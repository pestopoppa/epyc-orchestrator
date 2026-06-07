#!/usr/bin/env python3
"""AutoPilot: Continuous recursive optimization for the EPYC orchestration stack.

Main controller loop: observe → reason → act → evaluate → record → meta-learn.

Usage:
    python autopilot.py start [--dry-run] [--max-trials N]
    python autopilot.py status
    python autopilot.py pause
    python autopilot.py resume
    python autopilot.py report
    python autopilot.py plot
    python autopilot.py checkpoint [--production-best]
    python autopilot.py restore [--checkpoint PATH]
    python autopilot.py digest [--no-state-update]
    python autopilot.py peaf [--min-n N]   # PEAF cheap-kill report (intake-571 spike)

Environment flags:
    EPYC_AUTOPILOT_PEAF=0   Disable PEAF (Prediction-Error-As-Feature). Default ON: the
                            controller is asked to forecast each trial's objectives and
                            surprise is logged alongside actuals (logging-only, never
                            feeds back into scoring). Disable for baseline A/B or if
                            controller drift is suspected. See `python autopilot.py peaf`.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

import yaml

from experiment_journal import ExperimentJournal, JournalEntry, scrub_legacy_scale_text
from pareto_archive import ParetoArchive, ParetoEntry
from safety_gate import Baseline, DEFAULT_BASELINE_PATH, EvalResult, SafetyGate
from eval_tower import EvalTower
from config_applicator import apply_params, health_check
from meta_optimizer import MetaOptimizer, SpeciesBudget
from progress_plots import PLOTS_DIR, generate_all_plots
import peaf
from species import Seeder, NumericSwarm, PromptForge, StructuralLab, EvolutionManager
from species.prompt_forge import CODE_MUTATION_ALLOWLIST
from digest import generate_digest, should_generate_today
from short_term_memory import ShortTermMemory, TrialOutcome
from self_criticism import SelfCriticism, generate_self_criticism
from phase_status import AsyncTaskRunner, PhaseTracker

# 2026-05-22 Tranche-5 refactor — extracted modules. Public names re-imported below.
from controller_io import (
    extract_action,
    extract_rationale,
    invoke_controller as _invoke_controller_impl,
    validate_single_variable as _validate_single_variable,
    _unwrap_action,
)
from planner_coordinator import plan_with_providers
from state_store import (
    append_blacklist as _append_blacklist_impl,
    check_blacklist,
    format_model_signatures,
    load_blacklist as _load_blacklist_impl,
    load_model_signatures as _load_model_signatures_impl,
    load_state as _load_state_impl,
    save_state as _save_state_impl,
)
from actions import dispatch_action, SkipOutcome
from src.autopilot_core.action_identity import (
    EPHEMERAL_ACTION_KEYS,
    action_signature,
    canonical_action,
    config_fingerprint,
)
from src.autopilot_core.learning_exclusions import (
    BENIGN_LEARNING_EXCLUSIONS,
    classify_learning_exclusion,
)
from src.autopilot_core.tier_specs import DEFAULT_FRONTIER_TIER, MIN_FRONTIER_EVAL_TIER, objectives_from, spec_for

# Preflight diagnostics from seeding infra
sys.path.insert(0, str(SCRIPT_DIR.parent / "benchmark"))
try:
    from seeding_infra import get_preflight_diagnostics
except ImportError:
    get_preflight_diagnostics = None  # type: ignore[assignment]

# Strategy store for species memory (B1)
from orchestration.repl_memory.strategy_store import StrategyStore

# Durable earlyoom control-plane protection (mirrors orchestrator_stack). Guarded so a
# resolution hiccup never blocks autopilot import/startup — it is strictly best-effort.
try:
    from scripts.server.stack_processes import set_oom_score_adj as _set_oom_score_adj
except Exception:  # pragma: no cover - import-path fallback
    def _set_oom_score_adj(pids: Any, adj: int = -1000) -> int:
        return 0

log = logging.getLogger("autopilot")

if TYPE_CHECKING:
    from autopilot_tui import AutoPilotTUI

_EPHEMERAL_ACTION_KEYS = EPHEMERAL_ACTION_KEYS
_action_signature = action_signature
_canonical_action = canonical_action
_config_fingerprint = config_fingerprint

STATE_PATH = ORCH_ROOT / "orchestration" / "autopilot_state.json"
LOCK_PATH = ORCH_ROOT / "orchestration" / ".autopilot.lock"
BLACKLIST_PATH = SCRIPT_DIR / "failure_blacklist.yaml"
ORCHESTRATOR_URL = "http://localhost:8000"

# 2026-05-23 constrained-creativity planner knobs (gated on stagnation).
# Lean prompt is the default; the rich rubric+synthesis fragment activates
# only when one of the stagnation signals fires, to avoid spending prompt
# budget when autopilot is mid-exploit on a working lead.
CREATIVITY_N = 5            # candidates the rich prompt asks the controller to generate
TAIL_WINDOW = 30            # lookback for action_distribution "under-used" classification
TAIL_SEED_COUNT = 3         # seeds (not candidates) passed to LLM as inspiration
STAGNATION_HV_EPS = 1e-3    # hv_slope_10 strictly below this triggers rich prompt
STAGNATION_STREAK = 3       # N consecutive same-action_type trials triggers rich prompt
PLOT_INTERVAL = 10  # Generate plots every N trials
# A "trial" is an experiment that COLLECTS METRICS (runs an eval). These meta /
# housekeeping actions intentionally return no EvalResult — they mutate bookkeeping
# state (knowledge distillation, memory reset) without measuring anything — so they must
# NOT consume a trial number. They still execute and are logged; they simply do not count.
# MAX_CONSECUTIVE_META halts the loop if the planner gets stuck emitting only meta actions
# (the 2026-05-31 gate-lock symptom: ~80 distill_knowledge in a row, counter ticking with
# zero trials run), turning a silent no-op spin into a loud, fast stop for operator review.
META_NOOP_ACTIONS = {"distill_knowledge", "reset_memories"}
MAX_CONSECUTIVE_META = 5

# 2026-06-04 — non-executing-action (invalid/skipped) handling. The dispatcher
# now returns a SkipOutcome (not bare None) for actions that fail validation or
# are dropped by a guard. These are first-class outcomes: journaled, counted,
# fingerprinted, blacklisted, and circuit-broken — closing the blind spot that
# let the planner re-sample an impossible action 119× (graph_router deadlock).
MAX_CONSECUTIVE_SKIP = int(os.environ.get("AUTOPILOT_MAX_CONSECUTIVE_SKIP", "4"))
# A repeated *invalid* signature (stable validator reason, e.g. an unmet flag
# dependency) is auto-blacklisted at this many occurrences. Only "invalid"
# outcomes are blacklisted — generic "skipped" ones use too coarse a pattern.
INVALID_SIGNATURE_BLACKLIST_THRESHOLD = int(
    os.environ.get("AUTOPILOT_INVALID_BLACKLIST_THRESHOLD", "2")
)

# 2026-06-04 — draft_critique authority. When the binding critic rejects/revises
# a draft, the SUBSTITUTED action (safe fallback / revised_action) is what runs —
# but the rejected draft must still feed the invalid-action feedback + blacklist,
# or the planner could re-draft the same rejected action forever while the critic
# silently swaps in seed_batch. A run of consecutive rejected drafts is a stuck
# planner and halts loudly (mirrors MAX_CONSECUTIVE_META / MAX_CONSECUTIVE_SKIP).
MAX_CONSECUTIVE_REJECTED_DRAFTS = int(
    os.environ.get("AUTOPILOT_MAX_CONSECUTIVE_REJECTED_DRAFTS", "4")
)

# 2026-06-04 — experiment-quota policy (separate memory maintenance from the
# optimization budget). Once the memory store is already large, an unbounded run
# of passive seed/distill actions is the planner rationalizing no-op work; cap
# consecutive passive actions and force a frontier-moving experiment instead.
PASSIVE_ACTIONS = {"seed_batch", "distill_knowledge", "distill_skillbank"}
QUOTA_MEMORY_THRESHOLD = int(
    os.environ.get("AUTOPILOT_QUOTA_MEMORY_THRESHOLD", "2000")
)
MAX_CONSECUTIVE_PASSIVE = int(
    os.environ.get("AUTOPILOT_MAX_CONSECUTIVE_PASSIVE", "3")
)
# numeric_trial with empty params is the safest self-configuring frontier action
# (Optuna suggests the values; no file/flag dependency to get wrong).
_QUOTA_NUMERIC_SURFACES = ("think_harder", "escalation", "monitor", "memrl_retrieval")


def _enforce_experiment_quota(
    action: dict[str, Any],
    state: dict[str, Any],
    memory_count: int,
    rationale: dict[str, Any] | None = None,
    trial_counter: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Cap consecutive passive (seed/distill) actions once memory is large.

    Below QUOTA_MEMORY_THRESHOLD the system is still legitimately seeding, so
    passive actions pass through (the counter is still tracked). At or above the
    threshold, a run of more than MAX_CONSECUTIVE_PASSIVE passive actions is
    replaced with a frontier-moving numeric_trial so the planner cannot
    rationalize passive work forever. Non-passive actions reset the counter.
    """
    atype = action.get("type", "")
    if atype not in PASSIVE_ACTIONS:
        state["consecutive_passive_actions"] = 0
        return action, rationale

    streak = int(state.get("consecutive_passive_actions", 0))
    if memory_count >= QUOTA_MEMORY_THRESHOLD and streak >= MAX_CONSECUTIVE_PASSIVE:
        surface = _QUOTA_NUMERIC_SURFACES[trial_counter % len(_QUOTA_NUMERIC_SURFACES)]
        log.warning(
            "Experiment quota: %d consecutive passive actions with memory_count=%d "
            ">= %d threshold; forcing frontier-moving numeric_trial(surface=%s) "
            "instead of another '%s'.",
            streak, memory_count, QUOTA_MEMORY_THRESHOLD, surface, atype,
        )
        state["consecutive_passive_actions"] = 0
        return (
            {"type": "numeric_trial", "surface": surface, "params": {}},
            {
                **(rationale or {}),
                "experiment_quota_forced": True,
                "experiment_quota_reason": (
                    f"{streak} consecutive passive actions at memory={memory_count}"
                ),
            },
        )

    state["consecutive_passive_actions"] = streak + 1
    return action, rationale


def _build_feature_flags_block(lab: Any) -> str:
    """Render live feature-flag state + dependency rules for the planner prompt.

    The planner previously had no view of which flags exist, what they depend on,
    or which are currently on — so it proposed graph_router (off) without
    specialist_routing (off) 119× in a row. This block gives it the same
    dependency rules the validator enforces, plus live state, so it can enable a
    missing dependency first instead of re-sampling an impossible flag.
    """
    try:
        current = lab.current_flags()
    except Exception:
        current = {}
    try:
        schema = lab.flag_schema()
    except Exception:
        schema = []
    if not schema:
        return "  (feature flag registry unavailable)"

    def state_str(name: str) -> str:
        if not current:
            return "?"
        return "ON" if current.get(name) else "OFF"

    on = sorted(n for n, v in current.items() if v) if current else []
    lines: list[str] = []
    if current:
        lines.append(f"Currently ON ({len(on)}): " + (", ".join(on) or "(none)"))
    else:
        lines.append("Currently ON: (unknown — orchestrator /config not reachable)")
    lines.append("")
    lines.append(
        "Flags WITH dependencies (single-variable rule, AP-9: enable a missing "
        "dependency in its OWN trial first, then the dependent flag NEXT trial):"
    )
    any_dep = False
    for spec in schema:
        deps = spec.get("dependencies") or []
        if not deps:
            continue
        any_dep = True
        dep_states = ", ".join(f"{d}={state_str(d)}" for d in deps)
        deps_met = bool(current) and all(current.get(d) for d in deps)
        verdict = "DEPS MET — eligible" if deps_met else "DEPS MISSING — do NOT propose yet"
        lines.append(
            f"  - {spec['name']} (currently {state_str(spec['name'])}) "
            f"requires [{dep_states}] → {verdict}"
        )
    if not any_dep:
        lines.append("  (no dependency-bearing flags in registry)")
    lines.append("")
    lines.append(
        "RULE: never propose structural_experiment for a flag whose dependencies "
        "are not all currently ON. Setting a flag that is already ON is a no-op "
        "and burns a trial."
    )
    return "\n".join(lines)


def _build_last_invalid_feedback(state: dict[str, Any]) -> str:
    """Render the last non-executing action + its reason for the planner prompt.

    This is the residue that was previously discarded. Showing it (with the
    repeat count and blacklist threshold) is what stops the planner re-emitting
    an action the validator already rejected.
    """
    counts = state.get("invalid_signature_counts", {}) or {}
    repeated = sorted(
        ((c, s) for s, c in counts.items() if c >= 2), reverse=True
    )[:5]

    def _repeated_block(lines: list[str]) -> str:
        # Persistent across the run (NOT cleared when a trial succeeds), so a
        # repeatedly-rejected/invalid signature still surfaces even after the
        # single-turn last_invalid_action has been cleared by a good trial.
        if repeated:
            lines.append("  Repeatedly non-executing signatures this run "
                         f"(auto-blacklisted at {INVALID_SIGNATURE_BLACKLIST_THRESHOLD}×):")
            for c, s in repeated:
                lines.append(f"    {c}×  {s[:160]}")
        return "\n".join(lines)

    act = state.get("last_invalid_action")
    if not act:
        if repeated:
            return _repeated_block(
                ["  (last action executed; but these signatures keep failing:)"]
            )
        return "  (none — the last action executed and produced metrics)"
    reason = state.get("last_invalid_reason", "")
    status = state.get("last_invalid_status", "skipped")
    sig = _action_signature(act)
    n = int(counts.get(sig, 0))
    lines = [
        f"⚠ Your LAST action did NOT execute (status={status}) — it collected no metrics.",
        f"  action: {json.dumps(act, default=str)}",
        f"  reason: {reason}",
    ]
    if status in ("invalid", "critic_rejected"):
        lines.append(
            f"  this exact action has failed {n}× and will be AUTO-BLACKLISTED at "
            f"{INVALID_SIGNATURE_BLACKLIST_THRESHOLD}×."
        )
    lines.append(
        "  DO NOT repeat it. Fix the stated reason (e.g. enable a missing "
        "dependency flag in its own trial first) or choose a different action."
    )
    return _repeated_block(lines)


def _record_skip_trial(
    journal: Any,
    trial_id: int,
    action: dict[str, Any],
    species: str,
    status: str,
    reason: str,
    memory_count: int,
) -> None:
    """Journal a non-executing trial so it leaves durable residue (audit + planner).

    Uses pareto_status="skipped" and outcome_status to keep these out of the
    frontier/quality math while still recording that the trial number was
    consumed and why.
    """
    from experiment_journal import DeficiencyCategory

    deficiency = (
        DeficiencyCategory.INVALID_ACTION.value
        if status == "invalid"
        else DeficiencyCategory.DISPATCH_SKIPPED.value
    )
    entry = JournalEntry(
        trial_id=trial_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        species=species or "dispatch",
        action_type=action.get("type", ""),
        tier=0,
        quality=0.0,
        speed=0.0,
        cost=0.0,
        reliability=0.0,
        pareto_status="skipped",
        config_snapshot=dict(action),
        reasoning=json.dumps(action, default=str),
        memory_count=memory_count,
        failure_analysis=reason,
        deficiency_category=deficiency,
        outcome_status=status,
    )
    journal.record(entry)


def _record_rejected_draft(
    state: dict[str, Any],
    draft_action: dict[str, Any],
    critique: Any,
    trial_id: int,
) -> bool:
    """Feed a critic-rejected/revised draft into the invalid-action machinery.

    In draft_critique mode the SUBSTITUTED action (safe fallback / revised) is
    what dispatches, so the rejected draft would otherwise vanish — letting the
    planner re-draft it forever while the critic silently swaps in seed_batch.
    Record it like a non-executing action: fingerprint + count (persistent),
    surface it in the next prompt, auto-blacklist on repeat, and bump the
    consecutive-rejected-draft streak. Returns True if it was blacklisted.

    Note: the draft is NOT journaled as a trial (no trial number is consumed —
    the substituted action runs and is journaled instead); this only records the
    feedback residue so the rejected draft cannot bypass the loop.
    """
    issues = []
    try:
        issues = list(getattr(critique, "issues", []) or [])
    except Exception:
        issues = []
    reason = "critic rejected: " + ("; ".join(issues) if issues else
                                    getattr(critique, "decision", "rejected"))
    sig = _action_signature(draft_action)
    sig_counts = state.setdefault("invalid_signature_counts", {})
    sig_counts[sig] = int(sig_counts.get(sig, 0)) + 1
    state["last_invalid_action"] = draft_action
    state["last_invalid_reason"] = reason
    state["last_invalid_status"] = "critic_rejected"
    state["consecutive_rejected_drafts"] = (
        int(state.get("consecutive_rejected_drafts", 0)) + 1
    )

    blacklisted = False
    if sig_counts[sig] >= INVALID_SIGNATURE_BLACKLIST_THRESHOLD:
        append_blacklist(
            draft_action, trial_id,
            f"Auto-blacklisted: {sig_counts[sig]}× critic-rejected — {reason[:80]}",
        )
        blacklisted = True
    log.warning(
        "Critic %s the draft %s (substituted); recorded as feedback "
        "[signature seen %d×, consecutive rejected drafts=%d]%s",
        getattr(critique, "decision", "rejected"),
        json.dumps(draft_action, default=str),
        sig_counts[sig], state["consecutive_rejected_drafts"],
        " — BLACKLISTED" if blacklisted else "",
    )
    return blacklisted

def _force_metric_action_after_meta(
    action: dict[str, Any],
    state: dict[str, Any],
    rationale: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replace repeated meta no-ops with a measured action."""
    if (
        action.get("type") in META_NOOP_ACTIONS
        and int(state.get("consecutive_meta_actions", 0)) > 0
    ):
        log.warning(
            "Planner proposed metric-free meta action '%s' after %d "
            "consecutive meta action(s); forcing seed_batch to restore "
            "measurement.",
            action.get("type"),
            int(state.get("consecutive_meta_actions", 0)),
        )
        return (
            {"type": "seed_batch", "n_questions": 10},
            {
                **(rationale or {}),
                "meta_action_forced_metric_trial": True,
            },
        )
    return action, rationale


def _env_float(name: str, default: float, *, minimum: float = 0.1) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


PAUSE_POLL_S = _env_float("AUTOPILOT_PAUSE_POLL_S", 1.0)
HEALTH_BACKOFF_S = _env_float("AUTOPILOT_HEALTH_BACKOFF_S", 10.0)


def _stream_points_to_path(stream: Any, path: Path) -> bool:
    try:
        stream_stat = os.fstat(stream.fileno())
        path_stat = path.stat()
        return (
            stream_stat.st_dev == path_stat.st_dev
            and stream_stat.st_ino == path_stat.st_ino
        )
    except OSError:
        return False


def _autopilot_logging_handlers(
    log_path: Path,
    stream: Any | None = None,
) -> list[logging.Handler]:
    stream = stream or sys.stderr
    handlers: list[logging.Handler] = [logging.StreamHandler(stream)]
    if not _stream_points_to_path(stream, log_path):
        handlers.append(logging.FileHandler(log_path, mode="a"))
    return handlers

# ── Controller Prompt Template ───────────────────────────────────

PROGRAM_PATH = SCRIPT_DIR / "program.md"

CONTROLLER_PROMPT_TEMPLATE = """\
You are the AutoPilot meta-reasoning controller for an LLM orchestration stack.
Your job: analyze current system state and propose the SINGLE best next action.

## Program (strategy & constraints — human-editable)

{program}

## Current State

### Pareto Archive
{pareto_summary}

### Pareto Frontier Geometry
{pareto_geometry}

### Journal Trustworthiness (bug_corrupted filtering)
{journal_trustworthiness}

### Hypotheses Under Test (last 3 trustworthy trials)
{hypotheses_under_test}

### Experiment Journal (last 20 entries)
{journal_summary}

### Seeder Status
{seeder_status}

### Recent Batch Telemetry (autopilot adapts n_questions to fit budget)
{batch_telemetry}

### Species Effectiveness
{species_effectiveness}

### System Health
- Orchestrator: {health_status}
- Memory count: {memory_count}
- Q-value converged: {converged}

### Slot Memory (KV cache usage — consider slot_compact if tokens > 4000)
{slot_memory}

### Action Availability
{action_availability}

### Species Budget
{budget}

### Suite Quality Trends (last 10 evals)
{suite_quality_trends}

### Recent Insights (cross-species, structured per action_type, bug-corrupted excluded)
{insights_structured}

### Recent Insights (legacy flat — for backward compatibility)
{insights}

### Exploration mode
Stagnation signal: {stagnation_signal}

{exploration_block}

### Short-Term Memory (accumulated learnings this session)
{short_term_memory}

### Self-Criticism from Last Trial
{last_criticism}

### Active Model Performance Signatures
{model_signatures}

### Blacklisted Configurations
{blacklist_text}

### Feature Flags (live state + dependency rules — read before any structural_experiment)
{feature_flags_block}

### Last Non-Executing Action (validator/dispatch feedback — MUST address before re-proposing)
{last_invalid_feedback}

### Plot Paths (reference for trend analysis)
{plot_paths}

## Action Guidelines

1. If memories < 500: ALWAYS prioritize seeding (seed_batch)
2. If Q-values converged and models not trained: trigger train_routing_models
3. If models trained and not enabled: try structural_experiment with routing
   features — but ONLY for a flag whose dependencies are ALL currently ON (see
   "Feature Flags" above). One flag per trial (AP-9): e.g. graph_router requires
   specialist_routing, so if specialist_routing is OFF, enable IT first (its own
   trial), then graph_router next trial. Never re-propose a flag the validator
   just rejected (see "Last Non-Executing Action").
4. If stagnating (hv_slope < 0.001): try prompt_mutation or widen numeric search
5. If quality regression after changes: rollback to last good checkpoint
6. Consider the species budget allocation when choosing actions
7. If any slot shows >4000 tokens cached, consider slot_compact to free KV memory
   (use keep_ratio=0.3, target the port with the highest token count)

## Available Actions

Respond with EXACTLY ONE action in a ```json:autopilot_actions block:

- Seed: {{"type": "seed_batch", "n_questions": 10-50, "suites": ["coder","math",...]}}
- Numeric: {{"type": "numeric_trial", "surface": "memrl_retrieval|think_harder|monitor|escalation", "params": {{}}}}
  (Leave params empty to let Optuna suggest; provide params to test specific values)
- Prompt: {{"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix|compress|few_shot_evolution", "description": "..."}}
- GEPA: {{"type": "gepa_optimize", "file": "frontdoor.md", "max_evals": 50, "description": "..."}}
  (AP-19: Evolutionary prompt optimization via GEPA — runs ~50 evals internally, returns best candidate)
- Code: {{"type": "code_mutation", "file": "src/escalation.py", "mutation": "targeted_fix", "description": "..."}}
  (Mutate Python code — ONLY files in allowlist: {code_targets})
- Structural: {{"type": "structural_experiment", "flags": {{"feature_name": true/false}}}}
- Prune: {{"type": "structural_prune", "file": "frontdoor.md", "block": "## Section Name", "description": "..."}}
  (Delete an instruction block from a .md prompt file — accepted only if quality >= baseline AND instruction_token_ratio decreases)
- Compact: {{"type": "slot_compact", "port": 8070, "slot_id": 0, "keep_ratio": 0.3, "scorer": "expected_attention", "keep_first": 5, "n_future": 128}}
  (AM KV compaction — compress KV cache on a server slot. Use after long-context queries to free memory. Evaluates quality post-compact.)
- Train: {{"type": "train_routing_models", "min_memories": 500}}
- Distill: {{"type": "distill_skillbank", "teacher": "claude", "categories": ["routing"]}}
- Reset: {{"type": "reset_memories", "keep_seen": true, "keep_skills": true}}
- Deep eval: {{"type": "deep_eval", "tier": 2}}
  (Only tier is supported: 0, 1, or 2. Do NOT include target_trial, suites, baseline_recheck, or instrumentation fields.)
- Rollback: {{"type": "rollback", "to_checkpoint": "production_best"}}
- Distill: {{"type": "distill_knowledge", "last_n": 10}}
  (Run every ~5 trials to extract insights from recent outcomes into strategy memory)

Include brief reasoning before the action block.

After the action block, ALSO emit a second fenced block tagged
`autopilot_rationale` carrying the chosen action's falsifier and its self-scored
rubric. This sidecar is observability-only — a missing or malformed block will
not abort the trial, but populating it lets future planner passes grade new
candidates against still-open hypotheses:

```json:autopilot_rationale
{{"falsifier": "<one-line predicted outcome whose absence invalidates this hypothesis>",
 "rubric_scores": {{"info_gain": <1-5>, "coherence": <1-5>, "usefulness": <1-5>,
   "synthesis_note": "<optional one-line on fusion / cleaner model>"}}}}
```
"""


# ── Exploration block (stagnation-gated creative-prompt fragment) ─

_EXPLORATION_LEAN = """\
Before emitting your single action, briefly enumerate 3–5 alternatives you
considered. For each, give a one-line reason for rejection OR pick it as
your action.
"""


_EXPLORATION_RICH_TEMPLATE = """\
The system is STAGNATING. Run the constrained-creativity protocol:

1. **Generate {n} candidate actions.** Each must be:
   - Non-obvious relative to the recent default (last 3 trials' action_types).
   - Consistent with the Pareto geometry, blacklist, model signatures, and
     the last 30 trials' evidence above — do NOT optimize for weirdness.
   - Inspired (not constrained) by the under-used action types below — these
     are directions explored ≤1 time in the last {window} trials:

{tail_seeds}

2. **For each candidate, write one line each:**
   - why-low-typicality (what makes this non-default given current evidence)
   - falsifier (the concrete observation that would invalidate it)

3. **Score each candidate 1–5 on three axes:**
   - **info_gain** — novelty + falsifiability + explanatory compression
     (does resolving this trial reduce posterior uncertainty more than the
     obvious next step?)
   - **coherence** — consistency with facts, signatures, blacklist, the
     last 30 trials, and the still-open hypotheses listed below
   - **usefulness** — expected Pareto improvement per unit compute, minus
     risk of being decorative nonsense

4. **Synthesize.** If your top-2 candidates can be FUSED into one action
   that dominates both (e.g. a numeric_trial whose params encode a
   structural_experiment's hypothesis), prefer the fusion. The best
   creative idea reduces uncertainty, not adds complexity.

5. **Quote, don't regenerate.** The chosen action's rubric_scores in the
   rationale sidecar must be copied verbatim from your candidate table —
   no re-grading after the fact.

### Still-open hypotheses (carry an explicit falsifier, not yet resolved)
{unfalsified}

### Axis-vote BT tiebreak on top-K Pareto candidates (P17.BT-2 hint)

**This is a cheap axis-vote proxy, not peer-ranked consensus.**
Hypervolume scalarization can hide candidates that consistently beat
their peers across the 4 objectives without being individually
hypervolume-dominant. Pairwise BT aggregation over the recorded 4D
objectives (axis-vote / Borda counting — no judge-model inference)
surfaces those candidates as alternative exploration seeds. **Treat as
a hint, not a directive** — the BT-picked seed is only worth chasing
when it disagrees with the hypervolume-top seed AND the BT diagnostics
are clean (no Condorcet cycles, no extreme dominance skew). Top-K
candidate selection is per-axis range-normalized so the candidate set
fed to BT is not biased by axis magnitude (speed in t/s vs reliability
in [0,1]); the remaining caveat is that the proxy is still axis-vote,
not peer-judged, so the hint reflects per-axis dominance patterns and
nothing more.

{bt_tiebreak_hint}
"""


_RECOVERY_ONLY_ACTIONS = {
    "distill_knowledge",
    "reset_memories",
    "rollback",
}


def _slot_compaction_viable(slot_memory_text: str) -> bool:
    """True when any production slot currently has cached tokens."""
    return any(
        int(m.group(1)) > 0
        for m in re.finditer(r"(\d+)\s+tokens cached", slot_memory_text or "")
    )


def _type_only_blacklisted_actions(blacklist: list[dict[str, Any]]) -> dict[str, str]:
    """Map action_type -> reason for exact type-only blacklist entries."""
    blocked: dict[str, str] = {}
    for entry in blacklist:
        pattern = entry.get("pattern", {})
        if not isinstance(pattern, dict):
            continue
        if set(pattern) == {"type"} and isinstance(pattern.get("type"), str):
            blocked[pattern["type"]] = entry.get("reason", "blacklisted")
    return blocked


def _build_action_availability(
    *,
    journal: ExperimentJournal,
    known_actions: list[str],
    memory_count: int,
    converged: bool,
    slot_memory_text: str,
    blacklist: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Return prompt text + viable tail-seed action types for the planner."""
    blocked: dict[str, str] = {}
    cautions: dict[str, str] = {}

    blocked.update(_type_only_blacklisted_actions(blacklist))

    if memory_count < 500:
        blocked["train_routing_models"] = (
            f"needs >=500 routing memories; current memory_count={memory_count}"
        )
    elif not converged:
        cautions["train_routing_models"] = (
            "routing memories exist, but the current seeder session is not "
            "converged yet; avoid training as a stagnation escape hatch"
        )

    if not _slot_compaction_viable(slot_memory_text):
        blocked["slot_compact"] = "all production slots are empty/offline right now"

    cautions["reset_memories"] = (
        "destructive recovery action; do not use for ordinary stagnation"
    )
    cautions["rollback"] = (
        "recovery-only action; use only after a concrete regression from a known-good point"
    )
    cautions["distill_knowledge"] = (
        "metric-free meta action; do not use to break a measurement stall"
    )

    try:
        insights = journal.insights_structured(n=120, exclude_bug_corrupted=True)
    except Exception:
        insights = {}

    for action_type in ("prompt_mutation", "train_routing_models"):
        info = insights.get(action_type)
        if not info:
            continue
        if info.get("successes", 0) == 0 and info.get("failures", 0) >= 1:
            cautions[action_type] = (
                f"recent evidence is negative: trials {info.get('trials_supporting', [])} "
                f"produced only failures ({info.get('observation', '')})"
            )

    lines = []
    if blocked:
        lines.append("Currently unavailable:")
        for action_type, reason in sorted(blocked.items()):
            lines.append(f"- `{action_type}`: {reason}")
    if cautions:
        lines.append("Use only with a concrete new falsifier:")
        for action_type, reason in sorted(cautions.items()):
            lines.append(f"- `{action_type}`: {reason}")
    if not lines:
        lines.append("(no action-specific availability constraints detected)")

    viable_tail_actions = [
        action_type
        for action_type in known_actions
        if action_type not in blocked
        and action_type not in _RECOVERY_ONLY_ACTIONS
        and not (action_type == "train_routing_models" and not converged)
    ]
    return "\n".join(lines), viable_tail_actions


def _build_exploration_block(
    journal: ExperimentJournal,
    archive: ParetoArchive,
    known_actions: list[str],
) -> tuple[str, str]:
    """Return (exploration_block_text, stagnation_signal_text).

    Selects between the lean fragment (default) and the rich constrained-
    creativity fragment when at least one stagnation signal fires:
      - hv_slope_10 strictly below STAGNATION_HV_EPS
      - trustworthy trial count < 5 (low-signal regime)
      - last STAGNATION_STREAK trials share an action_type
    """
    reasons: list[str] = []

    # Pareto-hypervolume slope (already computed by archive.geometry()).
    # The threshold auto-calibrates from recent hv_slope noise where possible;
    # falls back to STAGNATION_HV_EPS (the hand-tuned constant) early on.
    hv_slope_10 = None
    try:
        geom = archive.geometry()
        hv_slope_10 = geom.get("hv_slope_10") if isinstance(geom, dict) else None
    except Exception:
        pass
    eps = STAGNATION_HV_EPS
    try:
        if hasattr(archive, "hv_slope_noise_floor"):
            eps = archive.hv_slope_noise_floor(floor_default=STAGNATION_HV_EPS)
    except Exception:
        eps = STAGNATION_HV_EPS
    if hv_slope_10 is not None and hv_slope_10 < eps:
        reasons.append(f"hv_slope_10={hv_slope_10:+.5f} < eps={eps:.5f}")

    # Trustworthiness low-signal flag.
    try:
        trust = journal.trustworthiness_score()
        if trust.get("low_signal"):
            reasons.append(f"trustworthy={trust.get('trustworthy', 0)} < 5")
    except Exception:
        pass

    # Action-type streak over the last STAGNATION_STREAK trials.
    try:
        recent = journal.recent(STAGNATION_STREAK)
        if len(recent) == STAGNATION_STREAK:
            types = {e.action_type for e in recent}
            if len(types) == 1:
                reasons.append(
                    f"last {STAGNATION_STREAK} trials all action_type={next(iter(types))}"
                )
    except Exception:
        pass

    if not reasons:
        return _EXPLORATION_LEAN, "none (lean prompt)"

    # Rich fragment: tail seeds + unfalsified hypotheses.
    try:
        tail = journal.tail_action_candidates(
            known_action_types=known_actions,
            last_n=TAIL_WINDOW,
            n_sample=TAIL_SEED_COUNT,
        )
    except Exception:
        tail = []
    if tail:
        tail_text = "\n".join(f"  - {t}" for t in tail)
    else:
        tail_text = "  (no under-used action types — every action_type seen recently)"

    try:
        unfalsified = journal.unfalsified_hypotheses(n=5)
    except Exception:
        unfalsified = []
    if unfalsified:
        unfalsified_text = "\n".join(
            f"  #{tid}: {hyp[:160]}\n     falsifier: {fal[:160]}"
            for tid, hyp, fal in unfalsified
        )
    else:
        unfalsified_text = "  (no recent trials with explicit falsifiers yet)"

    # P17.BT-2: axis-vote BT tiebreak on top-K Pareto candidates. Cheap
    # proxy — pairwise inputs come from Borda counting over the recorded
    # 4D objectives, NOT from judge-model peer ranking (that is P17.BT-4
    # and is INFERENCE-GATED). Surfaces candidates that beat peers across
    # axes but aren't hypervolume-dominant.
    bt_tiebreak_text = ""
    bt_signal = ""
    try:
        bt = archive.bt_tiebreak_topk(k=5)
        if bt and bt.get("top_k_trial_ids"):
            lines = [f"  {bt.get('note', '')}"]
            log_skills = bt.get("log_skills", {})
            for rank_pos, tid in enumerate(bt.get("ranking", []), start=1):
                lines.append(
                    f"    {rank_pos}. trial #{tid}  log-skill={log_skills.get(tid, 0.0):+.2f}"
                )
            if bt.get("warnings"):
                lines.append("  diagnostics:")
                for w in bt["warnings"]:
                    lines.append(f"    - {w}")
            bt_tiebreak_text = "\n".join(lines)
            note = bt.get("note", "")
            if "disagrees" in note:
                bt_signal = "BT-tiebreak disagrees with hypervolume top"
    except Exception as exc:  # noqa: BLE001 — defensive; BT must never block exploration
        bt_tiebreak_text = f"  (BT tiebreak unavailable: {exc!s})"

    if not bt_tiebreak_text:
        bt_tiebreak_text = "  (frontier too sparse for BT tiebreak)"

    block = _EXPLORATION_RICH_TEMPLATE.format(
        n=CREATIVITY_N,
        window=TAIL_WINDOW,
        tail_seeds=tail_text,
        unfalsified=unfalsified_text,
        bt_tiebreak_hint=bt_tiebreak_text,
    )
    # Append BT signal to the stagnation reason text for journal/digest logs.
    if bt_signal:
        reasons.append(bt_signal)
    return block, "; ".join(reasons)


# ── State Management ─────────────────────────────────────────────


# State / blacklist / signatures helpers moved to state_store.py (2026-05-22 refactor).
# Wrappers below preserve the original autopilot.py API by supplying STATE_PATH,
# BLACKLIST_PATH, and the model-quality-signatures path.

_MODEL_SIGNATURES_PATH = ORCH_ROOT / "orchestration" / "model_quality_signatures.yaml"


def _maybe_reimport_pareto_from_journal(
    archive: "ParetoArchive",
    journal: "ExperimentJournal",
    trial_id: int,
) -> bool:
    """Re-add a single journal entry to the Pareto archive if missing.

    Per handoffs/active/autopilot-exogenous-restart-resilience.md Section 5.7.
    Handles the corruption window where the journal advanced (line 837)
    but archive.save (line 929) was never reached → on restart the journal
    has the entry but the on-disk Pareto archive doesn't.

    Returns True if the entry was re-imported, False otherwise.

    Edge cases:
      - SKIP if no JournalEntry matches trial_id.
      - SKIP if entry.bug_corrupted_by is non-empty (placeholders + tagged
        trials must never enter the archive).
      - SKIP if the archive already has an entry with this trial_id.
      - For valid entries: construct ParetoEntry from journal fields and
        call archive.update() — let the dominance check re-classify;
        do NOT preserve the JournalEntry's stale pareto_status.
    """
    entry = next(
        (e for e in journal.all_entries() if e.trial_id == trial_id), None
    )
    if entry is None:
        log.info("Pareto re-import: no journal entry for trial %d", trial_id)
        return False
    if entry.bug_corrupted_by:
        log.info(
            "Pareto re-import: trial %d is bug_corrupted_by=%s, skipping",
            trial_id, entry.bug_corrupted_by,
        )
        return False
    # Trusted within-noise rows (mad_noise / reproduction_confirmed) are managed as
    # robust-median REPRESENTATIVES, not raw per-trial points. Re-importing one as a single
    # noisy sample would contradict that policy, so skip — the representative is rebuilt from
    # the persisted reproduction cluster on the next reproduction.
    excl_by = (entry.eval_details or {}).get("learning_exclusion", {}).get("by", "")
    if excl_by in BENIGN_LEARNING_EXCLUSIONS:
        log.info(
            "Pareto re-import: trial %d is a trusted within-noise exclusion (%s) — "
            "representative-managed, not raw-re-imported.",
            trial_id, excl_by,
        )
        return False
    # Already in archive? (use the private _all_entries list — it's the
    # only enumeration of every observed trial; the public surface exposes
    # only frontier-related views, and re-importing a duplicate would
    # double-count in archive statistics.)
    existing_ids = {e.trial_id for e in getattr(archive, "_all_entries", [])}
    if trial_id in existing_ids:
        log.info("Pareto re-import: trial %d already in archive", trial_id)
        return False
    p_entry = ParetoEntry(
        trial_id=entry.trial_id,
        objectives=spec_for(entry.tier).objectives_from_row({
            "quality": entry.quality,
            "speed": entry.speed,
            "cost": entry.cost,
            "reliability": entry.reliability,
        }) or (entry.quality, entry.speed, -entry.cost, entry.reliability),
        config_snapshot=entry.config_snapshot,
        git_tag=entry.git_tag,
        eval_tier=entry.tier,
        reasoning=(entry.reasoning or "")[:200],
        parent_trial=entry.parent_trial,
        memory_count=entry.memory_count,
        active_flags=list(entry.active_flags or []),
        species=entry.species,
        timestamp=entry.timestamp,
    )
    new_status = archive.update(p_entry)
    log.info(
        "Pareto re-import: trial %d added (status=%s)", trial_id, new_status
    )
    # Persist the re-imported entry immediately so a subsequent crash
    # doesn't lose it again.
    try:
        archive.save({"trial_counter": max(trial_id + 1, 0)})
    except Exception as exc:
        log.warning("Pareto re-import: archive.save failed: %s", exc)
    return True


def _recover_from_in_flight_trial(
    state: dict[str, Any],
    journal: "ExperimentJournal",
    archive: "ParetoArchive",
    trial_counter: int,
) -> int:
    """Apply the Phase 6b in_flight_trial recovery sequence.

    Returns the new trial_counter. Mutates `state` in place but does NOT
    save_state() — caller owns persistence so save can be paired with
    other startup-time state mutations.

    No-op (returns trial_counter unchanged) if state["in_flight_trial"]
    is None.

    Two recovery cases (handoff Section 5.7):
      (a) journal_max >= prior_in_flight.trial_id → trial was journaled
          before the crash. Bump trial_counter past it and attempt to
          re-import a missing Pareto entry from the journal.
      (b) journal_max < prior_in_flight.trial_id → trial died BEFORE
          journal.record. Write an AUTOPILOT_KILLED placeholder so the
          planner sees the gap, and bump trial_counter past it.

    On exit, state["in_flight_trial"] is None.
    """
    prior_in_flight = state.get("in_flight_trial")
    if prior_in_flight is None:
        return trial_counter

    journal_max = journal.next_trial_id() - 1
    prior_tid = int(prior_in_flight.get("trial_id", -1))
    log.warning(
        "Autopilot recovery: detected in_flight_trial marker "
        "(trial_id=%d, host_pid=%s, host_started_at=%s). "
        "Journal max trial_id=%d.",
        prior_tid,
        prior_in_flight.get("host_pid"),
        prior_in_flight.get("host_started_at"),
        journal_max,
    )
    if journal_max >= prior_tid:
        new_counter = max(trial_counter, journal_max + 1)
        log.info(
            "Recovery: trial %d was journaled before crash; "
            "bumping trial_counter %d → %d",
            prior_tid, trial_counter, new_counter,
        )
        trial_counter = new_counter
        state["trial_counter"] = trial_counter
        try:
            _maybe_reimport_pareto_from_journal(archive, journal, prior_tid)
        except Exception as exc:
            log.warning("Pareto re-import for trial %d failed: %s", prior_tid, exc)
    else:
        log.warning(
            "Recovery: trial %d died BEFORE journal.record. Writing "
            "AUTOPILOT_KILLED placeholder.",
            prior_tid,
        )
        try:
            placeholder = JournalEntry(
                trial_id=prior_tid,
                timestamp=datetime.now(timezone.utc).isoformat(),
                species="(killed)",
                action_type=(prior_in_flight.get("action") or {}).get("type", "unknown"),
                tier=0,
                quality=0.0, speed=0.0, cost=0.0, reliability=0.0,
                pareto_status="dominated",
                failure_analysis=(
                    f"Autopilot process killed before journal.record() "
                    f"(prior host_pid={prior_in_flight.get('host_pid')}, "
                    f"prior host_started_at={prior_in_flight.get('host_started_at')}, "
                    f"died at trial_id={prior_tid})."
                ),
                bug_corrupted_by="autopilot_killed_mid_trial",
                bug_corrupted_reason="incomplete trial; no eval evidence available",
                deficiency_category="autopilot_killed_mid_trial",
            )
            journal.record(placeholder)
            trial_counter = prior_tid + 1
            state["trial_counter"] = trial_counter
        except Exception as exc:
            log.error(
                "Failed to write AUTOPILOT_KILLED placeholder for trial %d: %s",
                prior_tid, exc,
            )
    state["in_flight_trial"] = None
    return trial_counter


def _default_state() -> dict[str, Any]:
    return {
        "trial_counter": 0,
        "session_id": None,
        "paused": False,
        "species_budget": SpeciesBudget().as_dict(),
        "td_errors": [],
        "seeder_state": {},
        # 2026-05-23 Phase 6b — autopilot self-crash recovery markers.
        # in_flight_trial is set immediately BEFORE dispatch_action and
        # cleared only AFTER the final atomic save_state. On startup,
        # _recover_in_flight_trial inspects it: if non-None, the prior
        # autopilot instance died mid-trial. See cmd_start's recovery
        # block for the exact taxonomy of cases handled.
        # autopilot_fleet_started_at is bumped on every autopilot start
        # so operators (and future watchdogs) can detect whether the
        # currently-running instance is the same one that recorded a
        # given trial.
        "in_flight_trial": None,
        "autopilot_fleet_started_at": None,
    }


def load_state() -> dict[str, Any]:
    return _load_state_impl(STATE_PATH, _default_state)


def save_state(state: dict[str, Any]) -> None:
    _save_state_impl(STATE_PATH, state)


def learning_exclusion_criticism(
    learning_excluded_by: str,
    learning_excluded_reason: str,
) -> SelfCriticism:
    """Controller-facing criticism for journaled-but-untrusted trials.

    SafetyGate can pass a trial while still marking it as `mad_noise`. Those
    entries are already excluded from Pareto/AP-22 learning, so their journal
    criticism must not say "keep" or "continue this surface"; that text is fed
    back into the planner and can create meta-action loops.
    """
    reason = learning_excluded_reason or "outcome was excluded from learning"
    if learning_excluded_by == "reproduction_confirmed":
        # Convergence, NOT a failure. The planner must read this as "the current
        # config is confirmed/converged on this surface" and move on — not as a
        # wasted/noisy trial demanding another attempt (which fuels the loop).
        return SelfCriticism(
            what_went_wrong="",
            why_it_happened=reason,
            what_should_change=(
                "Treat the current config as CONFIRMED / converged on this surface. "
                "Do NOT re-run this surface expecting a fresh win, and do NOT read "
                "repeated confirmations as a noisy or broken instrument — explore a "
                "different surface or idle cleanly until new signal is available."
            ),
            optimization_directions=[],
            keep_or_revert="excluded",
            keep_revert_reasoning=(
                "reproduction confirms the existing kept gain (convergence); no NEW "
                "Pareto point, but the config remains validated and trustworthy"
            ),
        )
    label = learning_excluded_by.replace("_", " ")
    return SelfCriticism(
        what_went_wrong=f"Trial excluded from learning: {label}",
        why_it_happened=reason,
        what_should_change=(
            "Do not treat this outcome as a keep or config-efficacy signal; "
            "require a clean, non-excluded metric trial before continuing this direction"
        ),
        optimization_directions=[],
        keep_or_revert="excluded",
        keep_revert_reasoning=(
            f"{reason}; archive/AP-22 learning skipped and planner trust excludes this trial"
        ),
    )


def _classify_meta_halt(journal: Any) -> str:
    """Classify a meta-action-loop halt as 'converged' vs 'stuck'.

    Convergence: the planner ran out of config moves because the recent metric
    trials REPRODUCED an already-established above-baseline config
    (``reproduction_confirmed``) rather than failing or corrupting. That is a
    benign terminal state — the planner should be pointed at a NEW surface, not
    treated as a malfunction or an instrument-noise problem (2026-05-31). Any
    other shape (bug-corruptions, kills, regressions, genuine gate-lock) is
    ``stuck`` and warrants operator review.
    """
    try:
        recent = journal.recent_hypotheses(n=5, exclude_bug_corrupted=False)
    except Exception:  # noqa: BLE001 — never let classification crash the halt
        return "stuck"
    repro = sum(
        1 for e in recent
        if getattr(e, "deficiency_category", "") == "reproduction_confirmed"
    )
    corrupt = sum(1 for e in recent if getattr(e, "bug_corrupted_by", ""))
    return "converged" if (repro >= 1 and corrupt == 0) else "stuck"


def load_blacklist() -> list[dict[str, Any]]:
    """Load failure blacklist from YAML."""
    return _load_blacklist_impl(BLACKLIST_PATH)


def load_model_signatures() -> dict[str, Any]:
    """Load model quality signatures from YAML."""
    return _load_model_signatures_impl(_MODEL_SIGNATURES_PATH)


def append_blacklist(action: dict[str, Any], trial_id: int, reason: str) -> None:
    """Auto-append a blacklist entry after rollback trigger."""
    _append_blacklist_impl(action, trial_id, reason, BLACKLIST_PATH)


# ── Slot Memory Visibility (AM KV Compaction) ──────────────────


# Primary production ports by role — query these for slot memory stats.
_SLOT_QUERY_PORTS: dict[str, list[int]] = {
    "frontdoor": [8070],
    "coder": [8070],  # shares server with frontdoor (same Qwen3.6-35B Q8 GGUF)
    "worker": [8072],
    "architect_general": [8083],
}


def _query_slot_memory() -> str:
    """Query llama-server /slots on production ports and return a summary.

    Returns a compact text block showing per-role slot memory usage so the
    controller can decide when slot_compact is worthwhile.
    """
    import httpx

    lines: list[str] = []
    for role, ports in _SLOT_QUERY_PORTS.items():
        for port in ports:
            try:
                resp = httpx.get(
                    f"http://localhost:{port}/slots", timeout=3.0
                )
                if resp.status_code != 200:
                    lines.append(f"  {role}:{port} — unreachable ({resp.status_code})")
                    continue
                slots = resp.json()
                if not isinstance(slots, list):
                    continue
                for s in slots:
                    sid = s.get("id", "?")
                    state = s.get("state", "?")
                    n_past = s.get("n_past", 0)
                    if n_past > 0:
                        lines.append(
                            f"  {role}:{port}/slot{sid} — {state}, {n_past} tokens cached"
                        )
            except Exception:
                lines.append(f"  {role}:{port} — offline")
    if not lines:
        return "  (all slots empty or servers offline)"
    return "\n".join(lines)


# ── Claude CLI Controller ────────────────────────────────────────
# Controller invocation, action extraction, and AP-9 single-variable
# validation moved to controller_io.py (2026-05-22 refactor). Wrapper below
# preserves the original cwd=ORCH_ROOT semantics; extract_action +
# _validate_single_variable + _unwrap_action are re-imported up top.


def invoke_controller(
    prompt: str,
    session_id: str | None = None,
    timeout: int = 300,
) -> tuple[str, str | None]:
    """Invoke Claude CLI for meta-reasoning."""
    return _invoke_controller_impl(prompt, session_id=session_id, timeout=timeout, cwd=ORCH_ROOT)


# ── Action Dispatch (moved to actions.py, 2026-05-22 refactor) ──
# dispatch_action() is re-imported from actions.py at the module top so the
# public name + signature are preserved for callers (run_loop, tests).



# ── Main Loop ────────────────────────────────────────────────────


def run_loop(
    max_trials: int | None = None,
    dry_run: bool = False,
    use_controller: bool = True,
    use_tui: bool = False,
) -> None:
    """Main optimization loop."""
    # Optional TUI for live inference monitoring
    tui = None
    if use_tui:
        try:
            from autopilot_tui import AutoPilotTUI
            tui = AutoPilotTUI()
            tui.__enter__()
        except Exception as e:
            log.warning("TUI not available: %s", e)
            tui = None

    try:
        _run_loop_inner(max_trials, dry_run, use_controller, tui)
    finally:
        if tui is not None:
            tui.__exit__(None, None, None)


def _run_loop_inner(
    max_trials: int | None,
    dry_run: bool,
    use_controller: bool,
    tui: "AutoPilotTUI | None" = None,
) -> None:
    """Inner loop (separated to ensure TUI cleanup via run_loop's finally)."""
    state = load_state()
    journal = ExperimentJournal()
    archive = ParetoArchive()
    # Clear the deliberate-rebase bypass ONLY once the frontier has actually rebuilt
    # (a prior run admitted >=1 point). Clearing it at startup while the frontier is
    # still empty would re-arm the frontier-lost guard before the bootstrap lands —
    # crash-looping any restart that happens before the first trial is admitted. Once
    # a point exists the guard never fires anyway, so this safely re-arms it.
    if state.get("_allow_empty_frontier_rebase") and archive.frontier_size(DEFAULT_FRONTIER_TIER) > 0:
        state.pop("_allow_empty_frontier_rebase", None)
        save_state(state)
        log.info("Rebase complete (frontier rebuilt) — cleared _allow_empty_frontier_rebase; guard re-armed.")
    gate = SafetyGate(
        consecutive_failures=state.get("consecutive_failures", 0),
        quality_history=state.get("quality_history", []),
        quality_history_by_tier=state.get("quality_history_by_tier", {}),
        baseline_state=state.get("baseline_state", {}),
    )
    tower = EvalTower(
        url=ORCHESTRATOR_URL,
        on_question=tui.set_prompt if tui is not None else None,
    )
    meta = MetaOptimizer()

    # Phase 5 — OrchestratorWatcher for exogenous-restart detection.
    # Single instance shared across the trial loop, the seeder, and the
    # eval tower. Disabled via AUTOPILOT_WATCHER_DISABLED=1 for tests/dev
    # without fleet markers; in that mode all methods no-op safely.
    try:
        from orchestrator_watch import OrchestratorWatcher
        watcher = OrchestratorWatcher(api_url=ORCHESTRATOR_URL)
        # Attach to the tower so its _eval_question call can pass it down
        # into call_orchestrator_forced (Phase 4 reads via
        # getattr(self, "watcher", None)).
        tower.watcher = watcher
        log.info(
            "OrchestratorWatcher attached (disabled=%s, marker dir=/mnt/raid0/llm/tmp)",
            watcher.disabled,
        )
    except Exception as exc:
        log.warning("OrchestratorWatcher init failed (%s); falling back to legacy retry-less path", exc)
        watcher = None

    seeder = Seeder(
        url=ORCHESTRATOR_URL,
        dry_run=dry_run,
        on_question=tui.set_prompt if tui is not None else None,
    )
    swarm = NumericSwarm()
    forge = PromptForge(auto_commit=not dry_run)
    lab = StructuralLab(orchestrator_url=ORCHESTRATOR_URL)
    evo = EvolutionManager(use_local_model=not use_controller)

    # AP-22: Short-term memory (accumulated learnings across trials)
    memory = ShortTermMemory()
    last_criticism_text = "(first trial — no prior criticism)"

    # B1: Strategy store for species memory
    strategy_store: StrategyStore | None = None
    try:
        strategy_store = StrategyStore()
        log.info("Strategy store loaded (%d entries)", strategy_store.count())
    except Exception as e:
        log.warning("Strategy store unavailable: %s", e)

    # B2: Failure blacklist
    blacklist = load_blacklist()

    # Load species budget from state
    if "species_budget" in state:
        b = state["species_budget"]
        meta.budget = SpeciesBudget(**b)

    # Restore seeder convergence state. Prefer the explicit state shape; fall
    # back to legacy td_errors-only persistence so existing state files still
    # reconstruct batch_count + convergence streak sensibly.
    seeder.restore_state(
        state.get("seeder_state")
        or {"td_errors": state.get("td_errors", [])}
    )

    trial_counter = state.get("trial_counter", 0)
    plot_paths: list[str] = []

    # 2026-05-23 Phase 6b — autopilot self-crash recovery.
    # Check for an in_flight_trial marker left by a prior crashed instance.
    # Recovery handles two cases (handoff Section 5.7):
    #   (a) journal_max >= prior_in_flight.trial_id → trial DID get recorded;
    #       we died between journal.record (line 837) and save_state (line 930).
    #       Re-sync trial_counter + attempt Pareto re-import from the journal.
    #   (b) journal_max <  prior_in_flight.trial_id → trial died BEFORE
    #       journal.record. Write a placeholder JournalEntry tagged
    #       bug_corrupted_by=autopilot_killed_mid_trial so the planner sees
    #       the gap and excludes it from hypothesis chains. Skip gate + archive.
    trial_counter = _recover_from_in_flight_trial(
        state, journal, archive, trial_counter,
    )
    # Bump the fleet-startup timestamp on every start (recovery path or
    # normal startup) so downstream watchers can detect autopilot
    # restarts the same way they detect orchestrator/llama restarts.
    state["autopilot_fleet_started_at"] = time.time()
    save_state(state)

    # Graceful shutdown handler
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        log.info("Shutdown requested (signal %d)", signum)
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log.info("AutoPilot starting (trial=%d, dry_run=%s)", trial_counter, dry_run)
    phase = PhaseTracker()
    async_tasks = AsyncTaskRunner()
    phase.set("starting", trial_id=trial_counter, dry_run=dry_run)

    while not shutdown_requested:
        async_tasks.reap(logger=log)
        phase.set("loop_start", trial_id=trial_counter)
        if max_trials and trial_counter >= max_trials:
            log.info("Max trials reached (%d)", max_trials)
            phase.set("max_trials_reached", trial_id=trial_counter, max_trials=max_trials)
            break

        # 2026-05-24 pause-bug fix: re-read state from disk at the top of
        # every iteration so that an external `autopilot.py pause` actually
        # takes effect on a running autopilot. Pre-fix the loop's local
        # `state` was loaded once at startup and only refreshed inside the
        # paused branch; meanwhile `save_state(state)` after each trial wrote
        # the cached `paused=False` back to disk, clobbering any externally-
        # set True. Net effect: `pause` was silently a no-op on a running
        # autopilot. See `feedback_autopilot_pause_broken_use_sigterm` memory.
        # Re-loading here is cheap (one small JSON file) and merges any
        # externally-set fields (paused, _in_cache_flush) without losing the
        # in-memory trial counters that get written back at end of iteration.
        was_paused = bool(state.get("paused"))
        try:
            disk_state = load_state()
            for key in ("paused", "_in_cache_flush"):
                if key in disk_state:
                    state[key] = disk_state[key]
        except Exception as _exc:
            log.warning("Failed to reload state at iteration top: %s", _exc)

        # Operator resumed (paused True→False): clear the meta-action-loop latch
        # so the planner starts fresh instead of re-tripping the guard on meta
        # action #1. Pre-fix, consecutive_meta_actions persisted across the halt
        # and a resume re-halted immediately (2026-05-31).
        if was_paused and not state.get("paused"):
            if state.get("consecutive_meta_actions") or state.get("_meta_halt_reason"):
                log.info(
                    "Resume after pause: clearing meta-loop latch "
                    "(consecutive_meta_actions %s→0, reason=%s)",
                    state.get("consecutive_meta_actions", 0),
                    state.get("_meta_halt_reason", ""),
                )
            state["consecutive_meta_actions"] = 0
            state.pop("_dispatch_deficiency", None)
            state.pop("_meta_halt_reason", None)
            save_state(state)

        if state.get("paused"):
            log.info("AutoPilot paused, waiting...")
            phase.set(
                "paused",
                trial_id=trial_counter,
                idle_reason="autopilot paused",
                poll_s=PAUSE_POLL_S,
            )
            time.sleep(PAUSE_POLL_S)
            continue

        # Check orchestrator health
        phase.set("health_check", trial_id=trial_counter, url=ORCHESTRATOR_URL)
        _health = health_check(ORCHESTRATOR_URL, retries=2)
        if not dry_run and not _health:
            log.error(
                "Orchestrator unhealthy [%s]: %s — waiting %.1fs...",
                _health.failure_reason, _health.failure_detail,
                HEALTH_BACKOFF_S,
            )
            phase.set(
                "health_backoff",
                trial_id=trial_counter,
                idle_reason="orchestrator unhealthy",
                failure_reason=_health.failure_reason,
                failure_detail=_health.failure_detail,
                backoff_s=HEALTH_BACKOFF_S,
            )
            time.sleep(HEALTH_BACKOFF_S)
            continue

        # Check preflight diagnostics for stack-level issues
        phase.set("preflight", trial_id=trial_counter)
        if get_preflight_diagnostics is not None:
            try:
                _pf = get_preflight_diagnostics()
                _last = _pf.get("last_preflight", {})
                if _last.get("status") == "failed":
                    log.warning(
                        "Preflight failed [%s]: %s — %s",
                        _last.get("stage", "?"),
                        _last.get("failure_reason", "?"),
                        _last.get("failure_detail", ""),
                    )
            except Exception:
                pass  # Preflight diagnostics are optional

        # ── 1. Observe ───────────────────────────────────────────
        phase.set("observe", trial_id=trial_counter)
        memory_count = seeder.get_memory_count() if not dry_run else 0
        converged = seeder.is_converged
        hv_slope = archive.hypervolume_slope(50)

        # ── 2. Reason ────────────────────────────────────────────
        if tui is not None:
            tui.set_status("selecting next trial (controller)…")
        if use_controller:
            phase.set(
                "planner_prompt_build",
                trial_id=trial_counter,
                idle_reason="building controller prompt",
            )
            # Load program.md (human-editable strategy file)
            try:
                program_text = PROGRAM_PATH.read_text()
            except OSError:
                program_text = "(program.md not found)"
            # B4/B5: Format insights for controller
            insights_text = journal.insights_text(n=10)

            # B2: Format blacklist for controller
            if blacklist:
                bl_lines = []
                for entry in blacklist:
                    bl_lines.append(f"  - {entry.get('pattern', {})} — {entry.get('reason', '')}")
                blacklist_text = "\n".join(bl_lines)
            else:
                blacklist_text = "  (none)"

            # AM compaction: query slot memory so controller can decide on compaction
            try:
                slot_memory_text = _query_slot_memory() if not dry_run else "  (dry run)"
            except Exception:
                slot_memory_text = "  (query failed)"

            # Load model signatures for hypothesis assessment
            model_sigs = load_model_signatures()
            model_signatures_text = format_model_signatures(model_sigs)

            # Adaptive-batch telemetry hint for the controller. Surfaces
            # recent seconds-per-question + recommended n_questions so the
            # planner can request batches that will actually fit the budget,
            # rather than asking for 10 every time and getting silently
            # scaled to 3 by _action_seed_batch's adaptive_batch_size().
            try:
                import sys as _sys
                _sys.path.insert(
                    0,
                    "/mnt/raid0/llm/epyc-orchestrator/scripts/benchmark",
                )
                from seeding_telemetry import (
                    batch_summary as _batch_summary,
                    adaptive_batch_size as _adaptive_n,
                )
                _bs = _batch_summary()
                _rate = _bs.get("median_s_per_q")
                if _rate is None:
                    batch_telemetry_text = (
                        "(no batch history yet — first seed_batch will use "
                        "the requested n; subsequent batches will auto-adapt)"
                    )
                else:
                    _budget_s = float(os.environ.get("SEEDING_BATCH_BUDGET_S", "900"))
                    _max_n_now, _max_reason = _adaptive_n(100)
                    batch_telemetry_text = (
                        f"recent median: {_rate:.0f}s/question over "
                        f"{_bs['n_recent']} batches\n"
                        f"current budget: {_budget_s:.0f}s "
                        f"(SEEDING_BATCH_BUDGET_S)\n"
                        f"realistic max n_questions right now: ~{_max_n_now} "
                        f"(scaler reason: {_max_reason})\n"
                        f"recent batches: "
                        + json.dumps(_bs.get("recent", []), separators=(",", ":"))
                        + "\n"
                        + "IMPORTANT: request seed_batch sizes that fit the "
                        "budget. Asking for 10 questions when the realistic "
                        "max is 3 just gets silently scaled down — better to "
                        "ask for 3 directly so you keep clean attribution."
                    )
            except Exception as _exc:
                batch_telemetry_text = f"(telemetry unavailable: {_exc})"

            # 2026-05-23 planner enrichment — four new sections injected
            # into the prompt:
            #   1. pareto_geometry — frontier shape + blocking points + gaps
            #   2. journal_trustworthiness — bug-corrupted ratio + low-signal flag
            #   3. hypotheses_under_test — last 3 trustworthy trials' hypotheses
            #   4. insights_structured — per-action-type observations + confidence
            #   5. stagnation_signal + exploration_block — stagnation-gated
            #      lean/rich creative-exploration fragment (2026-05-23 upgrade)
            try:
                trust = journal.trustworthiness_score()
                trust_lines = [
                    f"total trials: {trust['total']}",
                    f"trustworthy:  {trust['trustworthy']}",
                    f"corrupted:    {trust['corrupted']}",
                    f"ratio:        {trust['ratio']:.1%}",
                ]
                if trust["corrupted_by"]:
                    trust_lines.append(
                        "corrupted-by breakdown: "
                        + ", ".join(f"{sha}:{n}" for sha, n in trust["corrupted_by"].items())
                    )
                if trust["low_signal"]:
                    trust_lines.append(
                        "WARNING: trustworthy < 5 — hypothesis-chain reasoning has low "
                        "signal. Prefer EXPLORE actions (seed_batch, structural_experiment, "
                        "tail-sampled creative actions) over EXPLOIT (rollback, "
                        "small numeric perturbations) until trustworthiness exceeds 5."
                    )
                # Convergence vs corruption (2026-05-31). reproduction_confirmed
                # trials REPRODUCE an already-kept above-baseline gain: they are
                # measurement-VALID confirmations (the planner has converged on a
                # surface), deliberately NOT counted in `corrupted` above. Surface
                # the count so the planner reads convergence, not "wasted trials".
                try:
                    _recent = journal.recent(10)
                    _repro = sum(
                        1 for e in _recent
                        if getattr(e, "deficiency_category", "") == "reproduction_confirmed"
                    )
                except Exception:
                    _repro = 0
                if _repro:
                    trust_lines.append(
                        f"reproduction_confirmed (convergence, last 10): {_repro} — these "
                        "reproduce an established above-baseline gain and are TRUSTWORTHY "
                        "confirmations (config converged on this surface), NOT noise or "
                        "corruption. Do not 'wait out a noise window' or re-run for a fresh "
                        "win; explore a NEW surface or idle cleanly."
                    )
                # Attribution guard (C, 2026-05-31): every exclusion above is a
                # measurement-VALID classification, NOT evidence of host noise or a
                # broken eval tower. The planner must not invent an "exogenous
                # host-load noise window" from exclusions alone (the 2026-05-31
                # meta-action loop did exactly that).
                trust_lines.append(
                    "ATTRIBUTION GUARD: corrupted-by tags are classifications "
                    "(mad_noise=within-noise improvement; reproduction_confirmed="
                    "convergence; a commit SHA=operator code-fix invalidation; "
                    "exogenous_operator_reload=service restart). NONE of these are "
                    "evidence of host/exogenous eval-noise or a broken instrument. Do "
                    "NOT claim a host-load 'noise window' or escalate 'eval tower stuck' "
                    "unless the host-health line below shows an actual throttle/cache/load "
                    "signal; absent that, treat repeated reproduction_confirmed as "
                    "CONVERGENCE and move to a new surface."
                )
                try:
                    from host_health import HostHealthState  # type: ignore
                    _hh = HostHealthState.snapshot()
                    _throttled, _trig = _hh.is_throttled()
                    _mem_warnings = _hh.memory_residency_warnings()
                    trust_lines.append(
                        "host-health: "
                        + ("THROTTLED — " + "; ".join(_trig)
                           if _throttled else
                           "nominal (no CPU-throttle / page-cache / load signal "
                           "→ a host-noise narrative is UNSUPPORTED)")
                    )
                    if _mem_warnings:
                        trust_lines.append(
                            "memory-residency: ADVISORY — "
                            + "; ".join(_mem_warnings)
                            + "; do not use drop_caches for this class of RAM"
                        )
                except Exception:
                    trust_lines.append(
                        "host-health: not collected this turn — no positive evidence of "
                        "host noise; do not assume a noise window exists."
                    )
                journal_trustworthiness_text = "\n".join(trust_lines)
            except Exception as _exc:
                journal_trustworthiness_text = f"(trustworthiness unavailable: {_exc})"

            try:
                hyps = journal.recent_hypotheses(n=3, exclude_bug_corrupted=True)
                if not hyps:
                    hypotheses_text = "(no recent trustworthy hypotheses)"
                else:
                    hyp_lines: list[str] = []
                    for e in hyps:
                        outcome = (
                            f"q={e.quality:.3f} sp={e.speed:.1f} → {e.pareto_status}"
                        )
                        hyp_lines.append(
                            f"#{e.trial_id} ({e.species}/{e.action_type}):\n"
                            f"  Hypothesis: {(e.hypothesis or '(none)')[:240]}\n"
                            f"  Outcome:    {outcome}"
                            + (f"\n  Self-criticism: {scrub_legacy_scale_text(e.self_criticism)[:200]}" if e.self_criticism else "")
                        )
                    hypotheses_text = "\n\n".join(hyp_lines)
            except Exception as _exc:
                hypotheses_text = f"(hypothesis chain unavailable: {_exc})"

            try:
                insights_structured_text = journal.insights_structured_text(
                    n=30, exclude_bug_corrupted=True
                )
            except Exception as _exc:
                insights_structured_text = f"(structured insights unavailable: {_exc})"

            try:
                pareto_geometry_text = archive.geometry_text(tier=DEFAULT_FRONTIER_TIER)
            except Exception as _exc:
                pareto_geometry_text = f"(geometry unavailable: {_exc})"

            try:
                _known_actions = [
                    "seed_batch", "numeric_trial", "prompt_mutation",
                    "gepa_optimize", "code_mutation", "structural_experiment",
                    "structural_prune", "slot_compact", "train_routing_models",
                    "distill_skillbank", "reset_memories", "deep_eval",
                    "rollback", "distill_knowledge",
                ]
                action_availability_text, viable_tail_actions = _build_action_availability(
                    journal=journal,
                    known_actions=_known_actions,
                    memory_count=memory_count,
                    converged=converged,
                    slot_memory_text=slot_memory_text,
                    blacklist=blacklist,
                )
                exploration_block, stagnation_signal = _build_exploration_block(
                    journal=journal,
                    archive=archive,
                    known_actions=viable_tail_actions,
                )
            except Exception as _exc:
                exploration_block = (
                    "Briefly enumerate 3–5 alternatives with one-line reject/accept "
                    "reasons before committing to your single action.\n"
                    f"(exploration-block assembly failed: {_exc})"
                )
                action_availability_text = "(action availability unavailable)"
                stagnation_signal = "unknown"

            prompt = CONTROLLER_PROMPT_TEMPLATE.format(
                program=program_text,
                pareto_summary=archive.summary_text(tier=DEFAULT_FRONTIER_TIER),
                pareto_geometry=pareto_geometry_text,
                journal_trustworthiness=journal_trustworthiness_text,
                hypotheses_under_test=hypotheses_text,
                journal_summary=journal.summary_text(20),
                seeder_status=json.dumps(seeder.convergence_status(), indent=2),
                batch_telemetry=batch_telemetry_text,
                species_effectiveness=json.dumps(
                    journal.species_effectiveness(), indent=2
                ),
                health_status="OK" if not dry_run else "dry_run",
                memory_count=memory_count,
                converged=converged,
                slot_memory=slot_memory_text,
                action_availability=action_availability_text,
                budget=json.dumps(meta.budget.as_dict(), indent=2),
                suite_quality_trends=_format_suite_trends(journal.suite_quality_trend(10)),
                insights=insights_text,
                insights_structured=insights_structured_text,
                stagnation_signal=stagnation_signal,
                exploration_block=exploration_block,
                short_term_memory=memory.to_text(),  # AP-22
                last_criticism=last_criticism_text,  # AP-23
                model_signatures=model_signatures_text,
                blacklist_text=blacklist_text,
                feature_flags_block=_build_feature_flags_block(lab),
                last_invalid_feedback=_build_last_invalid_feedback(state),
                code_targets=", ".join(CODE_MUTATION_ALLOWLIST),
                plot_paths="\n".join(f"  - {p}" for p in plot_paths) or "  (none yet)",
            ) + peaf.peaf_prompt_addendum()

            phase.set(
                "planner_invoke",
                trial_id=trial_counter,
                prompt_chars=len(prompt),
                session_id=state.get("session_id"),
                idle_reason="planner subprocess running",
            )
            planner_provider_state = state.setdefault("planner_providers", {})
            if not isinstance(planner_provider_state, dict):
                planner_provider_state = {}
                state["planner_providers"] = planner_provider_state
            planner_decision = plan_with_providers(
                prompt,
                session_id=state.get("session_id"),
                cwd=ORCH_ROOT,
                planner_state=planner_provider_state,
                stagnation_signal=stagnation_signal,
            )
            phase.set(
                "planner_parse",
                trial_id=trial_counter,
                response_chars=len(planner_decision.canonical_text or ""),
                session_id=planner_decision.session_id,
                draft_provider=planner_decision.draft_provider,
                critic_provider=planner_decision.critic_provider,
                planner_mode=planner_decision.mode,
                degraded=planner_decision.degraded,
                fallback_reason=planner_decision.fallback_reason,
            )
            if planner_decision.fallback_reason:
                log.warning(
                    "Planner fallback/degraded mode: %s",
                    planner_decision.fallback_reason,
                )
            if planner_decision.critique is not None:
                log.info(
                    "Planner critique by %s: %s confidence=%.2f issues=%s",
                    planner_decision.critique.provider or "(none)",
                    planner_decision.critique.decision,
                    planner_decision.critique.confidence,
                    planner_decision.critique.issues,
                )
            state["session_id"] = planner_decision.session_id
            action = planner_decision.action
            predicted_objectives = planner_decision.predicted_objectives
            rationale = planner_decision.rationale

            # draft_critique authority: a BINDING reject/revise substituted the
            # planner's draft (safe fallback / revised_action is what `action`
            # now holds). Record the rejected draft so it cannot bypass the
            # invalid-action feedback + blacklist — otherwise the planner could
            # re-draft the same rejected action forever while the critic
            # silently swaps in the fallback. The substituted `action` still
            # flows through meta/quota/blacklist/dispatch below unchanged.
            crit = planner_decision.critique
            draft_action = planner_decision.draft_action
            if (
                planner_decision.mode == "draft_critique"
                and crit is not None
                and crit.decision in ("reject", "revise")
                and draft_action
                and draft_action != action
            ):
                _record_rejected_draft(state, draft_action, crit, trial_counter)
                blacklist = load_blacklist()  # may have grown
                if (
                    int(state.get("consecutive_rejected_drafts", 0))
                    >= MAX_CONSECUTIVE_REJECTED_DRAFTS
                ):
                    # Durable halt mirroring the meta/skip breakers: the planner
                    # keeps drafting actions the critic overrides — operator call.
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "critic_reject_loop"
                    save_state(state)
                    log.error(
                        "Critic rejected/revised %d consecutive planner drafts — "
                        "pausing for operator review (stuck planner the critic "
                        "keeps overriding).",
                        int(state.get("consecutive_rejected_drafts", 0)),
                    )
                    phase.set("critic_reject_loop_halt", trial_id=trial_counter)
                    break
            else:
                # Draft accepted (approve / not critiqued / no substitution).
                state["consecutive_rejected_drafts"] = 0
        else:
            # Autonomous mode: species selection by budget
            phase.set("autonomous_select", trial_id=trial_counter)
            species = meta.select_species()
            action = _auto_action(species, memory_count, converged, seeder)
            predicted_objectives = {}  # PEAF: autonomous mode has no controller forecast
            rationale = {"falsifier": "", "rubric_scores": {}}  # no controller call
            stagnation_signal = ""  # gate is controller-only; autonomous mode skips it
            state["consecutive_rejected_drafts"] = 0  # no critic in autonomous mode

        if not action:
            log.warning("No action proposed, defaulting to seed_batch")
            action = {"type": "seed_batch", "n_questions": 10}

        # Meta actions are allowed as occasional bookkeeping, but a repeated
        # metric-free action means the planner is avoiding the experiment loop.
        action, rationale = _force_metric_action_after_meta(action, state, rationale)

        # Experiment quota: once memory is large, cap consecutive passive
        # (seed/distill) actions so the planner cannot rationalize no-op work
        # forever — force a frontier-moving experiment instead.
        action, rationale = _enforce_experiment_quota(
            action, state, memory_count, rationale, trial_counter,
        )

        # ── 3. Act ───────────────────────────────────────────────
        # B2: Check failure blacklist before dispatch
        blocked_reason = check_blacklist(action, blacklist)
        if blocked_reason:
            log.warning(
                "Trial %d: action blacklisted (%s), requesting new action",
                trial_counter, blocked_reason,
            )
            action = {"type": "seed_batch", "n_questions": 10}

        log.info("Trial %d: %s", trial_counter, json.dumps(action))
        phase.set(
            "action_selected",
            trial_id=trial_counter,
            action_type=action.get("type", ""),
        )

        # Update TUI with trial info
        if tui is not None:
            species_hint = action.get("type", "unknown")
            tui.set_trial(trial_counter, species_hint)
            # Show the action description as the "current prompt" in TUI
            prompt_preview = action.get("description", "")
            if not prompt_preview:
                prompt_preview = json.dumps(action, indent=2)[:500]
            tui.set_prompt(prompt_preview)

        # Phase 6b — write in_flight_trial marker BEFORE dispatch_action.
        # Atomic save_state (Phase 6a) guarantees this either lands fully
        # or not at all. The marker is cleared after the final save_state
        # at the end of the trial; if autopilot crashes in between, the
        # next startup's recovery block sees the marker and either:
        #   - finds the trial in the journal (case a) → re-syncs counter
        #   - finds nothing (case b) → writes AUTOPILOT_KILLED placeholder
        # Both cases prevent silent corruption of the planner's view.
        state["in_flight_trial"] = {
            "trial_id": trial_counter,
            "action": action,
            "started_at": time.time(),
            "host_pid": os.getpid(),
            "host_started_at": state.get("autopilot_fleet_started_at"),
        }
        save_state(state)

        if dry_run:
            phase.set("dispatch_dry_run", trial_id=trial_counter, action_type=action.get("type", ""))
            eval_result = EvalResult(
                tier=0, quality=2.5, speed=15.0, cost=0.3, reliability=0.95
            )
            species_name = action.get("type", "unknown").split("_")[0]
        else:
            phase.set(
                "dispatch_action",
                trial_id=trial_counter,
                action_type=action.get("type", ""),
                idle_reason="running selected action",
            )
            eval_result, species_name = dispatch_action(
                action, seeder, swarm, forge, lab, tower, gate, archive,
                journal, state, strategy_store=strategy_store, evo=evo,
                watcher=watcher,
            )
            phase.set(
                "dispatch_complete",
                trial_id=trial_counter,
                action_type=action.get("type", ""),
                species=species_name,
            )

        # ── 4. Evaluate ─────────────────────────────────────────
        if eval_result is None or isinstance(eval_result, SkipOutcome):
            action_type = action.get("type", "")
            if eval_result is None and action_type in META_NOOP_ACTIONS:
                # Meta/housekeeping action: it executed its effect but collected
                # no metrics, so it is NOT an independent trial — do not consume a
                # trial number. Track consecutive meta actions so a stuck planner
                # that only emits no-ops (gate-lock symptom) halts loudly instead
                # of spinning the counter forever.
                meta_streak = int(state.get("consecutive_meta_actions", 0)) + 1
                state["consecutive_meta_actions"] = meta_streak
                state["in_flight_trial"] = None
                save_state(state)
                log.info(
                    "Meta action '%s' executed (no eval, no metrics) — not counted "
                    "as a trial (consecutive meta=%d, trial stays at %d)",
                    action_type, meta_streak, trial_counter,
                )
                if meta_streak >= MAX_CONSECUTIVE_META:
                    # Durable, terminal-until-operator-resume halt. Previously
                    # this only `break`-ed the inner loop; an external supervisor
                    # then restarted the process straight back into the same
                    # gate-locked state (consecutive_meta_actions persisted on
                    # disk), so it re-halted on meta action #1 (2026-05-31).
                    # Setting paused=True LATCHES the halt: the iteration-top
                    # paused gate idles this process AND any restart until the
                    # operator explicitly resumes (clearing paused resets the
                    # counter — see top of loop), so a supervisor cannot re-enter
                    # the no-op loop.
                    halt_reason = _classify_meta_halt(journal)
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "meta_action_loop"
                    state["_meta_halt_reason"] = halt_reason
                    save_state(state)
                    if halt_reason == "converged":
                        log.warning(
                            "Planner emitted %d consecutive metric-free meta actions; "
                            "recent metric trials are reproduction-confirmed — the planner "
                            "has CONVERGED on the current surface (benign, not a malfunction "
                            "and not instrument noise). Pausing cleanly; resume after "
                            "pointing it at a NEW surface.",
                            meta_streak,
                        )
                    else:
                        log.error(
                            "Planner emitted %d consecutive metric-free meta actions "
                            "without running a single trial — pausing for operator review "
                            "(likely a gate-lock or stuck planner).",
                            meta_streak,
                        )
                    phase.set(
                        "meta_loop_halt",
                        trial_id=trial_counter,
                        meta_streak=meta_streak,
                        halt_reason=halt_reason,
                    )
                    break
                continue

            # Non-executing action (invalid flags / AP-9 scope / dirty-tree
            # fence / unknown type / handler no-op). This was the blind spot: the
            # old path bumped the counter and `continue`d, discarding the reason —
            # so the planner re-sampled an impossible action 119× until max_trials
            # (the graph_router deadlock). Now it leaves durable residue: the
            # reason is journaled, fingerprinted, counted, fed back to the next
            # planner prompt, blacklisted on repeat, and circuit-broken on a run.
            if isinstance(eval_result, SkipOutcome):
                skip_status = eval_result.status
                skip_reason = eval_result.reason
            else:
                skip_status = "skipped"
                skip_reason = f"{action_type} returned no eval result (handler no-op)"

            sig = _action_signature(action)
            sig_counts = state.setdefault("invalid_signature_counts", {})
            sig_counts[sig] = int(sig_counts.get(sig, 0)) + 1
            skip_streak = int(state.get("consecutive_skip_actions", 0)) + 1
            state["consecutive_skip_actions"] = skip_streak
            state["last_invalid_action"] = action
            state["last_invalid_reason"] = skip_reason
            state["last_invalid_status"] = skip_status

            try:
                _record_skip_trial(
                    journal, trial_counter, action, species_name,
                    skip_status, skip_reason, memory_count,
                )
            except Exception:
                log.debug("skip-trial journal write failed", exc_info=True)

            log.warning(
                "Trial %d %s (%s): %s [signature seen %d×, consecutive skips=%d]",
                trial_counter, skip_status, action_type, skip_reason,
                sig_counts[sig], skip_streak,
            )
            phase.set(
                "dispatch_skip",
                trial_id=trial_counter,
                action_type=action_type,
                skip_status=skip_status,
            )

            # Auto-blacklist a repeated INVALID signature (stable validator
            # reason → safe to pattern-match). "skipped" outcomes are NOT
            # blacklisted: their coarse pattern would over-match (e.g. one
            # scope-violating numeric_trial would ban all numeric_trials).
            if (
                skip_status == "invalid"
                and sig_counts[sig] >= INVALID_SIGNATURE_BLACKLIST_THRESHOLD
            ):
                append_blacklist(
                    action, trial_counter,
                    f"Auto-blacklisted: {sig_counts[sig]}× invalid — {skip_reason[:80]}",
                )
                blacklist = load_blacklist()

            trial_counter += 1
            state["trial_counter"] = trial_counter
            state["in_flight_trial"] = None
            save_state(state)

            # Hard circuit-breaker mirroring MAX_CONSECUTIVE_META: a run of
            # non-executing actions means a stuck planner / impossible action
            # space. Latch paused=True so a supervisor restart cannot re-enter
            # the loop (same durable-halt semantics as the meta-action guard).
            if skip_streak >= MAX_CONSECUTIVE_SKIP:
                state["paused"] = True
                state["_dispatch_deficiency"] = "skip_action_loop"
                save_state(state)
                log.error(
                    "Planner emitted %d consecutive non-executing actions "
                    "(last: %s — %s); pausing for operator review (stuck planner "
                    "or impossible action space).",
                    skip_streak, action_type, skip_reason,
                )
                phase.set(
                    "skip_loop_halt",
                    trial_id=trial_counter,
                    skip_streak=skip_streak,
                )
                break
            continue

        # AP-13: Emit grep-parseable metrics
        phase.set("safety_gate", trial_id=trial_counter, species=species_name)
        log.info("\n%s", eval_result.to_grep_lines(trial_counter, species_name))

        # ── Pre-gate exogenous-reload classification (handoff Phase 5) ──
        # If the trial picked up an operator/external service reload (and at
        # least one question stayed unrecovered after retry), classify it as
        # bug-corrupted BEFORE running the safety gate or admitting it to the
        # Pareto archive. The aggregate quality/reliability numbers are
        # partially based on missing data; treating it as a real failure would
        # increment consecutive_failures + trigger spurious rollback + admit
        # a (0, 0, ?, ?) point to the archive.
        #
        # Trials that detected exogenous reloads but RECOVERED via retry are
        # sound — they just carry audit metadata in eval_details so the
        # operator can see this trial weathered a reload cleanly.
        has_exo_unrecovered = (
            getattr(eval_result, "n_exogenous_unrecovered", 0) > 0
        )
        has_exo_recovered = (
            getattr(eval_result, "n_exogenous_recovered", 0) > 0
        )

        if has_exo_unrecovered:
            # Bypass safety gate + archive update. Trial is journaled below
            # as a bug-corrupted placeholder for audit; the planner's
            # trustworthiness gate excludes it from learning surfaces.
            from safety_gate import SafetyVerdict  # type: ignore
            verdict = SafetyVerdict(passed=True)
            failure_analysis = (
                f"Excluded: unrecovered operator/service reload during trial "
                f"({eval_result.n_exogenous_unrecovered}/{eval_result.n_questions} "
                f"questions affected)"
            )
            log.warning(
                "Trial %d skipped safety gate + archive: %s",
                trial_counter, failure_analysis,
            )
        else:
            # Safety gate
            verdict = gate.check(eval_result)
            failure_analysis = gate.analyze_failure(eval_result, verdict)
            if not verdict:
                log.warning(
                    "Safety violations: %s", "; ".join(verdict.violations)
                )
                if gate.should_rollback():
                    log.error("Consecutive failure limit reached, rolling back")
                    state["_dispatch_deficiency"] = "consecutive_failures"  # AP-14
                    # B2: Auto-append failing config to blacklist
                    append_blacklist(
                        action, trial_counter,
                        f"Auto-blacklisted: 3 consecutive failures ending at trial {trial_counter}",
                    )
                    blacklist = load_blacklist()  # Reload after append
                    lab.restore_checkpoint()
                    gate.reset_failures()

        # ── 4b. Self-Criticism (AP-23/AP-24) ────────────────────
        phase.set("self_criticism", trial_id=trial_counter, species=species_name)
        # Get baseline and previous per-suite for comparison
        baseline_q = gate.baseline.quality_for_tier(eval_result.tier) if gate.baseline else 0.0
        prev_suite = {}
        recent = journal.by_species(species_name)
        if recent:
            prev_details = recent[-1].eval_details
            if isinstance(prev_details, dict):
                prev_suite = prev_details.get("per_suite_quality", {})

        prior_criticism_text = last_criticism_text
        criticism = generate_self_criticism(
            action=action,
            eval_result=eval_result,
            verdict=verdict,
            failure_analysis=failure_analysis,
            baseline_quality=baseline_q,
            prev_per_suite=prev_suite,
        )
        last_criticism_text = criticism.as_text()

        # ── 5. Record ────────────────────────────────────────────
        phase.set("record_trial", trial_id=trial_counter, species=species_name)
        # Classify whether this trial should be excluded from learning
        # surfaces (Pareto archive + AP-22 short-term memory). Two paths:
        # exogenous reload (Phase 5, partially-missing data) and MAD-noise
        # improvements (intake-421, noise inflates strategy memory). Both
        # tag bug_corrupted_by so the planner's trustworthiness gate
        # excludes the trial; both skip archive.update so the Pareto
        # frontier isn't distorted; both still journal so the operator can
        # audit. See classify_learning_exclusion() near the top of this
        # module for the priority order + reason strings.
        learning_excluded_by, learning_excluded_reason, exclusion_def_cat = (
            classify_learning_exclusion(verdict, eval_result)
        )
        # Policy correction (2026-06-04 — see autopilot-continuous-optimization handoff):
        # a TRUSTED within-quality-noise measurement (mad_noise / reproduction_confirmed)
        # stays excluded from AP-22 / strategy learning, but must NOT be auto-excluded from
        # the MULTI-OBJECTIVE Pareto archive. The MAD test is quality-only; a trial flat on
        # quality can still be NON-DOMINATED on speed / cost / reliability and belongs on the
        # frontier. Admit ONE representative point per stable config fingerprint, dominance
        # tested on robust-MEDIAN objectives across the reproduction cluster (so a lucky
        # single-trial speed sample / host-throughput variance can't manufacture frontier
        # geometry). This also subsumes the old empty-frontier bootstrap: on an empty tier the
        # first trusted within-noise point becomes the seed via the same median path. AP-22
        # suppression remains tied to the mad_noise/reproduction_confirmed tag (criticism
        # below). Non-trusted exclusions (exogenous reload, etc.) still skip entirely.
        if (
            learning_excluded_by in ("mad_noise", "reproduction_confirmed")
            and eval_result.tier >= MIN_FRONTIER_EVAL_TIER
        ):
            fingerprint = _config_fingerprint(action)
            pareto_status, rep_objs = archive.upsert_representative(
                fingerprint,
                eval_result.tier,
                objectives_from(eval_result),
                trial_id=trial_counter,
                config_snapshot=action,
                species=species_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                memory_count=memory_count,
                reasoning=json.dumps(action),
            )
            log.info(
                "Trial %d: Pareto representative admission %s (T%d fp=%s n=%d) "
                "median_objs=%s — AP-22/strategy learning still excluded (%s)",
                trial_counter, pareto_status, eval_result.tier, fingerprint,
                archive.reproduction_count(eval_result.tier, fingerprint),
                [round(float(x), 3) for x in rep_objs], learning_excluded_by,
            )
            criticism = learning_exclusion_criticism(
                learning_excluded_by, learning_excluded_reason
            )
        elif learning_excluded_by:
            pareto_status = "dominated"  # placeholder for JournalEntry only
            log.info(
                "Trial %d: archive.update SKIPPED (learning_excluded_by=%s)",
                trial_counter, learning_excluded_by,
            )
            criticism = learning_exclusion_criticism(
                learning_excluded_by, learning_excluded_reason
            )
        elif not verdict.passed:
            # Safety verdict FAILED and this is not a benign within-noise exclusion
            # (e.g. a genuine per-suite regression at adequate n, possibly co-tagged
            # mad_noise on the quality axis — see classify_learning_exclusion). A
            # failed trial must NEVER clean-update the Pareto archive or raise the
            # baseline. It is a real failed experiment, NOT corrupted data, so we
            # leave bug_corrupted_by unset; the deficiency category is taken from
            # verdict.categories below. (2026-06-06: previously such a trial fell
            # through to the clean-update branch and could admit/raise on a failure.)
            pareto_status = "dominated"  # placeholder for JournalEntry only
            log.info(
                "Trial %d: archive.update SKIPPED (safety verdict failed: %s)",
                trial_counter, ", ".join(verdict.categories) or "unspecified",
            )
        else:
            pareto_status = archive.update(
                ParetoEntry(
                    trial_id=trial_counter,
                    objectives=objectives_from(eval_result),
                    config_snapshot=action,
                    species=species_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    eval_tier=eval_result.tier,
                    memory_count=memory_count,
                    reasoning=json.dumps(action),
                )
            )
            baseline_update = gate.update_baseline(eval_result, source_trial_id=trial_counter)
            if baseline_update.updated:
                log.info(
                    "Trial %d: T%d baseline auto-raised %.3f → %.3f",
                    trial_counter,
                    baseline_update.tier,
                    baseline_update.previous_quality or 0.0,
                    baseline_update.new_quality,
                )
            else:
                log.info(
                    "Trial %d: baseline update skipped (%s)",
                    trial_counter,
                    baseline_update.reason,
                )

        # Extract hypothesis and expected mechanism from action/controller
        hypothesis = action.get("description", "")
        # AP-15: Fallback for species that don't provide description
        if not hypothesis:
            action_type = action.get("type", "")
            if action_type == "seed_batch":
                hypothesis = f"Seed {action.get('n_questions', 10)} questions across {action.get('suites', 'all')}"
            elif action_type == "numeric_trial":
                hypothesis = f"Optimize {action.get('surface', 'unknown')} surface"
            elif action_type == "structural_experiment":
                hypothesis = f"Toggle flags: {action.get('flags', {})}"
            elif action_type in ("train_routing_models", "distill_skillbank", "rollback"):
                hypothesis = action_type.replace("_", " ").title()
        expected_mechanism = (
            action.get("mutation", "") or action.get("surface", "") or action.get("type", "")
        )

        # B1: Store strategy on Pareto frontier improvements
        if pareto_status == "frontier" and strategy_store is not None:
            try:
                strategy_store.store(
                    description=f"{action.get('type', '')}: {hypothesis}",
                    insight=f"q={eval_result.quality:.3f} s={eval_result.speed:.1f} mechanism={expected_mechanism}",
                    source_trial_id=trial_counter,
                    species=species_name,
                )
            except Exception as e:
                log.warning("Strategy store write failed: %s", e)

        # B3: Capture execution traces for next PromptForge iteration
        recent_trace_text = state.get("last_traces", "")
        if not dry_run:
            recent_trace_text = tower.capture_recent_traces(50)
            state["last_traces"] = recent_trace_text

        # J9/HLE-4: observe-only harness metrics. Compute after SafetyGate and
        # ParetoArchive decisions so these diagnostics cannot affect current
        # trial acceptance or baseline mutation.
        if not getattr(eval_result, "harness_metrics", None):
            from hle_metrics import compute_hle_observe_payload  # type: ignore

            hle_payload = compute_hle_observe_payload(
                eval_result,
                action=action,
                verdict=verdict,
                failure_analysis=failure_analysis,
                prior_criticism=prior_criticism_text,
                recent_traces=recent_trace_text,
            )
            eval_result.metric_schema_version = hle_payload["metric_schema_version"]
            eval_result.harness_metrics = hle_payload["harness_metrics"]
            eval_result.oracle_adequacy = hle_payload["oracle_adequacy"]

        # Git tag
        phase.set("post_trial_artifacts", trial_id=trial_counter, species=species_name)
        git_tag = ""
        if not dry_run:
            git_tag = f"autopilot/trial-{trial_counter}"
            _git_tag(git_tag, f"Trial {trial_counter}: {species_name}/{action.get('type', '')}")

        # Compute trial lineage (AP-3): find most recent trial from same species
        parent_trial_id = None
        config_diff: dict[str, Any] = {}
        species_history = journal.by_species(species_name)
        if species_history:
            parent = species_history[-1]
            parent_trial_id = parent.trial_id
            # Compute config diff: keys that changed between parent and current
            prev_cfg = parent.config_snapshot
            for key in set(list(prev_cfg.keys()) + list(action.keys())):
                old_val = prev_cfg.get(key)
                new_val = action.get(key)
                if old_val != new_val:
                    config_diff[key] = {"old": old_val, "new": new_val}

        # Build active_flags from action context
        active_flags_dict = action.get("flags", {})
        active_flags_list = [
            f"{k}={v}" for k, v in active_flags_dict.items()
        ] if active_flags_dict else []

        # AP-14: Extract deficiency category from safety verdict + dispatch side channel
        deficiency_category = ""
        if not verdict.passed:
            deficiency_category = verdict.categories[0] if verdict.categories else ""
        if not deficiency_category:
            deficiency_category = state.pop("_dispatch_deficiency", "")

        # Apply the learning-exclusion decision computed above. Both exogenous
        # reload and mad_noise produce a single bug_corrupted_by tag + an
        # eval_details["learning_exclusion"] audit record. The deficiency
        # category is overridden so the planner can distinguish exclusion
        # reasons from genuine safety-gate failures.
        # Benign convergence exclusions (reproduction_confirmed) skip the Pareto
        # archive (via learning_excluded_by above) but must NOT populate
        # bug_corrupted_by — otherwise trustworthiness_score() and the journal
        # trust render would treat a valid confirmation like a kill / reload /
        # commit-invalidation, and the planner would narrate a "noisy instrument"
        # (2026-05-31 incident → meta-action loop).
        if learning_excluded_by and learning_excluded_by not in BENIGN_LEARNING_EXCLUSIONS:
            bug_corrupted_by = learning_excluded_by
            bug_corrupted_reason = learning_excluded_reason
        else:
            bug_corrupted_by = ""
            bug_corrupted_reason = ""
        metric_schema_version = getattr(eval_result, "metric_schema_version", 1)
        harness_metrics = getattr(eval_result, "harness_metrics", {}) or {}
        oracle_adequacy = getattr(eval_result, "oracle_adequacy", {}) or {}
        eval_details_dict: dict[str, Any] = {
            "per_suite_quality": eval_result.per_suite_quality,
            "routing_distribution": eval_result.routing_distribution,
            "details": eval_result.details,
            "metric_schema_version": metric_schema_version,
            "harness_metrics": harness_metrics,
            "oracle_adequacy": oracle_adequacy,
            "speed_metric_mode": getattr(eval_result, "speed_metric_mode", "median_request_tps"),
            "median_request_speed": getattr(eval_result, "median_request_speed", 0.0),
            "aggregate_speed": getattr(eval_result, "aggregate_speed", 0.0),
            "eval_concurrency": getattr(eval_result, "eval_concurrency", 1),
            "eval_wall_s": getattr(eval_result, "eval_wall_s", 0.0),
            "sum_request_elapsed_s": getattr(eval_result, "sum_request_elapsed_s", 0.0),
            "ece": eval_result.ece,
            "auroc": eval_result.auroc,
            "calibration_violations": eval_result.calibration_violations,
            "gepa_ratio": state.get("gepa_ratio", 0.30),
            # Tool-use telemetry (2026-06-01) — surfaced at top level so the planner
            # can measure and incentivize productive tool use. tokens_generated/speed
            # already credit tool-turn generation; these are the explicit signal.
            "mean_tools_used": getattr(eval_result, "mean_tools_used", 0.0),
            "tool_use_rate": getattr(eval_result, "tool_use_rate", 0.0),
            "total_tool_calls": getattr(eval_result, "total_tool_calls", 0),
            # The decision-grade signal: marginal usefulness of tools (NaN until
            # enough sample). Steer by THIS, not raw tool_use_rate. Now computed
            # per-suite then averaged, so trivially-correct no-tool suites can't
            # contaminate the sign (was −0.4 cross-suite at cutover for all-passing
            # tool calls). Per-suite breakdown below for the suite under test.
            "tool_helpfulness": getattr(eval_result, "tool_helpfulness", float("nan")),
            "per_suite_tool_helpfulness": getattr(eval_result, "per_suite_tool_helpfulness", {}),
        }
        if learning_excluded_by:
            deficiency_category = exclusion_def_cat
            eval_details_dict["learning_exclusion"] = {
                "by": learning_excluded_by,
                "reason": learning_excluded_reason,
            }
        if has_exo_unrecovered or has_exo_recovered:
            # Surface audit info regardless — recovered trials carry this
            # so the operator can see "this trial weathered a reload"
            # without the planner downgrading it.
            eval_details_dict["exogenous_retries"] = {
                "n_recovered": getattr(eval_result, "n_exogenous_recovered", 0),
                "n_unrecovered": getattr(eval_result, "n_exogenous_unrecovered", 0),
                "n_external_restart": getattr(eval_result, "n_external_restart", 0),
                "question_ids": list(getattr(eval_result, "exogenous_question_ids", [])),
                "marker_observations": list(getattr(eval_result, "exogenous_marker_log", [])),
            }

        journal.record(
            JournalEntry(
                trial_id=trial_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                species=species_name,
                action_type=action.get("type", ""),
                tier=eval_result.tier,
                quality=eval_result.quality,
                speed=eval_result.speed,
                cost=eval_result.cost,
                reliability=eval_result.reliability,
                pareto_status=pareto_status,
                git_tag=git_tag,
                config_snapshot=action,
                config_diff=config_diff,
                parent_trial=parent_trial_id,
                memory_count=memory_count,
                active_flags=active_flags_list,
                failure_analysis=failure_analysis,
                eval_details=eval_details_dict,
                metric_schema_version=metric_schema_version,
                harness_metrics=harness_metrics,
                oracle_adequacy=oracle_adequacy,
                reasoning=json.dumps(action),
                hypothesis=hypothesis,
                expected_mechanism=expected_mechanism,
                deficiency_category=deficiency_category,
                instruction_token_count=eval_result.instruction_token_count,
                instruction_token_ratio=eval_result.instruction_token_ratio,
                self_criticism=criticism.as_text(),  # AP-23
                keep_revert_decision=criticism.keep_or_revert,  # AP-24
                optimization_directions=criticism.directions_text(),  # AP-24
                predicted_objectives=predicted_objectives,  # PEAF (intake-571 spike)
                surprise_score=peaf.compute_surprise(
                    predicted_objectives, peaf.actual_objectives_from_eval(eval_result)
                ),
                falsifier=rationale.get("falsifier", ""),
                rubric_scores=rationale.get("rubric_scores", {}),
                stagnation_signal=stagnation_signal,
                bug_corrupted_by=bug_corrupted_by,
                bug_corrupted_reason=bug_corrupted_reason,
            )
        )

        # AP-16: Track last instruction ratio for structural pruning comparison
        state["_last_instruction_ratio"] = eval_result.instruction_token_ratio

        # AP-22: Update short-term memory with trial outcome.
        # Skip when learning_excluded_by is set — the trial is journaled
        # for audit, but its outcome must not feed strategy memory
        # (exogenous reload = partially-missing data; mad_noise =
        # noise-level improvement that would inflate the memory with
        # false positives per intake-421). Behavior change vs pre-2026-05-27:
        # exogenous-reload trials used to update AP-22; they no longer do.
        if not learning_excluded_by:
            memory.update(TrialOutcome(
                trial_id=trial_counter,
                species=species_name,
                action_type=action.get("type", ""),
                quality=eval_result.quality,
                speed=eval_result.speed,
                passed=verdict.passed,
                hypothesis=hypothesis,
                failure_analysis=failure_analysis,
                self_criticism=criticism.as_text(),
                optimization_directions=criticism.directions_text(),
                keep_revert=criticism.keep_or_revert,
                per_suite_quality=eval_result.per_suite_quality or {},
            ))
        else:
            log.info(
                "Trial %d: AP-22 short-term memory SKIPPED (learning_excluded_by=%s)",
                trial_counter, learning_excluded_by,
            )

        # ── 6. Meta-learn ───────────────────────────────────────
        if meta.should_rebalance(trial_counter):
            meta.rebalance(
                species_effectiveness=journal.species_effectiveness(window=50),
                hv_slope=hv_slope,
                memory_count=memory_count,
                is_converged=converged,
            )
            state["species_budget"] = meta.budget.as_dict()

        # Context budget management: auto-checkpoint at intervals
        if trial_counter > 0 and trial_counter % 25 == 0 and not dry_run:
            log.info("Auto-checkpoint at trial %d", trial_counter)
            phase.set("checkpoint", trial_id=trial_counter)
            lab.checkpoint_state(
                trial_id=trial_counter,
                notes=f"Auto-checkpoint at trial {trial_counter}",
            )

        # Persist seeder convergence state on every metric-bearing trial so a
        # restart does not reset train_routing_models eligibility.
        state["seeder_state"] = seeder.export_state()
        state["td_errors"] = state["seeder_state"]["td_errors"]

        # Generate plots periodically
        if trial_counter % PLOT_INTERVAL == 0:
            async_tasks.submit_subprocess(
                f"plots-trial-{trial_counter}",
                [sys.executable, str(SCRIPT_DIR / "autopilot.py"), "plot"],
                cwd=ORCH_ROOT,
            )
            plot_paths = [
                str(PLOTS_DIR / name)
                for name in (
                    "hypervolume_trend.png",
                    "pareto_frontier_2d.png",
                    "species_effectiveness.png",
                    "per_suite_quality.png",
                    "memory_convergence.png",
                    "trial_timeline.png",
                )
            ]
            phase.set(
                "async_plots_scheduled",
                trial_id=trial_counter,
                action_type=action.get("type", ""),
            )

        # Save state
        phase.set("save_state", trial_id=trial_counter, species=species_name)
        trial_counter += 1
        state["trial_counter"] = trial_counter
        state["consecutive_meta_actions"] = 0  # a real metric-collecting trial ran
        # A trial executed and produced metrics — clear the non-executing-action
        # residue so the next prompt's "Last Non-Executing Action" feedback only
        # reflects an UNRESOLVED skip (the per-signature counts persist for the
        # blacklist threshold across the whole run).
        state["consecutive_skip_actions"] = 0
        state["last_invalid_action"] = None
        state["last_invalid_reason"] = ""
        state["last_invalid_status"] = ""
        state["consecutive_failures"] = gate.consecutive_failures
        state["quality_history"] = gate.quality_history
        state["quality_history_by_tier"] = gate.quality_history_by_tier
        state["baseline_state"] = gate.baseline.to_state_dict()
        archive.save(state)
        save_state(state)

        # Phase 6b — clear in_flight_trial marker AFTER final save_state.
        # This is the closing half of the WAL pattern: a crash between
        # the pre-dispatch marker write and this clear leaves the marker
        # in place, which triggers the recovery branch on the next
        # startup. By the time we reach here both the journal and the
        # Pareto archive are durable on disk, so it is safe to clear.
        state["in_flight_trial"] = None
        save_state(state)

        log.info(
            "Trial %d complete: q=%.3f s=%.1f → %s (HV=%.4f)",
            trial_counter - 1,
            eval_result.quality,
            eval_result.speed,
            pareto_status,
            archive.hypervolume(),
        )

        # Daily digest: once per UTC day, append a markdown snapshot to
        # progress/YYYY-MM/YYYY-MM-DD-autopilot.md. Append-only and never
        # touches existing handoffs — review remains a manual step.
        if should_generate_today(state):
            try:
                state["last_digest_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                save_state(state)
                async_tasks.submit_subprocess(
                    f"digest-{state['last_digest_date']}",
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "autopilot.py"),
                        "digest",
                        "--no-state-update",
                    ],
                    cwd=ORCH_ROOT,
                )
                phase.set(
                    "async_digest_scheduled",
                    trial_id=trial_counter - 1,
                    digest_date=state["last_digest_date"],
                )
                log.info("Daily digest scheduled asynchronously")
            except Exception as e:
                # Digest failure must NEVER block the optimization loop.
                log.warning("Daily digest generation failed: %s", e)

    # Shutdown: checkpoint + save
    phase.set("shutting_down", trial_id=trial_counter)
    async_tasks.reap(logger=log)
    async_tasks.shutdown()
    log.info("AutoPilot shutting down (trial=%d)", trial_counter)
    archive.save(state)
    save_state(state)
    if strategy_store is not None:
        strategy_store.close()
    if not dry_run:
        lab.checkpoint_state(trial_id=trial_counter, notes="Shutdown checkpoint")
    phase.clear("autopilot process exiting")


def _format_suite_trends(
    trends: dict[str, list[tuple[int, float]]],
) -> str:
    """Format suite quality trends for the controller prompt."""
    if not trends:
        return "  (no suite data yet)"
    lines = []
    for suite, points in sorted(trends.items()):
        vals = [q for _, q in points]
        direction = ""
        if len(vals) >= 3:
            recent_avg = sum(vals[-3:]) / 3
            older_avg = sum(vals[:3]) / 3
            delta = recent_avg - older_avg
            if delta < -0.05:
                direction = " ↓ DECLINING"
            elif delta > 0.05:
                direction = " ↑ improving"
        trail = " → ".join(f"{q:.2f}" for _, q in points[-5:])
        lines.append(f"  {suite}: {trail}{direction}")
    return "\n".join(lines)


def _auto_action(
    species: str,
    memory_count: int,
    converged: bool,
    seeder: Seeder,
) -> dict[str, Any]:
    """Generate an action without LLM controller (autonomous fallback)."""
    if memory_count < 500 or species == "seeder":
        return {"type": "seed_batch", "n_questions": 10}
    elif species == "numeric_swarm":
        return {"type": "numeric_trial", "surface": "memrl_retrieval"}
    elif species == "prompt_forge":
        # AP-21: GEPA ratio knob — dynamic proportion of GEPA vs LLM mutation.
        # Default 0.30 (30% GEPA). Tunable via autopilot_state["gepa_ratio"].
        # Set to 1.0 after AR-3 data confirms GEPA dominance on Pareto frontier.
        import random
        _state = load_state()
        gepa_ratio = float(_state.get("gepa_ratio", 0.30))
        if random.random() < gepa_ratio:
            return {"type": "gepa_optimize", "file": "frontdoor.md", "max_evals": 50}
        return {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"}
    elif species == "structural_lab":
        if converged:
            return {"type": "train_routing_models", "min_memories": 500}
        return {"type": "structural_experiment", "flags": {"think_harder": True}}
    elif species == "evolution_manager":
        return {"type": "distill_knowledge", "last_n": 10}
    return {"type": "seed_batch", "n_questions": 10}


def _git_tag(tag: str, message: str) -> None:
    """Create a git tag."""
    try:
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", message],
            capture_output=True, timeout=10,
            cwd=str(ORCH_ROOT),
        )
    except Exception:
        log.debug("Git tagging failed", exc_info=True)


# ── CLI Commands ─────────────────────────────────────────────────


def cmd_start(args: argparse.Namespace) -> None:
    """Start the optimization loop."""
    # Process lock
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: Another AutoPilot instance is running")
        sys.exit(1)

    # Durable earlyoom protection for THIS process. The autopilot is comm=python
    # (collides with runaway python evals, so it cannot be earlyoom --ignore'd by
    # name) and is a long-lived control-plane process; set oom_score_adj=-1000 now
    # that we hold the singleton lock (earlyoom skips exactly -1000 in both oom_score
    # and --sort-by-rss modes). Only this pid — the transient GEPA/planner subprocesses
    # must stay killable. Best-effort (sudo -n). See earlyoom-oom-protection.md.
    _set_oom_score_adj([os.getpid()])

    run_loop(
        max_trials=args.max_trials,
        dry_run=args.dry_run,
        use_controller=not args.no_controller,
        use_tui=args.tui,
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Show current status."""
    state = load_state()
    archive = ParetoArchive()
    journal = ExperimentJournal()

    print("AutoPilot Status")
    print("=" * 50)
    print(f"Trial counter: {state.get('trial_counter', 0)}")
    print(f"Paused: {state.get('paused', False)}")
    print(f"Session ID: {state.get('session_id', 'none')}")
    print()
    print(archive.summary_text(tier=DEFAULT_FRONTIER_TIER))
    print()
    print(journal.summary_text(10))


def cmd_pause(args: argparse.Namespace) -> None:
    state = load_state()
    state["paused"] = True
    save_state(state)
    print("AutoPilot paused")


def cmd_resume(args: argparse.Namespace) -> None:
    state = load_state()
    state["paused"] = False
    save_state(state)
    print("AutoPilot resumed")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate markdown report."""
    journal = ExperimentJournal()
    archive = ParetoArchive()

    print("# AutoPilot Optimization Report")
    print()
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("## Summary")
    print(journal.summary_text())
    print()
    print("## Pareto Frontier")
    print(archive.summary_text(tier=DEFAULT_FRONTIER_TIER))
    print()
    print("## Species Effectiveness")
    eff = journal.species_effectiveness()
    for sp, stats in eff.items():
        print(f"  {sp}: {stats['pareto']:.0f}/{stats['total']:.0f} ({stats['rate']:.1%})")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots."""
    archive = ParetoArchive()
    journal = ExperimentJournal()
    state = load_state()
    td_errors = [(i, e) for i, e in enumerate(state.get("td_errors", []))]
    try:
        paths = generate_all_plots(archive, journal, td_errors, raise_on_error=True)
    except Exception as e:
        # Exit non-zero so the async reaper (phase_status.reap) surfaces this as
        # "[async] plots-trial-N failed" instead of falsely logging "complete".
        # A swallowed ImportError (missing matplotlib) silently froze the
        # dashboard panels for days; never let that happen quietly again.
        log.error("Plot generation failed: %s", e)
        sys.exit(1)
    for p in paths:
        print(f"  {p}")


def cmd_checkpoint(args: argparse.Namespace) -> None:
    lab = StructuralLab()
    state = load_state()
    cp = lab.checkpoint_state(
        trial_id=state.get("trial_counter", 0),
        mark_production_best=args.production_best,
        notes="Manual checkpoint",
    )
    print(f"Checkpoint created: {cp}")


def cmd_restore(args: argparse.Namespace) -> None:
    lab = StructuralLab()
    path = Path(args.checkpoint) if args.checkpoint else None
    result = lab.restore_checkpoint(path)
    print(f"Restore result: {result}")


def cmd_digest(args: argparse.Namespace) -> None:
    """Generate an autopilot digest snapshot on demand.

    Useful between automated daily generations, or to verify the digest
    writer works without waiting for the next trial-loop iteration.
    """
    state = load_state()
    archive = ParetoArchive()
    archive.load(state)
    swarm = NumericSwarm()
    lab = StructuralLab()
    journal = ExperimentJournal()
    path = generate_digest(
        swarm=swarm, lab=lab, archive=archive, state=state, journal=journal,
    )
    if not args.no_state_update:
        state["last_digest_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        save_state(state)
    print(f"Digest written: {path}")


def cmd_peaf(args: argparse.Namespace) -> None:
    """Print PEAF correlation report (intake-571 spike cheap-kill check).

    Walks the experiment journal for entries with non-None surprise_score
    and a parent_trial, computes Pearson r² between surprise and the
    (entry.quality - parent.quality) delta, and reports the cheap-kill
    decision threshold (r² < 0.10 over min_n predicted trials → abandon).
    """
    journal = ExperimentJournal()
    report = peaf.journal_peaf_correlation(journal.all_entries(), min_n=args.min_n)
    enabled = "ON (default)" if peaf.is_peaf_enabled() else "OFF (EPYC_AUTOPILOT_PEAF explicitly disabled — set to 1 to re-enable)"
    print(f"PEAF status: {enabled}")
    print(f"Total journal entries: {journal.count()}")
    print(f"Entries with PEAF prediction + parent: {report['n_predicted']}")
    if report["mean_surprise"] is not None:
        print(f"Mean surprise (L1, normalised): {report['mean_surprise']:.4f}")
    if report["r_squared"] is None:
        print(f"r²: n/a (need at least {args.min_n} predicted trials with a parent)")
    else:
        print(f"r² (surprise vs Δquality from parent): {report['r_squared']:.4f}")
    print(f"Decision: {report['decision']}")
    if report["decision"] == "abandon":
        print("  → r² < 0.10 — PEAF signal does not correlate with config-quality gradient; consider abandoning.")
    elif report["decision"] == "continue":
        print("  → r² ≥ 0.10 — PEAF signal correlates; consider promoting surprise as Pareto co-objective.")
    else:
        print("  → keep collecting.")


def _read_baseline_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _drop_top_level_yaml_block(text: str, key: str) -> str:
    pattern = rf"(?ms)^({re.escape(key)}:\n(?:^[ \t]+.*\n?|^\n)*)"
    return re.sub(pattern, "", text).rstrip() + "\n"


def _format_baseline_tier_yaml(baseline: Baseline) -> str:
    data = {
        "baselines_by_tier": {
            int(tier): quality for tier, quality in sorted(baseline.baselines_by_tier.items())
        },
        "per_suite_quality_by_tier": {
            int(tier): suites
            for tier, suites in sorted(baseline.per_suite_quality_by_tier.items())
        },
        "per_suite_counts_by_tier": {
            int(tier): counts
            for tier, counts in sorted(baseline.per_suite_counts_by_tier.items())
        },
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()


def _write_baseline_yaml_tiers(path: Path, baseline: Baseline) -> None:
    """Update YAML seed tier fields without dropping unrelated calibration tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text()
        text = _drop_top_level_yaml_block(text, "baselines_by_tier")
        text = _drop_top_level_yaml_block(text, "per_suite_quality_by_tier")
        text = _drop_top_level_yaml_block(text, "per_suite_counts_by_tier")
        path.write_text(text.rstrip() + "\n\n" + _format_baseline_tier_yaml(baseline) + "\n")
        return

    data = {
        "quality": baseline.quality,
        "speed": baseline.speed,
        "cost": baseline.cost,
        "reliability": baseline.reliability,
        "frontdoor_speed": baseline.frontdoor_speed,
        "per_suite_quality": baseline.per_suite_quality,
        "baselines_by_tier": baseline.baselines_by_tier,
        "per_suite_quality_by_tier": baseline.per_suite_quality_by_tier,
        "per_suite_counts_by_tier": baseline.per_suite_counts_by_tier,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _migrate_flat_baseline_to_tier(
    baseline: Baseline,
    *,
    target_tier: int = 2,
) -> bool:
    """Move the historical flat quality seed into its documented eval tier."""
    if target_tier in baseline.baselines_by_tier:
        return False
    tier_quality = Baseline._validate_quality(
        baseline.quality,
        None,
        f"baseline migration T{target_tier}",
        baseline.source_path or DEFAULT_BASELINE_PATH,
    )
    if tier_quality is None:
        return False
    baseline.baselines_by_tier[target_tier] = tier_quality
    if target_tier not in baseline.per_suite_quality_by_tier:
        baseline.per_suite_quality_by_tier[target_tier] = dict(baseline.per_suite_quality)
    return True


def _apply_calibrated_baseline_result(baseline: Baseline, result: EvalResult) -> None:
    tier = int(result.tier)
    quality = Baseline._validate_quality(
        result.quality,
        None,
        f"T{tier} calibration result",
        baseline.source_path or DEFAULT_BASELINE_PATH,
    )
    if quality is None:
        raise ValueError(f"T{tier} calibration produced invalid quality: {result.quality!r}")
    if result.n_questions <= 0:
        raise ValueError("Baseline calibration requires a non-empty evaluation result")
    if result.reliability <= 0:
        raise ValueError(
            f"T{tier} calibration produced zero reliability; refusing to persist baseline"
        )
    baseline.baselines_by_tier[tier] = quality
    baseline.per_suite_quality_by_tier[tier] = dict(result.per_suite_quality)
    # Persist the per-suite question counts the baseline was measured at so the
    # per-suite regression gate's threshold knows the baseline's own sampling
    # resolution (3/n quantum); without this a calibration refresh leaves the
    # baseline-side count term inactive (2026-06-07).
    baseline.per_suite_counts_by_tier[tier] = dict(getattr(result, "per_suite_counts", {}) or {})


def calibrate_baseline(
    *,
    tier: int = DEFAULT_FRONTIER_TIER,
    n: int | None = None,
    seed: int = 42,
    baseline_path: Path | None = None,
    migrate_only: bool = False,
    write: bool = True,
) -> tuple[Baseline, EvalResult | None, bool]:
    """Migrate flat baseline state and optionally seed a calibrated tier baseline."""
    path = baseline_path or DEFAULT_BASELINE_PATH
    state = load_state()
    baseline = Baseline.load(path, state=state.get("baseline_state") or {})
    migrated_t2 = _migrate_flat_baseline_to_tier(baseline, target_tier=2)

    result: EvalResult | None = None
    if not migrate_only:
        tower = EvalTower(url=ORCHESTRATOR_URL)
        result = tower.evaluate(tier=tier, n=n, seed=seed)
        _apply_calibrated_baseline_result(baseline, result)

    state["baseline_state"] = baseline.to_state_dict()
    if write:
        _write_baseline_yaml_tiers(path, baseline)
        save_state(state)
    return baseline, result, migrated_t2


def cmd_calibrate_baseline(args: argparse.Namespace) -> None:
    baseline, result, migrated_t2 = calibrate_baseline(
        tier=args.tier,
        n=args.n,
        seed=args.seed,
        baseline_path=Path(args.baseline_path) if args.baseline_path else None,
        migrate_only=args.migrate_only,
        write=not args.dry_run,
    )
    write_label = "dry-run" if args.dry_run else "persisted"
    print(f"Baseline calibration {write_label}")
    print(f"  T2 flat migration: {'applied' if migrated_t2 else 'already present'}")
    if result is not None:
        print(
            f"  T{result.tier} calibrated: q={result.quality:.3f} "
            f"r={result.reliability:.3f} n={result.n_questions}"
        )
    print(
        "  baselines_by_tier: "
        + ", ".join(
            f"T{tier}={quality:.3f}"
            for tier, quality in sorted(baseline.baselines_by_tier.items())
        )
    )


# ── Entry Point ──────────────────────────────────────────────────


def main() -> None:
    log_path = ORCH_ROOT / "logs" / "autopilot.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # force=True overrides any basicConfig already called at import time
    # (seed_specialist_routing.py calls basicConfig at module level)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=_autopilot_logging_handlers(log_path),
        force=True,
    )

    parser = argparse.ArgumentParser(
        description="AutoPilot: Continuous recursive optimization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = subparsers.add_parser("start")
    p_start.add_argument("--dry-run", action="store_true")
    p_start.add_argument("--max-trials", type=int, default=None)
    p_start.add_argument("--no-controller", action="store_true",
                         help="Use autonomous species selection instead of Claude CLI")
    p_start.add_argument("--tui", action="store_true",
                         help="Live Rich TUI for inference monitoring (hang detection)")
    p_start.set_defaults(func=cmd_start)

    # status
    p_status = subparsers.add_parser("status")
    p_status.set_defaults(func=cmd_status)

    # pause / resume
    p_pause = subparsers.add_parser("pause")
    p_pause.set_defaults(func=cmd_pause)
    p_resume = subparsers.add_parser("resume")
    p_resume.set_defaults(func=cmd_resume)

    # report
    p_report = subparsers.add_parser("report")
    p_report.set_defaults(func=cmd_report)

    # plot
    p_plot = subparsers.add_parser("plot")
    p_plot.set_defaults(func=cmd_plot)

    # checkpoint
    p_cp = subparsers.add_parser("checkpoint")
    p_cp.add_argument("--production-best", action="store_true")
    p_cp.set_defaults(func=cmd_checkpoint)

    # restore
    p_restore = subparsers.add_parser("restore")
    p_restore.add_argument("--checkpoint", type=str, default=None)
    p_restore.set_defaults(func=cmd_restore)

    # monitor — standalone TUI (read-only, doesn't own autopilot process)
    p_monitor = subparsers.add_parser(
        "monitor",
        help="Live TUI monitor (standalone, read-only — run in a separate terminal)",
    )
    p_monitor.set_defaults(func=cmd_monitor)

    # reset-memory — clear short-term memory (AP-22)
    p_reset_mem = subparsers.add_parser(
        "reset-memory",
        help="Clear short-term memory (start fresh for next session)",
    )
    p_reset_mem.set_defaults(func=cmd_reset_memory)

    # digest — generate an autopilot snapshot on demand
    p_digest = subparsers.add_parser(
        "digest",
        help="Generate an autopilot digest snapshot (progress/YYYY-MM/YYYY-MM-DD-autopilot.md)",
    )
    p_digest.add_argument(
        "--no-state-update",
        action="store_true",
        help="Do NOT update state['last_digest_date'] — useful for ad-hoc snapshots that should not delay the next automatic generation.",
    )
    p_digest.set_defaults(func=cmd_digest)

    # peaf — Prediction-Error-As-Feature cheap-kill correlation report (intake-571 spike)
    p_peaf = subparsers.add_parser(
        "peaf",
        help="PEAF cheap-kill report: Pearson r² between surprise_score and Δquality from parent trial. Abandon at r²<0.10 over min_n predicted trials.",
    )
    p_peaf.add_argument(
        "--min-n",
        type=int,
        default=200,
        help="Minimum predicted-trials sample size before computing r² (default: 200, per intake-571 cheap-kill criterion).",
    )
    p_peaf.set_defaults(func=cmd_peaf)

    # calibrate-baseline — one-shot YAML/state seed migration + tier calibration
    p_calibrate = subparsers.add_parser(
        "calibrate-baseline",
        help="Migrate flat baseline into T2 and optionally run a one-shot tier calibration.",
    )
    p_calibrate.add_argument(
        "--tier",
        type=int,
        default=DEFAULT_FRONTIER_TIER,
        help="Eval tier to calibrate (default: canonical T1).",
    )
    p_calibrate.add_argument(
        "--n",
        type=int,
        default=None,
        help="Question count override for the calibration eval.",
    )
    p_calibrate.add_argument("--seed", type=int, default=42)
    p_calibrate.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="Baseline YAML path (defaults to orchestration/autopilot_baseline.yaml).",
    )
    p_calibrate.add_argument(
        "--migrate-only",
        action="store_true",
        help="Only migrate the flat baseline into T2; do not run EvalTower.",
    )
    p_calibrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Run calibration logic without writing YAML or state.",
    )
    p_calibrate.set_defaults(func=cmd_calibrate_baseline)

    args = parser.parse_args()
    args.func(args)


def cmd_reset_memory(args: argparse.Namespace) -> None:
    """Clear short-term memory."""
    mem = ShortTermMemory()
    mem.clear()
    print("Short-term memory cleared.")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Launch standalone TUI monitor (read-only)."""
    from autopilot_tui import AutoPilotTUI
    print("Starting standalone TUI monitor (read-only)...")
    print("Press Ctrl+C to exit.\n")
    with AutoPilotTUI():
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
