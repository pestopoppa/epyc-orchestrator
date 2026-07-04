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
from collections import deque
from dataclasses import asdict
import fcntl
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

import yaml

from src.registry.stack_priors import live_stack_slot_query_ports
from src.registry.capability_registry import (
    build_action_availability_section,
    load_capability_registry,
)
from src.autopilot_core.journal_reconstruction import (
    parse_journal_ts,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.journal_snapshot_replay import archive_payload_from_verified_snapshot
from experiment_journal import ExperimentJournal, JournalEntry, scrub_legacy_scale_text
from pareto_archive import (
    ParetoArchive,
    ParetoArchive as _ConcreteParetoArchive,
    ParetoEntry,
    pareto_archive_from_journal_rows,
)
from safety_gate import Baseline, DEFAULT_BASELINE_PATH, EvalResult, SafetyGate
from eval_tower import EvalTower
from config_applicator import apply_params  # re-exported for actions.py monkeypatch compatibility  # noqa: F401
from config_applicator import health_check
from meta_optimizer import MetaOptimizer, SpeciesBudget
from progress_plots import PLOTS_DIR, generate_all_plots
import peaf
from species import Seeder, NumericSwarm, PromptForge, StructuralLab, EvolutionManager
from species.prompt_forge import CODE_MUTATION_ALLOWLIST
from digest import generate_digest, should_generate_today
from short_term_memory import ShortTermMemory
from self_criticism import SelfCriticism, generate_self_criticism
from phase_status import AsyncTaskRunner, PhaseTracker

# 2026-05-22 Tranche-5 refactor — extracted modules. Public names re-imported below.
import controller_io
from controller_io import PLANNER_ARCHIVE_PATH, invoke_controller as _invoke_controller_impl
from planner_coordinator import plan_with_providers, uncritiqued_dispatch_block_reason
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
from src.autopilot_core.planner_evidence import (
    DEFAULT_EVIDENCE_CORE_ID,
    format_planner_evidence_section,
)
from src.autopilot_core.sequential_verdict import baseline_profile_from_trials
from src.autopilot_core.tier_specs import (
    DEFAULT_FRONTIER_TIER,
    LEGACY_OBJECTIVE_POLICY,
    MIN_FRONTIER_EVAL_TIER,
    TASK_RATE_OBJECTIVE_POLICY,
    goodput_qph_from,
    objectives_from,
    spec_for,
    task_rate_objectives_from,
    task_rate_qph_from,
    task_rate_qph_from_row,
)
from src.autopilot_core.baseline_ledger import (
    apply_baseline_ledger_authority,
    format_baseline_ledger_summary,
    reconcile_baseline_ledger,
)

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

ARCHIVE_SOURCE_STATE = "state"
ARCHIVE_SOURCE_JOURNAL_CURRENT_RUN = "journal-current-run"
ARCHIVE_SOURCE_JOURNAL_ALL = "journal-all"
ARCHIVE_SOURCE_CHOICES = (
    ARCHIVE_SOURCE_STATE,
    ARCHIVE_SOURCE_JOURNAL_CURRENT_RUN,
    ARCHIVE_SOURCE_JOURNAL_ALL,
)

if TYPE_CHECKING:
    from autopilot_tui import AutoPilotTUI

_EPHEMERAL_ACTION_KEYS = EPHEMERAL_ACTION_KEYS
_action_signature = action_signature
_canonical_action = canonical_action
_config_fingerprint = config_fingerprint

STATE_PATH = ORCH_ROOT / "orchestration" / "autopilot_state.json"
LOCK_PATH = ORCH_ROOT / "orchestration" / ".autopilot.lock"
BLACKLIST_PATH = SCRIPT_DIR / "failure_blacklist.yaml"
# Prompt-budget cap (2026-06-10): only the most-recent N blacklist entries are
# RENDERED into the planner prompt (the full list is always enforced at dispatch
# by check_blacklist()). Keeps the unbounded-growth blacklist from dominating the
# ~80KB prompt. Operator-tunable.
BLACKLIST_RENDER_CAP = int(os.environ.get("AUTOPILOT_BLACKLIST_RENDER_CAP", "18"))
PRIOR_DECISION_DIGEST_CAP = int(
    os.environ.get("AUTOPILOT_PRIOR_DECISION_DIGEST_CAP", "8")
)
ORCHESTRATOR_URL = "http://localhost:8000"
SEQ_BASELINE_PROFILE_LIMIT = int(
    os.environ.get("AUTOPILOT_SEQ_BASELINE_PROFILE_LIMIT", "120")
)
SEQ_PRIOR_OBS_LIMIT = int(os.environ.get("AUTOPILOT_SEQ_PRIOR_OBS_LIMIT", "120"))
SEQ_BASELINE_REFRESH_CADENCE = int(
    os.environ.get("AUTOPILOT_SEQ_BASELINE_REFRESH_CADENCE", "10")
)
SEQ_BASELINE_REFERENCE_STALE_AFTER_S = float(
    os.environ.get("AUTOPILOT_SEQ_BASELINE_REFERENCE_STALE_AFTER_S", str(48 * 3600))
)
SEQ_PROMOTION_FINAL_CONFIRM_E = float(
    os.environ.get("AUTOPILOT_SEQ_PROMOTION_FINAL_CONFIRM_E", "100")
)
SEQ_PROMOTION_DELTA_CI_ALPHA = float(
    os.environ.get("AUTOPILOT_SEQ_PROMOTION_DELTA_CI_ALPHA", "0.05")
)
SEQ_PROMOTION_FRESH_EVAL_TIER = int(
    os.environ.get("AUTOPILOT_SEQ_PROMOTION_FRESH_EVAL_TIER", "2")
)
SEQ_CANDIDATE_REPLAY_ENABLED = os.environ.get(
    "AUTOPILOT_SEQ_CANDIDATE_REPLAY", "1"
).strip().lower() not in {"0", "false", "no", "off"}
SEQ_CANDIDATE_REPLAY_MIN_COMBINED_E = float(
    os.environ.get("AUTOPILOT_SEQ_CANDIDATE_REPLAY_MIN_COMBINED_E", "0.9")
)
SEQ_CANDIDATE_REPLAY_MIN_QUALITY_E = float(
    os.environ.get("AUTOPILOT_SEQ_CANDIDATE_REPLAY_MIN_QUALITY_E", "1.0")
)
SEQ_CANDIDATE_REPLAY_MAX_K = int(
    os.environ.get("AUTOPILOT_SEQ_CANDIDATE_REPLAY_MAX_K", "12")
)
SAFE_FALLBACK_SEED_N = 14
FALLBACK_SEED_CANDIDATES = (14, 16, 18, 20, 24, 30)

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
# Plots also refresh on a wall-clock timer so a long mid-decade gap (e.g. a
# multi-hour tier-2 eval) doesn't leave the dashboard stale, and a blocking
# lifecycle render is time-capped so shutdown can never hang on it.
PLOT_MAX_AGE_S = 600.0
PLOT_SYNC_TIMEOUT_S = 180.0
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
_FALLBACK_NUMERIC_SURFACES = ("think_harder", "escalation", "monitor", "memrl_retrieval")
_PLANNER_HINTS_ENABLED = os.environ.get(
    "AUTOPILOT_PLANNER_HINTS", ""
).strip().lower() in {"1", "true", "yes", "on"}
_PLANNER_SUPPRESSED_NUMERIC_SURFACES: set[str] = set()
_PLANNER_DENYLISTED_FEATURE_FLAGS: set[str] = set()
_PLANNER_STRATEGY_HINT_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "structural_lab",
        "tool use sentinel lane native tools repl react_mode activation latency",
    ),
    (
        "seeder",
        "seed_batch tool_use repl native tools code math retrieval tool_helpfulness",
    ),
    (
        "prompt_forge",
        "prompt mutation repl tool helpfulness verbosity CALL native tools",
    ),
    (
        "numeric_swarm",
        "tool output compression repl tool budget latency tool_helpfulness",
    ),
)


def _configured_numeric_surfaces() -> tuple[str, ...]:
    """Return the NumericSwarm surfaces available to planner/forced trials."""
    try:
        from species.numeric_swarm import SURFACES as _NS_SURFACES
    except Exception:
        surfaces = _FALLBACK_NUMERIC_SURFACES
    else:
        surfaces = tuple(
            surface
            for surface in _NS_SURFACES
            if isinstance(surface, str) and surface.strip()
        ) or _FALLBACK_NUMERIC_SURFACES

    suppressed = _PLANNER_SUPPRESSED_NUMERIC_SURFACES
    if not suppressed:
        return surfaces
    return tuple(surface for surface in surfaces if surface not in suppressed)


def _planner_convention_bindings(
    strategy_store: StrategyStore | None,
    journal: ExperimentJournal | None,
    *,
    species: str,
) -> set[str]:
    """Return live convention bindings for planner visibility."""
    if not _PLANNER_HINTS_ENABLED or strategy_store is None:
        return set()
    if not hasattr(strategy_store, "retrieve_conventions"):
        log.warning(
            "Skipping %s planner convention bindings: StrategyStore lacks "
            "retrieve_conventions()",
            species,
        )
        return set()
    try:
        conventions = strategy_store.retrieve_conventions(
            species=species,
            journal=journal,
        )
    except TypeError:
        log.warning(
            "Skipping %s planner convention bindings: incompatible "
            "retrieve_conventions() signature",
            species,
        )
        return set()
    except Exception as exc:
        log.warning("Skipping %s planner convention bindings: %s", species, exc)
        return set()

    bindings: set[str] = set()
    for entry in conventions:
        metadata = getattr(entry, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("bind_status", "")).strip().lower() != "live":
            continue
        identifiers = metadata.get("bind_identifiers", [])
        if not isinstance(identifiers, list):
            continue
        bindings.update(
            str(identifier).strip()
            for identifier in identifiers
            if str(identifier).strip()
        )
    return bindings


def _refresh_planner_convention_bindings(
    strategy_store: StrategyStore | None,
    journal: ExperimentJournal | None,
    *,
    reason: str = "planner_turn",
) -> None:
    """Refresh default-off convention bindings for prompts and validation."""
    previous_flags = set(_PLANNER_DENYLISTED_FEATURE_FLAGS)
    previous_surfaces = set(_PLANNER_SUPPRESSED_NUMERIC_SURFACES)
    _PLANNER_DENYLISTED_FEATURE_FLAGS.clear()
    _PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()
    if not _PLANNER_HINTS_ENABLED:
        controller_io.set_suppressed_numeric_surfaces(set())
        return

    _PLANNER_DENYLISTED_FEATURE_FLAGS.update(
        _planner_convention_bindings(
            strategy_store,
            journal,
            species="structural_lab",
        )
    )
    _PLANNER_SUPPRESSED_NUMERIC_SURFACES.update(
        _planner_convention_bindings(
            strategy_store,
            journal,
            species="numeric_swarm",
        )
    )
    controller_io.set_suppressed_numeric_surfaces(_PLANNER_SUPPRESSED_NUMERIC_SURFACES)
    current_flags = set(_PLANNER_DENYLISTED_FEATURE_FLAGS)
    current_surfaces = set(_PLANNER_SUPPRESSED_NUMERIC_SURFACES)
    if (
        reason == "startup"
        or current_flags != previous_flags
        or current_surfaces != previous_surfaces
    ):
        log.info(
            "Planner convention bindings refreshed (%s): denylisted_flags=%s "
            "suppressed_numeric_surfaces=%s",
            reason,
            sorted(current_flags),
            sorted(current_surfaces),
        )


def _install_planner_convention_bindings(
    strategy_store: StrategyStore | None,
    journal: ExperimentJournal | None,
) -> None:
    """Install planner convention bindings at startup."""
    _refresh_planner_convention_bindings(
        strategy_store,
        journal,
        reason="startup",
    )


def _planner_strategy_entry_line(entry: Any) -> str:
    metadata = getattr(entry, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    species = str(getattr(entry, "species", "") or "unknown")
    entry_type = str(getattr(entry, "entry_type", "") or "raw")
    title = (
        str(getattr(entry, "title", "") or "").strip()
        or str(getattr(entry, "description", "") or "").strip()
        or "strategy"
    )
    content = (
        str(getattr(entry, "generalized_content", "") or "").strip()
        or str(getattr(entry, "insight", "") or "").strip()
    )
    if len(content) > 260:
        content = content[:257].rstrip() + "..."

    tags = [f"{species}/{entry_type}"]
    bind_status = str(metadata.get("bind_status", "") or "").strip()
    if bind_status:
        tags.append(f"bind={bind_status}")
    identifiers = metadata.get("bind_identifiers", [])
    if isinstance(identifiers, list) and identifiers:
        joined = ",".join(str(item) for item in identifiers[:4])
        tags.append(f"orchestrator_ids={joined}")
        if any(
            str(item).strip().lower()
            in {"tool", "tools", "tool_use", "repl", "react_mode", "call"}
            for item in identifiers
        ):
            tags.append("scope=orchestrator_eval_tools_not_planner_tools")
    source_handoff = str(metadata.get("source_handoff", "") or "").strip()
    if source_handoff:
        tags.append(f"handoff={source_handoff}")

    return f"- [{' | '.join(tags)}] {title}: {content}".strip()


def _build_planner_strategy_hints(
    strategy_store: StrategyStore | None,
    journal: ExperimentJournal | None,
    *,
    max_rows: int = 10,
) -> str:
    """Render bounded StrategyStore rows into the planner prompt each turn."""
    if not _PLANNER_HINTS_ENABLED:
        return "(disabled; set AUTOPILOT_PLANNER_HINTS=1 to include StrategyStore rows)"
    if strategy_store is None:
        return "(unavailable: StrategyStore did not load)"

    rows: list[Any] = []
    seen: set[str] = set()

    def add_entries(entries: Any) -> None:
        if not entries:
            return
        for entry in entries:
            entry_id = str(getattr(entry, "id", "") or "")
            if entry_id and entry_id in seen:
                continue
            if entry_id:
                seen.add(entry_id)
            rows.append(entry)
            if len(rows) >= max_rows:
                return

    try:
        if hasattr(strategy_store, "retrieve_for_journal"):
            for species, query in _PLANNER_STRATEGY_HINT_QUERIES:
                add_entries(
                    strategy_store.retrieve_for_journal(
                        query,
                        journal=journal,
                        k=3,
                        species=species,
                    )
                )
                if len(rows) >= max_rows:
                    break
    except Exception as exc:
        rows.append(
            {
                "id": "strategy-retrieval-error",
                "species": "system",
                "entry_type": "error",
                "title": "StrategyStore retrieval unavailable",
                "generalized_content": str(exc),
            }
        )

    try:
        if len(rows) < max_rows and hasattr(strategy_store, "retrieve_conventions"):
            for species in ("structural_lab", "numeric_swarm", "seeder", "prompt_forge"):
                add_entries(
                    strategy_store.retrieve_conventions(
                        species=species,
                        journal=journal,
                        limit=3,
                    )
                )
                if len(rows) >= max_rows:
                    break
    except Exception as exc:
        rows.append(
            {
                "id": "strategy-conventions-error",
                "species": "system",
                "entry_type": "error",
                "title": "StrategyStore conventions unavailable",
                "generalized_content": str(exc),
            }
        )

    if not rows:
        return "(no StrategyStore rows matched the current planner hint queries)"

    return "\n".join(
        _planner_strategy_entry_line(entry)
        if not isinstance(entry, dict)
        else (
            f"- [{entry.get('species', 'system')}/{entry.get('entry_type', 'error')}] "
            f"{entry.get('title', 'StrategyStore error')}: "
            f"{entry.get('generalized_content', '')}"
        )
        for entry in rows[:max_rows]
    )


_QUOTA_NUMERIC_SURFACES = _configured_numeric_surfaces()


def _blacklisted_action_skip(action: dict[str, Any], blocked_reason: str) -> SkipOutcome:
    """Convert a pre-dispatch blacklist hit into journalable planner feedback."""
    action_type = str(action.get("type") or "unknown")
    return SkipOutcome("invalid", f"action blacklisted: {blocked_reason}", action_type)


def _question_outcome_map(question_results: Any) -> dict[str, bool]:
    """Normalize compact question result rows into a sequential verdict map."""
    if not isinstance(question_results, list):
        return {}
    outcomes: dict[str, bool] = {}
    for item in question_results:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        if not qid:
            continue
        outcomes[qid] = bool(item.get("correct"))
    return outcomes


def _parse_journal_timestamp(timestamp: Any) -> float | None:
    if not timestamp:
        return None
    try:
        text = str(timestamp)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def _seq_baseline_reference_state(
    journal: ExperimentJournal,
    *,
    tier: int,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Return freshness/cadence state for the seq baseline reference profile."""
    now = time.time() if now_ts is None else float(now_ts)
    trusted_profile_trials = 0
    trials_since_reference = 0
    latest_profile_trial_id = None
    latest_reference_trial_id = None
    latest_reference_ts = None

    for entry in reversed(journal.entries_with_supersessions()):
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        if getattr(entry, "outcome_status", "ok") in {"invalid", "skipped"}:
            continue
        try:
            if int(getattr(entry, "tier", -1)) != int(tier):
                continue
        except (TypeError, ValueError):
            continue
        eval_details = getattr(entry, "eval_details", {}) or {}
        if not isinstance(eval_details, dict):
            continue
        if not _question_outcome_map(eval_details.get("question_results")):
            continue

        trusted_profile_trials += 1
        entry_ts = _parse_journal_timestamp(getattr(entry, "timestamp", ""))
        if latest_profile_trial_id is None:
            latest_profile_trial_id = getattr(entry, "trial_id", None)

        if latest_reference_trial_id is None and eval_details.get(
            "seq_baseline_reference_draw"
        ):
            latest_reference_trial_id = getattr(entry, "trial_id", None)
            latest_reference_ts = entry_ts
        elif latest_reference_trial_id is None:
            trials_since_reference += 1

    age_s = (
        max(0.0, now - latest_reference_ts)
        if latest_reference_ts is not None
        else None
    )
    stale_reference = (
        age_s is not None and age_s > SEQ_BASELINE_REFERENCE_STALE_AFTER_S
    )
    due = (
        SEQ_BASELINE_REFRESH_CADENCE > 0
        and (
            latest_reference_trial_id is None
            or trials_since_reference >= SEQ_BASELINE_REFRESH_CADENCE
            or stale_reference
        )
    )
    reason = ""
    if latest_reference_trial_id is None:
        reason = "no marked seq baseline-reference draw"
    elif stale_reference:
        reason = f"baseline reference age {age_s:.0f}s exceeds stale threshold"
    elif trials_since_reference >= SEQ_BASELINE_REFRESH_CADENCE:
        reason = (
            f"{trials_since_reference} trusted profile trials since baseline "
            "reference draw"
        )

    return {
        "tier": int(tier),
        "trusted_profile_trials": trusted_profile_trials,
        "latest_profile_trial_id": latest_profile_trial_id,
        "latest_reference_trial_id": latest_reference_trial_id,
        "latest_reference_age_s": age_s,
        "latest_reference_ts": latest_reference_ts,
        "trials_since_reference": trials_since_reference,
        "stale_reference": bool(stale_reference),
        "due": bool(due),
        "reason": reason,
    }


def _seq_baseline_draw_action() -> dict[str, Any]:
    raw = os.environ.get("AUTOPILOT_SEQ_BASELINE_DRAW_ACTION", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("type"):
                return parsed
        except json.JSONDecodeError:
            log.warning("Invalid AUTOPILOT_SEQ_BASELINE_DRAW_ACTION JSON; using default")
    # n=14 avoids known blacklisted seed_batch sizes while still collecting a
    # small reference draw through the standard metric-collecting path.
    return {"type": "seed_batch", "n_questions": 14}


def _seed_action_candidates(
    preferred: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add(candidate: dict[str, Any]) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    if isinstance(preferred, dict) and preferred.get("type") == "seed_batch":
        add(dict(preferred))
        if "suites" in preferred:
            no_suite = dict(preferred)
            no_suite.pop("suites", None)
            add(no_suite)

    for n_questions in FALLBACK_SEED_CANDIDATES:
        add({"type": "seed_batch", "n_questions": int(n_questions)})
        add(
            {
                "type": "seed_batch",
                "n_questions": int(n_questions),
                "suites": ["coder", "math"],
            }
        )
    return candidates


def _first_unblacklisted_seed_action(
    blacklist: list[dict[str, Any]],
    *,
    preferred: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    last_blocked = ""
    for candidate in _seed_action_candidates(preferred):
        blocked = check_blacklist(candidate, blacklist)
        if not blocked:
            return candidate, ""
        last_blocked = blocked
    return None, last_blocked or "all measured seed fallbacks are blacklisted"


def _seed_fallback_exhaustion_reason(blacklist: list[dict[str, Any]]) -> str | None:
    """Return a planner-facing reason when every measured seed fallback is blocked."""
    candidate, reason = _first_unblacklisted_seed_action(blacklist)
    if candidate is not None:
        return None
    return reason or "all measured seed fallbacks are blacklisted"


def _critic_fallback_seed_skip(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
) -> SkipOutcome | None:
    """Skip a critic seed fallback when the measured seed ladder is exhausted."""
    if action.get("type") != "seed_batch":
        return None
    reason = _seed_fallback_exhaustion_reason(blacklist)
    if not reason:
        return None
    return SkipOutcome(
        "skipped",
        f"critic fallback seed_batch unavailable: {reason}",
        "planner_coordinator",
    )


def _first_unblacklisted_numeric_trial_action(
    blacklist: list[dict[str, Any]],
    *,
    trial_counter: int = 0,
) -> tuple[dict[str, Any] | None, str]:
    last_blocked = ""
    surfaces = _configured_numeric_surfaces()
    if not surfaces:
        return None, (
            "all quota numeric_trial surfaces are suppressed by planner "
            "conventions"
        )
    for offset in range(len(surfaces)):
        surface = surfaces[(trial_counter + offset) % len(surfaces)]
        candidate = {"type": "numeric_trial", "surface": surface, "params": {}}
        blocked = check_blacklist(candidate, blacklist)
        if not blocked:
            return candidate, ""
        last_blocked = blocked
    return None, last_blocked or "all quota numeric_trial surfaces are blacklisted"


def _replace_blacklisted_seed_fallback(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None = None,
    *,
    reason_label: str = "blacklisted fallback",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if action.get("type") != "seed_batch":
        return action, rationale
    blocked = check_blacklist(action, blacklist)
    if not blocked:
        return action, rationale
    replacement, _ = _first_unblacklisted_seed_action(
        blacklist,
        preferred=action,
    )
    if replacement is None:
        log.warning(
            "%s seed action %s is blacklisted (%s), and no measured seed "
            "fallback remains unblocked.",
            reason_label,
            json.dumps(action, default=str),
            blocked,
        )
        return action, rationale
    next_rationale = {
        **(rationale or {}),
        "fallback_seed_reselected": True,
        "fallback_seed_reselected_reason": blocked,
        "fallback_seed_reselected_from": dict(action),
        "fallback_seed_reselected_context": reason_label,
    }
    log.warning(
        "%s seed action %s is blacklisted (%s); using measured fallback %s.",
        reason_label,
        json.dumps(action, default=str),
        blocked,
        json.dumps(replacement, default=str),
    )
    return replacement, next_rationale


def _replace_blacklisted_autonomous_action(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Keep no-controller mode from burning trials on known-blocked/meta actions."""
    blocked = check_blacklist(action, blacklist)
    is_meta_noop = action.get("type") in META_NOOP_ACTIONS
    if not blocked and not is_meta_noop:
        return action, rationale
    reason = blocked or "autonomous meta action does not collect metrics"
    replacement, fallback_reason = _first_unblacklisted_seed_action(blacklist)
    if replacement is None:
        log.warning(
            "Autonomous action %s is not dispatchable (%s), and no measured seed "
            "fallback remains unblocked: %s",
            json.dumps(action, default=str),
            reason,
            fallback_reason,
        )
        return action, rationale
    next_rationale = {
        **(rationale or {}),
        "autonomous_blacklisted_replaced": True,
        "autonomous_blacklisted_reason": reason,
        "autonomous_blacklisted_from": dict(action),
    }
    log.warning(
        "Autonomous action %s is not dispatchable (%s); using measured seed fallback %s.",
        json.dumps(action, default=str),
        reason,
        json.dumps(replacement, default=str),
    )
    return replacement, next_rationale


def _seq_baseline_reference_block_key(reference: dict[str, Any]) -> str:
    ref_id = reference.get("latest_reference_trial_id")
    ref_token = "none" if ref_id is None else str(ref_id)
    return f"tier={reference.get('tier')}:reference={ref_token}"


def _maybe_force_seq_baseline_draw(
    action: dict[str, Any],
    *,
    state: dict[str, Any],
    journal: ExperimentJournal,
    tier: int,
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Force the 01c baseline-reference cadence when seq shadowing is enabled."""
    if not enabled:
        return action, rationale, None
    reference = _seq_baseline_reference_state(journal, tier=tier)
    if not reference["due"]:
        return action, rationale, None

    forced = _seq_baseline_draw_action()
    reference_key = _seq_baseline_reference_block_key(reference)
    blocked_state = state.get("seq_baseline_draw_blocked")
    if (
        isinstance(blocked_state, dict)
        and blocked_state.get("reference_key") == reference_key
        and blocked_state.get("action") == forced
    ):
        return action, rationale, None
    blocked_reason = check_blacklist(forced, blacklist)
    if blocked_reason:
        fallback, fallback_reason = _first_unblacklisted_seed_action(
            blacklist,
            preferred=forced,
        )
        if fallback is not None:
            forced = fallback
            blocked_reason = ""
        else:
            state["seq_baseline_draw_blocked"] = {
                "trial_id": trial_counter,
                "action": forced,
                "reason": fallback_reason or blocked_reason,
                "reference": reference,
                "reference_key": reference_key,
            }
            log.warning(
                "Seq baseline-reference draw due but forced action is blacklisted: %s",
                fallback_reason or blocked_reason,
            )
            return action, rationale, None
    next_rationale = dict(rationale or {})
    next_rationale["seq_baseline_reference_draw"] = True
    next_rationale["seq_baseline_reference_reason"] = reference["reason"]
    state["seq_baseline_draw_blocked"] = None
    state["seq_baseline_draw_forced"] = {
        "trial_id": trial_counter,
        "action": forced,
        "reference": reference,
    }
    log.info(
        "Forcing seq baseline-reference draw at trial %d: %s",
        trial_counter,
        reference["reason"],
    )
    return forced, next_rationale, reference


def _seq_combined_e(seq: dict[str, Any] | None) -> float | None:
    if not isinstance(seq, dict):
        return None
    try:
        e_quality = float(seq["E_quality"])
        e_rate = float(seq["E_rate_noninf"])
    except (KeyError, TypeError, ValueError):
        return None
    return min(e_quality, e_rate)


def _seq_promotion_delta_ci(seq: dict[str, Any] | None) -> dict[str, Any]:
    """Return the Phase-2.4 one-sided non-regression CI for a promotion eval."""
    if not isinstance(seq, dict):
        return {
            "status": "missing",
            "excludes_regression": False,
            "reason": "missing seq block",
        }
    try:
        mean_delta = float(seq["z"])
        n_eff = int(seq["r_eff"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "missing",
            "excludes_regression": False,
            "reason": "missing z/r_eff promotion-delta evidence",
        }
    if n_eff <= 0:
        return {
            "status": "insufficient",
            "n_eff": n_eff,
            "mean_delta": round(mean_delta, 6),
            "excludes_regression": False,
            "reason": "no effective paired questions in promotion eval",
        }

    alpha = max(1e-12, min(0.5, SEQ_PROMOTION_DELTA_CI_ALPHA))
    half_width = math.sqrt(2.0 * math.log(1.0 / alpha) / n_eff)
    lower_bound = mean_delta - half_width
    return {
        "status": "ok",
        "confidence": round(1.0 - alpha, 6),
        "alpha": alpha,
        "n_eff": n_eff,
        "mean_delta": round(mean_delta, 6),
        "half_width": round(half_width, 6),
        "lower_bound": round(lower_bound, 6),
        "excludes_regression": lower_bound >= 0.0,
    }


def _annotate_seq_promotion_finalization(
    seq: dict[str, Any] | None,
    *,
    baseline_reference: dict[str, Any] | None,
    is_fresh_eval: bool,
    fresh_eval_context: dict[str, Any] | None = None,
) -> bool | None:
    """Add baseline-promotion finalization metadata to a seq journal block."""
    if not isinstance(seq, dict):
        return None
    reference = baseline_reference or {}
    combined_e = _seq_combined_e(seq)
    delta_ci = _seq_promotion_delta_ci(seq) if is_fresh_eval else None
    stale_reference = bool(reference.get("stale_reference"))
    seq["baseline_reference"] = {
        "tier": reference.get("tier"),
        "latest_reference_trial_id": reference.get("latest_reference_trial_id"),
        "latest_reference_age_s": reference.get("latest_reference_age_s"),
        "trials_since_reference": reference.get("trials_since_reference"),
        "stale_reference": stale_reference,
    }
    seq["baseline_reference_state"] = (
        "stale-reference" if stale_reference else "fresh"
    )
    seq["baseline_promotion_required_E"] = SEQ_PROMOTION_FINAL_CONFIRM_E
    seq["baseline_promotion_fresh_eval"] = bool(is_fresh_eval)
    if combined_e is not None:
        seq["baseline_promotion_combined_E"] = round(combined_e, 6)
    if delta_ci is not None:
        seq["baseline_promotion_delta_ci"] = delta_ci
    if fresh_eval_context:
        seq["baseline_promotion_fresh_eval_for"] = {
            "candidate": fresh_eval_context.get("candidate"),
            "source_trial_id": fresh_eval_context.get("source_trial_id"),
        }
    finalized = (
        bool(seq.get("confirmed"))
        and bool(is_fresh_eval)
        and not stale_reference
        and combined_e is not None
        and combined_e >= SEQ_PROMOTION_FINAL_CONFIRM_E
        and delta_ci is not None
        and bool(delta_ci.get("excludes_regression"))
    )
    seq["baseline_promotion_finalized"] = finalized
    return finalized


def _maybe_force_seq_promotion_fresh_eval(
    action: dict[str, Any],
    *,
    state: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Force one large fresh eval for a pending seq-confirmed promotion."""
    pending = state.get("seq_pending_promotion_fresh_eval")
    if not enabled or not isinstance(pending, dict):
        return action, rationale, None
    attempts = int(pending.get("attempts", 0) or 0)
    if attempts >= 1:
        return action, rationale, None
    tier = max(MIN_FRONTIER_EVAL_TIER, int(pending.get("tier") or SEQ_PROMOTION_FRESH_EVAL_TIER))
    candidate_action = pending.get("action")
    replay_blocker = _seq_promotion_replay_blocker(candidate_action)
    if replay_blocker:
        state.pop("seq_pending_promotion_fresh_eval", None)
        state.pop("_seq_promotion_candidate_replay", None)
        state["seq_last_promotion_blocked"] = {
            "trial_id": trial_counter,
            "candidate": pending.get("candidate"),
            "source_trial_id": pending.get("source_trial_id"),
            "reason": replay_blocker,
        }
        log.warning(
            "Seq promotion fresh eval blocked for candidate %s: %s",
            pending.get("candidate"),
            replay_blocker,
        )
        return action, rationale, None
    forced = {"type": "deep_eval", "tier": tier}
    blocked_reason = check_blacklist(forced, blacklist)
    if blocked_reason:
        pending["attempts"] = attempts + 1
        pending["blocked_reason"] = blocked_reason
        pending["blocked_at_trial"] = trial_counter
        pending["blocked_action"] = forced
        state["seq_pending_promotion_fresh_eval"] = pending
        log.warning(
            "Seq promotion fresh eval due but forced action is blacklisted: %s",
            blocked_reason,
        )
        return action, rationale, None

    pending["attempts"] = attempts + 1
    pending["forced_trial_id"] = trial_counter
    state["seq_pending_promotion_fresh_eval"] = pending
    state["_seq_promotion_candidate_replay"] = {
        "trial_id": trial_counter,
        "candidate": pending.get("candidate"),
        "source_trial_id": pending.get("source_trial_id"),
        "action": candidate_action,
    }
    next_rationale = dict(rationale or {})
    next_rationale["seq_promotion_fresh_eval"] = True
    next_rationale["seq_promotion_candidate"] = pending.get("candidate")
    log.info(
        "Forcing seq promotion fresh eval at trial %d for candidate %s",
        trial_counter,
        pending.get("candidate"),
    )
    return forced, next_rationale, dict(pending)


def _seq_promotion_replay_blocker(action: Any) -> str:
    """Return why a seq-promotion candidate cannot be replayed for fresh eval."""
    if not isinstance(action, dict):
        return "candidate action is missing or not an object"
    action_type = str(action.get("type") or "")
    if action_type == "numeric_trial":
        params = action.get("params")
        if not isinstance(params, dict) or not params:
            return "candidate numeric_trial lacks replayable applied params"
    elif action_type == "structural_experiment":
        flags = action.get("flags")
        if not isinstance(flags, dict) or not flags:
            return "candidate structural_experiment lacks replayable flags"
    else:
        return f"candidate action type is not replayable: {action_type or 'unknown'}"

    scope_err = controller_io.validate_single_variable(action)
    if scope_err:
        return f"candidate action violates AP-9: {scope_err}"
    return ""


def _seq_candidate_replay_payload(
    journal: ExperimentJournal,
    *,
    tier: int,
    min_combined_e: float = SEQ_CANDIDATE_REPLAY_MIN_COMBINED_E,
    min_quality_e: float = SEQ_CANDIDATE_REPLAY_MIN_QUALITY_E,
    max_k: int = SEQ_CANDIDATE_REPLAY_MAX_K,
    core_id: str = DEFAULT_EVIDENCE_CORE_ID,
) -> dict[str, Any] | None:
    """Pick an accumulating replayable candidate that needs more W8 power."""
    best: dict[str, Any] | None = None
    try:
        entries = journal.entries_with_supersessions()
    except Exception:  # noqa: BLE001 - diagnostics should not halt AutoPilot
        return None
    latest_by_candidate: dict[str, Any] = {}
    for entry in entries:
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        if getattr(entry, "outcome_status", "ok") in {"invalid", "skipped"}:
            continue
        try:
            if int(getattr(entry, "tier", -1)) != int(tier):
                continue
        except (TypeError, ValueError):
            continue
        seq = getattr(entry, "seq", {}) or {}
        if not isinstance(seq, dict):
            continue
        if str(seq.get("core_id") or core_id) != core_id:
            continue
        candidate = str(seq.get("candidate") or "")
        if not candidate:
            continue
        previous = latest_by_candidate.get(candidate)
        previous_trial = int(getattr(previous, "trial_id", -1) or -1) if previous else -1
        trial_id = int(getattr(entry, "trial_id", -1) or -1)
        if trial_id >= previous_trial:
            latest_by_candidate[candidate] = entry

    for entry in latest_by_candidate.values():
        keep_revert = str(getattr(entry, "keep_revert_decision", "") or "").strip()
        if keep_revert in {"revert", "excluded"}:
            # AP-24 verdict-failed rows keep outcome_status="ok"; replay must
            # honor the explicit keep/revert decision or W8 can force
            # already-reverted configs back into the measurement loop.
            continue
        seq = getattr(entry, "seq", {}) or {}
        if seq.get("confirmed") is True or str(seq.get("state") or "") != "accumulating":
            continue
        candidate = str(seq.get("candidate") or "")
        try:
            k = int(seq.get("k") or 0)
        except (TypeError, ValueError):
            k = 0
        if max_k > 0 and k >= max_k:
            continue
        action = getattr(entry, "config_snapshot", {}) or {}
        if _seq_promotion_replay_blocker(action):
            continue
        combined = _seq_combined_e(seq)
        if combined is None:
            combined = seq.get("baseline_promotion_combined_E")
        try:
            combined_f = float(combined)
        except (TypeError, ValueError):
            continue
        if combined_f < min_combined_e:
            continue
        try:
            e_quality = float(seq.get("E_quality") or 0.0)
        except (TypeError, ValueError):
            e_quality = 0.0
        if e_quality < min_quality_e:
            continue
        try:
            e_rate = float(seq.get("E_rate_noninf") or 0.0)
        except (TypeError, ValueError):
            e_rate = 0.0
        payload = {
            "candidate": candidate,
            "source_trial_id": int(getattr(entry, "trial_id", 0) or 0),
            "action": dict(action),
            "k": k,
            "combined_E": round(combined_f, 6),
            "E_quality": round(e_quality, 6),
            "E_rate_noninf": round(e_rate, 6),
        }
        key = (-k, combined_f, e_quality, e_rate, payload["source_trial_id"])
        best_key = (
            -int(best.get("k", 0)),
            float(best.get("combined_E", 0.0)),
            float(best.get("E_quality", 0.0)),
            float(best.get("E_rate_noninf", 0.0)),
            int(best.get("source_trial_id", 0)),
        ) if best else None
        if best is None or key > best_key:
            best = payload
    return best


def _maybe_force_seq_candidate_replay(
    action: dict[str, Any],
    *,
    state: dict[str, Any],
    journal: ExperimentJournal,
    tier: int,
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Replay an accumulating W8 candidate before it is abandoned underpowered."""
    if not enabled or not SEQ_CANDIDATE_REPLAY_ENABLED:
        return action, rationale, None
    if isinstance(state.get("seq_pending_promotion_fresh_eval"), dict):
        return action, rationale, None
    payload = _seq_candidate_replay_payload(journal, tier=tier)
    if payload is None:
        return action, rationale, None
    forced = dict(payload["action"])
    blocked_reason = check_blacklist(forced, blacklist)
    if blocked_reason:
        state["seq_candidate_replay_blocked"] = {
            "trial_id": trial_counter,
            "candidate": payload["candidate"],
            "source_trial_id": payload["source_trial_id"],
            "reason": blocked_reason,
            "action": forced,
            "combined_E": payload["combined_E"],
        }
        return action, rationale, None
    state["seq_candidate_replay_forced"] = {
        "trial_id": trial_counter,
        "candidate": payload["candidate"],
        "source_trial_id": payload["source_trial_id"],
        "action": forced,
        "k": payload["k"],
        "combined_E": payload["combined_E"],
        "E_quality": payload["E_quality"],
        "E_rate_noninf": payload["E_rate_noninf"],
    }
    state.pop("seq_candidate_replay_blocked", None)
    next_rationale = dict(rationale or {})
    next_rationale["seq_candidate_replay"] = True
    next_rationale["seq_candidate"] = payload["candidate"]
    next_rationale["seq_candidate_source_trial_id"] = payload["source_trial_id"]
    log.info(
        "Forcing seq candidate replay at trial %d for candidate %s "
        "(source trial %s, k=%s, combined_E=%s)",
        trial_counter,
        payload["candidate"],
        payload["source_trial_id"],
        payload["k"],
        payload["combined_E"],
    )
    return forced, next_rationale, dict(payload)


def _update_seq_promotion_fresh_eval_state(
    state: dict[str, Any],
    *,
    seq: dict[str, Any] | None,
    action: dict[str, Any],
    eval_result: EvalResult,
    trial_counter: int,
    is_fresh_eval: bool,
    finalized: bool | None,
    baseline_update: Any | None = None,
) -> None:
    """Maintain the bounded pending fresh-eval state for seq baseline promotion."""
    if not isinstance(seq, dict):
        return
    baseline_update_reason = getattr(baseline_update, "reason", None)
    delta_ci = seq.get("baseline_promotion_delta_ci")
    if finalized:
        state.pop("_seq_promotion_candidate_replay", None)
        if baseline_update is not None and not bool(getattr(baseline_update, "updated", False)):
            state.pop("seq_pending_promotion_fresh_eval", None)
            blocked = {
                "trial_id": trial_counter,
                "candidate": seq.get("candidate"),
                "reason": "baseline-update-refused",
                "baseline_update_reason": baseline_update_reason,
                "combined_E": seq.get("baseline_promotion_combined_E"),
            }
            if delta_ci is not None:
                blocked["delta_ci"] = delta_ci
            state["seq_last_promotion_blocked"] = blocked
            return
        state.pop("seq_pending_promotion_fresh_eval", None)
        finalized_state = {
            "trial_id": trial_counter,
            "candidate": seq.get("candidate"),
            "combined_E": seq.get("baseline_promotion_combined_E"),
            "baseline_update_reason": baseline_update_reason,
        }
        if delta_ci is not None:
            finalized_state["delta_ci"] = delta_ci
        state["seq_last_promotion_finalized"] = finalized_state
        return
    if seq.get("baseline_reference_state") == "stale-reference":
        state.pop("_seq_promotion_candidate_replay", None)
        state.pop("seq_pending_promotion_fresh_eval", None)
        state["seq_last_promotion_blocked"] = {
            "trial_id": trial_counter,
            "candidate": seq.get("candidate"),
            "reason": "stale-reference",
        }
        return
    if is_fresh_eval:
        state.pop("_seq_promotion_candidate_replay", None)
        state.pop("seq_pending_promotion_fresh_eval", None)
        blocked = {
            "trial_id": trial_counter,
            "candidate": seq.get("candidate"),
            "reason": (
                "fresh-eval did not confirm"
                if not seq.get("confirmed")
                else "fresh-eval did not reach finalization threshold"
            ),
            "combined_E": seq.get("baseline_promotion_combined_E"),
        }
        if delta_ci is not None:
            blocked["delta_ci"] = delta_ci
        state["seq_last_promotion_blocked"] = blocked
        return
    if not seq.get("confirmed"):
        return
    state["seq_pending_promotion_fresh_eval"] = {
        "candidate": seq.get("candidate"),
        "core_id": seq.get("core_id") or DEFAULT_EVIDENCE_CORE_ID,
        "source_trial_id": trial_counter,
        "tier": int(getattr(eval_result, "tier", DEFAULT_FRONTIER_TIER) or DEFAULT_FRONTIER_TIER),
        "action": dict(action),
        "combined_E": seq.get("baseline_promotion_combined_E"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempts": 0,
    }


def _seq_inputs_for_trial(
    *,
    journal: ExperimentJournal,
    action: dict[str, Any],
    tier: int,
    candidate_override: str | None = None,
) -> dict[str, Any]:
    """Build default-off W4 shadow inputs from trusted prior journal evidence."""
    candidate = candidate_override or _config_fingerprint(action)
    prior_quality_obs: list[tuple[int | None, float]] = []
    prior_rate_obs: list[tuple[int | None, float]] = []
    baseline_trials: list[dict[str, bool]] = []
    baseline_task_rates: list[float] = []

    for entry in reversed(journal.entries_with_supersessions()):
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        if getattr(entry, "outcome_status", "ok") in {"invalid", "skipped"}:
            continue
        try:
            if int(getattr(entry, "tier", -1)) != int(tier):
                continue
        except (TypeError, ValueError):
            continue
        eval_details = getattr(entry, "eval_details", {}) or {}
        if not isinstance(eval_details, dict):
            continue

        outcome_map = _question_outcome_map(eval_details.get("question_results"))
        if outcome_map and len(baseline_trials) < SEQ_BASELINE_PROFILE_LIMIT:
            baseline_trials.append(outcome_map)
        if len(baseline_task_rates) < SEQ_BASELINE_PROFILE_LIMIT:
            row = {
                "quality": getattr(entry, "quality", 0.0),
                "n_questions": len(outcome_map),
                "eval_details": eval_details,
            }
            task_rate = task_rate_qph_from_row(row)
            if task_rate > 0.0:
                baseline_task_rates.append(task_rate)

        seq = getattr(entry, "seq", {}) or {}
        if not isinstance(seq, dict):
            continue
        if str(seq.get("candidate") or "") != candidate:
            continue
        if str(seq.get("core_id") or DEFAULT_EVIDENCE_CORE_ID) != DEFAULT_EVIDENCE_CORE_ID:
            continue
        if len(prior_quality_obs) < SEQ_PRIOR_OBS_LIMIT and "z" in seq:
            try:
                prior_quality_obs.append((entry.trial_id, float(seq["z"])))
            except (TypeError, ValueError):
                pass
        if len(prior_rate_obs) < SEQ_PRIOR_OBS_LIMIT and "z_rate" in seq:
            try:
                prior_rate_obs.append((entry.trial_id, float(seq["z_rate"])))
            except (TypeError, ValueError):
                pass

    baseline_profile = baseline_profile_from_trials(reversed(baseline_trials))
    baseline_task_rate = (
        sum(baseline_task_rates) / len(baseline_task_rates)
        if baseline_task_rates
        else None
    )
    return {
        "candidate": candidate,
        "core_id": DEFAULT_EVIDENCE_CORE_ID,
        "baseline_profile": baseline_profile,
        "baseline_task_rate": baseline_task_rate,
        "prior_quality_obs": list(reversed(prior_quality_obs)),
        "prior_rate_obs": list(reversed(prior_rate_obs)),
        "baseline_reference": _seq_baseline_reference_state(journal, tier=tier),
    }


def _format_blacklist_pattern(pattern: Any) -> str:
    if not isinstance(pattern, dict) or not pattern:
        return "{}"
    return json.dumps(pattern, sort_keys=True, separators=(",", ":"), default=str)


def _format_blacklist_for_prompt(blacklist: list[dict[str, Any]]) -> str:
    """Render blacklist prompt context without hiding older enforced patterns."""
    if not blacklist:
        return "  (none)"

    cap = max(0, BLACKLIST_RENDER_CAP)
    shown = blacklist[-cap:] if cap else []
    older = blacklist[:-cap] if cap else list(blacklist)
    lines: list[str] = []

    if shown:
        lines.append(
            f"  Recent entries ({len(shown)} newest; all {len(blacklist)} enforced):"
        )
        for entry in shown:
            reason = str(entry.get("reason", ""))
            if len(reason) > 80:
                reason = reason[:79] + "..."
            lines.append(
                f"  - {_format_blacklist_pattern(entry.get('pattern', {}))} -- {reason}"
            )

    if older:
        lines.append(
            f"  Older enforced patterns ({len(older)}; reasons omitted, still blocked):"
        )
        for entry in older:
            suffix = ""
            if entry.get("source_trial") is not None:
                suffix = f" (source_trial={entry['source_trial']})"
            lines.append(
                f"    - {_format_blacklist_pattern(entry.get('pattern', {}))}{suffix}"
            )

    return "\n".join(lines)


def _enforce_experiment_quota(
    action: dict[str, Any],
    state: dict[str, Any],
    memory_count: int,
    rationale: dict[str, Any] | None = None,
    trial_counter: int = 0,
    blacklist: list[dict[str, Any]] | None = None,
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
        forced, blocked_reason = _first_unblacklisted_numeric_trial_action(
            blacklist or [],
            trial_counter=trial_counter,
        )
        if forced is None:
            log.warning(
                "Experiment quota due after %d consecutive passive actions, but "
                "all numeric_trial quota surfaces are blacklisted: %s",
                streak,
                blocked_reason,
            )
            state["experiment_quota_blocked"] = {
                "trial_id": trial_counter,
                "reason": blocked_reason,
                "action": action,
            }
            state["consecutive_passive_actions"] = streak + 1
            return action, rationale
        surface = forced["surface"]
        log.warning(
            "Experiment quota: %d consecutive passive actions with memory_count=%d "
            ">= %d threshold; forcing frontier-moving numeric_trial(surface=%s) "
            "instead of another '%s'.",
            streak, memory_count, QUOTA_MEMORY_THRESHOLD, surface, atype,
        )
        state["consecutive_passive_actions"] = 0
        state["experiment_quota_blocked"] = None
        return (
            forced,
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


def _maybe_force_frontier_rerun_action(
    action: dict[str, Any],
    state: dict[str, Any],
    *,
    journal: ExperimentJournal | None = None,
    archive: ParetoArchive | None = None,
    blacklist: list[dict[str, Any]] | None = None,
    rationale: dict[str, Any] | None = None,
    trial_counter: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Turn an active frontier-rerun marker into a concrete numeric trial."""
    marker = state.get("frontier_rerun_required")
    if not (isinstance(marker, dict) and marker.get("required")):
        return action, rationale
    min_trials = _frontier_rerun_min_trials(marker)
    completed_trials = _frontier_rerun_completed_numeric_trials(marker, journal)
    pending = state.get("frontier_rerun_pending_clear")
    if isinstance(pending, dict) and pending.get("trial_id") is not None:
        if journal is not None and completed_trials < min_trials:
            action, forced_rationale = _force_frontier_rerun_numeric_action(
                action,
                state,
                marker=marker,
                blacklist=blacklist or [],
                rationale=rationale,
                trial_counter=trial_counter,
                completed_trials=completed_trials,
                min_trials=min_trials,
            )
            return action, forced_rationale
        if journal is not None and completed_trials >= min_trials:
            _clear_frontier_rerun_marker(
                state,
                marker=marker,
                pending=pending,
                completed_trials=completed_trials,
                min_trials=min_trials,
                archive=archive,
            )
            return action, {
                **(rationale or {}),
                "frontier_rerun_cleared": True,
                "frontier_rerun_completed_numeric_trials": completed_trials,
                "frontier_rerun_min_numeric_trials": min_trials,
                "frontier_rerun_archive_snapshot": state["frontier_rerun_required"].get(
                    "archive_snapshot"
                ),
            }
        return action, {
            **(rationale or {}),
            "frontier_rerun_pending_clear": True,
            "frontier_rerun_pending_trial_id": pending.get("trial_id"),
            "frontier_rerun_completed_numeric_trials": completed_trials,
            "frontier_rerun_min_numeric_trials": min_trials,
        }
    if action.get("type") == "numeric_trial":
        state["frontier_rerun_forced"] = None
        state["frontier_rerun_pending_clear"] = {
            "trial_id": trial_counter,
            "action": action,
            "reason": str(marker.get("reason") or "frontier rerun required"),
        }
        return action, {
            **(rationale or {}),
            "frontier_rerun_satisfied_by_selected_action": True,
        }

    return _force_frontier_rerun_numeric_action(
        action,
        state,
        marker=marker,
        blacklist=blacklist or [],
        rationale=rationale,
        trial_counter=trial_counter,
        completed_trials=completed_trials,
        min_trials=min_trials,
    )


def _clear_frontier_rerun_marker(
    state: dict[str, Any],
    *,
    marker: dict[str, Any],
    pending: dict[str, Any],
    completed_trials: int,
    min_trials: int,
    archive: ParetoArchive | None = None,
) -> None:
    """Persist that the era-scoped frontier rerun has satisfied its gate."""
    reason = str(marker.get("reason") or "frontier rerun required")
    archive_snapshot = _frontier_rerun_archive_snapshot(archive)
    state["frontier_rerun_required"] = {
        **marker,
        "required": False,
        "cleared_at": datetime.now(timezone.utc).isoformat(),
        "cleared_after_trial_id": pending.get("trial_id"),
        "completed_numeric_trials": completed_trials,
        "min_numeric_trials": min_trials,
        "archive_snapshot": archive_snapshot,
        "reason": (
            f"frontier rerun satisfied: {completed_trials}/{min_trials} "
            f"current-era numeric trials complete; {reason}"
        ),
    }
    state["frontier_rerun_forced"] = None
    state.pop("frontier_rerun_pending_clear", None)
    state.pop("frontier_rerun_blocked", None)


def _frontier_rerun_archive_snapshot(archive: ParetoArchive | None) -> dict[str, Any]:
    """Compact evidence that the current-era archive was rebuilt at rerun clear."""
    if archive is None:
        return {"status": "unavailable", "reason": "archive not supplied"}
    try:
        summary = dict(archive.summary(tier=DEFAULT_FRONTIER_TIER))
        frontier = archive.frontier(tier=DEFAULT_FRONTIER_TIER)
    except Exception as exc:  # pragma: no cover - defensive operator evidence only
        return {"status": "error", "reason": str(exc)}

    return {
        "status": "ok",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "tier": DEFAULT_FRONTIER_TIER,
        "frontier_size": int(summary.get("frontier_size") or 0),
        "total_entries": int(summary.get("total_entries") or 0),
        "hypervolume": summary.get("hypervolume"),
        "best_quality": summary.get("best_quality"),
        "best_speed": summary.get("best_speed"),
        "trial_ids": [entry.trial_id for entry in frontier],
    }


def _frontier_rerun_min_trials(marker: dict[str, Any]) -> int:
    try:
        return max(1, int(marker.get("min_numeric_trials") or 1))
    except (TypeError, ValueError):
        return 1


def _frontier_rerun_opened_ts(marker: dict[str, Any]) -> float | None:
    for key in ("rerun_started_at", "opened_at"):
        ts = parse_journal_ts(marker.get(key))
        if ts is not None:
            return ts
    return None


def _frontier_rerun_completed_numeric_trials(
    marker: dict[str, Any],
    journal: ExperimentJournal | None,
) -> int:
    if journal is None:
        return 0
    opened_ts = _frontier_rerun_opened_ts(marker)
    entries = (
        journal.entries_with_supersessions()
        if hasattr(journal, "entries_with_supersessions")
        else journal.all_entries()
    )
    completed = 0
    for entry in entries:
        if entry.bug_corrupted_by:
            continue
        if entry.action_type != "numeric_trial":
            continue
        if entry.tier < MIN_FRONTIER_EVAL_TIER:
            continue
        ts = parse_journal_ts(entry.timestamp)
        if opened_ts is not None and (ts is None or ts < opened_ts):
            continue
        completed += 1
    return completed


def _force_frontier_rerun_numeric_action(
    action: dict[str, Any],
    state: dict[str, Any],
    *,
    marker: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    completed_trials: int,
    min_trials: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    forced, blocked_reason = _first_unblacklisted_numeric_trial_action(
        blacklist,
        trial_counter=trial_counter,
    )
    if forced is None:
        log.warning(
            "Frontier rerun required, but all numeric_trial surfaces are "
            "blacklisted: %s",
            blocked_reason,
        )
        state["frontier_rerun_blocked"] = {
            "trial_id": trial_counter,
            "reason": blocked_reason,
            "action": action,
            "completed_numeric_trials": completed_trials,
            "min_numeric_trials": min_trials,
        }
        return action, rationale

    reason = str(marker.get("reason") or "frontier rerun required")
    log.warning(
        "Frontier rerun required (%s); forcing numeric_trial(surface=%s) "
        "instead of '%s' (%d/%d completed numeric trials).",
        reason,
        forced["surface"],
        action.get("type", "unknown"),
        completed_trials,
        min_trials,
    )
    state["frontier_rerun_forced"] = {
        "trial_id": trial_counter,
        "reason": reason,
        "original_action": action,
        "forced_action": forced,
        "completed_numeric_trials": completed_trials,
        "min_numeric_trials": min_trials,
    }
    state["frontier_rerun_pending_clear"] = {
        "trial_id": trial_counter,
        "action": forced,
        "reason": reason,
        "completed_numeric_trials": completed_trials,
        "min_numeric_trials": min_trials,
    }
    state.pop("frontier_rerun_blocked", None)
    return (
        forced,
        {
            **(rationale or {}),
            "frontier_rerun_forced": True,
            "frontier_rerun_reason": reason,
            "frontier_rerun_completed_numeric_trials": completed_trials,
            "frontier_rerun_min_numeric_trials": min_trials,
        },
    )


def _build_feature_flags_block(
    lab: Any,
    *,
    denylisted_flags: set[str] | None = None,
) -> str:
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
    denylisted_flags = set(denylisted_flags or set())
    if denylisted_flags:
        lines.append(
            "Convention-denylisted flags: "
            + ", ".join(sorted(denylisted_flags))
            + " (do not propose structural_experiment for these flags)"
        )
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
    if denylisted_flags:
        lines.append(
            "RULE: never propose structural_experiment for convention-denylisted "
            "flags; the dispatcher will reject them before evaluation."
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


def _build_prior_planner_decision_digest(
    archive_path: Path = PLANNER_ARCHIVE_PATH,
    *,
    limit: int = PRIOR_DECISION_DIGEST_CAP,
) -> str:
    """Render bounded planner continuity from archive records, not chat resume."""
    if limit <= 0:
        return "  (disabled by AUTOPILOT_PRIOR_DECISION_DIGEST_CAP)"
    if not archive_path.exists():
        return "  (none yet — planner archive has no prior decisions)"

    try:
        tail: deque[str] = deque(maxlen=max(limit * 6, limit))
        with open(archive_path) as fh:
            for line in fh:
                if line.strip():
                    tail.append(line)
    except Exception as exc:
        return f"  (prior decision digest unavailable: {exc})"

    rows: list[str] = []
    for raw in reversed(tail):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if record.get("type") == "planner_coordinator":
            critique = record.get("critique_decision") or "none"
            degraded = "degraded" if record.get("degraded") else "clean"
            fallback = str(record.get("fallback_reason") or "").strip()
            issues = record.get("critique_issues") or []
            issue_text = ""
            if isinstance(issues, list) and issues:
                issue_text = " issues=" + "; ".join(str(i)[:80] for i in issues[:2])
            line = (
                f"  - action={record.get('action_type') or '?'} "
                f"draft={record.get('draft_provider') or '?'} "
                f"critic={record.get('critic_provider') or 'none'} "
                f"verdict={critique} status={degraded}"
            )
            if fallback:
                line += f" fallback={fallback[:120]}"
            rows.append(line + issue_text)
        elif record.get("type") == "planner_provider_call" and not record.get("ok", True):
            line = (
                f"  - provider={record.get('provider') or '?'} "
                f"role={record.get('role') or '?'} "
                f"status={record.get('status') or 'failed'}"
            )
            error = str(record.get("error") or "").strip()
            if error:
                line += f" error={error[:120]}"
            rows.append(line)

        if len(rows) >= limit:
            break

    if not rows:
        return "  (none yet — no coordinator decisions in planner archive tail)"
    rows.reverse()
    return "\n".join(rows)


def _build_repo_readiness_advisory(
    pickup_path: Path | None = None,
    *,
    limit: int = 5,
) -> str:
    """Render a passive repo-readiness pickup artifact for planner context."""
    raw_path = str(pickup_path or os.environ.get(REPO_READINESS_PICKUP_ENV, "")).strip()
    if not raw_path:
        return f"  (disabled; set {REPO_READINESS_PICKUP_ENV}=<advisory pickup json> to include)"

    path = Path(raw_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"  (repo-readiness pickup unavailable: {exc})"

    mode = str(payload.get("mode") or "")
    if mode != "advisory_only" or bool(payload.get("authority_gate")):
        return (
            "  (repo-readiness pickup ignored: expected mode=advisory_only "
            "and authority_gate=false)"
        )

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return "  (repo-readiness pickup has no candidate items)"

    lines = [
        "  Planner context only. This is NOT an acceptance gate and MUST NOT "
        "override owning handoffs, GitNexus impact, or measurement gates.",
    ]
    source_count = payload.get("source_item_count")
    item_count = payload.get("item_count")
    generated_at = payload.get("generated_at")
    meta: list[str] = []
    if generated_at:
        meta.append(f"generated_at={generated_at}")
    if source_count is not None:
        meta.append(f"source_items={source_count}")
    if item_count is not None:
        meta.append(f"shown_items={item_count}")
    if meta:
        lines.append("  " + ", ".join(meta))

    for item in items[: max(0, limit)]:
        if not isinstance(item, dict):
            continue
        priority = str(item.get("priority") or "?")
        repo = str(item.get("repo") or "?")
        criterion = str(item.get("criterion_id") or "?")
        objective = str(item.get("objective") or item.get("acceptance") or "").strip()
        item_id = str(item.get("id") or f"{repo}:{criterion}")
        objective_text = objective[:140] if objective else "(no objective)"
        lines.append(
            f"  - {priority} {item_id}: repo={repo} criterion={criterion}; "
            f"{objective_text}"
        )

    rules = payload.get("pickup_rules")
    if isinstance(rules, list) and rules:
        lines.append("  Required before acting: " + "; ".join(str(r) for r in rules[:4]))
    else:
        lines.append(
            "  Required before acting: review owning handoff; run GitNexus impact; "
            "rerun the scorer after any patch."
        )
    return "\n".join(lines)


def _latest_fable_gate_report_path(reports_dir: Path | None = None) -> Path | None:
    """Return the newest generated Fable gate report artifact, if present."""
    base = reports_dir or ORCH_ROOT / "orchestration" / "reports"
    candidates: list[Path] = []
    for pattern in FABLE_GATE_REPORT_GLOBS:
        candidates.extend(path for path in base.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _fable_gate_evidence_brief(action: dict[str, Any]) -> str:
    evidence = action.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        return ""

    wanted_keys = (
        "latest_seq_trial_id",
        "latest_combined_E",
        "latest_required_E",
        "latest_fresh_eval",
        "latest_seq_state",
        "open_requirements",
        "telemetry_collection_reason",
        "telemetry_collection_blocker",
        "canary_role_sample_deficit",
        "canary_arm_volume_deficit",
        "canary_arm_balance_deficits",
    )
    parts: list[str] = []
    for key in wanted_keys:
        if key not in evidence:
            continue
        value = evidence.get(key)
        if isinstance(value, float):
            value_text = f"{value:.6g}"
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text[:120]}")
    return "; ".join(parts[:6])


def _build_fable_gate_advisory(
    report_path: Path | None = None,
    *,
    reports_dir: Path | None = None,
    limit: int = 4,
) -> str:
    """Render latest generated Fable gate next-actions for planner context.

    This intentionally consumes an existing report artifact instead of running
    fable5_gate_report.py in the planner loop. The report generator performs
    live process and evidence checks that are too expensive for every planner
    turn; this block only makes the latest durable snapshot visible.
    """
    path = report_path or _latest_fable_gate_report_path(reports_dir)
    if path is None:
        return (
            "  (no Fable gate report artifact found; run "
            "uv run python scripts/autopilot/fable5_gate_report.py --json --strict)"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stat = path.stat()
    except Exception as exc:
        return f"  (Fable gate advisory unavailable: {exc})"

    summary = payload.get("summary") if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    actions = payload.get("next_actions") if isinstance(payload, dict) else []
    if not isinstance(actions, list):
        actions = []

    now_s = datetime.now(timezone.utc).timestamp()
    age_s = max(0, int(now_s - stat.st_mtime))
    freshness = "fresh" if age_s <= FABLE_GATE_ADVISORY_MAX_AGE_S else "stale"
    lines = [
        "  Planner context only. This is NOT an acceptance gate and MUST NOT "
        "override owning handoffs, GitNexus impact, or measurement gates.",
        (
            f"  latest_artifact={path.name} age_s={age_s} freshness={freshness} "
            f"ready={bool(payload.get('ready'))}"
        ),
    ]

    active = summary.get("active_next_action_keys")
    blocked = summary.get("blocked_next_action_keys")
    if active:
        lines.append(f"  active_next_actions={active}")
    if blocked:
        lines.append(f"  blocked_next_actions={blocked}")

    if not actions:
        lines.append("  (report contains no next_actions)")
        return "\n".join(lines)

    status_rank = {"active": 0, "ready": 1, "blocked": 2}
    sorted_actions = sorted(
        (action for action in actions if isinstance(action, dict)),
        key=lambda action: (
            status_rank.get(str(action.get("status") or ""), 9),
            str(action.get("priority") or "P9"),
        ),
    )
    for action in sorted_actions[: max(0, limit)]:
        key = str(action.get("key") or "?")
        status = str(action.get("status") or "?")
        priority = str(action.get("priority") or "?")
        reason = str(action.get("reason") or "").replace("\n", " ").strip()
        line = f"  - {priority} {status} {key}"
        if reason:
            line += f": {reason[:180]}"
        evidence = _fable_gate_evidence_brief(action)
        if evidence:
            line += f" [{evidence}]"
        elif status == "blocked":
            blocked_by = action.get("blocked_by")
            if isinstance(blocked_by, list) and blocked_by:
                line += f" [blocked_by={str(blocked_by[0])[:140]}]"
        lines.append(line)

    return "\n".join(lines)


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
    blacklist: list[dict[str, Any]] | None = None,
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
        forced, forced_rationale = _replace_blacklisted_seed_fallback(
            {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N},
            blacklist or [],
            {
                **(rationale or {}),
                "meta_action_forced_metric_trial": True,
            },
            reason_label="meta-action replacement",
        )
        return forced, forced_rationale
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

CONSTITUTION_PATH = SCRIPT_DIR / "constitution.md"
SYSTEM_CARD_PATH = SCRIPT_DIR / "system_card.md"
STACK_PRIORS_PATH = ORCH_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
REPO_READINESS_PICKUP_ENV = "AUTOPILOT_REPO_READINESS_PICKUP"
FABLE_GATE_REPORT_GLOBS = (
    "fable5_gate_report_*.json",
    "fable5_gate_*.json",
)
FABLE_GATE_ADVISORY_MAX_AGE_S = int(
    os.environ.get("AUTOPILOT_FABLE_GATE_ADVISORY_MAX_AGE_S", "14400")
)

CONTROLLER_PROMPT_TEMPLATE = """\
You are the AutoPilot meta-reasoning controller for an LLM orchestration stack.
Your job: analyze current system state and propose the SINGLE best next action.
This is a non-interactive JSON controller call, not a Claude Code Plan Mode
session. Do not create files, do not write a plan under ~/.claude/plans, and do
not use planner-side mutation tools. If the best move is operator-facing, say so
briefly in reasoning and still emit the closest valid AutoPilot action block.

## Controller Constitution (durable human-authored policy)

{constitution}

## Generated System Card (live facts, regenerated from repository state)

{system_card}

## Current State

### Pareto Archive
{pareto_summary}

### Pareto Frontier Geometry
{pareto_geometry}

### Evidence Power and Sequential Candidate Status
{planner_evidence}

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

### Fable 5 Gate Advisory (latest generated report, non-authority)
{fable_gate_advisory}

### StrategyStore Planner Hints (refreshed each planner turn)
Planner tool boundary: StrategyStore rows mentioning tools, REPL, CALL, or
tool-use refer to orchestrator/model execution inside AutoPilot actions and
evals. They are not permission for this controller process to call planner-side
tools. The planner process is read-only: while drafting the JSON action, use
only Read/Grep/Glob for repository inspection when needed. Never use Bash, Edit,
MultiEdit, Write, NotebookEdit, apply_patch, or any other planner-side
mutation/execution tool to satisfy a hint. If a mutation is warranted, return an
AutoPilot action such as code_mutation, prompt_mutation, structural_experiment,
or numeric_trial and let the orchestrator dispatch it.
{planner_strategy_hints}

### Repo-Readiness Advisory Pickup (default-off, non-authority)
{repo_readiness_advisory}

### Species Budget
{budget}

### Suite Quality Trends (last 10 evals)
{suite_quality_trends}

### Recent Insights (cross-species, structured per action_type, bug-corrupted excluded)
{insights_structured}

### Exploration mode
Stagnation signal: {stagnation_signal}

{exploration_block}

### Short-Term Memory (accumulated learnings this session)
{short_term_memory}

### Prior Planner Decisions (bounded archive digest — replaces chat-session resume)
{prior_planner_decisions}

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
   (use keep_ratio=0.3, target a live port from Slot Memory or the generated system card)

## Available Actions

Respond with EXACTLY ONE action in a ```json:autopilot_actions block:

- Seed: {{"type": "seed_batch", "n_questions": 10-50, "suites": ["coder","math",...]}}
- Numeric: {{"type": "numeric_trial", "surface": "{numeric_surface_options}", "params": {{}}}}
  (Leave params empty to let Optuna suggest; provide params to test specific values)
- Prompt: {{"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix|compress|few_shot_evolution", "description": "..."}}
- GEPA: {{"type": "gepa_optimize", "file": "frontdoor.md", "max_evals": 50, "description": "..."}}
  (AP-19: Evolutionary prompt optimization via GEPA — runs ~50 evals internally, returns best candidate)
- Code: {{"type": "code_mutation", "file": "src/escalation.py", "mutation": "targeted_fix", "description": "..."}}
  (Mutate Python code — ONLY files in allowlist: {code_targets})
- Structural: {{"type": "structural_experiment", "flags": {{"feature_name": true/false}}}}
- Prune: {{"type": "structural_prune", "file": "frontdoor.md", "block": "## Section Name", "description": "..."}}
  (Delete an instruction block from a .md prompt file — accepted only if quality >= baseline AND instruction_token_ratio decreases)
- Compact: {{"type": "slot_compact", "port": 0, "slot_id": 0, "keep_ratio": 0.3, "scorer": "expected_attention", "keep_first": 5, "n_future": 128}}
  (AM KV compaction — replace port=0 with a live port from Slot Memory or the generated system card before emitting the action. Use after long-context queries to free memory. Evaluates quality post-compact.)
- Train: {{"type": "train_routing_models", "min_memories": 500}}
  (min_memories: integer 1-100000; default 500. Do NOT set it to the current
  memory_count — the validator REJECTS any value above 100000.)
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


def _read_guidance_file(path: Path, missing_label: str) -> str:
    try:
        return path.read_text()
    except OSError:
        return f"({missing_label} not found)"


def _render_system_card(state: dict[str, Any] | None = None) -> str:
    """Render live controller facts, failing closed if generation breaks."""
    try:
        from gen_system_card import generate_system_card

        return generate_system_card(ORCH_ROOT, state_override=state)
    except Exception as exc:
        return (
            "# AutoPilot Generated System Card\n\n"
            "SYSTEM CARD GENERATION FAILED.\n\n"
            "## Degraded Stack Guidance\n\n"
            f"- generator_error: {type(exc).__name__}: {exc}\n"
            "- Live role, port, tier, throughput, baseline, and trust-boundary facts "
            "are unavailable.\n"
            "- Do not use checked-in `system_card.md`, historical program text, "
            "handoffs, memories, or old logs as authoritative stack truth.\n"
            "- Choose an observational or no-stack-change action, or pause for "
            "operator repair, until `scripts/autopilot/gen_system_card.py --check` "
            "passes again.\n"
        )


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
    suppressed_numeric_surfaces: set[str] | None = None,
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
    # Validator caps train_routing_models.min_memories at 100000 (controller_io.py).
    # When the live corpus already exceeds the cap, say so explicitly so the planner
    # does NOT infer "use the current memory_count" and cross the hidden schema limit
    # (root cause of the #776 invalid-action pause, 2026-06-11).
    if memory_count > 100_000 and "train_routing_models" not in blocked:
        _cap_note = (
            f"min_memories is capped at 100000 — do NOT set it to the live "
            f"memory_count ({memory_count}); valid range 1-100000 (default 500)"
        )
        _existing = cautions.get("train_routing_models")
        cautions["train_routing_models"] = (
            f"{_existing}; {_cap_note}" if _existing else _cap_note
        )

    if not _slot_compaction_viable(slot_memory_text):
        blocked["slot_compact"] = (
            "no queried slot currently has cached tokens to compact; this is not "
            "evidence that the eval instrument or host is contaminated"
        )

    seed_exhaustion = _seed_fallback_exhaustion_reason(blacklist)
    if seed_exhaustion:
        blocked["seed_batch"] = (
            "all configured measured seed fallback candidates are blacklisted; "
            f"last reason: {seed_exhaustion}"
        )

    suppressed_numeric_surfaces = set(suppressed_numeric_surfaces or set())
    if suppressed_numeric_surfaces:
        available_numeric_surfaces = _configured_numeric_surfaces()
        if available_numeric_surfaces:
            cautions["numeric_trial"] = (
                "convention-suppressed numeric surfaces are unavailable: "
                + ", ".join(sorted(suppressed_numeric_surfaces))
                + "; use only the numeric surfaces shown in the action schema"
            )
        else:
            blocked["numeric_trial"] = (
                "all numeric surfaces are suppressed by planner conventions: "
                + ", ".join(sorted(suppressed_numeric_surfaces))
            )

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

    try:
        lines.append(build_action_availability_section(load_capability_registry()))
    except Exception as exc:
        lines.append(f"Capability registry levers (generated): unavailable ({exc})")

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

    block = _EXPLORATION_RICH_TEMPLATE.format(
        n=CREATIVITY_N,
        window=TAIL_WINDOW,
        tail_seeds=tail_text,
        unfalsified=unfalsified_text,
    )
    return block, "; ".join(reasons)


# ── State Management ─────────────────────────────────────────────


# State / blacklist / signatures helpers moved to state_store.py (2026-05-22 refactor).
# Wrappers below preserve the original autopilot.py API by supplying STATE_PATH,
# BLACKLIST_PATH, and the model-quality-signatures path.

_MODEL_SIGNATURES_PATH = ORCH_ROOT / "orchestration" / "model_quality_signatures.yaml"
_MODEL_DESCRIPTORS_PATH = ORCH_ROOT / "orchestration" / "model_descriptors.yaml"


def _maybe_reimport_pareto_from_journal(
    archive: "ParetoArchive",
    journal: "ExperimentJournal",
    trial_id: int,
) -> bool:
    """Re-add a single journal entry to the Pareto archive if missing.

    Per handoffs/active/autopilot-exogenous-restart-resilience.md Section 5.7.
    Handles the historical corruption window where the journal advanced but
    the derived state-cache archive was not persisted → on restart the journal
    has the entry but the on-disk Pareto archive does not.

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
    entries = (
        journal.entries_with_supersessions()
        if hasattr(journal, "entries_with_supersessions")
        else journal.all_entries()
    )
    entry = next((e for e in entries if e.trial_id == trial_id), None)
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
    # The startup journal-authority sync persists the reconstructed archive
    # after recovery finishes. Do not write a one-off archive snapshot here;
    # it can diverge from the folded append-only journal view.
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


def _normalize_state_before_save(state: dict[str, Any]) -> None:
    if not state.get("paused"):
        state.pop("pause_reason", None)


def save_state(state: dict[str, Any]) -> None:
    _normalize_state_before_save(state)
    _save_state_impl(STATE_PATH, state)


def _merge_external_control_fields(
    state: dict[str, Any],
    disk_state: dict[str, Any] | None = None,
) -> list[str]:
    """Preserve operator control fields changed while a trial was in-flight.

    The loop keeps a long-lived in-memory state dict during dispatch/eval. An
    external ``autopilot.py pause`` writes ``paused=True`` to disk, but the
    trial-end save path can otherwise overwrite it with stale ``paused=False``.
    Merge only control-plane fields, never counters or in-flight metadata.
    """
    if disk_state is None:
        try:
            disk_state = load_state()
        except Exception as exc:
            log.warning("Failed to reload control fields before state save: %s", exc)
            return []

    changed: list[str] = []
    for key in ("paused", "_in_cache_flush"):
        if key in disk_state and disk_state.get(key) != state.get(key):
            state[key] = disk_state[key]
            changed.append(key)
    return changed


def _archive_entry_count(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    entries = payload.get("all_entries")
    return len(entries) if isinstance(entries, list) else 0


def _journal_archive_payload_for_authority(
    journal: ExperimentJournal,
    *,
    deinflate_before_ts: float | None = None,
    deinflate_factor: float = 1.0,
    exclude_before_ts: float | None = None,
) -> dict[str, Any] | None:
    rows = _journal_rows_for_archive(journal)
    if hasattr(journal, "ledger_events"):
        snapshot_payload = archive_payload_from_verified_snapshot(
            rows,
            journal.ledger_events(),
        )
        if snapshot_payload is not None:
            return snapshot_payload
    return reconstruct_archive_from_journal_rows(
        rows,
        None,
        current_run_only=False,
        deinflate_before_ts=deinflate_before_ts,
        deinflate_factor=deinflate_factor,
        exclude_before_ts=exclude_before_ts,
    )


def _archive_epoch_params_from_state(
    state: dict[str, Any],
) -> tuple[float | None, float, float | None]:
    """Extract Pareto replay epoch params from autopilot state."""
    strict_epoch = _numeric_swarm_epoch_label_from_state(state) is not None
    deinflate_before_ts: float | None = None
    try:
        deinflate_before_ts = float(state.get("pareto_epoch_ts") or 0.0) or None
    except (TypeError, ValueError) as exc:
        if strict_epoch:
            raise ValueError("invalid pareto_epoch_ts for active speed era") from exc
    deinflate_factor: float = 1.0
    try:
        deinflate_factor = float(state.get("pareto_pre_epoch_speed_factor", 1.0))
    except (TypeError, ValueError) as exc:
        if strict_epoch:
            raise ValueError("invalid pareto_pre_epoch_speed_factor for active speed era") from exc
    exclude_before_ts: float | None = None
    try:
        exclude_before_ts = float(state.get("pareto_exclude_before_ts") or 0.0) or None
    except (TypeError, ValueError) as exc:
        if strict_epoch:
            raise ValueError("invalid pareto_exclude_before_ts for active speed era") from exc
    if strict_epoch and (deinflate_before_ts is None or exclude_before_ts is None):
        raise ValueError(
            "active speed era requires pareto_epoch_ts and pareto_exclude_before_ts"
        )
    return deinflate_before_ts, deinflate_factor, exclude_before_ts


def _numeric_swarm_epoch_label_from_state(state: dict[str, Any]) -> str | None:
    """Return the persistent NumericSwarm study era for the live speed instrument."""
    eras = state.get("active_instrument_eras")
    if isinstance(eras, dict):
        value = eras.get("autopilot_speed")
        if isinstance(value, str) and value.strip():
            return value.strip()
    marker = state.get("frontier_rerun_required")
    if isinstance(marker, dict) and marker.get("required"):
        raise ValueError("frontier rerun requires active_instrument_eras.autopilot_speed")
    return None


def _apply_journal_archive_authority(
    state: dict[str, Any],
    journal: ExperimentJournal,
    archive: ParetoArchive,
) -> bool | None:
    """Apply the append-only journal archive fold to memory.

    Returns True when cached state changed, False when it already matched, and
    None when the journal has no archive-bearing rows.
    """
    deinflate_before_ts, deinflate_factor, exclude_before_ts = (
        _archive_epoch_params_from_state(state)
    )
    archive_payload = _journal_archive_payload_for_authority(
        journal,
        deinflate_before_ts=deinflate_before_ts,
        deinflate_factor=deinflate_factor,
        exclude_before_ts=exclude_before_ts,
    )
    if archive_payload is None:
        return None
    changed = "pareto_archive" in state
    state.pop("pareto_archive", None)
    archive._replace_from_archive_payload(archive_payload)
    return changed


def _sync_startup_archive_from_journal_authority(
    state: dict[str, Any],
    journal: ExperimentJournal,
    archive: ParetoArchive,
) -> bool:
    """Remove stale startup archive cache before writing lifecycle state.

    Preflight repairs archive authority from the append-only journal, but a
    direct ``autopilot.py start`` must uphold the same invariant. Otherwise the
    first startup save (fleet timestamp, pause-loop bookkeeping, etc.) can
    preserve a stale cached ``state["pareto_archive"]`` before any trial runs.
    """
    if state.get("_allow_empty_frontier_rebase"):
        return False
    before_count = _archive_entry_count(state.get("pareto_archive"))
    deinflate_before_ts, deinflate_factor, exclude_before_ts = (
        _archive_epoch_params_from_state(state)
    )
    journal_count = _archive_entry_count(
        _journal_archive_payload_for_authority(
            journal,
            deinflate_before_ts=deinflate_before_ts,
            deinflate_factor=deinflate_factor,
            exclude_before_ts=exclude_before_ts,
        )
    )
    changed = _apply_journal_archive_authority(state, journal, archive)
    if changed is not True:
        return False
    log.warning(
        "Startup archive cache removed; journal fold is authoritative "
        "(cached_state_entries %d, journal_entries %d)",
        before_count,
        journal_count,
    )
    return True


def _baseline_state_for_startup_gate(
    state: dict[str, Any],
    journal: ExperimentJournal,
) -> dict[str, Any]:
    """Return the startup baseline authority payload for SafetyGate.

    ``baseline_state`` stays authoritative while present. After W4 cutover
    removes that cache, the append-only promotion ledger is allowed to seed the
    gate only when the same reconciliation rules say the fold is cutover-ready.
    """
    state_baseline = state.get("baseline_state")
    if isinstance(state_baseline, dict) and state_baseline:
        return state_baseline
    reconciliation = reconcile_baseline_ledger(
        journal.baseline_promotion_events(),
        None,
    )
    if reconciliation.cutover_ready and isinstance(reconciliation.folded_state, dict):
        log.info("Using baseline promotion ledger fold for SafetyGate startup baseline")
        return reconciliation.folded_state
    return {}


def _save_state_with_journal_archive_authority(
    state: dict[str, Any],
    journal: ExperimentJournal,
    archive: ParetoArchive,
    *,
    context: str,
) -> bool:
    """Persist state with append-only journal folds as cache authority."""
    archive_changed = _apply_journal_archive_authority(state, journal, archive)
    baseline_changed = apply_baseline_ledger_authority(
        state,
        journal.baseline_promotion_events(),
    )
    if archive_changed is None:
        log.warning(
            "Journal archive authority unavailable during %s; saving lifecycle "
            "state without a legacy archive cache",
            context,
        )
        save_state(state)
    else:
        if archive_changed:
            log.info("State archive cache removed during %s", context)
        save_state(state)
    if baseline_changed:
        log.info("State baseline cache removed during %s", context)
    return archive_changed is not None


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
    """Load descriptor-backed model signatures for planner context."""
    return _load_model_signatures_impl(_MODEL_SIGNATURES_PATH, _MODEL_DESCRIPTORS_PATH)


def append_blacklist(action: dict[str, Any], trial_id: int, reason: str) -> None:
    """Auto-append a blacklist entry after rollback trigger."""
    _append_blacklist_impl(action, trial_id, reason, BLACKLIST_PATH)


# ── Slot Memory Visibility (AM KV Compaction) ──────────────────


# Degraded fallback only. Runtime slot visibility should follow generated stack
# priors so the planner does not miss newly added live llama-server ports.
_FALLBACK_SLOT_QUERY_PORTS: dict[str, list[int]] = {
    "frontdoor": [8070],
    "coder": [8070],  # shares server with frontdoor (same Qwen3.6-35B Q8 GGUF)
    "worker": [8072],
    "architect_general": [8083],
}


def _slot_query_ports_from_stack_priors(
    stack_priors_path: Path = STACK_PRIORS_PATH,
) -> dict[str, list[int]]:
    """Return primary live llama-server ports by role from generated stack priors."""
    return live_stack_slot_query_ports(stack_priors_path)


def _slot_query_ports() -> dict[str, list[int]]:
    """Return live slot-query ports, falling back only when stack priors fail."""
    return _slot_query_ports_from_stack_priors() or dict(_FALLBACK_SLOT_QUERY_PORTS)


def _query_slot_memory() -> str:
    """Query llama-server /slots on production ports and return a summary.

    Returns a compact text block showing per-role slot memory usage so the
    controller can decide when slot_compact is worthwhile.
    """
    import httpx

    cached_lines: list[str] = []
    empty_ports: list[str] = []
    unavailable_ports: list[str] = []
    for role, ports in _slot_query_ports().items():
        for port in ports:
            try:
                resp = httpx.get(
                    f"http://localhost:{port}/slots", timeout=3.0
                )
                if resp.status_code != 200:
                    unavailable_ports.append(f"{role}:{port} http {resp.status_code}")
                    continue
                slots = resp.json()
                if not isinstance(slots, list):
                    continue
                port_cached = 0
                for s in slots:
                    sid = s.get("id", "?")
                    state = s.get("state", "?")
                    n_past = s.get("n_past", 0)
                    if n_past > 0:
                        port_cached += int(n_past)
                        cached_lines.append(
                            f"  {role}:{port}/slot{sid} — {state}, {n_past} tokens cached"
                        )
                if port_cached == 0:
                    empty_ports.append(f"{role}:{port}")
            except Exception:
                unavailable_ports.append(f"{role}:{port}")
    lines = list(cached_lines)
    if empty_ports:
        lines.append(
            "  healthy queried ports with empty KV cache: " + ", ".join(empty_ports)
        )
    if unavailable_ports:
        lines.append(
            "  unavailable configured replica ports: " + ", ".join(unavailable_ports)
        )
        lines.append(
            "  note: unavailable replicas only affect slot_compact targeting; "
            "do not infer eval-instrument contamination from this line"
        )
    if not lines:
        return "  (no slot data returned; slot_compact has no target)"
    return "\n".join(lines)


# ── Claude CLI Controller ────────────────────────────────────────
# Controller invocation, action extraction, and AP-9 single-variable
# validation moved to controller_io.py (2026-05-22 refactor). Wrapper below
# preserves the original cwd=ORCH_ROOT semantics.


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


def _make_eval_progress_callback(
    *,
    phase: PhaseTracker,
    tui: "AutoPilotTUI | None",
    trial_id: Callable[[], int],
    action: Callable[[], dict[str, Any] | None],
) -> Callable[[str], None]:
    def _callback(prompt: str) -> None:
        if tui is not None:
            tui.set_prompt(prompt)
        current_action = action() or {}
        phase.set(
            "dispatch_action",
            trial_id=trial_id(),
            action_type=current_action.get("type", ""),
            idle_reason="evaluating question",
            prompt_preview=str(prompt)[:240],
        )

    return _callback


def _make_eval_batch_progress_callback(
    *,
    phase: PhaseTracker,
    trial_id: Callable[[], int],
    action: Callable[[], dict[str, Any] | None],
) -> Callable[[dict[str, Any]], None]:
    def _callback(progress: dict[str, Any]) -> None:
        current_action = action() or {}
        phase.set(
            "dispatch_action",
            trial_id=trial_id(),
            action_type=current_action.get("type", ""),
            idle_reason="evaluating question",
            eval_label=progress.get("label"),
            eval_completed_questions=progress.get("completed_questions"),
            eval_total_questions=progress.get("total_questions"),
            eval_correct_questions=progress.get("correct_questions"),
            eval_correct_pct=progress.get("correct_pct"),
            eval_concurrency=progress.get("concurrency"),
        )

    return _callback


def _run_loop_inner(
    max_trials: int | None,
    dry_run: bool,
    use_controller: bool,
    tui: "AutoPilotTUI | None" = None,
) -> None:
    """Inner loop (separated to ensure TUI cleanup via run_loop's finally)."""
    state = load_state()
    journal = ExperimentJournal()
    _deinfl_ts, _deinfl_factor, _exclude_ts = _archive_epoch_params_from_state(state)
    archive_payload = _journal_archive_payload_for_authority(
        journal,
        deinflate_before_ts=_deinfl_ts,
        deinflate_factor=_deinfl_factor,
        exclude_before_ts=_exclude_ts,
    )
    if archive_payload is not None and not state.get("_allow_empty_frontier_rebase"):
        archive = ParetoArchive.from_archive_payload(archive_payload, read_only=False)
    else:
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
        baseline_state=_baseline_state_for_startup_gate(state, journal),
    )
    tower = EvalTower(
        url=ORCHESTRATOR_URL,
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
    )
    swarm = NumericSwarm(epoch_label=_numeric_swarm_epoch_label_from_state(state))
    forge = PromptForge(auto_commit=not dry_run)
    lab = StructuralLab(orchestrator_url=ORCHESTRATOR_URL)
    evo = EvolutionManager(use_local_model=not use_controller)

    # AP-22: Short-term memory (accumulated learnings across trials)
    memory = ShortTermMemory()
    memory.refresh_from_journal(journal)
    last_criticism_text = "(first trial — no prior criticism)"

    # B1: Strategy store for species memory
    strategy_store: StrategyStore | None = None
    try:
        strategy_store = StrategyStore()
        log.info("Strategy store loaded (%d entries)", strategy_store.count())
    except Exception as e:
        log.warning("Strategy store unavailable: %s", e)
    _install_planner_convention_bindings(strategy_store, journal)

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
    _sync_startup_archive_from_journal_authority(state, journal, archive)
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
    current_action: dict[str, Any] | None = None
    eval_progress_callback = _make_eval_progress_callback(
        phase=phase,
        tui=tui,
        trial_id=lambda: trial_counter,
        action=lambda: current_action,
    )
    eval_batch_progress_callback = _make_eval_batch_progress_callback(
        phase=phase,
        trial_id=lambda: trial_counter,
        action=lambda: current_action,
    )
    tower.on_question = eval_progress_callback
    tower.on_progress = eval_batch_progress_callback
    seeder.on_question = eval_progress_callback

    # ── Plot freshness, decoupled from the trial loop ──────────────────────
    # Pre-2026-06-07, plots regenerated ONLY at `trial_counter % PLOT_INTERVAL
    # == 0`, fired from inside this loop. So any stop on a non-multiple-of-N
    # (operator pause, SIGTERM, max-trials, internal halt) froze the PNGs at the
    # last decade boundary — observed: a run reached trial 707/708 but the
    # dashboard PNGs stayed at the trial-700 render. We now ALSO refresh on a
    # wall-clock timer (mid-decade) and force a synchronous render on every
    # lifecycle transition so the operator's view always reflects the last
    # completed trial.
    plot_clock = {"last_ts": 0.0, "last_pause_trial": -1}

    def _refresh_plots(*, sync: bool, reason: str) -> None:
        """Regenerate the dashboard PNGs from current on-disk state.

        ``sync=True`` blocks until the child ``autopilot.py plot`` completes —
        for lifecycle transitions where the process may exit right after.
        ``sync=False`` submits the existing non-blocking async subprocess
        (steady state). Both read freshly-saved state/journal from disk, so a
        ``sync=True`` caller must have ``save_state()``-ed first.
        """
        plot_clock["last_ts"] = time.time()
        cmd = [sys.executable, str(SCRIPT_DIR / "autopilot.py"), "plot"]
        if sync:
            try:
                subprocess.run(cmd, cwd=ORCH_ROOT, timeout=PLOT_SYNC_TIMEOUT_S, check=False)
                log.info("Plots refreshed synchronously (%s, trial=%d)", reason, trial_counter)
            except Exception as exc:
                log.warning("Synchronous plot refresh (%s) failed: %s", reason, exc)
        else:
            async_tasks.submit_subprocess(f"plots-trial-{trial_counter}", cmd, cwd=ORCH_ROOT)

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
            # Refresh plots once on ENTERING a pause episode. trial_counter is
            # frozen while paused, so the guard fires exactly once per episode
            # (not every PAUSE_POLL_S poll) and re-fires only if trials advanced
            # before a later re-pause. This covers operator `pause` AND internal
            # halts (meta-loop latch), which rarely land on a %N boundary.
            if plot_clock["last_pause_trial"] != trial_counter:
                plot_clock["last_pause_trial"] = trial_counter
                _refresh_plots(sync=True, reason="pause")
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
        critic_fallback_skip: SkipOutcome | None = None
        if use_controller:
            phase.set(
                "planner_prompt_build",
                trial_id=trial_counter,
                idle_reason="building controller prompt",
            )
            # Load split controller guidance (W8): durable human policy plus
            # generated live system card. program.md remains historical only.
            constitution_text = _read_guidance_file(
                CONSTITUTION_PATH,
                "constitution.md",
            )
            system_card_text = _render_system_card(state)
            # B2: Format blacklist for controller. The full blacklist is always
            # enforced at dispatch by check_blacklist(). The prompt view shows
            # recent entries with reasons plus compact older patterns so the
            # planner/critic do not unknowingly re-propose hidden hard blocks.
            blacklist_text = _format_blacklist_for_prompt(blacklist)

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

            # Stepping-stone lane (intake-772 Darwin Gödel Machine): append a diverse sample of
            # dominated-but-novel configs to the geometry block so the planner sees exploration
            # seeds beyond the frontier. Observe-only + fully guarded — never breaks the prompt.
            # Gated by AUTOPILOT_STEPPING_STONES (default on); set to "0" for the frontier-only
            # arm of the ablation in scripts/autopilot/STEPPING_STONE_ABLATION_PROTOCOL.md.
            if os.environ.get("AUTOPILOT_STEPPING_STONES", "1") != "0":
                try:
                    _stepping_text = archive.stepping_stones_text(
                        tier=DEFAULT_FRONTIER_TIER,
                        limit=8,
                    )
                    if _stepping_text:
                        pareto_geometry_text = f"{pareto_geometry_text}\n\n{_stepping_text}"
                except Exception as _exc:
                    pareto_geometry_text = (
                        f"{pareto_geometry_text}\n\n(stepping-stones unavailable: {_exc})"
                    )

            try:
                planner_evidence_text = format_planner_evidence_section(
                    asdict(entry)
                    for entry in journal.entries_with_supersessions()
                )
            except Exception as _exc:
                planner_evidence_text = f"(planner evidence unavailable: {_exc})"

            _refresh_planner_convention_bindings(
                strategy_store,
                journal,
                reason=f"planner_turn:{trial_counter}",
            )

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
                    suppressed_numeric_surfaces=_PLANNER_SUPPRESSED_NUMERIC_SURFACES,
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

            planner_strategy_hints_text = _build_planner_strategy_hints(
                strategy_store,
                journal,
            )

            prompt = CONTROLLER_PROMPT_TEMPLATE.format(
                constitution=constitution_text,
                system_card=system_card_text,
                pareto_summary=archive.summary_text(tier=DEFAULT_FRONTIER_TIER),
                pareto_geometry=pareto_geometry_text,
                planner_evidence=planner_evidence_text,
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
                fable_gate_advisory=_build_fable_gate_advisory(),
                planner_strategy_hints=planner_strategy_hints_text,
                repo_readiness_advisory=_build_repo_readiness_advisory(),
                budget=json.dumps(meta.budget.as_dict(), indent=2),
                suite_quality_trends=_format_suite_trends(journal.suite_quality_trend(10)),
                insights_structured=insights_structured_text,
                stagnation_signal=stagnation_signal,
                exploration_block=exploration_block,
                short_term_memory=memory.refresh_from_journal(journal),  # W5 generated STM
                prior_planner_decisions=_build_prior_planner_decision_digest(),
                last_criticism=last_criticism_text,  # AP-23
                model_signatures=model_signatures_text,
                blacklist_text=blacklist_text,
                feature_flags_block=_build_feature_flags_block(
                    lab,
                    denylisted_flags=_PLANNER_DENYLISTED_FEATURE_FLAGS,
                ),
                last_invalid_feedback=_build_last_invalid_feedback(state),
                numeric_surface_options="|".join(_configured_numeric_surfaces()),
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

            # Degraded/uncritiqued safety gate (2026-06-07, tightened 2026-06-12):
            # two cases (see uncritiqued_dispatch_block_reason).
            #   (A) NO critique object — the PRIMARY failed and the draft fell back
            #       to the critic provider: uncritiqued AND untrusted, so dispatch
            #       only if observational, else PAUSE.
            #   (B) verdict "unavailable" — the PRIMARY drafted fine but the binding
            #       critic could not review it (timeout/empty/unparseable/circuit).
            #       RISK CLASS ALONE IS NOT A SUFFICIENT GUARD (the @708 failure was
            #       "low-risk" seed looping): HIGH → pause; seed_batch/passive →
            #       pause unless explicitly observational+one-shot; MEDIUM experiment
            #       → proceed ONLY IF novel and non-looping (not blacklisted, not a
            #       recurring-invalid signature, carries a real falsifier). The gate
            #       is pure, so supply the blacklist + invalid-signature context here.
            # Mirrors the critic_reject_loop / meta / skip durable halts.
            _gate_action = planner_decision.action
            _gate_is_bl = bool(
                isinstance(_gate_action, dict)
                and check_blacklist(_gate_action, blacklist)
            )
            _gate_is_rep = bool(
                isinstance(_gate_action, dict)
                and _action_signature(_gate_action)
                in (state.get("invalid_signature_counts", {}) or {})
            )
            _uncritiqued_block = uncritiqued_dispatch_block_reason(
                planner_decision,
                is_blacklisted=_gate_is_bl,
                is_repeated=_gate_is_rep,
            )
            if _uncritiqued_block:
                state["paused"] = True
                state["_dispatch_deficiency"] = _uncritiqued_block
                save_state(state)
                log.error(
                    "Planner ran degraded with NO critic verdict and drafted a "
                    "non-observational action %r — pausing (%s) for operator review "
                    "instead of dispatching unreviewed (fallback_reason=%r).",
                    action.get("type") if isinstance(action, dict) else action,
                    _uncritiqued_block,
                    planner_decision.fallback_reason,
                )
                phase.set("critic_unavailable_halt", trial_id=trial_counter)
                break

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
                critic_fallback_skip = _critic_fallback_seed_skip(action, blacklist)
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
            action, rationale = _replace_blacklisted_autonomous_action(
                action,
                blacklist,
                rationale,
            )

        if not action:
            if getattr(planner_decision, "providers_unavailable", False):
                # BOTH planner models are offline (no usable response from any
                # attempted model). Don't loop a model-free seed_batch silently and
                # don't route through the LLM-critic gate (it's also offline). Pick a
                # DETERMINISTIC, statistically-grounded Optuna numeric_trial if the
                # numeric-swarm surfaces are available; otherwise pause for the
                # operator. (cross-model failover, 2026-06-12)
                _det_surface = None
                try:
                    from species.numeric_swarm import SURFACES as _NS_SURFACES

                    if _NS_SURFACES:
                        _surface_names = sorted(_NS_SURFACES.keys())
                        _det_surface = (
                            "memrl_retrieval"
                            if "memrl_retrieval" in _NS_SURFACES
                            else _surface_names[0]
                        )
                except Exception:
                    _det_surface = None

                if _det_surface is not None:
                    # numeric_trial without "params" → Optuna fills them at execution
                    # (trial 969 ran exactly this shape; the dispatch path handles it).
                    action = {"type": "numeric_trial", "surface": _det_surface}
                    log.warning(
                        "Both planner models offline → deterministic Optuna "
                        "numeric_trial fallback (surface=%s)",
                        _det_surface,
                    )
                else:
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "planners_offline_no_deterministic_fallback"
                    save_state(state)
                    log.error(
                        "Both planner models offline and no deterministic fallback "
                        "available — pausing for operator."
                    )
                    phase.set("planners_offline_halt", trial_id=trial_counter)
                    break
            else:
                log.warning("No action proposed, defaulting to seed_batch")
                action = {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}

        # Meta actions are allowed as occasional bookkeeping, but a repeated
        # metric-free action means the planner is avoiding the experiment loop.
        action, rationale = _force_metric_action_after_meta(
            action,
            state,
            rationale,
            blacklist,
        )

        # Experiment quota: once memory is large, cap consecutive passive
        # (seed/distill) actions so the planner cannot rationalize no-op work
        # forever — force a frontier-moving experiment instead.
        action, rationale = _enforce_experiment_quota(
            action, state, memory_count, rationale, trial_counter, blacklist,
        )

        seq_fresh_eval_context: dict[str, Any] | None = None
        action, rationale, seq_fresh_eval_context = _maybe_force_seq_promotion_fresh_eval(
            action,
            state=state,
            blacklist=blacklist,
            rationale=rationale,
            trial_counter=trial_counter,
            enabled=gate.use_sequential,
        )
        seq_baseline_draw_reference: dict[str, Any] | None = None
        seq_candidate_replay_context: dict[str, Any] | None = None
        if seq_fresh_eval_context is None:
            action, rationale, seq_baseline_draw_reference = _maybe_force_seq_baseline_draw(
                action,
                state=state,
                journal=journal,
                tier=DEFAULT_FRONTIER_TIER,
                blacklist=blacklist,
                rationale=rationale,
                trial_counter=trial_counter,
                enabled=gate.use_sequential,
            )
        if seq_fresh_eval_context is None and seq_baseline_draw_reference is None:
            action, rationale, seq_candidate_replay_context = _maybe_force_seq_candidate_replay(
                action,
                state=state,
                journal=journal,
                tier=DEFAULT_FRONTIER_TIER,
                blacklist=blacklist,
                rationale=rationale,
                trial_counter=trial_counter,
                enabled=gate.use_sequential,
            )

        action, rationale = _maybe_force_frontier_rerun_action(
            action,
            state,
            journal=journal,
            archive=archive,
            blacklist=blacklist,
            rationale=rationale,
            trial_counter=trial_counter,
        )

        # ── 3. Act ───────────────────────────────────────────────
        # B2: Check failure blacklist before dispatch
        pre_dispatch_skip: SkipOutcome | None = critic_fallback_skip
        if pre_dispatch_skip is not None:
            log.warning(
                "Trial %d: %s",
                trial_counter,
                pre_dispatch_skip.reason,
            )
        else:
            action, rationale = _replace_blacklisted_seed_fallback(
                action,
                blacklist,
                rationale,
                reason_label="pre-dispatch",
            )
            blocked_reason = check_blacklist(action, blacklist)
            if blocked_reason:
                log.warning(
                    "Trial %d: action blacklisted (%s), recording invalid skip",
                    trial_counter, blocked_reason,
                )
                pre_dispatch_skip = _blacklisted_action_skip(action, blocked_reason)

        log.info("Trial %d: %s", trial_counter, json.dumps(action))
        current_action = action
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
        if pre_dispatch_skip is None:
            state["in_flight_trial"] = {
                "trial_id": trial_counter,
                "action": action,
                "started_at": time.time(),
                "host_pid": os.getpid(),
                "host_started_at": state.get("autopilot_fleet_started_at"),
            }
            save_state(state)

        if pre_dispatch_skip is not None:
            phase.set(
                "dispatch_precheck_skip",
                trial_id=trial_counter,
                action_type=pre_dispatch_skip.action_type,
                skip_status=pre_dispatch_skip.status,
            )
            eval_result = pre_dispatch_skip
            species_name = pre_dispatch_skip.action_type or action.get("type", "unknown")
        elif dry_run:
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
        seq_finalized = False

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
            seq_inputs = _seq_inputs_for_trial(
                journal=journal,
                action=action,
                tier=eval_result.tier,
                candidate_override=(
                    str(seq_fresh_eval_context.get("candidate"))
                    if seq_fresh_eval_context
                    else None
                ),
            )
            verdict = gate.check(
                eval_result,
                question_results=list(getattr(eval_result, "question_results", []) or []),
                task_rate=task_rate_qph_from(eval_result),
                baseline_profile=seq_inputs["baseline_profile"],
                baseline_task_rate=seq_inputs["baseline_task_rate"],
                prior_quality_obs=seq_inputs["prior_quality_obs"],
                prior_rate_obs=seq_inputs["prior_rate_obs"],
                candidate=seq_inputs["candidate"],
                core_id=seq_inputs["core_id"],
            )
            seq_finalized = _annotate_seq_promotion_finalization(
                verdict.seq,
                baseline_reference=seq_inputs.get("baseline_reference"),
                is_fresh_eval=seq_fresh_eval_context is not None,
                fresh_eval_context=seq_fresh_eval_context,
            )
            if verdict.seq is not None and verdict.seq.get(
                "baseline_reference_state"
            ) == "stale-reference":
                verdict.categories.append("seq_stale_reference")
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
        baseline_update = None
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
            seq_confirmed = (
                bool(verdict.seq.get("baseline_promotion_finalized"))
                if verdict.seq is not None
                else None
            )
            baseline_update = gate.update_baseline(
                eval_result,
                source_trial_id=trial_counter,
                seq_confirmed=seq_confirmed,
            )
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

        _update_seq_promotion_fresh_eval_state(
            state,
            seq=getattr(verdict, "seq", None),
            action=action,
            eval_result=eval_result,
            trial_counter=trial_counter,
            is_fresh_eval=seq_fresh_eval_context is not None,
            finalized=seq_finalized if baseline_update is not None else False,
            baseline_update=baseline_update,
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

        # Compute trial lineage (AP-3) before BSV observe so BSV-3 dependency rows
        # can carry the same parent-trial identity as the journal entry.
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

        # J11/BSV-2: observe-only behavior-signature differential vs the last frontier-accepted
        # incumbent. Flag-gated (AUTOPILOT_BSV_OBSERVE=1), default-OFF. Same observe-only zone as
        # HLE (after SafetyGate + ParetoArchive) so it CANNOT affect trial acceptance, Pareto
        # promotion, routing, or baseline mutation — it only journals diagnostics.
        bsv_payload: dict = {}
        if os.environ.get("AUTOPILOT_BSV_OBSERVE") == "1":
            try:
                from bsv_observe import (  # type: ignore
                    build_conflict_report,
                    build_mutation_dependency_entry,
                    compute_bsv_observe_payload,
                )

                incumbent_signature = state.get("bsv_incumbent_signature")
                bsv_payload = compute_bsv_observe_payload(
                    eval_result,
                    species_name=species_name,
                    trial_id=trial_counter,
                    archive_member_id=f"trial:{trial_counter}",
                    incumbent_signature=incumbent_signature,
                    incumbent_archive_member_id=state.get("bsv_incumbent_archive_member_id"),
                )
                # Incumbent = last frontier entry that ALSO PASSED the SafetyGate (verdict truthy =
                # SafetyVerdict.passed). archive.update runs even on gate-FAILED trials, so frontier
                # alone could promote a failed trial's signature as the incumbent (finding #1).
                if pareto_status == "frontier" and verdict:
                    archive_member_id = bsv_payload.get("archive_member_id")
                    signature = bsv_payload.get("signature")
                    dependency_entry = build_mutation_dependency_entry(
                        trial_id=trial_counter,
                        action=action,
                        parent_trial=parent_trial_id,
                        bsv_payload=bsv_payload,
                        incumbent_signature=incumbent_signature,
                        pareto_status=pareto_status,
                    )
                    existing_ledger = state.get("bsv_mutation_dependency_ledger", [])
                    if not isinstance(existing_ledger, list):
                        existing_ledger = []
                    conflict_report = build_conflict_report(dependency_entry, existing_ledger)
                    bsv_payload["mutation_dependency"] = dependency_entry
                    bsv_payload["conflict_report"] = conflict_report
                    if conflict_report.get("severity") in {"watch", "blocking"}:
                        log.warning(
                            "BSV-3 conflict review signal for trial %d: severity=%s conflicts=%s",
                            trial_counter,
                            conflict_report.get("severity"),
                            conflict_report.get("conflict_count"),
                        )
                    state["bsv_mutation_dependency_ledger"] = [
                        *existing_ledger[-499:],
                        dependency_entry,
                    ]
                    state["bsv_incumbent_signature"] = signature
                    state["bsv_incumbent_archive_member_id"] = archive_member_id
                    if archive_member_id and signature:
                        archive_signatures = state.setdefault("bsv_archive_signatures", {})
                        archive_signatures[archive_member_id] = {
                            "trial_id": trial_counter,
                            "signature_hash": bsv_payload.get("signature_hash"),
                            "signature_confidence": bsv_payload.get("signature_confidence"),
                            "severity_vs_previous_incumbent": bsv_payload.get("severity"),
                            "reasons": list(bsv_payload.get("reasons") or [])[:8],
                            "conflict_severity": (bsv_payload.get("conflict_report") or {}).get("severity"),
                            "conflict_count": (bsv_payload.get("conflict_report") or {}).get("conflict_count"),
                            "signature": signature,
                        }
            except Exception as _bsv_err:  # observe-only must never disrupt the trial loop
                log.debug("BSV observe skipped (trial %s): %s", trial_counter, _bsv_err)

        # Git tag
        phase.set("post_trial_artifacts", trial_id=trial_counter, species=species_name)
        git_tag = ""
        if not dry_run:
            git_tag = f"autopilot/trial-{trial_counter}"
            _git_tag(git_tag, f"Trial {trial_counter}: {species_name}/{action.get('type', '')}")

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
        legacy_objectives = list(objectives_from(eval_result))
        task_rate_objectives = list(task_rate_objectives_from(eval_result))
        eval_details_dict: dict[str, Any] = {
            "per_suite_quality": eval_result.per_suite_quality,
            "routing_distribution": eval_result.routing_distribution,
            "question_results": list(getattr(eval_result, "question_results", []) or []),
            "details": eval_result.details,
            "metric_schema_version": metric_schema_version,
            "harness_metrics": harness_metrics,
            "oracle_adequacy": oracle_adequacy,
            "bsv_observe": bsv_payload,  # J11/BSV-2 observe-only diff ({} unless AUTOPILOT_BSV_OBSERVE=1)
            "objective_policy_live": LEGACY_OBJECTIVE_POLICY,
            "objective_policy_shadow": TASK_RATE_OBJECTIVE_POLICY,
            "objectives_legacy_v1": legacy_objectives,
            "objectives_task_rate_v1": task_rate_objectives,
            "task_rate_qph": task_rate_qph_from(eval_result),
            "goodput_qph": goodput_qph_from(eval_result),
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
        if seq_baseline_draw_reference is not None:
            eval_details_dict["seq_baseline_reference_draw"] = True
            eval_details_dict["seq_baseline_reference_reason"] = (
                seq_baseline_draw_reference.get("reason", "")
            )
        if seq_fresh_eval_context is not None:
            eval_details_dict["seq_promotion_fresh_eval"] = {
                "candidate": seq_fresh_eval_context.get("candidate"),
                "source_trial_id": seq_fresh_eval_context.get("source_trial_id"),
            }
        if seq_candidate_replay_context is not None:
            eval_details_dict["seq_candidate_replay"] = {
                "candidate": seq_candidate_replay_context.get("candidate"),
                "source_trial_id": seq_candidate_replay_context.get("source_trial_id"),
                "k": seq_candidate_replay_context.get("k"),
                "combined_E": seq_candidate_replay_context.get("combined_E"),
            }

        journal_entry = JournalEntry(
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
            seq=verdict.seq or {},
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
        journal.record(journal_entry)
        if strategy_store is not None:
            try:
                strategy_store.store_frontier_journal_entry(journal_entry)
            except Exception as e:
                log.warning("Strategy store journal projection failed: %s", e)

        # AP-16: Track last instruction ratio for structural pruning comparison
        state["_last_instruction_ratio"] = eval_result.instruction_token_ratio

        # W5: rebuild AP-22 prompt memory from the folded append-only journal view
        # instead of mutating short_term_memory.md from in-memory trial state. The
        # renderer applies the same trust filters for corrupted, non-ok, and
        # learning-excluded rows, so excluded trials stay auditable in the ledger
        # without feeding planner memory.
        memory.refresh_from_journal(journal)

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

        # Generate plots on a trial-count OR wall-clock cadence, so a long
        # mid-decade gap still refreshes instead of waiting for the next %N
        # boundary (which may never arrive if the run stops first).
        if (
            trial_counter % PLOT_INTERVAL == 0
            or (time.time() - plot_clock["last_ts"]) >= PLOT_MAX_AGE_S
        ):
            _refresh_plots(sync=False, reason="periodic")
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
        baseline_state = gate.baseline.to_state_dict()
        state["baseline_state"] = baseline_state
        merged_control = _merge_external_control_fields(state)
        if merged_control:
            log.info(
                "Preserved external control state before trial-end save: %s",
                ", ".join(merged_control),
            )
        try:
            _append_baseline_promotion_event(
                journal=journal,
                baseline_update=baseline_update,
                eval_result=eval_result,
                source_trial_id=trial_counter - 1,
                pareto_status=pareto_status,
                baseline_state=baseline_state,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Baseline promotion event append failed for trial %d: %s",
                trial_counter - 1,
                exc,
            )
        _save_state_with_journal_archive_authority(
            state,
            journal,
            archive,
            context=f"trial {trial_counter} final save",
        )

        # Phase 6b — clear in_flight_trial marker AFTER final save_state.
        # This is the closing half of the WAL pattern: a crash between
        # the pre-dispatch marker write and this clear leaves the marker
        # in place, which triggers the recovery branch on the next
        # startup. By the time we reach here both the journal and the
        # Pareto archive are durable on disk, so it is safe to clear.
        state["in_flight_trial"] = None
        merged_control = _merge_external_control_fields(state)
        if merged_control:
            log.info(
                "Preserved external control state before in-flight clear: %s",
                ", ".join(merged_control),
            )
        _save_state_with_journal_archive_authority(
            state,
            journal,
            archive,
            context=f"trial {trial_counter} in-flight clear",
        )

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
    _save_state_with_journal_archive_authority(
        state,
        journal,
        archive,
        context="shutdown",
    )
    # Final synchronous plot render so the operator's last view reflects the
    # final completed trial. Covers SIGTERM/SIGINT and the max-trials break (both
    # exit the loop to here); the %N-boundary regen almost never lands on the
    # exact stopping trial. Skipped in dry_run (no real metrics produced).
    if not dry_run:
        _refresh_plots(sync=True, reason="shutdown")
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
        return {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}
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
    return {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}


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


def _journal_rows_for_archive(journal: ExperimentJournal) -> list[dict[str, Any]]:
    rows = [asdict(entry) for entry in journal.all_entries()]
    if hasattr(journal, "supersession_events"):
        rows.extend(journal.supersession_events())
    return rows


def _archive_for_read_command(
    journal: ExperimentJournal,
    *,
    source: str = ARCHIVE_SOURCE_JOURNAL_ALL,
) -> tuple[ParetoArchive, str]:
    """Archive view for operator read commands.

    Journal reconstruction is the default read path. The explicit ``state``
    source remains a legacy fallback for one release.
    """
    if source == ARCHIVE_SOURCE_STATE:
        return ParetoArchive(), ARCHIVE_SOURCE_STATE
    if source not in ARCHIVE_SOURCE_CHOICES:
        raise ValueError(f"unknown archive source: {source}")

    state = load_state()
    deinflate_before_ts, deinflate_factor, exclude_before_ts = (
        _archive_epoch_params_from_state(state)
    )
    archive = pareto_archive_from_journal_rows(
        _journal_rows_for_archive(journal),
        None,
        current_run_only=(source == ARCHIVE_SOURCE_JOURNAL_CURRENT_RUN),
        deinflate_before_ts=deinflate_before_ts,
        deinflate_factor=deinflate_factor,
        exclude_before_ts=exclude_before_ts,
    )
    if archive is None:
        return (
            _ConcreteParetoArchive.from_archive_payload({}, read_only=True),
            f"{source}->empty-fallback",
        )
    return archive, source


def _append_baseline_promotion_event(
    *,
    journal: ExperimentJournal,
    baseline_update: Any,
    eval_result: EvalResult,
    source_trial_id: int,
    pareto_status: str,
    baseline_state: dict[str, Any],
) -> dict[str, Any] | None:
    if baseline_update is None or not baseline_update.updated:
        return None
    return journal.append_baseline_promotion_event(
        source_trial_id=source_trial_id,
        tier=baseline_update.tier,
        previous_quality=baseline_update.previous_quality,
        new_quality=baseline_update.new_quality,
        reason=baseline_update.reason,
        proof=baseline_update.proof,
        result_metrics={
            "quality": eval_result.quality,
            "speed": eval_result.speed,
            "cost": eval_result.cost,
            "reliability": eval_result.reliability,
            "n_questions": eval_result.n_questions,
            "pareto_status": pareto_status,
        },
        baseline_state=baseline_state,
    )


def _baseline_promotion_summary_lines(
    state: dict[str, Any], journal: ExperimentJournal,
) -> list[str]:
    """Read-only baseline-as-ledger preview for operator commands."""
    reconciliation = reconcile_baseline_ledger(
        journal.baseline_promotion_events(),
        state.get("baseline_state"),
    )
    return format_baseline_ledger_summary(reconciliation)


def _frontier_rerun_summary_lines(
    state: dict[str, Any], journal: ExperimentJournal,
) -> list[str]:
    """Read-only frontier-rerun marker summary for operator commands."""
    marker = state.get("frontier_rerun_required")
    if not (isinstance(marker, dict) and marker.get("required")):
        return ["Frontier rerun: not required"]

    completed = _frontier_rerun_completed_numeric_trials(marker, journal)
    min_trials = _frontier_rerun_min_trials(marker)
    reason = str(marker.get("reason") or "frontier rerun required")
    lines = [
        f"Frontier rerun: required ({completed}/{min_trials} numeric trials complete)",
        f"Frontier rerun reason: {reason}",
    ]
    opened = marker.get("rerun_started_at") or marker.get("opened_at")
    if opened:
        lines.append(f"Frontier rerun opened: {opened}")

    pending = state.get("frontier_rerun_pending_clear")
    if isinstance(pending, dict) and pending.get("trial_id") is not None:
        action = pending.get("action") if isinstance(pending.get("action"), dict) else {}
        action_type = action.get("type", "unknown")
        surface = action.get("surface")
        action_label = f"{action_type}/{surface}" if surface else str(action_type)
        lines.append(
            f"Frontier rerun pending: trial #{pending.get('trial_id')} {action_label}"
        )

    blocked = state.get("frontier_rerun_blocked")
    if isinstance(blocked, dict):
        lines.append(f"Frontier rerun blocked: {blocked.get('reason', 'unknown')}")

    return lines


def cmd_status(args: argparse.Namespace) -> None:
    """Show current status."""
    state = load_state()
    journal = ExperimentJournal()
    archive, archive_source = _archive_for_read_command(
        journal,
        source=getattr(args, "archive_source", ARCHIVE_SOURCE_JOURNAL_ALL),
    )

    print("AutoPilot Status")
    print("=" * 50)
    print(f"Trial counter: {state.get('trial_counter', 0)}")
    print(f"Paused: {state.get('paused', False)}")
    print(f"Session ID: {state.get('session_id', 'none')}")
    if archive_source != ARCHIVE_SOURCE_STATE:
        print(f"Archive source: {archive_source}")
    for line in _baseline_promotion_summary_lines(state, journal):
        print(line)
    for line in _frontier_rerun_summary_lines(state, journal):
        print(line)
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
    state.pop("pause_reason", None)
    if state.get("_dispatch_deficiency") == "skip_action_loop":
        state["consecutive_skip_actions"] = 0
        state["last_invalid_action"] = None
        state["last_invalid_reason"] = None
        state["last_invalid_status"] = None
    state.pop("_dispatch_deficiency", None)
    state.pop("_meta_halt_reason", None)
    save_state(state)
    print("AutoPilot resumed")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate markdown report."""
    journal = ExperimentJournal()
    archive, archive_source = _archive_for_read_command(
        journal,
        source=getattr(args, "archive_source", ARCHIVE_SOURCE_JOURNAL_ALL),
    )

    print("# AutoPilot Optimization Report")
    print()
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    if archive_source != ARCHIVE_SOURCE_STATE:
        print(f"Archive source: {archive_source}")
    print()
    print("## Baseline Promotion Ledger")
    for line in _baseline_promotion_summary_lines(load_state(), journal):
        print(line)
    print()
    print("## Frontier Rerun")
    for line in _frontier_rerun_summary_lines(load_state(), journal):
        print(line)
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
    journal = ExperimentJournal()
    archive, archive_source = _archive_for_read_command(
        journal,
        source=getattr(args, "archive_source", ARCHIVE_SOURCE_JOURNAL_ALL),
    )
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
    if archive_source != ARCHIVE_SOURCE_STATE:
        print(f"Archive source: {archive_source}")
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
    journal = ExperimentJournal()
    archive, archive_source = _archive_for_read_command(
        journal,
        source=getattr(args, "archive_source", ARCHIVE_SOURCE_JOURNAL_ALL),
    )
    swarm = NumericSwarm(epoch_label=_numeric_swarm_epoch_label_from_state(state))
    lab = StructuralLab()
    path = generate_digest(
        swarm=swarm,
        lab=lab,
        archive=archive,
        state=state,
        journal=journal,
        archive_source=archive_source,
        output_root=Path(args.output_root) if getattr(args, "output_root", None) else None,
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
        if tier == 0:
            result = tower.eval_t0()
        elif tier == 1:
            result = tower.eval_t1(n=n or 100, seed=seed)
        elif tier == 2:
            result = tower.eval_t2(n=n or 500, seed=seed)
        else:
            raise ValueError(f"Unknown eval tier: {tier}")
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
    p_status.add_argument(
        "--archive-source",
        choices=ARCHIVE_SOURCE_CHOICES,
        default=ARCHIVE_SOURCE_JOURNAL_ALL,
        help=(
            "Archive read source for this operator command only. "
            "Defaults to journal-all; state is a legacy fallback."
        ),
    )
    p_status.set_defaults(func=cmd_status)

    # pause / resume
    p_pause = subparsers.add_parser("pause")
    p_pause.set_defaults(func=cmd_pause)
    p_resume = subparsers.add_parser("resume")
    p_resume.set_defaults(func=cmd_resume)

    # report
    p_report = subparsers.add_parser("report")
    p_report.add_argument(
        "--archive-source",
        choices=ARCHIVE_SOURCE_CHOICES,
        default=ARCHIVE_SOURCE_JOURNAL_ALL,
        help=(
            "Archive read source for this operator command only. "
            "Defaults to journal-all; state is a legacy fallback."
        ),
    )
    p_report.set_defaults(func=cmd_report)

    # plot
    p_plot = subparsers.add_parser("plot")
    p_plot.add_argument(
        "--archive-source",
        choices=ARCHIVE_SOURCE_CHOICES,
        default=ARCHIVE_SOURCE_JOURNAL_ALL,
        help=(
            "Archive read source for this operator command only. "
            "Defaults to journal-all; state is a legacy fallback."
        ),
    )
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
    p_digest.add_argument(
        "--archive-source",
        choices=ARCHIVE_SOURCE_CHOICES,
        default=ARCHIVE_SOURCE_JOURNAL_ALL,
        help=(
            "Archive read source for this operator command only. "
            "Defaults to journal-all; state is a legacy fallback."
        ),
    )
    p_digest.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Optional progress root override for this digest write. "
            "Useful for ad-hoc smoke snapshots outside the root progress tree."
        ),
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
