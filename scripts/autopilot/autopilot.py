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
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import yaml

from experiment_journal import ExperimentJournal, JournalEntry
from pareto_archive import ParetoArchive, ParetoEntry
from safety_gate import EvalResult, SafetyGate
from eval_tower import EvalTower
from config_applicator import apply_params, health_check
from meta_optimizer import MetaOptimizer, SpeciesBudget
from progress_plots import generate_all_plots
from species import Seeder, NumericSwarm, PromptForge, StructuralLab, EvolutionManager
from species.prompt_forge import CODE_MUTATION_ALLOWLIST
from short_term_memory import ShortTermMemory, TrialOutcome
from self_criticism import SelfCriticism, generate_self_criticism

# Strategy store for species memory (B1)
sys.path.insert(0, str(ORCH_ROOT))
from orchestration.repl_memory.strategy_store import StrategyStore

log = logging.getLogger("autopilot")

STATE_PATH = ORCH_ROOT / "orchestration" / "autopilot_state.json"
LOCK_PATH = ORCH_ROOT / "orchestration" / ".autopilot.lock"
BLACKLIST_PATH = SCRIPT_DIR / "failure_blacklist.yaml"
ORCHESTRATOR_URL = "http://localhost:8000"
PLOT_INTERVAL = 10  # Generate plots every N trials

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

### Experiment Journal (last 20 entries)
{journal_summary}

### Seeder Status
{seeder_status}

### Species Effectiveness
{species_effectiveness}

### System Health
- Orchestrator: {health_status}
- Memory count: {memory_count}
- Q-value converged: {converged}

### Species Budget
{budget}

### Suite Quality Trends (last 10 evals)
{suite_quality_trends}

### Recent Insights (cross-species)
{insights}

### Short-Term Memory (accumulated learnings this session)
{short_term_memory}

### Self-Criticism from Last Trial
{last_criticism}

### Blacklisted Configurations
{blacklist_text}

### Plot Paths (reference for trend analysis)
{plot_paths}

## Action Guidelines

1. If memories < 500: ALWAYS prioritize seeding (seed_batch)
2. If Q-values converged and models not trained: trigger train_routing_models
3. If models trained and not enabled: try structural_experiment with routing features
4. If stagnating (hv_slope < 0.001): try prompt_mutation or widen numeric search
5. If quality regression after changes: rollback to last good checkpoint
6. Consider the species budget allocation when choosing actions

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
- Train: {{"type": "train_routing_models", "min_memories": 500}}
- Distill: {{"type": "distill_skillbank", "teacher": "claude", "categories": ["routing"]}}
- Reset: {{"type": "reset_memories", "keep_seen": true, "keep_skills": true}}
- Deep eval: {{"type": "deep_eval", "tier": 2}}
- Rollback: {{"type": "rollback", "to_checkpoint": "production_best"}}
- Distill: {{"type": "distill_knowledge", "last_n": 10}}
  (Run every ~5 trials to extract insights from recent outcomes into strategy memory)

Include brief reasoning before the action block.
"""


# ── State Management ─────────────────────────────────────────────


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "trial_counter": 0,
        "session_id": None,
        "paused": False,
        "species_budget": SpeciesBudget().as_dict(),
        "td_errors": [],
    }


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ── Failure Blacklist (B2) ───────────────────────────────────


def load_blacklist() -> list[dict[str, Any]]:
    """Load failure blacklist from YAML."""
    if not BLACKLIST_PATH.exists():
        return []
    try:
        data = yaml.safe_load(BLACKLIST_PATH.read_text()) or {}
        return data.get("blacklist", [])
    except Exception as e:
        log.warning("Could not load blacklist: %s", e)
        return []


def check_blacklist(action: dict[str, Any], blacklist: list[dict[str, Any]]) -> str | None:
    """Check if action matches any blacklist pattern.

    Returns the reason string if blocked, None if allowed.
    """
    if not isinstance(action, dict):
        return None
    for entry in blacklist:
        pattern = entry.get("pattern", {})
        if not isinstance(pattern, dict):
            continue
        if pattern and all(action.get(k) == v for k, v in pattern.items()):
            return entry.get("reason", "blacklisted")
    return None


def append_blacklist(action: dict[str, Any], trial_id: int, reason: str) -> None:
    """Auto-append a blacklist entry after rollback trigger."""
    # Build a pattern from the action's key fields
    pattern = {}
    for key in ("type", "surface", "file", "mutation", "flags"):
        if key in action:
            pattern[key] = action[key]
    if not pattern:
        return

    entry = {
        "pattern": pattern,
        "reason": reason,
        "added": datetime.now(timezone.utc).isoformat(),
        "source_trial": trial_id,
    }

    data = {"blacklist": []}
    if BLACKLIST_PATH.exists():
        try:
            data = yaml.safe_load(BLACKLIST_PATH.read_text()) or {"blacklist": []}
        except Exception:
            pass
    data.setdefault("blacklist", []).append(entry)
    BLACKLIST_PATH.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    log.info("Blacklisted pattern: %s (reason: %s)", pattern, reason)


# ── Claude CLI Controller ────────────────────────────────────────


def invoke_controller(
    prompt: str,
    session_id: str | None = None,
    timeout: int = 300,
) -> tuple[str, str | None]:
    """Invoke Claude CLI for meta-reasoning.

    Returns (response_text, session_id).
    """
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--allowedTools", "Read,Grep,Glob",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ORCH_ROOT),
        )
        stdout, stderr = proc.communicate(timeout=timeout)

        if proc.returncode != 0:
            log.error("Controller failed (rc=%d): %s", proc.returncode, stderr[:500])
            return "", session_id

        try:
            response = json.loads(stdout)
            new_session = response.get("session_id", session_id)
            return response.get("result", stdout), new_session
        except json.JSONDecodeError:
            return stdout, session_id

    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("Controller timed out after %ds", timeout)
        return "", session_id
    except FileNotFoundError:
        log.error("Claude CLI not found")
        return "", session_id


def _unwrap_action(data: Any) -> dict[str, Any] | None:
    """Unwrap action from list or validate dict."""
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    if isinstance(data, dict) and "type" in data:
        return data
    return None


def extract_action(text: str) -> dict[str, Any] | None:
    """Extract structured action from controller response."""
    marker = "```json:autopilot_actions"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.index("```", start)
        try:
            data = json.loads(text[start:end].strip())
            return _unwrap_action(data)
        except json.JSONDecodeError as e:
            log.error("Failed to parse action JSON: %s", e)
            return None

    # Fallback: look for any JSON block
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        try:
            data = json.loads(text[start:end].strip())
            if isinstance(data, dict) and "type" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ── Action Dispatch ──────────────────────────────────────────────


def _validate_single_variable(action: dict[str, Any]) -> str | None:
    """AP-9: Validate that an action proposes a single-variable change.

    Returns an error message if the action violates the single-variable
    constraint, or None if it passes.
    """
    action_type = action.get("type", "")

    if action_type in ("prompt_mutation", "gepa_optimize"):
        # Must target exactly one file
        target = action.get("file", "")
        if not target:
            return f"{action_type} must specify a single target file"
        if "," in target or ";" in target:
            return f"{action_type} targets multiple files: {target}"

    elif action_type == "code_mutation":
        target = action.get("file", "")
        if not target:
            return "code_mutation must specify a single target file"

    elif action_type == "structural_experiment":
        flags = action.get("flags", {})
        if len(flags) > 1:
            return (
                f"structural_experiment changes {len(flags)} flags at once "
                f"({list(flags.keys())}); limit to 1 for clean attribution"
            )

    elif action_type == "numeric_trial":
        params = action.get("params", {})
        # Optuna-suggested params are fine (controlled search), but explicit
        # multi-param overrides violate single-variable principle
        if len(params) > 1:
            return (
                f"numeric_trial sets {len(params)} params explicitly; "
                "limit to 1 for clean attribution (Optuna suggestions exempt)"
            )

    return None


def dispatch_action(
    action: dict[str, Any],
    seeder: Seeder,
    swarm: NumericSwarm,
    forge: PromptForge,
    lab: StructuralLab,
    tower: EvalTower,
    gate: SafetyGate,
    archive: ParetoArchive,
    journal: ExperimentJournal,
    state: dict[str, Any],
    strategy_store: StrategyStore | None = None,
    evo: EvolutionManager | None = None,
) -> tuple[EvalResult | None, str]:
    """Execute an action and return (eval_result, species_name)."""
    action_type = action.get("type", "")

    # AP-9: Single-variable scope enforcement
    scope_err = _validate_single_variable(action)
    if scope_err:
        log.warning("AP-9 scope violation: %s — skipping trial", scope_err)
        return None, action_type
    log.info("Dispatching action: %s", action_type)

    if action_type == "seed_batch":
        n = action.get("n_questions", 10)
        suites = action.get("suites")
        result = seeder.run_batch(n_questions=n, suites=suites)
        # After seeding, run T0 eval
        eval_result = tower.hybrid_eval()
        return eval_result, "seeder"

    elif action_type == "numeric_trial":
        surface = action.get("surface", "memrl_retrieval")
        explicit_params = action.get("params", {})

        if explicit_params:
            # Apply explicit params
            apply_params(explicit_params)
        else:
            # Let Optuna suggest
            trial = swarm.suggest_trial(surface)
            apply_params(trial["params"])
            state["_current_optuna_trial"] = {
                "surface": surface,
                "trial_number": trial["trial_number"],
            }

        eval_result = tower.hybrid_eval()
        # Report to Optuna if we have a trial
        if "_current_optuna_trial" in state and eval_result:
            t = state.pop("_current_optuna_trial")
            swarm.report_result(
                t["surface"], t["trial_number"], eval_result.objectives
            )
        return eval_result, "numeric_swarm"

    elif action_type == "prompt_mutation":
        target = action.get("file", "frontdoor.md")
        mutation_type = action.get("mutation", "targeted_fix")
        description = action.get("description", "")

        # Gather failure context from recent journal entries (AP-1)
        recent_failures = journal.recent_failures(species="prompt_forge", n=5)
        failure_context = "\n\n".join(
            f"Trial #{f.trial_id} ({f.action_type}):\n{f.failure_analysis}"
            for f in recent_failures
        )

        # B5: Cross-species fertilization — prepend insights from all species
        cross_insights = journal.insights_text(n=5)
        if cross_insights and cross_insights != "(no insights yet)":
            failure_context = (
                f"## Cross-Species Insights\n{cross_insights}\n\n"
                + failure_context
            )

        # B1: Strategy store retrieval — add past strategy insights
        if strategy_store is not None:
            query = f"{target} {mutation_type} {description}"
            strategies = strategy_store.retrieve(query, k=3)
            if strategies:
                strategy_lines = "\n".join(
                    f"- Trial #{s.source_trial_id} ({s.species}): {s.description} → {s.insight}"
                    for s in strategies
                )
                failure_context = (
                    f"## Past Strategy Insights\n{strategy_lines}\n\n"
                    + failure_context
                )

        # B3: Execution trace feedback — add recent inference traces
        last_traces = state.get("last_traces", "")
        if last_traces:
            failure_context = (
                f"## Recent Execution Traces\n{last_traces}\n\n"
                + failure_context
            )

        # Get per-suite quality from most recent eval
        last_entries = journal.recent(1)
        last_per_suite = (
            last_entries[-1].eval_details.get("per_suite_quality")
            if last_entries else None
        )

        mutation = forge.propose_mutation(
            target_file=target,
            mutation_type=mutation_type,
            failure_context=failure_context,
            per_suite_quality=last_per_suite,
            description=description,
        )
        forge.apply_mutation(mutation)
        eval_result = tower.hybrid_eval()

        # Revert if quality drops
        verdict = gate.check(eval_result)
        if not verdict:
            log.warning("Prompt mutation failed safety gate, reverting")
            forge.revert_mutation(mutation)
            return eval_result, "prompt_forge"

        # AP-10: Simplicity criterion — reject mutations that add
        # disproportionate complexity for marginal quality gain.
        orig_len = len(mutation.original_content)
        new_len = len(mutation.mutated_content)
        if orig_len > 0:
            size_increase = (new_len - orig_len) / orig_len
            # Compare quality against last known quality from journal
            last_quality = 0.0
            recent = journal.recent(1)
            if recent:
                last_quality = recent[-1].quality
            quality_delta = eval_result.quality - last_quality
            if size_increase > 0.20 and quality_delta < 0.02:
                log.warning(
                    "Simplicity criterion: prompt grew %.0f%% for %.3f quality gain, reverting",
                    size_increase * 100,
                    quality_delta,
                )
                forge.revert_mutation(mutation)
                return eval_result, "prompt_forge"
            # Block catastrophic shrinkage (>50% reduction) — likely destructive
            if size_increase < -0.50:
                log.warning(
                    "Simplicity criterion: prompt shrank %.0f%% — likely destructive, reverting",
                    abs(size_increase) * 100,
                )
                forge.revert_mutation(mutation)
                state["_dispatch_deficiency"] = "shrinkage"  # AP-14
                return eval_result, "prompt_forge"

        # AP-7: Prompt change accepted — invalidate stale Optuna trials
        swarm.mark_epoch(f"prompt_mutation:{target}/{mutation_type}")
        return eval_result, "prompt_forge"

    elif action_type == "gepa_optimize":
        # AP-19: GEPA evolutionary prompt optimization
        target = action.get("file", "frontdoor.md")
        max_evals = action.get("max_evals", 50)
        description = action.get("description", f"GEPA optimize {target}")

        log.info("GEPA optimize: %s (max_evals=%d)", target, max_evals)

        mutation = forge.propose_mutation(
            target_file=target,
            mutation_type="gepa",
            description=description,
            eval_tower=tower,
            gepa_max_evals=max_evals,
        )

        # No-op mutation means GEPA failed
        if mutation.original_content == mutation.mutated_content:
            log.warning("GEPA produced no mutation for %s", target)
            eval_result = tower.hybrid_eval()
            return eval_result, "prompt_forge"

        forge.apply_mutation(mutation)
        eval_result = tower.hybrid_eval()

        # Safety gate check (same as prompt_mutation)
        verdict = gate.check(eval_result)
        if not verdict:
            log.warning("GEPA mutation failed safety gate, reverting")
            forge.revert_mutation(mutation)
            return eval_result, "prompt_forge"

        # Simplicity criterion (AP-10)
        orig_len = len(mutation.original_content)
        new_len = len(mutation.mutated_content)
        if orig_len > 0:
            size_increase = (new_len - orig_len) / orig_len
            last_quality = 0.0
            recent = journal.recent(1)
            if recent:
                last_quality = recent[-1].quality
            quality_delta = eval_result.quality - last_quality
            if size_increase > 0.20 and quality_delta < 0.02:
                log.warning(
                    "GEPA simplicity criterion: prompt grew %.0f%% for %.3f gain, reverting",
                    size_increase * 100, quality_delta,
                )
                forge.revert_mutation(mutation)
                return eval_result, "prompt_forge"
            if size_increase < -0.50:
                log.warning(
                    "GEPA simplicity criterion: prompt shrank %.0f%%, reverting",
                    abs(size_increase) * 100,
                )
                forge.revert_mutation(mutation)
                state["_dispatch_deficiency"] = "shrinkage"
                return eval_result, "prompt_forge"

        swarm.mark_epoch(f"gepa_optimize:{target}")
        return eval_result, "prompt_forge"

    elif action_type == "code_mutation":
        # Meta-Harness Tier 2: Python code mutation
        target = action.get("file", "")
        mutation_type = action.get("mutation", "targeted_fix")
        description = action.get("description", "")

        # Gather context (same as prompt_mutation)
        recent_failures = journal.recent_failures(species="prompt_forge", n=5)
        failure_context = "\n\n".join(
            f"Trial #{f.trial_id} ({f.action_type}):\n{f.failure_analysis}"
            for f in recent_failures
        )
        cross_insights = journal.insights_text(n=5)
        if cross_insights and cross_insights != "(no insights yet)":
            failure_context = f"## Cross-Species Insights\n{cross_insights}\n\n" + failure_context
        last_traces = state.get("last_traces", "")
        if last_traces:
            failure_context = f"## Recent Execution Traces\n{last_traces}\n\n" + failure_context

        last_entries = journal.recent(1)
        last_per_suite = (
            last_entries[-1].eval_details.get("per_suite_quality")
            if last_entries else None
        )

        try:
            mutation = forge.propose_code_mutation(
                target_file=target,
                mutation_type=mutation_type,
                failure_context=failure_context,
                per_suite_quality=last_per_suite,
                description=description,
            )
        except (ValueError, FileNotFoundError) as e:
            log.error("Code mutation blocked: %s", e)
            return None, "prompt_forge"

        if not mutation.syntax_valid:
            log.warning("Code mutation failed syntax validation, skipping")
            return None, "prompt_forge"

        forge.apply_code_mutation(mutation)
        eval_result = tower.hybrid_eval()

        verdict = gate.check(eval_result)
        if not verdict:
            log.warning("Code mutation failed safety gate, reverting")
            forge.revert_code_mutation(mutation)
            return eval_result, "prompt_forge"

        # Simplicity criterion for code — block both excessive growth AND catastrophic shrinkage
        orig_len = len(mutation.original_content)
        new_len = len(mutation.mutated_content)
        if orig_len > 0:
            size_change = (new_len - orig_len) / orig_len
            last_quality = 0.0
            recent = journal.recent(1)
            if recent:
                last_quality = recent[-1].quality
            quality_delta = eval_result.quality - last_quality
            # Block excessive growth with insufficient quality gain
            if size_change > 0.20 and quality_delta < 0.02:
                log.warning(
                    "Simplicity criterion: code grew %.0f%% for %.3f quality gain, reverting",
                    size_change * 100, quality_delta,
                )
                forge.revert_code_mutation(mutation)
                return eval_result, "prompt_forge"
            # Block catastrophic shrinkage (>50% reduction) — likely destructive
            if size_change < -0.50:
                log.warning(
                    "Simplicity criterion: code shrank %.0f%% — likely destructive, reverting",
                    abs(size_change) * 100,
                )
                forge.revert_code_mutation(mutation)
                state["_dispatch_deficiency"] = "shrinkage"  # AP-14
                return eval_result, "prompt_forge"

        swarm.mark_epoch(f"code_mutation:{target}/{mutation_type}")
        return eval_result, "prompt_forge"

    elif action_type == "structural_experiment":
        flags = action.get("flags", {})
        validation = lab.propose_flag_experiment(flags)
        if validation.get("status") != "valid":
            log.warning("Invalid flag experiment: %s", validation)
            return None, "structural_lab"

        lab.apply_flag_experiment(flags)
        eval_result = tower.hybrid_eval()

        # Revert if quality drops
        verdict = gate.check(eval_result)
        if not verdict:
            log.warning("Structural experiment failed safety gate, reverting")
            # Revert flags
            reverted = {k: not v for k, v in flags.items()}
            lab.apply_flag_experiment(reverted)
        else:
            # AP-7: Structural change accepted — invalidate stale Optuna trials
            swarm.mark_epoch(f"structural_experiment:{flags}")

        return eval_result, "structural_lab"

    elif action_type == "structural_prune":
        # AP-17: Block-level deletion from .md prompt files
        target = action.get("file", "")
        block_id = action.get("block", "")

        if not target or not block_id:
            log.warning("structural_prune requires 'file' and 'block'")
            return None, "structural_lab"

        # Only allow pruning .md files in prompts directory
        prompts_dir = Path(__file__).resolve().parents[2] / "orchestration" / "prompts"
        target_path = prompts_dir / target
        if not target_path.exists() or not target.endswith(".md"):
            log.warning("Prune target not found or not .md: %s", target_path)
            return None, "structural_lab"

        original_content = target_path.read_text()
        pruned_content = lab.prune_block(original_content, block_id)
        if pruned_content is None or pruned_content == original_content:
            log.warning("Block '%s' not found in %s", block_id, target)
            return None, "structural_lab"

        # Save deleted block in action for journal rollback
        deleted_lines = original_content.split("\n")
        pruned_lines = pruned_content.split("\n")
        action["_deleted_block"] = "\n".join(
            line for line in deleted_lines if line not in pruned_lines
        )

        # Apply pruning
        target_path.write_text(pruned_content)
        pre_ratio = state.get("_last_instruction_ratio", 0.0)

        eval_result = tower.hybrid_eval()

        # Acceptance: safety gate passes AND instruction_token_ratio decreased
        verdict_result = gate.check(eval_result)
        ratio_decreased = eval_result.instruction_token_ratio < pre_ratio

        if not verdict_result or not ratio_decreased:
            reasons = []
            if not verdict_result:
                reasons.append(f"safety gate: {verdict_result.violations}")
            if not ratio_decreased:
                reasons.append(
                    f"ratio not decreased: {eval_result.instruction_token_ratio:.4f} "
                    f">= {pre_ratio:.4f}"
                )
            log.warning("Structural prune rejected: %s", "; ".join(reasons))
            target_path.write_text(original_content)
            return eval_result, "structural_lab"

        # Accepted — invalidate stale Optuna trials
        swarm.mark_epoch(f"structural_prune:{target}/{block_id}")
        return eval_result, "structural_lab"

    elif action_type == "train_routing_models":
        min_mem = action.get("min_memories", 500)
        lab.checkpoint_state(
            trial_id=state.get("trial_counter", 0),
            notes="Pre-training checkpoint",
        )
        result = lab.train_routing_models(min_memories=min_mem)
        log.info("Training result: %s", result)
        eval_result = tower.hybrid_eval()
        return eval_result, "structural_lab"

    elif action_type == "distill_skillbank":
        teacher = action.get("teacher", "claude")
        categories = action.get("categories", ["routing"])
        lab.checkpoint_state(
            trial_id=state.get("trial_counter", 0),
            notes="Pre-distillation checkpoint",
        )
        result = lab.distill_skillbank(teacher=teacher, categories=categories)
        log.info("Distillation result: %s", result)
        eval_result = tower.hybrid_eval()
        return eval_result, "structural_lab"

    elif action_type == "reset_memories":
        keep_seen = action.get("keep_seen", True)
        keep_skills = action.get("keep_skills", True)
        result = lab.reset_and_reseed(
            keep_seen=keep_seen,
            keep_skills=keep_skills,
            trial_id=state.get("trial_counter", 0),
        )
        log.info("Reset result: %s", result)
        return None, "structural_lab"

    elif action_type == "deep_eval":
        tier = action.get("tier", 2)
        eval_result = tower.evaluate(tier=tier)
        return eval_result, "seeder"

    elif action_type == "rollback":
        to_cp = action.get("to_checkpoint", "production_best")
        if to_cp == "production_best":
            lab.restore_checkpoint()
        else:
            lab.restore_checkpoint(Path(to_cp))
        gate.reset_failures()
        eval_result = tower.hybrid_eval()
        return eval_result, "structural_lab"

    elif action_type == "distill_knowledge":
        # Evolution Manager: knowledge distillation (no eval, no system change)
        last_n = action.get("last_n", 10)
        if evo is not None and strategy_store is not None:
            result = evo.distill(
                journal_entries=journal.all_entries(),
                strategy_store=strategy_store,
                last_n=last_n,
                trial_id=state.get("trial_counter", 0),
            )
            log.info("Knowledge distillation: %s", result)
        else:
            log.warning("distill_knowledge requires evo + strategy_store")
        return None, "evolution_manager"

    else:
        log.warning("Unknown action type: %s", action_type)
        return None, "unknown"


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
    tui_ctx = None
    if use_tui:
        try:
            from autopilot_tui import AutoPilotTUI
            tui = AutoPilotTUI()
            tui_ctx = tui.__enter__()
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
    gate = SafetyGate(
        consecutive_failures=state.get("consecutive_failures", 0),
    )
    tower = EvalTower(
        url=ORCHESTRATOR_URL,
        on_question=tui.set_prompt if tui is not None else None,
    )
    meta = MetaOptimizer()

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

    # Load TD errors from state
    if "td_errors" in state:
        seeder._td_errors = [(i, e) for i, e in enumerate(state["td_errors"])]

    trial_counter = state.get("trial_counter", 0)
    plot_paths: list[str] = []

    # Graceful shutdown handler
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        log.info("Shutdown requested (signal %d)", signum)
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log.info("AutoPilot starting (trial=%d, dry_run=%s)", trial_counter, dry_run)

    while not shutdown_requested:
        if max_trials and trial_counter >= max_trials:
            log.info("Max trials reached (%d)", max_trials)
            break

        if state.get("paused"):
            log.info("AutoPilot paused, waiting...")
            time.sleep(10)
            state = load_state()
            continue

        # Check orchestrator health
        _health = health_check(ORCHESTRATOR_URL, retries=2)
        if not dry_run and not _health:
            log.error(
                "Orchestrator unhealthy [%s]: %s — waiting 30s...",
                _health.failure_reason, _health.failure_detail,
            )
            time.sleep(30)
            continue

        # ── 1. Observe ───────────────────────────────────────────
        memory_count = seeder.get_memory_count() if not dry_run else 0
        converged = seeder.is_converged
        hv_slope = archive.hypervolume_slope(50)

        # ── 2. Reason ────────────────────────────────────────────
        if use_controller:
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

            prompt = CONTROLLER_PROMPT_TEMPLATE.format(
                program=program_text,
                pareto_summary=archive.summary_text(),
                journal_summary=journal.summary_text(20),
                seeder_status=json.dumps(seeder.convergence_status(), indent=2),
                species_effectiveness=json.dumps(
                    journal.species_effectiveness(), indent=2
                ),
                health_status="OK" if not dry_run else "dry_run",
                memory_count=memory_count,
                converged=converged,
                budget=json.dumps(meta.budget.as_dict(), indent=2),
                suite_quality_trends=_format_suite_trends(journal.suite_quality_trend(10)),
                insights=insights_text,
                short_term_memory=memory.to_text(),  # AP-22
                last_criticism=last_criticism_text,  # AP-23
                blacklist_text=blacklist_text,
                code_targets=", ".join(CODE_MUTATION_ALLOWLIST),
                plot_paths="\n".join(f"  - {p}" for p in plot_paths) or "  (none yet)",
            )

            response, session_id = invoke_controller(
                prompt, state.get("session_id")
            )
            state["session_id"] = session_id
            action = extract_action(response)
        else:
            # Autonomous mode: species selection by budget
            species = meta.select_species()
            action = _auto_action(species, memory_count, converged, seeder)

        if not action:
            log.warning("No action proposed, defaulting to seed_batch")
            action = {"type": "seed_batch", "n_questions": 10}

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

        # Update TUI with trial info
        if tui is not None:
            species_hint = action.get("type", "unknown")
            tui.set_trial(trial_counter, species_hint)
            # Show the action description as the "current prompt" in TUI
            prompt_preview = action.get("description", "")
            if not prompt_preview:
                prompt_preview = json.dumps(action, indent=2)[:500]
            tui.set_prompt(prompt_preview)

        if dry_run:
            eval_result = EvalResult(
                tier=0, quality=2.5, speed=15.0, cost=0.3, reliability=0.95
            )
            species_name = action.get("type", "unknown").split("_")[0]
        else:
            eval_result, species_name = dispatch_action(
                action, seeder, swarm, forge, lab, tower, gate, archive,
                journal, state, strategy_store=strategy_store, evo=evo,
            )

        # ── 4. Evaluate ─────────────────────────────────────────
        if eval_result is None:
            trial_counter += 1
            state["trial_counter"] = trial_counter
            save_state(state)
            continue

        # AP-13: Emit grep-parseable metrics
        log.info("\n%s", eval_result.to_grep_lines(trial_counter, species_name))

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
        # Get baseline and previous per-suite for comparison
        baseline_q = gate.baseline.quality if gate.baseline else 0.0
        prev_suite = {}
        recent = journal.by_species(species_name)
        if recent:
            prev_details = recent[-1].eval_details
            if isinstance(prev_details, dict):
                prev_suite = prev_details.get("per_suite_quality", {})

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
        pareto_status = archive.update(
            ParetoEntry(
                trial_id=trial_counter,
                objectives=eval_result.objectives,
                config_snapshot=action,
                species=species_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                eval_tier=eval_result.tier,
                memory_count=memory_count,
                reasoning=json.dumps(action),
            )
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
        if not dry_run:
            state["last_traces"] = tower.capture_recent_traces(50)

        # Git tag
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
        expected_mechanism = action.get("mutation", "") or action.get("surface", "") or action.get("type", "")

        # AP-14: Extract deficiency category from safety verdict + dispatch side channel
        deficiency_category = ""
        if not verdict.passed:
            deficiency_category = verdict.categories[0] if verdict.categories else ""
        if not deficiency_category:
            deficiency_category = state.pop("_dispatch_deficiency", "")

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
                eval_details={
                    "per_suite_quality": eval_result.per_suite_quality,
                    "routing_distribution": eval_result.routing_distribution,
                    "details": eval_result.details,
                },
                reasoning=json.dumps(action),
                hypothesis=hypothesis,
                expected_mechanism=expected_mechanism,
                deficiency_category=deficiency_category,
                instruction_token_count=eval_result.instruction_token_count,
                instruction_token_ratio=eval_result.instruction_token_ratio,
                self_criticism=criticism.as_text(),  # AP-23
                keep_revert_decision=criticism.keep_or_revert,  # AP-24
                optimization_directions=criticism.directions_text(),  # AP-24
            )
        )

        # AP-16: Track last instruction ratio for structural pruning comparison
        state["_last_instruction_ratio"] = eval_result.instruction_token_ratio

        # AP-22: Update short-term memory with trial outcome
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
            lab.checkpoint_state(
                trial_id=trial_counter,
                notes=f"Auto-checkpoint at trial {trial_counter}",
            )

        # Generate plots periodically
        if trial_counter % PLOT_INTERVAL == 0:
            td_errors = seeder.td_errors
            state["td_errors"] = [e for _, e in td_errors]
            paths = generate_all_plots(archive, journal, td_errors)
            plot_paths = [str(p) for p in paths]

        # Save state
        trial_counter += 1
        state["trial_counter"] = trial_counter
        state["consecutive_failures"] = gate.consecutive_failures
        archive.save(state)
        save_state(state)

        log.info(
            "Trial %d complete: q=%.3f s=%.1f → %s (HV=%.4f)",
            trial_counter - 1,
            eval_result.quality,
            eval_result.speed,
            pareto_status,
            archive.hypervolume(),
        )

    # Shutdown: checkpoint + save
    log.info("AutoPilot shutting down (trial=%d)", trial_counter)
    archive.save(state)
    save_state(state)
    if strategy_store is not None:
        strategy_store.close()
    if not dry_run:
        lab.checkpoint_state(trial_id=trial_counter, notes="Shutdown checkpoint")


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
        # AP-19: 30% chance of GEPA evolutionary optimization, 70% LLM mutation
        import random
        if random.random() < 0.30:
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
        pass


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
    print(archive.summary_text())
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
    print(archive.summary_text())
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
    paths = generate_all_plots(archive, journal, td_errors)
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


# ── Entry Point ──────────────────────────────────────────────────


def main() -> None:
    # force=True overrides any basicConfig already called at import time
    # (seed_specialist_routing.py calls basicConfig at module level)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                ORCH_ROOT / "logs" / "autopilot.log", mode="a"
            ),
        ],
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
    with AutoPilotTUI() as tui:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
