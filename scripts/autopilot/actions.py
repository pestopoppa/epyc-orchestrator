"""Autopilot action handlers and dispatcher.

Extracted from autopilot.py during the 2026-05-22 Tranche-5 refactor. Each
action type from the controller's response maps to one `_action_<type>`
handler; `dispatch_action()` is the public facade that:
  1. validates single-variable scope (AP-9)
  2. routes to the matching handler

Behavior is preserved verbatim — no semantic changes to any action. The
handler signatures pack the autopilot's many species/state objects into an
`_ActionContext` bundle so each handler stays readable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from controller_io import validate_single_variable
from safety_gate import EvalResult, SafetyGate


def _apply_params(*args, **kwargs):
    """Call apply_params via the autopilot module so tests' monkeypatches stick.

    Tests historically monkeypatch `autopilot.apply_params`; importing the
    function directly here would bypass that. Lazy lookup through sys.modules
    avoids a circular import (actions.py is imported by autopilot.py at the
    bottom of autopilot's imports, by which time `apply_params` is bound).
    """
    import sys
    # autopilot is imported as either 'autopilot' (when scripts/autopilot is on
    # sys.path — its normal load mode) or 'scripts.autopilot.autopilot' (when
    # tests import via the package path).
    mod = sys.modules.get("autopilot") or sys.modules.get("scripts.autopilot.autopilot")
    if mod is None:
        # Fallback: import config_applicator directly (no monkeypatch in play)
        from config_applicator import apply_params as _ap
        return _ap(*args, **kwargs)
    return mod.apply_params(*args, **kwargs)

if TYPE_CHECKING:
    from experiment_journal import ExperimentJournal
    from pareto_archive import ParetoArchive
    from eval_tower import EvalTower
    from species import (
        Seeder,
        NumericSwarm,
        PromptForge,
        StructuralLab,
        EvolutionManager,
    )
    from orchestration.repl_memory.strategy_store import StrategyStore

log = logging.getLogger("autopilot")


@dataclass
class _ActionContext:
    """Dependency bundle passed to each action handler."""
    seeder: "Seeder"
    swarm: "NumericSwarm"
    forge: "PromptForge"
    lab: "StructuralLab"
    tower: "EvalTower"
    gate: SafetyGate
    archive: "ParetoArchive"
    journal: "ExperimentJournal"
    state: dict[str, Any]
    strategy_store: "StrategyStore | None" = None
    evo: "EvolutionManager | None" = None


# -----------------------------------------------------------------------------
# Per-action handlers — one per action_type
# -----------------------------------------------------------------------------


def _action_seed_batch(action: dict[str, Any], ctx: _ActionContext):
    n = action.get("n_questions", 10)
    suites = action.get("suites")
    ctx.seeder.run_batch(n_questions=n, suites=suites)
    # After seeding, run T0 eval
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "seeder"


def _action_numeric_trial(action: dict[str, Any], ctx: _ActionContext):
    surface = action.get("surface", "memrl_retrieval")
    explicit_params = action.get("params", {})

    if explicit_params:
        # Apply explicit params
        apply_result = _apply_params(explicit_params)
        if apply_result.get("status") == "error":
            log.warning(
                "Skipping numeric trial eval; explicit params were not applied: %s",
                apply_result.get("errors") or apply_result,
            )
            return None, "numeric_swarm"
    else:
        # Let Optuna suggest
        trial = ctx.swarm.suggest_trial(surface)
        apply_result = _apply_params(trial["params"])
        if apply_result.get("status") == "error":
            reason = "; ".join(apply_result.get("errors", [])) or str(apply_result)
            ctx.swarm.mark_failed(surface, trial["trial_number"], reason)
            log.warning(
                "Skipping numeric trial eval; suggested params were not applied: %s",
                reason,
            )
            return None, "numeric_swarm"
        ctx.state["_current_optuna_trial"] = {
            "surface": surface,
            "trial_number": trial["trial_number"],
        }

    eval_result = ctx.tower.hybrid_eval()
    # Report to Optuna if we have a trial
    if "_current_optuna_trial" in ctx.state and eval_result:
        t = ctx.state.pop("_current_optuna_trial")
        ctx.swarm.report_result(
            t["surface"], t["trial_number"], eval_result.objectives
        )
    return eval_result, "numeric_swarm"


def _build_mutation_context(
    action: dict[str, Any], ctx: _ActionContext,
) -> tuple[str, dict | None]:
    """Shared failure-context + per-suite-quality assembly used by mutation handlers."""
    target = action.get("file", "")
    mutation_type = action.get("mutation", "targeted_fix")
    description = action.get("description", "")

    # Gather failure context from recent journal entries (AP-1)
    recent_failures = ctx.journal.recent_failures(species="prompt_forge", n=5)
    failure_context = "\n\n".join(
        f"Trial #{f.trial_id} ({f.action_type}):\n{f.failure_analysis}"
        for f in recent_failures
    )

    # B5: Cross-species fertilization — prepend insights from all species
    cross_insights = ctx.journal.insights_text(n=5)
    if cross_insights and cross_insights != "(no insights yet)":
        failure_context = (
            f"## Cross-Species Insights\n{cross_insights}\n\n"
            + failure_context
        )

    # B1: Strategy store retrieval — add past strategy insights
    if ctx.strategy_store is not None:
        query = f"{target} {mutation_type} {description}"
        strategies = ctx.strategy_store.retrieve(query, k=3)
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
    last_traces = ctx.state.get("last_traces", "")
    if last_traces:
        failure_context = (
            f"## Recent Execution Traces\n{last_traces}\n\n"
            + failure_context
        )

    # Get per-suite quality from most recent eval
    last_entries = ctx.journal.recent(1)
    last_per_suite = (
        last_entries[-1].eval_details.get("per_suite_quality")
        if last_entries else None
    )

    return failure_context, last_per_suite


def _simplicity_check(
    mutation, eval_result, ctx: _ActionContext, *, kind: str, log_label: str,
) -> tuple[bool, str | None]:
    """AP-10 simplicity criterion. Returns (passed, deficiency_marker).

    `passed=False` means the caller should revert. `deficiency_marker` is set
    to "shrinkage" when catastrophic shrinkage is detected, else None.
    """
    orig_len = len(mutation.original_content)
    new_len = len(mutation.mutated_content)
    if orig_len <= 0:
        return True, None
    size_change = (new_len - orig_len) / orig_len
    last_quality = 0.0
    recent = ctx.journal.recent(1)
    if recent:
        last_quality = recent[-1].quality
    quality_delta = eval_result.quality - last_quality
    if size_change > 0.20 and quality_delta < 0.02:
        log.warning(
            "%s simplicity criterion: %s grew %.0f%% for %.3f quality gain, reverting",
            log_label, kind, size_change * 100, quality_delta,
        )
        return False, None
    if size_change < -0.50:
        log.warning(
            "%s simplicity criterion: %s shrank %.0f%% — likely destructive, reverting",
            log_label, kind, abs(size_change) * 100,
        )
        return False, "shrinkage"
    return True, None


def _action_prompt_mutation(action: dict[str, Any], ctx: _ActionContext):
    target = action.get("file", "frontdoor.md")
    mutation_type = action.get("mutation", "targeted_fix")
    description = action.get("description", "")
    failure_context, last_per_suite = _build_mutation_context(action, ctx)

    try:
        mutation = ctx.forge.propose_mutation(
            target_file=target,
            mutation_type=mutation_type,
            failure_context=failure_context,
            per_suite_quality=last_per_suite,
            description=description,
        )
    except FileNotFoundError:
        log.warning("Prompt file not found: %s (may have been removed in refactoring)", target)
        return None, "prompt_forge"
    ctx.forge.apply_mutation(mutation)
    eval_result = ctx.tower.hybrid_eval()

    # Revert if quality drops
    verdict = ctx.gate.check(eval_result)
    if not verdict:
        log.warning("Prompt mutation failed safety gate, reverting")
        ctx.forge.revert_mutation(mutation)
        return eval_result, "prompt_forge"

    # AP-10 simplicity criterion
    passed, deficiency = _simplicity_check(
        mutation, eval_result, ctx, kind="prompt", log_label="Simplicity criterion:",
    )
    if not passed:
        ctx.forge.revert_mutation(mutation)
        if deficiency == "shrinkage":
            ctx.state["_dispatch_deficiency"] = "shrinkage"  # AP-14
        return eval_result, "prompt_forge"

    # AP-7: Prompt change accepted — invalidate stale Optuna trials
    ctx.swarm.mark_epoch(f"prompt_mutation:{target}/{mutation_type}")
    return eval_result, "prompt_forge"


def _action_gepa_optimize(action: dict[str, Any], ctx: _ActionContext):
    # AP-19: GEPA evolutionary prompt optimization
    target = action.get("file", "frontdoor.md")
    max_evals = action.get("max_evals", 50)
    description = action.get("description", f"GEPA optimize {target}")

    log.info("GEPA optimize: %s (max_evals=%d)", target, max_evals)

    try:
        mutation = ctx.forge.propose_mutation(
            target_file=target,
            mutation_type="gepa",
            description=description,
            eval_tower=ctx.tower,
            gepa_max_evals=max_evals,
        )
    except FileNotFoundError:
        log.warning("Prompt file not found: %s", target)
        return None, "prompt_forge"

    # No-op mutation means GEPA failed
    if mutation.original_content == mutation.mutated_content:
        log.warning("GEPA produced no mutation for %s", target)
        eval_result = ctx.tower.hybrid_eval()
        return eval_result, "prompt_forge"

    ctx.forge.apply_mutation(mutation)
    eval_result = ctx.tower.hybrid_eval()

    # Safety gate check
    verdict = ctx.gate.check(eval_result)
    if not verdict:
        log.warning("GEPA mutation failed safety gate, reverting")
        ctx.forge.revert_mutation(mutation)
        return eval_result, "prompt_forge"

    # AP-10 simplicity criterion
    passed, deficiency = _simplicity_check(
        mutation, eval_result, ctx, kind="prompt", log_label="GEPA",
    )
    if not passed:
        ctx.forge.revert_mutation(mutation)
        if deficiency == "shrinkage":
            ctx.state["_dispatch_deficiency"] = "shrinkage"
        return eval_result, "prompt_forge"

    ctx.swarm.mark_epoch(f"gepa_optimize:{target}")
    return eval_result, "prompt_forge"


def _action_code_mutation(action: dict[str, Any], ctx: _ActionContext):
    # Meta-Harness Tier 2: Python code mutation
    target = action.get("file", "")
    mutation_type = action.get("mutation", "targeted_fix")
    description = action.get("description", "")
    failure_context, last_per_suite = _build_mutation_context(action, ctx)

    try:
        mutation = ctx.forge.propose_code_mutation(
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

    ctx.forge.apply_code_mutation(mutation)
    eval_result = ctx.tower.hybrid_eval()

    verdict = ctx.gate.check(eval_result)
    if not verdict:
        log.warning("Code mutation failed safety gate, reverting")
        ctx.forge.revert_code_mutation(mutation)
        return eval_result, "prompt_forge"

    # AP-10 simplicity check (for code)
    passed, deficiency = _simplicity_check(
        mutation, eval_result, ctx, kind="code", log_label="Simplicity criterion:",
    )
    if not passed:
        ctx.forge.revert_code_mutation(mutation)
        if deficiency == "shrinkage":
            ctx.state["_dispatch_deficiency"] = "shrinkage"  # AP-14
        return eval_result, "prompt_forge"

    ctx.swarm.mark_epoch(f"code_mutation:{target}/{mutation_type}")
    return eval_result, "prompt_forge"


def _action_structural_experiment(action: dict[str, Any], ctx: _ActionContext):
    flags = action.get("flags", {})
    validation = ctx.lab.propose_flag_experiment(flags)
    if validation.get("status") != "valid":
        log.warning("Invalid flag experiment: %s", validation)
        return None, "structural_lab"

    ctx.lab.apply_flag_experiment(flags)
    eval_result = ctx.tower.hybrid_eval()

    # Revert if quality drops
    verdict = ctx.gate.check(eval_result)
    if not verdict:
        log.warning("Structural experiment failed safety gate, reverting")
        # Revert flags
        reverted = {k: not v for k, v in flags.items()}
        ctx.lab.apply_flag_experiment(reverted)
    else:
        # AP-7: Structural change accepted — invalidate stale Optuna trials
        ctx.swarm.mark_epoch(f"structural_experiment:{flags}")

    return eval_result, "structural_lab"


def _action_structural_prune(action: dict[str, Any], ctx: _ActionContext):
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
    pruned_content = ctx.lab.prune_block(original_content, block_id)
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
    pre_ratio = ctx.state.get("_last_instruction_ratio", 0.0)

    eval_result = ctx.tower.hybrid_eval()

    # Acceptance: safety gate passes AND instruction_token_ratio decreased
    verdict_result = ctx.gate.check(eval_result)
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
    ctx.swarm.mark_epoch(f"structural_prune:{target}/{block_id}")
    return eval_result, "structural_lab"


def _action_train_routing_models(action: dict[str, Any], ctx: _ActionContext):
    min_mem = action.get("min_memories", 500)
    ctx.lab.checkpoint_state(
        trial_id=ctx.state.get("trial_counter", 0),
        notes="Pre-training checkpoint",
    )
    result = ctx.lab.train_routing_models(min_memories=min_mem)
    log.info("Training result: %s", result)
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "structural_lab"


def _action_distill_skillbank(action: dict[str, Any], ctx: _ActionContext):
    teacher = action.get("teacher", "claude")
    categories = action.get("categories", ["routing"])
    ctx.lab.checkpoint_state(
        trial_id=ctx.state.get("trial_counter", 0),
        notes="Pre-distillation checkpoint",
    )
    result = ctx.lab.distill_skillbank(teacher=teacher, categories=categories)
    log.info("Distillation result: %s", result)
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "structural_lab"


def _action_reset_memories(action: dict[str, Any], ctx: _ActionContext):
    keep_seen = action.get("keep_seen", True)
    keep_skills = action.get("keep_skills", True)
    result = ctx.lab.reset_and_reseed(
        keep_seen=keep_seen,
        keep_skills=keep_skills,
        trial_id=ctx.state.get("trial_counter", 0),
    )
    log.info("Reset result: %s", result)
    return None, "structural_lab"


def _action_deep_eval(action: dict[str, Any], ctx: _ActionContext):
    tier = action.get("tier", 2)
    eval_result = ctx.tower.evaluate(tier=tier)
    return eval_result, "seeder"


def _action_rollback(action: dict[str, Any], ctx: _ActionContext):
    to_cp = action.get("to_checkpoint", "production_best")
    if to_cp == "production_best":
        ctx.lab.restore_checkpoint()
    else:
        ctx.lab.restore_checkpoint(Path(to_cp))
    ctx.gate.reset_failures()
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "structural_lab"


def _action_distill_knowledge(action: dict[str, Any], ctx: _ActionContext):
    # Evolution Manager: knowledge distillation (no eval, no system change)
    last_n = action.get("last_n", 10)
    if ctx.evo is not None and ctx.strategy_store is not None:
        result = ctx.evo.distill(
            journal_entries=ctx.journal.all_entries(),
            strategy_store=ctx.strategy_store,
            last_n=last_n,
            trial_id=ctx.state.get("trial_counter", 0),
        )
        log.info("Knowledge distillation: %s", result)
    else:
        log.warning("distill_knowledge requires evo + strategy_store")
    return None, "evolution_manager"


def _action_slot_compact(action: dict[str, Any], ctx: _ActionContext):
    # Expected Attention KV Compression: score and evict KV cache entries
    # Uses the kv_compress module for telemetry, gap guardrails, and structured results.
    from kv_compress import compress_slot, auto_compress_all

    port = action.get("port")
    slot_id = action.get("slot_id", 0)
    keep_ratio = action.get("keep_ratio", 0.5)
    keep_first = action.get("keep_first", 4)
    scorer = action.get("scorer", "expected_attention")
    layer_weights = action.get("layer_weights")
    n_future = action.get("n_future", 128)
    use_covariance = action.get("use_covariance", True)

    if port:
        # Single-port compression
        result = compress_slot(
            port=port, slot_id=slot_id, keep_ratio=keep_ratio,
            scorer=scorer, keep_first=keep_first, n_future=n_future,
            use_covariance=use_covariance, layer_weights=layer_weights,
        )
        if result.success:
            log.info(
                "KV compact port=%d slot=%d: evicted=%d keep=%.0f%% scorer=%s time=%.1fms",
                port, slot_id, result.n_evicted, keep_ratio * 100, scorer, result.elapsed_ms,
            )
        else:
            log.warning("KV compact failed on port %d: %s", port, result.error)
    else:
        # Compress all production slots
        results = auto_compress_all(
            threshold=action.get("threshold", 0.80),
            keep_ratio=keep_ratio, scorer=scorer, keep_first=keep_first,
            n_future=n_future, use_covariance=use_covariance,
            layer_weights=layer_weights,
        )
        for role, r in results.items():
            if r and r.success:
                log.info("KV compact %s: evicted=%d", role, r.n_evicted)

    # Evaluate quality after compaction to measure impact
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "slot_management"


# -----------------------------------------------------------------------------
# Dispatcher — maps action_type → handler
# -----------------------------------------------------------------------------

_ACTION_HANDLERS = {
    "seed_batch":             _action_seed_batch,
    "numeric_trial":          _action_numeric_trial,
    "prompt_mutation":        _action_prompt_mutation,
    "gepa_optimize":          _action_gepa_optimize,
    "code_mutation":          _action_code_mutation,
    "structural_experiment":  _action_structural_experiment,
    "structural_prune":       _action_structural_prune,
    "train_routing_models":   _action_train_routing_models,
    "distill_skillbank":      _action_distill_skillbank,
    "reset_memories":         _action_reset_memories,
    "deep_eval":              _action_deep_eval,
    "rollback":               _action_rollback,
    "distill_knowledge":      _action_distill_knowledge,
    "slot_compact":           _action_slot_compact,
}


def dispatch_action(
    action: dict[str, Any],
    seeder: "Seeder",
    swarm: "NumericSwarm",
    forge: "PromptForge",
    lab: "StructuralLab",
    tower: "EvalTower",
    gate: SafetyGate,
    archive: "ParetoArchive",
    journal: "ExperimentJournal",
    state: dict[str, Any],
    strategy_store: "StrategyStore | None" = None,
    evo: "EvolutionManager | None" = None,
) -> tuple[EvalResult | None, str]:
    """Execute an action and return (eval_result, species_name)."""
    action_type = action.get("type", "")

    # AP-9: Single-variable scope enforcement
    scope_err = validate_single_variable(action)
    if scope_err:
        log.warning("AP-9 scope violation: %s — skipping trial", scope_err)
        return None, action_type
    log.info("Dispatching action: %s", action_type)

    handler = _ACTION_HANDLERS.get(action_type)
    if handler is None:
        log.warning("Unknown action type: %s", action_type)
        return None, "unknown"

    ctx = _ActionContext(
        seeder=seeder, swarm=swarm, forge=forge, lab=lab, tower=tower,
        gate=gate, archive=archive, journal=journal, state=state,
        strategy_store=strategy_store, evo=evo,
    )
    return handler(action, ctx)
