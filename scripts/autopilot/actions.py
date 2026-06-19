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
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from controller_io import validate_single_variable
from orchestration.repl_memory.strategy_store import excluded_strategy_evidence_trial_ids
from safety_gate import EvalResult, SafetyGate


def _apply_params(*args, **kwargs):
    """Call apply_params via the autopilot module so tests' monkeypatches stick.

    Tests historically monkeypatch `autopilot.apply_params`; importing the
    function directly here would bypass that. Lazy lookup through sys.modules
    avoids a circular import (actions.py is imported by autopilot.py at the
    bottom of autopilot's imports, by which time `apply_params` is bound).
    """
    import sys
    # autopilot is imported as either 'autopilot' (normal load mode),
    # 'scripts.autopilot.autopilot' (package-path tests), or '__main__'
    # (direct script execution). Prefer the package path when both aliases are
    # present so tests that monkeypatch that module cannot leak into real
    # config application.
    for module_name in ("scripts.autopilot.autopilot", "autopilot", "__main__"):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, "apply_params"):
            return mod.apply_params(*args, **kwargs)

    # Fallback: import config_applicator directly (no monkeypatch in play).
    from config_applicator import apply_params as _ap
    return _ap(*args, **kwargs)

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
_SKILL_EFFICACY_GATE_ENV = "AUTOPILOT_SKILL_EFFICACY_GATE"


@dataclass
class SkipOutcome:
    """Structured residue for an action that did NOT execute an eval.

    Returned (in the eval-result slot of the ``(result, species)`` tuple) instead
    of a bare ``None`` so the main loop can treat a non-executing action as a
    first-class trial outcome: journal it, fingerprint it, count it, and feed the
    reason back to the planner / blacklist. Before this existed, the dispatcher's
    ``return None`` path silently dropped the only actionable signal — which let
    the planner re-sample an impossible action forever (the 2026-06 graph_router
    deadlock: 119 identical invalid ``structural_experiment`` dispatches, each
    rejected with "graph_router feature requires specialist_routing feature",
    none of it ever reaching the planner).

    status: "invalid"  — failed pre-execution validation (actionable; e.g. a
                          feature flag whose dependency is not enabled). Eligible
                          for signature blacklisting because the reason is stable.
            "skipped"  — dropped by a dispatcher guard (AP-9 scope, dirty-tree
                          fence, unknown type, handler no-op). Counted + halted on
                          runaway, but NOT auto-blacklisted (the coarse pattern
                          would over-match, e.g. blacklisting all numeric_trials).
    reason:  human/planner-readable explanation (the validator/guard message).
    """
    status: str
    reason: str
    action_type: str = ""


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
    # 2026-05-23 Phase 5 — OrchestratorWatcher for exogenous-restart
    # detection. None = legacy behavior (no retry, no metadata propagation).
    # Phase 6/7 deployments construct one in autopilot.py main() and pass
    # it through dispatch_action → _action_seed_batch → Seeder.run_batch.
    watcher: Any | None = None


# -----------------------------------------------------------------------------
# Per-action handlers — one per action_type
# -----------------------------------------------------------------------------


def _action_seed_batch(action: dict[str, Any], ctx: _ActionContext):
    requested_n = int(action.get("n_questions", 10))
    suites = action.get("suites")

    # 2026-05-22: adaptive batch size — scale requested_n down to what
    # recent batch wall-clock suggests will fit within
    # SEEDING_BATCH_BUDGET_S (default 900s). Without this, the autopilot
    # asks for 10 questions every time and a 240s/question rate gives
    # ~40-min batches that crowd out other trials.
    try:
        import sys
        sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator/scripts/benchmark")
        from seeding_telemetry import (
            adaptive_batch_size as _adaptive_n,
            record_batch_duration as _record_batch,
        )
        adapted_n, reason = _adaptive_n(requested_n)
        if adapted_n != requested_n:
            log.warning(
                "[adaptive-batch] scaling seed_batch from %d → %d (%s)",
                requested_n, adapted_n, reason,
            )
        else:
            log.info("[adaptive-batch] keeping seed_batch n=%d (%s)", requested_n, reason)
        n = adapted_n
    except Exception as exc:
        log.warning("[adaptive-batch] telemetry import failed (%s) — using requested n=%d", exc, requested_n)
        n = requested_n
        _record_batch = None  # type: ignore[assignment]

    import time as _time
    _batch_start = _time.perf_counter()
    # 2026-05-23 Phase 4: pass the watcher (if ctx supplies one) so the
    # seeder's per-role calls can detect exogenous service reloads. Phase 5
    # wires the watcher into ctx; for now ctx.watcher may be None (backward
    # compatible — Seeder.run_batch's watcher kwarg defaults to None).
    seeder_result = ctx.seeder.run_batch(
        n_questions=n,
        suites=suites,
        watcher=getattr(ctx, "watcher", None),
    )
    _batch_elapsed = _time.perf_counter() - _batch_start

    # Record duration so the next batch can adapt
    if _record_batch is not None:
        try:
            _record_batch(n, _batch_elapsed)
            log.info(
                "[adaptive-batch] recorded duration: %dq in %.0fs (%.0fs/q)",
                n, _batch_elapsed, _batch_elapsed / max(n, 1),
            )
        except Exception:
            pass

    # After seeding, run T0 eval
    eval_result = ctx.tower.hybrid_eval()

    # 2026-05-23 Phase 4 — merge seeding-phase exogenous-restart metadata
    # into the trial-level EvalResult. The seed phase happens BEFORE
    # tower.hybrid_eval; if a reload corrupted the seed phase but the
    # later eval came up clean, the EvalResult would (without this merge)
    # look entirely sound and the trial would not be tagged. Per the
    # handoff Section 5.4: aggregate counters via additive merge,
    # concatenate the per-question id list and marker log.
    if seeder_result is not None:
        eval_result.n_exogenous_recovered += seeder_result.n_exogenous_recovered
        eval_result.n_exogenous_unrecovered += seeder_result.n_exogenous_unrecovered
        eval_result.n_external_restart += seeder_result.n_external_restart
        # Avoid double-counting question ids if the eval phase also tagged
        # the same id (different question pool, but defensive).
        existing_ids = set(eval_result.exogenous_question_ids)
        for qid in seeder_result.exogenous_question_ids:
            if qid not in existing_ids:
                eval_result.exogenous_question_ids.append(qid)
                existing_ids.add(qid)
        eval_result.exogenous_marker_log.extend(seeder_result.exogenous_marker_log)

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
        f"Trial #{f.trial_id} ({f.action_type}):\n"
        f"{ctx.journal.failure_analysis_for_prompt(f)}"
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
        if hasattr(ctx.strategy_store, "retrieve_for_journal"):
            strategies = ctx.strategy_store.retrieve_for_journal(
                query,
                journal=ctx.journal,
                k=3,
            )
        else:
            excluded_trial_ids = excluded_strategy_evidence_trial_ids(ctx.journal)
            strategies = ctx.strategy_store.retrieve(
                query,
                k=3,
                excluded_trial_ids=excluded_trial_ids,
            )
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


def _autopilot_attr(name: str) -> Any | None:
    """Resolve helpers from the already-loaded autopilot module without import cycles."""
    import sys

    for module_name in ("scripts.autopilot.autopilot", "autopilot", "__main__"):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    return None


def _record_seq_action_gate(
    eval_result: EvalResult,
    *,
    applied: bool,
    reason: str = "",
    candidate: str = "",
    core_id: str = "",
) -> None:
    details = getattr(eval_result, "details", None)
    if isinstance(details, dict):
        details["seq_action_gate_check"] = {
            "enabled": True,
            "applied": applied,
            "reason": reason,
            "candidate": candidate,
            "core_id": core_id,
        }


def _action_gate_check(
    action: dict[str, Any],
    ctx: _ActionContext,
    eval_result: EvalResult,
):
    """Run the action-local revert gate, threading W4 seq inputs when available."""
    if not getattr(ctx.gate, "use_sequential", False):
        return ctx.gate.check(eval_result)

    seq_inputs_for_trial = _autopilot_attr("_seq_inputs_for_trial")
    task_rate_qph_from = _autopilot_attr("task_rate_qph_from")
    if seq_inputs_for_trial is None or task_rate_qph_from is None:
        _record_seq_action_gate(
            eval_result,
            applied=False,
            reason="autopilot_seq_helpers_unavailable",
        )
        log.warning("Sequential action gate fell back to legacy check: helpers unavailable")
        return ctx.gate.check(eval_result)
    if ctx.journal is None:
        _record_seq_action_gate(eval_result, applied=False, reason="journal_unavailable")
        log.warning("Sequential action gate fell back to legacy check: journal unavailable")
        return ctx.gate.check(eval_result)

    try:
        seq_inputs = seq_inputs_for_trial(
            journal=ctx.journal,
            action=action,
            tier=eval_result.tier,
        )
        verdict = ctx.gate.check(
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
        _record_seq_action_gate(
            eval_result,
            applied=verdict.seq is not None,
            reason="" if verdict.seq is not None else "seq_inputs_not_ready",
            candidate=seq_inputs.get("candidate", ""),
            core_id=seq_inputs.get("core_id", ""),
        )
        return verdict
    except Exception as exc:  # noqa: BLE001 - action revert gate must remain non-fatal
        _record_seq_action_gate(
            eval_result,
            applied=False,
            reason=f"seq_action_gate_error:{type(exc).__name__}",
        )
        log.warning("Sequential action gate fell back to legacy check: %s", exc)
        return ctx.gate.check(eval_result)


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


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _skill_efficacy_without_result(ctx: _ActionContext) -> EvalResult | None:
    """Run the no-artifact arm for K-SKILL-1 when its gate is explicitly enabled."""
    if not _env_flag_enabled(_SKILL_EFFICACY_GATE_ENV):
        return None
    return ctx.tower.hybrid_eval()


def _skill_efficacy_accepts(
    *,
    without_result: EvalResult | None,
    with_result: EvalResult,
    artifact_kind: str,
    target: str,
    mutation_type: str,
) -> bool:
    """Default-off EV-10a accept-path hook.

    ``without_result`` is the pre-mutation no-artifact arm; ``with_result`` is the
    post-mutation arm. When the env flag is off, ``without_result`` is ``None`` and
    this helper is a no-op.
    """
    if without_result is None:
        return True

    from skill_efficacy import evaluate_skill_efficacy

    verdict = evaluate_skill_efficacy(
        without_result.per_suite_quality,
        with_result.per_suite_quality,
    )
    detail = {
        "enabled": True,
        "artifact_kind": artifact_kind,
        "target": target,
        "mutation_type": mutation_type,
        "accept": verdict.accept,
        "aggregate_delta": verdict.aggregate_delta,
        "per_suite_delta": verdict.per_suite_delta,
        "regressed_suites": verdict.regressed_suites,
        "reason": verdict.reason,
        "without_per_suite_quality": dict(without_result.per_suite_quality),
        "with_per_suite_quality": dict(with_result.per_suite_quality),
    }
    with_result.details["skill_efficacy"] = detail
    if not verdict.accept:
        log.warning(
            "Skill efficacy gate rejected %s mutation on %s: %s",
            artifact_kind,
            target,
            verdict.reason,
        )
    return verdict.accept


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
    if not getattr(mutation, "safety_valid", True):
        log.warning(
            "Prompt mutation failed transfer safety, skipping: %s",
            getattr(mutation, "safety_reason", "unsafe"),
        )
        return None, "prompt_forge"
    skill_without = _skill_efficacy_without_result(ctx)
    ctx.forge.apply_mutation(mutation)
    eval_result = ctx.tower.hybrid_eval()

    # Revert if quality drops
    verdict = _action_gate_check(action, ctx, eval_result)
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

    if not _skill_efficacy_accepts(
        without_result=skill_without,
        with_result=eval_result,
        artifact_kind="prompt",
        target=target,
        mutation_type=mutation_type,
    ):
        ctx.forge.revert_mutation(mutation)
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

    skill_without = _skill_efficacy_without_result(ctx)
    ctx.forge.apply_mutation(mutation)
    eval_result = ctx.tower.hybrid_eval()

    # Safety gate check
    verdict = _action_gate_check(action, ctx, eval_result)
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

    if not _skill_efficacy_accepts(
        without_result=skill_without,
        with_result=eval_result,
        artifact_kind="prompt",
        target=target,
        mutation_type="gepa",
    ):
        ctx.forge.revert_mutation(mutation)
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
    if not getattr(mutation, "safety_valid", True):
        log.warning(
            "Code mutation failed transfer safety, skipping: %s",
            getattr(mutation, "safety_reason", "unsafe"),
        )
        return None, "prompt_forge"
    if getattr(mutation, "mutated_content", None) == getattr(
        mutation, "original_content", None
    ):
        log.warning("Code mutation produced no file changes, skipping eval")
        return (
            SkipOutcome(
                "skipped",
                "code_mutation produced no file changes",
                "code_mutation",
            ),
            "prompt_forge",
        )

    skill_without = _skill_efficacy_without_result(ctx)
    ctx.forge.apply_code_mutation(mutation)
    eval_result = ctx.tower.hybrid_eval()

    verdict = _action_gate_check(action, ctx, eval_result)
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

    if not _skill_efficacy_accepts(
        without_result=skill_without,
        with_result=eval_result,
        artifact_kind="code",
        target=target,
        mutation_type=mutation_type,
    ):
        ctx.forge.revert_code_mutation(mutation)
        return eval_result, "prompt_forge"

    ctx.swarm.mark_epoch(f"code_mutation:{target}/{mutation_type}")
    return eval_result, "prompt_forge"


def _action_structural_experiment(action: dict[str, Any], ctx: _ActionContext):
    flags = action.get("flags", {})
    validation = ctx.lab.propose_flag_experiment(flags)
    status = validation.get("status")
    if status != "valid":
        log.warning("Invalid flag experiment: %s", validation)
        # Surface the validator's reason instead of dropping it (graph_router
        # deadlock fix). The errors list carries the exact fix, e.g.
        # "graph_router feature requires specialist_routing feature".
        #
        # Map the validator status onto the SkipOutcome status so only STABLE
        # validation failures ("invalid") are blacklist-eligible. A transient
        # "error" (orchestrator unreachable, exception) must NOT be blacklisted,
        # or a momentary blip could permanently ban a perfectly valid flag.
        if status == "invalid":
            reason = "; ".join(validation.get("errors", [])) or "invalid flag experiment"
            return SkipOutcome("invalid", reason, "structural_experiment"), "structural_lab"
        reason = str(validation.get("error", "flag experiment error"))
        return SkipOutcome("skipped", reason, "structural_experiment"), "structural_lab"

    apply_result = ctx.lab.apply_flag_experiment(flags)
    eval_result = ctx.tower.hybrid_eval()
    eval_result.details.setdefault("flag_attestation", apply_result.get("attestation"))
    eval_result.details.setdefault("flag_apply_result", apply_result)

    # Revert if quality drops
    verdict = _action_gate_check(action, ctx, eval_result)
    if not verdict:
        log.warning("Structural experiment failed safety gate, reverting")
        # Revert flags
        reverted = {k: not v for k, v in flags.items()}
        revert_result = ctx.lab.apply_flag_experiment(reverted)
        eval_result.details["flag_revert_result"] = revert_result
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
    verdict_result = _action_gate_check(action, ctx, eval_result)
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
    if isinstance(result, dict) and result.get("status") == "not_available":
        return (
            SkipOutcome(
                "skipped",
                "distill_skillbank unavailable: DistillationPipeline not available",
                "distill_skillbank",
            ),
            "structural_lab",
        )
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
    if ctx.evo is None or ctx.strategy_store is None:
        log.warning("distill_knowledge requires evo + strategy_store")
        return (
            SkipOutcome(
                "invalid",
                "distill_knowledge unavailable: missing evo or strategy_store",
                "distill_knowledge",
            ),
            "evolution_manager",
        )
    journal_entries = (
        ctx.journal.entries_with_supersessions()
        if hasattr(ctx.journal, "entries_with_supersessions")
        else ctx.journal.all_entries()
    )
    result = ctx.evo.distill(
        journal_entries=journal_entries,
        strategy_store=ctx.strategy_store,
        last_n=last_n,
        trial_id=ctx.state.get("trial_counter", 0),
    )
    log.info("Knowledge distillation: %s", result)
    if isinstance(result, dict) and result.get("status") == "failed":
        reason = result.get("reason") or "unknown failure"
        return (
            SkipOutcome(
                "invalid",
                f"distill_knowledge failed: {reason}",
                "distill_knowledge",
            ),
            "evolution_manager",
        )
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


# -----------------------------------------------------------------------------
# Dirty-tree fence — never let a file-mutating action commit (or even write on
# top of) pre-existing uncommitted work in its commit target.
#
# The forge stages differently per path, so the guard scope differs:
#   * code_mutation -> `git add <single file>`  => check that one file
#   * prompt_mutation / gepa_optimize
#                   -> `git add <prompts dir>`  => check the WHOLE prompts dir
#     (a dirty *sibling* prompt would otherwise be swept into the commit).
#   * structural_prune -> direct write to one prompt file => check that file
# Fires regardless of auto_commit (a mutation must not write over unrelated
# uncommitted work either) and fails CLOSED: any git error => treat as dirty
# and skip the mutation rather than risk committing someone else's work.
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "orchestration" / "prompts"
_CODE_FILE_MUTATORS = {"code_mutation"}
_PROMPT_DIR_MUTATORS = {"prompt_mutation", "gepa_optimize"}
_PROMPT_FILE_MUTATORS = {"structural_prune"}


def _pathspec_has_pending_changes(pathspec: Path) -> bool:
    """True if ``git status --porcelain`` reports any change (modified, staged,
    or untracked) under ``pathspec``. Fail-closed: returns True on git error."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(pathspec)],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 — fail closed on any git/subprocess error
        return True
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _mutation_dirty_target_reason(action: dict[str, Any]) -> str | None:
    """Return a skip reason if a file-mutating action would commit pre-existing
    uncommitted changes in its commit target; ``None`` if the action is
    non-mutating or its target is clean."""
    action_type = action.get("type", "")
    if action_type in _CODE_FILE_MUTATORS:
        target = action.get("file", "")
        if not target:
            return None  # missing-file is handled by the scope validator
        path = (_REPO_ROOT / target).resolve()
        if _pathspec_has_pending_changes(path):
            return (
                f"{action_type} target '{target}' has pre-existing uncommitted "
                "changes; skipping to avoid committing unrelated work"
            )
    elif action_type in _PROMPT_DIR_MUTATORS:
        # The prompt commit path stages the whole prompts dir, so any dirty
        # sibling prompt would be swept in — check the entire directory.
        if _pathspec_has_pending_changes(_PROMPTS_DIR):
            return (
                f"{action_type} would stage the whole prompts dir, which has "
                "pre-existing uncommitted changes; skipping to avoid committing "
                "unrelated work"
            )
    elif action_type in _PROMPT_FILE_MUTATORS:
        target = action.get("file", "")
        if not target:
            return None  # missing-file is handled by the scope validator
        path = (_PROMPTS_DIR / target).resolve()
        if _pathspec_has_pending_changes(path):
            return (
                f"{action_type} target '{target}' has pre-existing uncommitted "
                "changes; skipping to avoid overwriting unrelated work"
            )
    return None


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
    watcher: Any | None = None,
) -> tuple[EvalResult | SkipOutcome | None, str]:
    """Execute an action and return (eval_result, species_name).

    The first element is an ``EvalResult`` for a metric-collecting trial, a
    ``SkipOutcome`` for an action that failed validation or was dropped by a
    dispatcher guard (so the main loop gets the reason as residue), or ``None``
    for a handler that ran a side effect but collected no metrics (meta no-op).

    `watcher`: OrchestratorWatcher (Phase 5). When non-None, action handlers
    that issue /chat traffic (seed_batch + variants) use it to detect operator
    reloads of the orchestrator or llama-servers and retry inline. None
    preserves pre-Phase-5 behavior exactly.
    """
    action_type = action.get("type", "")

    # AP-9: Single-variable scope enforcement
    scope_err = validate_single_variable(action)
    if scope_err:
        log.warning("AP-9 scope violation: %s — skipping trial", scope_err)
        return SkipOutcome("skipped", f"AP-9 scope violation: {scope_err}", action_type), action_type
    # Dirty-tree fence (see _mutation_dirty_target_reason): a file-mutating
    # action must never commit — or write over — pre-existing uncommitted work.
    dirty_reason = _mutation_dirty_target_reason(action)
    if dirty_reason:
        log.warning("Dirty-tree fence: %s — skipping trial", dirty_reason)
        return SkipOutcome("skipped", f"Dirty-tree fence: {dirty_reason}", action_type), action_type

    log.info("Dispatching action: %s", action_type)

    handler = _ACTION_HANDLERS.get(action_type)
    if handler is None:
        log.warning("Unknown action type: %s", action_type)
        return SkipOutcome("skipped", f"unknown action type: {action_type}", action_type), "unknown"

    ctx = _ActionContext(
        seeder=seeder, swarm=swarm, forge=forge, lab=lab, tower=tower,
        gate=gate, archive=archive, journal=journal, state=state,
        strategy_store=strategy_store, evo=evo, watcher=watcher,
    )
    if hasattr(ctx.tower, "set_trial_context"):
        ctx.tower.set_trial_context(ctx.state.get("trial_counter"))
    return handler(action, ctx)
