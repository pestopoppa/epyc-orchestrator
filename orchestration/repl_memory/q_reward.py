"""Pure reward computation for QScorer.

Extracted from q_scorer.py during the 2026-05-22 Task-G refactor. The
`compute_reward` function is config-driven and side-effect-free; QScorer's
`_compute_reward` method now delegates to it.

The contrastive and SPO+ adjustments stay in QScorer because they require
`self.store` + `self.embedder` (similarity retrieval, memory lookup) and
aren't pure config-only math.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .progress_logger import EventType, ProgressEntry

if TYPE_CHECKING:
    from .q_scorer import ScoringConfig

logger = logging.getLogger(__name__)

# Roles that are legitimately absent from baseline_tps_by_role: decommissioned
# roles that only appear in historical replay, and the test role. Warning on
# these would be noise during any rescore of retained logs.
_KNOWN_UNPRICED_ROLES = frozenset({"architect_coding", "mock", "plan_review"})
_warned_unpriced_roles: set[str] = set()

# RTG-09: same "absence must be visible, never silent" contract as
# _warned_unpriced_roles above, tracked separately because a role can be
# priced on tokens/sec (baseline_tps_by_role) while still missing a duration
# baseline (baseline_duration_by_role) -- the two artifacts have independent
# provenance and independent coverage.
_warned_unpriced_duration_roles: set[str] = set()
# Fires at most once per process: the caller did not supply task_duration_s at
# all (as opposed to the role having no baseline). Distinct from the per-role
# warning above because this is a CALLER defect (a call site with no
# task_started/task_completed pairing wired up), not a coverage gap in the
# derived baseline artifact.
_warned_missing_duration_arg = False


def _warn_unpriced_role(role: str) -> None:
    """Warn once per distinct role that has no baseline_tps entry.

    A role with no baseline skips every cost dimension and scores the full base
    reward. That silent miss is what disabled the entire cost/speed half of this
    function until 2026-07-21, so it is now surfaced rather than swallowed.
    """
    if role in _KNOWN_UNPRICED_ROLES or role in _warned_unpriced_roles:
        return
    _warned_unpriced_roles.add(role)
    logger.warning(
        "Reward cost penalty SKIPPED: role %r has no baseline_tps_by_role entry, "
        "so this task scores the full base reward. Either register the role in the "
        "q_scorer priors or fix the caller supplying it.",
        role[:80],
    )


def _warn_unpriced_role_duration(role: str) -> None:
    """Warn once per distinct role that has no baseline_duration entry.

    Mirrors `_warn_unpriced_role`: the duration DIMENSION is skipped for this
    role (never defaulted to zero penalty by fabricating a baseline), but the
    skip must be visible so a genuinely new/unregistered role does not quietly
    escape the primary speed axis the same way the pre-2026-07-21 role-key bug
    let every role escape the whole cost/speed half of the reward.
    """
    if role in _KNOWN_UNPRICED_ROLES or role in _warned_unpriced_duration_roles:
        return
    _warned_unpriced_duration_roles.add(role)
    logger.warning(
        "Reward DURATION penalty SKIPPED: role %r has no baseline_duration_by_role "
        "entry, so the primary speed axis does not price this task (tokens/sec "
        "secondary term, if priced, still applies). Run "
        "scripts/analysis/derive_duration_baselines.py to refresh coverage or "
        "register the role there.",
        role[:80],
    )


def _warn_missing_duration_arg() -> None:
    """Warn once (process-lifetime) that a caller never supplied task_duration_s.

    compute_reward cannot derive wall-clock duration itself -- it has no
    access to the task_started event, only whatever the caller passes. A
    caller that never wires task_duration_s silently loses the entire primary
    speed axis for every task it scores, which is exactly the shape of defect
    this reward has already suffered once (the role-key miss). Surfaced here
    so a new/unwired call site is diagnosable from the log instead of being a
    quiet, permanent zero on this dimension.
    """
    global _warned_missing_duration_arg
    if _warned_missing_duration_arg:
        return
    _warned_missing_duration_arg = True
    logger.warning(
        "Reward DURATION penalty SKIPPED: caller did not pass task_duration_s to "
        "compute_reward, so the primary wall-clock speed axis is inert for this "
        "call site. Wire task_started -> task_completed elapsed time through, or "
        "this task scores with no duration signal at all."
    )


def compute_reward(
    task_outcome: ProgressEntry,
    gate_results: List[ProgressEntry],
    escalations: List[ProgressEntry],
    plan_reviews: Optional[List[ProgressEntry]],
    cost_metrics: Optional[Dict[str, Any]],
    *,
    config: "ScoringConfig",
    task_duration_s: Optional[float] = None,
) -> float:
    """Compute reward from task outcome with optional cost penalty.

    Quality reward formula:
      - Base: success=1.0, failure=-0.5
      - Penalty for gate failures: -0.1 per failure
      - Penalty for escalations: -0.15 per escalation
      - Plan review bonus: +0.1 if approved, -0.2 if corrected

    Cost penalty (xRouter-style, correctness-gated), two speed dimensions:
      - PRIMARY (RTG-09): wall-clock task duration, `task_duration_s` (the
        caller-supplied task_completed.timestamp - task_started.timestamp).
        Graded 0..cost_lambda_duration between the role's measured p50 (no
        penalty) and p90 (full weight), saturating beyond p90. This axis sees
        orchestration/tool overhead that tokens/sec cannot.
      - SECONDARY (demoted 2026-09-24): tokens/sec-derived latency,
        cost_ratio = actual_elapsed / expected_elapsed,
        penalty = cost_penalty_lambda * max(0, cost_ratio - 1.0).
      - Both are only applied when quality reward > 0 (correct answers).
      - `task_duration_s=None` (caller did not wire it) or a role with no
        `baseline_duration_by_role` entry SKIPS the duration dimension only
        (warned once, never silent, never defaulted to a fabricated penalty)
        -- the other dimensions are independent and still apply.

    Returns the final reward clamped to [-1, 1].
    """
    if task_outcome.outcome == "success":
        base_reward = config.success_reward
    elif task_outcome.outcome == "partial":
        base_reward = config.partial_reward
    else:
        base_reward = config.failure_reward

    # Gate failure penalties.
    #
    # CJ-8: ONLY `GATE_FAILED` is charged. `EventType.GATE_INCONCLUSIVE` — a gate
    # that timed out, whose checker raised, or whose name does not exist — is
    # deliberately absent from this sum and from the `gate_results` bin that
    # feeds it. A gate that never ran produced no evidence about the model, and
    # charging -0.1 for it converts a harness defect into negative learning
    # signal about a subject that was never checked. Infra failure is the
    # ABSENCE of a measurement, never a bad one; the same rule the TASK_FAILED
    # disposition gate applies one layer up.
    gate_failures = sum(1 for g in gate_results if g.event_type == EventType.GATE_FAILED)
    gate_penalty = gate_failures * 0.1

    # Escalation penalties (unnecessary escalations are wasteful)
    escalation_penalty = len(escalations) * 0.15

    # Plan review adjustments (architect-in-the-loop)
    plan_review_adj = 0.0
    if plan_reviews:
        for pr in plan_reviews:
            decision = pr.data.get("decision", "")
            if decision == "ok":
                plan_review_adj += 0.1  # Approved — routing was correct
            else:
                plan_review_adj -= 0.2  # Corrected — routing needed fixing

    reward = base_reward - gate_penalty - escalation_penalty + plan_review_adj

    # Cost penalty: only penalize correct answers that were slower than expected.
    # Incorrect answers already receive low/zero reward — no cost signal needed.
    if cost_metrics and reward > 0:
        tokens_gen = cost_metrics.get("tokens_generated", 0)
        # cost_metrics is the TASK_COMPLETED entry's data dict, which carries the
        # role under "producer_role" (and "final_answer_role"); it has never
        # carried a bare "role" key. Reading only "role" resolved baseline_tps to
        # 0, which failed the guard below and silently disabled ALL THREE cost
        # dimensions plus the teacher shaping — leaving reward == base_reward.
        # Measured 2026-07-21 over 20,521 task_completed entries: "role" present
        # 0 times, "producer_role" present 20,521 times. "role" is kept first for
        # back-compat with any caller that does supply it.
        role = (
            cost_metrics.get("role")
            or cost_metrics.get("producer_role")
            or cost_metrics.get("final_answer_role")
            or ""
        )
        baseline_tps = config.baseline_tps_by_role.get(role, 0)
        if role and baseline_tps <= 0:
            # An unresolvable role skips EVERY cost dimension below and hands the
            # task the full base reward. That is exactly how this whole subsystem
            # went dark, so it must never be silent again. Known-benign cases:
            # decommissioned roles in historical replay (architect_coding, retired
            # 2026-06) and the "mock" test role. Anything else means either a new
            # role was added without a baseline_tps_by_role entry, or unvalidated
            # input reached the telemetry (see _normalize_role_field in
            # src/api/models/requests.py).
            _warn_unpriced_role(role)

        # Dimension 0 (RTG-09): wall-clock task-duration penalty -- the
        # PRIMARY speed axis. Wall-clock, not tokens/sec: tokens/sec is
        # gameable through tool calls and blind to orchestration/tool
        # overhead (DAR handoff: median wall/model-compute overhead 1.60x,
        # p90 9.09x over 19,433 tasks; worker_vision spends ~0.4s of model
        # compute inside ~11.9s of wall clock).
        #
        # Graded, not a single ratio threshold: 0 penalty at the role's
        # measured p50, scaling linearly to full `cost_lambda_duration`
        # weight at p90, saturating (never exceeding full weight) beyond it.
        # This is deliberately continuous -- CJ-8/L665's broader point is that
        # a success/failure-shaped reward wastes the graded signal the
        # underlying measurement actually carries.
        #
        # Two independent "missing" cases, both SKIP (never fabricate a
        # penalty) and both warn at most once so the skip is never silent --
        # the exact defect shape the role-key bug already taught this file:
        #   (a) caller never supplied task_duration_s at all;
        #   (b) role has no baseline_duration_by_role entry (coverage gap,
        #       or a role too new/rare to have cleared MIN_N_PER_ROLE).
        if task_duration_s is None:
            _warn_missing_duration_arg()
        else:
            duration_baseline = config.baseline_duration_by_role.get(role) if role else None
            if duration_baseline is None:
                if role:
                    _warn_unpriced_role_duration(role)
            else:
                p50 = duration_baseline.get("p50_s", 0.0)
                p90 = duration_baseline.get("p90_s", 0.0)
                if p90 > p50 > 0:
                    duration_frac = (task_duration_s - p50) / (p90 - p50)
                elif p50 > 0:
                    # Degenerate baseline (p90 <= p50, e.g. a low-n role with
                    # identical quantiles) -- fall back to a plain ratio past
                    # p50 rather than divide by zero or skip outright.
                    duration_frac = (task_duration_s / p50) - 1.0
                else:
                    duration_frac = 0.0
                duration_penalty = config.cost_lambda_duration * max(
                    0.0, min(1.0, duration_frac)
                )
                reward -= duration_penalty

        # Prefer generation_ms (clean generation time excluding prompt eval)
        # over elapsed_seconds (polluted by prompt processing time)
        gen_ms = cost_metrics.get("generation_ms", 0)
        if gen_ms > 0:
            elapsed = gen_ms / 1000.0
        else:
            elapsed = cost_metrics.get("elapsed_seconds", 0)

        # Dimension 1: Latency penalty (tokens/sec). DEMOTED to a SECONDARY
        # signal 2026-09-24 (RTG-09) -- see module docstring / Dimension 0
        # above. Still useful as a within-role compute-efficiency signal, just
        # no longer the axis the reward leans on for task-execution speed.
        if baseline_tps > 0 and tokens_gen > 0 and elapsed > 0:
            expected_elapsed = tokens_gen / baseline_tps
            cost_ratio = elapsed / expected_elapsed  # >1 = slower than expected
            cost_penalty = config.cost_penalty_lambda * max(0.0, cost_ratio - 1.0)
            reward -= cost_penalty

        # Dimension 2: Quality gap penalty — penalize using expensive model
        # when a cheaper one could suffice.
        if role in config.baseline_quality_by_role:
            model_quality = config.baseline_quality_by_role[role]
            quality_gap = max(0.0, model_quality - 0.75)  # 0.75 ≈ worker baseline
            reward -= config.cost_lambda_quality_gap * quality_gap

        # Dimension 3: Memory tier penalty — discourage WARM models when HOT works.
        if role in config.memory_cost_by_role:
            mem_cost = config.memory_cost_by_role[role]
            if mem_cost > 1.0:
                reward -= config.cost_lambda_memory * (mem_cost - 1.0)

        # Teacher telemetry shaping (regret + speedup bonus).
        regret = float(cost_metrics.get("regret", 0.0) or 0.0)
        reward -= config.teacher_regret_penalty * max(0.0, regret)

        speedup = float(
            cost_metrics.get("speedup_vs_teacher", cost_metrics.get("speedup", 1.0)) or 1.0
        )
        if speedup > 1.0:
            reward += config.teacher_speedup_bonus * min(speedup - 1.0, 1.0)

        # Web research source diversity bonus (Search-R1).
        wr_diversity = float(cost_metrics.get("wr_source_diversity", 0) or 0)
        wr_accuracy = float(cost_metrics.get("wr_accuracy", 0) or 0)
        if wr_diversity > 0 and wr_accuracy > 0:
            reward += 0.05 * wr_diversity

    # Final reward (clamped to [-1, 1])
    return max(-1.0, min(1.0, reward))
