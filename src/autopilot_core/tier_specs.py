"""Per-tier scoring specifications for the autopilot Pareto archive + safety gate.

Quality (`fraction_correct * 3`) is computed on a per-tier question set and the tiers differ
in difficulty (T0 ~10 easy q saturates ~2.4; T1 ~50 mixed ~1.6-1.9; T2 ~480 incl. GPQA/olympiad
~1.16; T3 expert/hard workflow rows), so quality is NOT comparable across tiers. The archive
and safety gate are therefore tier-segregated: each eval tier >= MIN_FRONTIER_EVAL_TIER keeps
its OWN frontier + baseline, and a trial is only ever ranked / gated against the SAME tier.

A `TierSpec` defines how a tier's objective tuple is built and its hypervolume reference point.
This keeps per-tier scoring **pluggable**: a future, more complex tier can carry different quality
semantics (its own scale/metric) by registering a different spec — without confounding existing
tiers or re-plumbing the archive/gate. All CURRENT tiers share the 4D
`(quality, speed, -cost, reliability)` shape and the canonical reference point.

This module is the SINGLE source of truth for tier scoring and is imported the same way
(`from src.autopilot_core.tier_specs import ...`) by both `scripts/autopilot` (which puts
ORCH_ROOT on sys.path) and `src/api`, so `TIER_SPECS` is one shared registry object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Canonical 4D objective shape + hypervolume reference point (worst acceptable values).
# (quality↑, speed↑, -cost↑ i.e. lower cost better, reliability↑)
DEFAULT_REFERENCE_POINT: tuple[float, ...] = (0.0, 0.0, -1.0, 0.0)
LEGACY_OBJECTIVE_POLICY = "legacy_4d_v1"
TASK_RATE_OBJECTIVE_POLICY = "task_rate_3d_v1"
TASK_RATE_REFERENCE_POINT: tuple[float, ...] = (0.0, 0.0, 0.0)

# T0 is a fast-reject sentinel tier (10q, quality saturates ~2.4 = 8/10) and never enters any
# frontier/baseline. Tiers >= this each keep their own segregated frontier + baseline.
MIN_FRONTIER_EVAL_TIER = 1
# The canonical "production" tier the controller optimizes and the dashboard shows by default.
# Other tiers (T2, T3) are broader/harder validation lanes with their own frontiers.
DEFAULT_FRONTIER_TIER = 1


def _default_objectives_from(result: Any) -> tuple[float, ...]:
    """4D objective tuple from an EvalResult-like object: (quality, speed, -cost, reliability)."""
    return (
        float(getattr(result, "quality", 0.0) or 0.0),
        float(getattr(result, "speed", 0.0) or 0.0),
        -float(getattr(result, "cost", 0.0) or 0.0),
        float(getattr(result, "reliability", 0.0) or 0.0),
    )


def _default_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """4D objective tuple from a journal-row dict (dashboard reconstruction), or None if unusable."""
    try:
        return (
            float(row.get("quality") or 0.0),
            float(row.get("speed") or 0.0),
            -float(row.get("cost") or 0.0),
            float(row.get("reliability") or 0.0),
        )
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nested(row: dict, *path: str) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _task_rate_inputs_from_row(row: dict) -> tuple[float, float]:
    eval_details = row.get("eval_details") or {}
    n_questions = (
        row.get("n_questions")
        or _nested(eval_details, "details", "total")
        or _nested(eval_details, "details", "n_questions")
        or _nested(eval_details, "details", "per_suite_counts_total")
    )
    if not n_questions:
        counts = _nested(eval_details, "details", "per_suite_counts")
        if isinstance(counts, dict):
            n_questions = sum(int(v) for v in counts.values() if int(v) > 0)
    eval_wall_s = (
        row.get("eval_wall_s")
        or eval_details.get("eval_wall_s")
        or _nested(eval_details, "details", "eval_wall_s")
    )
    return _as_float(n_questions), _as_float(eval_wall_s)


def task_rate_qph_from(result: Any) -> float:
    """Questions completed per eval-wall-hour; 0 when wall/n are unavailable."""
    n_questions = _as_float(
        getattr(result, "n_questions", None)
        or _nested(getattr(result, "details", {}) or {}, "total")
    )
    eval_wall_s = _as_float(
        getattr(result, "eval_wall_s", None)
        or _nested(getattr(result, "details", {}) or {}, "eval_wall_s")
    )
    if n_questions <= 0 or eval_wall_s <= 0:
        return 0.0
    return n_questions / (eval_wall_s / 3600.0)


def task_rate_qph_from_row(row: dict) -> float:
    """Questions completed per eval-wall-hour from a journal row."""
    n_questions, eval_wall_s = _task_rate_inputs_from_row(row)
    if n_questions <= 0 or eval_wall_s <= 0:
        return 0.0
    return n_questions / (eval_wall_s / 3600.0)


# ── SEQ-B: the paired rate measurement for the sequential non-inferiority axis ──
#
# `task_rate_qph_from` / `task_rate_qph_from_row` above are the Pareto/goodput rate
# metrics. They are deliberately NOT changed here: they feed archived objectives and
# `eval_details.goodput_qph`, and rescaling them would silently rewrite recorded history.
#
# The sequential rate axis needs something they do not provide: the CANDIDATE side and
# the INCUMBENT side must be the SAME measurement, or an unchanged config scores a rate
# regression. They were not the same measurement:
#
#   * `EvalTower._aggregate_decision_partitions` returns an EvalResult whose
#     `n_questions` / `details.total` counts only the DECISION partition (audit-shadow
#     questions excluded), while `eval_wall_s` is `max(r.eval_wall_s)` over the FULL
#     batch — the wall clock of every question that was actually asked. So the
#     candidate's `task_rate_qph_from` divides 55 questions by the wall time of 65.
#   * the incumbent comparator built in `autopilot._seq_inputs_for_trial` passed
#     `n_questions=len(outcome_map)`, i.e. the FULL question_results list that
#     `_aggregate_decision_partitions` explicitly copies over from the full result — 65.
#
# Measured over the 396 journaled sequential trials (2026-08-04): 381 carry
# `audit_shadow_excluded_partitions == ['audit']`, and the candidate/incumbent rate
# ratio is exactly 55/65 = 0.8462 (median; 50/60 = 0.8333 at p25). A candidate
# IDENTICAL to the incumbent therefore scored y = -0.1538 => z_rate = -0.208, i.e.
# negative rate evidence on every single trial, forever.
#
# The functions below are the ONE measurement both sides of the rate axis use. The
# numerator is the number of questions the wall clock ACTUALLY covers.

# A trial whose eval wall clock implies less than this per question did not measure a
# throughput — it aborted (fast-reject / crashed batch). Journal evidence: the 7 rows
# below 1.0 s/question are ALL `pareto_status=dominated`, and the extreme is trial 1302
# at 0.0008 s/question (65 questions in 0.054 s => 4.3 MILLION questions/hour). The
# legitimate distribution is p10 = 11.7 s/question, median 17.4. One such row inside a
# 120-row arithmetic MEAN moves the comparator by ~36,000 qph and pins every subsequent
# candidate at the `rate_noninferiority_z` clip floor. 1.0 s/question is far below any
# real LLM eval question and is a validity filter, not a tuning knob.
SEQ_RATE_MIN_SECONDS_PER_QUESTION = 1.0


def _seq_rate_question_count(question_results: Any, declared: Any) -> float:
    """Questions the eval wall clock actually covers.

    Prefers the observed per-question ledger (deduplicated by qid, matching
    `autopilot._question_outcome_map`) over the declared decision-partition count, so
    both sides of the paired comparison count the same questions.
    """
    if isinstance(question_results, (list, tuple)) and question_results:
        qids = set()
        for item in question_results:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("qid") or item.get("question_id") or "").strip()
            if qid:
                qids.add(qid)
        if qids:
            return float(len(qids))
        return float(len(question_results))
    return _as_float(declared)


def seq_task_rate_qph(
    *,
    question_results: Any,
    n_questions: Any,
    eval_wall_s: Any,
) -> float | None:
    """Paired questions-per-eval-wall-hour for the sequential rate axis.

    Returns ``None`` — never ``0.0`` — when the trial did not MEASURE a rate. The
    distinction is load-bearing: `task_rate_qph_from` returns 0.0 as an "unavailable"
    sentinel, and the safety gate's rate axis guard tested `task_rate is not None`, so an
    unmeasurable trial was fed to `rate_noninferiority_z` as a *measured* throughput of
    zero questions/hour => y = -1 => the clip floor z = -0.9. A missing measurement must
    SKIP the axis (wealth multiplied by exactly 1.0), the same doctrine
    `rebuild_candidate_view` already applies to out-of-domain z (SEQ-3a).
    """
    n = _seq_rate_question_count(question_results, n_questions)
    wall = _as_float(eval_wall_s)
    if n <= 0 or wall <= 0:
        return None
    if wall / n < SEQ_RATE_MIN_SECONDS_PER_QUESTION:
        return None
    return n / (wall / 3600.0)


def seq_task_rate_qph_from(result: Any) -> float | None:
    """Paired rate for a live EvalResult (candidate side of the rate axis)."""
    details = getattr(result, "details", {}) or {}
    return seq_task_rate_qph(
        question_results=getattr(result, "question_results", None),
        n_questions=(
            getattr(result, "n_questions", None) or _nested(details, "total")
        ),
        eval_wall_s=(
            getattr(result, "eval_wall_s", None) or _nested(details, "eval_wall_s")
        ),
    )


def seq_task_rate_qph_from_row(row: dict) -> float | None:
    """Paired rate for a journal row (incumbent side of the rate axis)."""
    eval_details = row.get("eval_details") or {}
    if not isinstance(eval_details, dict):
        eval_details = {}
    declared, eval_wall_s = _task_rate_inputs_from_row(row)
    return seq_task_rate_qph(
        question_results=eval_details.get("question_results"),
        n_questions=declared,
        eval_wall_s=eval_wall_s,
    )


def goodput_qph_from(result: Any) -> float:
    """Solved-question rate: quality-scaled task_rate on the 0-3 quality scale."""
    return (_as_float(getattr(result, "quality", 0.0)) / 3.0) * task_rate_qph_from(result)


def goodput_qph_from_row(row: dict) -> float:
    """Solved-question rate from a journal row."""
    return (_as_float(row.get("quality")) / 3.0) * task_rate_qph_from_row(row)


def task_rate_objectives_from(result: Any, tier: int | None = None) -> tuple[float, ...]:
    """Shadow 3D vector: (quality, task_rate_qph, reliability)."""
    _ = tier  # Reserved for future tier-specific rate semantics.
    return (
        _as_float(getattr(result, "quality", 0.0)),
        task_rate_qph_from(result),
        _as_float(getattr(result, "reliability", 0.0)),
    )


def task_rate_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """Shadow 3D vector from a journal row: (quality, task_rate_qph, reliability)."""
    try:
        return (
            float(row.get("quality") or 0.0),
            task_rate_qph_from_row(row),
            float(row.get("reliability") or 0.0),
        )
    except (TypeError, ValueError):
        return None


# ── W3 live-vector flip (2026-08-04, operator): dominance ranks TASKS/HOUR ──────
#
# The operator flipped the live dominance vector off tokens/second. Three findings
# shaped HOW, and none of them is the flag-flip the W3b-C tripwire implied:
#
# 1. The 3D `task_rate_3d_v1` shadow vector CANNOT become the live vector. Consumers
#    index the tuple POSITIONALLY past axis 1 — `safety_gate.py` reconstructs
#    `cost=-objectives[2], reliability=objectives[3]` and refuses any entry with
#    `len(objectives) < 4` ("frontier representative missing objective tuple"), and
#    `pareto_archive.py` reads `[2]`/`[3]` for its frontier summaries. Going 3D would
#    not have raised — it would have SILENTLY blocked every baseline promotion behind a
#    misleading message. `objectives_from` is documented as the single chokepoint, but a
#    chokepoint on CONSTRUCTION is not one on CONSUMPTION. So the 4D shape is preserved
#    and only axis 1's UNIT changes: t/s -> questions/hour, still "rate, higher better".
#
# 2. The rate metric had to be the CORRECTED one. `task_rate_qph_from` divides the
#    decision-partition question count by the FULL-batch wall clock, so `n` moving
#    43 -> 38 drops the objective ~12% with no change in real throughput (measured:
#    trial 775 qph=202.9 @ 51.5 t/s vs trial 778 qph=170.5 @ 49.8 t/s — a 19% objective
#    gap from a 3% speed difference). It also returns 0.0 for "unavailable" on 128 of
#    1466 journal rows, which archives an unmeasured trial as maximally slow.
#    `seq_task_rate_qph_from` counts the questions the wall clock actually covers,
#    returns None rather than 0.0, and rejects aborted batches.
#
# 3. An unmeasured rate must SKIP the archive, never enter it as a number. That is the
#    same doctrine as the `throughput_unmeasured` safety-gate category and SEQ-3a's
#    out-of-domain z handling: absence is not zero.
# The serving scheduler is part of a questions/hour instrument. v1 measured
# serial/client-global eval placement; v2 measures certified per-resource lanes
# with prompt-weighted admission and a model-judge scorer tail. Same tuple
# shape, different denominator instrument: never mix them in one frontier.
PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY = "task_rate_4d_v1"
RATE_4D_OBJECTIVE_POLICY = "task_rate_4d_v2_resource_lanes"


class UnmeasuredObjectiveError(ValueError):
    """Raised when a dominance axis was not measured on this trial.

    Callers MUST skip archiving rather than substituting a value. Zero is a real,
    maximally-bad throughput and would silently dominate-out a config whose rate simply
    was not captured.
    """


def _rate_objectives_from(result: Any) -> tuple[float, ...]:
    """Live 4D vector: (quality, seq_task_rate_qph, -cost, reliability)."""
    rate = seq_task_rate_qph_from(result)
    if rate is None:
        raise UnmeasuredObjectiveError(
            "task rate unmeasured (missing question ledger / eval_wall_s, or the batch "
            "aborted below the s/question validity floor)"
        )
    return (
        _as_float(getattr(result, "quality", 0.0)),
        float(rate),
        -_as_float(getattr(result, "cost", 0.0)),
        _as_float(getattr(result, "reliability", 0.0)),
    )


def _rate_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """Live 4D vector from a journal row; None when the row did not measure a rate."""
    rate = seq_task_rate_qph_from_row(row)
    if rate is None:
        return None
    try:
        return (
            float(row.get("quality") or 0.0),
            float(rate),
            -float(row.get("cost") or 0.0),
            float(row.get("reliability") or 0.0),
        )
    except (TypeError, ValueError):
        return None


def _policy_aware_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """Objective tuple built under the policy the ROW was recorded under.

    A journal row is evidence from a specific instrument. Pre-flip rows carry axis 1 in
    tokens/second and have no question ledger at all, so building them with the rate
    builder drops every one of them and reconstruction returns an EMPTY archive — which
    is how this first showed up (52 tests, `assert archive is not None`).

    Reading the policy off the row keeps historical replay faithful. It deliberately does
    NOT make the two series comparable: both are 4D, so `dominates` cannot catch a mixed
    frontier by shape. Separation is enforced at the ARCHIVE EPOCH instead — the live
    frontier restarts at the flip so it only ever holds one unit.
    """
    eval_details = row.get("eval_details")
    policy = ""
    if isinstance(eval_details, dict):
        policy = str(eval_details.get("objective_policy_live") or "")
    if not policy:
        policy = str(row.get("objective_policy_live") or "")
    if policy in {
        PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
        RATE_4D_OBJECTIVE_POLICY,
    }:
        return _rate_objectives_from_row(row)
    return _default_objectives_from_row(row)


def objectives_measurable(result: Any) -> bool:
    """True when this result carries every axis the live dominance vector needs."""
    return seq_task_rate_qph_from(result) is not None


# Public names for the PRE-FLIP tokens/second vector. Journalling and replay of the
# legacy series must build it explicitly rather than calling `objectives_from`, which
# now returns the live tasks/hour vector — the two are the same SHAPE and different
# UNITS, so a mislabelled call is invisible at the call site.
legacy_objectives_from = _default_objectives_from
legacy_objectives_from_row = _default_objectives_from_row

# Public names for the POST-FLIP tasks/hour vector, for callers that must pin a policy
# explicitly rather than inherit the row's stamp.
rate_objectives_from = _rate_objectives_from
rate_objectives_from_row = _rate_objectives_from_row


@dataclass(frozen=True)
class TierSpec:
    """How one eval tier's quality is scored for the Pareto archive + safety gate."""
    tier: int
    label: str
    reference_point: tuple[float, ...] = DEFAULT_REFERENCE_POINT
    objectives_from: Callable[[Any], tuple[float, ...]] = _rate_objectives_from
    objectives_from_row: Callable[[dict], tuple[float, ...] | None] = _policy_aware_objectives_from_row


# Registry: tier -> spec. All current tiers share the 4D shape, with axis 1 = questions/hour
# as of the 2026-08-04 W3 flip (`RATE_4D_OBJECTIVE_POLICY`). `_default_objectives_from{,_row}`
# are retained for LEGACY replay of pre-flip archives, whose axis 1 is tokens/second — the two
# units are not comparable, so a pre-flip archive must be replayed under its own policy, never
# merged into the live frontier. A future divergent-scoring tier is a new entry here, NOT a
# re-plumb of the archive/gate.
TIER_SPECS: dict[int, TierSpec] = {
    0: TierSpec(0, "T0 (10q sentinel, fast-reject)"),
    1: TierSpec(1, "T1 (50q gate)"),
    2: TierSpec(2, "T2 (480q comprehensive)"),
    3: TierSpec(3, "T3 (expert/hard workflow eval)"),
}


def spec_for(tier: int) -> TierSpec:
    """TierSpec for a tier; defaults to the 4D shape for any unregistered tier."""
    s = TIER_SPECS.get(int(tier))
    return s if s is not None else TierSpec(int(tier), f"T{int(tier)}")


def objectives_from(result: Any, tier: int | None = None) -> tuple[float, ...]:
    """Canonical objective construction — use this everywhere instead of `EvalResult.objectives`.

    `tier` defaults to `result.tier`. This is the single chokepoint so a future tier with
    different scoring semantics is handled by its spec, not by ad-hoc tuple-building at call sites.
    """
    t = int(tier if tier is not None else getattr(result, "tier", DEFAULT_FRONTIER_TIER))
    return spec_for(t).objectives_from(result)


def reference_point_for(tier: int) -> tuple[float, ...]:
    """Hypervolume reference point for a tier."""
    return spec_for(tier).reference_point
