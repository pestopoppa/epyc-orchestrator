"""Learning-exclusion policy shared by autopilot runtime and reports."""

from __future__ import annotations

from typing import Any

BENIGN_LEARNING_EXCLUSIONS = frozenset(
    {"reproduction_confirmed", "mad_noise", "seq_accumulating", "seq_stale_reference"}
)
WITHIN_NOISE_EXCLUSIONS = BENIGN_LEARNING_EXCLUSIONS


def classify_learning_exclusion(verdict: Any, eval_result: Any) -> tuple[str, str, str]:
    """Decide whether a trial should be excluded from strategy learning.

    Returns ``(learning_excluded_by, learning_excluded_reason,
    deficiency_category_override)``. An empty first value means "include
    normally."
    """
    has_exo_unrecovered = getattr(eval_result, "n_exogenous_unrecovered", 0) > 0
    if has_exo_unrecovered:
        preview_ids = list(getattr(eval_result, "exogenous_question_ids", []))[:10]
        n_q = getattr(eval_result, "n_questions", 0) or len(
            getattr(eval_result, "exogenous_question_ids", []) or []
        )
        reason = (
            f"{eval_result.n_exogenous_unrecovered}/{n_q} questions "
            f"remained unrecovered after detected service reload "
            f"(sample ids: {preview_ids})"
        )
        return "exogenous_operator_reload", reason, "exogenous_reload"

    categories = getattr(verdict, "categories", None) or []
    # The MAD test is QUALITY-ONLY. A within-noise quality reading must not launder
    # a FAILED safety verdict (per-suite regression, quality floor, throughput, …)
    # into a "trusted within-noise representative" — that path admits the trial to
    # the Pareto frontier (autopilot.py upsert_representative) and suppresses its
    # deficiency. 2026-06-06: trial 707 failed three per-suite regression checks yet
    # was admitted as mad_noise. Only treat the within-noise tags as benign when the
    # verdict OTHERWISE PASSED; otherwise fall through to the normal failed-trial
    # path (deficiency from verdict.categories), which skips archive admission
    # without mislabelling the trial as corrupted data.
    verdict_passed = bool(getattr(verdict, "passed", True))
    if "mad_noise" in categories and verdict_passed:
        if "reproduction_confirmed" in categories:
            return (
                "reproduction_confirmed",
                "within-noise reproduction of an already-established above-"
                "baseline config: convergence/confirmation of an existing gain, "
                "not a new improvement and not corrupted data",
                "reproduction_confirmed",
            )
        return (
            "mad_noise",
            "quality improvement was within MAD noise band per safety_gate "
            "rolling-history significance test",
            "mad_noise",
        )

    # LEDGER-W4 (01c §3): the sequential e-process verdict, produced only when the
    # default-off AUTOPILOT_SEQ_VERDICT path runs in safety_gate. Three states:
    #   seq_accumulating — E below both confirm/refute thresholds: not-yet-decided,
    #     within-noise → exclude from learning (benign, like mad_noise) until a
    #     verdict resolves. Only honored when the verdict OTHERWISE PASSED, mirroring
    #     the mad_noise guard above (a within-noise reading must not launder a failed
    #     safety verdict into a trusted representative).
    #   seq_confirmed — a confirmed improvement (E_quality>=20 AND E_rate_noninf>=20):
    #     include normally (falls through to the empty include path below).
    #   seq_refuted — a refuted experiment: handled by the normal failed-trial path
    #     (deficiency from verdict.categories), so it is not special-cased here.
    if "seq_stale_reference" in categories and verdict_passed:
        return (
            "seq_stale_reference",
            "sequential e-process reference profile is stale; excluded from learning "
            "until a baseline-reference draw refreshes the profile (LEDGER-W4)",
            "seq_stale_reference",
        )
    if "seq_accumulating" in categories and verdict_passed:
        return (
            "seq_accumulating",
            "sequential e-process still accumulating evidence (wealth below the "
            "confirm/refute thresholds); within-noise, excluded from learning "
            "until the e-process resolves (LEDGER-W4, AUTOPILOT_SEQ_VERDICT)",
            "seq_accumulating",
        )

    return "", "", ""
