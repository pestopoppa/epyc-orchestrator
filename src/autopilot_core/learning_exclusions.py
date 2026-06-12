"""Learning-exclusion policy shared by autopilot runtime and reports."""

from __future__ import annotations

from typing import Any

BENIGN_LEARNING_EXCLUSIONS = frozenset({"reproduction_confirmed", "mad_noise"})
WITHIN_NOISE_EXCLUSIONS = BENIGN_LEARNING_EXCLUSIONS

EXOGENOUS_CATEGORY_REASONS = {
    "exogenous_cache_flush": (
        "host page-cache flush or NUMA rewarm overlapped the trial; "
        "throughput was measured under an external remediation window"
    ),
    "exogenous_attestation_stale": (
        "running-state attestation freshness precondition failed; "
        "trial cannot be trusted as a clean system-state measurement"
    ),
    "exogenous_attestation_changed": (
        "running-state attestation artifact changed during the trial; "
        "trial spanned a system-state boundary"
    ),
}


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
    for category, fallback_reason in EXOGENOUS_CATEGORY_REASONS.items():
        if category in categories:
            reason = fallback_reason
            details = getattr(eval_result, "details", None)
            if isinstance(details, dict):
                trust = details.get("attestation_trust")
                if isinstance(trust, dict):
                    if category == "exogenous_attestation_changed":
                        reason = trust.get("change_reason") or reason
                    elif category == "exogenous_attestation_stale":
                        gate = trust.get("gate") or trust.get("after") or {}
                        if isinstance(gate, dict):
                            reason = gate.get("reason") or gate.get("status") or reason
            return category, reason, category

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

    return "", "", ""
