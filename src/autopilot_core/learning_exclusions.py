"""Learning-exclusion policy shared by autopilot runtime and reports."""

from __future__ import annotations

from typing import Any

BENIGN_LEARNING_EXCLUSIONS = frozenset({"reproduction_confirmed", "mad_noise"})
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
    if "mad_noise" in categories:
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
