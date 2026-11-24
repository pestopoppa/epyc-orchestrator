"""AEI-DROPS-MANIFEST: what was dropped from an eval aggregate, and graded-fraction imbalance.

A quality number is only comparable between two arms when both graded about the same share of
the rows they attempted. ``drops_manifest`` states n scheduled, n scored, n dropped by reason, and
``graded_fraction = n_scored / n_attempted``. ``graded_fraction_flag`` flags a candidate-vs-baseline
comparison whose graded fractions differ by more than ``GRADED_FRACTION_GAP`` (5 points). Pure and
offline; no gate reads this module, and scorer semantics are unchanged.

``n_scored`` is the quality denominator (scored plus task_failed rows, ETR-1). Dropped rows are the
infra_failed and scoring_failed rows, keyed by their structural reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GRADED_FRACTION_GAP = 0.05
_UNCLASSIFIED = "unclassified"


def drops_manifest(
    *,
    n_scheduled: int,
    n_attempted: int,
    n_scored: int,
    infra_failed_reasons: Mapping[str, int] | None = None,
    scoring_failed: int = 0,
    scoring_failed_reasons: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build the manifest. ``n_dropped_by_reason`` keys are ``infra_failed:<reason>`` and
    ``scoring_failed:<reason>``; ``n_dropped`` is their sum."""
    by_reason: dict[str, int] = {}
    for reason, count in (infra_failed_reasons or {}).items():
        by_reason[f"infra_failed:{reason or _UNCLASSIFIED}"] = int(count)
    sf_reasons = dict(scoring_failed_reasons or {})
    sf_missing = int(scoring_failed) - sum(int(v) for v in sf_reasons.values())
    if sf_missing > 0:
        sf_reasons[_UNCLASSIFIED] = sf_reasons.get(_UNCLASSIFIED, 0) + sf_missing
    for reason, count in sf_reasons.items():
        by_reason[f"scoring_failed:{reason or _UNCLASSIFIED}"] = int(count)
    n_attempted = int(n_attempted)
    return {
        "n_scheduled": int(n_scheduled),
        "n_attempted": n_attempted,
        "n_scored": int(n_scored),
        "n_dropped": sum(by_reason.values()),
        "n_dropped_by_reason": dict(sorted(by_reason.items())),
        "graded_fraction": (int(n_scored) / n_attempted) if n_attempted else None,
    }


def graded_fraction_flag(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    gap: float = GRADED_FRACTION_GAP,
) -> dict[str, Any]:
    """Flag a quality delta whose arms graded different shares of their attempts.

    Each argument is an aggregate ``details`` dict (or a bare drops manifest) carrying
    ``graded_fraction``. ``imbalanced`` is True when the two fractions differ by more than ``gap``.
    """
    try:
        bf = baseline["graded_fraction"]
        cf = candidate["graded_fraction"]
        if bf is None or cf is None:
            raise ValueError
        bf, cf = float(bf), float(cf)
    except (KeyError, TypeError, ValueError):
        return {"comparable": False, "reason": "missing graded_fraction"}
    diff = cf - bf
    return {
        "comparable": True,
        "baseline_graded_fraction": bf,
        "candidate_graded_fraction": cf,
        "graded_fraction_diff": round(diff, 6),
        "gap_threshold": gap,
        "imbalanced": abs(diff) > gap + 1e-12,
    }
