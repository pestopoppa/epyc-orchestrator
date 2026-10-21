"""EV-18: intention-to-treat quality beside the per-protocol (exclusion-based) quality.

Per-protocol quality drops unparseable answers from the denominator (``scoring_failed``
under ``EXCLUDE_UNPARSEABLE_ANSWERS``). Intention-to-treat counts them as failures. Both are
reported; a delta whose sign or significance differs between the two is scorer-dependent and
must be re-scored by a second independent scorer before anyone acts on it. Pure; no gate reads
this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

QUALITY_SCALE = 3.0
Z_CRITICAL = 1.96


def itt_counts(
    *, correct: int, n_scored: int, parse_failure_count: int, scoring_failed_count: int
) -> dict[str, Any]:
    """ITT denominator = per-protocol denominator + unparseable answers that were excluded.

    ``parse_failure_count`` is a scorer-side counter; only rows actually excluded as
    ``scoring_failed`` can have left the denominator, so the addition is capped by
    ``scoring_failed_count`` (it is exactly the count when every excluded row was a parse
    failure, and a conservative lower bound on the ITT penalty otherwise).
    """
    added = max(0, min(int(parse_failure_count), int(scoring_failed_count)))
    denom = int(n_scored) + added
    return {
        "itt_unparseable_counted_as_failure": added,
        "itt_denominator": denom,
        "quality_itt": (correct / denom) * QUALITY_SCALE if denom else 0.0,
    }


def _two_prop_z(c1: int, n1: int, c2: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return 0.0
    p1, p2 = c1 / n1, c2 / n2
    pooled = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return (p2 - p1) / se if se > 0 else 0.0


def itt_delta_flag(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare candidate-minus-baseline under both estimands.

    Each argument needs ``correct``, ``n_scored`` and ``itt_denominator`` (from ``details``).
    Returns the two deltas, two-proportion z for each, and ``scorer_dependent`` when the sign
    or the |z| >= 1.96 verdict differs.
    """
    try:
        bc, cc = int(baseline["correct"]), int(candidate["correct"])
        bn, cn = int(baseline["n_scored"]), int(candidate["n_scored"])
        bi, ci = int(baseline["itt_denominator"]), int(candidate["itt_denominator"])
    except (KeyError, TypeError, ValueError):
        return {"comparable": False, "reason": "missing correct/n_scored/itt_denominator"}
    pp = (cc / cn if cn else 0.0) - (bc / bn if bn else 0.0)
    itt = (cc / ci if ci else 0.0) - (bc / bi if bi else 0.0)
    z_pp, z_itt = _two_prop_z(bc, bn, cc, cn), _two_prop_z(bc, bi, cc, ci)
    sign_differs = (pp > 0) != (itt > 0) and not (pp == 0 and itt == 0)
    sig_differs = (abs(z_pp) >= Z_CRITICAL) != (abs(z_itt) >= Z_CRITICAL)
    return {
        "comparable": True,
        "delta_per_protocol": round(pp * QUALITY_SCALE, 6),
        "delta_itt": round(itt * QUALITY_SCALE, 6),
        "z_per_protocol": round(z_pp, 4),
        "z_itt": round(z_itt, 4),
        "sign_differs": sign_differs,
        "significance_differs": sig_differs,
        "scorer_dependent": sign_differs or sig_differs,
        "requires_second_scorer_rescore": sign_differs or sig_differs,
    }
