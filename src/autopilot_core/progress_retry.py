"""progress_aware_retry_cap (default OFF): an attempt that advanced is not charged.

Reference: rowflow 603fd7099 (a deadline end that made progress does not charge
the auto-requeue cap).  "Advanced" here means the attempt scored rows that no
earlier attempt in the same retry series had scored.  The first attempt of a
series has nothing to advance past and is always charged.  Pure helpers; the
caps themselves stay where they live.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def progress_aware_retry_cap_enabled() -> bool:
    """Flag read; fails closed (OFF) if the feature registry is unavailable."""
    try:
        from src.features import features

        return bool(features().progress_aware_retry_cap)
    except Exception:  # noqa: BLE001 - a flag read must never break a retry cap
        return False


def qids_of(rows: Any) -> set[str]:
    """Scored row ids from an outcomes mapping or a list of per-question dicts."""
    if isinstance(rows, Mapping):
        return {str(k) for k in rows}
    out: set[str] = set()
    if isinstance(rows, (list, tuple)):
        for item in rows:
            if isinstance(item, Mapping):
                qid = str(item.get("qid") or item.get("question_id") or "").strip()
                if qid:
                    out.add(qid)
    return out


def advanced(prior_qids: Iterable[str] | None, attempt_qids: Iterable[str]) -> bool:
    """True iff the attempt scored a row absent from every prior attempt.

    ``prior_qids is None`` means there is no earlier attempt: not an advance.
    """
    if prior_qids is None:
        return False
    return bool(set(attempt_qids) - set(prior_qids))


def charged_attempts(attempt_qid_sets: Iterable[Iterable[str]]) -> int:
    """Attempts charged against a cap when the flag is on."""
    seen: set[str] | None = None
    charged = 0
    for qids in attempt_qid_sets:
        qs = set(qids)
        if not advanced(seen, qs):
            charged += 1
        seen = qs if seen is None else (seen | qs)
    return charged
