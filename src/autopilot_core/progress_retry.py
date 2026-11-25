"""progress_aware_retry_cap (default OFF): an attempt that advanced is not charged.

Reference: rowflow 603fd7099 (a deadline end that made progress does not charge
the auto-requeue cap).  The exemption covers only an attempt that ended INCOMPLETE
(infra, timeout, partial, unmeasured) AND scored rows no earlier attempt in the
series had scored.  A completed evaluation that failed on quality is always
charged.  The first attempt of a series is always charged.  Pure helpers; the
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


def eval_incomplete(result: Any) -> bool:
    """True iff the eval ended incomplete: rows lost to infra/timeout/partial, or unmeasured.

    Derived from what EvalResult already records (infra_failed_count, partial_count,
    n_exogenous_unrecovered, quality_measured).  A completed evaluation that failed
    on quality is NOT incomplete and is always charged.
    """
    def _n(name: str) -> int:
        try:
            return int(getattr(result, name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    return bool(
        _n("infra_failed_count") > 0
        or _n("partial_count") > 0
        or _n("n_exogenous_unrecovered") > 0
        or getattr(result, "quality_measured", True) is False
    )


def exempt(prior_qids: Iterable[str] | None, attempt_qids: Iterable[str], incomplete: bool) -> bool:
    """Exempt from the cap only if the attempt ended incomplete AND scored new rows."""
    return bool(incomplete) and advanced(prior_qids, attempt_qids)


def charged_attempts(attempts: Iterable[tuple[Iterable[str], bool]]) -> int:
    """Attempts charged against a cap when the flag is on.

    ``attempts`` is (qids, incomplete) per attempt, in order.
    """
    seen: set[str] | None = None
    charged = 0
    for qids, incomplete in attempts:
        qs = set(qids)
        if not exempt(seen, qs, incomplete):
            charged += 1
        seen = qs if seen is None else (seen | qs)
    return charged
