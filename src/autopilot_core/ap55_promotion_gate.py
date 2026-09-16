"""AP-55 (b) + (c): same-regime seed re-run and candidate-batch homogeneity.

WHY
---
AP-55 (a)/(d) stamp every trial with an infra fingerprint and a
COMPARABLE / NON_COMPARABLE / UNVERIFIED verdict against its tier baseline's
regime. That alone does not stop a winner being named from numbers measured in
two regimes, nor from a batch whose members are statistically indistinguishable.
The external evidence (``intake-1362#record``): a 28-candidate search was
homogeneous within regime (chi-square p=0.24, p=0.50); its only clear pooled shift
coincided with a provider regime change; same-commit re-runs moved 12.1% -> 8.7%.

WHAT
----
(b) :func:`seed_rerun_verdict` — before a candidate is compared, the incumbent
    must have been RE-RUN (a marked baseline-reference draw) inside the
    candidate's infra regime. ``MISSING`` when no fingerprinted re-run exists,
    ``NON_COMPARABLE`` when every fingerprinted re-run is in another regime,
    otherwise the regime verdict of the newest re-run that is not
    NON_COMPARABLE (``COMPARABLE`` or ``UNVERIFIED``).
(c) :func:`batch_homogeneity` — before a winner is named, a chi-square test of
    homogeneity over the pass/fail counts of every candidate judged against the
    same baseline revision inside the candidate's regime (the seed re-runs are
    members too). ``HOMOGENEOUS`` (p >= alpha) means the leader is not
    separable from the batch, so no winner can be named from it.

:func:`promotion_gate` folds both legs under a mode:

* ``shadow``  (default) — compute and record; never hold.
* ``enforce`` — hold on a definitive negative: seed re-run MISSING or
  NON_COMPARABLE; batch HOMOGENEOUS; candidate without a fingerprint.
* ``strict``  — additionally hold unless the seed re-run is COMPARABLE and every
  batch member is COMPARABLE (UNVERIFIED never passes).

Binding is an operator decision (it changes what counts as a promotion, which is
human-amendment-only per MEASUREMENT.md), so the default is ``shadow``.

Missing data never yields COMPARABLE: an unfingerprinted row is excluded from
every batch and never counts as a re-run, and a kernel component read only from
the on-disk binary (server PIDs invisible) compares as UNVERIFIED
(see ``infra_fingerprint.compare_infra_fingerprints``). Rows written before this
module are read as-is and never back-filled.

Pure: no I/O, no inference, no process access. Never raises from
:func:`promotion_gate`.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Iterable, Mapping

from src.autopilot_core.infra_fingerprint import (
    COMPARABLE,
    NON_COMPARABLE,
    UNVERIFIED,
    compare_infra_fingerprints,
    fingerprint_digest,
)

AP55_GATE_SCHEMA_VERSION = 1

MISSING = "MISSING"
HOMOGENEOUS = "HOMOGENEOUS"
HETEROGENEOUS = "HETEROGENEOUS"
INSUFFICIENT = "INSUFFICIENT"

MODES = ("shadow", "enforce", "strict")
DEFAULT_ALPHA = 0.05
#: Cochran's rule of thumb: the chi-square approximation is unreliable when an
#: expected cell count is below this. Recorded, not acted on.
_MIN_EXPECTED = 5.0

_EXCLUDED_OUTCOMES = {"invalid", "skipped"}


def gate_mode(env: Mapping[str, str] | None = None) -> str:
    """``AUTOPILOT_AP55_PROMOTION_GATE``; anything unknown is ``shadow``."""
    raw = (env if env is not None else os.environ).get("AUTOPILOT_AP55_PROMOTION_GATE", "")
    mode = str(raw).strip().lower()
    return mode if mode in MODES else "shadow"


def gate_alpha(env: Mapping[str, str] | None = None) -> float:
    raw = (env if env is not None else os.environ).get("AUTOPILOT_AP55_HOMOGENEITY_ALPHA", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_ALPHA
    return value if 0.0 < value < 1.0 else DEFAULT_ALPHA


# ── statistics ────────────────────────────────────────────────────────────────


def chi2_sf(x: float, df: int) -> float:
    """Survival function of the chi-square distribution for integer ``df``.

    Closed forms (no scipy on the host): even df is a Poisson tail, odd df adds
    an erfc term. Exact to floating-point precision.
    """
    if df < 1:
        raise ValueError("df must be >= 1")
    if x <= 0.0:
        return 1.0
    half = x / 2.0
    if df % 2 == 0:
        term = math.exp(-half)
        total = term
        for i in range(1, df // 2):
            term *= half / i
            total += term
        return min(1.0, total)
    total = math.erfc(math.sqrt(half))
    term = math.exp(-half) * math.sqrt(half) / math.gamma(1.5)
    k = 1.5
    for _ in range((df - 1) // 2):
        total += term
        term *= half / k
        k += 1.0
    return min(1.0, total)


def chi2_homogeneity(counts: Iterable[tuple[int, int]]) -> dict[str, Any]:
    """Chi-square test of homogeneity on ``(correct, scored)`` rows (k x 2 table)."""
    rows = [(int(c), int(n)) for c, n in counts if int(n) > 0]
    k = len(rows)
    total_n = sum(n for _, n in rows)
    total_c = sum(c for c, _ in rows)
    out: dict[str, Any] = {"k": k, "n": total_n, "df": max(0, k - 1)}
    if k < 2:
        out.update(statistic=None, p_value=None, min_expected=None)
        return out
    p_pool = total_c / total_n
    stat = 0.0
    min_expected = math.inf
    for c, n in rows:
        exp_c = n * p_pool
        exp_w = n * (1.0 - p_pool)
        min_expected = min(min_expected, exp_c, exp_w)
        if exp_c > 0:
            stat += (c - exp_c) ** 2 / exp_c
        if exp_w > 0:
            stat += ((n - c) - exp_w) ** 2 / exp_w
    out.update(
        statistic=round(stat, 6),
        p_value=chi2_sf(stat, k - 1),
        min_expected=round(min_expected, 3),
        low_expected_counts=min_expected < _MIN_EXPECTED,
    )
    return out


# ── row access (JournalEntry objects or plain dicts) ─────────────────────────


def _get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _eval_details(row: Any) -> Mapping[str, Any]:
    value = _get(row, "eval_details", {}) or {}
    return value if isinstance(value, Mapping) else {}


def _trusted_row(row: Any, tier: int) -> bool:
    if _get(row, "bug_corrupted_by", ""):
        return False
    if (_get(row, "outcome_status", "ok") or "ok") in _EXCLUDED_OUTCOMES:
        return False
    try:
        return int(_get(row, "tier", -1)) == int(tier)
    except (TypeError, ValueError):
        return False


def is_seed_rerun(row: Any) -> bool:
    return bool(_eval_details(row).get("seq_baseline_reference_draw"))


def _row_fingerprint(row: Any) -> Mapping[str, Any] | None:
    fp = _get(row, "infra_fingerprint", None)
    return fp if isinstance(fp, Mapping) and fingerprint_digest(fp) else None


def _baseline_revision(row: Any) -> Any:
    pin = _get(row, "baseline_pin", {}) or {}
    return pin.get("baseline_revision") if isinstance(pin, Mapping) else None


# ── (b) seed re-run ──────────────────────────────────────────────────────────


def seed_rerun_verdict(
    entries: Iterable[Any],
    *,
    tier: int,
    candidate_fingerprint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Is there a re-run of the incumbent inside the candidate's regime?"""
    out: dict[str, Any] = {
        "status": MISSING,
        "trial_id": None,
        "differing_components": [],
        "unverified_components": [],
        "non_comparable_reruns": 0,
        "unfingerprinted_reruns": 0,
    }
    if not fingerprint_digest(candidate_fingerprint):
        out["status"] = UNVERIFIED
        out["reason"] = "candidate carries no infra fingerprint"
        return out
    latest_foreign: dict[str, Any] | None = None
    for row in reversed(list(entries)):
        if not _trusted_row(row, tier) or not is_seed_rerun(row):
            continue
        fp = _row_fingerprint(row)
        if fp is None:
            out["unfingerprinted_reruns"] += 1
            continue
        verdict = compare_infra_fingerprints(candidate_fingerprint, fp, reference_label="seed_rerun")
        if verdict["status"] == NON_COMPARABLE:
            out["non_comparable_reruns"] += 1
            if latest_foreign is None:
                latest_foreign = {"trial_id": _get(row, "trial_id"), **verdict}
            continue
        out.update(
            status=verdict["status"],
            trial_id=_get(row, "trial_id"),
            unverified_components=list(verdict["unverified_components"]),
            reason=f"seed re-run {_get(row, 'trial_id')}: {verdict['reason']}",
        )
        return out
    if latest_foreign is not None:
        out.update(
            status=NON_COMPARABLE,
            trial_id=latest_foreign["trial_id"],
            differing_components=list(latest_foreign["differing_components"]),
            reason="no seed re-run in this regime; newest differs in "
            + ",".join(latest_foreign["differing_components"]),
        )
    else:
        out["reason"] = "no fingerprinted seed re-run for this tier"
    return out


# ── (c) batch homogeneity ────────────────────────────────────────────────────


def batch_homogeneity(
    entries: Iterable[Any],
    *,
    tier: int,
    baseline_revision: Any,
    candidate_key: str,
    candidate_counts: tuple[int, int],
    candidate_fingerprint: Mapping[str, Any] | None,
    counts_fn: Callable[[Any], tuple[int, int]],
    key_fn: Callable[[Any], str],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Chi-square homogeneity over the in-regime batch judged against one incumbent.

    Members are pooled by ``key_fn`` (config identity) so replays of one config are
    one candidate; seed re-runs are pooled under ``"seed"``. ``counts_fn`` returns
    ``(correct, scored)`` from a row's quality-admissible outcomes.
    """
    out: dict[str, Any] = {
        "status": INSUFFICIENT,
        "alpha": alpha,
        "baseline_revision": baseline_revision,
        "members": {},
        "batch_regime": COMPARABLE,
        "cross_regime_excluded": 0,
        "unfingerprinted_excluded": 0,
    }
    if not fingerprint_digest(candidate_fingerprint):
        out.update(status=UNVERIFIED, batch_regime=UNVERIFIED,
                   reason="candidate carries no infra fingerprint")
        return out
    if baseline_revision is None:
        out["reason"] = "no baseline revision: no incumbent defines a batch"
        return out
    pooled: dict[str, list[int]] = {}
    trial_ids: dict[str, list[Any]] = {}

    def _add(key: str, counts: tuple[int, int], trial_id: Any) -> None:
        c, n = int(counts[0]), int(counts[1])
        if n <= 0:
            return
        slot = pooled.setdefault(key, [0, 0])
        slot[0] += c
        slot[1] += n
        trial_ids.setdefault(key, []).append(trial_id)

    for row in entries:
        if not _trusted_row(row, tier) or _baseline_revision(row) != baseline_revision:
            continue
        fp = _row_fingerprint(row)
        if fp is None:
            out["unfingerprinted_excluded"] += 1
            continue
        verdict = compare_infra_fingerprints(candidate_fingerprint, fp)
        if verdict["status"] == NON_COMPARABLE:
            out["cross_regime_excluded"] += 1
            continue
        if verdict["status"] != COMPARABLE:
            out["batch_regime"] = UNVERIFIED
        key = "seed" if is_seed_rerun(row) else str(key_fn(row) or "")
        if not key:
            continue
        _add(key, counts_fn(row), _get(row, "trial_id"))
    _add(candidate_key or "candidate", candidate_counts, "current")
    # A candidate is always measured in its own regime; an unreadable component on
    # its own side makes the whole batch unverified.
    if compare_infra_fingerprints(candidate_fingerprint, candidate_fingerprint)["status"] != COMPARABLE:
        out["batch_regime"] = UNVERIFIED

    test = chi2_homogeneity((c, n) for c, n in pooled.values())
    out["test"] = test
    out["members"] = {
        key: {"correct": c, "scored": n, "rate": round(c / n, 6), "trials": trial_ids[key]}
        for key, (c, n) in sorted(pooled.items())
    }
    ranked = sorted(pooled.items(), key=lambda kv: kv[1][0] / kv[1][1], reverse=True)
    out["leader"] = ranked[0][0] if ranked else None
    ck = candidate_key or "candidate"
    out["candidate_rank"] = next((i + 1 for i, (k, _) in enumerate(ranked) if k == ck), None)
    if test["p_value"] is None:
        out["reason"] = "fewer than two batch members in this regime"
    elif test["p_value"] >= alpha:
        out["status"] = HOMOGENEOUS
        out["reason"] = f"p={test['p_value']:.3f} >= {alpha}: leader not separable from batch"
    else:
        out["status"] = HETEROGENEOUS
        out["reason"] = f"p={test['p_value']:.4f} < {alpha}"
    return out


# ── fold ─────────────────────────────────────────────────────────────────────


def _holds(mode: str, seed: Mapping[str, Any], batch: Mapping[str, Any]) -> list[str]:
    if mode == "shadow":
        return []
    reasons: list[str] = []
    sstat = seed.get("status")
    if sstat in (MISSING, NON_COMPARABLE):
        reasons.append(f"seed_rerun:{sstat}")
    elif sstat == UNVERIFIED and (mode == "strict" or seed.get("trial_id") is None):
        # trial_id None: the candidate itself has no fingerprint.
        reasons.append("seed_rerun:UNVERIFIED")
    bstat = batch.get("status")
    if bstat in (HOMOGENEOUS, UNVERIFIED):
        reasons.append(f"batch:{bstat}")
    if mode == "strict" and batch.get("batch_regime") != COMPARABLE:
        reasons.append(f"batch_regime:{batch.get('batch_regime')}")
    return reasons


def promotion_gate(
    entries: Iterable[Any],
    *,
    tier: int,
    baseline_revision: Any,
    candidate_key: str,
    candidate_counts: tuple[int, int],
    candidate_fingerprint: Mapping[str, Any] | None,
    counts_fn: Callable[[Any], tuple[int, int]],
    key_fn: Callable[[Any], str],
    mode: str | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Both AP-55 legs plus the hold decision. Never raises."""
    resolved_mode = mode if mode in MODES else gate_mode()
    resolved_alpha = alpha if alpha is not None else gate_alpha()
    try:
        rows = list(entries)
        seed = seed_rerun_verdict(rows, tier=tier, candidate_fingerprint=candidate_fingerprint)
        batch = batch_homogeneity(
            rows,
            tier=tier,
            baseline_revision=baseline_revision,
            candidate_key=candidate_key,
            candidate_counts=candidate_counts,
            candidate_fingerprint=candidate_fingerprint,
            counts_fn=counts_fn,
            key_fn=key_fn,
            alpha=resolved_alpha,
        )
        reasons = _holds(resolved_mode, seed, batch)
        return {
            "schema_version": AP55_GATE_SCHEMA_VERSION,
            "mode": resolved_mode,
            "seed_rerun": seed,
            "batch_homogeneity": batch,
            "hold": bool(reasons),
            "hold_reasons": reasons,
        }
    except Exception as exc:  # noqa: BLE001 - the gate must never lose a trial
        err = f"{type(exc).__name__}: {exc}"[:200]
        hold = resolved_mode != "shadow"
        return {
            "schema_version": AP55_GATE_SCHEMA_VERSION,
            "mode": resolved_mode,
            "status": "error",
            "error": err,
            # An errored gate cannot certify anything: fail closed when binding.
            "hold": hold,
            "hold_reasons": ["gate_error"] if hold else [],
        }


def gate_summary(gate: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compact form for eval_details / the claim tuple / adapters."""
    if not isinstance(gate, Mapping) or not gate:
        return {}
    seed = gate.get("seed_rerun") or {}
    batch = gate.get("batch_homogeneity") or {}
    test = batch.get("test") or {}
    return {
        "mode": gate.get("mode", ""),
        "seed_rerun": seed.get("status", gate.get("status", "")),
        "seed_rerun_trial_id": seed.get("trial_id"),
        "batch_homogeneity": batch.get("status", gate.get("status", "")),
        "batch_regime": batch.get("batch_regime", ""),
        "batch_k": test.get("k"),
        "batch_p_value": test.get("p_value"),
        "hold": bool(gate.get("hold")),
        "hold_reasons": list(gate.get("hold_reasons") or []),
    }
