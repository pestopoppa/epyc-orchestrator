"""TD-18 calibration receipt upgrade (zero inference, pure functions).

Clean-room implementation of the receipt additions TD-18 lists, written from the
metric definitions (not copied from any external source). Everything here takes
already-scored rows ``(confidence, outcome, kind, cluster)`` and returns plain
dicts; it never calls a model. Seeds are explicit so two runs reproduce.

The extras are reported PER QUESTION KIND by ``build_receipt_upgrade``:

  * ``ece_noise_floor``      ECE of a perfectly calibrated predictor with the
                             same confidences (outcomes resampled ~ Bernoulli(c)).
                             An observed ECE inside this band is not evidence of
                             miscalibration.
  * ``clustered_bootstrap``  record/template-clustered bootstrap CI for ECE, and
                             ``paired_clustered_bootstrap_diff`` for ECE(A)-ECE(B).
  * ``selective``            tie-aware coverage at a stated error budget, and AURC.
  * ``nll`` / refit-T        NLL and refit temperature at several log floors and
                             clamps (sensitivity), plus a search-bound flag.
  * ``endpoints``            exact-0 / exact-1 confidence rates.
  * ``temperature_oof``      one in-distribution temperature plus a group-disjoint
                             out-of-fold check.
  * ``model_pin``            caller-supplied pin recorded verbatim.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.llm_primitives.stat_tests import expected_calibration_error

SCHEMA = "typed_decisions.calibration_receipt_upgrade.v1"
T_BOUNDS = (0.05, 20.0)
DEFAULT_LOG_FLOORS = (1e-12, 1e-6, 1e-3)
DEFAULT_CLAMPS = (0.0, 1e-3, 1e-2)


def _ece(conf: Sequence[float], y: Sequence[float]) -> float:
    value = expected_calibration_error(conf, y, n_bins=10)
    return 0.0 if value is None else float(value)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def ece_noise_floor(conf: Sequence[float], *, n_resamples: int = 500, seed: int = 0) -> dict[str, Any]:
    """ECE distribution of a perfectly calibrated predictor with these confidences."""
    rng = random.Random(seed)
    draws = sorted(
        _ece(conf, [1.0 if rng.random() < c else 0.0 for c in conf]) for _ in range(n_resamples)
    ) if conf else []
    return {
        "n": len(conf), "n_resamples": n_resamples, "seed": seed,
        "mean": sum(draws) / len(draws) if draws else None,
        "p95": _quantile(draws, 0.95) if draws else None,
    }


def _cluster_index(clusters: Sequence[Any]) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    return groups


def clustered_bootstrap_ece(
    conf: Sequence[float], y: Sequence[float], clusters: Sequence[Any],
    *, n_boot: int = 500, seed: int = 0,
) -> dict[str, Any]:
    """Percentile CI for ECE, resampling whole clusters (records/templates).

    ECE is positively biased in small samples, so the point estimate can sit
    outside this percentile interval; compare against ``ece_noise_floor``.
    """
    groups = _cluster_index(clusters)
    keys = list(groups)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot if keys else 0):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in groups[k]]
        stats.append(_ece([conf[i] for i in idx], [y[i] for i in idx]))
    stats.sort()
    return {
        "n": len(conf), "n_clusters": len(keys), "n_boot": n_boot, "seed": seed,
        "ece": _ece(conf, y) if conf else None,
        "ci95": [_quantile(stats, 0.025), _quantile(stats, 0.975)] if stats else None,
    }


def paired_clustered_bootstrap_diff(
    conf_a: Sequence[float], conf_b: Sequence[float], y: Sequence[float],
    clusters: Sequence[Any], *, n_boot: int = 500, seed: int = 0,
) -> dict[str, Any]:
    """ECE(A) - ECE(B) on the SAME rows, resampling clusters jointly (paired)."""
    if not (len(conf_a) == len(conf_b) == len(y) == len(clusters)):
        raise ValueError("paired arms must share rows")
    groups = _cluster_index(clusters)
    keys = list(groups)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot if keys else 0):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in groups[k]]
        yy = [y[i] for i in idx]
        diffs.append(_ece([conf_a[i] for i in idx], yy) - _ece([conf_b[i] for i in idx], yy))
    diffs.sort()
    point = (_ece(conf_a, y) - _ece(conf_b, y)) if y else None
    return {
        "n_clusters": len(keys), "n_boot": n_boot, "seed": seed, "diff": point,
        "ci95": [_quantile(diffs, 0.025), _quantile(diffs, 0.975)] if diffs else None,
        "ci_excludes_zero": bool(diffs) and (_quantile(diffs, 0.025) > 0 or _quantile(diffs, 0.975) < 0),
    }


def selective_prediction(
    conf: Sequence[float], y: Sequence[float], *, error_budget: float = 0.1,
) -> dict[str, Any]:
    """Tie-aware risk-coverage: tied confidences enter together, never in input order.

    ``coverage_at_budget`` is the largest coverage whose error rate is <= budget
    (whole tie groups only). ``aurc`` integrates risk over coverage with the
    risk evaluated at tie-group boundaries (trapezoid between boundaries).
    """
    n = len(conf)
    if n == 0:
        return {"n": 0, "error_budget": error_budget, "coverage_at_budget": None, "aurc": None}
    order: dict[float, list[float]] = defaultdict(list)
    for c, o in zip(conf, y):
        order[float(c)].append(float(o))
    taken = errors = 0
    cov_prev, risk_prev = 0.0, 0.0
    area = 0.0
    best = 0.0
    first = True
    for c in sorted(order, reverse=True):
        outs = order[c]
        taken += len(outs)
        errors += sum(1 for o in outs if o < 0.5)
        cov, risk = taken / n, errors / taken
        if first:
            risk_prev, first = risk, False
        area += (cov - cov_prev) * (risk_prev + risk) / 2
        if risk <= error_budget:
            best = cov
        cov_prev, risk_prev = cov, risk
    return {
        "n": n, "error_budget": error_budget, "coverage_at_budget": best, "aurc": area,
        "n_tie_groups": len(order),
    }


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _prep(conf: Sequence[float], floor: float, clamp: float) -> list[float]:
    lo, hi = max(floor, clamp), 1.0 - max(floor, clamp)
    return [min(max(float(c), lo), hi) for c in conf]


def nll(conf: Sequence[float], y: Sequence[float], *, floor: float = 1e-12, clamp: float = 0.0) -> float | None:
    """Mean binary NLL of ``confidence`` as P(correct); ``floor`` guards log(0)."""
    if not conf:
        return None
    p = _prep(conf, floor, clamp)
    return -sum(math.log(pi) if yi >= 0.5 else math.log(1.0 - pi) for pi, yi in zip(p, y)) / len(p)


def fit_temperature(
    conf: Sequence[float], y: Sequence[float], *, floor: float = 1e-12, clamp: float = 0.0,
    bounds: tuple[float, float] = T_BOUNDS,
) -> dict[str, Any]:
    """Refit a single temperature on logit(confidence) by log-grid + golden refinement."""
    if not conf:
        return {"T": None, "nll": None, "on_bound": False, "bounds": list(bounds)}
    z = [_logit(p) for p in _prep(conf, floor, clamp)]

    def loss(t: float) -> float:
        tot = 0.0
        for zi, yi in zip(z, y):
            s = zi / t
            # stable -log sigmoid
            tot += math.log1p(math.exp(-s)) if yi >= 0.5 else math.log1p(math.exp(s))
        return tot / len(z)

    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    grid = [lo + (hi - lo) * k / 200 for k in range(201)]
    k_best = min(range(201), key=lambda k: loss(math.exp(grid[k])))
    a = grid[max(k_best - 1, 0)]
    b = grid[min(k_best + 1, 200)]
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - phi * (b - a), a + phi * (b - a)
        if loss(math.exp(c)) < loss(math.exp(d)):
            b = d
        else:
            a = c
    t = math.exp((a + b) / 2)
    on_bound = t <= bounds[0] * 1.02 or t >= bounds[1] / 1.02
    return {"T": t, "nll": loss(t), "on_bound": on_bound, "bounds": list(bounds)}


def floor_clamp_sensitivity(
    conf: Sequence[float], y: Sequence[float],
    floors: Sequence[float] = DEFAULT_LOG_FLOORS, clamps: Sequence[float] = DEFAULT_CLAMPS,
) -> list[dict[str, Any]]:
    rows = []
    for floor in floors:
        for clamp in clamps:
            fit = fit_temperature(conf, y, floor=floor, clamp=clamp)
            rows.append({
                "log_floor": floor, "clamp": clamp, "nll": nll(conf, y, floor=floor, clamp=clamp),
                "refit_T": fit["T"], "refit_nll": fit["nll"], "T_on_bound": fit["on_bound"],
            })
    return rows


def endpoint_rates(conf: Sequence[float]) -> dict[str, Any]:
    n = len(conf)
    z = sum(1 for c in conf if float(c) == 0.0)
    o = sum(1 for c in conf if float(c) == 1.0)
    return {"n": n, "exact_0": z, "exact_1": o,
            "exact_0_rate": z / n if n else None, "exact_1_rate": o / n if n else None}


def temperature_oof(
    conf: Sequence[float], y: Sequence[float], groups: Sequence[Any], *, k: int = 5, seed: int = 0,
    floor: float = 1e-6, clamp: float = 1e-3,
) -> dict[str, Any]:
    """One in-distribution T, plus a group-disjoint K-fold out-of-fold check.

    Folds partition GROUPS (no group on both sides). Reports pooled out-of-fold NLL
    and ECE with the fold temperatures vs the raw confidences and vs the single T.
    """
    gi = _cluster_index(groups)
    keys = sorted(gi, key=str)
    random.Random(seed).shuffle(keys)
    k_eff = min(k, len(keys))
    single = fit_temperature(conf, y, floor=floor, clamp=clamp)
    if k_eff < 2:
        return {"single": single, "oof": None, "reason": "fewer than 2 groups"}
    fold_of = {key: i % k_eff for i, key in enumerate(keys)}
    oof_conf: list[float] = [0.0] * len(conf)
    fold_T = []
    for f in range(k_eff):
        tr = [i for key, idxs in gi.items() if fold_of[key] != f for i in idxs]
        te = [i for key, idxs in gi.items() if fold_of[key] == f for i in idxs]
        fit = fit_temperature([conf[i] for i in tr], [y[i] for i in tr], floor=floor, clamp=clamp)
        fold_T.append(fit["T"])
        for i in te:
            p = _prep([conf[i]], floor, clamp)[0]
            oof_conf[i] = 1.0 / (1.0 + math.exp(-_logit(p) / fit["T"]))
    return {
        "single": single, "k": k_eff, "fold_T": fold_T, "log_floor": floor, "clamp": clamp,
        "oof": {
            "nll_raw": nll(conf, y, floor=floor, clamp=clamp), "nll_calibrated": nll(oof_conf, y),
            "ece_raw": _ece(conf, y), "ece_calibrated": _ece(oof_conf, y),
        },
    }


def build_receipt_upgrade(
    rows: Sequence[Mapping[str, Any]], *, model_pin: Mapping[str, Any] | None,
    error_budget: float = 0.1, seed: int = 0, n_boot: int = 500,
) -> dict[str, Any]:
    """rows: dicts with ``kind``, ``confidence``, ``correct`` (bool), optional ``cluster``.

    Rows with ``correct is None`` (unlabeled) are ignored. Returns one block per kind.
    """
    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("correct") is not None:
            by_kind[str(r["kind"])].append(r)
    kinds: dict[str, Any] = {}
    for kind in sorted(by_kind):
        rs = by_kind[kind]
        conf = [float(r["confidence"]) for r in rs]
        y = [1.0 if r["correct"] else 0.0 for r in rs]
        clusters = [r.get("cluster", r.get("question_id", i)) for i, r in enumerate(rs)]
        kinds[kind] = {
            "n": len(rs),
            "ece": _ece(conf, y),
            "ece_noise_floor": ece_noise_floor(conf, seed=seed),
            "clustered_bootstrap": clustered_bootstrap_ece(conf, y, clusters, n_boot=n_boot, seed=seed),
            "selective": selective_prediction(conf, y, error_budget=error_budget),
            "floor_clamp_sensitivity": floor_clamp_sensitivity(conf, y),
            "endpoints": endpoint_rates(conf),
            "temperature_oof": temperature_oof(conf, y, clusters, seed=seed),
        }
    return {
        "schema": SCHEMA,
        "model_pin": dict(model_pin) if model_pin is not None else None,
        "model_pin_missing": model_pin is None,
        "seed": seed,
        "per_kind": kinds,
    }
