"""One-sided degradation statistics for paired per-item evals (EV-DEGR1 / TQ3-PWR1 / NEW-14).

ADDITIVE sibling of :mod:`scripts.autopilot.paired_stats`. It never touches
``mcnemar_verdict`` / ``McNemarResult`` and exports NO P-PAIRED verdict string: the
two-sided P-PAIRED McNemar stays the preference/decision surface and this module
is a *diagnostic* printed beside it.

Re-implemented from the method description of arXiv 2602.10144 (intake-2484#00,
intake-2484#03): exact one-sided McNemar (binomial on discordant pairs), pooled /
Fisher / max-drop aggregation across suites, any-of-three flag, and SNR power
arithmetic (Eq 11). Not derived from the CC BY-NC upstream repo (intake-2491#record).

Orientation: ``b`` = baseline correct / candidate wrong (degradation evidence),
``c`` = baseline wrong / candidate correct. H0: P(b | discordant) = 1/2; the
alternative is degradation (b > c). Monte Carlo uses a fixed seed so results are
reproducible. A suite with b + c == 0 contributes p = 1.0 and z = 0 (never 0.0,
never an exception).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats as _sps

from scripts.autopilot.paired_stats import QuestionOutcome

Z_ALPHA05_ONE_SIDED = 1.6448536269514722
Z_POWER80 = 0.8416212335729143
# 1.645 + 0.842 = 2.486 (paper constant for one-sided alpha 0.05, 80% power)
Z_SUM_05_80 = 2.486
DEFAULT_SEED = 20261022
DEFAULT_SIMS = 20000
DEFAULT_MARGIN = 0.02  # declared 2 pp non-inferiority margin (NEW-14)


def one_sided_degradation_p(b: int, c: int) -> float:
    """Exact one-sided binomial p for degradation: P(X >= b | n=b+c, p=.5). n==0 -> 1.0."""
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return float(_sps.binomtest(b, n, 0.5, alternative="greater").pvalue)


def degradation_verdict(b: int, c: int, *, alpha: float = 0.05) -> dict[str, Any]:
    """Diagnostic dict: n_discordant, direction, one-sided exact p. Not a P-PAIRED verdict."""
    n = b + c
    direction = "baseline_better" if b > c else "candidate_better" if c > b else "tie"
    p = one_sided_degradation_p(b, c)
    return {
        "method": "exact_one_sided_mcnemar",
        "b_baseline_correct_candidate_wrong": b,
        "c_baseline_wrong_candidate_correct": c,
        "n_discordant": n,
        "direction": direction,
        "p_one_sided": p,
        "alpha": alpha,
        "degraded": bool(p < alpha),
    }


def power_required_n(p_flip: float, delta: float) -> float:
    """Required N (items) = p_flip * (2.486/delta)^2, one-sided alpha .05, 80% power."""
    if not (0 < p_flip <= 1) or delta <= 0:
        raise ValueError("need 0 < p_flip <= 1 and delta > 0")
    return p_flip * (Z_SUM_05_80 / delta) ** 2


def power_at_n(n: int, p_flip: float, delta: float, *, alpha: float = 0.05) -> float:
    """Power = Phi(sqrt(N/p_flip)*delta - z_alpha) (SNR form, normal approximation)."""
    if n <= 0 or not (0 < p_flip <= 1) or delta < 0:
        raise ValueError("need n > 0, 0 < p_flip <= 1, delta >= 0")
    z_a = float(_sps.norm.isf(alpha))
    return float(_sps.norm.cdf(math.sqrt(n / p_flip) * delta - z_a))


def min_detectable_delta(n: int, p_flip: float) -> float:
    """Smallest drop detectable at 80% power one-sided alpha .05 with n items."""
    if n <= 0 or not (0 < p_flip <= 1):
        raise ValueError("need n > 0 and 0 < p_flip <= 1")
    return Z_SUM_05_80 * math.sqrt(p_flip / n)


def required_n_table(p_flips: list[float], deltas: list[float]) -> list[dict[str, Any]]:
    """Rows of {p_flip, delta, required_n(ceil)} for documentation tables."""
    return [
        {"p_flip": p, "delta": d, "required_n": math.ceil(power_required_n(p, d))}
        for p in p_flips
        for d in deltas
    ]


def margin_sizing(
    n: int | None, p_flip: float | None, margin: float = DEFAULT_MARGIN
) -> dict[str, Any]:
    """Size a gate against a declared margin (NEW-14). Unknown inputs stay None."""
    out: dict[str, Any] = {"margin": margin, "p_flip": p_flip, "n": n}
    out["required_n"] = math.ceil(power_required_n(p_flip, margin)) if p_flip is not None else None
    out["power_at_n"] = (
        power_at_n(n, p_flip, margin) if n is not None and p_flip is not None else None
    )
    out["margin_supported_by_n"] = (
        min_detectable_delta(n, p_flip) if n is not None and p_flip is not None else None
    )
    return out


def count_discordant_by_suite(
    baseline: Mapping[str, QuestionOutcome],
    candidate: Mapping[str, QuestionOutcome],
) -> dict[str, tuple[int, int]]:
    """Per-suite (b, c) over qids present in both vectors (shape of paired_stats.trial_vectors)."""
    out: dict[str, list[int]] = {}
    for qid in sorted(set(baseline) & set(candidate)):
        a, o = baseline[qid], candidate[qid]
        bc = out.setdefault(a.suite, [0, 0])
        if a.correct and not o.correct:
            bc[0] += 1
        elif o.correct and not a.correct:
            bc[1] += 1
    return {s: (v[0], v[1]) for s, v in out.items()}


def _z(b: int, c: int) -> float:
    n = b + c
    return 0.0 if n == 0 else (b - c) / math.sqrt(n)


def aggregate_degradation(
    suites: Mapping[str, tuple[int, int]],
    *,
    alpha: float = 0.05,
    sims: int = DEFAULT_SIMS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Pooled / Fisher / max-drop tests plus the any-of-three flag (diagnostic only).

    ``suites`` maps suite -> (b, c). Zero-discordant suites contribute p=1.0 and z=0.
    """
    per = {
        s: {
            "b": b,
            "c": c,
            "n_discordant": b + c,
            "p_one_sided": one_sided_degradation_p(b, c),
            "z": _z(b, c),
        }
        for s, (b, c) in sorted(suites.items())
    }
    tb = sum(b for b, _ in suites.values())
    tc = sum(c for _, c in suites.values())
    p_pooled = one_sided_degradation_p(tb, tc)

    ps = [max(v["p_one_sided"], 1e-300) for v in per.values()]
    if ps:
        fisher_stat = -2.0 * sum(math.log(p) for p in ps)
        p_fisher = float(_sps.chi2.sf(fisher_stat, 2 * len(ps)))
    else:
        p_fisher = 1.0

    # max-drop: null = independent Binomial(n_k, .5) per suite; statistic = max per-suite z.
    ns = [b + c for b, c in suites.values()]
    obs = max((v["z"] for v in per.values()), default=0.0)
    if not ns or all(n == 0 for n in ns):
        p_max = 1.0
    else:
        rng = np.random.default_rng(seed)
        zmax = np.full(sims, -np.inf)
        for n in ns:
            if n == 0:
                zk = np.zeros(sims)
            else:
                bk = rng.binomial(n, 0.5, size=sims)
                zk = (2 * bk - n) / math.sqrt(n)
            zmax = np.maximum(zmax, zk)
        p_max = float((1 + np.sum(zmax >= obs - 1e-12)) / (1 + sims))
        p_max = min(1.0, p_max)

    flags = {"pooled": p_pooled < alpha, "fisher": p_fisher < alpha, "max_drop": p_max < alpha}
    return {
        "method": "any_of_three_one_sided",
        "alpha": alpha,
        "seed": seed,
        "sims": sims,
        "n_suites": len(per),
        "b_total": tb,
        "c_total": tc,
        "n_discordant": tb + tc,
        "p_pooled": p_pooled,
        "p_fisher": p_fisher,
        "p_max_drop": p_max,
        "flags": flags,
        "any_degraded": any(flags.values()),
        "per_suite": per,
    }


def degradation_from_vectors(
    baseline: Mapping[str, QuestionOutcome],
    candidate: Mapping[str, QuestionOutcome],
    **kw: Any,
) -> dict[str, Any]:
    """Convenience: paired_stats-style vectors -> aggregate diagnostic (emit beside mcnemar_verdict)."""
    return aggregate_degradation(count_discordant_by_suite(baseline, candidate), **kw)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``verdict --b B --c C`` prints the one-sided diagnostic; ``size`` prints required N / power."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--b", type=int, required=True, help="baseline correct / candidate wrong")
    v.add_argument("--c", type=int, required=True, help="baseline wrong / candidate correct")
    v.add_argument("--alpha", type=float, default=0.05)
    s = sub.add_parser("size")
    s.add_argument("--p-flip", type=float, required=True, help="measured same-model flip floor")
    s.add_argument("--n", type=int, default=None, help="available paired items")
    s.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    a = ap.parse_args(argv)
    if a.cmd == "verdict":
        print(json.dumps(degradation_verdict(a.b, a.c, alpha=a.alpha), indent=2))
    else:
        print(json.dumps(margin_sizing(a.n, a.p_flip, a.margin), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
