#!/usr/bin/env python3
"""DAR handoff L535 — Decisive-subset restriction + policy lift (zero inference).

The 2026-07-21 audit found the routing signal is CONCENTRATED: for ~78% of
matched objectives the within-objective role gap is ~0 (cost-dominated), and
essentially all discriminative signal lives in a ~22% minority. A policy that
improves the minority is INVISIBLE when lift is reported over all traffic.

This script:
  1. Builds the within-objective matched set (>=2 roles, >=5 obs each).
  2. Splits every (objective, role) sample into two disjoint seeded folds.
  3. Selects the decisive subset AND the best role on the selection fold only
     (gap > threshold AND an exact binomial null), then scores policy lift on
     the other fold (DAR-SPLIT-1: the earlier version selected on all
     observations, which leaked selection noise into the lift).

This is the metric DAR-2 (contrastive) and any future policy eval should use.
Non-randomised assignment confound stands (OBSERVATION only, MEASUREMENT.md).

Usage:
    python scripts/analysis/dar_decisive_subset.py [--gap 0.05] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

try:  # run as a script (scripts/analysis on sys.path) or imported as a module
    import dar_common as dc
except ImportError:  # pragma: no cover
    from scripts.analysis import dar_common as dc

# Metric direction: per-observation outcome is success in {0,1}; HIGHER success
# rate is better. gap = best-role rate minus worst-role rate (>=0); lift =
# (success of the role picked on the selection fold, measured on the held-out
# fold) minus (mean role success on the held-out fold); higher lift is better.
SEED = 20261109      # recorded: seeds the fold assignment and the bootstrap
ALPHA = 0.05         # per-objective selection level (Bonferroni over roles)
BOOT = 2000


def binom_sf(k: int, n: int, p: float) -> float:
    """Exact P(X >= k), X~Binomial(n,p)."""
    if n <= 0 or k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * lp + (n - i) * lq for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def split_folds(obj_role_seq, seed: int = SEED):
    """Seeded random split of each (objective, role) observation list into two
    DISJOINT folds. Returns {obj: {role: (A_list, B_list)}}."""
    rng = random.Random(seed)
    out = {}
    for obj in sorted(obj_role_seq):
        for role in sorted(obj_role_seq[obj]):
            seq = list(obj_role_seq[obj][role])
            idx = list(range(len(seq)))
            rng.shuffle(idx)
            h = len(idx) // 2
            A = [seq[i] for i in idx[:h]]
            B = [seq[i] for i in idx[h:]]
            out.setdefault(obj, {})[role] = (A, B)
    return out


def _rate(xs):
    return sum(xs) / len(xs) if xs else 0.0


def select_on_fold(folds_obj, sel: int, gap_thresh: float, alpha: float = ALPHA):
    """Decide decisiveness from the SELECTION fold only (sel=0 -> A, 1 -> B).

    Decisive iff gap > gap_thresh AND an exact binomial null rejects: under H0
    the best role's selection-fold successes ~ Binomial(n_best, p0), p0 = pooled
    success rate of the OTHER roles; one-sided, Bonferroni over the roles.
    Returns (decisive, best_role) or (False, None) if not testable.
    """
    roles = {r: f[sel] for r, f in folds_obj.items() if f[sel] and folds_obj[r][1 - sel]}
    if len(roles) < 2:
        return False, None
    rates = {r: _rate(x) for r, x in roles.items()}
    best = max(sorted(rates), key=lambda r: rates[r])
    gap = rates[best] - min(rates.values())
    if gap <= gap_thresh:
        return False, best
    others = [v for r, x in roles.items() if r != best for v in x]
    p0 = _rate(others)
    p = binom_sf(sum(roles[best]), len(roles[best]), p0) * len(roles)
    return p < alpha, best


def split_policy_lift(obj_role_seq, gap_thresh: float = 0.05, sel: int = 0,
                      seed: int = SEED, alpha: float = ALPHA, folds=None):
    """Leakage-free lift: decisiveness AND best role chosen on fold `sel`;
    lift scored on the other fold only. Also scores the ALL-matched set (no
    decisive filter) on the held-out fold for comparison."""
    folds = folds or split_folds(obj_role_seq, seed)
    ev = 1 - sel
    dec, allm = [], []   # (lift, weight)
    n_tested = 0
    for obj, fo in folds.items():
        roles_ev = {r: f[ev] for r, f in fo.items() if f[ev] and f[sel]}
        if len(roles_ev) < 2:
            continue
        n_tested += 1
        is_dec, best = select_on_fold(fo, sel, gap_thresh, alpha)
        rates_ev = {r: _rate(x) for r, x in roles_ev.items()}
        lift = rates_ev[best] - statistics.mean(rates_ev.values())
        w = sum(len(x) for x in roles_ev.values())
        allm.append((lift, w))
        if is_dec:
            dec.append((lift, w))
    return {"n_tested": n_tested, "decisive": dec, "all": allm}


def _wmean(items):
    den = sum(w for _, w in items)
    return sum(l * w for l, w in items) / den if den else 0.0


def boot_ci(items, seed: int = SEED, n: int = BOOT):
    """Percentile 95% CI of the weighted-mean lift, bootstrapping OBJECTIVES."""
    if not items:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(items)
    vals = sorted(_wmean([items[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return (vals[int(0.025 * n)], vals[int(0.975 * n) - 1])


def selection_null(res, alpha: float = ALPHA):
    """Is the number of selected objectives more than the per-objective alpha
    false-positive rate would give by chance? Exact binomial, one-sided."""
    n, k = res["n_tested"], len(res["decisive"])
    return {"n_tested": n, "n_selected": k, "expected_by_chance": n * alpha,
            "p_selected_exceeds_null": binom_sf(k, n, alpha)}


def summarize(res):
    lo, hi = boot_ci(res["decisive"])
    alo, ahi = boot_ci(res["all"])
    return {
        "decisive_lift_pp": _wmean(res["decisive"]) * 100,
        "decisive_ci95_pp": [lo * 100, hi * 100],
        "decisive_n_objectives": len(res["decisive"]),
        "decisive_weight": sum(w for _, w in res["decisive"]),
        "all_matched_lift_pp": _wmean(res["all"]) * 100,
        "all_matched_ci95_pp": [alo * 100, ahi * 100],
        "all_matched_n_objectives": len(res["all"]),
        "all_matched_weight": sum(w for _, w in res["all"]),
        "selection_null": selection_null(res),
    }


def run(gap_thresh: float, out_dir: Path | None) -> dict:
    rows = dc.load_rows("routing")
    _, matched = dc.matched_set(rows, min_obs=5, min_roles=2)

    obj_role_seq: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        if r.objective in matched and r.action_type == "routing":
            if matched[r.objective].get(r.role):
                obj_role_seq.setdefault(r.objective, {}).setdefault(r.role, []).append(r.success)

    total_routing = sum(1 for r in rows if r.action_type == "routing")
    folds = split_folds(obj_role_seq, SEED)
    a2b = summarize(split_policy_lift(obj_role_seq, gap_thresh, 0, folds=folds))
    b2a = summarize(split_policy_lift(obj_role_seq, gap_thresh, 1, folds=folds))
    # legacy (leaky) all-observation gap count, kept only for old-vs-new comparison
    gaps = {o: dc.objective_role_gap(s) for o, s in matched.items()}
    legacy_decisive = sum(1 for g in gaps.values() if g > gap_thresh)

    result = {
        "task": "DAR-SPLIT-1 split-half decisive subset + binomial null",
        "snapshot": dc.snapshot_meta(),
        "metric_direction": "success rate, higher is better; lift higher is better",
        "seed": SEED, "alpha": ALPHA, "bootstrap": BOOT,
        "gap_threshold_pp": gap_thresh * 100,
        "matched_objectives": len(matched),
        "legacy_all_obs_decisive_objectives": legacy_decisive,
        "total_routing_decisions": total_routing,
        "A_select_B_eval": a2b,
        "B_select_A_eval": b2a,
        "note": ("decisive subset and best role chosen on one fold, lift scored on "
                 "the disjoint other fold; selection needs gap>threshold AND exact "
                 "binomial null (Bonferroni over roles) at alpha."),
    }
    print(f"[DAR-SPLIT-1] seed={SEED} gap>{gap_thresh*100:.0f}pp alpha={ALPHA}")
    print(f"snapshot {result['snapshot']['snapshot_ts_utc']}  matched objectives {len(matched):,}")
    for name, s in (("A->B", a2b), ("B->A", b2a)):
        print(f"{name}: decisive lift {s['decisive_lift_pp']:+.2f}pp "
              f"CI95 [{s['decisive_ci95_pp'][0]:+.2f},{s['decisive_ci95_pp'][1]:+.2f}] "
              f"n_obj={s['decisive_n_objectives']} w={s['decisive_weight']:,}; "
              f"all-matched {s['all_matched_lift_pp']:+.2f}pp; null {s['selection_null']}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dar_decisive_subset_split.json").write_text(json.dumps(result, indent=2))
        print(f"\nartifact: {out_dir/'dar_decisive_subset_split.json'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=float, default=0.05, help="decisive gap threshold (fraction)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.gap, args.out)


if __name__ == "__main__":
    main()
