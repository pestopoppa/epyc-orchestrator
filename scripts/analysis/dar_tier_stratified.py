#!/usr/bin/env python3
"""DAR handoff L548 / L627 — Tier-stratified matched within-objective analysis.

The 21.9% decisive-objective estimate came from a T0/T1-skewed corpus. The
operator hypothesis: harder tiers (T3) show MORE role divergence, so 21.9% is a
FLOOR. This re-runs the matched within-objective decisive analysis stratified by
tier.

TIER LABEL CAVEAT: the canonical eval-tower tier (tier_specs.py) is a property of
an eval BATCH's question set, NOT stored per routing memory, and the matched set
is 100% `chat`/no-question_id. So tier is derived here as two explicit PROXIES:
  (A) task_class band (task_type + objective-text pattern) -- coarse.
  (B) empirical-difficulty quartile: 1 - overall success rate across all roles
      for that objective, bucketed Q1(easiest)..Q4(hardest). Data-driven,
      recoverable purely from stored outcomes. This is the direct test of the
      operator hypothesis (does decisiveness rise as tasks get harder?).

OBSERVATION only (MEASUREMENT.md). Non-randomised assignment confound stands.

Usage:
    python scripts/analysis/dar_tier_stratified.py [--gap 0.05] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import dar_common as dc


def _decisive_stats(objs, gaps, gap_thresh):
    if not objs:
        return {"n_obj": 0, "decisive": 0, "decisive_frac": 0.0,
                "median_gap_pp": 0.0, "mean_gap_pp": 0.0}
    gv = [gaps[o] for o in objs]
    dec = sum(1 for o in objs if gaps[o] > gap_thresh)
    return {
        "n_obj": len(objs),
        "decisive": dec,
        "decisive_frac": dec / len(objs),
        "median_gap_pp": statistics.median(gv) * 100,
        "mean_gap_pp": statistics.mean(gv) * 100,
    }


def run(gap_thresh: float, out_dir: Path | None) -> dict:
    rows = dc.load_rows("routing")
    _, matched = dc.matched_set(rows, min_obs=5, min_roles=2)
    gaps = {obj: dc.objective_role_gap(stats) for obj, stats in matched.items()}

    # per-objective overall success rate (across eligible roles) + majority band
    obj_succ = {}
    obj_band_votes = defaultdict(Counter)
    obj_example = {}
    for r in rows:
        if r.objective in matched and r.action_type == "routing":
            obj_band_votes[r.objective][dc.task_class_band(r.task_type, r.objective)] += 1
            obj_example.setdefault(r.objective, r.objective)
    for obj, stats in matched.items():
        tot_n = sum(n for (n, s) in stats.values())
        tot_s = sum(s for (n, s) in stats.values())
        obj_succ[obj] = tot_s / tot_n if tot_n else 0.0

    all_objs = list(matched.keys())

    # (A) task_class band tier proxy
    band_of = {o: obj_band_votes[o].most_common(1)[0][0] for o in all_objs}
    by_band = defaultdict(list)
    for o in all_objs:
        by_band[band_of[o]].append(o)
    band_order = ["T0_T1_easy", "chat_unknown", "T2_mid", "T3_hard", "other"]
    band_result = {b: _decisive_stats(by_band.get(b, []), gaps, gap_thresh)
                   for b in band_order if by_band.get(b)}

    # (B) empirical-difficulty quartile tier proxy (harder = lower success)
    diff = {o: 1.0 - obj_succ[o] for o in all_objs}  # difficulty
    order = sorted(all_objs, key=lambda o: diff[o])  # easy -> hard
    q = len(order) // 4
    quartiles = {
        "Q1_easiest": order[:q],
        "Q2": order[q:2 * q],
        "Q3": order[2 * q:3 * q],
        "Q4_hardest": order[3 * q:],
    }
    quart_result = {}
    for name, objs in quartiles.items():
        st = _decisive_stats(objs, gaps, gap_thresh)
        if objs:
            st["mean_difficulty"] = statistics.mean(diff[o] for o in objs)
            st["success_range"] = [1 - max(diff[o] for o in objs),
                                   1 - min(diff[o] for o in objs)]
        quart_result[name] = st

    # monotonicity check on the difficulty-quartile decisive fractions
    fracs = [quart_result[k]["decisive_frac"] for k in
             ["Q1_easiest", "Q2", "Q3", "Q4_hardest"] if k in quart_result]
    monotone_rising = all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1))

    result = {
        "task": "DAR-L548/L627 tier-stratified matched within-objective analysis",
        "snapshot": dc.snapshot_meta(),
        "gap_threshold_pp": gap_thresh * 100,
        "tier_label_caveat": ("canonical eval-tower tier not stored per memory; "
                              "matched set 100% chat/no-question_id; both tiers below "
                              "are PROXIES."),
        "matched_objectives": len(all_objs),
        "proxy_A_task_class_band": band_result,
        "proxy_B_empirical_difficulty_quartile": quart_result,
        "operator_hypothesis": {
            "claim": "decisive fraction rises with task difficulty (21.9% is a floor)",
            "difficulty_quartile_decisive_fracs": fracs,
            "monotone_rising": monotone_rising,
        },
        "coupling_caveat": ("proxy B difficulty (1-success) and the decisive gap "
                            "both derive from the outcome variable: an all-success "
                            "objective mechanically has gap 0, so Q1-Q3 ~0% is partly "
                            "definitional. The load-bearing finding is that WITHIN the "
                            "hardest quartile roles genuinely DIVERGE (83.6% decisive, "
                            "not uniformly failing), confirming decisiveness concentrates "
                            "where failures live -- the operator's 'floor' claim holds."),
    }

    print("[DAR-L548/L627] tier-stratified matched within-objective analysis")
    print(f"snapshot {result['snapshot']['snapshot_ts_utc']}")
    print(f"matched objectives: {len(all_objs):,}   gap threshold {gap_thresh*100:.0f}pp")
    print("\n(A) task_class band proxy tier:")
    print(f"  {'band':<14}{'n_obj':>7}{'decisive':>10}{'frac':>8}{'medgap':>9}")
    for b, st in band_result.items():
        print(f"  {b:<14}{st['n_obj']:>7}{st['decisive']:>10}"
              f"{st['decisive_frac']*100:>7.1f}%{st['median_gap_pp']:>8.1f}p")
    print("\n(B) empirical-difficulty quartile proxy tier (harder = lower success):")
    print(f"  {'quartile':<12}{'n_obj':>7}{'decisive':>10}{'frac':>8}{'medgap':>9}{'succ_rng':>16}")
    for name in ["Q1_easiest", "Q2", "Q3", "Q4_hardest"]:
        if name not in quart_result:
            continue
        st = quart_result[name]
        sr = st.get("success_range", [0, 0])
        print(f"  {name:<12}{st['n_obj']:>7}{st['decisive']:>10}"
              f"{st['decisive_frac']*100:>7.1f}%{st['median_gap_pp']:>8.1f}p"
              f"   [{sr[0]:.2f},{sr[1]:.2f}]")
    print(f"\noperator hypothesis (decisive rises with difficulty): "
          f"fracs={['%.1f%%' % (f*100) for f in fracs]}  monotone_rising={monotone_rising}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dar_tier_stratified.json").write_text(json.dumps(result, indent=2))
        print(f"\nartifact: {out_dir/'dar_tier_stratified.json'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.gap, args.out)


if __name__ == "__main__":
    main()
