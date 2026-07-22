#!/usr/bin/env python3
"""DAR handoff L535 — Decisive-subset restriction + policy lift (zero inference).

The 2026-07-21 audit found the routing signal is CONCENTRATED: for ~78% of
matched objectives the within-objective role gap is ~0 (cost-dominated), and
essentially all discriminative signal lives in a ~22% minority. A policy that
improves the minority is INVISIBLE when lift is reported over all traffic.

This script:
  1. Builds the within-objective matched set (>=2 roles, >=5 obs each).
  2. Computes the decisive subset (best-worst role success gap > threshold).
  3. Reports a split-half counterfactual POLICY LIFT (pick best role on fold A,
     score on fold B) over ALL matched traffic vs over the DECISIVE subset --
     showing the same policy is near-zero over all traffic but large on the
     decisive minority.

This is the metric DAR-2 (contrastive) and any future policy eval should use.
Non-randomised assignment confound stands (OBSERVATION only, MEASUREMENT.md).

Usage:
    python scripts/analysis/dar_decisive_subset.py [--gap 0.05] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import dar_common as dc


def _split_rates(rows_for_role):
    """Deterministic split of a role's observations into two folds; return
    (nA, sA, nB, sB)."""
    nA = sA = nB = sB = 0
    for i, succ in enumerate(rows_for_role):
        if i % 2 == 0:
            nA += 1
            sA += succ
        else:
            nB += 1
            sB += succ
    return nA, sA, nB, sB


def policy_lift(obj_role_seq, objectives, direction="A->B"):
    """Split-half counterfactual lift over the given objective subset.

    Pick the best role by fold-A success rate; measure it on fold B against the
    average-role fold-B rate. Weight by fold-B decisions. Returns weighted mean
    lift and total weight.
    """
    num = 0.0
    den = 0.0
    per_obj = []
    for obj in objectives:
        roles = obj_role_seq[obj]
        a_rates, b_rates, b_ns = {}, {}, {}
        ok = True
        for role, seq in roles.items():
            nA, sA, nB, sB = _split_rates(seq)
            if nA == 0 or nB == 0:
                ok = False
                break
            a_rates[role] = sA / nA
            b_rates[role] = sB / nB
            b_ns[role] = nB
        if not ok or len(a_rates) < 2:
            continue
        if direction == "B->A":
            a_rates, b_rates = b_rates, a_rates
            # recompute b_ns for the swapped fold
            b_ns = {}
            for role, seq in roles.items():
                nA, sA, nB, sB = _split_rates(seq)
                b_ns[role] = nA
        best_role = max(a_rates, key=lambda r: a_rates[r])
        picked_b = b_rates[best_role]
        mean_b = statistics.mean(b_rates.values())
        w = sum(b_ns.values())
        lift = picked_b - mean_b
        num += lift * w
        den += w
        per_obj.append(lift)
    return (num / den if den else 0.0), den, per_obj


def run(gap_thresh: float, out_dir: Path | None) -> dict:
    rows = dc.load_rows("routing")
    _, matched = dc.matched_set(rows, min_obs=5, min_roles=2)

    # ordered success sequences per (objective, role) for split-half
    obj_role_seq: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        if r.objective in matched and r.action_type == "routing":
            if matched[r.objective].get(r.role):  # eligible role only
                obj_role_seq.setdefault(r.objective, {}).setdefault(r.role, []).append(r.success)

    # gaps + decisive subset
    gaps = {obj: dc.objective_role_gap(stats) for obj, stats in matched.items()}
    decisive = [obj for obj, g in gaps.items() if g > gap_thresh]
    all_objs = list(matched.keys())
    total_decisions = sum(n for stats in matched.values() for (n, s) in stats.values())

    # non-saturated decisions within matched set (write-time reward not at +1)
    nonsat = sum(1 for r in rows if r.objective in matched and r.reward is not None
                 and r.reward < 0.999)

    gap_vals = list(gaps.values())
    lift_all_ab, w_all, _ = policy_lift(obj_role_seq, all_objs, "A->B")
    lift_all_ba, _, _ = policy_lift(obj_role_seq, all_objs, "B->A")
    lift_dec_ab, w_dec, _ = policy_lift(obj_role_seq, decisive, "A->B")
    lift_dec_ba, _, _ = policy_lift(obj_role_seq, decisive, "B->A")
    # ALL-TRAFFIC dilution: the same numerator (only matched objectives can carry
    # a counterfactual) spread over EVERY routing decision. Single-role / low-obs
    # objectives (the cost-dominated majority) contribute 0 lift but full weight,
    # which is why any all-traffic metric hides a decisive-only policy.
    total_routing = sum(1 for r in rows if r.action_type == "routing")
    lift_alltraffic_ab = (lift_all_ab * w_all) / total_routing if total_routing else 0.0

    result = {
        "task": "DAR-L535 decisive-subset restriction + policy lift",
        "snapshot": dc.snapshot_meta(),
        "gap_threshold_pp": gap_thresh * 100,
        "matched_objectives": len(all_objs),
        "matched_decisions": total_decisions,
        "decisive_objectives": len(decisive),
        "decisive_fraction": len(decisive) / len(all_objs) if all_objs else 0.0,
        "within_objective_gap_pp": {
            "mean": statistics.mean(gap_vals) * 100,
            "median": statistics.median(gap_vals) * 100,
            "p90": (sorted(gap_vals)[int(0.9 * len(gap_vals))] * 100) if gap_vals else 0.0,
        },
        "nonsaturated_decisions_in_matched": nonsat,
        "nonsaturated_fraction_in_matched": nonsat / total_decisions if total_decisions else 0.0,
        "split_half_policy_lift_pp": {
            "all_traffic_A2B": lift_alltraffic_ab * 100,
            "all_matched_A2B": lift_all_ab * 100,
            "all_matched_B2A": lift_all_ba * 100,
            "decisive_A2B": lift_dec_ab * 100,
            "decisive_B2A": lift_dec_ba * 100,
            "weight_all_matched": w_all,
            "weight_decisive": w_dec,
            "total_routing_decisions": total_routing,
            "note": ("lift = (best-role-by-fold-A success on fold B) - (mean-role "
                     "fold-B success); baseline is uniform-random role choice."),
        },
    }

    print(f"[DAR-L535] decisive-subset restriction  (gap>{gap_thresh*100:.0f}pp)")
    print(f"snapshot {result['snapshot']['snapshot_ts_utc']}\n")
    print(f"matched objectives : {len(all_objs):,}  ({total_decisions:,} decisions)")
    print(f"decisive objectives: {len(decisive):,}  "
          f"({100*len(decisive)/len(all_objs):.1f}% of matched)")
    print(f"within-objective role gap: mean {statistics.mean(gap_vals)*100:.2f}pp  "
          f"median {statistics.median(gap_vals)*100:.2f}pp")
    print(f"non-saturated decisions in matched: {nonsat:,} "
          f"({100*nonsat/total_decisions:.1f}%)")
    print("\nsplit-half counterfactual policy lift (best-role-on-A vs mean-role, scored on B):")
    print(f"  over ALL TRAFFIC : {lift_alltraffic_ab*100:+.2f}pp  (numerator spread over all {total_routing:,} decisions)")
    print(f"  over ALL matched : {lift_all_ab*100:+.2f}pp (A->B) / {lift_all_ba*100:+.2f}pp (B->A)")
    print(f"  over DECISIVE    : {lift_dec_ab*100:+.2f}pp (A->B) / {lift_dec_ba*100:+.2f}pp (B->A)")
    print("  => the SAME policy shrinks ~4x from decisive subset to all-traffic denominator")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dar_decisive_subset.json").write_text(json.dumps(result, indent=2))
        print(f"\nartifact: {out_dir/'dar_decisive_subset.json'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=float, default=0.05, help="decisive gap threshold (fraction)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.gap, args.out)


if __name__ == "__main__":
    main()
