#!/usr/bin/env python3
"""DAR handoff L491 — Write-path / update_count audit (zero inference, read-only).

Answers two questions the handoff raises:
  1. Is `update_count = 0` on ~99.7% of rows an intentional append-only replay
     buffer, or a dedup/update defect? Evidence: if the SAME (objective, role)
     appears as many separate update_count=0 rows, the store is pure-appending
     instead of routing writes through update_q_value() -- i.e. the TD apparatus
     DAR-1/2/3 build on is effectively dead code in production.
  2. Is DAR-2 (contrastive Q) live-EFFECTIVE, not merely live-ON? DAR-2 skips
     memories still at default Q and only fires when >=1 alternative role for the
     same objective carries a LEARNED (update_count>0) Q. This bounds how often
     DAR-2 could possibly have fired.

All outputs are OBSERVATIONS (MEASUREMENT.md).

Usage:
    python scripts/analysis/dar_write_path_audit.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import dar_common as dc


def run(out_dir: Path | None) -> dict:
    rows = dc.load_rows("routing")
    n = len(rows)

    uc_dist = Counter(r.update_count for r in rows)
    n_uc0 = uc_dist[0]

    # (1) dedup / append-only evidence -----------------------------------
    pair_counts: dict[tuple, int] = defaultdict(int)  # (objective, role) -> rows
    for r in rows:
        if r.update_count == 0 and r.objective is not None:
            pair_counts[(r.objective, r.role)] += 1
    total_uc0_pairable = sum(pair_counts.values())
    distinct_pairs = len(pair_counts)
    dup_pairs = sum(1 for c in pair_counts.values() if c > 1)
    max_pair = max(pair_counts.values()) if pair_counts else 0
    # how many rows are "redundant copies" that a TD-update store would have
    # collapsed into an update instead of an insert
    redundant_rows = total_uc0_pairable - distinct_pairs
    dup_factor = total_uc0_pairable / distinct_pairs if distinct_pairs else 0.0

    # (2) DAR-2 effectiveness --------------------------------------------
    # rows with a learned (moved) Q
    learned = [r for r in rows if r.update_count > 0]
    n_learned = len(learned)
    # objectives that have >=1 role carrying a learned Q (DAR-2's precondition:
    # an alternative with a non-default learned Q to contrast against)
    obj_has_learned_role = defaultdict(set)  # objective -> set(roles with learned Q)
    obj_all_roles = defaultdict(set)
    for r in rows:
        if r.objective is None:
            continue
        obj_all_roles[r.objective].add(r.role)
        if r.update_count > 0:
            obj_has_learned_role[r.objective].add(r.role)
    # a routing write can benefit from DAR-2 only if, at write time, another role
    # for the same objective already has a learned Q. Upper bound: rows whose
    # objective has >=1 role with a learned Q AND >=2 distinct roles overall.
    dar2_eligible_rows = 0
    for r in rows:
        if r.objective is None:
            continue
        roles_learned = obj_has_learned_role.get(r.objective, set())
        if roles_learned and len(obj_all_roles[r.objective]) >= 2:
            # exclude the trivial case where the only learned role is r itself and
            # no alternative exists
            alt = roles_learned - {r.role}
            if alt or (r.role not in roles_learned):
                dar2_eligible_rows += 1

    intentional = dup_factor > 5  # thousands of identical pairs => pure append
    verdict = ("APPEND-ONLY buffer (store() inserts a fresh row per observation; "
               "update_q_value / TD path is effectively bypassed in production)"
               if intentional else
               "plausibly dedup-updating (low duplication)")

    result = {
        "task": "DAR-L491 write-path / update_count audit",
        "snapshot": dc.snapshot_meta(),
        "n_routing": n,
        "update_count_distribution": dict(sorted(uc_dist.items())[:12]),
        "update_count_0_fraction": n_uc0 / n,
        "dedup_evidence": {
            "update_count_0_rows": total_uc0_pairable,
            "distinct_objective_role_pairs": distinct_pairs,
            "duplicated_pairs": dup_pairs,
            "max_rows_for_one_pair": max_pair,
            "redundant_rows_a_TD_store_would_have_collapsed": redundant_rows,
            "duplication_factor": dup_factor,
            "verdict": verdict,
        },
        "dar2_effectiveness": {
            "rows_with_learned_Q_update_count_gt_0": n_learned,
            "learned_fraction": n_learned / n,
            "rows_where_DAR2_could_fire_upper_bound": dar2_eligible_rows,
            "dar2_fireable_fraction_upper_bound": dar2_eligible_rows / n,
            "note": ("DAR-2 skips default-Q memories and needs an alternative role "
                     "with a learned Q; this is the ceiling on its live effect."),
        },
    }

    print(f"[DAR-L491] write-path / update_count audit  ({n:,} routing rows)")
    print(f"snapshot {result['snapshot']['snapshot_ts_utc']}\n")
    print("update_count distribution:")
    for uc, c in sorted(uc_dist.items())[:10]:
        print(f"  update_count={uc:<3} {c:>9,}  {100*c/n:6.3f}%")
    print(f"\nupdate_count=0 fraction: {100*n_uc0/n:.3f}%")
    print("\ndedup evidence (update_count=0 rows):")
    print(f"  rows                       {total_uc0_pairable:>9,}")
    print(f"  distinct (objective,role)  {distinct_pairs:>9,}")
    print(f"  duplicated pairs           {dup_pairs:>9,}")
    print(f"  max rows for one pair      {max_pair:>9,}")
    print(f"  duplication factor         {dup_factor:>9.1f}x")
    print(f"  => {verdict}")
    print("\nDAR-2 effectiveness ceiling:")
    print(f"  rows with learned Q (uc>0) {n_learned:>9,}  ({100*n_learned/n:.3f}%)")
    print(f"  rows DAR-2 could fire on   {dar2_eligible_rows:>9,}  "
          f"({100*dar2_eligible_rows/n:.3f}%)  [upper bound]")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dar_write_path_audit.json").write_text(json.dumps(result, indent=2))
        print(f"\nartifact: {out_dir/'dar_write_path_audit.json'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    run(ap.parse_args().out)


if __name__ == "__main__":
    main()
