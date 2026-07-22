#!/usr/bin/env python3
"""DAR handoff L490 — Reward-saturation audit (zero inference, read-only snapshot).

Inverts `q = 0.5 + reward/2` on the `update_count = 0` rows and histograms the
recovered reward globally, by role, and by task_class (the task_class dimension
extends the 2026-07-21 role-only audit). Reports per-decision reward entropy.

Pre-registered decision flip (DAR L490): if per-decision reward entropy < 1 bit
AND role-conditional means differ < 2pp, close DAR-3/4/5 as
`not_pursued -- signal-bound`. This script reports both legs so the split
verdict is reproducible.

All outputs are OBSERVATIONS (MEASUREMENT.md); pre-fix instrument era.

Usage:
    python scripts/analysis/dar_reward_saturation_audit.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import dar_common as dc


def _stats(rewards: list[float]) -> dict:
    n = len(rewards)
    mean = sum(rewards) / n if n else 0.0
    var = sum((r - mean) ** 2 for r in rewards) / n if n else 0.0
    return {"n": n, "mean": mean, "sd": math.sqrt(var),
            "se": math.sqrt(var / n) if n else 0.0}


def run(out_dir: Path | None) -> dict:
    rows = dc.load_rows("routing")
    uc0 = [r for r in rows if r.update_count == 0 and r.reward is not None]
    rewards = [r.reward for r in uc0]

    # Histogram (0.1 bins)
    hist = defaultdict(int)
    for r in rewards:
        hist[round(r, 1)] += 1
    total = len(rewards)
    hist_sorted = sorted(hist.items(), key=lambda kv: -kv[1])

    ent_bin = dc.entropy(rewards, 0.1)
    sat = sum(1 for r in rewards if r >= 0.999)
    ent_binary = 0.0
    for p in (sat / total, 1 - sat / total):
        if p > 0:
            ent_binary -= p * math.log2(p)

    # By role
    by_role = defaultdict(list)
    for r in uc0:
        by_role[r.role].append(r.reward)
    role_means = {role: _stats(v) for role, v in by_role.items() if len(v) >= 50}
    role_sorted = sorted(role_means.items(), key=lambda kv: -kv[1]["mean"])
    # Decision metric: spread over the CANONICAL primary roles (n>=5000). The
    # low-n `:direct`/`plan_review:*` sub-actions have noisy means that inflate a
    # naive max-min spread, so they are excluded from the gate figure (but listed).
    primary = [s["mean"] for _, s in role_sorted if s["n"] >= 5000]
    role_spread = (max(primary) - min(primary)) if primary else 0.0

    # By task_class band (NEW dimension: L490 "by role and task_class")
    by_tc = defaultdict(list)
    for r in uc0:
        band = dc.task_class_band(r.task_type, r.objective)
        by_tc[band].append(r.reward)
    tc_means = {b: _stats(v) for b, v in by_tc.items() if len(v) >= 50}
    tc_sorted = sorted(tc_means.items(), key=lambda kv: -kv[1]["mean"])

    # Raw task_type (finer) for the eval-labelled minority
    by_tt = defaultdict(list)
    for r in uc0:
        if (r.task_type or "chat") != "chat":
            by_tt[r.task_type].append(r.reward)
    tt_means = {t: _stats(v) for t, v in by_tt.items() if len(v) >= 30}
    tt_sorted = sorted(tt_means.items(), key=lambda kv: -kv[1]["mean"])

    cond1 = ent_bin < 1.0
    cond2 = role_spread < 0.02
    verdict = ("CLOSE as signal-bound" if (cond1 and cond2)
               else "DOES NOT close as signal-bound (split verdict)")

    result = {
        "task": "DAR-L490 reward-saturation audit",
        "snapshot": dc.snapshot_meta(),
        "n_routing": len(rows),
        "n_update_count_0": total,
        "reward_histogram": [{"reward": r, "count": c, "share": c / total}
                             for r, c in hist_sorted[:12]],
        "entropy_bits_0p1": ent_bin,
        "entropy_bits_binary_r1_vs_else": ent_binary,
        "saturated_at_r1_share": sat / total,
        "role_means": {r: s for r, s in role_sorted},
        "role_spread_pp": role_spread * 100,
        "task_class_means": {b: s for b, s in tc_sorted},
        "task_type_means_nonchat": {t: s for t, s in tt_sorted},
        "decision_flip": {
            "cond1_entropy_lt_1bit": cond1,
            "cond2_role_spread_lt_2pp": cond2,
            "verdict": verdict,
        },
    }

    # ---- print ----
    print(f"[DAR-L490] reward-saturation audit  ({total:,} update_count=0 rows)")
    print(f"snapshot {result['snapshot']['snapshot_ts_utc']}\n")
    print("reward histogram (0.1 bin):")
    for r, c in hist_sorted[:8]:
        print(f"  r={r:+.1f}  {c:>9,}  {100*c/total:6.2f}%")
    print(f"\nper-decision entropy: {ent_bin:.4f} bits (0.1-bin) | "
          f"{ent_binary:.4f} bits (binary r==1 vs else)")
    print("\nreward by ROLE (n>=50):")
    for role, s in role_sorted:
        print(f"  {role:<22} n={s['n']:>8,}  mean={s['mean']:+.4f}  se={s['se']:.4f}")
    print(f"  role spread = {role_spread*100:.2f}pp")
    print("\nreward by TASK_CLASS band (proxy tier):")
    for b, s in tc_sorted:
        print(f"  {b:<14} n={s['n']:>8,}  mean={s['mean']:+.4f}  se={s['se']:.4f}")
    if tt_sorted:
        print("\nreward by raw task_type (non-chat eval traffic, n>=30):")
        for t, s in tt_sorted[:15]:
            print(f"  {t:<20} n={s['n']:>7,}  mean={s['mean']:+.4f}")
    print(f"\nDECISION FLIP: cond1(entropy<1bit)={cond1}  "
          f"cond2(role spread<2pp)={cond2}  =>  {verdict}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dar_reward_saturation_audit.json").write_text(
            json.dumps(result, indent=2))
        print(f"\nartifact: {out_dir/'dar_reward_saturation_audit.json'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    run(ap.parse_args().out)


if __name__ == "__main__":
    main()
