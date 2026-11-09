#!/usr/bin/env python3
"""AP-29b(a): replay-only comparison of lexicographic ranking vs the scalarized proxy.

Reads persisted autopilot journal rows (no inference, no Autopilot). Both rankings are computed
inside ONE comparability era: same ``metric_schema_version``, same ``objective_policy_live``
(the row-level mirror of the state-file ``pareto_objective_policy``) and same
``speed_metric_mode`` (speed is not comparable across modes).

Rankings (best first):
  proxy : quality - lam * max(0, 1 - speed / baseline_speed)          (lam=0.15)
  lexi  : (1) quality gate  quality >= gate_quality
          (2) reliability gate reliability >= gate_reliability
          (3) survivors sit on / are ordered by Pareto rank over (cost low, speed high);
              ties broken by quality desc.  Gate-failing rows rank after all survivors, by quality desc.
baseline_speed, gate_quality, gate_reliability = era medians of the first ``--seed-n`` scored rows.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ORCH = Path(__file__).resolve().parents[2]
DEFAULT_SHARDS = ("autopilot_journal.jsonl", "autopilot_journal_1.jsonl")


def load_rows(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ed = d.get("eval_details") or {}
            if not ed or d.get("bug_corrupted_by"):
                continue
            rows.append(
                {
                    "trial_id": d.get("trial_id"),
                    "ts": d.get("timestamp"),
                    "quality": float(d.get("quality") or 0.0),
                    "speed": float(d.get("speed") or 0.0),
                    "cost": float(d.get("cost") or 0.0),
                    "reliability": float(d.get("reliability") or 0.0),
                    "era": (
                        d.get("metric_schema_version"),
                        ed.get("objective_policy_live"),
                        ed.get("speed_metric_mode"),
                    ),
                }
            )
    rows.sort(key=lambda r: r["ts"] or "")
    return rows


def proxy_score(r, base_speed, lam):
    return r["quality"] - lam * max(0.0, 1.0 - r["speed"] / base_speed)


def pareto_ranks(items):
    """Non-dominated sorting on (cost low, speed high). Returns {id(item): rank}."""
    remaining, rank, out = list(items), 0, {}
    while remaining:
        front = [
            a
            for a in remaining
            if not any(
                b["cost"] <= a["cost"] and b["speed"] >= a["speed"]
                and (b["cost"] < a["cost"] or b["speed"] > a["speed"])
                for b in remaining
            )
        ]
        for a in front:
            out[id(a)] = rank
        remaining = [a for a in remaining if id(a) not in out]
        rank += 1
    return out


def lexi_order(rows, gate_q, gate_r):
    surv = [r for r in rows if r["quality"] >= gate_q and r["reliability"] >= gate_r]
    fail = [r for r in rows if not (r["quality"] >= gate_q and r["reliability"] >= gate_r)]
    pr = pareto_ranks(surv)
    surv.sort(key=lambda r: (pr[id(r)], -r["quality"], r["cost"], -r["speed"]))
    fail.sort(key=lambda r: -r["quality"])
    return surv + fail, len(surv)


def kendall_tau(a_order, b_order):
    pos = {id(r): i for i, r in enumerate(b_order)}
    seq = [pos[id(r)] for r in a_order]
    n, conc = len(seq), 0
    for i in range(n):
        for j in range(i + 1, n):
            conc += 1 if seq[i] < seq[j] else -1
    return conc / (n * (n - 1) / 2) if n > 1 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--journal-dir", type=Path, default=ORCH / "orchestration")
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--seed-n", type=int, default=5)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--era-policy", default="legacy_4d_v1")
    ap.add_argument("--era-speed-mode", default="aggregate_batch_tps")
    args = ap.parse_args(argv)

    rows = load_rows([args.journal_dir / s for s in DEFAULT_SHARDS])
    era = (1, args.era_policy, args.era_speed_mode)
    rows = [r for r in rows if r["era"] == era and r["speed"] > 0]
    if len(rows) < 20:
        print(json.dumps({"status": "insufficient_rows", "n": len(rows), "era": era}))
        return 2
    seed = rows[: args.seed_n]
    base_speed = statistics.median(r["speed"] for r in seed)
    gate_q = statistics.median(r["quality"] for r in seed)
    gate_r = statistics.median(r["reliability"] for r in seed)
    proxy_order = sorted(rows, key=lambda r: -proxy_score(r, base_speed, args.lam))
    lexi, n_surv = lexi_order(rows, gate_q, gate_r)
    k = args.top
    top_p = {id(r) for r in proxy_order[:k]}
    top_l = {id(r) for r in lexi[:k]}
    cost_vals = {r["cost"] for r in rows}
    out = {
        "era": list(era),
        "n_rows": len(rows),
        "baseline_speed": base_speed,
        "gate_quality": gate_q,
        "gate_reliability": gate_r,
        "lam": args.lam,
        "lexi_survivors": n_surv,
        "distinct_cost_values": len(cost_vals),
        "kendall_tau_all": round(kendall_tau(proxy_order, lexi), 4),
        f"top{k}_overlap": len(top_p & top_l),
        f"proxy_top{k}_failing_lexi_gate": sum(
            1 for r in proxy_order[:k] if not (r["quality"] >= gate_q and r["reliability"] >= gate_r)
        ),
        "proxy_top_trials": [r["trial_id"] for r in proxy_order[:k]],
        "lexi_top_trials": [r["trial_id"] for r in lexi[:k]],
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
