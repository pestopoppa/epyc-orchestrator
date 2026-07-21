#!/usr/bin/env python3
"""Replay historical task completions through the fixed reward function.

WHY THIS EXISTS
---------------
`compute_reward` gates its entire cost/speed half behind a role lookup:

    role = cost_metrics.get("role", "")
    baseline_tps = config.baseline_tps_by_role.get(role, 0)

`cost_metrics` is the TASK_COMPLETED entry's ``data`` dict, which carries
``producer_role`` / ``final_answer_role`` and has never carried a bare
``role``. The lookup therefore resolved ``baseline_tps`` to 0, every guard
failed, and reward collapsed to ``base_reward``. Measured 2026-07-21 over
20,521 production completions: ``role`` present 0 times, ``producer_role``
present 20,521 times.

Consequence: stored ``q_value``s are a near-constant. Every learned-routing
experiment was fitting a target carrying ~0 bits.

The progress logs retain everything needed to recompute the reward properly,
so this is a REPLAY, not a re-run: no inference, no model, no GPU.

WHAT IT EMITS
-------------
A JSONL artifact, one row per completed task, carrying both the pre-fix and
post-fix reward so the instrument-era boundary is auditable, plus the raw
timing inputs needed to design the wall-clock speed axis.

It also reports the wall-clock/model-compute overhead ratio per role, which is
the evidence that a tokens-per-second speed term is blind to orchestration and
tool time (median ~1.6x, p90 ~9x as of 2026-07-21).

MEASUREMENT NOTE
----------------
Outputs are OBSERVATIONS under MEASUREMENT.md. Pre-fix and post-fix rewards are
DIFFERENT INSTRUMENTS and must not be compared or co-trained without recording
the era boundary. This script never mutates the episodic store; persisting
rescored values is a separate, deliberate step.

Usage:
    python scripts/analysis/rescore_rewards_from_progress.py \
        --out /mnt/raid0/llm/epyc-orchestrator/orchestration/reports/reward_rescore_YYYYMMDD.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestration.repl_memory.q_reward import compute_reward  # noqa: E402
from orchestration.repl_memory.q_scorer import ScoringConfig  # noqa: E402

DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "progress"

# Keys the pre-fix code could never resolve. Stripping them reproduces the old
# behaviour exactly, which is what makes the before/after comparison honest.
_ROLE_KEYS_ADDED_BY_FIX = ("producer_role", "final_answer_role")


class _Entry:
    """Minimal stand-in for ProgressEntry — compute_reward only reads these."""

    __slots__ = ("outcome", "data", "event_type")

    def __init__(self, outcome: str, data: dict) -> None:
        self.outcome = outcome
        self.data = data
        self.event_type = None


def _parse_ts(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _entropy(values, bin_width=0.1):
    counts = collections.Counter(round(v / bin_width) for v in values)
    total = len(values)
    if not total:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def rescore(log_dir: Path, out_path: Path | None):
    config = ScoringConfig()
    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no progress logs under {log_dir}")

    starts: dict[str, datetime] = {}
    rows = []
    skipped = 0

    for path in files:
        with path.open(errors="ignore") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("event_type")
                task_id = event.get("task_id")
                if etype == "task_started" and task_id:
                    started = _parse_ts(event.get("timestamp"))
                    if started:
                        starts[task_id] = started
                    continue
                if etype != "task_completed":
                    continue

                data = event.get("data") or {}
                outcome = event.get("outcome") or "success"
                data_prefix = {
                    k: v for k, v in data.items() if k not in _ROLE_KEYS_ADDED_BY_FIX
                }
                try:
                    reward_pre = compute_reward(
                        _Entry(outcome, data_prefix), [], [], None,
                        data_prefix, config=config,
                    )
                    reward_post = compute_reward(
                        _Entry(outcome, data), [], [], None, data, config=config,
                    )
                except Exception:  # malformed row — count it, never silently drop
                    skipped += 1
                    continue

                completed = _parse_ts(event.get("timestamp"))
                started = starts.get(task_id)
                wall = (
                    (completed - started).total_seconds()
                    if started and completed
                    else None
                )
                gen_s = (data.get("generation_ms") or 0) / 1000.0
                pe_s = (data.get("prompt_eval_ms") or 0) / 1000.0
                model_s = gen_s + pe_s

                rows.append({
                    "task_id": task_id,
                    "timestamp": event.get("timestamp"),
                    "producer_role": data.get("producer_role"),
                    "final_answer_role": data.get("final_answer_role"),
                    "outcome": outcome,
                    "tokens_generated": data.get("tokens_generated"),
                    "generation_s": round(gen_s, 3),
                    "prompt_eval_s": round(pe_s, 3),
                    "model_compute_s": round(model_s, 3),
                    "wall_clock_s": round(wall, 3) if wall is not None else None,
                    # The speed axis the tokens/sec term cannot see.
                    "overhead_ratio": (
                        round(wall / model_s, 3)
                        if wall is not None and model_s > 0.05
                        else None
                    ),
                    "reward_pre_fix": round(reward_pre, 6),
                    "reward_post_fix": round(reward_post, 6),
                })

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    _report(rows, files, skipped, out_path)
    return rows


def _report(rows, files, skipped, out_path):
    pre = [r["reward_pre_fix"] for r in rows]
    post = [r["reward_post_fix"] for r in rows]
    print(f"progress files replayed : {len(files)}")
    print(f"completed tasks rescored: {len(rows):,}   (malformed skipped: {skipped})")
    if not rows:
        return

    def line(label, vals):
        sat = sum(1 for v in vals if v >= 0.999)
        print(
            f"  {label:<10} mean={statistics.mean(vals):+.4f}  "
            f"sd={statistics.pstdev(vals):.4f}  "
            f"at r=+1.0: {sat:,} ({100 * sat / len(vals):.1f}%)  "
            f"entropy={_entropy(vals):.4f} bits"
        )

    print("\nreward distribution:")
    line("PRE-FIX", pre)
    line("POST-FIX", post)

    by_role = collections.defaultdict(list)
    for r in rows:
        if r["producer_role"]:
            by_role[r["producer_role"]].append(r["reward_post_fix"])
    print("\npost-fix reward by role:")
    for role, vals in sorted(by_role.items(), key=lambda kv: -len(kv[1])):
        if len(vals) < 50:
            continue
        print(f"  {role:<22} n={len(vals):>7,}  mean={statistics.mean(vals):+.4f}")

    ratios = [r["overhead_ratio"] for r in rows if r["overhead_ratio"]]
    if ratios:
        q = statistics.quantiles(ratios, n=10)
        print(
            f"\nwall-clock / model-compute over {len(ratios):,} tasks: "
            f"median {statistics.median(ratios):.2f}x  p90 {q[8]:.2f}x"
        )
        print("  (the gap is orchestration + tool time — invisible to a tokens/sec term)")

    if out_path:
        print(f"\nartifact: {out_path}")
    print(
        "\nMEASUREMENT: observations only. PRE/POST are different instruments — "
        "record the era boundary before any mixed comparison or cross-era training."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=None, help="JSONL artifact path")
    args = ap.parse_args()
    rescore(args.log_dir, args.out)


if __name__ == "__main__":
    main()
