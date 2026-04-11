#!/usr/bin/env python3
"""Aggregate tool output compression metrics from seeding diagnostics JSONL.

Usage:
    python scripts/analysis/compression_stats.py [--file PATH] [--last N]

Options:
    --file PATH   Path to diagnostics JSONL (default: logs/seeding_diagnostics.jsonl)
    --last N      Only consider the last N records (default: all)
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def _cmd_prefix(cmd: str) -> str:
    """Extract command prefix for grouping (e.g. 'pytest', 'git diff')."""
    parts = cmd.strip().split()
    if not parts:
        return "(empty)"
    if parts[0] in ("python", "python3") and len(parts) > 2 and parts[1] == "-m":
        return parts[2]
    if parts[0] == "git" and len(parts) > 1:
        return f"git {parts[1]}"
    return parts[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path,
                        default=Path("logs/seeding_diagnostics.jsonl"))
    parser.add_argument("--last", type=int, default=0,
                        help="Only consider last N records")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if args.last > 0:
        records = records[-args.last:]

    # Extract compression metrics
    compressed = []
    for rec in records:
        cm = rec.get("compression_metrics", {})
        if cm and cm.get("output_original_chars", 0) > 0:
            compressed.append(cm)

    total = len(records)
    n_compressed = len(compressed)

    print(f"Total diagnostic records: {total}")
    print(f"Records with compression: {n_compressed} ({n_compressed/max(total,1)*100:.1f}%)")
    print()

    if not compressed:
        print("No compression metrics found.")
        return

    ratios = [c["output_ratio"] for c in compressed]
    orig_total = sum(c["output_original_chars"] for c in compressed)
    comp_total = sum(c["output_compressed_chars"] for c in compressed)
    saved = orig_total - comp_total

    print(f"{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    print(f"{'Mean ratio':<30} {mean(ratios):>11.1%}")
    print(f"{'Median ratio':<30} {median(ratios):>11.1%}")
    print(f"{'Min ratio (best)':<30} {min(ratios):>11.1%}")
    print(f"{'Max ratio (worst)':<30} {max(ratios):>11.1%}")
    print(f"{'Total original chars':<30} {orig_total:>12,}")
    print(f"{'Total compressed chars':<30} {comp_total:>12,}")
    print(f"{'Total chars saved':<30} {saved:>12,}")
    print()

    # Breakdown by command prefix
    by_cmd: dict[str, list[float]] = defaultdict(list)
    by_cmd_saved: dict[str, int] = Counter()
    for c in compressed:
        prefix = _cmd_prefix(c.get("command", ""))
        by_cmd[prefix].append(c["output_ratio"])
        by_cmd_saved[prefix] += c["output_original_chars"] - c["output_compressed_chars"]

    print(f"{'Command':<20} {'Count':>6} {'Mean Ratio':>11} {'Chars Saved':>12}")
    print("-" * 52)
    for prefix in sorted(by_cmd, key=lambda p: -len(by_cmd[p])):
        vals = by_cmd[prefix]
        print(f"{prefix:<20} {len(vals):>6} {mean(vals):>10.1%} {by_cmd_saved[prefix]:>12,}")


if __name__ == "__main__":
    main()
