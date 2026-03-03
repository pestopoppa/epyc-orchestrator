#!/usr/bin/env python3
"""Generate orchestration/eval_registry.yaml from checkpoint data.

Reads JSONL checkpoint files, computes per-suite statistics, and writes
a registry with scoring methods, question counts, pass rates, and trends.

Usage:
    python update_eval_registry.py [--checkpoint-dir DIR] [--output PATH]

Preserves `curated: true` entries on regeneration (merges, not overwrites).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _find_checkpoint_dir() -> Path:
    """Find the default checkpoint directory."""
    candidates = [
        Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/checkpoints"),
        Path("orchestration/checkpoints"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def _load_checkpoints(checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Load all JSONL checkpoint files from the directory."""
    records = []
    if not checkpoint_dir.is_dir():
        return records
    for path in sorted(checkpoint_dir.glob("*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return records


def _compute_suite_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-suite statistics from checkpoint records."""
    suites: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "passed": 0,
        "scoring_methods": set(),
        "timestamps": [],
        "recent_passed": 0,
        "recent_total": 0,
    })

    for rec in records:
        suite = rec.get("suite", "unknown")
        s = suites[suite]
        s["total"] += 1
        if rec.get("passed", False):
            s["passed"] += 1
        sm = rec.get("scoring_method", "")
        if sm:
            s["scoring_methods"].add(sm)
        ts = rec.get("ts", "")
        if ts:
            s["timestamps"].append(ts)

    # Compute recent stats (last 20% of records per suite)
    for suite, s in suites.items():
        recent_count = max(1, s["total"] // 5)
        suite_records = [r for r in records if r.get("suite") == suite]
        recent = suite_records[-recent_count:]
        s["recent_total"] = len(recent)
        s["recent_passed"] = sum(1 for r in recent if r.get("passed", False))

    return dict(suites)


def _compute_trend(overall_rate: float, recent_rate: float) -> str:
    """Classify trend based on overall vs recent pass rate."""
    if abs(recent_rate - overall_rate) < 0.05:
        return "stable"
    elif recent_rate > overall_rate:
        return "improving"
    else:
        return "declining"


def _load_existing_registry(path: Path) -> dict[str, Any]:
    """Load existing registry to preserve curated entries."""
    if not path.is_file():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def build_registry(
    checkpoint_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build the eval registry from checkpoint data.

    Args:
        checkpoint_dir: Directory containing JSONL checkpoint files.
        output_path: Where to write the registry YAML.

    Returns:
        The registry dict.
    """
    records = _load_checkpoints(checkpoint_dir)
    stats = _compute_suite_stats(records)

    # Load existing registry to preserve curated entries
    existing = _load_existing_registry(output_path)
    existing_suites = existing.get("suites", {})

    registry: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
        "total_records": len(records),
        "suites": {},
    }

    for suite, s in sorted(stats.items()):
        overall_rate = s["passed"] / s["total"] if s["total"] > 0 else 0.0
        recent_rate = (
            s["recent_passed"] / s["recent_total"]
            if s["recent_total"] > 0
            else 0.0
        )
        trend = _compute_trend(overall_rate, recent_rate)

        scoring_methods = sorted(s["scoring_methods"])
        last_ts = max(s["timestamps"]) if s["timestamps"] else ""

        entry: dict[str, Any] = {
            "scoring_method": scoring_methods[0] if len(scoring_methods) == 1 else scoring_methods,
            "question_count": s["total"],
            "recent_pass_rate": round(recent_rate, 3),
            "overall_pass_rate": round(overall_rate, 3),
            "trend": trend,
            "last_run": last_ts,
        }

        # Preserve curated fields from existing registry
        old = existing_suites.get(suite, {})
        if old.get("curated"):
            entry["curated"] = True
            # Preserve known_issues if curated
            if "known_issues" in old:
                entry["known_issues"] = old["known_issues"]

        registry["suites"][suite] = entry

    return registry


def write_registry(registry: dict[str, Any], output_path: Path) -> None:
    """Write the registry dict to YAML."""
    try:
        import yaml
    except ImportError:
        # Fallback: write as JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Auto-generated eval registry. Do not edit manually.\n")
        f.write(f"# Generated: {registry.get('generated_at', '')}\n")
        f.write(f"# Source: {registry.get('checkpoint_dir', '')}\n\n")
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate eval registry from checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory with JSONL checkpoint files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML path",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or _find_checkpoint_dir()
    output_path = args.output or Path(
        "/mnt/raid0/llm/epyc-orchestrator/orchestration/eval_registry.yaml"
    )

    registry = build_registry(checkpoint_dir, output_path)
    write_registry(registry, output_path)

    n_suites = len(registry.get("suites", {}))
    n_records = registry.get("total_records", 0)
    print(f"Wrote {n_suites} suites ({n_records} records) to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
