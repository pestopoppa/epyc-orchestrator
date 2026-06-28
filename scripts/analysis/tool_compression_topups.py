#!/usr/bin/env python3
"""Summarize bash-compressor downstream top-up telemetry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_MONITOR_PATH = Path("/mnt/raid0/llm/epyc-root/logs/tool_compression_monitor.jsonl")
DEFAULT_MIN_COMPRESSED_CALLS = 100


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_records(path: Path, since: datetime | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("tool") != "run_bash_compressed":
                continue
            if since is not None:
                ts = _parse_timestamp(record.get("timestamp"))
                if ts is None or ts < since:
                    continue
            records.append(record)
    return records


def summarize(
    records: list[dict[str, Any]],
    *,
    min_compressed_calls: int = DEFAULT_MIN_COMPRESSED_CALLS,
) -> dict[str, Any]:
    compressed_calls = len(records)
    followups = [record for record in records if record.get("top_up_candidate") is True]
    reasons = Counter(str(record.get("followup_reason") or "unknown") for record in followups)
    strategies = Counter(str(record.get("compressor_strategy") or "unknown") for record in records)
    passes_threshold = (len(followups) / compressed_calls) <= 0.10 if compressed_calls else None
    has_enough_observations = compressed_calls >= min_compressed_calls

    if compressed_calls == 0:
        rollout_decision = "awaiting_compressed_calls"
    elif not has_enough_observations:
        rollout_decision = "awaiting_minimum_observations"
    elif passes_threshold:
        rollout_decision = "promote_candidate"
    else:
        rollout_decision = "keep_optional_or_drop_candidate"

    return {
        "compressed_calls": compressed_calls,
        "followups": len(followups),
        "top_up_rate": round(len(followups) / compressed_calls, 4) if compressed_calls else 0.0,
        "top_up_rate_threshold": 0.10,
        "min_compressed_calls": min_compressed_calls,
        "has_enough_observations": has_enough_observations,
        "passes_threshold": passes_threshold,
        "ready_for_rollout_decision": compressed_calls > 0 and has_enough_observations,
        "rollout_decision": rollout_decision,
        "followup_reasons": dict(sorted(reasons.items())),
        "compressor_strategies": dict(sorted(strategies.items())),
    }


def _format_markdown(summary: dict[str, Any], path: Path, days: int | None) -> str:
    window = f"last {days} days" if days is not None else "all records"
    pass_state = summary["passes_threshold"]
    gate = "n/a" if pass_state is None else ("PASS" if pass_state else "FAIL")
    lines = [
        "# Tool Compression Top-Up Summary",
        "",
        f"- Source: `{path}`",
        f"- Window: {window}",
        f"- Telemetry status: `{summary.get('telemetry_status', 'unknown')}`",
        f"- Compressed calls: {summary['compressed_calls']}",
        f"- Minimum compressed calls: {summary['min_compressed_calls']}",
        f"- Follow-ups: {summary['followups']}",
        f"- Top-up rate: {summary['top_up_rate']:.2%}",
        f"- Gate (`<=10%`): {gate}",
        f"- Rollout decision: `{summary['rollout_decision']}`",
    ]
    if summary["followup_reasons"]:
        lines.append("- Follow-up reasons:")
        for reason, count in summary["followup_reasons"].items():
            lines.append(f"  - `{reason}`: {count}")
    if summary["compressor_strategies"]:
        lines.append("- Compressor strategies:")
        for strategy, count in summary["compressor_strategies"].items():
            lines.append(f"  - `{strategy}`: {count}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_MONITOR_PATH)
    parser.add_argument("--days", type=int, default=7, help="Lookback window; use 0 for all records")
    parser.add_argument(
        "--min-calls",
        type=int,
        default=DEFAULT_MIN_COMPRESSED_CALLS,
        help="Minimum compressed calls required before a rollout decision is ready",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    args = parser.parse_args()

    days = args.days if args.days > 0 else None
    since = None
    if days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=days)

    min_calls = max(int(args.min_calls), 1)
    summary = summarize(load_records(args.path, since=since), min_compressed_calls=min_calls)
    if not args.path.exists():
        summary["telemetry_status"] = "missing_file"
        summary["ready_for_rollout_decision"] = False
        summary["rollout_decision"] = "awaiting_telemetry_file"
    else:
        summary["telemetry_status"] = "present"
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_format_markdown(summary, args.path, days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
