#!/usr/bin/env python3
"""Read-only Internal Interaction P1 bake report.

Consumes progress JSONL and summarizes the two counters that gate P2/J17:

- delegated completion cache lookup/hit/miss/rate metadata
- ``ContentionDenied`` HTTP 503 progress events

The report is intentionally conservative: a clean gate requires an elapsed
window, observable delegation-cache traffic, and no second-half rate rise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROGRESS_DIR = _REPO_ROOT / "logs/progress"
DEFAULT_MIN_HOURS = 48.0


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SystemExit(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _jsonl_paths(progress_dir: Path, start: datetime, end: datetime) -> list[Path]:
    paths: list[Path] = []
    day = start.date()
    while day <= end.date():
        path = progress_dir / f"{day.isoformat()}.jsonl"
        if path.exists():
            paths.append(path)
        day += timedelta(days=1)
    return paths


def _iter_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row
        except OSError:
            continue


def _window_rows(
    rows: Iterable[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        ts = _parse_ts(str(row.get("timestamp") or ""))
        if ts is None:
            continue
        if start <= ts <= end:
            selected.append(row)
    return selected


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _summarize(rows: list[dict[str, Any]], *, hours: float) -> dict[str, Any]:
    cache_rows: list[dict[str, Any]] = []
    lookups = 0
    hits = 0
    misses = 0
    contention_events = 0
    contention_task_ids: list[str] = []

    for row in rows:
        data = row.get("data")
        data = data if isinstance(data, dict) else {}
        if "delegation_cache_lookups" in data:
            cache_rows.append(row)
            lookups += int(data.get("delegation_cache_lookups") or 0)
            hits += int(data.get("delegation_cache_hits") or 0)
            misses += int(data.get("delegation_cache_misses") or 0)
        if row.get("event_type") == "routing_fallback" and data.get("kind") == "contention_denied":
            contention_events += 1
            task_id = row.get("task_id")
            if isinstance(task_id, str) and task_id:
                contention_task_ids.append(task_id)

    return {
        "progress_rows": len(rows),
        "delegation_cache_completion_rows": len(cache_rows),
        "delegation_cache_lookups": lookups,
        "delegation_cache_hits": hits,
        "delegation_cache_misses": misses,
        "delegation_cache_hit_rate": _rate(hits, lookups),
        "delegation_cache_miss_rate": _rate(misses, lookups),
        "contention_denied_events": contention_events,
        "contention_denied_per_hour": round(contention_events / hours, 6)
        if hours > 0
        else None,
        "contention_denied_task_ids": contention_task_ids[:20],
    }


def build_report(
    *,
    progress_dir: Path = DEFAULT_PROGRESS_DIR,
    since: datetime | None = None,
    until: datetime | None = None,
    min_hours: float = DEFAULT_MIN_HOURS,
    min_delegation_lookups: int = 1,
) -> dict[str, Any]:
    end = until or datetime.now(timezone.utc)
    start = since or (end - timedelta(hours=min_hours))
    if start >= end:
        raise SystemExit("--since must be before --until")

    paths = _jsonl_paths(progress_dir, start, end)
    rows = _window_rows(_iter_rows(paths), start=start, end=end)
    duration_hours = max(0.0, (end - start).total_seconds() / 3600.0)
    midpoint = start + (end - start) / 2
    first_rows = _window_rows(rows, start=start, end=midpoint)
    second_rows = _window_rows(rows, start=midpoint, end=end)

    full = _summarize(rows, hours=duration_hours)
    first = _summarize(first_rows, hours=max(duration_hours / 2.0, 0.0))
    second = _summarize(second_rows, hours=max(duration_hours / 2.0, 0.0))

    blockers: list[str] = []
    if duration_hours < min_hours:
        blockers.append(f"window_too_short:{duration_hours:.2f}h<{min_hours:.2f}h")
    if full["delegation_cache_lookups"] < min_delegation_lookups:
        blockers.append(
            "delegation_cache_observations_too_small:"
            f"{full['delegation_cache_lookups']}<{min_delegation_lookups}"
        )

    cache_rise = None
    first_miss = first["delegation_cache_miss_rate"]
    second_miss = second["delegation_cache_miss_rate"]
    if first_miss is not None and second_miss is not None:
        cache_rise = second_miss > first_miss
        if cache_rise:
            blockers.append(
                f"delegation_cache_miss_rate_rose:{first_miss:.6f}->{second_miss:.6f}"
            )
    elif full["delegation_cache_lookups"] >= min_delegation_lookups:
        blockers.append("delegation_cache_split_comparison_unavailable")

    contention_rise = None
    first_contention = first["contention_denied_per_hour"]
    second_contention = second["contention_denied_per_hour"]
    if first_contention is not None and second_contention is not None:
        contention_rise = second_contention > first_contention
        if contention_rise:
            blockers.append(
                "contention_denied_rate_rose:"
                f"{first_contention:.6f}->{second_contention:.6f}"
            )

    return {
        "schema_version": "internal_interaction_bake_report.v1",
        "progress_dir": str(progress_dir),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "duration_hours": round(duration_hours, 6),
        "min_hours": min_hours,
        "min_delegation_lookups": min_delegation_lookups,
        "log_paths": [str(path) for path in paths],
        "full_window": full,
        "first_half": first,
        "second_half": second,
        "rise_checks": {
            "delegation_cache_miss_rate_rose": cache_rise,
            "contention_denied_rate_rose": contention_rise,
        },
        "gate_ready": not blockers,
        "blockers": blockers,
    }


def render_markdown(report: dict[str, Any]) -> str:
    full = report["full_window"]
    first = report["first_half"]
    second = report["second_half"]
    lines = [
        "# Internal Interaction Bake Report",
        "",
        f"- Window: `{report['window_start']}` to `{report['window_end']}`",
        f"- Duration hours: `{report['duration_hours']}` (min `{report['min_hours']}`)",
        f"- Gate ready: `{str(report['gate_ready']).lower()}`",
        f"- Blockers: {', '.join(report['blockers']) if report['blockers'] else 'none'}",
        "",
        "## Full Window",
        "",
        f"- Progress rows: `{full['progress_rows']}`",
        f"- Delegation cache completion rows: `{full['delegation_cache_completion_rows']}`",
        f"- Delegation cache lookups/hits/misses: `{full['delegation_cache_lookups']}` / "
        f"`{full['delegation_cache_hits']}` / `{full['delegation_cache_misses']}`",
        f"- Delegation cache miss rate: `{full['delegation_cache_miss_rate']}`",
        f"- ContentionDenied events: `{full['contention_denied_events']}`",
        f"- ContentionDenied per hour: `{full['contention_denied_per_hour']}`",
        "",
        "## Split Check",
        "",
        "| metric | first half | second half | rose |",
        "|---|---:|---:|---:|",
        "| delegation cache miss rate | "
        f"{first['delegation_cache_miss_rate']} | {second['delegation_cache_miss_rate']} | "
        f"{report['rise_checks']['delegation_cache_miss_rate_rose']} |",
        "| contention denied per hour | "
        f"{first['contention_denied_per_hour']} | {second['contention_denied_per_hour']} | "
        f"{report['rise_checks']['contention_denied_rate_rose']} |",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-dir", type=Path, default=DEFAULT_PROGRESS_DIR)
    parser.add_argument("--since", default=None, help="ISO timestamp, inclusive")
    parser.add_argument("--until", default=None, help="ISO timestamp, inclusive; defaults to now")
    parser.add_argument("--min-hours", type=float, default=DEFAULT_MIN_HOURS)
    parser.add_argument("--min-delegation-lookups", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when gate is blocked")
    args = parser.parse_args(argv)

    report = build_report(
        progress_dir=args.progress_dir,
        since=_parse_ts(args.since),
        until=_parse_ts(args.until),
        min_hours=args.min_hours,
        min_delegation_lookups=args.min_delegation_lookups,
    )
    rendered = render_markdown(report)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(rendered)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(rendered, end="")
    if args.strict and not report["gate_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
