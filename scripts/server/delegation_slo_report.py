#!/usr/bin/env python3
"""Delegation SLO Report — daily summary of orchestration latency and reliability.

Parses progress JSONL logs and reports:
- p50 / p95 / p99 end-to-end task latency
- Success / failure / timeout rates
- Delegation lineage distribution (single-hop vs multi-hop)
- Escalation frequency and paths
- Per-role latency breakdown

Usage:
    # Today's report
    python3 scripts/server/delegation_slo_report.py

    # Specific date
    python3 scripts/server/delegation_slo_report.py --date 2026-04-03

    # Date range
    python3 scripts/server/delegation_slo_report.py --from 2026-04-01 --to 2026-04-04

    # JSON output
    python3 scripts/server/delegation_slo_report.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/logs/progress")


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, skipping malformed lines."""
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return entries


def percentile(values: list[float], p: float) -> float:
    """Compute percentile from sorted list."""
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(values):
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def extract_elapsed_from_details(details: str | None) -> float | None:
    """Extract elapsed seconds from outcome_details like 'Direct answer mode (frontdoor), 832.979s'."""
    if not details:
        return None
    # Look for pattern like "123.456s" at end
    parts = details.rsplit(",", 1)
    if len(parts) == 2:
        time_part = parts[1].strip()
        if time_part.endswith("s"):
            try:
                return float(time_part[:-1])
            except ValueError:
                pass
    return None


def generate_report(
    entries: list[dict[str, Any]],
    *,
    as_json: bool = False,
) -> str | dict:
    """Generate SLO report from progress log entries."""

    # Index events by task_id
    started: dict[str, datetime] = {}
    completed: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    escalations: list[dict] = []
    routing_decisions: dict[str, dict] = {}

    for entry in entries:
        etype = entry.get("event_type")
        tid = entry.get("task_id", "")
        ts_str = entry.get("timestamp", "")

        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue

        if etype == "task_started":
            started[tid] = ts
        elif etype == "task_completed":
            completed[tid] = {**entry, "_ts": ts}
        elif etype == "task_failed":
            failed[tid] = {**entry, "_ts": ts}
        elif etype == "escalation_triggered":
            escalations.append(entry)
        elif etype == "routing_decision":
            routing_decisions[tid] = entry

    # Compute latencies (from started→completed/failed)
    success_latencies: list[float] = []
    failure_latencies: list[float] = []
    all_latencies: list[float] = []
    per_role_latencies: dict[str, list[float]] = defaultdict(list)
    lineage_counts: Counter = Counter()
    final_roles: Counter = Counter()

    for tid, start_ts in started.items():
        if tid in completed:
            end_ts = completed[tid]["_ts"]
            lat = (end_ts - start_ts).total_seconds()
            # Also try to get precise elapsed from outcome_details
            precise = extract_elapsed_from_details(
                completed[tid].get("outcome_details")
            )
            if precise is not None:
                lat = precise

            success_latencies.append(lat)
            all_latencies.append(lat)

            data = completed[tid].get("data", {})
            lineage = data.get("delegation_lineage", [])
            lineage_counts[len(lineage)] += 1
            final_role = data.get("final_answer_role", "unknown")
            final_roles[final_role] += 1
            per_role_latencies[final_role].append(lat)

        elif tid in failed:
            end_ts = failed[tid]["_ts"]
            lat = (end_ts - start_ts).total_seconds()
            precise = extract_elapsed_from_details(
                failed[tid].get("outcome_details")
            )
            if precise is not None:
                lat = precise
            failure_latencies.append(lat)
            all_latencies.append(lat)

    total_tasks = len(started)
    completed_count = len([t for t in started if t in completed])
    failed_count = len([t for t in started if t in failed])
    in_flight = total_tasks - completed_count - failed_count
    timeout_count = sum(
        1 for tid in failed
        if "timeout" in (failed[tid].get("outcome_details") or "").lower()
        or "timed out" in (failed[tid].get("outcome_details") or "").lower()
    )

    # Escalation paths
    esc_paths: Counter = Counter()
    for esc in escalations:
        data = esc.get("data", {})
        fr = data.get("from_tier", "?")
        to = data.get("to_tier", "?")
        esc_paths[f"{fr} → {to}"] += 1

    report = {
        "period": {
            "tasks_total": total_tasks,
            "completed": completed_count,
            "failed": failed_count,
            "in_flight": in_flight,
            "timeout": timeout_count,
        },
        "rates": {
            "success_rate": round(completed_count / total_tasks * 100, 1) if total_tasks else 0,
            "failure_rate": round(failed_count / total_tasks * 100, 1) if total_tasks else 0,
            "timeout_rate": round(timeout_count / total_tasks * 100, 1) if total_tasks else 0,
        },
        "latency_all": {
            "p50": round(percentile(all_latencies, 50), 2),
            "p95": round(percentile(all_latencies, 95), 2),
            "p99": round(percentile(all_latencies, 99), 2),
            "mean": round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0,
            "count": len(all_latencies),
        },
        "latency_success": {
            "p50": round(percentile(success_latencies, 50), 2),
            "p95": round(percentile(success_latencies, 95), 2),
            "p99": round(percentile(success_latencies, 99), 2),
            "mean": round(sum(success_latencies) / len(success_latencies), 2) if success_latencies else 0,
            "count": len(success_latencies),
        },
        "delegation": {
            "lineage_distribution": dict(sorted(lineage_counts.items())),
            "final_role_distribution": dict(final_roles.most_common()),
        },
        "escalation": {
            "total": len(escalations),
            "paths": dict(esc_paths.most_common()),
        },
        "per_role_latency": {
            role: {
                "p50": round(percentile(lats, 50), 2),
                "p95": round(percentile(lats, 95), 2),
                "count": len(lats),
            }
            for role, lats in sorted(per_role_latencies.items())
        },
    }

    if as_json:
        return report

    # Format as text
    lines = []
    lines.append("=" * 60)
    lines.append("  DELEGATION SLO REPORT")
    lines.append("=" * 60)

    p = report["period"]
    r = report["rates"]
    lines.append(f"\nTasks: {p['tasks_total']} total, {p['completed']} completed, "
                 f"{p['failed']} failed, {p['in_flight']} in-flight")
    lines.append(f"Rates: {r['success_rate']}% success, {r['failure_rate']}% failure, "
                 f"{r['timeout_rate']}% timeout")

    la = report["latency_all"]
    ls = report["latency_success"]
    lines.append(f"\nLatency (all):     p50={la['p50']}s  p95={la['p95']}s  p99={la['p99']}s  mean={la['mean']}s  (n={la['count']})")
    lines.append(f"Latency (success): p50={ls['p50']}s  p95={ls['p95']}s  p99={ls['p99']}s  mean={ls['mean']}s  (n={ls['count']})")

    d = report["delegation"]
    lines.append("\nDelegation lineage:")
    for hops, count in sorted(d["lineage_distribution"].items()):
        label = "single-hop" if hops == 1 else f"{hops}-hop"
        lines.append(f"  {label}: {count}")

    lines.append("\nFinal answer role:")
    for role, count in d["final_role_distribution"].items():
        lines.append(f"  {role}: {count}")

    e = report["escalation"]
    if e["total"]:
        lines.append(f"\nEscalations: {e['total']}")
        for path, count in e["paths"].items():
            lines.append(f"  {path}: {count}")

    prl = report["per_role_latency"]
    if prl:
        lines.append("\nPer-role latency:")
        for role, stats in prl.items():
            lines.append(f"  {role}: p50={stats['p50']}s  p95={stats['p95']}s  (n={stats['count']})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Delegation SLO Report")
    parser.add_argument("--date", help="Single date (YYYY-MM-DD)")
    parser.add_argument("--from", dest="from_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--log-dir", default=str(LOG_DIR), help="Progress log directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if args.date:
        dates = [args.date]
    elif args.from_date and args.to_date:
        from datetime import timedelta
        start = datetime.strptime(args.from_date, "%Y-%m-%d")
        end = datetime.strptime(args.to_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    else:
        dates = [datetime.now(timezone.utc).strftime("%Y-%m-%d")]

    all_entries = []
    for date_str in dates:
        path = log_dir / f"{date_str}.jsonl"
        all_entries.extend(parse_jsonl(path))

    if not all_entries:
        print(f"No log entries found for {', '.join(dates)}", file=sys.stderr)
        sys.exit(1)

    report = generate_report(all_entries, as_json=args.json)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(report)


if __name__ == "__main__":
    main()
