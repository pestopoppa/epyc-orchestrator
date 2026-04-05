#!/usr/bin/env python3
"""Chain Anomaly Detector — flag delegation and escalation anomalies.

Parses progress JSONL logs and detects:
- Repeated escalation paths (same from→to firing disproportionately)
- High failure rates for delegated tasks (specialist timeout, forced synthesis)
- Sequential fallback patterns (repeated escalation chains per task)
- Role concentration anomalies (one role handling >80% of completions)
- Stale tasks (started but never completed within the log window)

Usage:
    # Today's anomalies
    python3 scripts/server/chain_anomaly_detector.py

    # Specific date
    python3 scripts/server/chain_anomaly_detector.py --date 2026-04-03

    # Date range
    python3 scripts/server/chain_anomaly_detector.py --from 2026-04-01 --to 2026-04-04

    # JSON output
    python3 scripts/server/chain_anomaly_detector.py --json
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

# Thresholds
ESCALATION_CONCENTRATION_THRESHOLD = 0.60  # >60% of escalations on one path
ROLE_CONCENTRATION_THRESHOLD = 0.80        # >80% completions by one role
FAILURE_RATE_THRESHOLD = 0.15              # >15% failure rate
STALE_TASK_THRESHOLD_S = 600               # Task started >10 min ago, never completed
MULTI_HOP_ANOMALY_THRESHOLD = 0.20        # >20% of completed tasks are multi-hop


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


def detect_anomalies(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect chain anomalies from progress log entries."""
    anomalies: list[dict[str, Any]] = []

    # Index by task
    started: dict[str, datetime] = {}
    completed: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    escalations: list[dict] = []

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

    total_started = len(started)
    total_completed = len(completed)
    total_failed = len(failed)
    total_resolved = total_completed + total_failed

    if total_started == 0:
        return anomalies

    # 1. Escalation path concentration
    if escalations:
        esc_paths: Counter = Counter()
        for esc in escalations:
            data = esc.get("data", {})
            fr = data.get("from_tier", "?")
            to = data.get("to_tier", "?")
            esc_paths[f"{fr} → {to}"] += 1

        total_esc = sum(esc_paths.values())
        for path, count in esc_paths.most_common(3):
            frac = count / total_esc
            if frac > ESCALATION_CONCENTRATION_THRESHOLD and count >= 3:
                anomalies.append({
                    "type": "escalation_concentration",
                    "severity": "warning",
                    "message": (
                        f"Escalation path '{path}' accounts for {frac:.0%} "
                        f"of all escalations ({count}/{total_esc})"
                    ),
                    "path": path,
                    "count": count,
                    "fraction": round(frac, 3),
                })

    # 2. Role concentration in completions
    final_roles: Counter = Counter()
    for tid, info in completed.items():
        role = info.get("data", {}).get("final_answer_role", "unknown")
        final_roles[role] += 1

    if total_completed > 10:
        for role, count in final_roles.most_common(1):
            frac = count / total_completed
            if frac > ROLE_CONCENTRATION_THRESHOLD:
                anomalies.append({
                    "type": "role_concentration",
                    "severity": "info",
                    "message": (
                        f"Role '{role}' handles {frac:.0%} of completions "
                        f"({count}/{total_completed}) — may indicate routing imbalance"
                    ),
                    "role": role,
                    "fraction": round(frac, 3),
                })

    # 3. Failure rate
    if total_resolved > 10:
        failure_rate = total_failed / total_resolved
        if failure_rate > FAILURE_RATE_THRESHOLD:
            anomalies.append({
                "type": "high_failure_rate",
                "severity": "warning",
                "message": (
                    f"Failure rate {failure_rate:.1%} exceeds threshold "
                    f"({total_failed}/{total_resolved} resolved tasks)"
                ),
                "failure_rate": round(failure_rate, 3),
            })

    # 4. Multi-hop delegation anomaly
    multi_hop_count = 0
    for tid, info in completed.items():
        lineage = info.get("data", {}).get("delegation_lineage", [])
        if len(lineage) > 1:
            multi_hop_count += 1

    if total_completed > 10:
        multi_hop_rate = multi_hop_count / total_completed
        if multi_hop_rate > MULTI_HOP_ANOMALY_THRESHOLD:
            anomalies.append({
                "type": "multi_hop_anomaly",
                "severity": "info",
                "message": (
                    f"Multi-hop delegation rate {multi_hop_rate:.1%} — "
                    f"{multi_hop_count}/{total_completed} tasks required >1 delegation hop"
                ),
                "multi_hop_rate": round(multi_hop_rate, 3),
            })

    # 5. Stale tasks (started but never resolved)
    latest_ts = max(
        (completed[t]["_ts"] for t in completed),
        default=datetime.now(timezone.utc),
    )
    stale_count = 0
    for tid, start_ts in started.items():
        if tid not in completed and tid not in failed:
            age = (latest_ts - start_ts).total_seconds()
            if age > STALE_TASK_THRESHOLD_S:
                stale_count += 1

    if stale_count > 0:
        stale_rate = stale_count / total_started
        if stale_rate > 0.30:
            anomalies.append({
                "type": "stale_tasks",
                "severity": "warning",
                "message": (
                    f"{stale_count} tasks ({stale_rate:.0%}) started but never completed "
                    f"within the log window"
                ),
                "stale_count": stale_count,
                "stale_rate": round(stale_rate, 3),
            })

    # 6. Repeated escalation chains per task (wave stalls)
    tasks_with_multiple_esc: Counter = Counter()
    for esc in escalations:
        tasks_with_multiple_esc[esc.get("task_id", "")] += 1
    wave_stalls = sum(1 for count in tasks_with_multiple_esc.values() if count >= 3)
    if wave_stalls >= 2:
        anomalies.append({
            "type": "wave_stall",
            "severity": "warning",
            "message": (
                f"{wave_stalls} tasks had 3+ escalation events — "
                "possible delegation loop or stuck specialist chain"
            ),
            "affected_tasks": wave_stalls,
        })

    # 7. Failure pattern analysis (from outcome_details)
    failure_patterns: Counter = Counter()
    for tid, info in failed.items():
        details = info.get("outcome_details", "") or ""
        if "timeout" in details.lower() or "timed out" in details.lower():
            failure_patterns["timeout"] += 1
        elif "0.2" in details and "s" in details:
            failure_patterns["instant_failure"] += 1
        else:
            failure_patterns["other"] += 1

    for pattern, count in failure_patterns.most_common():
        if count >= 5:
            anomalies.append({
                "type": "failure_pattern",
                "severity": "info",
                "message": f"Failure pattern '{pattern}' occurred {count} times",
                "pattern": pattern,
                "count": count,
            })

    return anomalies


def format_report(anomalies: list[dict[str, Any]], as_json: bool = False) -> str:
    """Format anomaly report."""
    if as_json:
        return json.dumps(anomalies, indent=2)

    if not anomalies:
        return "No chain anomalies detected."

    lines = ["=" * 60, "  CHAIN ANOMALY REPORT", "=" * 60, ""]

    severity_order = {"warning": 0, "info": 1}
    sorted_anomalies = sorted(anomalies, key=lambda a: severity_order.get(a.get("severity", ""), 2))

    for a in sorted_anomalies:
        sev = a.get("severity", "info").upper()
        msg = a.get("message", "")
        atype = a.get("type", "")
        lines.append(f"  [{sev}] {atype}: {msg}")

    lines.append("")
    lines.append(f"Total anomalies: {len(anomalies)}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Chain Anomaly Detector")
    parser.add_argument("--date", help="Single date (YYYY-MM-DD)")
    parser.add_argument("--from", dest="from_date", help="Start date")
    parser.add_argument("--to", dest="to_date", help="End date")
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument("--json", action="store_true")
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
        all_entries.extend(parse_jsonl(log_dir / f"{date_str}.jsonl"))

    if not all_entries:
        print(f"No log entries found for {', '.join(dates)}", file=sys.stderr)
        sys.exit(1)

    anomalies = detect_anomalies(all_entries)
    print(format_report(anomalies, as_json=args.json))


if __name__ == "__main__":
    main()
