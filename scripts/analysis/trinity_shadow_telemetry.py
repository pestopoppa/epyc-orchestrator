#!/usr/bin/env python3
"""Summarize Trinity role-shadow telemetry from progress JSONL logs.

TR-3.3/3.4 require production-like role-shadow telemetry before the
role-aware routing path can advance. This script is intentionally read-only:
it scans `routing_decision` progress rows, reports coverage and role
distribution, and keeps the one-week collection gate separate from the
non-degeneracy check.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("logs/progress")
DEFAULT_REPORT_DIR = Path("orchestration/reports")


@dataclass
class TrinityTelemetryReport:
    log_paths: list[Path]
    total_routing_rows: int = 0
    role_bearing_rows: int = 0
    malformed_rows: int = 0
    role_counts: Counter[str] = field(default_factory=Counter)
    strategy_counts: Counter[str] = field(default_factory=Counter)
    decision_source_counts: Counter[str] = field(default_factory=Counter)
    first_role_ts: datetime | None = None
    last_role_ts: datetime | None = None
    min_days: float = 7.0
    max_top_role_pct: float = 95.0

    @property
    def missing_role_rows(self) -> int:
        return max(0, self.total_routing_rows - self.role_bearing_rows)

    @property
    def role_coverage_pct(self) -> float:
        if self.total_routing_rows == 0:
            return 0.0
        return 100.0 * self.role_bearing_rows / self.total_routing_rows

    @property
    def observed_days(self) -> float:
        if self.first_role_ts is None or self.last_role_ts is None:
            return 0.0
        span = self.last_role_ts - self.first_role_ts
        return max(0.0, span.total_seconds() / 86_400.0)

    @property
    def top_role_pct(self) -> float:
        if not self.role_counts or self.role_bearing_rows == 0:
            return 0.0
        return 100.0 * self.role_counts.most_common(1)[0][1] / self.role_bearing_rows

    @property
    def distribution_non_degenerate(self) -> bool:
        return len(self.role_counts) >= 2 and self.top_role_pct < self.max_top_role_pct

    @property
    def collection_window_satisfied(self) -> bool:
        return self.observed_days >= self.min_days


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _iter_log_paths(log_dir: Path, from_date: str | None, to_date: str | None) -> list[Path]:
    paths = [Path(p) for p in sorted(glob.glob(str(log_dir / "*.jsonl")))]
    out: list[Path] = []
    for path in paths:
        stem = path.stem
        if from_date and stem < from_date:
            continue
        if to_date and stem > to_date:
            continue
        out.append(path)
    return out


def summarize_trinity_telemetry(
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
    from_date: str | None = None,
    to_date: str | None = None,
    min_days: float = 7.0,
    max_top_role_pct: float = 95.0,
) -> TrinityTelemetryReport:
    report = TrinityTelemetryReport(
        log_paths=_iter_log_paths(log_dir, from_date, to_date),
        min_days=min_days,
        max_top_role_pct=max_top_role_pct,
    )
    for path in report.log_paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    report.malformed_rows += 1
                    continue
                if not isinstance(row, dict):
                    report.malformed_rows += 1
                    continue
                if row.get("event_type") != "routing_decision":
                    continue
                report.total_routing_rows += 1
                data = row.get("data") if isinstance(row.get("data"), dict) else {}
                strategy = data.get("strategy") or "unknown"
                source = data.get("decision_source") or "unknown"
                report.strategy_counts[str(strategy)] += 1
                report.decision_source_counts[str(source)] += 1
                role = data.get("assigned_role")
                if not role:
                    continue
                role_text = str(role)
                report.role_bearing_rows += 1
                report.role_counts[role_text] += 1
                ts = _parse_timestamp(row.get("timestamp"))
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if report.first_role_ts is None or ts < report.first_role_ts:
                    report.first_role_ts = ts
                if report.last_role_ts is None or ts > report.last_role_ts:
                    report.last_role_ts = ts
    return report


def _count_lines(counter: Counter[str], total: int) -> list[str]:
    if not counter:
        return ["  - none"]
    lines: list[str] = []
    for key, count in counter.most_common():
        pct = 100.0 * count / max(1, total)
        lines.append(f"  - `{key}`: {count:,} ({pct:.1f}%)")
    return lines


def render_report(report: TrinityTelemetryReport) -> str:
    first_ts = report.first_role_ts.isoformat() if report.first_role_ts else "n/a"
    last_ts = report.last_role_ts.isoformat() if report.last_role_ts else "n/a"
    lines = [
        "# Trinity Shadow Telemetry Report",
        "",
        "## Summary",
        "",
        f"- log files scanned: {len(report.log_paths)}",
        f"- routing_decision rows: {report.total_routing_rows:,}",
        f"- rows with assigned_role: {report.role_bearing_rows:,} ({report.role_coverage_pct:.1f}%)",
        f"- rows missing assigned_role: {report.missing_role_rows:,}",
        f"- first role timestamp: {first_ts}",
        f"- last role timestamp: {last_ts}",
        f"- observed role-bearing span: {report.observed_days:.3f} days",
        f"- malformed rows skipped: {report.malformed_rows:,}",
        "",
        "## Role Distribution",
        "",
        *_count_lines(report.role_counts, report.role_bearing_rows),
        "",
        "## Strategy Distribution",
        "",
        *_count_lines(report.strategy_counts, report.total_routing_rows),
        "",
        "## Decision Source Distribution",
        "",
        *_count_lines(report.decision_source_counts, report.total_routing_rows),
        "",
        "## TR-3.3 / TR-3.4 Verdict",
        "",
        (
            "- TR-3.3 collection window: "
            + (
                "PASS"
                if report.collection_window_satisfied
                else (
                    "PENDING — observed span "
                    f"{report.observed_days:.3f}d < required {report.min_days:.1f}d"
                )
            )
        ),
        (
            "- TR-3.4 non-degenerate distribution: "
            + (
                "PASS"
                if report.distribution_non_degenerate
                else (
                    "PENDING — top role share "
                    f"{report.top_role_pct:.1f}% >= {report.max_top_role_pct:.1f}% "
                    "or fewer than two roles observed"
                )
            )
        ),
        "",
        "Interpretation: telemetry persistence is working once `assigned_role` appears in "
        "progress rows. Do not promote TR-4/5 until TR-3.3 has a clean production-like "
        "window and the distribution remains non-degenerate.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--from", dest="from_date", help="First YYYY-MM-DD log date")
    parser.add_argument("--to", dest="to_date", help="Last YYYY-MM-DD log date")
    parser.add_argument("--min-days", type=float, default=7.0)
    parser.add_argument("--max-top-role-pct", type=float, default=95.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = summarize_trinity_telemetry(
        log_dir=args.log_dir,
        from_date=args.from_date,
        to_date=args.to_date,
        min_days=args.min_days,
        max_top_role_pct=args.max_top_role_pct,
    )
    rendered = render_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
