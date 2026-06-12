#!/usr/bin/env python3
"""Generate weekly economics ledgers from existing orchestration logs.

The ledger is deliberately read-only. It summarizes cloud planner spend,
manual cloud spend, local eval wall time, and throughput proxies without
mutating the live autopilot state or journal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PLANNER_ARCHIVE = REPO / "logs" / "planner_archive.jsonl"
DEFAULT_JOURNAL_DIR = REPO / "orchestration"
DEFAULT_CLOUD_COSTS = REPO / "orchestration" / "cloud_costs.yaml"
DEFAULT_REPORT_DIR = REPO / "orchestration" / "reports"
DEFAULT_PROGRESS_ROOT = Path(os.environ.get("EPYC_ROOT_PROGRESS", "/mnt/raid0/llm/epyc-root/progress"))
DEFAULT_ORCH_PROGRESS = REPO / "logs" / "progress"


@dataclass
class PlannerSpend:
    calls: int = 0
    billable_calls: int = 0
    total_usd: float = 0.0
    duration_s: float = 0.0
    malformed_rows: int = 0
    by_provider_usd: Counter[str] = field(default_factory=Counter)
    by_purpose_usd: Counter[str] = field(default_factory=Counter)


@dataclass
class ManualSpend:
    entries: int = 0
    total_usd: float = 0.0
    by_provider_usd: Counter[str] = field(default_factory=Counter)
    by_purpose_usd: Counter[str] = field(default_factory=Counter)
    source_exists: bool = False


@dataclass
class LocalInference:
    trials: int = 0
    eval_wall_s: float = 0.0
    malformed_rows: int = 0
    by_consumer_s: Counter[str] = field(default_factory=Counter)
    by_tier_trials: Counter[str] = field(default_factory=Counter)

    @property
    def eval_hours(self) -> float:
        return self.eval_wall_s / 3600.0


@dataclass
class ThroughputProxy:
    progress_files: int = 0
    progress_decision_markers: int = 0
    halt_resume_mentions: int = 0
    routing_decisions: int = 0
    task_completions: int = 0
    task_duration_s: list[float] = field(default_factory=list)
    malformed_rows: int = 0

    @property
    def median_task_duration_s(self) -> float | None:
        if not self.task_duration_s:
            return None
        return float(median(self.task_duration_s))


@dataclass
class EconomicsLedger:
    window_start: datetime
    window_end: datetime
    planner: PlannerSpend
    manual: ManualSpend
    local: LocalInference
    throughput: ThroughputProxy

    @property
    def total_cloud_usd(self) -> float:
        return self.planner.total_usd + self.manual.total_usd


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                dt = datetime.combine(date.fromisoformat(text[:10]), time.min)
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                malformed += 1
    return rows, malformed


def _in_window(ts: datetime | None, start: datetime, end: datetime) -> bool:
    return ts is not None and start <= ts < end


def _planner_timestamp(row: dict[str, Any]) -> datetime | None:
    return _parse_dt(row.get("ts_iso") or row.get("timestamp") or row.get("ts"))


def _planner_provider(row: dict[str, Any]) -> str:
    provider = row.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    if row.get("total_cost_usd") is not None:
        return "claude"
    return "unknown"


def _planner_purpose(row: dict[str, Any]) -> str:
    if row.get("type") == "planner_coordinator":
        mode = row.get("mode")
        return f"planner_coordinator:{mode}" if mode else "planner_coordinator"
    role = row.get("role")
    if isinstance(role, str) and role.strip():
        return f"planner:{role.strip()}"
    subtype = row.get("subtype")
    if subtype in {"success", "failed", "timeout", "file_not_found"}:
        return "planner:cloud_session"
    if isinstance(subtype, str) and subtype.strip():
        return f"planner:{subtype.strip()}"
    return "planner:unknown"


def _summarize_planner(path: Path, start: datetime, end: datetime) -> PlannerSpend:
    rows, malformed = _read_jsonl(path)
    out = PlannerSpend(malformed_rows=malformed)
    for row in rows:
        if not _in_window(_planner_timestamp(row), start, end):
            continue
        out.calls += 1
        duration = _as_float(row.get("duration_s"))
        if duration is None:
            ms = _as_float(row.get("duration_ms"))
            duration = ms / 1000.0 if ms is not None else None
        if duration is not None:
            out.duration_s += duration
        cost = _as_float(row.get("total_cost_usd"))
        if cost is None:
            continue
        provider = _planner_provider(row)
        purpose = _planner_purpose(row)
        out.billable_calls += 1
        out.total_usd += cost
        out.by_provider_usd[provider] += cost
        out.by_purpose_usd[purpose] += cost
    return out


def _load_cloud_entries(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    data = yaml.safe_load(path.read_text()) or {}
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("entries") or data.get("costs") or []
    else:
        entries = []
    return [item for item in entries if isinstance(item, dict)], True


def _summarize_manual(path: Path, start: datetime, end: datetime) -> ManualSpend:
    rows, exists = _load_cloud_entries(path)
    out = ManualSpend(source_exists=exists)
    for row in rows:
        ts = _parse_dt(row.get("date") or row.get("timestamp") or row.get("ts"))
        if not _in_window(ts, start, end):
            continue
        amount = _as_float(row.get("amount_usd") or row.get("cost_usd") or row.get("usd"))
        if amount is None:
            continue
        provider = str(row.get("provider") or "manual").strip() or "manual"
        purpose = str(row.get("purpose") or "manual").strip() or "manual"
        out.entries += 1
        out.total_usd += amount
        out.by_provider_usd[provider] += amount
        out.by_purpose_usd[purpose] += amount
    return out


def _journal_timestamp(row: dict[str, Any]) -> datetime | None:
    return _parse_dt(row.get("timestamp") or row.get("ts_iso") or row.get("ts"))


def _eval_wall_s(row: dict[str, Any]) -> float | None:
    direct = _as_float(row.get("eval_wall_s"))
    if direct is not None:
        return direct
    eval_details = row.get("eval_details")
    if not isinstance(eval_details, dict):
        return None
    nested = _as_float(eval_details.get("eval_wall_s"))
    if nested is not None:
        return nested
    details = eval_details.get("details")
    if isinstance(details, dict):
        return _as_float(details.get("eval_wall_s"))
    return None


def _journal_files(journal_dir: Path) -> list[Path]:
    if journal_dir.is_file():
        return [journal_dir]
    if not journal_dir.exists():
        return []
    return sorted(journal_dir.glob("autopilot_journal*.jsonl"))


def _summarize_local(journal_dir: Path, start: datetime, end: datetime) -> LocalInference:
    out = LocalInference()
    for path in _journal_files(journal_dir):
        rows, malformed = _read_jsonl(path)
        out.malformed_rows += malformed
        for row in rows:
            if not _in_window(_journal_timestamp(row), start, end):
                continue
            wall_s = _eval_wall_s(row)
            if wall_s is None:
                continue
            action = str(row.get("action_type") or "unknown").strip() or "unknown"
            species = str(row.get("species") or "unknown").strip() or "unknown"
            tier = str(row.get("tier") if row.get("tier") is not None else "unknown")
            consumer = f"{action}:{species}"
            out.trials += 1
            out.eval_wall_s += wall_s
            out.by_consumer_s[consumer] += wall_s
            out.by_tier_trials[tier] += 1
    return out


_PROGRESS_MARKER_WORDS = (
    "decision",
    "verdict",
    "gated",
    "checked off",
    "blocked",
    "restart",
    "resume",
    "halt",
    "landed",
    "next work",
)


def _progress_file_date(path: Path) -> datetime | None:
    try:
        return datetime.combine(date.fromisoformat(path.stem[:10]), time.min, tzinfo=timezone.utc)
    except ValueError:
        return None


def _summarize_progress_markers(progress_root: Path, start: datetime, end: datetime) -> ThroughputProxy:
    out = ThroughputProxy()
    if not progress_root.exists():
        return out
    files = sorted(progress_root.glob("*/*.md"))
    for path in files:
        day = _progress_file_date(path)
        if not _in_window(day, start, end):
            continue
        out.progress_files += 1
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            normalized = line.lower()
            if any(word in normalized for word in _PROGRESS_MARKER_WORDS):
                out.progress_decision_markers += 1
            if "halt" in normalized or "resume" in normalized or "restart" in normalized:
                out.halt_resume_mentions += 1
    return out


def _progress_event_files(progress_dir: Path, start: datetime, end: datetime) -> list[Path]:
    if not progress_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(progress_dir.glob("*.jsonl")):
        day = _progress_file_date(path)
        if _in_window(day, start, end):
            files.append(path)
    return files


def _merge_event_throughput(
    out: ThroughputProxy,
    progress_dir: Path,
    start: datetime,
    end: datetime,
) -> None:
    started: dict[str, datetime] = {}
    for path in _progress_event_files(progress_dir, start, end):
        rows, malformed = _read_jsonl(path)
        out.malformed_rows += malformed
        for row in rows:
            ts = _parse_dt(row.get("timestamp") or row.get("ts_iso") or row.get("ts"))
            if not _in_window(ts, start, end):
                continue
            event_type = row.get("event_type")
            task_id = str(row.get("task_id") or "")
            if event_type == "routing_decision":
                out.routing_decisions += 1
            elif event_type == "task_started" and task_id and ts is not None:
                started[task_id] = ts
            elif event_type == "task_completed":
                out.task_completions += 1
                if task_id and task_id in started and ts is not None:
                    elapsed = (ts - started[task_id]).total_seconds()
                    if elapsed >= 0:
                        out.task_duration_s.append(elapsed)


def _summarize_throughput(
    progress_root: Path,
    orch_progress_dir: Path,
    start: datetime,
    end: datetime,
) -> ThroughputProxy:
    out = _summarize_progress_markers(progress_root, start, end)
    _merge_event_throughput(out, orch_progress_dir, start, end)
    return out


def summarize_economics(
    *,
    week_start: date | datetime | None = None,
    days: int = 7,
    planner_archive: Path = DEFAULT_PLANNER_ARCHIVE,
    journal_dir: Path = DEFAULT_JOURNAL_DIR,
    cloud_costs: Path = DEFAULT_CLOUD_COSTS,
    progress_root: Path = DEFAULT_PROGRESS_ROOT,
    orch_progress_dir: Path = DEFAULT_ORCH_PROGRESS,
    now: datetime | None = None,
) -> EconomicsLedger:
    now = now or datetime.now(timezone.utc)
    if week_start is None:
        start_date = now.astimezone(timezone.utc).date() - timedelta(days=days - 1)
    elif isinstance(week_start, datetime):
        start_date = week_start.astimezone(timezone.utc).date()
    else:
        start_date = week_start
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=days)
    return EconomicsLedger(
        window_start=start,
        window_end=end,
        planner=_summarize_planner(planner_archive, start, end),
        manual=_summarize_manual(cloud_costs, start, end),
        local=_summarize_local(journal_dir, start, end),
        throughput=_summarize_throughput(progress_root, orch_progress_dir, start, end),
    )


def _money(value: float) -> str:
    return f"${value:.4f}"


def _hours(value: float) -> str:
    return f"{value:.2f}h"


def _top_counter(counter: Counter[str], *, money: bool = False, seconds_to_hours: bool = False, limit: int = 8) -> list[str]:
    if not counter:
        return ["  - none"]
    lines: list[str] = []
    for key, value in counter.most_common(limit):
        if money:
            rendered = _money(float(value))
        elif seconds_to_hours:
            rendered = _hours(float(value) / 3600.0)
        else:
            rendered = str(value)
        lines.append(f"  - `{key}`: {rendered}")
    return lines


def render_report(ledger: EconomicsLedger) -> str:
    window = (
        f"{ledger.window_start.date().isoformat()} through "
        f"{(ledger.window_end - timedelta(days=1)).date().isoformat()} UTC"
    )
    median_task = ledger.throughput.median_task_duration_s
    median_task_text = f"{median_task:.2f}s" if median_task is not None else "n/a"
    lines = [
        "# Economic Ledger",
        "",
        f"Window: {window}",
        "",
        "## Cloud spend",
        "",
        f"- planner archive billable calls: {ledger.planner.billable_calls} / {ledger.planner.calls}",
        f"- planner archive spend: {_money(ledger.planner.total_usd)}",
        f"- manual cloud spend: {_money(ledger.manual.total_usd)}"
        + ("" if ledger.manual.source_exists else " (no `cloud_costs.yaml` found)"),
        f"- total cloud spend: {_money(ledger.total_cloud_usd)}",
        f"- planner wall time: {_hours(ledger.planner.duration_s / 3600.0)}",
        "",
        "### Planner spend by provider",
        *_top_counter(ledger.planner.by_provider_usd, money=True),
        "",
        "### Planner spend by purpose",
        *_top_counter(ledger.planner.by_purpose_usd, money=True),
        "",
        "### Manual spend by purpose",
        *_top_counter(ledger.manual.by_purpose_usd, money=True),
        "",
        "## Local inference",
        "",
        f"- autopilot eval trials with wall time: {ledger.local.trials}",
        f"- eval wall time: {_hours(ledger.local.eval_hours)}",
        "",
        "### Local eval wall time by consumer",
        *_top_counter(ledger.local.by_consumer_s, seconds_to_hours=True),
        "",
        "## Decision throughput proxy",
        "",
        "This section is a proxy. The repo does not yet expose a canonical operator-decision event stream.",
        f"- root progress files scanned: {ledger.throughput.progress_files}",
        f"- progress decision markers: {ledger.throughput.progress_decision_markers}",
        f"- halt/resume/restart mentions: {ledger.throughput.halt_resume_mentions}",
        f"- automated routing decisions: {ledger.throughput.routing_decisions}",
        f"- completed interactive tasks: {ledger.throughput.task_completions}",
        f"- median task duration from progress JSONL: {median_task_text}",
        "",
        "## Parse health",
        "",
        f"- planner malformed rows: {ledger.planner.malformed_rows}",
        f"- journal malformed rows: {ledger.local.malformed_rows}",
        f"- progress malformed rows: {ledger.throughput.malformed_rows}",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-start", type=str, help="UTC start date, YYYY-MM-DD. Defaults to trailing window.")
    parser.add_argument("--days", type=int, default=7, help="Window length in days.")
    parser.add_argument("--planner-archive", type=Path, default=DEFAULT_PLANNER_ARCHIVE)
    parser.add_argument("--journal-dir", type=Path, default=DEFAULT_JOURNAL_DIR)
    parser.add_argument("--cloud-costs", type=Path, default=DEFAULT_CLOUD_COSTS)
    parser.add_argument("--progress-root", type=Path, default=DEFAULT_PROGRESS_ROOT)
    parser.add_argument("--orch-progress-dir", type=Path, default=DEFAULT_ORCH_PROGRESS)
    parser.add_argument("--output", type=Path, help="Markdown report path. Defaults under orchestration/reports.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    week_start = date.fromisoformat(args.week_start) if args.week_start else None
    ledger = summarize_economics(
        week_start=week_start,
        days=args.days,
        planner_archive=args.planner_archive,
        journal_dir=args.journal_dir,
        cloud_costs=args.cloud_costs,
        progress_root=args.progress_root,
        orch_progress_dir=args.orch_progress_dir,
    )
    output = args.output
    if output is None:
        output = DEFAULT_REPORT_DIR / f"economic_ledger_{ledger.window_start.date().isoformat()}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(ledger))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
