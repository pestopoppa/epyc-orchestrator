#!/usr/bin/env python3
"""Build a live-token coverage probe for the F1 real-task corpus lane.

The probe is intentionally offline/read-only with respect to runtime state: it
harvests progress logs through ``harvest_tasks.py``, inspects the prompt-free
output rows, and reports whether active processes predate token-telemetry code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tasks import harvest_tasks


DEFAULT_TELEMETRY_FILES = [
    Path("src/api/routes/chat_pipeline/telemetry.py"),
    Path("src/api/routes/chat_pipeline/direct_stage.py"),
    Path("orchestration/repl_memory/progress_logger.py"),
]
DEFAULT_PROGRESS_LOG_DIR = REPO_ROOT / "logs/progress"
DEFAULT_WORKLOAD_MODEL = Path("orchestration/workload_model.yaml")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def iso(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _has_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict | list | str):
        return bool(value)
    return True


def telemetry_mtimes(paths: list[Path], *, root: Path) -> dict[str, str | None]:
    mtimes: dict[str, str | None] = {}
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        if not path.exists():
            mtimes[str(raw_path)] = None
            continue
        mtimes[str(raw_path)] = iso(dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC))
    return mtimes


def latest_mtime(mtimes: dict[str, str | None]) -> dt.datetime | None:
    values = []
    for value in mtimes.values():
        if not value:
            continue
        values.append(dt.datetime.fromisoformat(value))
    return max(values) if values else None


def active_autopilot_processes(*, now: dt.datetime | None = None) -> list[dict[str, Any]]:
    """Return live AutoPilot processes from ps without requiring procfs parsing."""
    now = now or utc_now()
    proc = subprocess.run(
        ["ps", "-eo", "pid=,etimes=,cmd="],
        check=False,
        text=True,
        capture_output=True,
    )
    processes: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return processes
    for raw in proc.stdout.splitlines():
        parts = raw.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid_text, etimes_text, cmd = parts
        if "scripts/autopilot/autopilot.py start" not in cmd:
            continue
        try:
            pid = int(pid_text)
            etimes = int(float(etimes_text))
        except ValueError:
            continue
        started_at = now - dt.timedelta(seconds=etimes)
        processes.append(
            {
                "pid": pid,
                "started_at": iso(started_at),
                "elapsed_s": etimes,
                "cmd": cmd,
            }
        )
    return processes


def build_deployment_check(
    *,
    processes: list[dict[str, Any]],
    telemetry_files: dict[str, str | None],
) -> dict[str, Any]:
    telemetry_latest = latest_mtime(telemetry_files)
    stale_processes: list[int] = []
    if telemetry_latest:
        for process in processes:
            started_raw = process.get("started_at")
            if not isinstance(started_raw, str):
                continue
            try:
                started_at = dt.datetime.fromisoformat(started_raw)
            except ValueError:
                continue
            if started_at < telemetry_latest:
                stale_processes.append(int(process["pid"]))

    return {
        "active_autopilot_processes": processes,
        "active_autopilot_pid": processes[0]["pid"] if processes else None,
        "active_autopilot_started_at": processes[0]["started_at"] if processes else None,
        "telemetry_files": telemetry_files,
        "latest_telemetry_mtime": iso(telemetry_latest),
        "stale_process_for_token_telemetry": bool(stale_processes),
        "stale_autopilot_pids": stale_processes,
    }


def summarize_probe(
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    generated_at: str,
    output_path: Path,
    manifest_path: Path,
    start_date: str,
    end_date: str,
    deployment_check: dict[str, Any],
) -> dict[str, Any]:
    counts = dict(manifest.get("counts") or {})
    token_payload_rows = sum(1 for row in rows if _has_payload(row.get("tokens")))
    wall_time_rows = sum(1 for row in rows if _has_payload(row.get("wall_s")))
    prompt_text_rows = sum(1 for row in rows if _has_payload(row.get("prompt")))
    prompt_ref_rows = sum(1 for row in rows if _has_payload(row.get("prompt_ref")))
    counts["wall_time_rows"] = wall_time_rows
    counts["token_payload_rows"] = token_payload_rows
    counts["prompt_text_rows"] = prompt_text_rows
    counts["prompt_ref_rows"] = prompt_ref_rows

    token_coverage = token_payload_rows > 0
    stale = bool(deployment_check.get("stale_process_for_token_telemetry"))
    if token_coverage:
        status = "token_payload_coverage_present"
    elif stale:
        status = "blocked_on_controlled_restart_or_new_post_restart_traffic"
    else:
        status = "token_payload_coverage_missing_from_current_window"

    return {
        "schema_version": "live_token_probe_summary.v2",
        "generated_at": generated_at,
        "builder": "scripts/tasks/probe_real_task_token_coverage.py",
        "probe_window": {"start_date": start_date, "end_date": end_date},
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "counts": counts,
        "source_scan": (manifest.get("sources") or {}).get("progress", {}),
        "deployment_check": deployment_check,
        "gate_readout": {
            "class_outcome_count_gate": int(counts.get("training_eligible") or counts.get("written") or 0) >= 100,
            "wall_time_coverage": wall_time_rows > 0,
            "token_payload_coverage": token_coverage,
            "privacy_prompt_text_free": prompt_text_rows == 0,
            "status": status,
        },
        "notes": [
            "This probe is prompt-free when run with the default compact harvester options.",
            "Do not interrupt a live W4/W6 accrual run solely for this F1 token-coverage check.",
            "If active AutoPilot predates telemetry files, refresh after controlled restart or natural post-restart traffic.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    deployment = summary["deployment_check"]
    lines = [
        "# Live Real-Task Token Coverage Probe",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Window: `{summary['probe_window']['start_date']}` to `{summary['probe_window']['end_date']}`",
        f"- Output inspected: `{summary['output_path']}`",
        f"- Manifest inspected: `{summary['manifest_path']}`",
        "",
        "## Readout",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Training-eligible rows | {counts.get('training_eligible', 0)} |",
        f"| Duplicate prompt attempts collapsed | {counts.get('duplicates_collapsed', 0)} |",
        f"| Rows with wall time | {counts.get('wall_time_rows', 0)} |",
        f"| Rows with token payloads | {counts.get('token_payload_rows', 0)} |",
        f"| Prompt text rows | {counts.get('prompt_text_rows', 0)} |",
        "",
        "## Gate Readout",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for key, value in summary["gate_readout"].items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## By Class", "", "| Class | Rows |", "|---|---:|"])
    for task_class, count in sorted(dict(counts.get("by_class") or {}).items()):
        lines.append(f"| {task_class} | {count} |")

    lines.extend(["", "## Deployment Check", ""])
    lines.append(f"- Latest telemetry mtime: `{deployment.get('latest_telemetry_mtime')}`")
    lines.append(f"- Stale for token telemetry: `{deployment.get('stale_process_for_token_telemetry')}`")
    lines.append(f"- Stale AutoPilot PIDs: `{deployment.get('stale_autopilot_pids')}`")
    lines.extend(["", "| PID | Started at | Elapsed s |"])
    lines.append("|---:|---|---:|")
    for process in deployment.get("active_autopilot_processes") or []:
        lines.append(f"| {process.get('pid')} | `{process.get('started_at')}` | {process.get('elapsed_s')} |")

    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _default_output_dir(now: dt.datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return Path(tempfile.gettempdir()) / f"f1-live-token-probe-{stamp}"


def run(args: argparse.Namespace) -> dict[str, str]:
    now = utc_now()
    start_date = args.start_date or now.date().isoformat()
    end_date = args.end_date or start_date
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_output_dir(now)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "real_tasks.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_json = Path(args.output_json).expanduser() if args.output_json else output_dir / "summary.json"
    summary_md = Path(args.output_md).expanduser() if args.output_md else output_dir / "summary.md"

    harvest_args = Namespace(
        progress_log_dir=str(Path(args.progress_log_dir).expanduser()),
        workload_model=str(Path(args.workload_model).expanduser()),
        output=str(rows_path),
        manifest=str(manifest_path),
        start_date=start_date,
        end_date=end_date,
        lab_task_records=[],
        historical_conversation_paths=[],
        include_historical_sidechains=False,
        limit=0,
        include_open=False,
        exclude_synthetic_like=True,
        dedupe_prompt=True,
        omit_prompt_text=True,
        compact_evidence=True,
        training_eligible_only=True,
    )
    harvest_tasks.run(harvest_args)
    manifest = load_json(manifest_path)
    rows = load_jsonl(rows_path)
    telemetry_files = telemetry_mtimes([Path(p) for p in args.telemetry_file], root=Path.cwd())
    deployment_check = build_deployment_check(
        processes=active_autopilot_processes(),
        telemetry_files=telemetry_files,
    )
    generated_at = iso(now) or ""
    summary = summarize_probe(
        manifest=manifest,
        rows=rows,
        generated_at=generated_at,
        output_path=rows_path,
        manifest_path=manifest_path,
        start_date=start_date,
        end_date=end_date,
        deployment_check=deployment_check,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    return {
        "output": str(rows_path),
        "manifest": str(manifest_path),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-log-dir", default=str(DEFAULT_PROGRESS_LOG_DIR))
    parser.add_argument("--workload-model", default=str(DEFAULT_WORKLOAD_MODEL))
    parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD lower bound; defaults to today UTC")
    parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD upper bound; defaults to start date")
    parser.add_argument("--output-dir", default="", help="Directory for harvested local-private rows and default reports")
    parser.add_argument("--output-json", default="", help="Optional summary JSON path")
    parser.add_argument("--output-md", default="", help="Optional summary Markdown path")
    parser.add_argument(
        "--telemetry-file",
        action="append",
        default=[str(path) for path in DEFAULT_TELEMETRY_FILES],
        help="Telemetry-bearing file used for stale-process comparison; repeatable",
    )
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
