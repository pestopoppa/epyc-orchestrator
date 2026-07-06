#!/usr/bin/env python3
"""Report self-running-lab shadow and promotion readiness without mutating state."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.lab import promote_job
from scripts.lab import run_shadow_jobs


DEFAULT_JOBS_FILE = "orchestration/lab_jobs.yaml"
DEFAULT_QUEUE = Path("orchestration/lab_review_queue")
DEFAULT_RECORDS = "task_records.jsonl"
DEFAULT_VERDICTS = "review_verdicts.jsonl"
AUTOPILOT_CMD_MARKER = "scripts/autopilot/autopilot.py start"
LLAMA_CMD_MARKER = "llama-server"


class ReadinessError(RuntimeError):
    """Raised for operator-facing readiness-report failures."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ReadinessError(f"jobs file not found: {path}") from exc
    if not isinstance(data, dict):
        raise ReadinessError(f"jobs file must contain a mapping: {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReadinessError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ReadinessError(f"{path}:{lineno}: JSONL row must be an object")
        rows.append(row)
    return rows


def _resolve_repo_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _records_for(records: list[dict[str, Any]], job_id: str) -> list[dict[str, Any]]:
    return [row for row in records if row.get("job_id") == job_id]


def _verdicts_for(verdicts: list[dict[str, Any]], job_id: str) -> list[dict[str, Any]]:
    return [row for row in verdicts if row.get("job_id") == job_id]


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("stage") or "unknown")
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _gold_tuple_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("tuple_path") or row.get("gold_tuple_path"))


def _review_status(
    records: list[dict[str, Any]], verdicts: list[dict[str, Any]]
) -> dict[str, Any]:
    verdict_run_ids = {str(row.get("run_id")) for row in verdicts if row.get("run_id")}
    pending = [
        str(row.get("run_id"))
        for row in records
        if row.get("run_id") and str(row.get("run_id")) not in verdict_run_ids
    ]
    return {
        "pending": len(pending),
        "pending_run_ids": pending[:25],
        "pending_run_ids_truncated": len(pending) > 25,
    }


def _active_processes(marker: str) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    self_pid = os.getpid()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_raw, _, cmd = line.partition(" ")
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        if pid == self_pid or marker not in cmd:
            continue
        if marker == LLAMA_CMD_MARKER and not (
            cmd.startswith(f"{LLAMA_CMD_MARKER} ") or f"/{LLAMA_CMD_MARKER} " in cmd
        ):
            continue
        rows.append({"pid": pid, "cmd": cmd})
    return rows


def _quiet_window_status() -> dict[str, Any]:
    autopilot = _active_processes(AUTOPILOT_CMD_MARKER)
    llama = _active_processes(LLAMA_CMD_MARKER)
    blockers: list[str] = []
    if os.environ.get("NIGHTSHIFT_INFERENCE_ACTIVE") == "1":
        blockers.append(
            "NIGHTSHIFT_INFERENCE_ACTIVE=1"
            f" ({os.environ.get('NIGHTSHIFT_INFERENCE_RSS_GB', 'unknown')}GB RSS)"
        )
    if autopilot:
        blockers.append(f"active AutoPilot process count: {len(autopilot)}")
    if llama:
        blockers.append(f"active llama-server process count: {len(llama)}")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "active_autopilot_processes": autopilot,
        "active_llama_process_count": len(llama),
        "active_llama_process_examples": llama[:5],
    }


def _promotion_decision(
    *,
    jobs_doc: dict[str, Any],
    job_id: str,
    target_stage: str,
    queue_dir: Path,
    task_records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    min_shadow_runs: int,
    min_reviewed_runs: int,
    autonomous_accept_rate: float,
) -> dict[str, Any]:
    decision = promote_job.evaluate_promotion(
        jobs_doc=jobs_doc,
        job_id=job_id,
        target_stage=target_stage,
        task_records=task_records,
        verdicts=verdicts,
        queue_dir=queue_dir,
        min_shadow_runs=min_shadow_runs,
        min_reviewed_runs=min_reviewed_runs,
        autonomous_accept_rate=autonomous_accept_rate,
    )
    return decision.as_dict()


def _job_readiness(
    *,
    jobs_doc: dict[str, Any],
    job: dict[str, Any],
    queue_dir: Path,
    task_records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    min_shadow_runs: int,
    min_reviewed_runs: int,
    autonomous_accept_rate: float,
) -> dict[str, Any]:
    job_id = str(job.get("job_id") or "")
    job_records = _records_for(task_records, job_id)
    job_verdicts = _verdicts_for(verdicts, job_id)
    review = _review_status(job_records, job_verdicts)
    promotion: dict[str, Any] = {}
    if job_id:
        promotion["reviewed"] = _promotion_decision(
            jobs_doc=jobs_doc,
            job_id=job_id,
            target_stage="reviewed",
            queue_dir=queue_dir,
            task_records=task_records,
            verdicts=verdicts,
            min_shadow_runs=min_shadow_runs,
            min_reviewed_runs=min_reviewed_runs,
            autonomous_accept_rate=autonomous_accept_rate,
        )
        promotion["autonomous"] = _promotion_decision(
            jobs_doc=jobs_doc,
            job_id=job_id,
            target_stage="autonomous",
            queue_dir=queue_dir,
            task_records=task_records,
            verdicts=verdicts,
            min_shadow_runs=min_shadow_runs,
            min_reviewed_runs=min_reviewed_runs,
            autonomous_accept_rate=autonomous_accept_rate,
        )
    return {
        "job_id": job_id,
        "title": job.get("title"),
        "stage": job.get("stage"),
        "enabled": job.get("enabled") is not False,
        "risk": job.get("risk"),
        "runtime_class": job.get("runtime_class") or (job.get("schedule") or {}).get("runtime_class"),
        "execution_mode": run_shadow_jobs.execution_mode(job),
        "active_safe": run_shadow_jobs.is_active_safe_job(job),
        "requires_quiet_window": not run_shadow_jobs.is_active_safe_job(job),
        "gated": bool(job.get("gates")),
        "gates": list(job.get("gates") or []),
        "scheduled_manual": run_shadow_jobs._scheduled_for(job, "manual"),
        "scheduled_nightly": run_shadow_jobs._scheduled_for(job, "nightly"),
        "task_records": {
            "total": len(job_records),
            "by_stage": _stage_counts(job_records),
        },
        "verdicts": {
            "total": len(job_verdicts),
            "by_stage": _stage_counts(job_verdicts),
            "gold_tuples": _gold_tuple_count(job_verdicts),
        },
        "review": review,
        "promotion": promotion,
    }


def build_report(
    *,
    jobs_doc: dict[str, Any],
    jobs_file: Path,
    queue_dir: Path,
    records_file: Path,
    verdicts_file: Path,
    task_records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    min_shadow_runs: int,
    min_reviewed_runs: int,
    autonomous_accept_rate: float,
    quiet_window: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    jobs = [job for job in jobs_doc.get("jobs", []) or [] if isinstance(job, dict)]
    runnable_nightly = run_shadow_jobs.select_jobs(
        jobs_doc,
        schedule="nightly",
        job_ids=[],
        include_disabled=False,
        allow_gated=False,
        max_jobs=0,
    )
    runnable_manual = run_shadow_jobs.select_jobs(
        jobs_doc,
        schedule="manual",
        job_ids=[],
        include_disabled=False,
        allow_gated=False,
        max_jobs=0,
    )
    active_safe_nightly = [
        job for job in runnable_nightly if run_shadow_jobs.is_active_safe_job(job)
    ]
    quiet_window_nightly = [
        job for job in runnable_nightly if not run_shadow_jobs.is_active_safe_job(job)
    ]
    rows = [
        _job_readiness(
            jobs_doc=jobs_doc,
            job=job,
            queue_dir=queue_dir,
            task_records=task_records,
            verdicts=verdicts,
            min_shadow_runs=min_shadow_runs,
            min_reviewed_runs=min_reviewed_runs,
            autonomous_accept_rate=autonomous_accept_rate,
        )
        for job in jobs
    ]
    promotion_ready = [
        row["job_id"]
        for row in rows
        if row["promotion"].get("reviewed", {}).get("eligible")
        or row["promotion"].get("autonomous", {}).get("eligible")
    ]
    pending_reviews = sum(row["review"]["pending"] for row in rows)
    pending_review_job_ids = [
        row["job_id"] for row in rows if row["review"]["pending"] > 0
    ]
    quiet = quiet_window or {
        "ready": None,
        "blockers": [],
        "active_autopilot_processes": [],
        "active_llama_process_count": None,
        "active_llama_process_examples": [],
    }
    quiet_ready = quiet.get("ready") is True
    return {
        "schema_version": "lab_readiness_report.v1",
        "generated_at": generated_at or utc_now(),
        "jobs_file": str(jobs_file),
        "queue_dir": str(queue_dir),
        "task_records_file": str(records_file),
        "verdicts_file": str(verdicts_file),
        "summary": {
            "jobs_total": len(jobs),
            "enabled_jobs": sum(1 for job in jobs if job.get("enabled") is not False),
            "shadow_jobs": sum(1 for job in jobs if job.get("stage") == "shadow"),
            "nightly_runnable": len(runnable_nightly),
            "nightly_active_safe_runnable": len(active_safe_nightly),
            "nightly_active_safe_ready_now": len(active_safe_nightly),
            "nightly_quiet_window_runnable": len(quiet_window_nightly),
            "nightly_quiet_window_ready_now": len(quiet_window_nightly) if quiet_ready else 0,
            "nightly_ready_now": len(active_safe_nightly)
            + (len(quiet_window_nightly) if quiet_ready else 0),
            "manual_runnable": len(runnable_manual),
            "task_records": len(task_records),
            "verdicts": len(verdicts),
            "gold_tuples": _gold_tuple_count(verdicts),
            "pending_reviews": pending_reviews,
            "pending_review_job_ids": pending_review_job_ids,
            "promotion_ready": len(promotion_ready),
            "promotion_ready_job_ids": promotion_ready,
        },
        "quiet_window": quiet,
        "jobs": rows,
    }


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    jobs_file = Path(args.jobs_file).expanduser()
    if not jobs_file.is_absolute():
        jobs_file = repo_root / jobs_file
    jobs_file = jobs_file.resolve()
    queue_dir = Path(args.queue_dir).expanduser() if args.queue_dir else DEFAULT_QUEUE
    if not queue_dir.is_absolute():
        queue_dir = repo_root / queue_dir
    queue_dir = queue_dir.resolve()
    records_file = _resolve_repo_path(
        repo_root, args.task_records_file or queue_dir / DEFAULT_RECORDS
    )
    verdicts_file = _resolve_repo_path(
        repo_root, args.verdicts_file or queue_dir / DEFAULT_VERDICTS
    )
    return build_report(
        jobs_doc=_load_yaml(jobs_file),
        jobs_file=jobs_file,
        queue_dir=queue_dir,
        records_file=records_file,
        verdicts_file=verdicts_file,
        task_records=_load_jsonl(records_file),
        verdicts=_load_jsonl(verdicts_file),
        min_shadow_runs=args.min_shadow_runs,
        min_reviewed_runs=args.min_reviewed_runs,
        autonomous_accept_rate=args.autonomous_accept_rate,
        quiet_window=(
            None if getattr(args, "skip_process_check", False) else _quiet_window_status()
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", default=DEFAULT_JOBS_FILE)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--queue-dir")
    parser.add_argument("--task-records-file")
    parser.add_argument("--verdicts-file")
    parser.add_argument("--min-shadow-runs", type=int, default=10)
    parser.add_argument("--min-reviewed-runs", type=int, default=20)
    parser.add_argument("--autonomous-accept-rate", type=float, default=0.90)
    parser.add_argument("--skip-process-check", action="store_true")
    parser.add_argument("--json", action="store_true", help="Accepted for consistency; output is always JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_from_args(args)
    except (ReadinessError, promote_job.PromotionError, run_shadow_jobs.ShadowBatchError) as exc:
        print(f"readiness_report: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
