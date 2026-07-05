#!/usr/bin/env python3
"""Run a bounded batch of shadow lab jobs through the review queue."""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from scripts.lab import run_job


DEFAULT_JOBS_FILE = "orchestration/lab_jobs.yaml"
DEFAULT_MAX_JOBS = 2
ACTIVE_SAFE_RUNTIME_CLASSES = frozenset({"active_safe", "active_safe_deterministic"})


class ShadowBatchError(RuntimeError):
    """Raised for operator-facing shadow batch failures."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ShadowBatchError(f"jobs file not found: {path}") from exc
    if not isinstance(data, dict):
        raise ShadowBatchError(f"jobs file must contain a mapping: {path}")
    return data


def _scheduled_for(job: dict[str, Any], schedule: str) -> bool:
    job_schedule = job.get("schedule") or {}
    schedule_type = job_schedule.get("type")
    cadence = job_schedule.get("cadence")
    if schedule == "manual":
        return schedule_type in {"manual", "manual_or_nightly"}
    if schedule == "nightly":
        return schedule_type in {"nightly", "manual_or_nightly"} and cadence == "daily"
    return False


def execution_mode(job: dict[str, Any]) -> str:
    execution = job.get("execution") or {}
    return str(execution.get("mode") or "model_chat")


def is_active_safe_job(job: dict[str, Any]) -> bool:
    """Return true for read-only deterministic jobs safe during live inference."""
    schedule = job.get("schedule") or {}
    runtime_class = str(job.get("runtime_class") or schedule.get("runtime_class") or "")
    marked_active_safe = (
        job.get("active_safe") is True or runtime_class in ACTIVE_SAFE_RUNTIME_CLASSES
    )
    return (
        marked_active_safe
        and job.get("risk") == "read_only"
        and execution_mode(job) == run_job.DETERMINISTIC_COMMAND_MODE
    )


def select_jobs(
    jobs_doc: dict[str, Any],
    *,
    schedule: str,
    job_ids: list[str],
    include_disabled: bool,
    allow_gated: bool,
    max_jobs: int,
    active_safe_only: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    requested = set(job_ids)
    for job in jobs_doc.get("jobs", []) or []:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or "")
        if requested and job_id not in requested:
            continue
        if job.get("stage") != "shadow":
            continue
        if job.get("enabled") is False and not include_disabled:
            continue
        if job.get("gates") and not allow_gated:
            continue
        if not requested and not _scheduled_for(job, schedule):
            continue
        if active_safe_only and not is_active_safe_job(job):
            continue
        selected.append(job)
        if max_jobs and len(selected) >= max_jobs:
            break
    if requested:
        found = {str(job.get("job_id")) for job in selected}
        missing = sorted(requested - found)
        if missing:
            raise ShadowBatchError(f"requested jobs were not runnable: {', '.join(missing)}")
    return selected


def _run_one(
    *,
    args: argparse.Namespace,
    job_id: str,
    repo_root: Path,
    jobs_file: Path,
) -> dict[str, Any]:
    run_args = Namespace(
        job_id=job_id,
        jobs_file=str(jobs_file),
        repo_root=str(repo_root),
        queue_dir=args.queue_dir,
        repo_map=args.repo_map,
        allow_disabled=args.include_disabled,
        allow_gated=args.allow_gated,
        run_id=None,
        max_context_chars=args.max_context_chars,
        dry_run_stub=args.dry_run_stub,
        response_fixture=None,
        execute_chat=args.execute_chat,
        execute_command=args.execute_command,
        api_url=args.api_url,
        timeout_s=args.timeout_s,
        print_output=False,
    )
    result = run_job.run_from_args(run_args)
    return result.as_dict()


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    jobs_file = Path(args.jobs_file).expanduser()
    if not jobs_file.is_absolute():
        jobs_file = repo_root / jobs_file
    jobs_file = jobs_file.resolve()
    if not args.dry_run_stub and not args.execute_chat and not args.execute_command:
        raise ShadowBatchError("select --dry-run-stub, --execute-chat, or --execute-command")
    jobs_doc = _load_yaml(jobs_file)
    jobs = select_jobs(
        jobs_doc,
        schedule=args.schedule,
        job_ids=args.job_id,
        include_disabled=args.include_disabled,
        allow_gated=args.allow_gated,
        active_safe_only=getattr(args, "active_safe_only", False),
        max_jobs=args.max_jobs,
    )
    rows: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        try:
            rows.append({"job_id": job_id, "ok": True, "result": _run_one(
                args=args,
                job_id=job_id,
                repo_root=repo_root,
                jobs_file=jobs_file,
            )})
        except Exception as exc:  # noqa: BLE001
            rows.append({"job_id": job_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            if not args.continue_on_error:
                break
    return {
        "schedule": args.schedule,
        "selected": [str(job.get("job_id")) for job in jobs],
        "n_selected": len(jobs),
        "n_ok": sum(1 for row in rows if row["ok"]),
        "n_failed": sum(1 for row in rows if not row["ok"]),
        "runs": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", default=DEFAULT_JOBS_FILE)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--queue-dir")
    parser.add_argument("--repo-map", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--schedule", choices=("nightly", "manual"), default="nightly")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--allow-gated", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS)
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--dry-run-stub", action="store_true")
    parser.add_argument("--execute-chat", action="store_true")
    parser.add_argument("--execute-command", action="store_true")
    parser.add_argument(
        "--active-safe-only",
        action="store_true",
        help="Select only read-only deterministic jobs marked active-safe.",
    )
    parser.add_argument("--api-url", default=run_job.DEFAULT_API_URL)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_from_args(args)
    except ShadowBatchError as exc:
        print(f"run_shadow_jobs: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
