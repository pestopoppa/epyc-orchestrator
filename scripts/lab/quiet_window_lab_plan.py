#!/usr/bin/env python3
"""Plan the first quiet-window self-running-lab batch without executing it."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.lab import readiness_report
from scripts.lab import run_shadow_jobs


DEFAULT_JOBS_FILE = "orchestration/lab_jobs.yaml"
DEFAULT_QUEUE = Path("orchestration/lab_review_queue")
DEFAULT_MAX_JOBS = 2


class QuietWindowLabPlanError(RuntimeError):
    """Raised for operator-facing lab-plan failures."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise QuietWindowLabPlanError(f"jobs file not found: {path}") from exc
    if not isinstance(data, dict):
        raise QuietWindowLabPlanError(f"jobs file must contain a mapping: {path}")
    return data


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _selected_job_summaries(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for job in jobs:
        schedule = job.get("schedule") or {}
        out.append(
            {
                "job_id": str(job.get("job_id") or ""),
                "title": job.get("title"),
                "risk": job.get("risk"),
                "stage": job.get("stage"),
                "execution_mode": run_shadow_jobs.execution_mode(job),
                "scheduled_nightly": run_shadow_jobs._scheduled_for(job, "nightly"),
                "max_runtime_s": schedule.get("max_runtime_s"),
                "context_modes": list((job.get("input_spec") or {}).get("context_modes") or []),
            }
        )
    return out


def build_plan(
    *,
    repo_root: Path,
    jobs_file: Path,
    queue_dir: Path,
    schedule: str,
    max_jobs: int,
    api_url: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    jobs_doc = _load_yaml(jobs_file)
    jobs = run_shadow_jobs.select_jobs(
        jobs_doc,
        schedule=schedule,
        job_ids=[],
        include_disabled=False,
        allow_gated=False,
        max_jobs=max_jobs,
        quiet_window_only=True,
    )
    quiet_window = readiness_report._quiet_window_status()
    selected_jobs = _selected_job_summaries(jobs)
    selected_runtime_caps = [
        float(row["max_runtime_s"])
        for row in selected_jobs
        if row.get("max_runtime_s") is not None
    ]
    effective_timeout_s = timeout_s
    if effective_timeout_s is None:
        effective_timeout_s = max(selected_runtime_caps, default=300.0)
    run_command = [
        "uv",
        "run",
        "python",
        "scripts/lab/run_shadow_jobs.py",
        "--quiet-window-only",
        "--execute-chat",
        "--continue-on-error",
        "--schedule",
        schedule,
        "--max-jobs",
        str(max_jobs),
        "--queue-dir",
        str(queue_dir),
        "--api-url",
        api_url,
        "--timeout-s",
        str(effective_timeout_s),
    ]
    review_command = [
        "uv",
        "run",
        "python",
        "scripts/lab/review_queue_report.py",
        "--json",
        "--queue-dir",
        str(queue_dir),
    ]
    batch_template = [
        "uv",
        "run",
        "python",
        "scripts/lab/apply_review_batch.py",
        "--batch",
        "<reviewed-verdicts.jsonl>",
        "--queue-dir",
        str(queue_dir),
    ]
    blockers = list(quiet_window.get("blockers") or [])
    if not selected_jobs:
        blockers.append("no quiet-window lab jobs selected")
    status = "ready" if not blockers else ("no_jobs" if not selected_jobs else "blocked")
    next_steps = [
        "Stop AutoPilot and wait for live llama-server inference to quiesce.",
        "Run the quiet-window batch command once; it writes lab_task_record.v1 rows and review artifacts.",
        "Run the review-queue report command and review each pending item.",
        "Apply reviewed verdicts with the batch-apply template so F2 tuples feed F3 gold data.",
    ]
    return {
        "schema_version": "quiet_window_lab_plan.v1",
        "generated_at": utc_now(),
        "ok": status in {"ready", "blocked", "no_jobs"},
        "status": status,
        "blockers": blockers,
        "quiet_window": quiet_window,
        "queue_dir": str(queue_dir),
        "timeout_s": effective_timeout_s,
        "selected_jobs": selected_jobs,
        "commands": {
            "run_quiet_window_batch": _cmd(run_command),
            "review_pending_items": _cmd(review_command),
            "apply_reviewed_verdict_batch_template": _cmd(batch_template),
        },
        "next_steps": next_steps,
    }


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    jobs_file = _resolve_path(repo_root, args.jobs_file)
    queue_dir = _resolve_path(repo_root, args.queue_dir)
    return build_plan(
        repo_root=repo_root,
        jobs_file=jobs_file,
        queue_dir=queue_dir,
        schedule=args.schedule,
        max_jobs=args.max_jobs,
        api_url=args.api_url,
        timeout_s=args.timeout_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", default=DEFAULT_JOBS_FILE)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--queue-dir", default=str(DEFAULT_QUEUE))
    parser.add_argument("--schedule", choices=("nightly", "manual"), default="nightly")
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS)
    parser.add_argument("--api-url", default=run_shadow_jobs.run_job.DEFAULT_API_URL)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Per-job chat timeout; defaults to the largest selected job max_runtime_s.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_from_args(args)
    except (QuietWindowLabPlanError, run_shadow_jobs.ShadowBatchError) as exc:
        print(f"quiet_window_lab_plan: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print("# Quiet-Window Lab Plan")
        print()
        print(f"- status: {report['status']}")
        print(f"- selected jobs: {', '.join(row['job_id'] for row in report['selected_jobs']) or '(none)'}")
        if report["blockers"]:
            print(f"- blockers: {'; '.join(report['blockers'])}")
        print()
        print("## Commands")
        for label, command in report["commands"].items():
            print(f"- {label}: `{command}`")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
