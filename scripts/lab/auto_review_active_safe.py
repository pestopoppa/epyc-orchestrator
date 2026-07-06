#!/usr/bin/env python3
"""Deterministically close active-safe lab review rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.lab import record_verdict
from scripts.lab import run_shadow_jobs


DEFAULT_JOBS_FILE = Path("orchestration/lab_jobs.yaml")
DEFAULT_QUEUE = Path("orchestration/lab_review_queue")
DEFAULT_RECORDS = "task_records.jsonl"
DEFAULT_VERDICTS = "review_verdicts.jsonl"


class AutoReviewError(RuntimeError):
    """Raised for operator-facing auto-review failures."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise AutoReviewError(f"jobs file not found: {path}") from exc
    if not isinstance(data, dict):
        raise AutoReviewError(f"jobs file must contain a mapping: {path}")
    return data


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AutoReviewError(f"output artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AutoReviewError(f"invalid JSON file {path}: {exc}") from exc


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
            raise AutoReviewError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise AutoReviewError(f"{path}:{lineno}: JSONL row must be an object")
        rows.append(row)
    return rows


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _active_safe_job_ids(jobs_doc: dict[str, Any]) -> set[str]:
    return {
        str(job["job_id"])
        for job in jobs_doc.get("jobs", []) or []
        if isinstance(job, dict)
        and job.get("job_id")
        and run_shadow_jobs.is_active_safe_job(job)
    }


def _existing_run_ids(verdicts: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("run_id")) for row in verdicts if row.get("run_id")}


def _output_path(queue_dir: Path, record: dict[str, Any]) -> Path:
    raw = ((record.get("artifacts") or {}).get("output"))
    if not raw:
        raise AutoReviewError(
            f"task_record for {record.get('job_id')}/{record.get('run_id')} has no output artifact"
        )
    path = Path(str(raw))
    if not path.is_absolute():
        path = queue_dir / path
    return path.resolve()


def _verdict_for_payload(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "reject", "automated active-safe review rejected non-object JSON output"
    if payload.get("ok") is True:
        return "accept", "automated active-safe review accepted output with ok=true"
    return "reject", "automated active-safe review rejected output without ok=true"


def auto_review(
    *,
    jobs_doc: dict[str, Any],
    queue_dir: Path,
    task_records_file: Path,
    verdicts_file: Path,
    task_records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    apply: bool,
    max_items: int,
) -> dict[str, Any]:
    active_safe_ids = _active_safe_job_ids(jobs_doc)
    existing = _existing_run_ids(verdicts)
    planned: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for record in task_records:
        job_id = str(record.get("job_id") or "")
        run_id = str(record.get("run_id") or "")
        if not job_id or not run_id:
            skipped.append({"job_id": job_id, "run_id": run_id, "reason": "missing job_id/run_id"})
            continue
        if run_id in existing:
            continue
        if job_id not in active_safe_ids:
            continue
        if len(planned) + len(applied) >= max_items:
            skipped.append({"job_id": job_id, "run_id": run_id, "reason": "max_items reached"})
            continue
        try:
            output_path = _output_path(queue_dir, record)
            payload = _load_json(output_path)
            verdict, notes = _verdict_for_payload(payload)
            item = {
                "job_id": job_id,
                "run_id": run_id,
                "verdict": verdict,
                "reviewer": "automated",
                "confidence": 1.0,
                "notes": notes,
                "output_path": _safe_rel(output_path, queue_dir),
                "write_gold_tuple": False,
            }
            if not apply:
                planned.append(item)
                continue
            result = record_verdict.record_verdict(
                queue_dir=queue_dir,
                task_records_file=task_records_file,
                verdicts_file=verdicts_file,
                job_id=job_id,
                run_id=run_id,
                verdict=verdict,
                reviewer="automated",
                stage=record.get("stage"),
                notes=notes,
                confidence=1.0,
                local_output=str(output_path),
                reference_output=None,
                cloud_reference_run_id=None,
                allow_duplicate=False,
                write_gold_tuple=False,
            )
            applied.append({**item, "verdict_artifact_path": result["verdict_artifact_path"]})
            existing.add(run_id)
        except (AutoReviewError, record_verdict.VerdictError) as exc:
            errors.append({"job_id": job_id, "run_id": run_id, "error": str(exc)})

    return {
        "schema_version": "lab_active_safe_auto_review_report.v1",
        "ok": not errors,
        "status": "ok" if not errors else "attention",
        "applied_mode": apply,
        "active_safe_job_ids": sorted(active_safe_ids),
        "planned": planned,
        "applied": applied,
        "skipped": skipped,
        "errors": errors,
        "summary": {
            "planned": len(planned),
            "applied": len(applied),
            "skipped": len(skipped),
            "errors": len(errors),
        },
        "queue_dir": str(queue_dir),
        "task_records_file": str(task_records_file),
        "verdicts_file": str(verdicts_file),
    }


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    jobs_file = _resolve_path(repo_root, args.jobs_file)
    queue_dir = _resolve_path(repo_root, args.queue_dir or DEFAULT_QUEUE)
    records_file = _resolve_path(repo_root, args.task_records_file or queue_dir / DEFAULT_RECORDS)
    verdicts_file = _resolve_path(repo_root, args.verdicts_file or queue_dir / DEFAULT_VERDICTS)
    return auto_review(
        jobs_doc=_load_yaml(jobs_file),
        queue_dir=queue_dir,
        task_records_file=records_file,
        verdicts_file=verdicts_file,
        task_records=_load_jsonl(records_file),
        verdicts=_load_jsonl(verdicts_file),
        apply=args.apply,
        max_items=args.max_items,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--jobs-file", default=str(DEFAULT_JOBS_FILE))
    parser.add_argument("--queue-dir")
    parser.add_argument("--task-records-file")
    parser.add_argument("--verdicts-file")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--apply", action="store_true", help="Write automated verdicts.")
    parser.add_argument("--json", action="store_true", help="Accepted for consistency; output is always JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_from_args(args)
    except AutoReviewError as exc:
        print(f"auto_review_active_safe: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
