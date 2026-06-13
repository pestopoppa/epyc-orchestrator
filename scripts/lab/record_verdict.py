#!/usr/bin/env python3
"""Record a reviewed lab-job verdict and F3 gold tuple."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_QUEUE = Path("orchestration/lab_review_queue")
DEFAULT_RECORDS = "task_records.jsonl"
DEFAULT_VERDICTS = "review_verdicts.jsonl"
VERDICTS = ("accept", "reject")
REVIEWERS = ("operator", "cloud_reference", "automated")


class VerdictError(RuntimeError):
    """Raised for operator-facing verdict capture failures."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise VerdictError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VerdictError(f"invalid JSON file {path}: {exc}") from exc


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
            raise VerdictError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise VerdictError(f"{path}:{lineno}: JSONL row must be an object")
        rows.append(row)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _task_record_for(
    *,
    task_records: list[dict[str, Any]],
    job_id: str,
    run_id: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in task_records
        if row.get("job_id") == job_id and row.get("run_id") == run_id
    ]
    if not matches:
        raise VerdictError(f"task_record not found for {job_id}/{run_id}")
    return matches[-1]


def _existing_verdict(
    *,
    verdicts: list[dict[str, Any]],
    job_id: str,
    run_id: str,
) -> dict[str, Any] | None:
    for row in reversed(verdicts):
        if row.get("job_id") == job_id and row.get("run_id") == run_id:
            return row
    return None


def _resolve_queue_path(queue_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = queue_dir / candidate
    return candidate.resolve()


def _local_output_path(
    *,
    queue_dir: Path,
    task_record: dict[str, Any],
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    artifact_path = ((task_record.get("artifacts") or {}).get("output"))
    resolved = _resolve_queue_path(queue_dir, artifact_path)
    if resolved is None:
        raise VerdictError("task_record has no output artifact; pass --local-output")
    return resolved


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def record_verdict(
    *,
    queue_dir: Path,
    task_records_file: Path,
    verdicts_file: Path,
    job_id: str,
    run_id: str,
    verdict: str,
    reviewer: str,
    stage: str | None,
    notes: str,
    confidence: float | None,
    local_output: str | None,
    reference_output: str | None,
    cloud_reference_run_id: str | None,
    allow_duplicate: bool,
) -> dict[str, Any]:
    task_records = _load_jsonl(task_records_file)
    verdict_rows = _load_jsonl(verdicts_file)
    if _existing_verdict(verdicts=verdict_rows, job_id=job_id, run_id=run_id) and not allow_duplicate:
        raise VerdictError(f"verdict already exists for {job_id}/{run_id}; pass --allow-duplicate")
    task_record = _task_record_for(task_records=task_records, job_id=job_id, run_id=run_id)
    resolved_stage = stage or str(task_record.get("stage") or "")
    if not resolved_stage:
        raise VerdictError("stage is required when task_record has no stage")
    local_path = _local_output_path(
        queue_dir=queue_dir,
        task_record=task_record,
        explicit_path=local_output,
    )
    local_payload = _load_json(local_path)
    reference_payload = None
    reference_path = None
    if reference_output:
        reference_path = Path(reference_output).expanduser().resolve()
        reference_payload = _load_json(reference_path)
    if reviewer == "cloud_reference" and reference_payload is None:
        raise VerdictError("--reviewer cloud_reference requires --reference-output")
    captured_at = utc_now()
    tuple_path = queue_dir / "gold_tuples" / job_id / f"{run_id}.json"
    gold_tuple = {
        "schema_version": "lab_gold_tuple.v1",
        "job_id": job_id,
        "run_id": run_id,
        "captured_at": captured_at,
        "stage": resolved_stage,
        "verdict": verdict,
        "reviewer": reviewer,
        "confidence": confidence,
        "notes": notes,
        "local_output": {
            "path": _safe_rel(local_path, queue_dir),
            "payload": local_payload,
        },
        "reference_output": {
            "path": _safe_rel(reference_path, queue_dir) if reference_path else None,
            "payload": reference_payload,
        },
        "task_record": task_record,
    }
    _atomic_write_json(tuple_path, gold_tuple)
    verdict_row = {
        "schema_version": "lab_review_verdict.v1",
        "job_id": job_id,
        "run_id": run_id,
        "stage": resolved_stage,
        "verdict": verdict,
        "reviewer": reviewer,
        "reviewed_at": captured_at,
        "confidence": confidence,
        "notes": notes,
        "tuple_path": _safe_rel(tuple_path, queue_dir),
    }
    if reviewer == "cloud_reference":
        verdict_row["reference_type"] = "cloud_reference"
        verdict_row["cloud_reference_run_id"] = cloud_reference_run_id or f"cloud-{run_id}"
    _append_jsonl(verdicts_file, verdict_row)
    return {
        "job_id": job_id,
        "run_id": run_id,
        "verdict": verdict,
        "tuple_path": str(tuple_path),
        "verdicts_file": str(verdicts_file),
    }


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    queue_dir = Path(args.queue_dir).expanduser() if args.queue_dir else DEFAULT_QUEUE
    if not queue_dir.is_absolute():
        queue_dir = repo_root / queue_dir
    queue_dir = queue_dir.resolve()
    task_records_file = Path(args.task_records_file or queue_dir / DEFAULT_RECORDS)
    verdicts_file = Path(args.verdicts_file or queue_dir / DEFAULT_VERDICTS)
    return record_verdict(
        queue_dir=queue_dir,
        task_records_file=task_records_file,
        verdicts_file=verdicts_file,
        job_id=args.job_id,
        run_id=args.run_id,
        verdict=args.verdict,
        reviewer=args.reviewer,
        stage=args.stage,
        notes=args.notes or "",
        confidence=args.confidence,
        local_output=args.local_output,
        reference_output=args.reference_output,
        cloud_reference_run_id=args.cloud_reference_run_id,
        allow_duplicate=args.allow_duplicate,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--verdict", required=True, choices=VERDICTS)
    parser.add_argument("--reviewer", required=True, choices=REVIEWERS)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--queue-dir")
    parser.add_argument("--task-records-file")
    parser.add_argument("--verdicts-file")
    parser.add_argument("--stage")
    parser.add_argument("--notes")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--local-output")
    parser.add_argument("--reference-output")
    parser.add_argument("--cloud-reference-run-id")
    parser.add_argument("--allow-duplicate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_from_args(args)
    except VerdictError as exc:
        print(f"record_verdict: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
