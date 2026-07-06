#!/usr/bin/env python3
"""Apply a JSONL batch of reviewed lab verdicts through record_verdict.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.lab import record_verdict


DEFAULT_QUEUE = Path("orchestration/lab_review_queue")
DEFAULT_RECORDS = "task_records.jsonl"
DEFAULT_VERDICTS = "review_verdicts.jsonl"
SCHEMA_VERSION = "lab_review_batch.v1"


class ApplyReviewBatchError(RuntimeError):
    """Raised for operator-facing batch-apply failures."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise ApplyReviewBatchError(f"batch file not found: {path}") from exc
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ApplyReviewBatchError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ApplyReviewBatchError(f"{path}:{lineno}: row must be an object")
        rows.append(row)
    return rows


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _required_str(row: dict[str, Any], key: str, idx: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ApplyReviewBatchError(f"batch row {idx}: {key} is required")
    return value


def _optional_float(row: dict[str, Any], key: str, idx: int) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    raise ApplyReviewBatchError(f"batch row {idx}: {key} must be a number when set")


def apply_batch(
    *,
    queue_dir: Path,
    task_records_file: Path,
    verdicts_file: Path,
    batch_rows: list[dict[str, Any]],
    allow_duplicates: bool,
    dry_run: bool,
) -> dict[str, Any]:
    applied: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, row in enumerate(batch_rows, start=1):
        try:
            job_id = _required_str(row, "job_id", idx)
            run_id = _required_str(row, "run_id", idx)
            verdict = _required_str(row, "verdict", idx)
            reviewer = _required_str(row, "reviewer", idx)
            if row.get("schema_version") not in {None, SCHEMA_VERSION}:
                raise ApplyReviewBatchError(
                    f"batch row {idx}: unsupported schema_version {row.get('schema_version')}"
                )
            item = {
                "row": idx,
                "job_id": job_id,
                "run_id": run_id,
                "verdict": verdict,
                "reviewer": reviewer,
            }
            if dry_run:
                planned.append(item)
                continue
            result = record_verdict.record_verdict(
                queue_dir=queue_dir,
                task_records_file=task_records_file,
                verdicts_file=verdicts_file,
                job_id=job_id,
                run_id=run_id,
                verdict=verdict,
                reviewer=reviewer,
                stage=row.get("stage"),
                notes=str(row.get("notes") or ""),
                confidence=_optional_float(row, "confidence", idx),
                local_output=row.get("local_output"),
                reference_output=row.get("reference_output"),
                cloud_reference_run_id=row.get("cloud_reference_run_id"),
                allow_duplicate=allow_duplicates or bool(row.get("allow_duplicate")),
            )
            applied.append({**item, "tuple_path": result["tuple_path"]})
        except (ApplyReviewBatchError, record_verdict.VerdictError) as exc:
            errors.append({"row": idx, "error": str(exc)})
    return {
        "schema_version": "lab_review_batch_apply_report.v1",
        "ok": not errors,
        "status": "ok" if not errors else "attention",
        "dry_run": dry_run,
        "rows": len(batch_rows),
        "applied": applied,
        "planned": planned,
        "errors": errors,
        "queue_dir": str(queue_dir),
        "task_records_file": str(task_records_file),
        "verdicts_file": str(verdicts_file),
    }


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    queue_dir = _resolve_path(repo_root, args.queue_dir or DEFAULT_QUEUE)
    task_records_file = _resolve_path(repo_root, args.task_records_file or queue_dir / DEFAULT_RECORDS)
    verdicts_file = _resolve_path(repo_root, args.verdicts_file or queue_dir / DEFAULT_VERDICTS)
    batch_file = _resolve_path(repo_root, args.batch_file)
    return apply_batch(
        queue_dir=queue_dir,
        task_records_file=task_records_file,
        verdicts_file=verdicts_file,
        batch_rows=_load_jsonl(batch_file),
        allow_duplicates=args.allow_duplicates,
        dry_run=args.dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-file", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--queue-dir")
    parser.add_argument("--task-records-file")
    parser.add_argument("--verdicts-file")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Accepted for consistency; output is always JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_from_args(args)
    except ApplyReviewBatchError as exc:
        print(f"apply_review_batch: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
