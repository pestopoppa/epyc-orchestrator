#!/usr/bin/env python3
"""Report pending self-running-lab reviews without recording verdicts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.lab import run_shadow_jobs


DEFAULT_JOBS_FILE = Path("orchestration/lab_jobs.yaml")
DEFAULT_QUEUE = Path("orchestration/lab_review_queue")
DEFAULT_RECORDS = "task_records.jsonl"
DEFAULT_VERDICTS = "review_verdicts.jsonl"


class ReviewQueueReportError(RuntimeError):
    """Raised for operator-facing review queue report failures."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ReviewQueueReportError(f"jobs file not found: {path}") from exc
    if not isinstance(data, dict):
        raise ReviewQueueReportError(f"jobs file must contain a mapping: {path}")
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
            raise ReviewQueueReportError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ReviewQueueReportError(f"{path}:{lineno}: JSONL row must be an object")
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


def _job_classes(jobs_doc: dict[str, Any]) -> dict[str, str]:
    classes: dict[str, str] = {}
    for job in jobs_doc.get("jobs", []) or []:
        if not isinstance(job, dict) or not job.get("job_id"):
            continue
        classes[str(job["job_id"])] = (
            "active_safe_deterministic"
            if run_shadow_jobs.is_active_safe_job(job)
            else "review_candidate"
        )
    return classes


def _latest_verdict_run_ids(verdicts: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("run_id")) for row in verdicts if row.get("run_id")}


def _record_output_path(queue_dir: Path, record: dict[str, Any]) -> Path | None:
    raw = ((record.get("artifacts") or {}).get("output"))
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = queue_dir / path
    return path.resolve()


def _record_command(
    *,
    job_id: str,
    run_id: str,
    reviewer: str,
    verdict: str,
    queue_dir: Path,
    reference_placeholder: str | None = None,
) -> str:
    argv = [
        "uv",
        "run",
        "python",
        "scripts/lab/record_verdict.py",
        "--job-id",
        job_id,
        "--run-id",
        run_id,
        "--verdict",
        verdict,
        "--reviewer",
        reviewer,
        "--queue-dir",
        str(queue_dir),
    ]
    if reviewer == "cloud_reference":
        argv.extend(["--reference-output", reference_placeholder or "<cloud-reference-output.json>"])
    return " ".join(shlex.quote(arg) for arg in argv)


def _batch_template_row(item: dict[str, Any]) -> dict[str, Any]:
    row = {
        "schema_version": "lab_review_batch.v1",
        "job_id": item["job_id"],
        "run_id": item["run_id"],
        "verdict": "<accept|reject>",
        "reviewer": item["next_reviewer"],
        "stage": item["stage"],
        "confidence": None,
        "notes": "",
        "local_output": item["output_path"],
        "reference_output": (
            "<cloud-reference-output.json>"
            if item["next_reviewer"] == "cloud_reference"
            else None
        ),
    }
    if item.get("record_class") == "active_safe_deterministic":
        row["write_gold_tuple"] = False
    return row


def _pending_items(
    *,
    queue_dir: Path,
    task_records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    job_classes: dict[str, str],
    max_items: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    verdict_run_ids = _latest_verdict_run_ids(verdicts)
    pending: list[dict[str, Any]] = []
    counts_by_class = {"active_safe_deterministic": 0, "review_candidate": 0}
    for record in task_records:
        run_id = str(record.get("run_id") or "")
        job_id = str(record.get("job_id") or "")
        if not run_id or not job_id or run_id in verdict_run_ids:
            continue
        record_class = job_classes.get(job_id, "review_candidate")
        counts_by_class[record_class] = counts_by_class.get(record_class, 0) + 1
        if len(pending) >= max_items:
            continue
        output_path = _record_output_path(queue_dir, record)
        output_exists = output_path.is_file() if output_path is not None else False
        next_reviewer = (
            "cloud_reference" if record_class == "review_candidate" else "automated"
        )
        pending.append(
            {
                "job_id": job_id,
                "run_id": run_id,
                "stage": str(record.get("stage") or ""),
                "record_class": record_class,
                "output_path": _safe_rel(output_path, queue_dir) if output_path else None,
                "output_exists": output_exists,
                "next_reviewer": next_reviewer,
                "operator_accept_command": _record_command(
                    job_id=job_id,
                    run_id=run_id,
                    reviewer="operator",
                    verdict="accept",
                    queue_dir=queue_dir,
                ),
                "operator_reject_command": _record_command(
                    job_id=job_id,
                    run_id=run_id,
                    reviewer="operator",
                    verdict="reject",
                    queue_dir=queue_dir,
                ),
                "cloud_reference_accept_command": _record_command(
                    job_id=job_id,
                    run_id=run_id,
                    reviewer="cloud_reference",
                    verdict="accept",
                    queue_dir=queue_dir,
                ),
                "cloud_reference_reject_command": _record_command(
                    job_id=job_id,
                    run_id=run_id,
                    reviewer="cloud_reference",
                    verdict="reject",
                    queue_dir=queue_dir,
                ),
            }
        )
    return pending, dict(sorted(counts_by_class.items()))


def build_report(
    *,
    jobs_doc: dict[str, Any],
    queue_dir: Path,
    records_file: Path,
    verdicts_file: Path,
    task_records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    max_items: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    job_classes = _job_classes(jobs_doc)
    pending, pending_by_class = _pending_items(
        queue_dir=queue_dir,
        task_records=task_records,
        verdicts=verdicts,
        job_classes=job_classes,
        max_items=max_items,
    )
    missing_outputs = [
        {"job_id": item["job_id"], "run_id": item["run_id"], "output_path": item["output_path"]}
        for item in pending
        if not item["output_exists"]
    ]
    review_batch_template = [_batch_template_row(item) for item in pending]
    total_pending = sum(pending_by_class.values())
    return {
        "schema_version": "lab_review_queue_report.v1",
        "generated_at": generated_at or utc_now(),
        "ok": not missing_outputs,
        "status": "attention" if total_pending else "ok",
        "blockers": [
            f"{len(missing_outputs)} pending review item(s) have missing output artifacts"
        ] if missing_outputs else [],
        "queue_dir": str(queue_dir),
        "task_records_file": str(records_file),
        "verdicts_file": str(verdicts_file),
        "summary": {
            "task_records": len(task_records),
            "verdicts": len(verdicts),
            "pending_reviews": total_pending,
            "pending_reviews_by_class": pending_by_class,
            "pending_review_candidates": pending_by_class.get("review_candidate", 0),
            "pending_active_safe": pending_by_class.get("active_safe_deterministic", 0),
            "items_returned": len(pending),
            "items_truncated": total_pending > len(pending),
        },
        "pending_items": pending,
        "review_batch_template": review_batch_template,
        "review_batch_template_jsonl": "\n".join(
            json.dumps(row, sort_keys=True) for row in review_batch_template
        ),
        "missing_outputs": missing_outputs,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Lab Review Queue Report",
        "",
        f"- generated_at: `{report.get('generated_at', '')}`",
        f"- status: `{report.get('status', '')}`",
        f"- pending_reviews: `{summary.get('pending_reviews', 0)}`",
        f"- pending_active_safe: `{summary.get('pending_active_safe', 0)}`",
        f"- pending_review_candidates: `{summary.get('pending_review_candidates', 0)}`",
        f"- queue_dir: `{report.get('queue_dir', '')}`",
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {b}" for b in blockers)
        lines.append("")

    pending = report.get("pending_items") or []
    if pending:
        lines.extend([
            "## Pending Items",
            "",
            "| job_id | run_id | class | stage | next_reviewer | output |",
            "|---|---|---|---|---|---|",
        ])
        for item in pending:
            lines.append(
                "| {job_id} | `{run_id}` | {record_class} | {stage} | {next_reviewer} | `{output_path}` |".format(
                    job_id=item.get("job_id", ""),
                    run_id=item.get("run_id", ""),
                    record_class=item.get("record_class", ""),
                    stage=item.get("stage", ""),
                    next_reviewer=item.get("next_reviewer", ""),
                    output_path=item.get("output_path", ""),
                )
            )
        lines.append("")
        lines.extend([
            "## Review Batch Template",
            "",
            "For active-safe deterministic rows, run `scripts/lab/auto_review_active_safe.py --apply`; for model-backed rows, edit `verdict`, `confidence`, and `notes`, then pass the JSONL to `scripts/lab/apply_review_batch.py`.",
            "",
            "```jsonl",
            str(report.get("review_batch_template_jsonl") or ""),
            "```",
            "",
        ])
    else:
        lines.extend(["No pending lab review items.", ""])

    missing_outputs = report.get("missing_outputs") or []
    if missing_outputs:
        lines.extend(["## Missing Outputs", ""])
        for item in missing_outputs:
            lines.append(
                f"- {item.get('job_id', '')} `{item.get('run_id', '')}` -> `{item.get('output_path', '')}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    jobs_file = _resolve_path(repo_root, args.jobs_file)
    queue_dir = _resolve_path(repo_root, args.queue_dir or DEFAULT_QUEUE)
    records_file = _resolve_path(repo_root, args.task_records_file or queue_dir / DEFAULT_RECORDS)
    verdicts_file = _resolve_path(repo_root, args.verdicts_file or queue_dir / DEFAULT_VERDICTS)
    return build_report(
        jobs_doc=_load_yaml(jobs_file),
        queue_dir=queue_dir,
        records_file=records_file,
        verdicts_file=verdicts_file,
        task_records=_load_jsonl(records_file),
        verdicts=_load_jsonl(verdicts_file),
        max_items=args.max_items,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--jobs-file", default=str(DEFAULT_JOBS_FILE))
    parser.add_argument("--queue-dir")
    parser.add_argument("--task-records-file")
    parser.add_argument("--verdicts-file")
    parser.add_argument("--max-items", type=int, default=25)
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout (default).")
    parser.add_argument("--markdown", action="store_true", help="Emit a markdown review packet to stdout.")
    parser.add_argument("--output-md", help="Also write a markdown review packet to this path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_from_args(args)
    except ReviewQueueReportError as exc:
        print(f"review_queue_report: {exc}", file=sys.stderr)
        return 2
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown(report), encoding="utf-8")
    if args.markdown:
        print(render_markdown(report), end="")
    else:
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
