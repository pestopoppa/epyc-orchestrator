#!/usr/bin/env python3
"""Evaluate or apply self-running-lab job promotions."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_QUEUE = Path("orchestration/lab_review_queue")
DEFAULT_RECORDS = "task_records.jsonl"
DEFAULT_VERDICTS = "review_verdicts.jsonl"
TARGET_STAGES = ("reviewed", "autonomous")


class PromotionError(RuntimeError):
    """Raised for operator-facing promotion failures."""


@dataclass(frozen=True)
class PromotionDecision:
    job_id: str
    target_stage: str
    eligible: bool
    reason: str
    counts: dict[str, Any]
    report_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "target_stage": self.target_stage,
            "eligible": self.eligible,
            "reason": self.reason,
            "counts": self.counts,
            "report_path": str(self.report_path) if self.report_path else None,
        }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise PromotionError(f"jobs file not found: {path}") from exc
    if not isinstance(data, dict):
        raise PromotionError(f"jobs file must contain a mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    text = yaml.safe_dump(data, sort_keys=False)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _job_by_id(jobs_doc: dict[str, Any], job_id: str) -> dict[str, Any]:
    for job in jobs_doc.get("jobs", []) or []:
        if isinstance(job, dict) and job.get("job_id") == job_id:
            return job
    raise PromotionError(f"job_id not found in jobs file: {job_id}")


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
            raise PromotionError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise PromotionError(f"{path}:{lineno}: JSONL row must be an object")
        rows.append(row)
    return rows


def _is_accepted(row: dict[str, Any]) -> bool:
    return row.get("verdict") in {"accept", "accepted", "pass", "passed"}


def _is_rejected(row: dict[str, Any]) -> bool:
    return row.get("verdict") in {"reject", "rejected", "fail", "failed"}


def _is_scored(row: dict[str, Any]) -> bool:
    return _is_accepted(row) or _is_rejected(row)


def _is_cloud_referenced(row: dict[str, Any]) -> bool:
    return (
        row.get("reference_type") == "cloud_reference"
        or row.get("reviewer") == "cloud_reference"
        or bool(row.get("cloud_reference_run_id"))
    )


def _has_gold_tuple(row: dict[str, Any]) -> bool:
    return bool(row.get("tuple_path") or row.get("gold_tuple_path"))


def _stage_for(row: dict[str, Any], records_by_run: dict[str, dict[str, Any]]) -> str:
    stage = row.get("stage")
    if stage:
        return str(stage)
    record = records_by_run.get(str(row.get("run_id", "")), {})
    return str(record.get("stage", ""))


def _filter_verdicts(
    *,
    job_id: str,
    stage: str,
    verdicts: list[dict[str, Any]],
    records_by_run: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in verdicts
        if row.get("job_id") == job_id
        and _stage_for(row, records_by_run) == stage
        and _is_scored(row)
    ]


def _accept_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _is_accepted(row)) / len(rows)


def evaluate_promotion(
    *,
    jobs_doc: dict[str, Any],
    job_id: str,
    target_stage: str,
    task_records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    min_shadow_runs: int = 10,
    min_reviewed_runs: int = 20,
    autonomous_accept_rate: float = 0.90,
) -> PromotionDecision:
    if target_stage not in TARGET_STAGES:
        raise PromotionError(f"unsupported target stage: {target_stage}")
    job = _job_by_id(jobs_doc, job_id)
    records_by_run = {
        str(row.get("run_id")): row
        for row in task_records
        if row.get("job_id") == job_id and row.get("run_id")
    }
    if target_stage == "reviewed":
        shadow = _filter_verdicts(
            job_id=job_id,
            stage="shadow",
            verdicts=verdicts,
            records_by_run=records_by_run,
        )
        cloud_scored = [row for row in shadow if _is_cloud_referenced(row)]
        gold_rows = [row for row in cloud_scored if _has_gold_tuple(row)]
        counts = {
            "shadow_scored": len(shadow),
            "shadow_cloud_scored": len(cloud_scored),
            "shadow_gold_tuples": len(gold_rows),
            "shadow_accept_rate": round(_accept_rate(cloud_scored), 4),
            "min_shadow_runs": min_shadow_runs,
        }
        if len(cloud_scored) < min_shadow_runs:
            return PromotionDecision(
                job_id,
                target_stage,
                False,
                "insufficient shadow verdicts scored against cloud-reference runs",
                counts,
            )
        if len(gold_rows) < min_shadow_runs:
            return PromotionDecision(
                job_id,
                target_stage,
                False,
                "cloud-reference verdicts must save F3 gold tuple paths",
                counts,
            )
        return PromotionDecision(
            job_id,
            target_stage,
            True,
            "shadow evidence satisfies reviewed-stage promotion gate",
            counts,
        )

    reviewed = _filter_verdicts(
        job_id=job_id,
        stage="reviewed",
        verdicts=verdicts,
        records_by_run=records_by_run,
    )
    gold_rows = [row for row in reviewed if _has_gold_tuple(row)]
    accept_rate = _accept_rate(gold_rows)
    counts = {
        "reviewed_scored": len(reviewed),
        "reviewed_gold_tuples": len(gold_rows),
        "reviewed_accept_rate": round(accept_rate, 4),
        "min_reviewed_runs": min_reviewed_runs,
        "required_accept_rate": autonomous_accept_rate,
    }
    if job.get("risk") != "read_only":
        return PromotionDecision(
            job_id,
            target_stage,
            False,
            "autonomous promotion is restricted to read_only jobs",
            counts,
        )
    if len(gold_rows) < min_reviewed_runs:
        return PromotionDecision(
            job_id,
            target_stage,
            False,
            "insufficient reviewed verdicts with F3 gold tuple paths",
            counts,
        )
    if accept_rate < autonomous_accept_rate:
        return PromotionDecision(
            job_id,
            target_stage,
            False,
            "reviewed accept rate is below autonomous threshold",
            counts,
        )
    return PromotionDecision(
        job_id,
        target_stage,
        True,
        "reviewed evidence satisfies autonomous promotion gate",
        counts,
    )


def write_report(
    *,
    queue_dir: Path,
    decision: PromotionDecision,
    generated_at: str,
    applied: bool,
) -> PromotionDecision:
    report = {
        "schema_version": "lab_promotion_report.v1",
        "generated_at": generated_at,
        "job_id": decision.job_id,
        "target_stage": decision.target_stage,
        "eligible": decision.eligible,
        "reason": decision.reason,
        "counts": decision.counts,
        "applied": applied,
        "next_action": (
            "review applied jobs-file diff"
            if applied
            else "apply with --apply --confirm-job-id after operator review"
            if decision.eligible
            else "collect more reviewed evidence"
        ),
    }
    safe_job = decision.job_id.replace("/", "_")
    stamp = generated_at.replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    path = queue_dir / "promotion_reports" / f"{safe_job}-{decision.target_stage}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return PromotionDecision(
        decision.job_id,
        decision.target_stage,
        decision.eligible,
        decision.reason,
        decision.counts,
        path,
    )


def apply_promotion(
    *,
    jobs_file: Path,
    jobs_doc: dict[str, Any],
    decision: PromotionDecision,
    confirm_job_id: str | None,
) -> None:
    if not decision.eligible:
        raise PromotionError("refusing to apply an ineligible promotion")
    if confirm_job_id != decision.job_id:
        raise PromotionError("--apply requires --confirm-job-id matching the promoted job")
    job = _job_by_id(jobs_doc, decision.job_id)
    job["stage"] = decision.target_stage
    job["enabled"] = True
    _write_yaml(jobs_file, jobs_doc)


def run_from_args(args: argparse.Namespace) -> PromotionDecision:
    repo_root = Path(args.repo_root).expanduser().resolve()
    jobs_file = Path(args.jobs_file).expanduser()
    if not jobs_file.is_absolute():
        jobs_file = repo_root / jobs_file
    jobs_file = jobs_file.resolve()
    queue_dir = Path(args.queue_dir).expanduser() if args.queue_dir else DEFAULT_QUEUE
    if not queue_dir.is_absolute():
        queue_dir = repo_root / queue_dir
    queue_dir = queue_dir.resolve()
    records_file = Path(args.task_records_file or queue_dir / DEFAULT_RECORDS)
    verdicts_file = Path(args.verdicts_file or queue_dir / DEFAULT_VERDICTS)
    jobs_doc = _load_yaml(jobs_file)
    decision = evaluate_promotion(
        jobs_doc=jobs_doc,
        job_id=args.job_id,
        target_stage=args.target_stage,
        task_records=_load_jsonl(records_file),
        verdicts=_load_jsonl(verdicts_file),
        min_shadow_runs=args.min_shadow_runs,
        min_reviewed_runs=args.min_reviewed_runs,
        autonomous_accept_rate=args.autonomous_accept_rate,
    )
    applied = False
    if args.apply:
        apply_promotion(
            jobs_file=jobs_file,
            jobs_doc=jobs_doc,
            decision=decision,
            confirm_job_id=args.confirm_job_id,
        )
        applied = True
    if not args.no_report:
        decision = write_report(
            queue_dir=queue_dir,
            decision=decision,
            generated_at=utc_now(),
            applied=applied,
        )
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--target-stage", required=True, choices=TARGET_STAGES)
    parser.add_argument("--jobs-file", default="orchestration/lab_jobs.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--queue-dir")
    parser.add_argument("--task-records-file")
    parser.add_argument("--verdicts-file")
    parser.add_argument("--min-shadow-runs", type=int, default=10)
    parser.add_argument("--min-reviewed-runs", type=int, default=20)
    parser.add_argument("--autonomous-accept-rate", type=float, default=0.90)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-job-id")
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        decision = run_from_args(args)
    except PromotionError as exc:
        print(f"promote_job: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(decision.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
