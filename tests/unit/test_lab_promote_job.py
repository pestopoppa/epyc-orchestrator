from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from scripts.lab import promote_job


def _write_jobs_file(path: Path, *, risk: str = "read_only") -> None:
    doc = {
        "version": 1,
        "schema_version": "lab_jobs.v1",
        "policy": {
            "review_queue": "orchestration/lab_review_queue/",
            "direct_repo_writes_allowed": False,
            "default_stage": "shadow",
        },
        "jobs": [
            {
                "job_id": "sample_job",
                "title": "Sample job",
                "stage": "shadow",
                "enabled": False,
                "risk": risk,
                "model_role": "verifier",
                "input_spec": {"sources": []},
                "output_contract": {"format": "json", "json_schema": {"type": "object"}},
            }
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _records(stage: str, n: int) -> list[dict]:
    return [
        {
            "schema_version": "lab_task_record.v1",
            "record_type": "task_record",
            "job_id": "sample_job",
            "run_id": f"{stage}-{idx}",
            "stage": stage,
        }
        for idx in range(n)
    ]


def _verdicts(stage: str, verdicts: list[str], *, cloud: bool = True) -> list[dict]:
    rows = []
    for idx, verdict in enumerate(verdicts):
        row = {
            "schema_version": "lab_review_verdict.v1",
            "job_id": "sample_job",
            "run_id": f"{stage}-{idx}",
            "stage": stage,
            "verdict": verdict,
            "reviewer": "cloud_reference" if cloud else "operator",
            "tuple_path": f"gold/{stage}-{idx}.json",
        }
        if cloud:
            row["cloud_reference_run_id"] = f"cloud-{idx}"
        rows.append(row)
    return rows


def _args(tmp_path: Path, jobs_file: Path, **overrides) -> Namespace:
    values = {
        "job_id": "sample_job",
        "target_stage": "reviewed",
        "jobs_file": str(jobs_file),
        "repo_root": str(tmp_path),
        "queue_dir": str(tmp_path / "queue"),
        "task_records_file": str(tmp_path / "queue" / "task_records.jsonl"),
        "verdicts_file": str(tmp_path / "queue" / "review_verdicts.jsonl"),
        "min_shadow_runs": 10,
        "min_reviewed_runs": 20,
        "autonomous_accept_rate": 0.90,
        "apply": False,
        "confirm_job_id": None,
        "no_report": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_reviewed_requires_ten_cloud_reference_gold_tuples(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)
    _write_jsonl(tmp_path / "queue" / "task_records.jsonl", _records("shadow", 9))
    _write_jsonl(
        tmp_path / "queue" / "review_verdicts.jsonl",
        _verdicts("shadow", ["accept"] * 9),
    )

    decision = promote_job.run_from_args(_args(tmp_path, jobs_file))
    assert not decision.eligible
    assert "insufficient shadow verdicts" in decision.reason

    _write_jsonl(tmp_path / "queue" / "task_records.jsonl", _records("shadow", 10))
    _write_jsonl(
        tmp_path / "queue" / "review_verdicts.jsonl",
        _verdicts("shadow", ["accept"] * 10),
    )
    decision = promote_job.run_from_args(_args(tmp_path, jobs_file))
    assert decision.eligible
    assert decision.counts["shadow_cloud_scored"] == 10
    assert decision.report_path and decision.report_path.exists()


def test_reviewed_requires_saved_gold_tuple_paths(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)
    verdicts = _verdicts("shadow", ["accept"] * 10)
    for row in verdicts:
        row.pop("tuple_path")
    _write_jsonl(tmp_path / "queue" / "task_records.jsonl", _records("shadow", 10))
    _write_jsonl(tmp_path / "queue" / "review_verdicts.jsonl", verdicts)

    decision = promote_job.run_from_args(_args(tmp_path, jobs_file))
    assert not decision.eligible
    assert "gold tuple" in decision.reason


def test_autonomous_requires_read_only_and_accept_rate(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file, risk="read_only")
    _write_jsonl(tmp_path / "queue" / "task_records.jsonl", _records("reviewed", 20))
    _write_jsonl(
        tmp_path / "queue" / "review_verdicts.jsonl",
        _verdicts("reviewed", ["accept"] * 17 + ["reject"] * 3, cloud=False),
    )

    decision = promote_job.run_from_args(
        _args(tmp_path, jobs_file, target_stage="autonomous")
    )
    assert not decision.eligible
    assert "accept rate" in decision.reason

    _write_jsonl(
        tmp_path / "queue" / "review_verdicts.jsonl",
        _verdicts("reviewed", ["accept"] * 18 + ["reject"] * 2, cloud=False),
    )
    decision = promote_job.run_from_args(
        _args(tmp_path, jobs_file, target_stage="autonomous")
    )
    assert decision.eligible
    assert decision.counts["reviewed_accept_rate"] == 0.9


def test_autonomous_rejects_write_reviewed_jobs(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file, risk="write_reviewed")
    _write_jsonl(tmp_path / "queue" / "task_records.jsonl", _records("reviewed", 20))
    _write_jsonl(
        tmp_path / "queue" / "review_verdicts.jsonl",
        _verdicts("reviewed", ["accept"] * 20, cloud=False),
    )

    decision = promote_job.run_from_args(
        _args(tmp_path, jobs_file, target_stage="autonomous")
    )
    assert not decision.eligible
    assert "read_only" in decision.reason


def test_apply_requires_confirmation_and_updates_job_stage(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)
    _write_jsonl(tmp_path / "queue" / "task_records.jsonl", _records("shadow", 10))
    _write_jsonl(
        tmp_path / "queue" / "review_verdicts.jsonl",
        _verdicts("shadow", ["accept"] * 10),
    )

    with pytest.raises(promote_job.PromotionError, match="confirm-job-id"):
        promote_job.run_from_args(
            _args(tmp_path, jobs_file, apply=True, confirm_job_id=None)
        )

    promote_job.run_from_args(
        _args(tmp_path, jobs_file, apply=True, confirm_job_id="sample_job")
    )
    updated = yaml.safe_load(jobs_file.read_text())
    job = updated["jobs"][0]
    assert job["stage"] == "reviewed"
    assert job["enabled"] is True
