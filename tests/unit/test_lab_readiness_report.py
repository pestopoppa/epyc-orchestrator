from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import yaml

from scripts.lab import readiness_report


def _write_jobs_file(path: Path) -> None:
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
                "job_id": "shadow_ready",
                "title": "Shadow-ready job",
                "stage": "shadow",
                "enabled": True,
                "risk": "read_only",
                "schedule": {"type": "nightly", "cadence": "daily"},
                "input_spec": {"sources": []},
                "output_contract": {"format": "json", "json_schema": {"type": "object"}},
            },
            {
                "job_id": "manual_disabled",
                "title": "Manual disabled job",
                "stage": "shadow",
                "enabled": False,
                "risk": "write_reviewed",
                "schedule": {"type": "manual"},
                "input_spec": {"sources": []},
                "output_contract": {"format": "json", "json_schema": {"type": "object"}},
            },
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _records(job_id: str, stage: str, n: int) -> list[dict]:
    return [
        {
            "schema_version": "lab_task_record.v1",
            "record_type": "task_record",
            "job_id": job_id,
            "run_id": f"{stage}-{idx}",
            "stage": stage,
        }
        for idx in range(n)
    ]


def _verdicts(job_id: str, stage: str, verdicts: list[str]) -> list[dict]:
    return [
        {
            "schema_version": "lab_review_verdict.v1",
            "job_id": job_id,
            "run_id": f"{stage}-{idx}",
            "stage": stage,
            "verdict": verdict,
            "reviewer": "cloud_reference",
            "cloud_reference_run_id": f"cloud-{idx}",
            "tuple_path": f"gold/{stage}-{idx}.json",
        }
        for idx, verdict in enumerate(verdicts)
    ]


def _args(tmp_path: Path, jobs_file: Path, **overrides) -> Namespace:
    queue = tmp_path / "queue"
    values = {
        "jobs_file": str(jobs_file),
        "repo_root": str(tmp_path),
        "queue_dir": str(queue),
        "task_records_file": str(queue / "task_records.jsonl"),
        "verdicts_file": str(queue / "review_verdicts.jsonl"),
        "min_shadow_runs": 10,
        "min_reviewed_runs": 20,
        "autonomous_accept_rate": 0.90,
    }
    values.update(overrides)
    return Namespace(**values)


def test_report_summarizes_schedulable_jobs_and_promotion_readiness(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)
    queue = tmp_path / "queue"
    _write_jsonl(queue / "task_records.jsonl", _records("shadow_ready", "shadow", 10))
    _write_jsonl(
        queue / "review_verdicts.jsonl",
        _verdicts("shadow_ready", "shadow", ["accept"] * 10),
    )

    report = readiness_report.run_from_args(_args(tmp_path, jobs_file))

    assert report["schema_version"] == "lab_readiness_report.v1"
    assert report["summary"]["jobs_total"] == 2
    assert report["summary"]["enabled_jobs"] == 1
    assert report["summary"]["nightly_runnable"] == 1
    assert "nightly_ready_now" in report["summary"]
    assert report["summary"]["manual_runnable"] == 0
    assert report["summary"]["promotion_ready_job_ids"] == ["shadow_ready"]
    ready = report["jobs"][0]
    assert ready["job_id"] == "shadow_ready"
    assert ready["scheduled_nightly"] is True
    assert ready["task_records"]["by_stage"] == {"shadow": 10}
    assert ready["verdicts"]["gold_tuples"] == 10
    assert ready["promotion"]["reviewed"]["eligible"] is True


def test_report_treats_missing_logs_as_zero_evidence(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)

    report = readiness_report.run_from_args(_args(tmp_path, jobs_file))

    assert report["summary"]["task_records"] == 0
    assert report["summary"]["verdicts"] == 0
    assert report["summary"]["promotion_ready"] == 0
    assert report["jobs"][0]["promotion"]["reviewed"]["eligible"] is False


def test_report_surfaces_pending_review_records(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)
    queue = tmp_path / "queue"
    _write_jsonl(queue / "task_records.jsonl", _records("shadow_ready", "shadow", 3))
    _write_jsonl(
        queue / "review_verdicts.jsonl",
        _verdicts("shadow_ready", "shadow", ["accept"]),
    )

    report = readiness_report.run_from_args(_args(tmp_path, jobs_file))

    assert report["summary"]["pending_reviews"] == 2
    assert report["summary"]["pending_review_job_ids"] == ["shadow_ready"]
    ready = report["jobs"][0]
    assert ready["review"] == {
        "pending": 2,
        "pending_run_ids": ["shadow-1", "shadow-2"],
        "pending_run_ids_truncated": False,
    }


def test_report_separates_schedulable_from_ready_now(tmp_path: Path, monkeypatch) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)

    def fake_active_processes(marker: str) -> list[dict]:
        if marker == readiness_report.AUTOPILOT_CMD_MARKER:
            return [{"pid": 123, "cmd": "uv run python scripts/autopilot/autopilot.py start"}]
        return []

    monkeypatch.setattr(readiness_report, "_active_processes", fake_active_processes)

    report = readiness_report.run_from_args(_args(tmp_path, jobs_file))

    assert report["summary"]["nightly_runnable"] == 1
    assert report["summary"]["nightly_ready_now"] == 0
    assert report["quiet_window"]["ready"] is False
    assert report["quiet_window"]["blockers"] == ["active AutoPilot process count: 1"]
    assert report["quiet_window"]["active_autopilot_processes"][0]["pid"] == 123


def test_report_skip_process_check_leaves_ready_now_unknown(tmp_path: Path) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    _write_jobs_file(jobs_file)

    report = readiness_report.run_from_args(
        _args(tmp_path, jobs_file, skip_process_check=True)
    )

    assert report["summary"]["nightly_runnable"] == 1
    assert report["summary"]["nightly_ready_now"] == 0
    assert report["quiet_window"]["ready"] is None


def test_active_processes_only_counts_llama_server_executables(monkeypatch) -> None:
    class FakeProc:
        returncode = 0
        stdout = "\n".join(
            [
                "101 /usr/local/bin/earlyoom --ignore ^(llama-server|sd-server)$",
                "102 /mnt/raid0/llm/llama.cpp/build/bin/llama-server --port 8080",
                "103 llama-server --port 8081",
            ]
        )

    monkeypatch.setattr(
        readiness_report.subprocess,
        "run",
        lambda *args, **kwargs: FakeProc(),
    )
    monkeypatch.setattr(readiness_report.os, "getpid", lambda: 999)

    rows = readiness_report._active_processes(readiness_report.LLAMA_CMD_MARKER)

    assert [row["pid"] for row in rows] == [102, 103]
