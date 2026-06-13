from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.lab import run_shadow_jobs


def _job(job_id: str, *, enabled: bool = True, gated: bool = False) -> dict:
    job = {
        "job_id": job_id,
        "stage": "shadow",
        "enabled": enabled,
        "schedule": {"type": "nightly", "cadence": "daily"},
    }
    if gated:
        job["gates"] = ["frontier-f5-intake-injection-hardening"]
    return job


def test_select_jobs_defaults_to_enabled_ungated_nightly() -> None:
    jobs_doc = {
        "jobs": [
            _job("enabled_1"),
            _job("disabled", enabled=False),
            _job("gated", gated=True),
            _job("enabled_2"),
        ]
    }

    selected = run_shadow_jobs.select_jobs(
        jobs_doc,
        schedule="nightly",
        job_ids=[],
        include_disabled=False,
        allow_gated=False,
        max_jobs=2,
    )

    assert [job["job_id"] for job in selected] == ["enabled_1", "enabled_2"]


def test_select_jobs_can_include_disabled_and_gated_when_explicit() -> None:
    jobs_doc = {"jobs": [_job("disabled", enabled=False), _job("gated", gated=True)]}

    selected = run_shadow_jobs.select_jobs(
        jobs_doc,
        schedule="nightly",
        job_ids=[],
        include_disabled=True,
        allow_gated=True,
        max_jobs=0,
    )

    assert [job["job_id"] for job in selected] == ["disabled", "gated"]


def test_requested_unrunnable_job_fails() -> None:
    jobs_doc = {"jobs": [_job("disabled", enabled=False)]}

    with pytest.raises(run_shadow_jobs.ShadowBatchError, match="not runnable"):
        run_shadow_jobs.select_jobs(
            jobs_doc,
            schedule="nightly",
            job_ids=["disabled"],
            include_disabled=False,
            allow_gated=False,
            max_jobs=2,
        )


def test_run_from_args_requires_execution_mode(tmp_path, monkeypatch) -> None:
    jobs_file = tmp_path / "lab_jobs.yaml"
    jobs_file.write_text("jobs: []\n")
    args = Namespace(
        repo_root=str(tmp_path),
        jobs_file=str(jobs_file),
        dry_run_stub=False,
        execute_chat=False,
    )

    with pytest.raises(run_shadow_jobs.ShadowBatchError, match="dry-run-stub"):
        run_shadow_jobs.run_from_args(args)
