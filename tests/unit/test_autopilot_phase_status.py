from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "autopilot"
sys.path.insert(0, str(_ROOT))

from phase_status import (  # noqa: E402
    AsyncTaskRunner,
    PhaseTracker,
    build_phase_health_report,
    format_phase_health_report,
)


def test_phase_tracker_writes_snapshot_and_jsonl(tmp_path):
    snapshot = tmp_path / "phase.json"
    events = tmp_path / "phase.jsonl"
    tracker = PhaseTracker(path=snapshot, events_path=events)

    tracker.set("planner_prompt_build", trial_id=7, idle_reason="building")
    payload = json.loads(snapshot.read_text())

    assert payload["phase"] == "planner_prompt_build"
    assert payload["trial_id"] == 7
    assert payload["idle_reason"] == "building"
    assert payload["pid"] > 0
    assert events.read_text().strip()


def test_async_task_runner_sync_fallback():
    runner = AsyncTaskRunner(enabled=False)

    result = runner.submit("add", lambda a, b: a + b, 2, 3)

    assert result == 5


def test_phase_health_report_accepts_fresh_alive_heartbeat(tmp_path, monkeypatch):
    snapshot = tmp_path / "phase.json"
    snapshot.write_text(
        json.dumps(
            {
                "phase": "dispatch_action",
                "pid": 123,
                "trial_id": 894,
                "action_type": "deep_eval",
                "updated_at": 100.0,
                "updated_at_iso": "2026-06-20T12:13:13+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("phase_status._process_exists", lambda pid: True)

    report = build_phase_health_report(path=snapshot, now=120.0, stale_after_s=60.0)

    assert report["ok"] is True
    assert report["status"] == "active"
    assert report["heartbeat_age_s"] == 20.0
    assert report["pid_alive"] is True
    assert report["trial_id"] == 894
    assert "Status: active" in "\n".join(format_phase_health_report(report))


def test_phase_health_report_blocks_stale_heartbeat(tmp_path, monkeypatch):
    snapshot = tmp_path / "phase.json"
    snapshot.write_text(
        json.dumps({"phase": "dispatch_action", "pid": 123, "updated_at": 100.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr("phase_status._process_exists", lambda pid: True)

    report = build_phase_health_report(path=snapshot, now=1001.0, stale_after_s=900.0)

    assert report["ok"] is False
    assert report["status"] == "stale"
    assert report["blockers"] == ["phase heartbeat is stale: 901.0s > 900.0s"]


def test_phase_health_report_blocks_dead_pid(tmp_path, monkeypatch):
    snapshot = tmp_path / "phase.json"
    snapshot.write_text(
        json.dumps({"phase": "dispatch_action", "pid": 123, "updated_at": 100.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr("phase_status._process_exists", lambda pid: False)

    report = build_phase_health_report(path=snapshot, now=120.0, stale_after_s=900.0)

    assert report["ok"] is False
    assert report["status"] == "pid_dead"
    assert report["blockers"] == ["phase heartbeat pid is not alive: 123"]


def test_phase_health_report_handles_missing_file(tmp_path):
    report = build_phase_health_report(
        path=tmp_path / "missing.json",
        now=120.0,
        stale_after_s=900.0,
    )

    assert report["ok"] is False
    assert report["status"] == "missing"
    assert "missing or unreadable" in report["blockers"][0]
