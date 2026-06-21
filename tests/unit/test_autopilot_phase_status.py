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
                "eval_label": "T2",
                "eval_completed_questions": 200,
                "eval_total_questions": 500,
                "eval_correct_questions": 144,
                "eval_correct_pct": 72.0,
                "eval_concurrency": 1,
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
    assert report["eval_completed_questions"] == 200
    assert report["eval_total_questions"] == 500
    formatted = "\n".join(format_phase_health_report(report))
    assert "Status: active" in formatted
    assert "Eval progress: 200/500 (72% correct)" in formatted


def test_phase_health_report_tails_eval_progress_when_heartbeat_lacks_counters(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "phase.json"
    snapshot.write_text(
        json.dumps(
            {
                "phase": "dispatch_action",
                "pid": 123,
                "trial_id": 902,
                "action_type": "deep_eval",
                "updated_at": 100.0,
                "updated_at_iso": "2026-06-20T12:13:13+00:00",
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "autopilot.log"
    log_path.write_text(
        "\n".join(
            [
                '2026-06-20 19:14:09 [autopilot] INFO: Trial 902: {"type": "deep_eval"}',
                "2026-06-20 19:29:11 [autopilot.eval] INFO: T2 progress: 50/500 (78% correct)",
                "2026-06-20 21:43:41 [autopilot.eval] INFO: T2 progress: 400/500 (70% correct)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("phase_status._process_exists", lambda pid: True)

    report = build_phase_health_report(
        path=snapshot,
        log_path=log_path,
        now=120.0,
        stale_after_s=60.0,
    )

    assert report["ok"] is True
    assert report["eval_label"] == "T2"
    assert report["eval_completed_questions"] == 400
    assert report["eval_total_questions"] == 500
    assert report["eval_correct_pct"] == 70.0
    assert report["eval_progress_source"] == "log_tail"
    formatted = "\n".join(format_phase_health_report(report))
    assert "Eval progress: 400/500 (70% correct)" in formatted


def test_phase_health_report_tails_numeric_trial_progress_from_recent_tmp_log(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "autopilot_phase.json"
    snapshot.write_text(
        json.dumps(
            {
                "phase": "dispatch_action",
                "pid": 123,
                "trial_id": 916,
                "action_type": "numeric_trial",
                "updated_at": 100.0,
                "updated_at_iso": "2026-06-21T03:33:26+00:00",
            }
        ),
        encoding="utf-8",
    )
    default_log = tmp_path / "logs" / "autopilot.log"
    default_log.parent.mkdir()
    default_log.write_text(
        "\n".join(
            [
                '2026-06-21 03:22:19 [autopilot] INFO: Trial 915: {"type": "seed_batch"}',
                "2026-06-21 03:22:18 [autopilot.eval] INFO: T1 progress: 60/60 (67% correct)",
            ]
        ),
        encoding="utf-8",
    )
    tmp_log_dir = tmp_path / "tmp"
    tmp_log_dir.mkdir()
    redirected_log = tmp_log_dir / "autopilot_w4w6_codex_pair.log"
    redirected_log.write_text(
        "\n".join(
            [
                '2026-06-21 03:22:56 [autopilot] INFO: Trial 916: {"type": "numeric_trial"}',
                "2026-06-21 03:27:14 [autopilot.eval] INFO: T1 progress: 10/60 (100% correct)",
                "2026-06-21 03:33:26 [autopilot.eval] INFO: T1 progress: 40/60 (70% correct)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("phase_status._process_exists", lambda pid: True)
    monkeypatch.setattr("phase_status.PHASE_PATH", snapshot)
    monkeypatch.setattr("phase_status.DEFAULT_AUTOPILOT_LOG_PATH", default_log)
    monkeypatch.setattr("phase_status.DEFAULT_TMP_AUTOPILOT_LOG_DIR", tmp_log_dir)

    report = build_phase_health_report(path=snapshot, now=120.0, stale_after_s=60.0)

    assert report["ok"] is True
    assert report["action_type"] == "numeric_trial"
    assert report["eval_label"] == "T1"
    assert report["eval_completed_questions"] == 40
    assert report["eval_total_questions"] == 60
    assert report["eval_correct_pct"] == 70.0
    assert report["eval_progress_source"] == "log_tail"
    assert report["eval_progress_log_path"] == str(redirected_log)


def test_phase_health_report_does_not_tail_other_trial_progress(tmp_path, monkeypatch):
    snapshot = tmp_path / "phase.json"
    snapshot.write_text(
        json.dumps(
            {
                "phase": "dispatch_action",
                "pid": 123,
                "trial_id": 903,
                "action_type": "deep_eval",
                "updated_at": 100.0,
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "autopilot.log"
    log_path.write_text(
        "\n".join(
            [
                '2026-06-20 19:14:09 [autopilot] INFO: Trial 902: {"type": "deep_eval"}',
                "2026-06-20 21:43:41 [autopilot.eval] INFO: T2 progress: 400/500 (70% correct)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("phase_status._process_exists", lambda pid: True)

    report = build_phase_health_report(
        path=snapshot,
        log_path=log_path,
        now=120.0,
        stale_after_s=60.0,
    )

    assert report["ok"] is True
    assert report["eval_completed_questions"] is None
    assert report.get("eval_progress_source") is None


def test_phase_health_report_keeps_heartbeat_eval_progress_over_log_tail(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "phase.json"
    snapshot.write_text(
        json.dumps(
            {
                "phase": "dispatch_action",
                "pid": 123,
                "trial_id": 902,
                "action_type": "deep_eval",
                "eval_label": "T1",
                "eval_completed_questions": 10,
                "eval_total_questions": 60,
                "eval_correct_pct": 100.0,
                "updated_at": 100.0,
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "autopilot.log"
    log_path.write_text(
        "\n".join(
            [
                '2026-06-20 19:14:09 [autopilot] INFO: Trial 902: {"type": "deep_eval"}',
                "2026-06-20 21:43:41 [autopilot.eval] INFO: T2 progress: 400/500 (70% correct)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("phase_status._process_exists", lambda pid: True)

    report = build_phase_health_report(
        path=snapshot,
        log_path=log_path,
        now=120.0,
        stale_after_s=60.0,
    )

    assert report["eval_label"] == "T1"
    assert report["eval_completed_questions"] == 10
    assert report["eval_total_questions"] == 60
    assert report["eval_correct_pct"] == 100.0
    assert report.get("eval_progress_source") is None


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
