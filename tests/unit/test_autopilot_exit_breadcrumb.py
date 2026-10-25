from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import autopilot  # type: ignore[import-not-found]  # noqa: E402
from tests.unit.autopilot_loop_harness import (  # noqa: E402
    fresh_state,
    planner_decision,
    run_loop_inner,
)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exit_breadcrumb_persists_signal_and_one_terminal_record(tmp_path: Path) -> None:
    path = tmp_path / "autopilot_exit_breadcrumb.jsonl"
    breadcrumb = autopilot.ExitBreadcrumb(path)
    breadcrumb.set_context(trial_id=1302)

    assert breadcrumb.write("signal_received", signal_number=15, signal_name="SIGTERM")
    assert breadcrumb.mark_terminal("loop_exit", exit_trigger="signal")
    assert breadcrumb.mark_terminal("interpreter_exit")

    records = _records(path)
    assert [record["reason"] for record in records] == ["signal_received", "loop_exit"]
    assert records[0]["trial_id"] == 1302
    assert records[0]["signal_name"] == "SIGTERM"
    assert records[1]["terminal"] is True


def test_run_loop_records_unhandled_exception_before_reraising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "autopilot_exit_breadcrumb.jsonl"
    monkeypatch.setattr(autopilot, "EXIT_BREADCRUMB_PATH", path)

    def fail_inner(*_args: object) -> None:
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(autopilot, "_run_loop_inner", fail_inner)

    with pytest.raises(RuntimeError, match="deliberate test failure"):
        autopilot.run_loop(use_controller=False)

    records = _records(path)
    assert [record["reason"] for record in records] == [
        "run_loop_started",
        "unhandled_exception",
    ]
    assert records[-1]["terminal"] is True
    assert records[-1]["exception_type"] == "RuntimeError"


def test_loop_breadcrumb_tracks_current_trial_when_signal_arrives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "autopilot_exit_breadcrumb.jsonl"
    breadcrumb = autopilot.ExitBreadcrumb(path)
    turns = {"count": 0}

    def stop_after_two_trials(_decision: object, handlers: dict[int, object]) -> None:
        turns["count"] += 1
        if turns["count"] == 3:
            handlers[signal.SIGTERM](signal.SIGTERM, None)  # type: ignore[operator]

    run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=True,
        higher_tier_probe=False,
        planner_decisions=[planner_decision({"type": "seed_batch"})],
        planner_hook=stop_after_two_trials,
        breadcrumb=breadcrumb,
        max_trials=3,
    )

    records = _records(path)
    signal_record = next(record for record in records if record["reason"] == "signal_received")
    terminal = next(record for record in records if record.get("terminal") is True)
    assert signal_record["trial_id"] == 2
    assert terminal["exit_trigger"] == "signal"


def test_loop_breadcrumb_names_the_durable_halt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "autopilot_exit_breadcrumb.jsonl"
    breadcrumb = autopilot.ExitBreadcrumb(path)

    run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=True,
        higher_tier_probe=False,
        planner_decisions=[planner_decision({"type": "seed_batch"}, degraded=True, critique=None)],
        breadcrumb=breadcrumb,
    )

    terminal = next(record for record in _records(path) if record.get("terminal") is True)
    assert terminal["exit_trigger"] == "critic_unavailable"
