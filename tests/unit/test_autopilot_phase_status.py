from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "autopilot"
sys.path.insert(0, str(_ROOT))

from phase_status import AsyncTaskRunner, PhaseTracker  # noqa: E402


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
