"""flush_cache_with_pause + the autopilot loop pause-bug fix.

2026-05-24: pre-fix, `autopilot.py pause` was a no-op on running autopilots
because `state.get("paused")` read a cached in-memory dict and `save_state()`
after each trial clobbered any externally-set True. The fix reloads state at
the top of every iteration. This test mocks out the subprocess+sudo paths and
verifies the wrapper does the right state-file dance.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))


host_health = importlib.import_module("host_health")
experiment_journal = importlib.import_module("experiment_journal")


def _write_state(path: Path, paused: bool, trial_counter: int = 1) -> None:
    path.write_text(json.dumps({"paused": paused, "trial_counter": trial_counter}, indent=2))


def test_exogenous_cache_flush_category_exists() -> None:
    """DeficiencyCategory.EXOGENOUS_CACHE_FLUSH must be defined for journal tagging."""
    assert hasattr(experiment_journal.DeficiencyCategory, "EXOGENOUS_CACHE_FLUSH")
    assert experiment_journal.DeficiencyCategory.EXOGENOUS_CACHE_FLUSH.value == "exogenous_cache_flush"


def test_flush_sets_paused_true_then_restores(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path, paused=False)

    captured_states: list[bool] = []

    def _fake_remediate() -> bool:
        # While flush runs, capture what the state file says — should be True.
        with open(state_path) as f:
            captured_states.append(json.load(f).get("paused", False))
        return True

    with mock.patch.object(host_health, "remediate", side_effect=_fake_remediate), \
         mock.patch.object(host_health, "_numa_interleave_rewarm", return_value={}), \
         mock.patch("time.sleep"):
        result = host_health.flush_cache_with_pause(state_path=state_path, rewarm=False)

    assert result["flush_ok"] is True
    assert result["paused_pre"] is False
    # During the flush, state was paused
    assert captured_states == [True]
    # After the flush, state was restored to False
    with open(state_path) as f:
        assert json.load(f)["paused"] is False


def test_flush_preserves_user_set_pause(tmp_path: Path) -> None:
    """If autopilot was already paused before the flush, leave it paused after."""
    state_path = tmp_path / "state.json"
    _write_state(state_path, paused=True)

    with mock.patch.object(host_health, "remediate", return_value=True), \
         mock.patch.object(host_health, "_numa_interleave_rewarm", return_value={}), \
         mock.patch("time.sleep"):
        result = host_health.flush_cache_with_pause(state_path=state_path, rewarm=False)

    assert result["paused_pre"] is True
    with open(state_path) as f:
        # Stay paused — the user/operator wants it paused
        assert json.load(f)["paused"] is True


def test_flush_runs_rewarm_when_enabled(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path, paused=False)

    fake_rewarm_results = {"/tmp/fake1.gguf": True, "/tmp/fake2.gguf": True}
    with mock.patch.object(host_health, "remediate", return_value=True), \
         mock.patch.object(host_health, "_numa_interleave_rewarm",
                           return_value=fake_rewarm_results) as mock_warm, \
         mock.patch("time.sleep"):
        result = host_health.flush_cache_with_pause(state_path=state_path, rewarm=True)

    assert mock_warm.called
    assert result["rewarm"] == fake_rewarm_results


def test_flush_skips_rewarm_when_disabled(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path, paused=False)

    with mock.patch.object(host_health, "remediate", return_value=True), \
         mock.patch.object(host_health, "_numa_interleave_rewarm",
                           return_value={"x": True}) as mock_warm, \
         mock.patch("time.sleep"):
        result = host_health.flush_cache_with_pause(state_path=state_path, rewarm=False)

    assert not mock_warm.called
    assert result["rewarm"] == {}


def test_flush_skips_rewarm_on_flush_failure(tmp_path: Path) -> None:
    """If flush itself failed (no sudo, no helper), skip rewarm — pages are still cached."""
    state_path = tmp_path / "state.json"
    _write_state(state_path, paused=False)

    with mock.patch.object(host_health, "remediate", return_value=False), \
         mock.patch.object(host_health, "_numa_interleave_rewarm",
                           return_value={"x": True}) as mock_warm, \
         mock.patch("time.sleep"):
        result = host_health.flush_cache_with_pause(state_path=state_path, rewarm=True)

    assert result["flush_ok"] is False
    assert not mock_warm.called


def test_flush_handles_missing_state_file(tmp_path: Path) -> None:
    """If state.json doesn't exist, log a warning but still proceed with flush."""
    state_path = tmp_path / "nonexistent.json"
    with mock.patch.object(host_health, "remediate", return_value=True), \
         mock.patch.object(host_health, "_numa_interleave_rewarm", return_value={}), \
         mock.patch("time.sleep"):
        result = host_health.flush_cache_with_pause(state_path=state_path, rewarm=False)
    assert result["flush_ok"] is True
    assert result["paused_pre"] is None
