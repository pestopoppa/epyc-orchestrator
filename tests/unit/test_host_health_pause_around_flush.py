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


def test_llama_server_memory_reader_aggregates_rollup(tmp_path: Path) -> None:
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "cmdline").write_bytes(b"/mnt/raid0/llm/llama.cpp/build/bin/llama-server\x00--port\x008070")
    (proc / "smaps_rollup").write_text(
        "Pss:                2048 kB\n"
        "Private_Dirty:      1024 kB\n"
        "Locked:              512 kB\n"
    )
    other = tmp_path / "456"
    other.mkdir()
    (other / "cmdline").write_bytes(b"python\x00worker.py")
    (other / "smaps_rollup").write_text(
        "Pss:                9999 kB\n"
        "Private_Dirty:      9999 kB\n"
        "Locked:             9999 kB\n"
    )

    assert host_health._read_llama_server_memory_mb(tmp_path) == (1, 2.0, 1.0, 0.5)


def test_memory_residency_warnings_are_advisory() -> None:
    state = host_health.HostHealthState(
        loadavg_1min=1.0,
        n_cores_online=64,
        mean_cur_mhz=2000.0,
        base_mhz=2000.0,
        page_cache_mb=500_000.0,
        mem_available_mb=500_000.0,
        unevictable_mb=160_000.0,
        mlocked_mb=160_000.0,
        llama_process_count=28,
        llama_pss_mb=650_000.0,
        llama_private_dirty_mb=400_000.0,
        llama_locked_mb=150_000.0,
        timestamp=0.0,
    )

    warnings = state.memory_residency_warnings()
    assert warnings
    assert "llama_private_dirty" in warnings[0]
    assert state.is_throttled() == (False, [])


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
