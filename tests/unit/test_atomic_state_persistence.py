"""Unit tests for atomic persistence + corrupt-state startup refusal.

Per handoffs/active/autopilot-exogenous-restart-resilience.md Phase 6a.

Covers:
  - save_state uses atomic temp + os.replace (no half-files visible
    to concurrent readers)
  - load_state on a corrupt JSON file exits 70 with the exact stderr
    format, leaves the file untouched, does NOT write a fresh state
  - pareto_archive.save round-trip preserves data
  - Crash-during-write semantics (monkeypatch os.replace to raise)
    leaves the original file intact + leaves a forensic .tmp.<pid>
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import state_store  # noqa: E402
from pareto_archive import ParetoArchive, ParetoEntry  # noqa: E402


# ───────── save_state atomic semantics ──────────


def test_save_state_writes_payload(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    state_store.save_state(p, {"trial_counter": 42, "session_id": "abc"})
    loaded = json.loads(p.read_text())
    assert loaded == {"trial_counter": 42, "session_id": "abc"}


def test_save_state_uses_atomic_replace_no_tmp_left_behind(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    state_store.save_state(p, {"a": 1})
    # No .tmp.<pid> file should linger after a successful save
    leftover = [x for x in tmp_path.iterdir() if ".tmp." in x.name]
    assert leftover == [], f"unexpected tmp files: {leftover}"


def test_save_state_overwrites_previous(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    state_store.save_state(p, {"version": 1})
    state_store.save_state(p, {"version": 2, "extra": "x"})
    assert json.loads(p.read_text()) == {"version": 2, "extra": "x"}


def test_save_state_crash_during_replace_preserves_original(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "state.json"
    # Write a known-good first version
    state_store.save_state(p, {"trial": 1})
    original = p.read_text()
    # Now make os.replace raise during the next save
    def boom(*a, **kw):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(state_store.os, "replace", boom)
    with pytest.raises(OSError):
        state_store.save_state(p, {"trial": 2})
    # Original file is untouched (atomicity contract)
    assert p.read_text() == original
    # A forensic .tmp.<pid> file should exist
    leftover = [x for x in tmp_path.iterdir() if ".tmp." in x.name]
    assert leftover, "expected a leftover tmp file for post-mortem"


# ───────── corrupt-state startup refusal ──────────


def test_load_state_corrupt_file_exits_70(tmp_path: Path, capsys) -> None:
    p = tmp_path / "state.json"
    # Truncated JSON
    p.write_text('{"trial_counter": 5, "incomp')
    with pytest.raises(SystemExit) as excinfo:
        state_store.load_state(p, lambda: {})
    assert excinfo.value.code == state_store.EXIT_CORRUPT_STATE == 70
    captured = capsys.readouterr()
    assert "FATAL: orchestration/autopilot_state.json is corrupt" in captured.err
    assert "JSONDecodeError" in captured.err
    assert str(p) in captured.err
    assert "Recovery options:" in captured.err
    assert "baseline-*.json" in captured.err
    assert "autopilot_checkpoints" in captured.err


def test_load_state_corrupt_file_does_not_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    corrupt = '{"trial_counter": 5, "broken'
    p.write_text(corrupt)
    try:
        state_store.load_state(p, lambda: {"FRESH": True})
    except SystemExit:
        pass
    # The corrupt file MUST still be on disk untouched. If load_state
    # silently reset, we'd see {"FRESH": True} here.
    assert p.read_text() == corrupt


def test_load_state_missing_file_uses_default_factory(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    assert not p.exists()
    out = state_store.load_state(p, lambda: {"hello": "world"})
    assert out == {"hello": "world"}
    # Default-factory invocation should NOT create the file
    assert not p.exists()


def test_load_state_valid_file_returns_parsed(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"trial_counter": 10}))
    assert state_store.load_state(p, lambda: {}) == {"trial_counter": 10}


# ───────── pareto archive atomic semantics ──────────


def test_pareto_save_atomic_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    archive = ParetoArchive(state_path=state_path)
    archive.update(ParetoEntry(trial_id=1, objectives=(1.0, 50.0, -0.5, 1.0)))
    archive.update(ParetoEntry(trial_id=2, objectives=(0.5, 80.0, -0.3, 0.9)))
    archive.save({"trial_counter": 2})
    # Re-load from the same path
    a2 = ParetoArchive(state_path=state_path)
    assert a2.frontier_size() == 2
    # And no .tmp file left behind
    leftover = [x for x in tmp_path.iterdir() if ".tmp." in x.name]
    assert leftover == []


def test_pareto_save_updates_caller_state_for_followup_save(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    archive = ParetoArchive(state_path=state_path)
    state = {
        "trial_counter": 1,
        "pareto_archive": {
            "frontier": [],
            "all_entries": [],
            "hypervolume_history": [],
        },
    }
    archive.update(ParetoEntry(trial_id=1, objectives=(1.0, 50.0, -0.5, 1.0)))

    archive.save(state)
    state_store.save_state(state_path, state)

    loaded = json.loads(state_path.read_text())
    assert loaded["pareto_archive"]["all_entries"][0]["trial_id"] == 1
    assert ParetoArchive(state_path=state_path).frontier_size() == 1


def test_pareto_save_crash_during_replace_preserves_original(
    tmp_path: Path, monkeypatch
) -> None:
    import pareto_archive as pa_mod
    state_path = tmp_path / "state.json"
    archive = ParetoArchive(state_path=state_path)
    archive.update(ParetoEntry(trial_id=1, objectives=(1.0, 50.0, -0.5, 1.0)))
    archive.save({"trial_counter": 1})
    original = state_path.read_text()
    # Force os.replace to raise on the next save
    real_replace = pa_mod.__dict__.get("os") or os  # paranoia
    def boom(*a, **kw):
        raise OSError("simulated replace failure")
    # pareto_archive imports os as _os inside save() — patch the live module
    monkeypatch.setattr("os.replace", boom)
    archive.update(ParetoEntry(trial_id=2, objectives=(0.5, 80.0, -0.3, 0.9)))
    with pytest.raises(OSError):
        archive.save({"trial_counter": 2})
    assert state_path.read_text() == original
