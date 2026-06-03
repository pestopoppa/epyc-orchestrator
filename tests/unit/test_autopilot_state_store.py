"""Tests for the extracted autopilot.state_store module."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

state_store = importlib.import_module("state_store")


# ----- load_state / save_state -----


def test_load_state_returns_default_when_file_missing(tmp_path) -> None:
    out = state_store.load_state(tmp_path / "no_such.json", lambda: {"trial_counter": 0})
    assert out == {"trial_counter": 0}


def test_load_state_reads_existing_json(tmp_path) -> None:
    f = tmp_path / "state.json"
    f.write_text(json.dumps({"trial_counter": 42, "paused": True}))
    out = state_store.load_state(f, lambda: {"trial_counter": 0})
    assert out == {"trial_counter": 42, "paused": True}


def test_save_state_writes_json_and_creates_parent(tmp_path) -> None:
    f = tmp_path / "nested" / "dir" / "state.json"
    state_store.save_state(f, {"trial_counter": 5})
    assert f.exists()
    assert json.loads(f.read_text()) == {"trial_counter": 5}


# ----- blacklist -----


def test_load_blacklist_returns_empty_when_missing(tmp_path) -> None:
    assert state_store.load_blacklist(tmp_path / "no_such.yaml") == []


def test_load_blacklist_parses_yaml(tmp_path) -> None:
    f = tmp_path / "bl.yaml"
    f.write_text(yaml.dump({"blacklist": [
        {"pattern": {"type": "prompt_mutation"}, "reason": "regression"},
    ]}))
    out = state_store.load_blacklist(f)
    assert len(out) == 1
    assert out[0]["reason"] == "regression"


def test_load_blacklist_handles_malformed_yaml(tmp_path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("not: valid: yaml: {{{")
    assert state_store.load_blacklist(f) == []


def test_check_blacklist_matches_full_pattern() -> None:
    bl = [{"pattern": {"type": "x", "file": "a.md"}, "reason": "no"}]
    assert state_store.check_blacklist(
        {"type": "x", "file": "a.md", "extra": 1}, bl
    ) == "no"


def test_check_blacklist_no_match_returns_none() -> None:
    bl = [{"pattern": {"type": "x"}, "reason": "no"}]
    assert state_store.check_blacklist({"type": "y"}, bl) is None


def test_check_blacklist_skips_empty_pattern() -> None:
    bl = [{"pattern": {}, "reason": "no"}]
    assert state_store.check_blacklist({"type": "x"}, bl) is None


def test_check_blacklist_non_dict_action_returns_none() -> None:
    assert state_store.check_blacklist(None, [{"pattern": {"type": "x"}, "reason": "no"}]) is None


def test_append_blacklist_creates_new_file(tmp_path) -> None:
    bl_path = tmp_path / "bl.yaml"
    state_store.append_blacklist(
        {"type": "prompt_mutation", "file": "x.md"}, trial_id=1, reason="bad",
        blacklist_path=bl_path,
    )
    data = yaml.safe_load(bl_path.read_text())
    assert len(data["blacklist"]) == 1
    assert data["blacklist"][0]["pattern"] == {"type": "prompt_mutation", "file": "x.md"}
    assert data["blacklist"][0]["reason"] == "bad"


def test_append_blacklist_appends_to_existing(tmp_path) -> None:
    bl_path = tmp_path / "bl.yaml"
    bl_path.write_text(yaml.dump({"blacklist": [
        {"pattern": {"type": "x"}, "reason": "first"},
    ]}))
    state_store.append_blacklist(
        {"type": "y"}, trial_id=2, reason="second", blacklist_path=bl_path,
    )
    data = yaml.safe_load(bl_path.read_text())
    assert len(data["blacklist"]) == 2
    assert data["blacklist"][1]["reason"] == "second"


def test_append_blacklist_skips_type_only_low_risk_action(tmp_path) -> None:
    bl_path = tmp_path / "bl.yaml"
    state_store.append_blacklist(
        {"type": "deep_eval"}, trial_id=2, reason="second", blacklist_path=bl_path,
    )
    assert not bl_path.exists()


def test_append_blacklist_keeps_specific_low_risk_pattern(tmp_path) -> None:
    bl_path = tmp_path / "bl.yaml"
    state_store.append_blacklist(
        {"type": "deep_eval", "tier": 2}, trial_id=2, reason="second",
        blacklist_path=bl_path,
    )
    data = yaml.safe_load(bl_path.read_text())
    assert data["blacklist"][0]["pattern"] == {"type": "deep_eval", "tier": 2}


def test_append_blacklist_skips_unpatternable_actions(tmp_path) -> None:
    """If no patternable fields are in the action, no entry is written."""
    bl_path = tmp_path / "bl.yaml"
    state_store.append_blacklist(
        {"unrelated": "key"}, trial_id=1, reason="bad", blacklist_path=bl_path,
    )
    assert not bl_path.exists()


# ----- model signatures -----


def test_load_model_signatures_returns_empty_when_missing(tmp_path) -> None:
    assert state_store.load_model_signatures(tmp_path / "no.yaml") == {}


def test_load_model_signatures_parses_yaml(tmp_path) -> None:
    f = tmp_path / "sigs.yaml"
    f.write_text(yaml.dump({"models": {
        "gemma-4-26B": {"role": "worker", "max_throughput_tps": 76.5},
    }}))
    out = state_store.load_model_signatures(f)
    assert "gemma-4-26B" in out
    assert out["gemma-4-26B"]["max_throughput_tps"] == 76.5


def test_format_model_signatures_empty_dict_message() -> None:
    out = state_store.format_model_signatures({})
    assert "no model signatures available" in out


def test_format_model_signatures_table_format() -> None:
    sigs = {
        "gemma-4-26B": {
            "role": "worker",
            "max_throughput_tps": 76.5,
            "per_suite": {"coder": "96%", "math": "84%", "general": "78%"},
        },
    }
    out = state_store.format_model_signatures(sigs)
    assert "| Model | Role |" in out
    assert "gemma-4-26B" in out
    assert "worker" in out
    # Top suite should be coder (96%), bottom should be general (78%)
    assert "coder (96%)" in out
    assert "general (78%)" in out
