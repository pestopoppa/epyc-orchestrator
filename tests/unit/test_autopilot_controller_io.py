"""Tests for the extracted autopilot.controller_io module."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

controller_io = importlib.import_module("controller_io")


# ----- extract_action -----


def test_extract_action_marker_block() -> None:
    text = """Some reasoning here.

```json:autopilot_actions
{"type": "seed_batch", "n_questions": 10}
```

More text after."""
    action = controller_io.extract_action(text)
    assert action == {"type": "seed_batch", "n_questions": 10}


def test_extract_action_marker_unwraps_list() -> None:
    text = """```json:autopilot_actions
[{"type": "numeric_trial", "surface": "x"}]
```"""
    action = controller_io.extract_action(text)
    assert action == {"type": "numeric_trial", "surface": "x"}


def test_extract_action_falls_back_to_generic_json_block() -> None:
    text = """No marker here.

```json
{"type": "rollback", "to_checkpoint": "production_best"}
```"""
    action = controller_io.extract_action(text)
    assert action == {"type": "rollback", "to_checkpoint": "production_best"}


def test_extract_action_returns_none_when_no_block() -> None:
    assert controller_io.extract_action("plain text, no json") is None


def test_extract_action_returns_none_when_no_type_field() -> None:
    text = """```json
{"foo": "bar"}
```"""
    assert controller_io.extract_action(text) is None


def test_extract_action_returns_none_on_invalid_json() -> None:
    text = """```json:autopilot_actions
{not valid json
```"""
    assert controller_io.extract_action(text) is None


# ----- validate_single_variable (AP-9) -----


def test_validate_prompt_mutation_requires_file() -> None:
    err = controller_io.validate_single_variable({"type": "prompt_mutation"})
    assert err and "must specify a single target file" in err


def test_validate_prompt_mutation_rejects_multi_file() -> None:
    err = controller_io.validate_single_variable(
        {"type": "prompt_mutation", "file": "a.md,b.md"}
    )
    assert err and "multiple files" in err


def test_validate_prompt_mutation_accepts_single_file() -> None:
    assert controller_io.validate_single_variable(
        {"type": "prompt_mutation", "file": "frontdoor.md"}
    ) is None


def test_validate_code_mutation_requires_file() -> None:
    err = controller_io.validate_single_variable({"type": "code_mutation"})
    assert err and "must specify a single target file" in err


def test_validate_structural_experiment_blocks_multi_flag() -> None:
    err = controller_io.validate_single_variable(
        {"type": "structural_experiment", "flags": {"a": True, "b": False}}
    )
    assert err and "2 flags at once" in err


def test_validate_structural_experiment_accepts_single_flag() -> None:
    assert controller_io.validate_single_variable(
        {"type": "structural_experiment", "flags": {"a": True}}
    ) is None


def test_validate_numeric_trial_blocks_multi_param() -> None:
    err = controller_io.validate_single_variable(
        {"type": "numeric_trial", "params": {"x": 1, "y": 2}}
    )
    assert err and "sets 2 params explicitly" in err


def test_validate_numeric_trial_accepts_empty_params_for_optuna() -> None:
    # Empty params = Optuna will suggest; exempt from single-variable rule
    assert controller_io.validate_single_variable(
        {"type": "numeric_trial", "params": {}}
    ) is None


def test_validate_unknown_action_type_passes() -> None:
    assert controller_io.validate_single_variable({"type": "unknown_thing"}) is None


# ----- _unwrap_action -----


def test_unwrap_action_dict_with_type() -> None:
    assert controller_io._unwrap_action({"type": "x"}) == {"type": "x"}


def test_unwrap_action_list_takes_first() -> None:
    assert controller_io._unwrap_action([{"type": "x"}, {"type": "y"}]) == {"type": "x"}


def test_unwrap_action_empty_list_none() -> None:
    assert controller_io._unwrap_action([]) is None


def test_unwrap_action_dict_without_type_none() -> None:
    assert controller_io._unwrap_action({"no_type": True}) is None
