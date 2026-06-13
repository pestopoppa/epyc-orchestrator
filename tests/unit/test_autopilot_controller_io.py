"""Tests for the extracted autopilot.controller_io module."""

from __future__ import annotations

import io
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

controller_io = importlib.import_module("controller_io")


class _FakePlannerProcess:
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        timeout: bool = False,
    ) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.timeout = timeout
        self.killed = False

    def wait(self, timeout: int) -> int:
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _redirect_planner_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    archive_path = tmp_path / "planner_archive.jsonl"
    monkeypatch.setattr(controller_io, "PLANNER_ARCHIVE_PATH", archive_path)
    monkeypatch.setattr(controller_io, "PLANNER_TAP_PATH", tmp_path / "planner_tap.log")
    return archive_path


def _read_archive_record(path: Path) -> dict:
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


# ----- invoke_controller archival -----


def test_invoke_controller_archives_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = _redirect_planner_logs(monkeypatch, tmp_path)
    proc = _FakePlannerProcess(
        stdout='{"type":"system","subtype":"init","session_id":"sid-fail"}\n',
        stderr="planner failed hard",
        returncode=2,
    )
    monkeypatch.setattr(controller_io.subprocess, "Popen", lambda *a, **k: proc)

    result, session_id = controller_io.invoke_controller(
        "prompt",
        session_id="resume-id",
        timeout=1,
        cwd=tmp_path,
    )

    assert result == ""
    assert session_id == "resume-id"
    record = _read_archive_record(archive_path)
    assert record["subtype"] == "failed"
    assert record["returncode"] == 2
    assert record["stderr_preview"] == "planner failed hard"
    assert record["session_id"] == "sid-fail"
    assert record["resume_session_id"] == "resume-id"
    assert record["n_events"] == 1


def test_invoke_controller_archives_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = _redirect_planner_logs(monkeypatch, tmp_path)
    proc = _FakePlannerProcess(
        stdout='{"type":"system","subtype":"init","session_id":"sid-timeout"}\n',
        timeout=True,
    )
    monkeypatch.setattr(controller_io.subprocess, "Popen", lambda *a, **k: proc)

    result, session_id = controller_io.invoke_controller(
        "prompt",
        session_id="resume-timeout",
        timeout=1,
        cwd=tmp_path,
    )

    assert result == ""
    assert session_id == "resume-timeout"
    assert proc.killed
    record = _read_archive_record(archive_path)
    assert record["subtype"] == "timeout"
    assert record["timed_out"] is True
    assert record["timeout_s"] == 1
    assert record["session_id"] == "sid-timeout"


def test_invoke_controller_archives_missing_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = _redirect_planner_logs(monkeypatch, tmp_path)

    def _raise_missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(controller_io.subprocess, "Popen", _raise_missing)

    result, session_id = controller_io.invoke_controller(
        "prompt",
        session_id="resume-missing",
        timeout=1,
        cwd=tmp_path,
    )

    assert result == ""
    assert session_id == "resume-missing"
    record = _read_archive_record(archive_path)
    assert record["subtype"] == "file_not_found"
    assert record["error"] == "Claude CLI not found"


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


# ----- extract_rationale -----


def test_extract_rationale_well_formed() -> None:
    text = """Action below.

```json:autopilot_actions
{"type": "seed_batch"}
```

```json:autopilot_rationale
{"falsifier": "no quality gain after 20 seeded questions",
 "rubric_scores": {"info_gain": 4, "coherence": 5, "usefulness": 3,
  "synthesis_note": "fused with numeric_trial"}}
```"""
    out = controller_io.extract_rationale(text)
    assert out["falsifier"] == "no quality gain after 20 seeded questions"
    assert out["rubric_scores"]["info_gain"] == 4
    assert out["rubric_scores"]["synthesis_note"] == "fused with numeric_trial"


def test_extract_rationale_missing_block_returns_defaults() -> None:
    out = controller_io.extract_rationale("no rationale here")
    assert out == {"falsifier": "", "rubric_scores": {}}


def test_extract_rationale_malformed_json_returns_defaults() -> None:
    text = """```json:autopilot_rationale
{not valid
```"""
    out = controller_io.extract_rationale(text)
    assert out == {"falsifier": "", "rubric_scores": {}}


def test_extract_rationale_unclosed_fence_returns_defaults() -> None:
    text = """```json:autopilot_rationale
{"falsifier": "x"}
"""
    out = controller_io.extract_rationale(text)
    assert out == {"falsifier": "", "rubric_scores": {}}


def test_extract_rationale_coerces_non_string_falsifier() -> None:
    text = """```json:autopilot_rationale
{"falsifier": 42, "rubric_scores": {"info_gain": 1}}
```"""
    out = controller_io.extract_rationale(text)
    assert out["falsifier"] == "42"
    assert out["rubric_scores"] == {"info_gain": 1}


def test_extract_rationale_non_dict_rubric_falls_back_to_empty() -> None:
    text = """```json:autopilot_rationale
{"falsifier": "x", "rubric_scores": "not a dict"}
```"""
    out = controller_io.extract_rationale(text)
    assert out["falsifier"] == "x"
    assert out["rubric_scores"] == {}


# ----- validate_single_variable (AP-9) -----


def test_validate_prompt_mutation_requires_file() -> None:
    err = controller_io.validate_single_variable({"type": "prompt_mutation"})
    assert err and "must specify a single target file" in err


def test_validate_prompt_mutation_rejects_multi_file() -> None:
    err = controller_io.validate_single_variable({"type": "prompt_mutation", "file": "a.md,b.md"})
    assert err and "multiple files" in err


def test_validate_prompt_mutation_accepts_single_file() -> None:
    assert (
        controller_io.validate_single_variable({"type": "prompt_mutation", "file": "frontdoor.md"})
        is None
    )


def test_validate_code_mutation_requires_file() -> None:
    err = controller_io.validate_single_variable({"type": "code_mutation"})
    assert err and "must specify a single target file" in err


def test_validate_structural_experiment_blocks_multi_flag() -> None:
    err = controller_io.validate_single_variable(
        {"type": "structural_experiment", "flags": {"a": True, "b": False}}
    )
    assert err and "2 flags at once" in err


def test_validate_structural_experiment_accepts_single_flag() -> None:
    assert (
        controller_io.validate_single_variable(
            {"type": "structural_experiment", "flags": {"a": True}}
        )
        is None
    )


def test_validate_numeric_trial_blocks_multi_param() -> None:
    err = controller_io.validate_single_variable(
        {"type": "numeric_trial", "params": {"x": 1, "y": 2}}
    )
    assert err and "sets 2 params explicitly" in err


def test_validate_numeric_trial_accepts_empty_params_for_optuna() -> None:
    # Empty params = Optuna will suggest; exempt from single-variable rule
    assert controller_io.validate_single_variable({"type": "numeric_trial", "params": {}}) is None


def test_validate_numeric_trial_rejects_unknown_surface() -> None:
    err = controller_io.validate_single_variable(
        {"type": "numeric_trial", "surface": "not_a_surface", "params": {}}
    )
    assert err and "surface must be one of" in err


def test_validate_mutation_rejects_unknown_keys_and_bad_enums() -> None:
    err = controller_io.validate_single_variable(
        {
            "type": "code_mutation",
            "file": "src/escalation.py",
            "mutation": "targeted_fix",
            "target_function": "route",
        }
    )
    assert err and "unsupported keys" in err
    assert "target_function" in err

    err = controller_io.validate_single_variable(
        {
            "type": "prompt_mutation",
            "file": "frontdoor.md",
            "mutation": "rewrite_everything",
        }
    )
    assert err and "mutation must be one of" in err


def test_validate_gepa_rejects_unbounded_max_evals() -> None:
    err = controller_io.validate_single_variable(
        {"type": "gepa_optimize", "file": "frontdoor.md", "max_evals": 5000}
    )
    assert err and "max_evals must be <=" in err


def test_validate_slot_compact_schema_matches_handler() -> None:
    assert (
        controller_io.validate_single_variable(
            {
                "type": "slot_compact",
                "port": 8070,
                "slot_id": 0,
                "keep_ratio": 0.3,
                "scorer": "expected_attention",
                "keep_first": 5,
                "n_future": 128,
                "use_covariance": True,
            }
        )
        is None
    )

    err = controller_io.validate_single_variable(
        {
            "type": "slot_compact",
            "port": 8070,
            "slot_id": 0,
            "keep_ratio": 0.3,
            "beta": 0.5,
            "keep_last": 10,
        }
    )
    assert err and "unsupported keys" in err
    assert "beta" in err
    assert "keep_last" in err


def test_validate_slot_compact_rejects_bad_ranges() -> None:
    err = controller_io.validate_single_variable(
        {
            "type": "slot_compact",
            "port": 8070,
            "slot_id": 0,
            "keep_ratio": 1.5,
        }
    )
    assert err and "keep_ratio must be <=" in err

    err = controller_io.validate_single_variable(
        {"type": "slot_compact", "port": 8070, "n_future": 0}
    )
    assert err and "n_future must be >=" in err


def test_validate_deep_eval_rejects_ignored_schema_fields() -> None:
    err = controller_io.validate_single_variable(
        {
            "type": "deep_eval",
            "tier": 2,
            "target_trial": 38,
            "suites": ["coder"],
            "baseline_recheck": True,
            "n_questions": 500,
            "seed": 1234,
        }
    )
    assert err and "unsupported keys" in err
    assert "target_trial" in err
    assert "n_questions" in err
    assert "seed" in err


def test_validate_deep_eval_requires_valid_tier() -> None:
    assert controller_io.validate_single_variable({"type": "deep_eval", "tier": 2}) is None
    assert controller_io.validate_single_variable({"type": "deep_eval"})
    assert controller_io.validate_single_variable({"type": "deep_eval", "tier": 3})
    assert controller_io.validate_single_variable({"type": "deep_eval", "tier": "2"})


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
