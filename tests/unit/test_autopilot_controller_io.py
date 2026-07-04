"""Tests for the extracted autopilot.controller_io module."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


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


def test_validate_numeric_trial_accepts_all_configured_surfaces() -> None:
    from species.numeric_swarm import SURFACES

    assert set(SURFACES) == controller_io._NUMERIC_SURFACES
    assert controller_io.validate_single_variable(
        {"type": "numeric_trial", "surface": "kv_compaction", "params": {}}
    ) is None


def test_validate_numeric_trial_rejects_unknown_surface() -> None:
    err = controller_io.validate_single_variable(
        {"type": "numeric_trial", "surface": "not_a_surface", "params": {}}
    )
    assert err and "surface must be one of" in err


def test_validate_numeric_trial_rejects_suppressed_surface() -> None:
    try:
        controller_io.set_suppressed_numeric_surfaces({"kv_compaction"})
        err = controller_io.validate_single_variable(
            {"type": "numeric_trial", "surface": "kv_compaction", "params": {}}
        )
        assert err and "surface must be one of" in err
        assert "kv_compaction" not in controller_io._NUMERIC_SURFACES
    finally:
        controller_io.set_suppressed_numeric_surfaces(set())


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
    assert controller_io.validate_single_variable(
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
    ) is None

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

    err = controller_io.validate_single_variable(
        {"type": "slot_compact", "port": 0}
    )
    assert err and "port must be >=" in err


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
    assert controller_io.validate_single_variable(
        {"type": "deep_eval", "tier": 2}
    ) is None
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


def test_invoke_controller_archives_timeout_before_return(monkeypatch) -> None:
    records = []
    statuses = []

    class FakeTimeoutProcess:
        stdout = iter(())
        stderr = None
        pid = 12345
        returncode = None

        def wait(self, timeout):
            raise controller_io.subprocess.TimeoutExpired(cmd="claude", timeout=timeout)

        def kill(self):
            pass

    monkeypatch.setattr(controller_io.subprocess, "Popen", lambda *a, **k: FakeTimeoutProcess())
    monkeypatch.setattr(controller_io, "_open_planner_tap", lambda: None)
    monkeypatch.setattr(controller_io, "_append_planner_archive", records.append)
    monkeypatch.setattr(
        controller_io,
        "_write_planner_subprocess_status",
        lambda **kwargs: statuses.append(kwargs),
    )

    text, session_id = controller_io.invoke_controller(
        "prompt",
        session_id="old-session",
        timeout=1,
    )

    assert text == ""
    assert session_id == "old-session"
    assert len(records) == 1
    assert records[0]["type"] == "planner_provider_call"
    assert records[0]["provider"] == "claude"
    assert records[0]["status"] == "timeout"
    assert records[0]["ok"] is False
    assert records[0]["resume_session_id"] == "old-session"
    assert [status["status"] for status in statuses] == ["running", "timeout"]
    assert statuses[0]["child_pid"] == 12345


def test_invoke_controller_pins_planner_model_args(monkeypatch) -> None:
    captured = {}

    class FakeTimeoutProcess:
        stdout = iter(())
        stderr = None
        pid = 12345
        returncode = None

        def wait(self, timeout):
            raise controller_io.subprocess.TimeoutExpired(cmd="claude", timeout=timeout)

        def kill(self):
            pass

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return FakeTimeoutProcess()

    monkeypatch.setenv("CLAUDECODE", "parent-session")
    monkeypatch.setenv("AUTOPILOT_CLAUDE_MODEL", "opus")
    monkeypatch.setenv("AUTOPILOT_CLAUDE_FALLBACK_MODEL", "sonnet")
    monkeypatch.setattr(controller_io.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(controller_io, "_open_planner_tap", lambda: None)
    monkeypatch.setattr(controller_io, "_append_planner_archive", lambda record: None)

    controller_io.invoke_controller("prompt", timeout=1)

    assert captured["cmd"][:3] == ["claude", "-p", "prompt"]
    assert "--permission-mode" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--permission-mode") + 1] == "default"
    assert "--safe-mode" in captured["cmd"]
    assert "--tools" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == "Read,Grep,Glob"
    assert "--allowedTools" in captured["cmd"]
    assert (
        captured["cmd"][captured["cmd"].index("--allowedTools") + 1]
        == "Read,Grep,Glob"
    )
    assert "--disallowedTools" in captured["cmd"]
    disallowed = set(
        captured["cmd"][captured["cmd"].index("--disallowedTools") + 1].split(",")
    )
    assert {"Bash", "Edit", "Task", "Write"}.issubset(disallowed)
    assert "MultiEdit" not in disallowed
    assert "MultiEdit" in controller_io.PLANNER_DISALLOWED_TOOLS
    assert disallowed.isdisjoint(controller_io.PLANNER_ALLOWED_TOOLS)
    assert "--model" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == "opus"
    assert "--fallback-model" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--fallback-model") + 1] == "sonnet"
    assert "CLAUDECODE" not in captured["env"]


def test_invoke_controller_rejects_disallowed_tool_use(monkeypatch) -> None:
    records = []
    statuses = []

    class FakeSuccessProcess:
        stderr = None
        pid = 12345
        returncode = 0
        stdout = iter(
            [
                '{"type":"system","subtype":"init","session_id":"new-session"}\n',
                (
                    '{"type":"assistant","message":{"content":['
                    '{"type":"tool_use","name":"Write","input":{"file_path":"x"}}'
                    ']}}\n'
                ),
                (
                    '{"type":"result","subtype":"success","session_id":"new-session",'
                    '"result":"```json:autopilot_actions\\n'
                    '{\\"type\\":\\"numeric_trial\\",\\"surface\\":\\"monitor\\",\\"params\\":{}}'
                    '\\n```"}\n'
                ),
            ]
        )

        def wait(self, timeout):
            return self.returncode

        def kill(self):
            pass

    monkeypatch.setattr(controller_io.subprocess, "Popen", lambda *a, **k: FakeSuccessProcess())
    monkeypatch.setattr(controller_io, "_open_planner_tap", lambda: None)
    monkeypatch.setattr(controller_io, "_append_planner_archive", records.append)
    monkeypatch.setattr(
        controller_io,
        "_write_planner_subprocess_status",
        lambda **kwargs: statuses.append(kwargs),
    )

    text, session_id = controller_io.invoke_controller(
        "prompt",
        session_id="old-session",
        timeout=1,
    )

    assert text == ""
    assert session_id is None
    assert records[-1]["status"] == "disallowed_tool_use"
    assert records[-1]["ok"] is False
    assert "Write" in records[-1]["error"]
    assert [status["status"] for status in statuses] == [
        "running",
        "disallowed_tool_use",
    ]
