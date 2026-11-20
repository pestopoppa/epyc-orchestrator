"""Harness-robustness guards in the REPL loop (all default OFF, flag-off byte-identical).

A. ``repl_syntax_loop_breaker``  - 2nd consecutive SyntaxError turn gets a one-shot hint, 3rd ends the task
                                   as ``harness:syntax_loop`` (infra-style artifact, no escalation).
B. ``repl_plan_narration_nudge`` - prose that only narrates a plan is nudged, not rescued into a bogus FINAL.
C. ``harness_failure_ratio_guard`` - REPL+tool failures / turns >= 0.5 over >= 4 turns => harness_suspect,
                                   ``_should_escalate`` returns False.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.escalation import ErrorCategory
from src.features import _FEATURE_REGISTRY, _REGISTRY_BY_NAME, Features
from src.graph.answer_resolution import _looks_like_plan_narration
from src.graph.decision_gates import _harness_failure_ratio_tripped, _should_escalate
from src.graph.helpers import (
    PLAN_NARRATION_NUDGE,
    SYNTAX_LOOP_HINT,
    _execute_turn,
)
from src.graph.state import TaskState
from src.repl_environment import REPLConfig, REPLEnvironment
from src.roles import Role

FLAGS = ("repl_syntax_loop_breaker", "repl_plan_narration_nudge", "harness_failure_ratio_guard")


async def _async_none(*_a, **_k):
    return None


def _drive(monkeypatch, raw_outputs, *, flags=(), state=None):
    """Return a coroutine factory running ``_execute_turn`` once per scripted raw LLM output."""
    state = state or TaskState(task_id="t", prompt="p", escalation_prompt="skip prompt build")
    repl = REPLEnvironment(context="test", config=REPLConfig(timeout_seconds=5))
    ctx = SimpleNamespace(state=state, deps=SimpleNamespace(repl=repl, primitives=SimpleNamespace(llm_call=lambda *a, **k: "x")))
    flagset = Features(
        state_history_snapshots=False, generalized_interrupts=False, session_token_budget=False,
        tool_output_compression=False, batch_edit_mode=False, interleaved_edit_rider=False,
        **{f: True for f in flags},
    )
    monkeypatch.setattr("src.features.features", lambda: flagset)
    outputs = list(raw_outputs)

    async def _llm(*_a, **_k):
        return outputs.pop(0), {}

    for name, val in {
        "_use_inline_calls_in_tests": lambda: False,
        "_answer_force_due": lambda _s, **_k: False,
        "_init_session_log": lambda _s, **_k: None,
        "_clear_stale_tool_outputs": lambda _s, **_k: 0,
        "_maybe_compact_context": _async_none,
        "_auto_seed_tasks_from_task_ir": lambda _s, **_k: None,
        "_auto_gather_context": lambda *_a, **_k: "",
        "_check_anti_pattern": lambda _c: "",
        "_maybe_refresh_session_summary": _async_none,
        "_session_log_prompt_block": lambda _s, **_k: "",
        "_budget_pressure_warnings": lambda _s, **_k: "",
        "_maybe_compress_for_escalation": lambda p, _s: p,
        "_call_llm_capturing_meta": _llm,
        "_bep_turn_trace": lambda *_a, **_k: None,
        "_backend_infra_sentinel": lambda _t: None,
        "_check_reasoning_length_alarm": lambda *_a, **_k: False,
        "_maybe_batch_edit_turn": _async_none,
        "_repl_loop_guard_enabled": lambda: False,
        "_persist_solution_file": lambda *_a, **_k: None,
        "_update_workspace_from_turn": lambda *_a, **_k: None,
        "_tap_write_repl_exec": lambda *_a, **_k: None,
        "_tap_write_repl_result": lambda *_a, **_k: None,
        "_get_exploration_tool_calls": lambda *_a, **_k: [],
        "_record_session_turn": Mock(),
    }.items():
        monkeypatch.setattr(f"src.graph.helpers.{name}", val)

    def run_turn():
        return asyncio.run(_execute_turn(ctx, "worker_general"))

    return state, run_turn


BAD_SYNTAX = "```python\nx = (1 +\n```"


# ── registration / surface ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("flag", FLAGS)
def test_flags_registered_default_off_and_on_autopilot_surface(flag):
    spec = _REGISTRY_BY_NAME[flag]
    assert spec.default_test is False and spec.default_prod is False
    assert getattr(Features(), flag) is False
    assert any(s.name == flag for s in _FEATURE_REGISTRY)
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert f'"{flag}"' in (root / "scripts/autopilot/config_applicator.py").read_text()
    assert f"  {flag}: baseline" in (root / "orchestration/runtime_flags.spec.yaml").read_text()


# ── A: syntax loop ─────────────────────────────────────────────────────────────────────────────
def test_syntax_loop_flag_on_counts_hints_at_2_and_ends_at_3(monkeypatch):
    state, run = _drive(monkeypatch, [BAD_SYNTAX] * 3, flags=("repl_syntax_loop_breaker",))
    out1 = run()
    assert out1[1].startswith("SyntaxError") and SYNTAX_LOOP_HINT not in out1[1]
    assert state.syntax_error_streak == 1
    out2 = run()
    assert out2[1].startswith("SyntaxError") and out2[1].endswith(SYNTAX_LOOP_HINT)
    assert out2[3] == {} or "_infra_failure" not in out2[3]
    out3 = run()
    assert out3[1].startswith("[FAILED: harness:syntax_loop")
    assert out3[2] is False
    assert out3[3]["_infra_failure"] is True
    assert out3[3]["harness_failure_reason"] == "harness:syntax_loop"


def test_syntax_loop_streak_resets_on_non_syntax_turn(monkeypatch):
    state, run = _drive(
        monkeypatch, [BAD_SYNTAX, "```python\nprint('ok')\n```", BAD_SYNTAX, BAD_SYNTAX],
        flags=("repl_syntax_loop_breaker",),
    )
    run()
    run()
    assert state.syntax_error_streak == 0
    run()
    out = run()
    assert out[1].endswith(SYNTAX_LOOP_HINT)  # streak restarted: hint at 2, not terminal


def test_syntax_loop_flag_off_is_inert(monkeypatch):
    state, run = _drive(monkeypatch, [BAD_SYNTAX] * 4)
    outs = [run() for _ in range(4)]
    for o in outs:
        assert o[1].startswith("SyntaxError")
        assert SYNTAX_LOOP_HINT not in o[1]
        assert "_infra_failure" not in o[3]
    assert state.syntax_error_streak == 0 and state.syntax_hint_sent is False


# ── B: plan narration ──────────────────────────────────────────────────────────────────────────
PLAN_POSITIVES = [
    "I will read the file first and then summarize it.",
    "Let me look at the repository layout.",
    "Next I need to inspect the failing test.",
    "First, I'll open the config.",
    "Plan: 1. read file 2. patch it",
    "Sure.\nLet's start by listing the directory.",
    "I'm going to run the tests.",
]
PLAN_NEGATIVES = [
    "The answer is B",
    "Answer: C",
    "I will go with D",
    "I'll go with A.",
    "I choose B",
    "B",
    "42",
    "Therefore the answer is 17",
    "",
    "Let me think.\n```python\nprint(1)\n```",  # has code: not prose-only
    "Let me be clear: the answer is C",  # explicit answer wins
]


@pytest.mark.parametrize("text", PLAN_POSITIVES)
def test_plan_heuristic_positive(text):
    assert _looks_like_plan_narration(text) is True


@pytest.mark.parametrize("text", PLAN_NEGATIVES)
def test_plan_heuristic_negative(text):
    assert _looks_like_plan_narration(text) is False


def test_plan_narration_flag_on_nudges_instead_of_final(monkeypatch):
    _, run = _drive(monkeypatch, ["I will read the file first."], flags=("repl_plan_narration_nudge",))
    out, err, is_final, artifacts = run()
    assert (out, err, is_final) == ("", None, False)
    assert artifacts == {"_nudge": PLAN_NARRATION_NUDGE}
    assert PLAN_NARRATION_NUDGE == "No code block found. Emit one ```python block now that performs the next step."


def test_plan_narration_flag_on_still_rescues_real_prose_answer(monkeypatch):
    _, run = _drive(monkeypatch, ["The answer is B"], flags=("repl_plan_narration_nudge",))
    out, err, is_final, artifacts = run()
    assert is_final is True and out == "B" and "_nudge" not in artifacts


def test_plan_narration_flag_off_unchanged(monkeypatch):
    _, run = _drive(monkeypatch, ["I will go with B"])
    out, err, is_final, _ = run()
    assert is_final is True and out == "B"


def test_plan_narration_flag_off_never_emits_plan_nudge(monkeypatch):
    _, run = _drive(monkeypatch, ["I will read the file first."])
    assert run()[3].get("_nudge") != PLAN_NARRATION_NUDGE


# ── C: ratio guard ─────────────────────────────────────────────────────────────────────────────
def _esc_ctx(turns, repl_fail=0, tool_fail=0):
    ctx = MagicMock()
    ctx.deps.config.no_escalate_categories = {ErrorCategory.FORMAT}
    ctx.deps.config.max_escalations = 3
    ctx.deps.config.max_retries = 2
    ctx.deps.config.max_regenerations = 0
    s = ctx.state
    s.escalation_count = 0
    s.consecutive_failures = 3
    s.last_error = ""
    s.role_history = ["worker"]
    s.turns = turns
    s.harness_repl_failures = repl_fail
    s.harness_tool_failures = tool_fail
    s.total_repl_failures = 0
    s.artifacts = {}
    return ctx


def _with_flag(on):
    flags = Features(harness_failure_ratio_guard=on)
    return patch("src.features.features", lambda: flags)


def test_ratio_tripped_thresholds():
    st = lambda t, r, k: SimpleNamespace(turns=t, harness_repl_failures=r, harness_tool_failures=k)  # noqa: E731
    assert _harness_failure_ratio_tripped(st(4, 2, 0)) is True
    assert _harness_failure_ratio_tripped(st(4, 1, 1)) is True  # repl + tool combine
    assert _harness_failure_ratio_tripped(st(4, 1, 0)) is False  # 0.25
    assert _harness_failure_ratio_tripped(st(6, 2, 0)) is False  # 0.33
    assert _harness_failure_ratio_tripped(st(3, 3, 0)) is False  # under 4 turns


def test_ratio_guard_on_blocks_escalation_and_marks_suspect():
    ctx = _esc_ctx(turns=4, repl_fail=2)
    with _with_flag(True), patch("src.graph.decision_gates._detect_role_cycle_impl", return_value=False):
        assert _should_escalate(ctx, ErrorCategory.CODE, Role.CODER_ESCALATION) is False
    assert ctx.state.artifacts["harness_suspect"] is True


def test_ratio_guard_on_below_threshold_still_escalates():
    ctx = _esc_ctx(turns=4, repl_fail=1)
    with _with_flag(True), patch("src.graph.decision_gates._detect_role_cycle_impl", return_value=False):
        assert _should_escalate(ctx, ErrorCategory.CODE, Role.CODER_ESCALATION) is True
    assert "harness_suspect" not in ctx.state.artifacts


def test_ratio_guard_on_under_four_turns_still_escalates():
    ctx = _esc_ctx(turns=3, repl_fail=3)
    with _with_flag(True), patch("src.graph.decision_gates._detect_role_cycle_impl", return_value=False):
        assert _should_escalate(ctx, ErrorCategory.CODE, Role.CODER_ESCALATION) is True


def test_ratio_guard_off_escalates_as_before():
    ctx = _esc_ctx(turns=4, repl_fail=4)
    with _with_flag(False), patch("src.graph.decision_gates._detect_role_cycle_impl", return_value=False):
        assert _should_escalate(ctx, ErrorCategory.CODE, Role.CODER_ESCALATION) is True
    assert "harness_suspect" not in ctx.state.artifacts


def test_ratio_tally_counts_repl_and_tool_failures_only_when_on(monkeypatch):
    on_state, run_on = _drive(
        monkeypatch, ["```python\nraise ValueError('x')\n```", "```python\nprint('[ERROR: tool down]')\n```"],
        flags=("harness_failure_ratio_guard",),
    )
    run_on()
    run_on()
    assert (on_state.harness_repl_failures, on_state.harness_tool_failures) == (1, 1)
    off_state, run_off = _drive(monkeypatch, ["```python\nraise ValueError('x')\n```"])
    run_off()
    assert (off_state.harness_repl_failures, off_state.harness_tool_failures) == (0, 0)
