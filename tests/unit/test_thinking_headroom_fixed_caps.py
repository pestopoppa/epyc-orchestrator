"""HF-THINK-CAPS-2: ``thinking_budget_headroom`` also guards the hardcoded llm_call caps
(stages.py escalation 2048, stream_adapter 1024); no model/server/network."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.backends import repl_thinking_budget as budget
from src.features import get_features, reset_features, set_features


def _flags(headroom: bool, thinking: bool = True) -> None:
    set_features(get_features(override={"repl_thinking_budget": thinking, "thinking_budget_headroom": headroom}))


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv(budget.ANSWER_FLOOR_ENV, raising=False)
    reset_features()
    yield
    reset_features()


@pytest.mark.parametrize("on,cap,inherited,expected", [
    (True, 2048, 4000, 1792),    # above room: clamped to cap - 256
    (True, 2048, 1792, 1792),    # exactly at room: unchanged
    (True, 2048, 1000, 1000),    # below room: untouched
    (True, 1024, 1024, 768),     # at the cap itself: clamped
    (True, 1024, 5000, 768),
    (True, 256, 900, None),      # no room above the floor: budget refused
    (False, 2048, 4000, 4000),   # flag off: untouched
    (False, 1024, 5000, 5000),
])
def test_scope_clamps_or_refuses(on, cap, inherited, expected):
    _flags(on)
    with budget.budget_scope(inherited):
        with budget.fixed_cap_budget_scope(cap):
            assert budget.current_budget() == expected
        assert budget.current_budget() == inherited  # restored


def test_scope_honours_floor_knob(monkeypatch):
    monkeypatch.setenv(budget.ANSWER_FLOOR_ENV, "1000")
    _flags(True)
    with budget.budget_scope(3000), budget.fixed_cap_budget_scope(2048):
        assert budget.current_budget() == 1048


def test_scope_noop_without_inherited_budget_or_thinking_flag():
    _flags(True)
    with budget.fixed_cap_budget_scope(2048):
        assert budget.current_budget() is None
    _flags(True, thinking=False)
    with budget.fixed_cap_budget_scope(2048):
        assert budget.current_budget() is None


def _escalate_seen_budget(inherited: int) -> tuple[int | None, dict]:
    import src.api.routes.chat_pipeline.stages as stages
    from src.roles import Role

    seen: list = []

    def llm_call(prompt, **kwargs):
        seen.append(budget.current_budget())
        return "better"

    primitives = MagicMock()
    primitives.llm_call.side_effect = llm_call
    with patch.object(stages, "_detect_output_quality_issue", lambda a: "repetition"), \
         patch.object(stages, "features", lambda: MagicMock(generation_monitor=True)):
        with budget.budget_scope(inherited):
            answer, role = stages._quality_escalate("bad bad", "p", primitives, Role.FRONTDOOR)
    assert answer == "better"
    return seen[0], primitives.llm_call.call_args.kwargs


def test_stages_escalation_on_clamps():
    _flags(True)
    seen, kwargs = _escalate_seen_budget(5000)
    assert seen == 1792 and kwargs["n_tokens"] == 2048


def test_stages_escalation_off_is_unchanged():
    _flags(False)
    seen, kwargs = _escalate_seen_budget(5000)
    assert seen == 5000 and kwargs["n_tokens"] == 2048


def _stream_repl_seen_budget(requested: int) -> tuple[int | None, dict]:
    from src.api.routes.chat_pipeline import stream_adapter

    seen: list = []

    def llm_call(prompt, **kwargs):
        seen.append(budget.current_budget())
        raise RuntimeError("stop after capture")

    primitives = MagicMock()
    primitives.llm_call.side_effect = llm_call
    request = MagicMock(
        prompt="hi", context=None, thinking_budget=requested, max_tokens=None, role="frontdoor",
        max_turns=2, force_role=None, real_mode=True,
    )
    routing = MagicMock(task_id="t1", reasoning_budget=None, generation_max_tokens=None)
    routing.task_ir = {"task_type": "chat"}
    state = MagicMock()
    state.progress_logger = None

    async def run():
        async for _ in stream_adapter._stream_repl(request, routing, primitives, state, 0.0):
            pass

    with patch.object(stream_adapter, "REPLEnvironment", MagicMock()), \
         patch.object(stream_adapter, "score_completed_task", MagicMock()), \
         patch.object(stream_adapter, "build_root_lm_prompt", MagicMock(return_value="rp")):
        asyncio.run(run())
    return seen[0], primitives.llm_call.call_args.kwargs


def test_stream_adapter_on_clamps():
    _flags(True)
    seen, kwargs = _stream_repl_seen_budget(5000)
    assert seen == 768 and kwargs["n_tokens"] == 1024


def test_stream_adapter_off_is_unchanged():
    _flags(False)
    seen, kwargs = _stream_repl_seen_budget(5000)
    assert seen == 5000 and kwargs["n_tokens"] == 1024
