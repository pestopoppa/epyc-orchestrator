"""MF-RR-1 source controls for native length cutoffs without an emitted action."""

from __future__ import annotations

import pytest
from pydantic_graph import GraphRunContext

from src import chat_completions_roles as cc
from src.features import get_features, reset_features, set_features
from src.graph.helpers import _execute_turn, _has_executable_action_or_tool
from src.roles import Role


@pytest.fixture(autouse=True)
def _clean_features():
    reset_features()
    yield
    reset_features()


class _ScriptedBackend:
    def __init__(self, outputs, reasons):
        self.outputs = list(outputs)
        self.reasons = list(reasons)
        self.requests = []

    def infer(self, role_config, request):
        from src.model_server import InferenceResult

        self.requests.append(request)
        index = len(self.requests) - 1
        return InferenceResult(
            role=request.role,
            output=self.outputs[index],
            tokens_generated=17,
            output_tokens_reported=17,
            output_tokens_source="server_usage",
            prompt_tokens=11,
            generation_speed=10.0,
            elapsed_time=0.1,
            success=True,
            completion_reason=self.reasons[index],
        )


def _enable_retry():
    set_features(get_features(override={
        "reasoning_runaway_retry": True,
        "thinking_roles_chat_lane": True,
    }))


def _primitives(backend):
    from src.llm_primitives import LLMPrimitives

    primitives = LLMPrimitives(mock_mode=False, server_urls={})
    primitives._backends = {str(Role.ARCHITECT_CRITIC): backend}
    return primitives


@pytest.mark.asyncio
async def test_length_cutoff_retries_once_and_request_builder_carries_thinking_off(
    graph_ctx, monkeypatch,
):
    from src.graph import helpers

    _enable_retry()
    set_features(get_features(override={
        "reasoning_runaway_retry": True,
        "thinking_roles_chat_lane": True,
        "session_token_budget": True,
    }))
    monkeypatch.setattr(cc, "thinking_chat_lane_roles", lambda: {str(Role.ARCHITECT_CRITIC)})
    retry_action = 'print("retry-action")\nFINAL("recovered")'
    backend = _ScriptedBackend(["unfinished thought", retry_action], ["length", "stop"])
    state, deps = graph_ctx(role=Role.ARCHITECT_CRITIC)
    deps.primitives = _primitives(backend)
    workspace_outputs = []
    trace = {}
    trajectory_calls = []
    executed_retry_actions = []
    update_workspace = helpers._update_workspace_from_turn
    execute_repl = deps.repl.execute

    def capture_workspace(state, role, raw_output, error):
        workspace_outputs.append(raw_output)
        return update_workspace(state, role, raw_output, error)

    def capture_repl_execution(code):
        if code == retry_action:
            executed_retry_actions.append(code)
        return execute_repl(code)

    monkeypatch.setattr(helpers, "_update_workspace_from_turn", capture_workspace)
    deps.repl.execute = capture_repl_execution
    monkeypatch.setattr(helpers, "_bep_turn_trace", lambda *args, **kwargs: trace.update(args=args, **kwargs))
    monkeypatch.setattr(
        helpers,
        "_trajectory_note_call",
        lambda state, role, prompt, meta, deps: trajectory_calls.append((prompt, dict(meta))),
    )

    output, error, is_final, _ = await _execute_turn(
        GraphRunContext(state=state, deps=deps),
        Role.ARCHITECT_CRITIC,
    )

    assert error is None and is_final is True and "recovered" in output
    assert len(backend.requests) == 2
    assert backend.requests[0].chat_template_kwargs is None
    assert backend.requests[1].chat_template_kwargs == {"enable_thinking": False}
    assert executed_retry_actions == [retry_action]
    assert state.last_code == retry_action
    assert workspace_outputs == [retry_action]
    assert state.aggregate_tokens == 34
    assert state._session_token_budget.input_tokens == 22
    assert state._session_token_budget.output_tokens == 34
    assert len(trajectory_calls) == 2
    assert trajectory_calls[1][1]["output_tokens_reported"] == 17
    assert trace["args"][2] == "unfinished thought"
    assert trace["action_source_output"] == retry_action
    assert trace["retry_attempted"] is True
    assert trace["primary_meta"]["completion_reason"] == "length"
    assert trace["retry_meta"]["completion_reason"] == "stop"


@pytest.mark.asyncio
async def test_length_cutoff_does_not_retry_when_selected_role_cannot_honor_thinking_off(
    graph_ctx, monkeypatch,
):
    _enable_retry()
    monkeypatch.setattr(cc, "thinking_chat_lane_roles", lambda: set())
    backend = _ScriptedBackend(["unfinished thought"], ["length"])
    state, deps = graph_ctx(role=Role.ARCHITECT_CRITIC)
    deps.primitives = _primitives(backend)

    await _execute_turn(
        GraphRunContext(state=state, deps=deps),
        Role.ARCHITECT_CRITIC,
    )

    assert len(backend.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,reason",
    [
        ('FINAL("already acted")', "length"),
        ('CALL("web_search", query="x")', "length"),
        ("unfinished thought", "tool_calls"),
        ("unfinished thought", None),
    ],
)
async def test_action_tool_call_and_missing_native_reason_do_not_retry(
    graph_ctx, monkeypatch, output, reason,
):
    _enable_retry()
    monkeypatch.setattr(cc, "thinking_chat_lane_roles", lambda: {str(Role.ARCHITECT_CRITIC)})
    backend = _ScriptedBackend([output], [reason])
    state, deps = graph_ctx(role=Role.ARCHITECT_CRITIC)
    deps.primitives = _primitives(backend)

    await _execute_turn(
        GraphRunContext(state=state, deps=deps),
        Role.ARCHITECT_CRITIC,
    )

    assert len(backend.requests) == 1


@pytest.mark.asyncio
async def test_default_off_is_single_call_even_for_native_length_cutoff(graph_ctx, monkeypatch):
    set_features(get_features(override={"thinking_roles_chat_lane": True}))
    monkeypatch.setattr(cc, "thinking_chat_lane_roles", lambda: {str(Role.ARCHITECT_CRITIC)})
    backend = _ScriptedBackend(["unfinished thought"], ["length"])
    state, deps = graph_ctx(role=Role.ARCHITECT_CRITIC)
    deps.primitives = _primitives(backend)

    await _execute_turn(
        GraphRunContext(state=state, deps=deps),
        Role.ARCHITECT_CRITIC,
    )

    assert len(backend.requests) == 1


@pytest.mark.asyncio
async def test_runaway_retry_is_one_shot_even_if_retry_also_hits_length(graph_ctx, monkeypatch):
    _enable_retry()
    monkeypatch.setattr(cc, "thinking_chat_lane_roles", lambda: {str(Role.ARCHITECT_CRITIC)})
    backend = _ScriptedBackend(["unfinished thought", "still unfinished"], ["length", "length"])
    state, deps = graph_ctx(role=Role.ARCHITECT_CRITIC)
    deps.primitives = _primitives(backend)

    await _execute_turn(GraphRunContext(state=state, deps=deps), Role.ARCHITECT_CRITIC)

    assert len(backend.requests) == 2


@pytest.mark.asyncio
async def test_retry_failure_preserves_primary_and_does_not_execute_retry_output(
    graph_ctx, monkeypatch, caplog,
):
    _enable_retry()
    monkeypatch.setattr(cc, "thinking_chat_lane_roles", lambda: {str(Role.ARCHITECT_CRITIC)})

    class FailingRetryBackend(_ScriptedBackend):
        def infer(self, role_config, request):
            if self.requests:
                self.requests.append(request)
                from src.exceptions import InferenceError

                raise InferenceError("scripted retry failure")
            return super().infer(role_config, request)

    backend = FailingRetryBackend(["unfinished thought", 'FINAL("must not run")'], ["length", "stop"])
    state, deps = graph_ctx(role=Role.ARCHITECT_CRITIC)
    deps.primitives = _primitives(backend)

    await _execute_turn(
        GraphRunContext(state=state, deps=deps),
        Role.ARCHITECT_CRITIC,
    )

    assert len(backend.requests) == 2
    assert state.last_code == ""
    assert "preserving primary output" in caplog.text


def test_action_detector_fails_closed_for_unparseable_or_structured_tool_calls():
    assert _has_executable_action_or_tool("```python\nprint(1)\n```", {}) is True
    assert _has_executable_action_or_tool("unfinished thought", {"tool_calls": [{"id": "x"}]}) is True
    assert _has_executable_action_or_tool("unfinished thought", {"completion_reason": "length"}) is False
    assert _has_executable_action_or_tool("```python\nnot valid python !!!\n```", {}) is True


def test_native_structured_tool_calls_are_present_in_call_local_inference_meta():
    from src.llm_primitives import LLMPrimitives
    from src.model_server import InferenceResult

    native_call = {"id": "call-1", "type": "function", "function": {"name": "read"}}

    class NativeToolBackend:
        def infer(self, role_config, request):
            return InferenceResult(
                role=request.role,
                output="",
                tokens_generated=2,
                generation_speed=1.0,
                elapsed_time=0.1,
                success=True,
                completion_reason="tool_calls",
                tool_calls=[native_call],
            )

    primitives = LLMPrimitives(mock_mode=False, server_urls={})
    primitives._backends = {str(Role.FRONTDOOR): NativeToolBackend()}
    result = primitives.chat_completion_call(
        [{"role": "user", "content": "read"}],
        role=str(Role.FRONTDOOR),
        tools=[{"type": "function", "function": {"name": "read"}}],
    )

    assert result["tool_calls"] == [native_call]
    assert primitives.get_last_inference_meta()["tool_calls"] == [native_call]
