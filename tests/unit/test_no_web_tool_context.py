"""WS-3 no_web reaches every request REPL, not only the /chat REPL/react/stream ones (A4-M1).

The delegated-specialist and architect REPLs (default structured path), both /v1
REPL loops and the legacy /chat/stream REPL used to pass no tool_context, so a
math/thinking task could still CALL('web_search', ...).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import ChatRequest
from src.tool_policy import tool_context_for

NO_WEB = {"no_web": True}


class _Stop(BaseException):
    """Escapes the callee's `except Exception` blocks once construction is recorded."""


def _recording_repl(seen: list[dict]):
    def factory(*_args, **kwargs):
        seen.append(kwargs.get("tool_context"))
        raise _Stop

    return factory


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [("math", NO_WEB), ("thinking", NO_WEB), ("chat", {}), ("coder", {}), (None, {})],
)
def test_tool_context_for(task_type, expected):
    assert tool_context_for(task_type) == expected


def test_architect_and_specialist_repls_get_tool_context():
    from src.api.routes import chat_delegation

    seen: list[dict] = []
    with patch.object(chat_delegation, "REPLEnvironment", _recording_repl(seen)):
        with pytest.raises(_Stop):
            chat_delegation._run_architect_decision(
                "p", "q", MagicMock(), "architect_general", None, tool_context=NO_WEB
            )
        with pytest.raises(_Stop):
            chat_delegation._run_specialist_loop(
                "q", "", "brief", "worker_math", "repl", MagicMock(), None, tool_context=NO_WEB
            )
    assert seen == [NO_WEB, NO_WEB]


def test_architect_delegation_threads_tool_context_to_both_phases():
    from src.api.routes import chat_delegation

    decisions = iter([("I|brief:check|to:coder_escalation", 1, 0), ("D|done", 1, 0)])
    arch = MagicMock(side_effect=lambda *a, **kw: next(decisions))
    spec = MagicMock(return_value=("report", 0, [], [], False, False, {}, []))
    primitives = MagicMock()
    primitives.total_tokens_generated = 0
    with (
        patch.object(chat_delegation, "_run_architect_decision", arch),
        patch.object(chat_delegation, "_run_specialist_loop", spec),
    ):
        chat_delegation._architect_delegated_answer(
            question="q",
            context="",
            primitives=primitives,
            state=SimpleNamespace(tool_registry=None),
            tool_context=NO_WEB,
        )
    assert arch.call_count == 2
    assert all(c.kwargs.get("tool_context") == NO_WEB for c in arch.call_args_list)
    assert spec.call_args.kwargs["tool_context"] == NO_WEB


def test_delegated_stage_derives_tool_context_from_task_ir():
    from src.api.routes.chat_pipeline import delegation_stage

    seen: list[dict] = []

    def record(**kwargs):
        seen.append(kwargs.get("tool_context"))
        raise _Stop

    routing = SimpleNamespace(task_id="t", task_ir={"task_type": "math"})
    with patch.object(delegation_stage, "_architect_delegated_answer", record):
        with pytest.raises(_Stop):
            delegation_stage._execute_delegated(
                ChatRequest(prompt="2+2", real_mode=True),
                routing,
                MagicMock(),
                SimpleNamespace(),
                time.perf_counter(),
                "architect_general",
                "delegated",
            )
    assert seen == [NO_WEB]


def test_v1_repl_gets_tool_context_for_math_role(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app
    from src.api.routes import openai_compat
    from src.api.state import get_state, reset_state
    from src.features import reset_features
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    reset_features()
    reset_state()
    primitives = MagicMock()
    primitives.llm_call.return_value = 'FINAL("4")'
    primitives.total_tokens_generated = 1
    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    seen: list[dict] = []
    real_repl = openai_compat.REPLEnvironment

    def recording(*args, **kwargs):
        seen.append(kwargs.get("tool_context"))
        return real_repl(*args, **kwargs)

    monkeypatch.setattr(openai_compat, "REPLEnvironment", recording)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            get_state().registry = MagicMock()
            for stream in (False, True):
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "worker_math",
                        "messages": [{"role": "user", "content": "2+2"}],
                        "stream": stream,
                    },
                )
                assert resp.status_code == 200, resp.text
    finally:
        reset_features()
    assert seen == [NO_WEB, NO_WEB]
