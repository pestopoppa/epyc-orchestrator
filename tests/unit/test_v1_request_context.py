"""/v1/chat/completions binds per-request cancellation and task id on every mode (A1-M1).

Before, only the streamed client/direct chunk path bound a request context, so the
non-streaming dispatch and the REPL loops ran with cancel_check=None and
task_id=None (region-lock holders and tap rows tagged 'None:<hex>').
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.state import get_state, reset_state
from src.features import reset_features
from src.llm_primitives import LLMPrimitives


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    monkeypatch.setenv("ORCHESTRATOR_MEMRL", "false")  # No real startup seeding.
    from tests.fixtures.api_startup import route_only_lifespan

    monkeypatch.setattr(app.router, "lifespan_context", route_only_lifespan)
    reset_features()
    reset_state()
    get_state()
    with TestClient(app, raise_server_exceptions=False) as c:
        state = get_state()
        if state.registry is None:
            state.registry = MagicMock()
        yield c
    reset_features()


def _install(monkeypatch) -> list[tuple[str | None, object]]:
    seen: list[tuple[str | None, object]] = []
    primitives = LLMPrimitives(mock_mode=True)

    def record():
        seen.append((primitives.get_request_task_id(), primitives.get_request_cancel_check()))

    def llm_call(*_args, **_kwargs):
        record()
        return 'FINAL("ok")'

    def chat_completion_call(*_args, **_kwargs):
        record()
        return {"content": "ok", "tool_calls": [], "finish_reason": "stop", "usage": {}}

    primitives.llm_call = llm_call
    primitives.chat_completion_call = chat_completion_call

    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    return seen


@pytest.mark.parametrize(
    "extra",
    [
        {"x_disable_repl": True},
        {"x_tool_mode": "client", "x_session_id": "s-1"},
        {},
        {"stream": True},
    ],
    ids=["direct", "client", "repl", "repl_stream"],
)
def test_every_v1_mode_binds_task_id_and_cancel_check(client, monkeypatch, extra):
    seen = _install(monkeypatch)
    body = {"model": "frontdoor", "messages": [{"role": "user", "content": "hi"}], **extra}

    resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200, resp.text
    assert seen, "no model call was made"
    chat_id = resp.json()["id"] if not extra.get("stream") else None
    for task_id, cancel_check in seen:
        assert task_id is not None and task_id.startswith("chatcmpl-")
        if chat_id is not None:
            assert task_id == chat_id
        assert callable(cancel_check)
