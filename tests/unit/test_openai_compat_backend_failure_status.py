#!/usr/bin/env python3
"""HS-OD-2 — backend failures must not be returned as assistant content.

`/v1/chat/completions` used to catch **every** exception into
`response_text = f"[ERROR] Backend failed: {e}"` and fall through to a normal
`OpenAIChatResponse`. A downstream harness then saw a successful completion whose
text merely began with `[ERROR]`: retry logic, error metrics and eval scorers all
treated it as a model answer, so every eval fan-out through `:8000` scored
backend outages as low-quality generations.

These tests pin the seam in both directions — a failure must surface as a real
error status (or a terminal SSE `error` event, since a stream cannot retract its
200), and the ordinary success path must still be a clean 200 with the model's
text. The second half matters as much as the first: a guard that only asserts the
failure would pass just as happily if the route started erroring on everything.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.state import get_state, reset_state
from src.features import reset_features
from src.scheduling.contention_gate import ContentionDenied


def _install_primitives(monkeypatch, *, llm_call):
    """Force real mode and hand the route a primitives double."""
    primitives = MagicMock()
    primitives.llm_call.side_effect = llm_call
    primitives.total_tokens_generated = 7

    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(
        llm_primitives_module, "LLMPrimitives", lambda **_kwargs: primitives
    )
    return primitives


@pytest.fixture
def client(monkeypatch):
    # `use_real_mode` is `registry is not None and not features().mock_mode`, and
    # mock_mode defaults True "for safety" — without both halves the route serves
    # a [MOCK] completion and every assertion below tests nothing.
    # NB the registry's env_var is the SUFFIX — the real variable is
    # ORCHESTRATOR_MOCK_MODE. Plain MOCK_MODE is silently ignored and the route
    # keeps serving [MOCK] completions, which is a green-looking no-op test.
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    reset_features()
    reset_state()
    get_state()
    with TestClient(app, raise_server_exceptions=False) as c:
        state = get_state()
        if state.registry is None:
            state.registry = MagicMock()
        yield c
    reset_features()


def _body(**overrides):
    body = {
        "model": "frontdoor",
        "messages": [{"role": "user", "content": "hello"}],
        "x_disable_repl": True,
    }
    body.update(overrides)
    return body


def test_backend_exception_is_502_not_a_200_completion(client, monkeypatch):
    """The core HS-OD-2 assertion."""
    _install_primitives(
        monkeypatch, llm_call=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("upstream died"))
    )

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 502, (
        f"backend failure returned {r.status_code}; a non-error status here is the "
        "fail-open shape HS-OD-2 exists to close"
    )
    payload = r.json()
    text = json.dumps(payload)
    assert "upstream died" in text
    # The failure must NOT be dressed as a completion.
    assert "choices" not in payload or not payload.get("choices")


def test_contention_denied_keeps_its_own_503(client, monkeypatch):
    """ContentionDenied has a dedicated handler; it must reach it."""
    _install_primitives(
        monkeypatch,
        llm_call=lambda *a, **k: (_ for _ in ()).throw(ContentionDenied("region busy")),
    )

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 503, (
        "ContentionDenied must reach its app-level handler; swallowing it turns "
        "documented back-pressure into a model answer"
    )
    assert "region busy" in json.dumps(r.json())


def test_success_path_is_still_a_clean_200(client, monkeypatch):
    """The compliant path — the half a failure-only guard would not catch."""
    _install_primitives(monkeypatch, llm_call=lambda *a, **k: "the real answer")

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 200
    payload = r.json()
    assert payload["choices"][0]["message"]["content"] == "the real answer"
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_streaming_backend_failure_emits_a_terminal_error_event(client, monkeypatch):
    """A stream cannot retract its 200, so the event body carries the signal."""
    _install_primitives(
        monkeypatch, llm_call=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("upstream died"))
    )

    r = client.post("/v1/chat/completions", json=_body(stream=True))
    events = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]")
    ]

    assert events, "stream produced no events"
    errors = [e for e in events if "error" in e]
    assert errors, f"streamed failure carried no error object: {events}"
    assert errors[0]["error"]["code"] == 502
    assert errors[0]["choices"][0]["finish_reason"] == "error"

    # The failure must never arrive as assistant content, and the stream must not
    # claim it stopped normally — that pairing is exactly what made an outage
    # indistinguishable from a low-quality generation.
    for event in events:
        for choice in event.get("choices", []):
            assert "upstream died" not in str(choice.get("delta", {}))
            assert choice.get("finish_reason") != "stop"


def test_streaming_success_path_still_streams_content(client, monkeypatch):
    """Compliant streaming path stays intact."""
    _install_primitives(monkeypatch, llm_call=lambda *a, **k: "hi")

    r = client.post("/v1/chat/completions", json=_body(stream=True))
    events = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]")
    ]

    assert not any("error" in e for e in events)
    streamed = "".join(
        c.get("delta", {}).get("content", "")
        for e in events
        for c in e.get("choices", [])
    )
    assert streamed == "hi"
    assert any(
        c.get("finish_reason") == "stop" for e in events for c in e.get("choices", [])
    )
