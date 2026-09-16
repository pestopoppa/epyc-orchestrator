#!/usr/bin/env python3
"""HS-4 P0.1 / P0.2 — /v1 client-executed tool mode and typed request keys.

Offline only: the route is exercised with a primitives double, the backend with
a fake HTTP client, and the primitives seam with a fake backend. The companion
golden test (``test_openai_compat_default_golden.py``) pins that the DEFAULT
mode is byte-identical to the pre-P0.1 route.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.state import get_state, reset_state
from src.features import reset_features
from src.scheduling.contention_gate import ContentionDenied

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
BASH_TOOL = {
    "type": "function",
    "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}},
}
MODEL_TOOL_CALL = {
    "id": "call_abc",
    "type": "function",
    "function": {"name": "read", "arguments": "{\"path\": \"README.md\"}"},
}


# ── route fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
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


def _install(monkeypatch, *, result=None, side_effect=None) -> MagicMock:
    primitives = MagicMock()
    primitives.total_tokens_generated = 11
    if side_effect is not None:
        primitives.chat_completion_call.side_effect = side_effect
    else:
        primitives.chat_completion_call.return_value = result
    primitives.llm_call.side_effect = AssertionError("client mode must not use llm_call")

    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    return primitives


def _body(**overrides) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "orchestrator",
        "messages": [{"role": "user", "content": "read the readme"}],
        "tools": [READ_TOOL, BASH_TOOL],
        "x_tool_mode": "client",
        # The session guard requires it in client mode (see guard tests below).
        "x_session_id": "ses_default",
    }
    body.update(overrides)
    return body


def _tool_result(content="", tool_calls=None, finish_reason="stop") -> dict[str, Any]:
    return {"content": content, "tool_calls": tool_calls or [], "finish_reason": finish_reason}


def _sse_events(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: "):]))
    assert text.rstrip().endswith("data: [DONE]")
    return events


# ── (a) tool call comes back as message.tool_calls ──────────────────────────


def test_nonstream_tool_call_is_returned_as_message_tool_calls(client, monkeypatch):
    primitives = _install(monkeypatch, result=_tool_result(tool_calls=[MODEL_TOOL_CALL]))

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == [MODEL_TOOL_CALL]

    # Forwarded to the backend path: structured messages + the client's tools.
    call = primitives.chat_completion_call.call_args
    assert call.args[0] == [{"role": "user", "content": "read the readme"}]
    assert call.kwargs["tools"] == [READ_TOOL, BASH_TOOL]
    assert call.kwargs["tool_choice"] is None
    primitives.llm_call.assert_not_called()


def test_stream_tool_call_is_emitted_as_tool_call_deltas(client, monkeypatch):
    second = {
        "id": "call_def",
        "type": "function",
        "function": {"name": "bash", "arguments": {"cmd": "ls"}},  # dict → JSON string
    }
    _install(
        monkeypatch,
        result=_tool_result(content="Let me look.", tool_calls=[MODEL_TOOL_CALL, second]),
    )

    r = client.post("/v1/chat/completions", json=_body(stream=True))

    assert r.status_code == 200
    events = _sse_events(r.text)
    content = "".join(e["choices"][0]["delta"].get("content") or "" for e in events)
    assert content == "Let me look."
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"

    tool_deltas = [
        tc
        for e in events
        for tc in (e["choices"][0]["delta"].get("tool_calls") or [])
    ]
    assert tool_deltas == [
        {"index": 0, **MODEL_TOOL_CALL},
        {
            "index": 1,
            "id": "call_def",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"},
        },
    ]
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert all(e["choices"][0]["finish_reason"] is None for e in events[:-1])


def test_stream_tool_call_only_turn_carries_role_on_first_delta(client, monkeypatch):
    _install(monkeypatch, result=_tool_result(tool_calls=[MODEL_TOOL_CALL]))

    events = _sse_events(client.post("/v1/chat/completions", json=_body(stream=True)).text)

    first = events[0]["choices"][0]["delta"]
    assert first["role"] == "assistant"
    assert first["content"] is None
    assert first["tool_calls"][0]["id"] == "call_abc"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_call_without_id_gets_one(client, monkeypatch):
    anonymous = {"type": "function", "function": {"name": "read", "arguments": "{}"}}
    _install(monkeypatch, result=_tool_result(tool_calls=[anonymous]))

    tc = client.post("/v1/chat/completions", json=_body()).json()["choices"][0]["message"][
        "tool_calls"
    ][0]
    assert tc["id"].startswith("call_") and len(tc["id"]) > len("call_")


# ── (c) a tool result message round-trips with its id ───────────────────────


@pytest.mark.parametrize("stream", [False, True])
def test_tool_result_history_round_trips_to_backend(client, monkeypatch, stream):
    primitives = _install(monkeypatch, result=_tool_result(content="The readme says hi."))
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "read the readme"},
        {"role": "assistant", "content": None, "tool_calls": [MODEL_TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_abc", "name": "read", "content": "hi"},
    ]

    r = client.post("/v1/chat/completions", json=_body(messages=messages, stream=stream))

    assert r.status_code == 200
    forwarded = primitives.chat_completion_call.call_args.args[0]
    assert forwarded == [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "read the readme"},
        {"role": "assistant", "content": None, "tool_calls": [MODEL_TOOL_CALL]},
        {"role": "tool", "content": "hi", "tool_call_id": "call_abc", "name": "read"},
    ]
    if stream:
        events = _sse_events(r.text)
        text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events)
        assert text == "The readme says hi."
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
    else:
        choice = r.json()["choices"][0]
        assert choice["message"]["content"] == "The readme says hi."
        assert choice["message"]["tool_calls"] is None
        assert choice["finish_reason"] == "stop"


def test_multipart_text_is_flattened_and_length_passes_through(client, monkeypatch):
    primitives = _install(monkeypatch, result=_tool_result(content="cut", finish_reason="length"))
    messages = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]

    r = client.post("/v1/chat/completions", json=_body(messages=messages))

    assert primitives.chat_completion_call.call_args.args[0] == [{"role": "user", "content": "a b"}]
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_image_input_is_refused_in_client_mode(client, monkeypatch):
    primitives = _install(monkeypatch, result=_tool_result(content="x"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }
    ]

    r = client.post("/v1/chat/completions", json=_body(messages=messages))

    assert r.status_code == 400
    assert "client" in r.json()["detail"]
    primitives.chat_completion_call.assert_not_called()


# ── tool_choice is honoured ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "choice",
    ["auto", "none", "required", {"type": "function", "function": {"name": "bash"}}],
)
def test_tool_choice_is_forwarded_verbatim(client, monkeypatch, choice):
    primitives = _install(monkeypatch, result=_tool_result(content="ok"))

    r = client.post("/v1/chat/completions", json=_body(tool_choice=choice))

    assert r.status_code == 200, r.text
    assert primitives.chat_completion_call.call_args.kwargs["tool_choice"] == choice


@pytest.mark.parametrize(
    "overrides",
    [
        {"tool_choice": "sometimes"},
        {"tool_choice": {"type": "function", "function": {"name": "not_declared"}}},
        {"tool_choice": {"type": "function"}},
        {"tool_choice": "required", "tools": None},
    ],
)
def test_invalid_tool_choice_is_422_in_client_mode(client, monkeypatch, overrides):
    primitives = _install(monkeypatch, result=_tool_result(content="ok"))

    r = client.post("/v1/chat/completions", json=_body(**overrides))

    assert r.status_code == 422, r.text
    primitives.chat_completion_call.assert_not_called()


def test_default_mode_keeps_permissive_tool_choice(client, monkeypatch):
    """The strict check is client-mode only; the REPL bridge is unchanged."""
    primitives = MagicMock()
    primitives.llm_call.return_value = "fine"
    primitives.total_tokens_generated = 1
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    body = _body(tool_choice="sometimes", x_disable_repl=True)
    body.pop("x_tool_mode")

    r = client.post("/v1/chat/completions", json=body)

    assert r.status_code == 200
    primitives.chat_completion_call.assert_not_called()


# ── routing still happens in client mode ────────────────────────────────────


@pytest.mark.parametrize(
    ("overrides", "expected_role"),
    [
        ({"model": "orchestrator"}, "frontdoor"),
        ({"model": "orchestrator", "x_orchestrator_role": "coder_escalation"}, "coder_escalation"),
        ({"model": "frontdoor", "x_force_model": "architect_general"}, "architect_general"),
    ],
)
def test_client_mode_uses_the_same_role_resolution(client, monkeypatch, overrides, expected_role):
    primitives = _install(monkeypatch, result=_tool_result(content="ok"))

    r = client.post("/v1/chat/completions", json=_body(x_show_routing=True, **overrides))

    assert r.status_code == 200
    role = primitives.chat_completion_call.call_args.kwargs["role"]
    assert str(getattr(role, "value", role)) == expected_role
    assert str(r.json()["x_orchestrator_metadata"]["role"]) == str(role)


def test_sampling_and_max_tokens_are_forwarded(client, monkeypatch):
    primitives = _install(monkeypatch, result=_tool_result(content="ok"))

    client.post(
        "/v1/chat/completions",
        json=_body(temperature=0.2, seed=7, top_p=0.9, top_k=20, max_tokens=300),
    )

    kwargs = primitives.chat_completion_call.call_args.kwargs
    assert kwargs["n_tokens"] == 300
    assert (kwargs["temperature"], kwargs["seed"], kwargs["top_p"], kwargs["top_k"]) == (
        0.2, 7, 0.9, 20,
    )


# ── failures keep HS-OD-2 semantics ─────────────────────────────────────────


def test_backend_failure_is_502(client, monkeypatch):
    _install(monkeypatch, side_effect=RuntimeError("upstream died"))

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 502
    assert "upstream died" in json.dumps(r.json())


def test_contention_denied_is_503(client, monkeypatch):
    _install(monkeypatch, side_effect=ContentionDenied("region busy"))

    assert client.post("/v1/chat/completions", json=_body()).status_code == 503


def test_stream_backend_failure_is_terminal_error_event(client, monkeypatch):
    _install(monkeypatch, side_effect=RuntimeError("upstream died"))

    events = _sse_events(client.post("/v1/chat/completions", json=_body(stream=True)).text)

    assert len(events) == 1
    assert events[0]["error"]["type"] == "backend_error"
    assert events[0]["choices"][0]["finish_reason"] == "error"


# ── P0.2: typed keys ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"x_tool_mode": "server"},
        {"x_tool_mode": ""},
        {"x_memory": "maybe"},
        {"x_memory": True},
        {"x_session_id": "has space"},
        {"x_session_id": ""},
        {"x_session_id": "x" * 129},
        {"x_user_id": "-leading-dash"},
        {"x_user_id": 42},
    ],
)
def test_bad_request_key_values_are_422(client, monkeypatch, overrides):
    primitives = _install(monkeypatch, result=_tool_result(content="ok"))

    r = client.post("/v1/chat/completions", json=_body(**overrides))

    assert r.status_code == 422, (overrides, r.text)
    primitives.chat_completion_call.assert_not_called()


KEYS = {
    "x_session_id": "ses_01J9ZABC",
    "x_user_id": "daniele@example.org",
    "x_memory": "off",
}


def test_keys_are_echoed_in_nonstream_metadata_and_trace(client, monkeypatch):
    primitives = _install(monkeypatch, result=_tool_result(tool_calls=[MODEL_TOOL_CALL]))

    r = client.post("/v1/chat/completions", json=_body(x_show_routing=True, **KEYS))

    meta = r.json()["x_orchestrator_metadata"]
    assert list(meta["request_keys"]) == ["x_session_id", "x_user_id", "x_memory", "x_tool_mode"]
    assert meta["request_keys"] == {**KEYS, "x_tool_mode": "client"}
    assert "memory_injection" not in meta
    assert meta["native_tool_contract"] == "client_execution"
    assert meta["response_tool_calls"] == "emitted"
    assert meta["tool_calls_emitted"] == ["read"]
    primitives.set_request_trace_keys.assert_called_once_with({**KEYS, "x_tool_mode": "client"})


def test_keys_are_echoed_in_stream_metadata(client, monkeypatch):
    _install(monkeypatch, result=_tool_result(content="no tools needed"))

    events = _sse_events(
        client.post(
            "/v1/chat/completions",
            json=_body(stream=True, x_show_routing=True, **{**KEYS, "x_memory": "on"}),
        ).text
    )

    meta = events[-1]["x_orchestrator_metadata"]
    assert meta["request_keys"] == {**KEYS, "x_memory": "on", "x_tool_mode": "client"}
    assert meta["memory_injection"] == "not_implemented"
    assert meta["response_tool_calls"] == "none"
    assert meta["tool_calls_emitted"] == []


def test_keys_in_default_mode_are_echoed_without_changing_the_repl_contract(client, monkeypatch):
    primitives = MagicMock()
    primitives.llm_call.return_value = "fine"
    primitives.total_tokens_generated = 1
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "frontdoor",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [READ_TOOL],
            "x_disable_repl": True,
            "x_show_routing": True,
            "x_tool_mode": "repl",
            "x_session_id": "ses_1",
        },
    )

    meta = r.json()["x_orchestrator_metadata"]
    assert meta["request_keys"] == {"x_tool_mode": "repl", "x_session_id": "ses_1"}
    assert meta["native_tool_contract"] == "internal_repl_execution"
    assert meta["response_tool_calls"] == "not_emitted"
    primitives.chat_completion_call.assert_not_called()
    primitives.set_request_trace_keys.assert_called_once_with(
        {"x_tool_mode": "repl", "x_session_id": "ses_1"}
    )


def test_absent_keys_leave_trace_untouched(client, monkeypatch):
    primitives = MagicMock()
    primitives.llm_call.return_value = "fine"
    primitives.total_tokens_generated = 1
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)

    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "x_disable_repl": True},
    )

    primitives.set_request_trace_keys.assert_not_called()


# ── backend: payload forwarding and tool_calls parsing (fake HTTP) ──────────


@pytest.fixture
def role_config():
    from src.registry_loader import (
        AccelerationConfig,
        MemoryConfig,
        ModelConfig,
        PerformanceMetrics,
        RoleConfig,
    )

    return RoleConfig(
        name="frontdoor",
        tier="A",
        description="test",
        model=ModelConfig(name="m", path="m.gguf", quant="Q4_K_M", size_gb=1.0),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


def _backend(use_chat_completions: bool):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig

    return LlamaServerBackend(
        config=ServerConfig(base_url="http://test:8080", use_chat_completions=use_chat_completions)
    )


def _http_response(message: dict[str, Any], finish_reason: str) -> Mock:
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        "timings": {"prompt_ms": 5.0, "predicted_ms": 5.0, "predicted_per_second": 30.0},
    }
    return response


@pytest.mark.parametrize("use_chat_completions", [True, False])
def test_backend_forwards_structured_payload_and_parses_tool_calls(role_config, use_chat_completions):
    """A chat payload goes to /v1/chat/completions even on a /completion backend."""
    from src.model_server import InferenceRequest

    backend = _backend(use_chat_completions)
    messages = [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": None, "tool_calls": [MODEL_TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "hi"},
    ]
    request = InferenceRequest(
        role="frontdoor",
        prompt="ignored-for-payload",
        n_tokens=64,
        chat_payload={"messages": messages, "tools": [READ_TOOL], "tool_choice": "required"},
    )
    captured: dict[str, Any] = {}

    def _post(path, json=None, timeout=None):
        captured["path"] = path
        captured.update(json or {})
        return _http_response(
            {"role": "assistant", "content": None, "tool_calls": [MODEL_TOOL_CALL]},
            "tool_calls",
        )

    with patch.object(backend.client, "post", side_effect=_post), patch(
        "src.registry.registry_loader.chat_template_kwargs_for_role", return_value=None
    ):
        # infer_stream_text must fall back to the batch path for a chat payload.
        result = backend.infer_stream_text(role_config, request)

    assert captured["path"] == "/v1/chat/completions"
    assert captured["messages"] == messages
    assert captured["tools"] == [READ_TOOL]
    assert captured["tool_choice"] == "required"
    assert captured["stream"] is False
    assert result.success is True
    assert result.output == ""
    assert result.tool_calls == [MODEL_TOOL_CALL]
    assert result.completion_reason == "tool_calls"


def test_backend_without_payload_ignores_tool_calls_and_keeps_single_turn(role_config):
    from src.model_server import InferenceRequest

    backend = _backend(True)
    request = InferenceRequest(role="frontdoor", prompt="hello", n_tokens=16)
    captured: dict[str, Any] = {}

    def _post(path, json=None, timeout=None):
        captured.update(json or {})
        return _http_response({"content": "hi", "tool_calls": [MODEL_TOOL_CALL]}, "stop")

    with patch.object(backend.client, "post", side_effect=_post), patch(
        "src.registry.registry_loader.chat_template_kwargs_for_role", return_value=None
    ):
        result = backend.infer(role_config, request)

    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in captured and "tool_choice" not in captured
    assert result.tool_calls == []
    assert result.output == "hi"


def test_chat_payload_survives_prefix_cache_slot_routing():
    """dataclasses.replace() in the prefix-cache router keeps the payload."""
    from dataclasses import replace

    from src.model_server import InferenceRequest

    req = InferenceRequest(role="frontdoor", prompt="p", chat_payload={"messages": []})
    assert replace(req, slot_id=3).chat_payload == {"messages": []}


# ── primitives seam: payload binding, batch-only, tool_calls return ─────────


class _FakeBackend:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.requests: list[Any] = []
        self.stream_calls = 0

    def infer(self, role_config, request):
        from src.model_server import InferenceResult

        self.requests.append(request)
        return InferenceResult(
            role="frontdoor",
            output="",
            tokens_generated=5,
            generation_speed=10.0,
            elapsed_time=0.1,
            success=True,
            completion_reason="tool_calls",
            tool_calls=list(self.tool_calls),
        )

    def infer_stream_text(self, role_config, request, on_chunk=None):
        self.stream_calls += 1
        raise AssertionError("client tool mode must not stream from the backend")


def _real_primitives(backend):
    from src.llm_primitives import LLMPrimitives

    primitives = LLMPrimitives(mock_mode=False, server_urls={})
    primitives._backends = {"frontdoor": backend}
    return primitives


def test_primitives_chat_completion_call_binds_payload_and_returns_tool_calls():
    backend = _FakeBackend([MODEL_TOOL_CALL])
    primitives = _real_primitives(backend)
    messages = [{"role": "user", "content": "read it"}]

    out = primitives.chat_completion_call(
        messages, role="frontdoor", tools=[READ_TOOL], tool_choice="auto", n_tokens=64, seed=3
    )

    assert out == {"content": "", "tool_calls": [MODEL_TOOL_CALL], "finish_reason": "tool_calls"}
    assert backend.stream_calls == 0
    sent = backend.requests[0]
    assert sent.chat_payload == {"messages": messages, "tools": [READ_TOOL], "tool_choice": "auto"}
    assert sent.n_tokens == 64 and sent.seed == 3
    # Payload is unbound after the call: a following llm_call is plain text.
    assert primitives.get_request_chat_payload() is None
    assert primitives.total_tokens_generated == 5
    assert primitives.call_log[-1].call_type == "chat_completion"


def test_primitives_chat_completion_call_raises_instead_of_inband_error():
    class _Broken(_FakeBackend):
        def infer(self, role_config, request):
            from src.model_server import InferenceResult

            return InferenceResult(
                role="frontdoor", output="", tokens_generated=0, generation_speed=0.0,
                elapsed_time=0.1, success=False, error_message="HTTP 500",
            )

    primitives = _real_primitives(_Broken([]))

    with pytest.raises(RuntimeError, match="HTTP 500"):
        primitives.chat_completion_call([{"role": "user", "content": "x"}], role="frontdoor")
    assert primitives.get_request_chat_payload() is None


def test_primitives_chat_completion_call_refuses_mock_mode():
    from src.llm_primitives import LLMPrimitives

    with pytest.raises(RuntimeError, match="mock_mode"):
        LLMPrimitives(mock_mode=True).chat_completion_call(
            [{"role": "user", "content": "x"}], role="frontdoor"
        )


def test_trace_keys_setter_drops_none_and_copies():
    from src.llm_primitives import LLMPrimitives

    primitives = LLMPrimitives(mock_mode=True)
    assert primitives.get_request_trace_keys() == {}
    keys = {"x_session_id": "s1", "x_memory": None}
    primitives.set_request_trace_keys(keys)
    assert primitives.get_request_trace_keys() == {"x_session_id": "s1"}
    primitives.get_request_trace_keys()["x_session_id"] = "mutated"
    assert primitives.get_request_trace_keys() == {"x_session_id": "s1"}


def test_trace_keys_reach_inference_tap_metadata(monkeypatch):
    """With the tap active, the typed keys land in the tap section metadata."""
    backend = _FakeBackend([MODEL_TOOL_CALL])
    primitives = _real_primitives(backend)
    primitives.server_urls = {"frontdoor": "http://127.0.0.1:65530"}
    primitives.set_request_trace_keys({"x_session_id": "ses_9", "x_tool_mode": "client"})
    primitives.health_tracker = None
    primitives.admission_controller = None

    captured: dict[str, Any] = {}

    class _Tap:
        def write_response(self, *_a):
            pass

        def write_timings(self, *_a):
            pass

        def set_metadata(self, **kw):
            captured.setdefault("set_metadata", []).append(kw)

    from contextlib import contextmanager

    @contextmanager
    def _tap_section(role, prompt, metadata=None):
        captured["metadata"] = dict(metadata or {})
        yield _Tap()

    import src.inference_tap as tap_mod

    monkeypatch.setattr(tap_mod, "is_active", lambda: True)
    monkeypatch.setattr(tap_mod, "tap_section", _tap_section)
    monkeypatch.setattr(tap_mod, "should_stream_role", lambda _r: True)
    monkeypatch.setattr(
        "src.llm_primitives.inference._per_region_locks_enabled", lambda: False
    )

    @contextmanager
    def _no_lock(*_a, **_kw):
        yield

    monkeypatch.setattr("src.inference_lock.inference_lock", _no_lock)

    out = primitives.chat_completion_call(
        [{"role": "user", "content": "x"}], role="frontdoor", tools=[READ_TOOL]
    )

    assert out["tool_calls"] == [MODEL_TOOL_CALL]
    assert backend.stream_calls == 0, "tap streaming must be skipped for a chat payload"
    assert captured["metadata"]["request_keys"] == {"x_session_id": "ses_9", "x_tool_mode": "client"}
    assert captured["set_metadata"] == [{"client_tool_calls": ["read"]}]


# ── P0.2 session guard (flag v1_client_session_guard) ───────────────────────


def _no_session(**overrides):
    body = _body(**overrides)
    body.pop("x_session_id")
    return body


def _default_mode_primitives(monkeypatch) -> MagicMock:
    primitives = MagicMock()
    primitives.llm_call.return_value = "fine"
    primitives.total_tokens_generated = 1
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    return primitives


def _default_body(**overrides):
    body = {"messages": [{"role": "user", "content": "hi"}], "x_disable_repl": True}
    body.update(overrides)
    return body


@pytest.mark.parametrize("stream", [False, True])
def test_guard_refuses_client_mode_without_session_id(client, monkeypatch, stream):
    primitives = _install(monkeypatch, result=_tool_result(content="ok"))

    r = client.post("/v1/chat/completions", json=_no_session(stream=stream))

    assert r.status_code == 422
    assert "x_session_id is required" in r.json()["detail"]
    assert "x_tool_mode" in r.json()["detail"]
    primitives.chat_completion_call.assert_not_called()


@pytest.mark.parametrize("ua", ["opencode/0.15.2", "ai-sdk/5.0 OpenCode (linux)"])
def test_guard_refuses_opencode_user_agent_without_session_id(client, monkeypatch, ua):
    primitives = _default_mode_primitives(monkeypatch)

    r = client.post("/v1/chat/completions", json=_default_body(), headers={"User-Agent": ua})

    assert r.status_code == 422
    assert "OpenCode user-agent" in r.json()["detail"]
    primitives.llm_call.assert_not_called()


def test_guard_passes_opencode_with_session_id(client, monkeypatch):
    _default_mode_primitives(monkeypatch)

    r = client.post(
        "/v1/chat/completions",
        json=_default_body(x_session_id="ses_1"),
        headers={"User-Agent": "opencode/0.15.2"},
    )

    assert r.status_code == 200


@pytest.mark.parametrize("ua", ["aider/0.86", "OpenAI/Python 1.99", "python-httpx/0.28", ""])
def test_guard_leaves_other_clients_alone(client, monkeypatch, ua):
    _default_mode_primitives(monkeypatch)

    r = client.post("/v1/chat/completions", json=_default_body(), headers={"User-Agent": ua})

    assert r.status_code == 200


def test_guard_can_be_disabled(client, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_V1_CLIENT_SESSION_GUARD", "0")
    reset_features()
    primitives = _install(monkeypatch, result=_tool_result(content="ok"))

    r = client.post(
        "/v1/chat/completions",
        json=_no_session(),
        headers={"User-Agent": "opencode/0.15.2"},
    )

    assert r.status_code == 200
    primitives.chat_completion_call.assert_called_once()


def test_guard_flag_is_on_by_default_in_production():
    from src.features import _FEATURE_REGISTRY

    spec = next(s for s in _FEATURE_REGISTRY if s.name == "v1_client_session_guard")
    assert spec.default_prod is True
    assert spec.env_var == "V1_CLIENT_SESSION_GUARD"
