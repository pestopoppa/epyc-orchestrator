"""tool_call_parse_error_retry: server tool_call_parse_error (batch body, stream event) and native
parser loss re-prompt once; a generic 5xx stays infra; a second failure is a no-escalate format error."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

from src.backends import tool_parse_retry as tpr
from src.backends.native_tool_parse import NativeToolParseLoss
from src.metrics import tool_call_parse_retry as metrics

TOOL = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
PARSE_ERR = {"error": {"code": 500, "type": "tool_call_parse_error", "message": "bad xml",
                       "retryable": True, "raw_excerpt": "</parameter></function><parameter=path>"}}
GENERIC_500 = {"error": {"code": 500, "type": "server_error", "message": "slot crashed"}}


def _role_config():
    from tests.unit.test_chat_stream_timings import _role_config as rc

    return rc()


def _err(body: dict) -> httpx.Response:
    return httpx.Response(500, json=body, request=httpx.Request("POST", "http://t/v1/chat/completions"))


def _ok(arguments: str = '{"path": "a.txt"}', raw: str | None = None) -> Mock:
    r = Mock()
    r.status_code = 200
    r.raise_for_status = Mock()
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "read", "arguments": arguments}}]}
    r.json.return_value = {
        "choices": [{"message": msg, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        "timings": {"prompt_ms": 5.0, "predicted_ms": 5.0, "predicted_per_second": 30.0},
    }
    if raw is not None:
        r.json.return_value["__verbose"] = {"content": raw}
    return r


def _lost() -> Mock:
    """Current-server parser loss: arguments emptied though the raw generation had them."""
    raw = '<tool_call>\n{"name":"read","arguments":"{\\"path\\":\\"x\\"}"}\n</tool_call>'
    return _ok(arguments="{}", raw=raw)


def _backend():
    from src.backends.llama_server import LlamaServerBackend, ServerConfig

    return LlamaServerBackend(config=ServerConfig(base_url="http://test:8080", use_chat_completions=True))


def _run(monkeypatch, responses, flag):
    import src.features as feature_module
    from src.model_server import InferenceRequest

    monkeypatch.setattr(feature_module.features(), "tool_call_parse_error_retry", flag)
    backend = _backend()
    messages = [{"role": "user", "content": "read it"}]
    request = InferenceRequest(role="frontdoor", prompt="x", n_tokens=64,
                               chat_payload={"messages": messages, "tools": [TOOL]})
    bodies: list[dict] = []
    queue = list(responses)

    def post(path, json=None, timeout=None):
        bodies.append(json)
        return queue.pop(0)

    with patch.object(backend.client, "post", side_effect=post), patch(
        "src.registry.registry_loader.chat_template_kwargs_for_role", return_value=None
    ):
        try:
            return backend.infer(_role_config(), request), None, bodies, messages
        except Exception as exc:  # noqa: BLE001
            return None, exc, bodies, messages


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset()
    yield
    metrics.reset()


# --- detection helpers ---------------------------------------------------------

def test_detection_keys_on_type_not_status():
    assert tpr.parse_error_from_payload(PARSE_ERR) is not None
    assert tpr.parse_error_from_text(json.dumps(PARSE_ERR)) is not None
    assert tpr.parse_error_from_payload(GENERIC_500) is None
    assert tpr.parse_error_from_text("Internal Server Error") is None
    assert tpr.parse_error_from_text("") is None
    assert tpr.parse_error_from_payload({"error": "tool call parse failed"}) is None  # wording is not enough


# --- batch (non-stream) ---------------------------------------------------------

def test_batch_body_retries_once_then_ok(monkeypatch):
    result, error, bodies, messages = _run(monkeypatch, [_err(PARSE_ERR), _ok()], flag=True)
    assert error is None and len(bodies) == 2
    assert bodies[1]["messages"] == [*messages, {"role": "user", "content": tpr.CORRECTION}]
    assert "raw_excerpt" not in json.dumps(bodies[1])          # excerpt never reaches the prompt
    assert result.success and metrics.total("frontdoor", "parse_error", "retried_ok") == 1
    ev = metrics.snapshot()["recent"][0]
    assert ev["trigger"] == "parse_error" and "<parameter=path>" in ev["detail"]  # telemetry only


def test_batch_double_failure_is_format_error_no_escalation(monkeypatch):
    from src.escalation import ErrorCategory
    from src.graph.error_classifier import classify_error
    from src.graph.state import GraphConfig

    result, error, bodies, _ = _run(monkeypatch, [_err(PARSE_ERR), _err(PARSE_ERR)], flag=True)
    assert len(bodies) == 2                                    # one retry, never a third call
    assert isinstance(error, tpr.ToolCallParseFailed) and isinstance(error, NativeToolParseLoss)
    assert metrics.total("frontdoor", "parse_error", "failed") == 1 and metrics.total(outcome="retried_ok") == 0
    category = classify_error(str(error))
    assert category == ErrorCategory.FORMAT and category in GraphConfig().no_escalate_categories
    assert "raw_excerpt" not in str(error) and "<parameter" not in str(error)


def test_generic_500_stays_infra_not_retried(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_err(GENERIC_500)], flag=True)
    assert error is None and len(bodies) == 1
    assert result.success is False and result.failure_reason == "http_status"
    assert metrics.total() == 0


def test_flag_off_parse_error_body_unchanged(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_err(PARSE_ERR)], flag=False)
    assert error is None and len(bodies) == 1
    assert result.failure_reason == "http_status"              # exactly the legacy infra result
    assert metrics.total() == 0


# --- native parser loss (current servers) ------------------------------------------

def test_native_parser_loss_retries_once(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_lost(), _ok()], flag=True)
    assert error is None and len(bodies) == 2 and result.success
    assert metrics.total("frontdoor", "parse_error", "retried_ok") == 1
    ev = metrics.snapshot()["recent"][0]
    assert ev["trigger"] == "parse_error" and ev["detail"].startswith("native_parser_loss:")


def test_native_parser_loss_double_failure(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_lost(), _lost()], flag=True)
    assert len(bodies) == 2 and isinstance(error, tpr.ToolCallParseFailed)
    assert metrics.total("frontdoor", "parse_error", "failed") == 1


def test_native_parser_loss_flag_off_raises_original(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_lost()], flag=False)
    assert len(bodies) == 1 and type(error) is NativeToolParseLoss
    assert metrics.total() == 0


def test_truncated_opener_not_retried(monkeypatch):
    r = _ok()
    r.json.return_value["choices"][0] = {
        "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}
    r.json.return_value["__verbose"] = {"content": "<tool_call>\n"}
    result, error, bodies, _ = _run(monkeypatch, [r], flag=True)
    assert len(bodies) == 1 and type(error) is NativeToolParseLoss
    assert metrics.total() == 0


# --- stream ---------------------------------------------------------------------------

def _stream_run(monkeypatch, streams, flag):
    import src.features as feature_module
    from src.model_server import InferenceRequest
    from tests.unit.test_stream_degeneration_guard import _FakeStream

    monkeypatch.setattr(feature_module.features(), "tool_call_parse_error_retry", flag)
    backend = _backend()
    request = InferenceRequest(role="frontdoor", prompt="hello", n_tokens=64)
    queue = [_FakeStream(lines) for lines in streams]
    bodies: list[dict] = []

    def fake_stream(method, url, json=None, timeout=None):
        bodies.append(json)
        return queue.pop(0)

    with patch.object(backend.client, "stream", side_effect=fake_stream):
        try:
            return backend.infer_stream_text(_role_config(), request, on_chunk=lambda c: None), None, bodies
        except Exception as exc:  # noqa: BLE001
            return None, exc, bodies


ERR_EVENT = ["data: " + json.dumps(PARSE_ERR)]          # closes with no [DONE]
GOOD = ["data: " + json.dumps({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop", "index": 0}]}),
        "data: [DONE]"]


def test_stream_error_event_without_done_retries_once(monkeypatch):
    result, error, bodies = _stream_run(monkeypatch, [ERR_EVENT, GOOD], flag=True)
    assert error is None and len(bodies) == 2 and result.success and result.output == "hi"
    assert tpr.CORRECTION in json.dumps(bodies[1]) and "raw_excerpt" not in json.dumps(bodies[1])
    assert metrics.total("frontdoor", "parse_error", "retried_ok") == 1


def test_stream_double_failure(monkeypatch):
    result, error, bodies = _stream_run(monkeypatch, [ERR_EVENT, ERR_EVENT], flag=True)
    assert len(bodies) == 2 and isinstance(error, tpr.ToolCallParseFailed)
    assert metrics.total("frontdoor", "parse_error", "failed") == 1


def test_stream_generic_error_event_stays_infra(monkeypatch):
    result, error, bodies = _stream_run(monkeypatch, [["data: " + json.dumps(GENERIC_500)]], flag=True)
    assert error is None and len(bodies) == 1
    assert result.success is False and result.failure_reason == "server_error"
    assert metrics.total() == 0


def test_stream_flag_off_unchanged(monkeypatch):
    result, error, bodies = _stream_run(monkeypatch, [ERR_EVENT], flag=False)
    assert error is None and len(bodies) == 1
    assert result.failure_reason == "server_error" and metrics.total() == 0


def test_no_model_fallback_for_format_failure():
    """The primitives-level fallback re-raises NativeToolParseLoss subclasses untouched."""
    import inspect

    from src.llm_primitives import inference

    assert "isinstance(primary_error, NativeToolParseLoss)" in inspect.getsource(inference)


def test_dashboard_snapshot_shape():
    metrics.record("frontdoor", "parse_error", "retried_ok", "x")
    assert metrics.snapshot()[metrics.RETRIES_TOTAL] == {"frontdoor/parse_error/retried_ok": 1}
    assert 'role="frontdoor",trigger="parse_error",outcome="retried_ok"} 1' in metrics.render_prometheus()
