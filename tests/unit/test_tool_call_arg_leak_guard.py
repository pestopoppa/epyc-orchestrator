"""tool_call_arg_leak_guard: pure validator, flag-off identity, one corrective retry, format failure,
and the both-flags-on case (one retry per call, not two)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

from src.backends import tool_parse_retry as tpr
from src.backends.native_tool_parse import NativeToolParseLoss
from src.backends.tool_arg_guard import (
    CORRECTION,
    LEAK_MARKERS,
    ToolCallArgLeak,
    find_call_leak,
    find_leak,
)
from src.metrics import tool_call_parse_retry as metrics


# --- validator -------------------------------------------------------------

@pytest.mark.parametrize("marker", LEAK_MARKERS)
def test_each_marker_is_found(marker):
    assert find_leak({"path": f"a{marker}b"}) == ("path", marker)


def test_documented_leak_example():
    hit = find_leak({"path": "x</parameter></function><parameter=path>y"})
    assert hit is not None and hit[0] == "path" and hit[1] in LEAK_MARKERS


def test_nested_dicts_and_lists():
    assert find_leak({"a": {"b": ["ok", {"c": "x <tool_call> y"}]}}) == ("a.b[1].c", "<tool_call>")
    assert find_leak({"edits": [{"new": "fine"}, {"new": "</function>"}]}) == ("edits[1].new", "</function>")
    assert find_leak(["<function=x"]) == ("[0]", "<function=")


@pytest.mark.parametrize("args", [
    {},
    {"cmd": "echo a < b && test 1 > 0"},
    {"text": "a < b"},
    {"html": "<div class='x'><p>hello</p></div>"},
    {"code": "if (a<b) { return Foo<Bar>(); }  // <parameter list>"},
    {"xml": "<function>not our marker</function"},
    {"n": 3, "flag": True, "none": None, "nested": {"l": [1, 2.5, "x"]}},
    {"<parameter=": "keys are not scanned"},
])
def test_legitimate_values_pass(args):
    assert find_leak(args) is None


def test_call_level_scan_decodes_json_arguments():
    call = {"function": {"name": "read", "arguments": json.dumps({"path": "x</parameter>"})}}
    assert find_call_leak([call]) == ("read", "path", "</parameter>")
    ok = {"function": {"name": "read", "arguments": json.dumps({"path": "a.txt"})}}
    assert find_call_leak([ok]) is None
    assert find_call_leak([]) is None


# --- backend ---------------------------------------------------------------

LEAKED = {"path": "a.txt</parameter></function><parameter=path>b"}
CLEAN = {"path": "a.txt"}
TOOL = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
PARSE_ERR = {"error": {"code": 500, "type": "tool_call_parse_error", "message": "bad", "retryable": True,
                       "raw_excerpt": "x"}}


def _resp(arguments: dict[str, Any]) -> Mock:
    r = Mock()
    r.status_code = 200
    r.raise_for_status = Mock()
    r.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read", "arguments": json.dumps(arguments)}}]},
                     "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        "timings": {"prompt_ms": 5.0, "predicted_ms": 5.0, "predicted_per_second": 30.0},
    }
    return r


def _err() -> httpx.Response:
    return httpx.Response(500, json=PARSE_ERR, request=httpx.Request("POST", "http://t/v1/chat/completions"))


def _role_config():
    from tests.unit.test_chat_stream_timings import _role_config as rc

    return rc()


def _run(monkeypatch, responses, leak=False, parse=False):
    import src.features as feature_module
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.model_server import InferenceRequest

    feats = feature_module.features()
    monkeypatch.setattr(feats, "tool_call_arg_leak_guard", leak)
    monkeypatch.setattr(feats, "tool_call_parse_error_retry", parse)
    backend = LlamaServerBackend(
        config=ServerConfig(base_url="http://test:8080", use_chat_completions=True)
    )
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


def test_flag_off_no_validation(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_resp(LEAKED)])
    assert error is None and len(bodies) == 1
    assert find_call_leak(result.tool_calls) is not None  # passes through untouched
    assert metrics.total() == 0


def test_flag_on_clean_call_untouched(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_resp(CLEAN)], leak=True)
    assert error is None and len(bodies) == 1 and metrics.total() == 0


def test_flag_on_leak_retries_once_then_succeeds(monkeypatch):
    result, error, bodies, messages = _run(monkeypatch, [_resp(LEAKED), _resp(CLEAN)], leak=True)
    assert error is None and len(bodies) == 2
    assert bodies[0]["messages"] == messages
    assert bodies[1]["messages"] == [*messages, {"role": "user", "content": CORRECTION}]
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == CLEAN
    assert metrics.total("frontdoor", "arg_leak", "retried_ok") == 1
    assert metrics.total(trigger="parse_error") == 0


def test_double_leak_is_a_format_failure_without_escalation(monkeypatch):
    from src.escalation import ErrorCategory
    from src.graph.error_classifier import classify_error
    from src.graph.state import GraphConfig

    result, error, bodies, _ = _run(monkeypatch, [_resp(LEAKED), _resp(LEAKED)], leak=True)
    assert len(bodies) == 2                          # exactly one retry, never a third call
    assert isinstance(error, ToolCallArgLeak) and isinstance(error, NativeToolParseLoss)
    assert metrics.total("frontdoor", "arg_leak", "failed") == 1
    assert metrics.total(outcome="retried_ok") == 0
    category = classify_error(str(error))
    assert category == ErrorCategory.FORMAT and category in GraphConfig().no_escalate_categories


def test_parse_flag_alone_does_not_scan_arguments(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_resp(LEAKED)], parse=True)
    assert error is None and len(bodies) == 1 and metrics.total() == 0


# --- both flags on: ONE retry per call, not two -----------------------------------

def test_both_on_leak_then_clean_is_one_retry(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_resp(LEAKED), _resp(CLEAN)], leak=True, parse=True)
    assert error is None and len(bodies) == 2
    assert metrics.total() == 1 and metrics.total("frontdoor", "arg_leak", "retried_ok") == 1


def test_both_on_parse_error_then_leak_is_failed_not_second_retry(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_err(), _resp(LEAKED)], leak=True, parse=True)
    assert len(bodies) == 2                           # still only one retry
    assert isinstance(error, ToolCallArgLeak)
    assert metrics.total("frontdoor", "arg_leak", "failed") == 1 and metrics.total(outcome="retried_ok") == 0


def test_both_on_parse_error_then_clean(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_err(), _resp(CLEAN)], leak=True, parse=True)
    assert error is None and len(bodies) == 2
    assert bodies[1]["messages"][-1] == {"role": "user", "content": tpr.CORRECTION}
    assert metrics.total("frontdoor", "parse_error", "retried_ok") == 1 and metrics.total() == 1


def test_both_on_clean_call_makes_one_request(monkeypatch):
    result, error, bodies, _ = _run(monkeypatch, [_resp(CLEAN)], leak=True, parse=True)
    assert error is None and len(bodies) == 1 and metrics.total() == 0
