"""AEI-THINK-HEADROOM: flag ``thinking_budget_headroom`` (default off); no model/server/network."""
from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest

from src.backends import repl_thinking_budget as budget
from src.features import Features, _FEATURE_REGISTRY, get_features, reset_features, set_features
from src.model_server import InferenceRequest
from tests.unit.test_thinking_roles_chat_lane import (
    REGISTRY_CTK, _FakeStream, _backend, _chat_response, _sse,
)

pytest_plugins = ("tests.unit.test_thinking_roles_chat_lane",)


def _flags(headroom: bool) -> None:
    set_features(get_features(override={"repl_thinking_budget": True, "thinking_budget_headroom": headroom}))


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv(budget.ANSWER_FLOOR_ENV, raising=False)
    reset_features()
    yield
    reset_features()


def test_flag_defaults_off_in_both_modes():
    assert Features().thinking_budget_headroom is False
    spec, = [s for s in _FEATURE_REGISTRY if s.name == "thinking_budget_headroom"]
    assert spec.default_test is spec.default_prod is False


def test_flag_is_wired_for_autopilot():
    from scripts.autopilot.config_applicator import ENV_PARAMS, HOT_SWAP_FEATURES, classify_params
    from scripts.autopilot.species.numeric_swarm import SURFACES

    assert "thinking_budget_headroom" in HOT_SWAP_FEATURES
    assert ENV_PARAMS["thinking"]["answer_floor_tokens"] == budget.ANSWER_FLOOR_ENV
    assert classify_params({"thinking.answer_floor_tokens": 128})["env_restart"] == {"thinking.answer_floor_tokens": 128}
    spec, = [s for s in SURFACES["thinking_headroom"] if s.name == "thinking.answer_floor_tokens"]
    assert (spec.low, spec.high, spec.param_type) == (64, 1024, "int")


@pytest.mark.parametrize("on,cap,asked,floor_env,expected", [
    (False, 1000, 900, None, 900),    # off: untouched
    (True, 1000, 900, None, 744),     # clamped to cap - 256
    (True, 1000, 500, None, 500),     # already below the room: untouched
    (True, 1000, 900, "100", 900),    # floor knob honoured (room 900)
    (True, 1000, 950, "100", 900),
    (True, 200, 150, None, 150),      # cap <= floor: no room, left unchanged
])
def test_clamp(monkeypatch, on, cap, asked, floor_env, expected):
    if floor_env:
        monkeypatch.setenv(budget.ANSWER_FLOOR_ENV, floor_env)
    _flags(on)
    payload = {"max_tokens": cap, "thinking_budget_tokens": asked}
    budget.clamp_payload_thinking_budget(payload)
    assert payload["thinking_budget_tokens"] == expected
    assert payload["max_tokens"] == cap


def test_clamp_noop_without_budget_or_cap():
    _flags(True)
    for payload in ({"max_tokens": 500}, {"thinking_budget_tokens": 400}, {}):
        before = dict(payload)
        budget.clamp_payload_thinking_budget(payload)
        assert payload == before


def test_exhausted_predicate():
    _flags(True)
    assert budget.is_thinking_exhausted("", "thought", "length")
    assert budget.is_thinking_exhausted("  ", "thought", "length")
    assert not budget.is_thinking_exhausted("answer", "thought", "length")
    assert not budget.is_thinking_exhausted("", "", "length")
    assert not budget.is_thinking_exhausted("", None, "length")
    assert not budget.is_thinking_exhausted("", "thought", "stop")
    assert not budget.is_thinking_exhausted("", "thought", "length", [{"id": "t"}])
    _flags(False)
    assert not budget.is_thinking_exhausted("", "thought", "length")


def _length_response(content: str, reasoning: str | None) -> Mock:
    response = _chat_response(content, reasoning)
    response.json.return_value["choices"][0]["finish_reason"] = "length"
    # a length stop at exactly max_tokens (below it would be reclassified context_limit)
    response.json.return_value["usage"]["completion_tokens"] = 1000
    return response


def _run(role_config, stream: bool, content: str, reasoning: str | None, n_tokens: int = 1000, tokens: int = 900):
    backend = _backend(True)
    request = InferenceRequest(role=role_config.name, prompt="p", n_tokens=n_tokens, thinking_budget=tokens)
    captured = {}

    def post(path, json=None, timeout=None):
        captured.update(json)
        return _length_response(content, reasoning)

    def streaming(method, path, json=None, timeout=None):
        captured.update(json)
        lines = []
        if reasoning:
            lines.append(_sse({"reasoning_content": reasoning}))
        if content:
            lines.append(_sse({"content": content}))
        lines += [_sse({}, "length"), "data: [DONE]"]
        return _FakeStream(lines)

    with patch.object(backend.client, "post", side_effect=post), \
         patch.object(backend.client, "stream", side_effect=streaming), \
         patch("src.registry.registry_loader.chat_template_kwargs_for_role", return_value=REGISTRY_CTK):
        if stream:
            result = backend._infer_stream_text_chat_completions(role_config, request, lambda _: None, time.time())
        else:
            result = backend._infer_chat_completions(role_config, request, time.time())
    return captured, result


@pytest.mark.parametrize("stream", [False, True])
def test_on_clamps_body_and_stamps_thinking_exhausted(role_config, stream):
    _flags(True)
    captured, result = _run(role_config, stream, "", "long thought")
    assert captured["thinking_budget_tokens"] == 744
    assert result.output == ""
    assert result.failure_reason == "thinking_exhausted"
    assert result.failure_stage == "generation"
    assert result.success is False and result.degraded is True
    assert result.completion_reason == "length"


@pytest.mark.parametrize("stream", [False, True])
def test_on_leaves_answered_length_reply_alone(role_config, stream):
    _flags(True)
    _, result = _run(role_config, stream, "partial answer", "thought")
    assert result.failure_reason == ""
    assert result.success is True


@pytest.mark.parametrize("stream", [False, True])
def test_off_is_behaviour_identical(role_config, stream):
    _flags(False)
    captured, result = _run(role_config, stream, "", "long thought")
    assert captured["thinking_budget_tokens"] == 900  # not clamped
    assert result.failure_reason == ""  # no stamp (reply is fast, so not empty_generation either)
    assert result.completion_reason == "length"
    assert result.success is True and result.degraded is False
