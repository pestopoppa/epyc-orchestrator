"""UFH14-B1f transport proof; no model/server/network invocation."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import time
from unittest.mock import patch

import pytest

from src.backends import repl_thinking_budget as budget
from src.features import get_features, reset_features, set_features
from src.model_server import InferenceRequest
from test_thinking_roles_chat_lane import (
    REGISTRY_CTK, _FakeBackend, _FakeStream, _backend, _chat_response,
    _completion_response, _infer, _primitives, _sse, role_config,
)


@pytest.fixture(autouse=True)
def reset_budget_features():
    reset_features()
    set_features(get_features(override={"repl_thinking_budget": False}))
    yield
    reset_features()


def turn_on():
    set_features(get_features(override={"repl_thinking_budget": True}))


def test_flag_defaults_off_and_has_own_catalog_entry():
    from src.features import Features, _FEATURE_REGISTRY
    assert Features().repl_thinking_budget is False
    spec, = [item for item in _FEATURE_REGISTRY if item.name == "repl_thinking_budget"]
    assert spec.default_test is spec.default_prod is False
    with budget.budget_scope(100):
        assert budget.current_budget() is None


def test_nested_unset_and_error_restore_previous_budget():
    turn_on()
    assert budget.current_budget() is None
    with budget.budget_scope(100):
        assert budget.current_budget() == 100
        with budget.budget_scope(0):
            assert budget.current_budget() is None
        with pytest.raises(RuntimeError):
            with budget.budget_scope(200):
                assert budget.current_budget() == 200
                raise RuntimeError("synthetic turn failed")
        assert budget.current_budget() == 100
    assert budget.current_budget() is None


def test_concurrent_budgets_and_unset_cross_real_thread_hops_without_leak():
    turn_on()
    async def run():
        ready = asyncio.Event()
        entered = []
        async def worker(value):
            with budget.budget_scope(value):
                entered.append(value)
                if len(entered) == 3:
                    ready.set()
                await ready.wait()
                await asyncio.sleep(0)
                observed = await asyncio.to_thread(budget.current_budget)
            return observed, budget.current_budget()
        return await asyncio.gather(worker(100), worker(200), worker(None))
    assert asyncio.run(run()) == [(100, None), (200, None), (None, None)]
    assert budget.current_budget() is None


def test_cancelled_task_resets_its_budget_before_outer_finally():
    turn_on()
    observed = []
    async def run():
        ready = asyncio.Event()
        async def worker():
            try:
                with budget.budget_scope(100):
                    ready.set()
                    await asyncio.Event().wait()
            finally:
                observed.append(budget.current_budget())
        task = asyncio.create_task(worker())
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert observed == [None]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("flag,tokens", [(False, 100), (True, None), (True, 0), (True, 100)])
def test_both_chat_body_builders_deliver_only_enabled_caller_budget(role_config, stream, flag, tokens):
    set_features(get_features(override={"repl_thinking_budget": flag}))
    backend = _backend(True)
    request = InferenceRequest(role=role_config.name, prompt="p", n_tokens=32, thinking_budget=tokens)
    captured = {}
    def post(path, json=None, timeout=None):
        captured.update(json)
        return _chat_response("done", None)
    def streaming(method, path, json=None, timeout=None):
        captured.update(json)
        return _FakeStream([_sse({"content": "done"}, "stop"), "data: [DONE]"])
    with patch.object(backend.client, "post", side_effect=post), \
         patch.object(backend.client, "stream", side_effect=streaming), \
         patch("src.registry.registry_loader.chat_template_kwargs_for_role", return_value=REGISTRY_CTK):
        if stream:
            result = backend._infer_stream_text_chat_completions(role_config, request, lambda _: None, time.perf_counter())
        else:
            result = backend._infer_chat_completions(role_config, request, time.perf_counter())
    assert result.success
    assert captured["chat_template_kwargs"] == REGISTRY_CTK
    if flag and tokens:
        assert captured["thinking_budget_tokens"] == tokens
        assert "Stop deliberating" in captured["reasoning_budget_message"]
    else:
        assert "thinking_budget_tokens" not in captured
        assert "reasoning_budget_message" not in captured


def test_thinking_off_and_non_chat_body_remain_unchanged(role_config):
    turn_on()
    request = InferenceRequest(role=role_config.name, prompt="p", thinking_budget=100)
    assert budget.request_fields(request, {"enable_thinking": False}) == {}
    backend = _backend(False)
    with patch("src.backends.llama_server._thinking_chat_lane", return_value=False):
        captured, result = _infer(backend, role_config, request, _completion_response("done"))
    assert result.success and captured["path"] == "/completion"
    assert "thinking_budget_tokens" not in captured["payload"]
    assert "reasoning_budget_message" not in captured["payload"]


def test_primitive_copies_budget_and_cache_replace_preserves_it():
    turn_on()
    backend = _FakeBackend(None)
    primitives = _primitives(backend)
    with budget.budget_scope(321):
        primitives._call_caching_backend(backend, "p", "architect_critic", n_tokens=8)
    primitives._call_caching_backend(backend, "q", "architect_critic", n_tokens=8)
    assert backend.requests[0].thinking_budget == 321
    assert replace(backend.requests[0], prompt="replacement").thinking_budget == 321
    assert backend.requests[1].thinking_budget is None


def test_actual_graph_worker_path_carries_concurrent_budgets_and_resets(monkeypatch):
    """Exercise production to_thread dispatch, disabling its pytest inline shortcut."""
    import threading
    from types import SimpleNamespace
    from src.graph.helpers import _call_llm_capturing_meta

    turn_on()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    parent_thread = threading.get_ident()
    barrier = threading.Barrier(3, timeout=5)
    observed_threads = []

    def call():
        observed_threads.append(threading.get_ident())
        barrier.wait()
        return budget.current_budget()

    async def turn(value):
        with budget.budget_scope(value):
            actual, meta = await _call_llm_capturing_meta(call, SimpleNamespace())
        return actual, meta, budget.current_budget()

    async def run():
        return await asyncio.gather(turn(111), turn(222), turn(None))

    assert asyncio.run(run()) == [(111, {}, None), (222, {}, None), (None, {}, None)]
    assert len(observed_threads) == 3
    assert all(ident != parent_thread for ident in observed_threads)
    assert budget.current_budget() is None


def test_worker_scope_resets_after_backend_failure_in_reused_executor():
    """Explicit worker-local scope also works with executors that don't copy context."""
    from concurrent.futures import ThreadPoolExecutor
    turn_on()
    def failed_turn():
        with budget.budget_scope(123):
            assert budget.current_budget() == 123
            raise RuntimeError("backend failed")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="backend failed"):
            pool.submit(failed_turn).result(timeout=5)
        assert pool.submit(budget.current_budget).result(timeout=5) is None
    assert budget.current_budget() is None


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_actual_stream_routes_reset_before_yield_error_or_generator_close(monkeypatch, legacy, fail):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import src.api.routes.chat as chat
    from src.api.routes.chat_pipeline import stream_adapter
    from src.api.models import ChatRequest

    turn_on()
    observed = []
    def call(*args, **kwargs):
        observed.append(budget.current_budget())
        if fail:
            raise RuntimeError("synthetic backend error")
        return "print('answer')"
    primitives = SimpleNamespace(llm_call=call)
    state = MagicMock(progress_logger=None, hybrid_router=None)
    request = ChatRequest(prompt="test", real_mode=True, mock_mode=False, thinking_budget=151,
                          max_turns=1, server_urls={"frontdoor": "http://invalid"})
    module = chat if legacy else stream_adapter
    monkeypatch.setattr(module, "REPLEnvironment", MagicMock())
    monkeypatch.setattr(module, "build_root_lm_prompt", lambda **_: "root")
    monkeypatch.setattr(module, "build_corpus_context", lambda **_: "")
    if legacy:
        monkeypatch.setattr(chat, "ensure_memrl_initialized", lambda _: None)
        monkeypatch.setattr(chat, "LLMPrimitives", lambda **_: primitives)
        monkeypatch.setattr(chat, "create_sse_response", lambda gen: gen)

    async def run():
        if legacy:
            generator = await chat.chat_stream(request, state)
        else:
            routing = SimpleNamespace(task_id="budget-test", task_ir={})
            generator = stream_adapter._stream_repl(request, routing, primitives, state, time.perf_counter())
        try:
            async for event in generator:
                # Context must never escape into an SSE consumer, even on error.
                assert budget.current_budget() is None
                if observed:
                    assert event["event"] == ("error" if fail else "token")
                    break
        finally:
            await generator.aclose()
        assert budget.current_budget() is None
    asyncio.run(run())
    assert observed == [151]


@pytest.mark.parametrize("cancel", [False, True])
def test_actual_nonstream_route_carries_budget_to_graph_worker_and_resets(monkeypatch, cancel):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from src.api.models import ChatRequest
    from src.api.routes.chat_pipeline import repl_executor
    from src.graph.helpers import _call_llm_capturing_meta

    turn_on()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(repl_executor, "REPLEnvironment", MagicMock())
    monkeypatch.setattr(repl_executor, "_should_use_two_stage", lambda **_: False)
    observed = []
    failure = asyncio.CancelledError if cancel else RuntimeError
    def call():
        observed.append(budget.current_budget())
        raise failure("synthetic graph backend failure")
    async def run_task(*args, **kwargs):
        return await _call_llm_capturing_meta(call, SimpleNamespace())
    monkeypatch.setattr(repl_executor, "run_task", run_task)
    request = ChatRequest(prompt="test", thinking_budget=171, real_mode=True)
    routing = SimpleNamespace(task_id="budget-nonstream", routing_strategy="rules",
                              formalization_applied=False, skill_context="", document_result=None,
                              task_ir={}, tool_required=False, tool_hint=None, difficulty_band=None)
    state = MagicMock(progress_logger=None, hybrid_router=None, session_store=None)
    primitives = SimpleNamespace()
    async def run():
        with pytest.raises(failure):
            await repl_executor._execute_repl_body(request, routing, primitives, state,
                                                   time.perf_counter(), "frontdoor", None)
        assert budget.current_budget() is None
    asyncio.run(run())
    assert observed == [171]


def test_real_content_cache_separates_budgets_and_reuses_matching_budget(tmp_path):
    from src.llm_cache import ContentAddressableCache
    turn_on()
    set_features(get_features(override={"repl_thinking_budget": True, "content_cache": True}))
    class CountingBackend(_FakeBackend):
        def infer(self, role_config, request):
            result = super().infer(role_config, request)
            result.output = f"answer-budget-{request.thinking_budget}"
            result.prompt_tokens = 10
            result.output_tokens_reported = 2
            result.output_tokens_source = "llama_server:/v1/chat/completions.usage.completion_tokens"
            return result
    backend = CountingBackend(None)
    primitives = _primitives(backend)
    primitives._content_cache = ContentAddressableCache(tmp_path)
    answers = []
    for tokens in [100, 200, 100]:
        with budget.budget_scope(tokens):
            answers.append(primitives._real_call("same prompt", "architect_critic", n_tokens=8))
    assert answers == ["answer-budget-100", "answer-budget-200", "answer-budget-100"]
    assert [request.thinking_budget for request in backend.requests] == [100, 200]
    assert (primitives.get_last_inference_meta() or {})["transport"] == "content_cache"
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.parametrize("flag,tokens", [(False, 100), (True, None), (True, 0)])
@pytest.mark.parametrize("sampling", [{}, {"temperature": 0.0, "seed": 7, "top_p": 0.8, "top_k": 64}])
def test_off_unset_zero_keep_legacy_sampling_and_content_keys(tmp_path, flag, tokens, sampling):
    from src.llm_cache import ContentAddressableCache
    from src.llm_primitives.inference import _sampling_cache_key
    set_features(get_features(override={"repl_thinking_budget": flag, "content_cache": True}))
    backend = _FakeBackend(None)
    primitives = _primitives(backend)
    primitives._content_cache = ContentAddressableCache(tmp_path)
    with patch.object(primitives._content_cache, "get_entry", wraps=primitives._content_cache.get_entry) as lookup:
        with budget.budget_scope(tokens):
            primitives._real_call("legacy prompt", "architect_critic", n_tokens=8, **sampling)
        expected_sampling = '{"seed":7,"temperature":0.0,"top_k":64,"top_p":0.8}' if sampling else ""
        assert _sampling_cache_key(**sampling, thinking_budget=budget.current_budget()) == expected_sampling
        expected = ContentAddressableCache.make_key("legacy prompt", "architect_critic", 8,
                    model_hash=getattr(primitives, "_model_hash", ""), sampling_key=expected_sampling)
        lookup.assert_called_once_with(expected)
    assert _sampling_cache_key(thinking_budget=None) == ""
    assert _sampling_cache_key(thinking_budget=0) == ""
