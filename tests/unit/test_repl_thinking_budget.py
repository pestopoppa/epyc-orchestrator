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
from tests.unit.test_thinking_roles_chat_lane import (
    REGISTRY_CTK, _FakeBackend, _FakeStream, _backend, _chat_response,
    _completion_response, _infer, _primitives, _sse,
)

pytest_plugins = ("tests.unit.test_thinking_roles_chat_lane",)


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
    with budget.request_budget_scope(
        type("Req", (), {"thinking_budget": 0, "max_tokens": None})(),
        type("Route", (), {"reasoning_budget": 200, "generation_max_tokens": 256})(),
    ):
        assert budget.current_budget() is None
        assert budget.current_generation_cap() is None
        assert budget.cap_generation_tokens(8192) == 8192


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


@pytest.mark.parametrize(
    "band,easy,hard,expected",
    [("easy", 256, 8192, 256), ("hard", 256, 8192, 8192),
     ("medium", 256, 8192, None), ("", 256, 8192, None),
     ("easy", None, 8192, None), ("hard", 256, None, None)],
)
def test_router_policy_uses_only_configured_existing_bands(band, easy, hard, expected):
    assert budget.routed_budget_for_band(band, easy=easy, hard=hard) == expected


def test_router_policy_is_empty_by_default():
    from src.config.models import ChatPipelineConfig

    config = ChatPipelineConfig()
    assert config.reasoning_budget_easy is None
    assert config.reasoning_budget_hard is None
    assert config.generation_max_tokens_easy is None
    assert config.generation_max_tokens_hard is None


def test_stream_assessor_skips_disabled_or_fully_overridden_policy(monkeypatch):
    from types import SimpleNamespace

    from src.api.routes.chat_pipeline import routing_decision
    from src.config.models import ChatPipelineConfig

    def must_not_assess(*args, **kwargs):
        raise AssertionError("difficulty assessor must not run for inactive budget policy")

    monkeypatch.setattr(routing_decision, "assess_difficulty", must_not_assess)
    configured = ChatPipelineConfig(
        reasoning_budget_easy=256,
        reasoning_budget_hard=2048,
        generation_max_tokens_easy=512,
        generation_max_tokens_hard=4096,
    )
    # Feature-off default avoids adding classification to legacy stream requests.
    assert budget.assess_request_budgets(
        SimpleNamespace(prompt="p", thinking_budget=0, max_tokens=None),
        "frontdoor", "t0", configured,
    ) == (None, None)

    turn_on()
    # Positive caller reasoning budget and explicit max_tokens override all policy outputs.
    assert budget.assess_request_budgets(
        SimpleNamespace(prompt="p", thinking_budget=1200, max_tokens=6000),
        "frontdoor", "t1", configured,
    ) == (None, None)


def test_stream_assessor_runs_once_when_configured_policy_applies(monkeypatch):
    from types import SimpleNamespace

    from src.api.routes.chat_pipeline import routing_decision
    from src.config.models import ChatPipelineConfig

    turn_on()
    calls = []
    monkeypatch.setattr(
        routing_decision,
        "assess_difficulty",
        lambda prompt, role, task_id: (calls.append((prompt, role, task_id)) or (0.2, "easy")),
    )
    result = budget.assess_request_budgets(
        SimpleNamespace(prompt="p", thinking_budget=0, max_tokens=None),
        "frontdoor", "t2",
        ChatPipelineConfig(reasoning_budget_easy=256, generation_max_tokens_easy=512),
    )
    assert result == (256, 512)
    assert calls == [("p", "frontdoor", "t2")]


def test_router_generation_policy_is_bounded_by_existing_admission_reserve():
    from src.api.routes.chat_pipeline.routing_decision import DEFAULT_ROUTING_DECODE_RESERVE_TOKENS

    assert budget.routed_generation_cap_for_band(
        "hard", easy=256, hard=32000
    ) == DEFAULT_ROUTING_DECODE_RESERVE_TOKENS
    assert budget.routed_generation_cap_for_band(
        "easy", easy=256, hard=32000
    ) == 256
    assert budget.routed_generation_cap_for_band(
        "medium", easy=256, hard=32000
    ) is None


def test_router_reasoning_budget_stays_within_admitted_completion_cap():
    from types import SimpleNamespace

    from src.api.routes.chat_pipeline.routing_decision import DEFAULT_ROUTING_DECODE_RESERVE_TOKENS
    from src.config.models import ChatPipelineConfig

    policy = ChatPipelineConfig(reasoning_budget_hard=32000)
    assert budget.routing_budget_outputs(
        SimpleNamespace(max_tokens=None), "hard", policy
    ) == (DEFAULT_ROUTING_DECODE_RESERVE_TOKENS, None)
    assert budget.routing_budget_outputs(
        SimpleNamespace(max_tokens=512), "hard", policy
    ) == (512, None)


def test_generation_cap_preserves_prefill_and_explicit_request_max():
    from types import SimpleNamespace

    turn_on()
    routing = SimpleNamespace(generation_max_tokens=512)
    assert budget.effective_generation_cap(SimpleNamespace(max_tokens=None), routing) == 512
    assert budget.effective_generation_cap(SimpleNamespace(max_tokens=9000), routing) is None
    with budget.request_budget_scope(SimpleNamespace(max_tokens=None), routing):
        assert budget.cap_generation_tokens(8192) == 512
        assert budget.cap_generation_tokens(-1) == 512
        assert budget.cap_generation_tokens(0) == 0
    assert budget.current_generation_cap() is None
    with pytest.raises(RuntimeError):
        with budget.request_budget_scope(SimpleNamespace(max_tokens=None), routing):
            raise RuntimeError("synthetic request failure")
    assert budget.current_budget() is None
    assert budget.current_generation_cap() is None


def test_llm_primitive_applies_routed_generation_cap_to_backend_request():
    turn_on()
    backend = _FakeBackend(None)
    primitives = _primitives(backend)
    from types import SimpleNamespace

    request = SimpleNamespace(thinking_budget=0, max_tokens=None)
    route = SimpleNamespace(reasoning_budget=321, generation_max_tokens=1200)
    with budget.request_budget_scope(request, route):
        primitives.llm_call("controlled", role="architect_critic", n_tokens=4096, skip_suffix=True)
    assert backend.requests[0].n_tokens == 1200
    assert backend.requests[0].thinking_budget == 321


def test_generation_cap_is_applied_before_cache_identity_and_backend_dispatch(tmp_path):
    from src.llm_cache import ContentAddressableCache

    turn_on()
    set_features(get_features(override={"repl_thinking_budget": True, "content_cache": True}))

    class CountingBackend(_FakeBackend):
        def infer(self, role_config, request):
            result = super().infer(role_config, request)
            result.output = f"answer-cap-{request.n_tokens}"
            result.prompt_tokens = 10
            result.output_tokens_reported = 2
            result.output_tokens_source = "llama_server:/v1/chat/completions.usage.completion_tokens"
            return result

    backend = CountingBackend(None)
    primitives = _primitives(backend)
    primitives._content_cache = ContentAddressableCache(tmp_path)
    answers = []
    request = type("Req", (), {"thinking_budget": 0, "max_tokens": None})()

    for cap in (1024, 512, 1024):
        route = type("Route", (), {"reasoning_budget": None,
                                    "generation_max_tokens": cap})()
        with budget.request_budget_scope(request, route):
            answers.append(
                primitives._real_call("same prompt", "architect_critic", n_tokens=4096)
            )

    assert answers == ["answer-cap-1024", "answer-cap-512", "answer-cap-1024"]
    assert [item.n_tokens for item in backend.requests] == [1024, 512]
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_feature_on_without_chat_budget_scope_preserves_nonchat_cache_and_token_cap(tmp_path):
    from src.config.models import ChatPipelineConfig
    from src.llm_cache import ContentAddressableCache

    turn_on()
    set_features(get_features(override={"repl_thinking_budget": True, "content_cache": True}))
    # A configured policy alone is inert; only chat routing installs its request scope.
    policy = ChatPipelineConfig(generation_max_tokens_easy=512)
    assert budget.routed_generation_cap_for_band(
        "easy", easy=policy.generation_max_tokens_easy, hard=None
    ) == 512

    class CountingBackend(_FakeBackend):
        def infer(self, role_config, request):
            result = super().infer(role_config, request)
            result.output = "legacy-uncapped"
            result.prompt_tokens = 10
            result.output_tokens_reported = 2
            result.output_tokens_source = "llama_server:/v1/chat/completions.usage.completion_tokens"
            return result

    backend = CountingBackend(None)
    primitives = _primitives(backend)
    primitives._content_cache = ContentAddressableCache(tmp_path)
    assert primitives._real_call("nonchat", "architect_critic", n_tokens=4096) == "legacy-uncapped"
    assert primitives._real_call("nonchat", "architect_critic", n_tokens=4096) == "legacy-uncapped"
    assert [item.n_tokens for item in backend.requests] == [4096]
    assert budget.current_generation_cap() is None


def test_explicit_request_reasoning_budget_precedes_router_budget():
    from types import SimpleNamespace

    assert budget.effective_request_budget(
        SimpleNamespace(thinking_budget=701), SimpleNamespace(reasoning_budget=9000)
    ) == 701
    # Existing zero means inherit, so the router value remains eligible.
    assert budget.effective_request_budget(
        SimpleNamespace(thinking_budget=0), SimpleNamespace(reasoning_budget=9000)
    ) == 9000
    assert budget.effective_request_budget(
        SimpleNamespace(thinking_budget=0), SimpleNamespace(reasoning_budget=None)
    ) is None


def test_direct_stage_scopes_router_budget_for_all_nested_calls(monkeypatch):
    turn_on()
    from types import SimpleNamespace
    from src.api.routes.chat_pipeline import direct_stage

    seen = []
    expected = SimpleNamespace(answer="controlled")
    monkeypatch.setattr(
        direct_stage,
        "_execute_direct_with_budget",
        lambda *args: (
            seen.append((budget.current_budget(), budget.current_generation_cap())), expected
        )[1],
    )
    result = direct_stage._execute_direct(
        SimpleNamespace(thinking_budget=0),
        SimpleNamespace(reasoning_budget=777, generation_max_tokens=512),
        SimpleNamespace(), SimpleNamespace(), time.perf_counter(), "frontdoor",
    )
    assert result is expected
    assert seen == [(777, 512)]
    assert budget.current_budget() is None
    assert budget.current_generation_cap() is None


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


def test_concurrent_reasoning_and_generation_caps_are_request_local():
    turn_on()

    async def run():
        ready = asyncio.Event()
        entered = []

        async def worker(reasoning, cap):
            request = type("Req", (), {"thinking_budget": 0, "max_tokens": None})()
            route = type("Route", (), {"reasoning_budget": reasoning,
                                        "generation_max_tokens": cap})()
            with budget.request_budget_scope(request, route):
                entered.append((budget.current_budget(), budget.current_generation_cap()))
                if len(entered) == 3:
                    ready.set()
                await ready.wait()
                await asyncio.sleep(0)
                observed = await asyncio.to_thread(
                    lambda: (budget.current_budget(), budget.current_generation_cap())
                )
            return observed, budget.current_budget(), budget.current_generation_cap()

        return await asyncio.gather(worker(111, 1000), worker(222, 2000), worker(None, None))

    assert asyncio.run(run()) == [
        ((111, 1000), None, None),
        ((222, 2000), None, None),
        ((None, None), None, None),
    ]


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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "caller_budget,router_budget,request_max_tokens,expected_budget,expected_max_tokens",
    [(0, 777, None, 777, 3200), (321, 777, None, 321, 3200),
     (0, None, None, None, 3200), (321, 777, 777, 321, 777)],
)
def test_router_budget_reaches_actual_chat_request_payload(
    role_config, stream, caller_budget, router_budget, request_max_tokens,
    expected_budget, expected_max_tokens,
):
    """Controlled request proves router selection reaches llama-server's wire payload."""
    turn_on()
    from types import SimpleNamespace

    request_values = SimpleNamespace(
        thinking_budget=caller_budget, max_tokens=request_max_tokens
    )
    from src.config.models import ChatPipelineConfig

    reasoning_budget, generation_max_tokens = budget.routing_budget_outputs(
        request_values,
        "hard",
        ChatPipelineConfig(
            reasoning_budget_hard=router_budget,
            generation_max_tokens_hard=3200,
        ),
    )
    route_output = SimpleNamespace(
        reasoning_budget=reasoning_budget,
        generation_max_tokens=generation_max_tokens,
    )
    with budget.request_budget_scope(request_values, route_output):
        request = InferenceRequest(
            role=role_config.name,
            prompt="controlled request",
            n_tokens=budget.cap_generation_tokens(request_max_tokens or 8192),
            thinking_budget=budget.current_budget(),
        )
        backend = _backend(True)
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
                result = backend._infer_stream_text_chat_completions(
                    role_config, request, lambda _: None, time.perf_counter()
                )
            else:
                result = backend._infer_chat_completions(role_config, request, time.perf_counter())

    assert result.success
    assert captured["max_tokens"] == expected_max_tokens
    if expected_budget is None:
        assert "thinking_budget_tokens" not in captured
    else:
        assert captured["thinking_budget_tokens"] == expected_budget
        assert "Stop deliberating" in captured["reasoning_budget_message"]


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
