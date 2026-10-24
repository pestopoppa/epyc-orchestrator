"""Per-request state on a SHARED LLMPrimitives (A1-02 / A2-01 / A1-M2).

`_init_primitives` hands one per-worker instance to every concurrent /chat request,
so counters, cache_prompt, recursion depth, early-stop and budget diagnostics must
not leak between requests.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from src.api.models import ChatRequest
from src.api.routes.chat_pipeline.routing import _init_primitives
from src.llm_primitives import LLMPrimitives


def _shared_primitives(real_call) -> LLMPrimitives:
    prims = LLMPrimitives(mock_mode=False)
    prims._backends = {"frontdoor": object(), "worker_general": object()}
    prims._real_call = real_call
    return prims


def test_concurrent_requests_keep_independent_counters_and_cache_prompt():
    """A (3 calls x 10 tokens) and B (1 call x 7 tokens, starting mid-A) on one instance."""
    a_first_done = threading.Event()
    b_done = threading.Event()
    seen_cache_prompt: dict[str, list] = {"A": [], "B": []}
    holder: dict[str, LLMPrimitives] = {}

    def fake_real_call(prompt, role, n_tokens, stop, **_kw):
        prims = holder["p"]
        who, tokens = prompt.split(":")
        seen_cache_prompt[who].append(prims.cache_prompt)
        prims.total_tokens_generated += int(tokens)
        return "ok"

    prims = _shared_primitives(fake_real_call)
    holder["p"] = prims
    state = SimpleNamespace(_real_primitives=prims, registry=None)

    async def request_a():
        p = _init_primitives(ChatRequest(prompt="a", real_mode=True, cache_prompt=True), state)
        await asyncio.to_thread(p.llm_call, "A:10", role="frontdoor")
        a_first_done.set()
        await asyncio.to_thread(b_done.wait, 5)
        for _ in range(2):
            await asyncio.to_thread(p.llm_call, "A:10", role="frontdoor")
        return p.total_tokens_generated, p.total_calls, len(p.call_log)

    async def request_b():
        await asyncio.to_thread(a_first_done.wait, 5)
        p = _init_primitives(ChatRequest(prompt="b", real_mode=True, cache_prompt=False), state)
        await asyncio.to_thread(p.llm_call, "B:7", role="frontdoor")
        b_done.set()
        return p.total_tokens_generated, p.total_calls, len(p.call_log)

    async def main():
        return await asyncio.gather(request_a(), request_b())

    a_stats, b_stats = asyncio.run(main())
    assert a_stats == (30, 3, 3)
    assert b_stats == (7, 1, 1)
    assert seen_cache_prompt == {"A": [True, True, True], "B": [False]}


def test_concurrent_top_level_calls_neither_recurse_nor_reroute():
    """7 in-flight top-level calls: no RecursionError, no depth override, depth back to 0."""
    release = threading.Event()
    entered = threading.Semaphore(0)
    roles: list[str] = []

    def blocking_real_call(prompt, role, n_tokens, stop, **_kw):
        roles.append(role)
        entered.release()
        release.wait(5)
        return "ok"

    prims = _shared_primitives(blocking_real_call)
    prims._depth_model_overrides_enabled = True
    prims._depth_role_overrides = {1: "worker_general"}
    errors: list[BaseException] = []

    def call():
        try:
            prims.llm_call("q", role="frontdoor")
        except BaseException as exc:  # noqa: BLE001 - the assertion is that nothing raises
            errors.append(exc)
            entered.release()

    threads = [threading.Thread(target=call) for _ in range(7)]
    for t in threads:
        t.start()
    for _ in threads:
        assert entered.acquire(timeout=5)
    release.set()
    for t in threads:
        t.join(5)

    assert errors == []
    assert roles == ["frontdoor"] * 7
    assert prims._recursion_depth == 0


def test_nested_chain_still_hits_the_recursion_limit():
    prims = LLMPrimitives(mock_mode=True)
    original = prims._llm_call_impl

    def nest(*args, **kwargs):
        return prims.llm_call("deeper")

    prims._llm_call_impl = nest
    with pytest.raises(RecursionError, match="Maximum recursion depth"):
        prims.llm_call("start")
    prims._llm_call_impl = original
    assert prims._recursion_depth == 0
    assert prims._max_recursion_depth_reached == prims.config.max_recursion_depth


def test_early_stop_and_budget_diagnostics_are_request_local():
    prims = LLMPrimitives(mock_mode=True)
    a_set = threading.Event()
    b_checked = threading.Event()
    seen: dict[str, object] = {}

    def request_a():
        prims.begin_request()
        with prims.request_context(deadline_s=10_000.0, request_id="A"):
            prims._early_stop_check = lambda text: True
            a_set.set()
            b_checked.wait(5)
            seen["a_stop"] = prims._early_stop_check
            seen["a_diag"] = prims.get_budget_diagnostics()["request_id"]

    def request_b():
        a_set.wait(5)
        prims.begin_request()
        with prims.request_context(request_id="B"):
            seen["b_stop"] = prims._early_stop_check
            seen["b_diag"] = prims.get_budget_diagnostics()["request_id"]
        b_checked.set()

    threads = [threading.Thread(target=request_a), threading.Thread(target=request_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert callable(seen["a_stop"])
    assert seen["b_stop"] is None
    assert seen["a_diag"] == "A"
    assert seen["b_diag"] == "B"
