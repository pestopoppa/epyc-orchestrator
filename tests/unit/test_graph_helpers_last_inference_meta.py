"""Unit tests for `src.graph.helpers._best_effort_last_inference_meta` (TD-21.33).

`_execute_turn` (`src/graph/helpers.py`) reads this meta AFTER
`await asyncio.to_thread(llm_call_fn, ...)` -- a genuine context-propagation edge the
per-call-safe `get_last_inference_meta()` getter cannot see across (a ContextVar `.set()`
inside a `to_thread` worker's copied context is invisible to the parent once the `await`
returns). `_best_effort_last_inference_meta` therefore PREFERS the getter (correct
whenever the read happens to share a context with the call -- e.g. the
`_use_inline_calls_in_tests()` branch) and falls back to the plain, possibly-racy
`_last_inference_meta` attribute otherwise -- identical to pre-TD-21.33 behavior in the
fallback case, never worse. No live requests; fake backends only.
"""
from __future__ import annotations

import asyncio
from unittest.mock import Mock

from src.graph.helpers import _best_effort_last_inference_meta
from src.llm_primitives import LLMPrimitives
from src.model_server import InferenceResult


def _result(role: str, completion_reason: str, tokens: int) -> InferenceResult:
    return InferenceResult(
        role=role, output="ignored", tokens_generated=tokens, generation_speed=1.0,
        elapsed_time=0.001, success=True, prompt_eval_ms=0.1, generation_ms=0.1,
        http_overhead_ms=0.0, completion_reason=completion_reason,
    )


def test_same_context_call_uses_the_per_call_safe_getter():
    """The `_use_inline_calls_in_tests()` shape: call and read share one context."""
    prims = LLMPrimitives(mock_mode=False, server_urls={"role_a": "http://localhost:9401"})
    backend = Mock(spec=[])
    backend.infer = Mock(return_value=_result("role_a", "stop", 5))
    prims._backends["role_a"] = backend

    prims._real_call("prompt", "role_a", n_tokens=8)
    meta = _best_effort_last_inference_meta(prims)

    assert meta["completion_reason"] == "stop"
    assert meta["tokens"] == 5


def test_cross_thread_call_falls_back_to_the_plain_attribute_without_a_concurrent_clobberer():
    """Production shape: the call runs inside `asyncio.to_thread`; the read happens in the
    PARENT context after the `await` returns, so the ContextVar the getter reads is empty
    there -- `_best_effort_last_inference_meta` must fall back to the plain attribute and
    still return the correct (uncontended) value, exactly as it did before TD-21.33.
    """
    prims = LLMPrimitives(mock_mode=False, server_urls={"role_a": "http://localhost:9402"})
    backend = Mock(spec=[])
    backend.infer = Mock(return_value=_result("role_a", "length", 9))
    prims._backends["role_a"] = backend

    async def _run():
        await asyncio.to_thread(prims._real_call, "prompt", "role_a", n_tokens=8)
        # Back in the PARENT context: the getter sees nothing from the child thread's copy.
        assert prims.get_last_inference_meta() is None
        return _best_effort_last_inference_meta(prims)

    meta = asyncio.run(_run())
    assert meta["completion_reason"] == "length"
    assert meta["tokens"] == 9


def test_cross_thread_call_with_a_concurrent_clobberer_is_a_KNOWN_unfixed_residual():
    """Documents the residual gap explicitly (this is NOT a claim of full safety): when
    the call crosses a to_thread boundary AND a concurrent request clobbers the shared
    plain attribute before this code reads it back, the fallback reports the OTHER call's
    data -- identical to the pre-TD-21.33 race, not introduced by this helper. A full fix
    needs the meta captured inside the thread and returned alongside the answer (the
    restructuring `src/api/routes/chat.py`'s edit-transaction path already does), which is
    out of this migration's scope.
    """
    prims = LLMPrimitives(
        mock_mode=False,
        server_urls={"role_a": "http://localhost:9403", "role_b": "http://localhost:9404"},
    )
    backend_a, backend_b = Mock(spec=[]), Mock(spec=[])
    backend_a.infer = Mock(return_value=_result("role_a", "length", 1))
    backend_b.infer = Mock(return_value=_result("role_b", "stop", 2))
    prims._backends["role_a"] = backend_a
    prims._backends["role_b"] = backend_b

    a_done = asyncio.Event()
    b_done = asyncio.Event()

    async def a_flow() -> dict:
        # `to_thread` wraps ONLY the write, exactly like `_execute_turn`'s
        # `await asyncio.to_thread(llm_call_fn, ...)` -- the read below runs back in the
        # PARENT context, having crossed the boundary that isolates the ContextVar write.
        await asyncio.to_thread(prims._real_call, "prompt A", "role_a", n_tokens=8)
        a_done.set()
        await b_done.wait()
        return _best_effort_last_inference_meta(prims)

    async def b_flow() -> None:
        await a_done.wait()
        await asyncio.to_thread(prims._real_call, "prompt B", "role_b", n_tokens=8)
        b_done.set()

    async def _run():
        return await asyncio.gather(a_flow(), b_flow())

    meta_a, _ = asyncio.run(_run())
    # Known, pre-existing (not introduced here) limitation: A observes B's data.
    assert meta_a["completion_reason"] == "stop"


def test_falls_back_for_a_double_without_the_new_getter_at_all():
    class _Bare:
        def __init__(self) -> None:
            self._last_inference_meta = {"completion_reason": "eos"}

    assert _best_effort_last_inference_meta(_Bare()) == {"completion_reason": "eos"}


def test_missing_meta_returns_empty_dict():
    class _Bare:
        pass

    assert _best_effort_last_inference_meta(_Bare()) == {}
