"""Model calls on request paths must not run on the uvicorn event loop (A1-01).

A blocking llm_call inline in an async handler freezes every other request on that
worker (health checks, SSE frames, disconnect watchers) for the whole generation.
Each case blocks the fake model call for up to 2 s and checks that the loop noticed
the call start promptly, i.e. it kept running while the call was in flight.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx

from src.api.models import ChatRequest


class _BlockingCall:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.started = threading.Event()
        self.release = threading.Event()
        self.started_at = 0.0

    def __call__(self, *_args, **_kwargs) -> str:
        self.started_at = time.perf_counter()
        self.started.set()
        self.release.wait(2)
        return self.answer


async def _loop_lag_during(call: _BlockingCall, coro) -> float:
    """Seconds between the call starting and the event loop observing it."""
    task = asyncio.ensure_future(coro)
    deadline = time.perf_counter() + 5
    while not call.started.is_set() and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    lag = time.perf_counter() - call.started_at
    call.release.set()
    await task
    return lag


def test_v1_non_streaming_direct_call_runs_off_the_event_loop(monkeypatch):
    from src.api import app
    from src.api.state import get_state, reset_state
    from src.features import reset_features
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    reset_features()
    reset_state()
    get_state().registry = MagicMock()
    call = _BlockingCall("hello")
    primitives = MagicMock()
    primitives.llm_call.side_effect = call
    primitives.total_tokens_generated = 1
    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "frontdoor",
                    "messages": [{"role": "user", "content": "hi"}],
                    "x_disable_repl": True,
                },
            )
            assert resp.status_code == 200, resp.text

    try:
        lag = asyncio.run(_loop_lag_during(call, request()))
    finally:
        reset_features()
    assert lag < 0.5


def test_cheap_first_call_runs_off_the_event_loop():
    from src.api.routes.chat import _try_cheap_first

    call = _BlockingCall("<answer>" + "a sufficiently long cheap answer " * 3 + "</answer>")
    primitives = MagicMock()
    primitives.llm_call.side_effect = call
    primitives.total_tokens_generated = 1
    routing = SimpleNamespace(
        task_id="t", factual_risk_band="low", routing_strategy="rules", skill_ids=[]
    )
    state = SimpleNamespace(progress_logger=None, hybrid_router=None, registry=None)

    coro = _try_cheap_first(
        ChatRequest(prompt="write a haiku about the sea", real_mode=True),
        routing,
        primitives,
        state,
        time.perf_counter(),
        "frontdoor",
        "direct",
    )
    lag = asyncio.run(_loop_lag_during(call, coro))
    assert lag < 0.5


def test_chat_routing_runs_off_the_event_loop(monkeypatch):
    """_route_request does embedder HTTP + FAISS on every real /chat."""
    from src.api import app
    from src.api.routes import chat
    from src.api.state import reset_state

    reset_state()
    call = _BlockingCall(None)
    real_route = chat._route_request

    def blocking_route(request, state):
        call()
        return real_route(request, state)

    monkeypatch.setattr(chat, "_route_request", blocking_route)

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.post("/chat", json={"prompt": "hi", "mock_mode": True})
            assert resp.status_code == 200, resp.text

    assert asyncio.run(_loop_lag_during(call, request())) < 0.5


def test_to_thread_keeping_context_keeps_vars_set_in_the_thread():
    from src.api.routes.chat_utils import to_thread_keeping_context
    from src.runtime import routing_stage_timing

    async def main():
        timing = await to_thread_keeping_context(routing_stage_timing.begin, "t1", "chat")
        return timing, routing_stage_timing.current()

    timing, current = asyncio.run(main())
    assert current is timing
