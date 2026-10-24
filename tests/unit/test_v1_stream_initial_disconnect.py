"""A client disconnect during the first streamed model call cancels that call."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.api.models import OpenAIChatRequest
from src.api.routes import openai_compat
from src.api.state import get_state, reset_state
from src.features import reset_features


@pytest.mark.asyncio
async def test_disconnect_during_first_streamed_generation_sets_backend_cancel(monkeypatch):
    """The route has not returned StreamingResponse while it awaits its first event."""
    started = threading.Event()
    observed_cancel = threading.Event()

    class BlockingPrimitives:
        total_tokens_generated = 0

        def set_request_trace_keys(self, _keys):
            pass

        @contextmanager
        def request_context(self, *, cancel_check, session_id=None):
            previous = getattr(self, "_active_cancel_check", None)
            self._active_cancel_check = cancel_check
            try:
                yield self
            finally:
                self._active_cancel_check = previous

        def get_request_cancel_check(self):
            return getattr(self, "_active_cancel_check", lambda: False)

        def chat_completion_call(self, *_args, **_kwargs):
            started.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if self.get_request_cancel_check()():
                    observed_cancel.set()
                    break
                time.sleep(0.01)
            return {"content": "answer", "tool_calls": [], "finish_reason": "stop", "usage": {}}

    primitives = BlockingPrimitives()
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    reset_features()
    reset_state()
    state = get_state()
    state.registry = MagicMock()
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kwargs: primitives)

    class DisconnectingRequest:
        headers = {}

        async def is_disconnected(self):
            return started.is_set()

    body = OpenAIChatRequest(
        model="frontdoor",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        x_tool_mode="client",
        x_session_id="disconnect-test",
    )
    response = await openai_compat.openai_chat_completions(body, DisconnectingRequest(), state)
    async for _ in response.body_iterator:
        pass

    assert observed_cancel.is_set(), (
        "the backend call remained live after the request disconnected while the "
        "route was awaiting its first streaming event"
    )
