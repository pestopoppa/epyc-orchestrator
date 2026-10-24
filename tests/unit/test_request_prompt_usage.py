"""Request-local prompt-usage aggregation for shared LLMPrimitives."""

from __future__ import annotations

import asyncio

from src.api.models import ChatResponse
from src.llm_primitives import LLMPrimitives


def test_prompt_usage_is_nested_request_scoped_and_requires_complete_backend_usage():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    assert primitives.get_request_prompt_tokens() is None
    with primitives.request_context(request_id="outer"):
        primitives._record_request_prompt_tokens(11)
        with primitives.request_context(request_id="nested"):
            primitives._record_request_prompt_tokens(0)
            assert primitives.get_request_prompt_tokens() == 11
        assert primitives.get_request_prompt_tokens() == 11
    assert primitives.get_request_prompt_tokens() is None

    with primitives.request_context(request_id="missing"):
        primitives._record_request_prompt_tokens(11)
        primitives._record_request_prompt_tokens(None)
        assert primitives.get_request_prompt_tokens() is None
    assert primitives.get_request_prompt_tokens() is None


def test_prompt_usage_explicit_zero_is_valid_and_exception_resets_scope():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    with primitives.request_context(request_id="zero"):
        primitives._record_request_prompt_tokens(0)
        assert primitives.get_request_prompt_tokens() == 0

    try:
        with primitives.request_context(request_id="exception"):
            primitives._record_request_prompt_tokens(None)
            raise RuntimeError("request failed")
    except RuntimeError:
        pass

    assert primitives.get_request_prompt_tokens() is None


def test_unmetered_generation_attempt_invalidates_otherwise_exact_request_sum():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    with primitives.request_context(request_id="external-generation"):
        primitives._record_request_prompt_tokens(17)
        primitives.mark_request_prompt_usage_unknown()
        assert primitives.get_request_prompt_tokens() is None

    with primitives.request_context(request_id="unaffected"):
        primitives._record_request_prompt_tokens(9)
        assert primitives.get_request_prompt_tokens() == 9


def test_prompt_usage_isolated_between_concurrent_requests_and_to_thread_calls():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    async def _request(request_id: str, prompt_counts: tuple[int, int]) -> int | None:
        with primitives.request_context(request_id=request_id):
            await asyncio.gather(
                asyncio.to_thread(primitives._record_request_prompt_tokens, prompt_counts[0]),
                asyncio.to_thread(primitives._record_request_prompt_tokens, prompt_counts[1]),
            )
            return primitives.get_request_prompt_tokens()

    async def _run() -> list[int | None]:
        return await asyncio.gather(
            _request("request-a", (5, 7)),
            _request("request-b", (20, 30)),
        )

    assert asyncio.run(_run()) == [12, 50]
    assert primitives.get_request_prompt_tokens() is None


def test_chat_response_serializes_unknown_prompt_usage_as_explicit_null():
    response = ChatResponse(
        answer="answer",
        turns=1,
        elapsed_seconds=0.1,
        mock_mode=False,
        real_mode=True,
    )

    assert response.model_dump()["prompt_tokens"] is None
    response.prompt_tokens = 0
    assert response.model_dump()["prompt_tokens"] == 0
