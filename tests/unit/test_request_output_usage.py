"""Request-local exact output usage never substitutes text estimates."""

from __future__ import annotations

import asyncio

from src.api.models import ChatResponse
from src.llm_primitives import LLMPrimitives


def test_output_usage_sums_native_counts_including_explicit_zero():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    with primitives.request_context(request_id="exact-output"):
        first = primitives._begin_request_output_tokens()
        second = primitives._begin_request_output_tokens()
        primitives._record_request_output_tokens(7, "llama_server:/completion.tokens_predicted", first)
        primitives._record_request_output_tokens(0, "llama_server:/v1/chat/completions.usage.completion_tokens", second)
        assert primitives.get_request_output_usage() == (
            7,
            "reported",
            ["llama_server:/completion.tokens_predicted", "llama_server:/v1/chat/completions.usage.completion_tokens"],
        )

    assert primitives.get_request_output_usage() == (None, "unknown", [])


def test_nested_request_contexts_share_the_outer_output_collector():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    with primitives.request_context(request_id="outer"):
        primitives._record_request_output_tokens(4, "llama_server:/completion.tokens_predicted")
        with primitives.request_context(request_id="nested"):
            primitives._record_request_output_tokens(6, "llama_server:/v1/chat/completions.usage.completion_tokens")
            assert primitives.get_request_output_usage() == (
                10,
                "reported",
                ["llama_server:/completion.tokens_predicted", "llama_server:/v1/chat/completions.usage.completion_tokens"],
            )
        assert primitives.get_request_output_usage() == (
            10,
            "reported",
            ["llama_server:/completion.tokens_predicted", "llama_server:/v1/chat/completions.usage.completion_tokens"],
        )


def test_output_usage_is_unknown_for_no_calls_missing_counts_and_pending_calls():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    with primitives.request_context(request_id="no-call"):
        assert primitives.get_request_output_usage() == (None, "unknown", [])

    with primitives.request_context(request_id="missing"):
        attempt = primitives._begin_request_output_tokens()
        primitives._record_request_output_tokens(None, None, attempt)
        assert primitives.get_request_output_usage() == (None, "unknown", [])

    with primitives.request_context(request_id="pending"):
        attempt = primitives._begin_request_output_tokens()
        primitives._record_request_output_tokens(4, "llama_server:/completion.tokens_predicted")
        assert primitives.get_request_output_usage() == (
            None,
            "unknown",
            ["llama_server:/completion.tokens_predicted"],
        )
        primitives._record_request_output_tokens(3, "llama_server:/completion.tokens_predicted", attempt)
        assert primitives.get_request_output_usage() == (
            7,
            "reported",
            ["llama_server:/completion.tokens_predicted"],
        )


def test_unmetered_generation_invalidates_complete_output_sum():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    with primitives.request_context(request_id="formalizer"):
        primitives._record_request_output_tokens(11, "llama_server:/completion.tokens_predicted")
        primitives.mark_request_output_usage_unknown()
        assert primitives.get_request_output_usage() == (
            None,
            "unknown",
            ["llama_server:/completion.tokens_predicted"],
        )


def test_output_usage_isolated_between_concurrent_request_contexts_and_threads():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    async def _request(request_id: str, values: tuple[int, int]) -> tuple[int | None, str, list[str]]:
        with primitives.request_context(request_id=request_id):
            attempts = [primitives._begin_request_output_tokens() for _ in values]
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        primitives._record_request_output_tokens,
                        value,
                        "llama_server:/completion.tokens_predicted",
                        attempt,
                    )
                    for value, attempt in zip(values, attempts)
                )
            )
            return primitives.get_request_output_usage()

    async def _run():
        return await asyncio.gather(
            _request("request-a", (5, 7)),
            _request("request-b", (20, 30)),
        )

    assert asyncio.run(_run()) == [
        (12, "reported", ["llama_server:/completion.tokens_predicted"]),
        (50, "reported", ["llama_server:/completion.tokens_predicted"]),
    ]


def test_chat_response_serializes_unknown_and_reported_output_usage():
    response = ChatResponse(
        answer="answer",
        turns=1,
        elapsed_seconds=0.1,
        mock_mode=False,
        real_mode=True,
    )
    assert response.model_dump()["output_tokens_reported"] is None
    assert response.model_dump()["output_tokens_status"] == "unknown"
    response.output_tokens_reported = 0
    response.output_tokens_status = "reported"
    response.output_tokens_sources = ["llama_server:/completion.tokens_predicted"]
    dumped = response.model_dump()
    assert dumped["output_tokens_reported"] == 0
    assert dumped["output_tokens_status"] == "reported"
    assert dumped["output_tokens_sources"] == ["llama_server:/completion.tokens_predicted"]
