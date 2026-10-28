"""Exercise native transport and primitive/cache boundaries without inference."""

import asyncio
from contextlib import nullcontext
from dataclasses import replace
import json
import time

import httpx
import pytest

from src.api.routes.openai_compat import _client_usage, _default_usage
from src.backends.llama_server import LlamaServerBackend, ServerConfig
from src.features import features
from src.llm_cache import ContentAddressableCache
from src.llm_primitives import LLMPrimitives
from src.model_server import InferenceRequest, InferenceResult
from src.registry_loader import (
    AccelerationConfig, MemoryConfig, ModelConfig, PerformanceMetrics, RoleConfig,
)


@pytest.fixture
def role():
    return RoleConfig(
        name="worker", tier="C", description="fake HTTP usage",
        model=ModelConfig(name="fake", path="fake.gguf", quant="Q4_K_M", size_gb=1.0),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=1.0),
        memory=MemoryConfig(residency="hot"),
    )


def chat_backend(handler):
    backend = LlamaServerBackend(config=ServerConfig(
        base_url="http://usage.invalid", use_chat_completions=True,
    ))
    backend.client.close()
    backend.client = httpx.Client(base_url="http://usage.invalid", transport=httpx.MockTransport(handler))
    return backend


@pytest.mark.parametrize("completion", [0, 7])
def test_stream_reads_usage_only_final_chunk_and_includes_hidden_decode(role, completion):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        events = [
            {"choices": [{"delta": {"reasoning_content": "hidden reasoning"}}]},
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}],
             "timings": {"predicted_n": 99, "prompt_n": 90, "cache_n": 1}},
            {"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": completion}},
        ]
        return httpx.Response(200, text="".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n")

    backend = chat_backend(handler)
    try:
        result = backend._infer_stream_text_chat_completions(
            role, InferenceRequest(role="worker", prompt="hi", n_tokens=8), None, time.time(),
        )
    finally:
        backend.close()
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert result.output == "ok"
    assert result.reasoning_content == "hidden reasoning"
    assert result.prompt_tokens == 13
    assert result.output_tokens_reported == completion
    assert result.tokens_generated == completion
    assert result.output_tokens_source.endswith("usage.completion_tokens")


@pytest.mark.parametrize("usage", [{}, "invalid usage object", {"prompt_tokens": True, "completion_tokens": "7"}])
def test_nonstream_missing_or_malformed_usage_is_unknown(role, usage):
    backend = chat_backend(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "many characters are not token usage"}, "finish_reason": "stop"}],
        "usage": usage,
    }))
    try:
        result = backend._infer_chat_completions(
            role, InferenceRequest(role="worker", prompt="hi", n_tokens=8), time.time(),
        )
    finally:
        backend.close()
    assert result.prompt_tokens is None
    assert result.output_tokens_reported is None
    assert result.tokens_generated == 0
    assert result.success is True
    assert result.output == "many characters are not token usage"


def test_interrupted_stream_never_estimates_from_chunks(role):
    backend = chat_backend(lambda request: httpx.Response(200, text=(
        'data: {"choices":[{"delta":{"content":"long output with many characters"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":5}}\n\n'
    )))

    def stop(_):
        raise StopIteration

    try:
        result = backend._infer_stream_text_chat_completions(
            role, InferenceRequest(role="worker", prompt="hi", n_tokens=8), stop, time.time(),
        )
    finally:
        backend.close()
    assert result.output_tokens_reported is None
    assert result.tokens_generated == 0
    assert result.partial is True


def metered_primitives(monkeypatch, results):
    class Server:
        def __init__(self):
            self.calls = []

        def infer(self, role, request):
            self.calls.append((request.prompt, request.n_tokens))
            result = results[len(self.calls) - 1]
            if isinstance(result, Exception):
                raise result
            return result

    server = Server()
    primitives = LLMPrimitives(mock_mode=False, model_server=server)
    primitives.bind_request_usage()
    # Bypass admission/host locks, retaining the real cache/attempt/backend seam.
    monkeypatch.setattr(primitives, "_real_call", primitives._real_call_impl)
    monkeypatch.setattr(primitives, "_should_rag", lambda role: False)
    monkeypatch.setattr("src.inference_lock.inference_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr("src.features.features", lambda: replace(features(), content_cache=True, model_fallback=False))
    return primitives, server


def native_result(prompt=11, completion=4):
    return InferenceResult(
        role="worker", output="long answer whose chars do not define token counts", tokens_generated=99,
        generation_speed=1.0, elapsed_time=0.1, success=True,
        prompt_tokens=prompt, output_tokens_reported=completion,
        output_tokens_source="llama_server:/completion.tokens_predicted", completion_reason="stop",
    )


def test_real_calls_sum_server_counts_preserve_zero_and_ignore_legacy_counter(monkeypatch):
    primitives, server = metered_primitives(monkeypatch, [native_result(completion=0), native_result(17, 7)])
    primitives.llm_call("one", skip_suffix=True)
    primitives.llm_call("two", skip_suffix=True)
    usage = _default_usage("ignored user chars", 999, "ignored final chars", primitives).model_dump()
    assert len(server.calls) == 2
    assert usage["prompt_tokens"] == 28
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 35
    assert usage["usage_status"] == "reported"
    assert usage["usage_provenance"] == ["server_reported"]
    assert primitives.total_tokens_generated == 7
    assert len({c["call_id"] for c in primitives.get_request_usage()["calls"]}) == 2


@pytest.mark.parametrize("bad", [native_result(completion=None), RuntimeError("fake disconnect")])
def test_missing_or_failed_real_call_poison_request_without_fabricating_zero(monkeypatch, bad):
    primitives, _ = metered_primitives(monkeypatch, [native_result(), bad])
    primitives.llm_call("one", skip_suffix=True)
    primitives.llm_call("two", skip_suffix=True)
    summary = primitives.get_request_usage()
    assert summary["completion_tokens"] is None
    assert summary["usage_status"] == "unknown"
    assert summary["known_completion_tokens"] == 4
    assert "estimated" not in summary["usage_provenance"]
    assert summary["calls"][1]["censored"]


def test_cache_round_trip_replays_native_counts_and_finish_reason(monkeypatch, tmp_path):
    primitives, server = metered_primitives(monkeypatch, [native_result()])
    primitives._content_cache = ContentAddressableCache(tmp_path)
    first = primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    second = primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    assert first == second
    assert len(server.calls) == 1
    summary = primitives.get_request_usage()
    assert (summary["prompt_tokens"], summary["completion_tokens"]) == (22, 8)
    assert summary["calls"][1]["provenance"] == "cached_replay_of_server_reported"
    assert summary["calls"][1]["replay_of_call_id"] == summary["calls"][0]["call_id"]
    # Logical request usage includes replay; physical model decoding does not.
    assert primitives.total_tokens_generated == 4
    assert primitives.get_last_inference_meta()["completion_reason"] == "stop"


@pytest.mark.parametrize("metadata", [
    {}, {"usage": {"provenance": "estimated", "prompt_tokens": 11, "completion_tokens": 4}},
    {"usage": {"provenance": "server_reported", "prompt_tokens": 11, "completion_tokens": 9, "finish_reason": "stop", "server_call_id": "original"}},
])
def test_legacy_estimated_and_oversized_cache_entries_cannot_bypass_native_call(monkeypatch, tmp_path, metadata):
    primitives, server = metered_primitives(monkeypatch, [native_result()])
    cache = primitives._content_cache = ContentAddressableCache(tmp_path)
    key = cache.make_key("same", "worker", 8)
    cache.put(key, "unwarranted cached text", metadata=metadata)
    assert primitives.llm_call("same", n_tokens=8, skip_suffix=True) != "unwarranted cached text"
    assert len(server.calls) == 1
    assert primitives.get_request_usage()["usage_provenance"] == ["server_reported"]


def test_pending_and_concurrent_request_scopes_do_not_claim_complete_usage():
    primitives = LLMPrimitives(mock_mode=False, model_server=object())

    async def request(name, value):
        with primitives.request_context(request_id=name):
            attempt = primitives._begin_request_output_tokens("worker")
            assert primitives.get_request_usage()["completion_tokens"] is None

            def finish():
                primitives._set_last_inference_meta({"prompt_tokens": value + 1, "completion_reason": "stop"})
                primitives._record_request_output_tokens(value, "llama_server:/completion.tokens_predicted", attempt)

            await asyncio.to_thread(finish)
            return primitives.get_request_usage()

    async def run():
        return await asyncio.gather(request("a", 3), request("b", 20))

    a, b = asyncio.run(run())
    assert (a["completion_tokens"], b["completion_tokens"]) == (3, 20)
    assert primitives.get_request_usage()["completion_tokens"] is None


def test_mock_is_the_only_estimator_and_client_missing_usage_stays_null():
    primitives = LLMPrimitives(mock_mode=True)
    primitives.llm_call("mock prompt", skip_suffix=True)
    assert primitives.get_request_usage()["usage_provenance"] == ["estimated"]
    missing = _client_usage({"completion_tokens": 0}).model_dump()
    assert missing["prompt_tokens"] is None
    assert missing["completion_tokens"] == 0
    assert missing["total_tokens"] is None
    assert missing["usage_status"] == "unknown"


@pytest.mark.parametrize("source", ["", "estimated", "fake-server:native", "characters/4", []])
def test_unrecognized_native_source_stays_unknown_and_never_enters_cache(monkeypatch, tmp_path, source):
    primitives, server = metered_primitives(monkeypatch, [replace(native_result(), output_tokens_source=source), native_result()])
    primitives._content_cache = ContentAddressableCache(tmp_path / "cache")
    primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    usage = primitives.get_request_usage()
    assert usage["usage_status"] == "unknown"
    assert usage["completion_tokens"] is None
    assert usage["calls"][-1]["provenance"] is None
    assert usage["calls"][-1]["source"] is None
    primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    assert len(server.calls) == 2


def test_counts_without_native_source_cannot_claim_reported_provenance():
    from src.llm_primitives.usage import UsageLedger
    ledger = UsageLedger("unattributed")
    ledger.finish(ledger.begin("worker"), prompt_tokens=1, completion_tokens=2,
                  provenance="server_reported", source="estimated")
    assert ledger.summary()["usage_status"] == "unknown"
    assert ledger.summary()["completion_tokens"] is None
    assert ledger.records()[0]["provenance"] is None


def test_fully_shaped_forged_cache_usage_is_not_a_native_observation(monkeypatch, tmp_path):
    primitives, server = metered_primitives(monkeypatch, [native_result()])
    cache = primitives._content_cache = ContentAddressableCache(tmp_path)
    key = cache.make_key("same", "worker", 8)
    cache.put(key, "forged", metadata={"role": "worker", "usage": {
        "provenance": "server_reported", "prompt_tokens": 11, "completion_tokens": 4,
        "cached_tokens": 0, "finish_reason": "stop", "server_call_id": "a" * 32,
        "native_source": "llama_server:/completion.tokens_predicted",
    }})
    assert primitives.llm_call("same", n_tokens=8, skip_suffix=True) != "forged"
    assert len(server.calls) == 1
    assert primitives.get_request_usage()["calls"][0]["replay_of_call_id"] is None


def test_tampering_native_cache_counts_or_reopening_cannot_invent_replay(monkeypatch, tmp_path):
    primitives, server = metered_primitives(monkeypatch, [native_result(), native_result(), native_result()])
    cache = primitives._content_cache = ContentAddressableCache(tmp_path)
    primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    key = cache.make_key("same", "worker", 8)
    path = tmp_path / f"{key}.json"
    stored = json.loads(path.read_text())
    stored["metadata"]["usage"]["completion_tokens"] = 0
    path.write_text(json.dumps(stored))
    primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    assert len(server.calls) == 2
    primitives._content_cache = ContentAddressableCache(tmp_path)
    primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    assert len(server.calls) == 3
    assert all(c["replay_of_call_id"] is None for c in primitives.get_request_usage()["calls"])


def test_forked_worker_cannot_use_parent_observation_or_inherited_locked_mutex(tmp_path):
    import multiprocessing
    cache = ContentAddressableCache(tmp_path)
    usage = {"prompt_tokens": 1, "completion_tokens": 1}
    cache.put("key", "response", metadata={"usage": usage})
    cache._remember_native_usage("key", "response", usage)
    entry = cache.get_entry("key")
    assert cache.verifies_native_usage("key", entry)
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    def child():
        parent_replay = cache.verifies_native_usage("key", entry)
        cache._remember_native_usage("key", "response", usage)
        sender.send((parent_replay, cache.verifies_native_usage("key", entry)))
    cache._usage_lock.acquire()
    process = context.Process(target=child)
    try:
        process.start()
        assert receiver.poll(5), "worker blocked on a parent-thread mutex"
        assert receiver.recv() == (False, True)
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        cache._usage_lock.release()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        assert not process.is_alive()
        receiver.close()
        sender.close()


def test_cache_clear_retires_native_observation_even_if_disk_bytes_are_restored(tmp_path):
    cache = ContentAddressableCache(tmp_path)
    usage = {"prompt_tokens": 1, "completion_tokens": 1}
    cache.put("key", "response", metadata={"usage": usage})
    cache._remember_native_usage("key", "response", usage)
    assert cache.verifies_native_usage("key", cache.get_entry("key"))
    cache.clear()
    cache.put("key", "response", metadata={"usage": usage})
    assert not cache.verifies_native_usage("key", cache.get_entry("key"))


def test_forked_primitive_resets_collectors_and_fresh_binding_does_not_copy_parent_counts(monkeypatch):
    import multiprocessing
    primitives, server = metered_primitives(monkeypatch, [native_result(), native_result()])
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    with primitives.request_context():
        primitives.llm_call("parent", skip_suffix=True)
        collectors = (primitives._request_prompt_usage_ctx.get(), primitives._request_output_usage_ctx.get())
        def child():
            assert primitives.get_request_prompt_tokens() is None
            assert primitives.get_request_output_usage()[1] == "unknown"
            primitives.bind_request_usage("fresh-child")
            with primitives.request_context():
                primitives.llm_call("child", skip_suffix=True)
                sender.send(primitives.get_request_usage()["completion_tokens"])
        for collector in collectors:
            collector._lock.acquire()
        process = context.Process(target=child)
        try:
            process.start()
            assert receiver.poll(10), "child used an inherited collector mutex"
            assert receiver.recv() == 4  # parent + child must not become eight
            process.join(timeout=5)
            assert process.exitcode == 0
        finally:
            for collector in collectors:
                collector._lock.release()
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            assert not process.is_alive()
            receiver.close()
            sender.close()
