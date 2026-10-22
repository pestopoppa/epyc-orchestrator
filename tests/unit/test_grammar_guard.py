"""Reject ``%llguidance`` grammars before they reach the frozen llama-server.

Production llama-server (v10) aborts the whole process on such a grammar, so
every entrypoint that forwards a caller grammar must return HTTP 400 without
making a backend call. Offline: all backends are mocks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api import app
from src.api.routes import passthrough as pt
from src.backends.grammar_guard import LLGUIDANCE_REJECTED_DETAIL, reject_unsafe_grammar
from src.backends.llama_server import LlamaServerBackend, ServerConfig
from src.model_server import InferenceRequest
from src.registry_loader import (
    AccelerationConfig,
    MemoryConfig,
    ModelConfig,
    PerformanceMetrics,
    RoleConfig,
)

BAD = '%llguidance {}\nstart: "a"'
GBNF = 'root ::= "yes" | "no"'


class TestRejectUnsafeGrammar:
    def test_rejects_leading_llguidance(self):
        with pytest.raises(HTTPException) as exc:
            reject_unsafe_grammar(BAD)
        assert exc.value.status_code == 400
        assert exc.value.detail == LLGUIDANCE_REJECTED_DETAIL

    @pytest.mark.parametrize("prefix", [" ", "\n", "\t\r\n  "])
    def test_rejects_with_leading_whitespace(self, prefix):
        with pytest.raises(HTTPException) as exc:
            reject_unsafe_grammar(prefix + BAD)
        assert exc.value.status_code == 400

    def test_plain_gbnf_passes(self):
        reject_unsafe_grammar(GBNF)
        # Only a LEADING marker aborts the server; mid-grammar text is a comment/literal.
        reject_unsafe_grammar('root ::= "%llguidance"')

    @pytest.mark.parametrize("value", [None, ""])
    def test_none_and_empty_pass(self, value):
        reject_unsafe_grammar(value)


# ── entrypoint: /v1/passthrough/{role}/... (raw body forwarded verbatim) ──────


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as c:
        yield c


@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
def test_passthrough_rejects_llguidance_without_backend_call(client, monkeypatch, endpoint):
    monkeypatch.setattr(pt, "_server_urls", lambda: {"architect_critic": "http://127.0.0.1:9"})
    monkeypatch.delenv(pt.PASSTHROUGH_ENV, raising=False)
    monkeypatch.delenv(pt.PASSTHROUGH_ROLES_ENV, raising=False)
    upstream = MagicMock()
    monkeypatch.setattr(pt, "run_upstream", upstream)
    monkeypatch.setattr(pt, "gate", upstream)
    body = {"messages": [{"role": "user", "content": "hi"}], "grammar": "  " + BAD}
    r = client.post(f"/v1/passthrough/architect_critic/{endpoint}", json=body)
    assert r.status_code == 400
    assert LLGUIDANCE_REJECTED_DETAIL in r.text
    upstream.assert_not_called()


# ── entrypoint: LlamaServerBackend (/completion and /v1/chat/completions) ─────


@pytest.fixture
def role_config():
    return RoleConfig(
        name="test_role",
        tier="C",
        description="Test role",
        model=ModelConfig(name="test-model", path="test-model.gguf", quant="Q4_K_M", size_gb=1.0),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


@pytest.mark.parametrize("use_chat", [False, True], ids=["completion", "chat_completions"])
def test_llama_server_backend_rejects_llguidance_without_http_call(role_config, use_chat):
    cfg = ServerConfig(
        base_url="http://localhost:8080", timeout=120, num_slots=4, connect_timeout=5,
        use_chat_completions=use_chat,
    )
    backend = LlamaServerBackend(config=cfg)
    request = InferenceRequest(role="test", prompt="hi", n_tokens=8, grammar=BAD)
    with patch.object(backend, "client") as http:
        with pytest.raises(HTTPException) as exc:
            backend.infer(role_config, request)
    assert exc.value.status_code == 400
    http.post.assert_not_called()
    http.stream.assert_not_called()


def test_llama_server_backend_payload_keeps_plain_gbnf(role_config):
    backend = LlamaServerBackend(config=ServerConfig(base_url="http://localhost:8080"))
    request = InferenceRequest(role="test", prompt="hi", n_tokens=8, grammar=GBNF)
    assert backend._build_payload(role_config, request)["grammar"] == GBNF


# ── entrypoint: WorkerPoolManager (call / batch / _http_call) ────────────────

aiohttp = pytest.importorskip("aiohttp", reason="aiohttp required for worker_pool tests")

from src.services.worker_pool import (  # noqa: E402
    WorkerConfig,
    WorkerInstance,
    WorkerPoolConfig,
    WorkerPoolManager,
    WorkerTier,
)


@pytest.fixture
def pool_and_worker():
    cfg = WorkerPoolConfig(
        enabled=True,
        workers={
            "explore": WorkerConfig(
                name="explore", port=8082, model_path="/models/explore.gguf",
                tier=WorkerTier.HOT, threads=4, slots=2, task_types=["explore"],
            ),
        },
    )
    pool = WorkerPoolManager(config=cfg)
    pool._http_session = MagicMock()
    return pool, WorkerInstance(config=cfg.workers["explore"])


async def test_worker_pool_http_call_rejects_llguidance_without_post(pool_and_worker):
    pool, worker = pool_and_worker
    with pytest.raises(HTTPException) as exc:
        await pool._http_call(worker, "p", 0.2, 16, grammar=BAD)
    assert exc.value.status_code == 400
    pool._http_session.post.assert_not_called()


async def test_worker_pool_call_and_batch_reject_llguidance_without_post(pool_and_worker):
    pool, _ = pool_and_worker
    pool.initialize = AsyncMock()
    pool._select_workers = AsyncMock()
    for coro in (pool.call("p", grammar=BAD), pool.batch(["a", "b"], grammar=BAD)):
        with pytest.raises(HTTPException) as exc:
            await coro
        assert exc.value.status_code == 400
    pool._select_workers.assert_not_called()
    pool._http_session.post.assert_not_called()
