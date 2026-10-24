"""Per-request primitives share one backend build per AppState (A2-04, A2-12).

/v1, voice and /chat/stream built a fresh LLMPrimitives per request, and with it
one LlamaServerBackend + httpx.Client per (role, URL) that nothing ever closed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import src.backends.concurrency_aware as CA
from src.backends.llama_server import LlamaServerBackend
from src.llm_primitives import LLMPrimitives

URLS = {
    "frontdoor": "full:http://localhost:18070,http://localhost:18080,http://localhost:18180",
    "worker_general": "http://localhost:18072,http://localhost:18172",
    "architect_general": "http://localhost:18083",
}


def _build(pool, **kwargs) -> LLMPrimitives:
    return LLMPrimitives(mock_mode=False, server_urls=dict(URLS), backend_pool=pool, **kwargs)


def test_config_builds_reuse_the_pooled_backends():
    pool: dict = {}
    real_init = LlamaServerBackend.__init__
    built: list[str] = []

    def counting_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self.config.base_url)

    with patch.object(LlamaServerBackend, "__init__", counting_init):
        first = _build(pool)
        n_first = len(built)
        others = [_build(pool) for _ in range(10)]

    assert n_first == 6
    assert len(built) == n_first, "warm builds must not create HTTP clients"
    for prims in others:
        assert prims._backends.keys() == first._backends.keys()
        assert all(prims._backends[r] is first._backends[r] for r in first._backends)
        assert prims._backends is not first._backends


def test_request_urls_owner_change_and_no_pool_build_fresh():
    pool: dict = {}
    shared = _build(pool)
    fresh = [
        _build(pool, server_urls_source="request"),
        _build(pool, registry=SimpleNamespace(_raw={})),
        _build(None),
    ]
    for prims in fresh:
        assert all(prims._backends[r] is not shared._backends[r] for r in shared._backends)


def test_session_maps_stay_bounded(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "0")
    monkeypatch.setattr(CA, "_MAX_TRACKED_SESSIONS", 8)
    stub = SimpleNamespace(config=SimpleNamespace(base_url="http://localhost:18080"))
    cab = CA.ConcurrencyAwareBackend(None, [stub], role="worker_general")

    for i in range(50):
        with cab._dispatch(session_id=f"s{i}"):
            pass
        cab._set_session_state(f"s{i}", state="assigned_quarter", quarter=0)

    assert len(cab._session_last_seen) <= 8
    assert set(cab._session_state) <= set(cab._session_last_seen) | {"s49"}
    assert "s49" in cab._session_last_seen
