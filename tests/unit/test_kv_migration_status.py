"""Phase E — KV migration status reporting under per-region-locks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ca_mod = importlib.import_module("src.backends.concurrency_aware")


class _StubBackend:
    def __init__(self, url: str = "http://localhost:0"):
        self.config = type("C", (), {"base_url": url})()
        self.url = url


def _build(role: str = "frontdoor"):
    return ca_mod.ConcurrencyAwareBackend(
        full_backend=_StubBackend(),
        quarter_backends=[_StubBackend(), _StubBackend(), _StubBackend(), _StubBackend()],
        role=role,
    )


def test_migration_disabled_under_per_region_locks(monkeypatch) -> None:
    """When PER_REGION_LOCKS=1, KV migration must be flagged disabled."""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    cab = _build()
    assert cab._kv_migration_enabled is False
    status = cab.kv_migration_status()
    assert status["enabled"] is False
    assert status["per_region_locks"] is True
    assert "follow-up" in status["reason"].lower() or "Phase E" in status["follow_up"]


def test_migration_enabled_when_per_region_locks_off(monkeypatch) -> None:
    """When PER_REGION_LOCKS=0, the legacy migration path is available (assuming httpx)."""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "0")
    cab = _build()
    # Should be True iff httpx is available; in this repo it is.
    assert cab._kv_migration_enabled is ca_mod._HTTPX_AVAILABLE
    status = cab.kv_migration_status()
    assert status["per_region_locks"] is False


def test_migration_status_keys_present(monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    cab = _build()
    s = cab.kv_migration_status()
    for key in ("enabled", "per_region_locks", "reason", "follow_up"):
        assert key in s


def test_chat_request_carries_migration_budget_ms() -> None:
    """The advisory request field should be parsable + default None."""
    sys.path.insert(0, str(ROOT))
    from src.api.models.requests import ChatRequest
    r = ChatRequest(prompt="x")
    assert r.migration_budget_ms is None
    r2 = ChatRequest(prompt="x", migration_budget_ms=200)
    assert r2.migration_budget_ms == 200
