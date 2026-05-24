"""Loose-end fix: ContentionDenied → HTTP 503 + Retry-After."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_contention_denied_class_is_runtime_error() -> None:
    from src.scheduling.contention_gate import ContentionDenied
    assert issubclass(ContentionDenied, RuntimeError)
    # Can be raised + caught
    with pytest.raises(ContentionDenied) as exc_info:
        raise ContentionDenied("test reason")
    assert "test reason" in str(exc_info.value)


def test_app_registers_503_handler_for_contention_denied() -> None:
    """The FastAPI app must have an exception handler mapping ContentionDenied → 503."""
    import os
    os.environ["PYTEST_CURRENT_TEST"] = "1"  # disables lifespan
    from src.api import app
    from src.scheduling.contention_gate import ContentionDenied
    # FastAPI stores handlers in app.exception_handlers keyed by exception class
    assert ContentionDenied in app.exception_handlers


def test_503_response_carries_retry_after() -> None:
    """End-to-end via TestClient — a route that raises ContentionDenied should
    return 503 with Retry-After header + structured body."""
    import os
    os.environ["PYTEST_CURRENT_TEST"] = "1"
    from fastapi.testclient import TestClient
    from src.api import app
    from src.scheduling.contention_gate import ContentionDenied

    # Inject a test-only route that raises
    @app.get("/__test_contention_denied")
    def _raise():
        raise ContentionDenied("test pair frontdoor+architect blocked")

    client = TestClient(app)
    resp = client.get("/__test_contention_denied")
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "5"
    body = resp.json()
    assert body["error"] == "contention_denied"
    assert "frontdoor" in body["detail"]
    assert body["retry_after_s"] == 5
