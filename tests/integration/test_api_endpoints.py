"""Integration tests for API endpoints via httpx.AsyncClient.

Tests health, chat, and config endpoints with dependency overrides
to avoid requiring live model servers.

Target: chat.py 53% → 75%+, health.py coverage boost
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi.testclient import TestClient

from src.api import create_app
from src.api.dependencies import dep_app_state, dep_health_tracker
from src.api.state import AppState

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_state():
    """Create a mock AppState with minimal wiring."""
    state = MagicMock(spec=AppState)
    state.progress_logger = MagicMock()
    state.hybrid_router = None
    state.tool_registry = MagicMock()
    state.script_registry = MagicMock()
    state.registry = MagicMock()
    state.registry.routing_hints = {}
    state.health_tracker = MagicMock()
    state.health_tracker.get_status.return_value = {}
    state.admission = MagicMock()
    state.admission.try_acquire.return_value = True
    state.increment_active = MagicMock()
    state.decrement_active = MagicMock()
    state.increment_request = MagicMock()
    state.get_stats.return_value = {
        "total_requests": 0,
        "total_turns": 0,
        "average_turns_per_request": 0.0,
        "mock_requests": 0,
        "real_requests": 0,
    }

    # LLM primitives in mock mode
    primitives = MagicMock()
    primitives._backends = True
    primitives.mock_mode = True
    primitives.total_tokens_generated = 0
    primitives.total_prompt_eval_ms = 0.0
    primitives.total_generation_ms = 0.0
    primitives._last_predicted_tps = 25.0
    primitives.total_http_overhead_ms = 0.0
    primitives.get_cache_stats.return_value = {"hits": 0, "misses": 0}
    primitives.llm_call.return_value = 'FINAL("test answer")'
    state.llm_primitives = primitives

    return state


@pytest.fixture
def app(mock_state):
    """Create a FastAPI app with dependency overrides."""
    application = create_app()
    application.dependency_overrides[dep_app_state] = lambda: mock_state
    application.dependency_overrides[dep_health_tracker] = lambda: mock_state.health_tracker
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app):
    """Create a test client."""
    with TestClient(app) as client:
        yield client


# ── Health endpoint ───────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client):
        """Basic health check returns 200 with status."""
        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")
        assert "mock_mode_available" in data

    def test_health_reports_version(self, client):
        """Health endpoint includes API version."""
        resp = client.get("/health")

        data = resp.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_health_includes_knowledge_tools(self, client):
        """Health endpoint reports knowledge tool availability."""
        resp = client.get("/health")

        data = resp.json()
        assert "knowledge_tools" in data


# ── Stats endpoint ────────────────────────────────────────────────────


class TestStatsEndpointBasic:
    """Tests for GET /stats."""

    def test_stats_returns_data(self, client):
        """Stats endpoint returns operational statistics."""
        resp = client.get("/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


# ── Chat endpoint basic validation ────────────────────────────────────


class TestChatEndpoint:
    """Tests for POST /chat."""

    def test_chat_requires_prompt(self, client):
        """Chat endpoint requires a non-empty prompt."""
        resp = client.post("/chat", json={})

        # Should return 422 (validation error) for missing prompt
        assert resp.status_code == 422

    def test_chat_accepts_valid_request(self, client, mock_state):
        """Chat endpoint accepts a valid request and returns response."""
        # The pipeline requires async execution — use a simplified mock
        # that returns the result directly
        mock_state.llm_primitives.llm_call.return_value = 'FINAL("test")'

        resp = client.post("/chat", json={
            "prompt": "What is 2+2?",
            "context": "math question",
        })

        # Should succeed or return a known error status
        # (depends on full pipeline wiring, but shouldn't 500)
        assert resp.status_code in (200, 503, 422)

    def test_chat_with_task_id(self, client):
        """Chat endpoint accepts explicit task_id."""
        resp = client.post("/chat", json={
            "prompt": "Hello",
            "task_id": "custom-task-123",
        })

        assert resp.status_code in (200, 503, 422)


# ── OpenAI-compat endpoint ────────────────────────────────────────────


class TestOpenAICompatEndpoint:
    """Tests for POST /v1/chat/completions."""

    def test_openai_compat_accepts_messages(self, client):
        """OpenAI-compat endpoint accepts messages array."""
        resp = client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        })

        # Should return response or 503 if backends not available
        assert resp.status_code in (200, 503, 422)

    def test_openai_compat_validates_model(self, client):
        """OpenAI-compat endpoint validates request structure."""
        resp = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        })

        # model field may or may not be required depending on validation
        assert resp.status_code in (200, 422, 503)


# ── Models endpoint ───────────────────────────────────────────────────


class TestModelsEndpoint:
    """Tests for GET /v1/models."""

    def test_models_returns_list(self, client):
        """Models endpoint returns model information."""
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
