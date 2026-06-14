"""Tests for Crawl4AI-backed web_research fetching."""

from __future__ import annotations

import json
from urllib.error import URLError

from src.tools.web import research
from src.tools.web.fetch import _fetch_cache


class _Response:
    def __init__(
        self,
        body: dict | str,
        *,
        status: int = 200,
        content_type: str = "application/json",
    ) -> None:
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        if isinstance(self._body, str):
            return self._body.encode("utf-8")
        return json.dumps(self._body).encode("utf-8")


def test_extract_crawl4ai_markdown_from_nested_result() -> None:
    payload = {
        "success": True,
        "results": [
            {
                "success": True,
                "markdown": {
                    "raw_markdown": "# Title\n\nUseful extracted text",
                },
            }
        ],
    }

    assert research._extract_crawl4ai_markdown(payload) == "# Title\n\nUseful extracted text"


def test_is_blocked_page_detects_status_and_interstitial_text() -> None:
    assert research._is_blocked_page("normal page text") is False
    assert research._is_blocked_page("Please verify you are a human before continuing") is True
    assert research._is_blocked_page("anything", status_code=403) is True


def test_fetch_page_uses_crawl4ai_when_available(monkeypatch) -> None:
    _fetch_cache.clear()
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        assert req.full_url == "http://localhost:11235/crawl"
        assert req.get_method() == "POST"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["urls"] == ["https://example.test/page"]
        return _Response(
            {
                "success": True,
                "results": [
                    {
                        "success": True,
                        "markdown": {"raw_markdown": "Crawl4AI extracted content"},
                    }
                ],
            }
        )

    monkeypatch.delenv("ORCHESTRATOR_CRAWL4AI_DISABLE", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_CRAWL4AI_URL", raising=False)
    monkeypatch.setattr(research.urllib.request, "urlopen", fake_urlopen)

    result = research._fetch_page("https://example.test/page")

    assert result["success"] is True
    assert result["content"] == "Crawl4AI extracted content"
    assert result["fetch_backend"] == "crawl4ai"
    assert result["cached"] is False
    assert len(calls) == 1


def test_fetch_page_falls_back_to_urllib_when_crawl4ai_fails(monkeypatch) -> None:
    _fetch_cache.clear()
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if req.full_url == "http://localhost:11235/crawl":
            raise URLError("crawl4ai unavailable")
        return _Response(
            "plain fallback content",
            content_type="text/plain",
        )

    monkeypatch.delenv("ORCHESTRATOR_CRAWL4AI_DISABLE", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_CRAWL4AI_URL", raising=False)
    monkeypatch.setattr(research.urllib.request, "urlopen", fake_urlopen)

    result = research._fetch_page("https://example.test/fallback")

    assert result["success"] is True
    assert result["content"] == "plain fallback content"
    assert result["fetch_backend"] == "urllib"
    assert calls == ["http://localhost:11235/crawl", "https://example.test/fallback"]


def test_poll_crawl4ai_task_returns_completed_result(monkeypatch) -> None:
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        return _Response(
            {
                "task_id": "crawl_123",
                "status": "completed",
                "result": {"markdown": "# Done"},
            }
        )

    monkeypatch.delenv("ORCHESTRATOR_CRAWL4AI_URL", raising=False)
    monkeypatch.setattr(research.urllib.request, "urlopen", fake_urlopen)

    result = research._poll_crawl4ai_task("crawl_123", timeout_seconds=1)

    assert result["status"] == "completed"
    assert calls == ["http://localhost:11235/job/crawl_123"]
