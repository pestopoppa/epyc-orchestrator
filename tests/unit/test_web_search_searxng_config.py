"""SearXNG web-search configuration tests."""

import importlib

from src.tools.web import search as search_mod


def test_default_searxng_url_matches_managed_service_port(monkeypatch):
    """Default to the managed SearXNG Docker service, not the embedder port."""
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    reloaded = importlib.reload(search_mod)
    assert reloaded._SEARXNG_URL == "http://localhost:8888"
