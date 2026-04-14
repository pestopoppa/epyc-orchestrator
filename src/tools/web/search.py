#!/usr/bin/env python3
"""Web search tool for knowledge retrieval.

Provides web search functionality with:
- SearXNG as primary backend (self-hosted JSON API, no scraping)
- DuckDuckGo/Brave as fallback (HTML scraping, used when SearXNG unavailable)
- Source registry filtering for trusted results
- Result summarization
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib.parse import quote_plus, urlencode

import threading
import time
import urllib.request
import urllib.error

from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import safe_execute

logger = logging.getLogger(__name__)

# Feature flag: use SearXNG as default search backend (SX-6)
_SEARXNG_DEFAULT = os.environ.get("ORCHESTRATOR_SEARXNG_DEFAULT", "1") == "1"
_SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8090")

# Rate limiter: minimum seconds between DDG requests to avoid bot detection
_DDG_MIN_INTERVAL = 2.0
_ddg_last_request: float = 0.0
_ddg_lock = threading.Lock()


def _search_duckduckgo(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Search DuckDuckGo and return results.

    Uses the DuckDuckGo HTML interface (no API key required).
    Rate-limited to avoid bot detection.

    Args:
        query: Search query.
        max_results: Maximum number of results.

    Returns:
        List of result dicts with title, url, snippet.
    """
    import subprocess

    # Rate limit: wait if we've searched recently
    global _ddg_last_request
    with _ddg_lock:
        elapsed = time.time() - _ddg_last_request
        if elapsed < _DDG_MIN_INTERVAL:
            time.sleep(_DDG_MIN_INTERVAL - elapsed)
        _ddg_last_request = time.time()

    encoded_query = quote_plus(query)

    # Try multiple search engines with fallback
    engines = [
        ("ddg", f"https://html.duckduckgo.com/html/?q={encoded_query}"),
        ("brave", f"https://search.brave.com/search?q={encoded_query}&source=web"),
    ]

    html = None
    engine_used = None
    for engine_name, url in engines:
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-L", "--max-time", "15", "--compressed",
                    "-H", "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                    "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "-H", "Accept-Language: en-US,en;q=0.5",
                    url,
                ],
                capture_output=True,
                timeout=20,
            )
            fetched = result.stdout.decode("utf-8", errors="replace")
            if result.returncode != 0 and not fetched.strip():
                continue  # try next engine (only skip if no output at all)
            # Check if we got real results (not a bot challenge page)
            if "result__a" in fetched or "snippet" in fetched:
                html = fetched
                engine_used = engine_name
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    if html is None:
        raise Exception("All search engines failed or rate-limited")

    # Parse results from HTML — engine-specific parsers
    from urllib.parse import unquote
    results = []

    if engine_used == "ddg":
        result_blocks = re.split(r'<div[^>]*class="[^"]*web-result[^"]*"', html)
        for block in result_blocks[1:]:
            if len(results) >= max_results:
                break
            link_match = re.search(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                block, re.DOTALL,
            )
            if not link_match:
                continue
            raw_url = link_match.group(1)
            title = re.sub(r'<[^>]+>', '', link_match.group(2)).strip()
            snippet_match = re.search(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|span|div)',
                block, re.DOTALL,
            )
            snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip() if snippet_match else ""
            if "uddg=" in raw_url:
                uddg_match = re.search(r"uddg=([^&]+)", raw_url)
                if uddg_match:
                    raw_url = unquote(uddg_match.group(1))
            elif raw_url.startswith("//"):
                raw_url = "https:" + raw_url
            if raw_url and title:
                results.append({"title": title, "url": raw_url, "snippet": snippet})

    elif engine_used == "brave":
        # Brave search HTML parser
        result_blocks = re.split(r'<div[^>]*class="snippet[^"]*"', html)
        for block in result_blocks[1:]:
            if len(results) >= max_results:
                break
            link_match = re.search(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            if not link_match:
                continue
            raw_url = link_match.group(1)
            title = re.sub(r'<[^>]+>', '', link_match.group(2)).strip()
            snippet_match = re.search(r'class="snippet-content[^"]*"[^>]*>(.*?)</p', block, re.DOTALL)
            snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip() if snippet_match else ""
            if raw_url and title and "brave.com" not in raw_url:
                results.append({"title": title, "url": raw_url, "snippet": snippet})

    logger.debug("Search engine=%s, results=%d", engine_used, len(results))
    return results


def _search_searxng(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Search via self-hosted SearXNG JSON API.

    Returns structured results with multi-engine provenance. Requires
    SearXNG container running on _SEARXNG_URL with JSON format enabled.

    Args:
        query: Search query.
        max_results: Maximum number of results.

    Returns:
        List of result dicts with title, url, snippet, score, engines.
    """
    params = {"q": query, "format": "json"}
    url = f"{_SEARXNG_URL}/search?{urlencode(params)}"

    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
    })

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        raise Exception(f"SearXNG request failed: {e}")

    # SX-4: Log unresponsive engines for telemetry
    unresponsive = data.get("unresponsive_engines", [])
    if unresponsive:
        engine_names = [entry[0] if isinstance(entry, list) else str(entry)
                        for entry in unresponsive]
        logger.info(
            "searxng unresponsive_engines: %s (query=%r)",
            ", ".join(engine_names), query,
        )

    results = []
    for r in data.get("results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "score": r.get("score", 0),
            "engines": r.get("engines", []),
        })

    logger.debug(
        "SearXNG search: results=%d, unresponsive=%d, query=%r",
        len(results), len(unresponsive), query,
    )
    return results


def web_search(
    query: str,
    max_results: int = 5,
    domain_filter: str | None = None,
) -> dict[str, Any]:
    """Search the web for information.

    Args:
        query: Search query.
        max_results: Maximum number of results to return.
        domain_filter: Optional domain to filter results (e.g., "docs.python.org").

    Returns:
        Dict with results list and metadata.
    """
    # Add domain filter to query if specified
    search_query = query
    if domain_filter:
        search_query = f"site:{domain_filter} {query}"

    # SX-6: Use SearXNG when enabled, fall back to DDG HTML scraping
    if _SEARXNG_DEFAULT:
        result = safe_execute(
            _search_searxng,
            search_query,
            max_results=max_results,
            timeout_seconds=15,
        )
        if result.success:
            return {
                "success": True,
                "results": result.data,
                "query": query,
                "result_count": len(result.data),
                "elapsed_ms": result.elapsed_ms,
                "backend": "searxng",
            }
        # SearXNG failed — fall through to DDG as fallback
        logger.warning("SearXNG failed (%s), falling back to DDG", result.error)

    result = safe_execute(
        _search_duckduckgo,
        search_query,
        max_results=max_results,
        timeout_seconds=20,
    )

    if not result.success:
        return {
            "success": False,
            "error": result.error,
            "query": query,
        }

    return {
        "success": True,
        "results": result.data,
        "query": query,
        "result_count": len(result.data),
        "elapsed_ms": result.elapsed_ms,
        "backend": "duckduckgo",
    }


def register_search_tool(registry: ToolRegistry) -> int:
    """Register the web_search tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="web_search",
        description="Search the web for information",
        category=ToolCategory.WEB,
        parameters={
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5)",
                "required": False,
            },
            "domain_filter": {
                "type": "string",
                "description": "Filter results to specific domain",
                "required": False,
            },
        },
        handler=web_search,
    )

    registry.register_tool(tool)
    return 1
