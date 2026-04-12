#!/usr/bin/env python3
"""Web search tool for knowledge retrieval.

Provides web search functionality with:
- DuckDuckGo as default backend (no API key required)
- Source registry filtering for trusted results
- Result summarization
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

import threading
import time

from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import safe_execute

logger = logging.getLogger(__name__)

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
