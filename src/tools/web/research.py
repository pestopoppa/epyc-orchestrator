#!/usr/bin/env python3
"""Deep web research tool using worker models for content synthesis.

Combines web search with parallel content fetching and worker-model
summarization to return dense, synthesized information instead of
bare search snippets.

Architecture:
    1. web_search() → top N URLs + snippets
    2. ThreadPoolExecutor → fetch full page content in parallel
    3. Worker model (explore, port 8082) → synthesize each page in parallel
    4. Return combined dense summaries to the calling model
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError

from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import safe_execute
from src.tools.web.fetch import _extract_content, _fetch_cache, _CACHE_TTL_SECONDS
from src.tools.web.search import web_search

logger = logging.getLogger(__name__)

# Worker endpoint for synthesis (explore model — Qwen2.5-7B)
_WORKER_URL = "http://localhost:8082/completion"
_WORKER_TIMEOUT = 45  # seconds per synthesis call
_FETCH_TIMEOUT = 15   # seconds per URL fetch
_MAX_FETCH_WORKERS = 5
_MAX_SYNTH_WORKERS = 3
_CONTENT_PER_PAGE = 6000  # chars of page content to send to worker
_SYNTH_MAX_TOKENS = 512   # worker output cap

# Relevance detection patterns for synthesis output instrumentation
_IRRELEVANT_PHRASES = (
    "not relevant",
    "does not contain",
    "no relevant information",
    "not related to",
    "does not address",
    "no information about",
    "doesn't contain",
    "doesn't address",
    "not directly relevant",
    "unable to find relevant",
)
_IRRELEVANT_MAX_CHARS = 120  # synthesis shorter than this is likely a "not relevant" dismissal


def _is_irrelevant_synthesis(synthesis: str) -> bool:
    """Detect if worker synthesis indicates the page was not relevant.

    Heuristic: the worker prompt instructs "If the page is not relevant,
    say so briefly", producing short dismissals containing negation phrases.
    """
    if not synthesis.strip():
        return True
    lower = synthesis.lower()
    if len(synthesis) < _IRRELEVANT_MAX_CHARS:
        return any(phrase in lower for phrase in _IRRELEVANT_PHRASES)
    return False


def _fetch_page(url: str, max_length: int = _CONTENT_PER_PAGE) -> dict[str, Any]:
    """Fetch a single URL and extract text content.

    Args:
        url: URL to fetch.
        max_length: Max content chars.

    Returns:
        Dict with url, content, success, and timing.
    """
    start = time.perf_counter()

    # Check cache
    if url in _fetch_cache:
        content, cached_at = _fetch_cache[url]
        if time.time() - cached_at < _CACHE_TTL_SECONDS:
            elapsed = (time.perf_counter() - start) * 1000
            truncated = len(content) > max_length
            return {
                "url": url,
                "content": content[:max_length],
                "success": True,
                "elapsed_ms": elapsed,
                "cached": True,
            }

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OrchestratorBot/1.0)",
        "Accept": "text/html,application/xhtml+xml,text/plain",
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read().decode("utf-8", errors="replace")

            if "text/html" in content_type:
                content = _extract_content(raw, url)
            else:
                content = raw

            # Cache full content
            _fetch_cache[url] = (content, time.time())

            elapsed = (time.perf_counter() - start) * 1000
            return {
                "url": url,
                "content": content[:max_length],
                "success": True,
                "elapsed_ms": elapsed,
                "cached": False,
            }

    except (HTTPError, URLError, Exception) as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "url": url,
            "content": "",
            "success": False,
            "error": str(e),
            "elapsed_ms": elapsed,
        }


_MIN_PARAGRAPH_LEN = 80


def _dedup_pages(
    pages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove duplicate paragraphs across pages (paragraph-level SHA256 dedup).

    Pages are processed in order — the first page to contain a paragraph keeps
    it; later pages have the duplicate removed.  Short paragraphs (< _MIN_PARAGRAPH_LEN
    chars) are always kept to avoid stripping headings and list items.

    Args:
        pages: Ordered list of page dicts (each must have a ``content`` key).

    Returns:
        Tuple of (deduped_pages, stats_dict).
    """
    seen: set[str] = set()
    stats = {"paragraphs_removed": 0, "chars_saved": 0, "pages_affected": 0}
    deduped: list[dict[str, Any]] = []

    for page in pages:
        content = page.get("content", "")
        paragraphs = content.split("\n\n")
        kept: list[str] = []
        page_had_removal = False

        for para in paragraphs:
            stripped = para.strip()
            if len(stripped) < _MIN_PARAGRAPH_LEN:
                kept.append(para)
                continue

            # Normalize: lowercase + collapse whitespace
            normalized = re.sub(r"\s+", " ", stripped.lower())
            h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

            if h in seen:
                stats["paragraphs_removed"] += 1
                stats["chars_saved"] += len(stripped)
                page_had_removal = True
            else:
                seen.add(h)
                kept.append(para)

        if page_had_removal:
            stats["pages_affected"] += 1

        deduped.append({**page, "content": "\n\n".join(kept)})

    return deduped, stats


def _synthesize_page(
    url: str,
    title: str,
    content: str,
    query: str,
) -> dict[str, Any]:
    """Send page content to worker model for query-focused synthesis.

    Args:
        url: Source URL.
        title: Page title from search results.
        content: Extracted page text.
        query: Original search query (for relevance focusing).

    Returns:
        Dict with url, title, synthesis, success.
    """
    if not content.strip():
        return {
            "url": url,
            "title": title,
            "synthesis": "",
            "success": False,
            "error": "Empty content",
        }

    prompt = (
        f"<|im_start|>system\n"
        f"You are a research assistant. Extract and synthesize the most relevant "
        f"information from the following web page content that answers or relates "
        f"to the query. Be concise but thorough — include specific facts, numbers, "
        f"names, and technical details. If the page is not relevant, say so briefly.\n"
        f"IMPORTANT: Only use information from the retrieved content below. "
        f"Do not add facts from your training data.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Query: {query}\n\n"
        f"Page: {title} ({url})\n\n"
        f"Content:\n{content}\n\n"
        f"Synthesize the relevant information from this page. Cite the source URL when stating specific facts.\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    payload = json.dumps({
        "prompt": prompt,
        "temperature": 0.1,
        "n_predict": _SYNTH_MAX_TOKENS,
        "stream": False,
        "stop": ["<|im_end|>"],
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(_WORKER_URL, data=payload, headers=headers)

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=_WORKER_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
            synthesis = data.get("content", "").strip()
            elapsed = (time.perf_counter() - start) * 1000

            return {
                "url": url,
                "title": title,
                "synthesis": synthesis,
                "success": True,
                "elapsed_ms": elapsed,
            }

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        logger.warning(f"Worker synthesis failed for {url}: {e}")
        return {
            "url": url,
            "title": title,
            "synthesis": "",
            "success": False,
            "error": str(e),
            "elapsed_ms": elapsed,
        }


def _web_research_impl(
    query: str,
    max_results: int = 5,
    max_pages: int = 3,
    domain_filter: str | None = None,
) -> dict[str, Any]:
    """Core implementation of web_research.

    Args:
        query: Search query.
        max_results: Max search results to retrieve.
        max_pages: Max pages to fetch and synthesize.
        domain_filter: Optional domain filter.

    Returns:
        Dict with synthesized results and metadata.
    """
    t0 = time.perf_counter()
    dedup_stats = {"paragraphs_removed": 0, "chars_saved": 0, "pages_affected": 0}

    # Step 1: Search
    search_result = web_search(query, max_results=max_results, domain_filter=domain_filter)
    if not search_result["success"]:
        return {
            "success": False,
            "error": f"Search failed: {search_result.get('error', 'unknown')}",
            "query": query,
        }

    search_backend = search_result.get("backend", "unknown")
    results = search_result["results"]
    if not results:
        return {
            "success": True,
            "query": query,
            "sources": [],
            "synthesis": "No search results found.",
            "search_elapsed_ms": search_result.get("elapsed_ms", 0),
        }

    # Step 2: Fetch top pages in parallel
    pages_to_fetch = results[:max_pages]
    fetched = {}

    with ThreadPoolExecutor(max_workers=_MAX_FETCH_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_page, r["url"]): r
            for r in pages_to_fetch
        }
        for future in as_completed(futures):
            result_meta = futures[future]
            try:
                fetch_result = future.result()
                fetched[result_meta["url"]] = {
                    **fetch_result,
                    "title": result_meta["title"],
                    "snippet": result_meta["snippet"],
                }
            except Exception as e:
                logger.warning(f"Fetch failed for {result_meta['url']}: {e}")
                fetched[result_meta["url"]] = {
                    "url": result_meta["url"],
                    "title": result_meta["title"],
                    "snippet": result_meta["snippet"],
                    "content": "",
                    "success": False,
                    "error": str(e),
                }

    # Step 3: Rank-ordered dedup, then synthesize with worker models in parallel
    successful_pages = [
        fetched[r["url"]]
        for r in pages_to_fetch
        if r["url"] in fetched
        and fetched[r["url"]].get("success")
        and fetched[r["url"]].get("content", "").strip()
    ]
    to_synthesize, dedup_stats = _dedup_pages(successful_pages)

    synthesized = []
    if to_synthesize:
        with ThreadPoolExecutor(max_workers=_MAX_SYNTH_WORKERS) as pool:
            futures = {
                pool.submit(
                    _synthesize_page,
                    f["url"],
                    f["title"],
                    f["content"],
                    query,
                ): f
                for f in to_synthesize
            }
            for future in as_completed(futures):
                try:
                    synth_result = future.result()
                    synthesized.append(synth_result)
                except Exception as e:
                    meta = futures[future]
                    logger.warning(f"Synthesis failed for {meta['url']}: {e}")

    # Step 3b: Relevance instrumentation — classify synthesis results
    irrelevant_pages = []
    relevant_pages = []
    for s in synthesized:
        if not s.get("success"):
            continue
        synthesis_text = s.get("synthesis", "")
        if _is_irrelevant_synthesis(synthesis_text):
            irrelevant_pages.append(s["url"])
            logger.info(
                "web_research relevance: IRRELEVANT page=%s query=%r synthesis_len=%d",
                s["url"], query, len(synthesis_text),
            )
        else:
            relevant_pages.append(s["url"])

    total_synth = len(relevant_pages) + len(irrelevant_pages)
    irrelevant_rate = len(irrelevant_pages) / total_synth if total_synth > 0 else 0.0
    if irrelevant_pages:
        logger.info(
            "web_research relevance summary: query=%r total=%d relevant=%d "
            "irrelevant=%d rate=%.1f%% backend=%s",
            query, total_synth, len(relevant_pages),
            len(irrelevant_pages), irrelevant_rate * 100, search_backend,
        )

    # Step 4: Build structured output
    sources = []
    for r in results:
        url = r["url"]
        source = {
            "title": r["title"],
            "url": url,
            "snippet": r["snippet"],
        }

        # Attach synthesis if available
        for s in synthesized:
            if s["url"] == url and s.get("success") and s.get("synthesis"):
                source["synthesis"] = s["synthesis"]
                source["relevant"] = url not in irrelevant_pages
                break

        # Fall back to snippet-only for unfetched/failed pages
        sources.append(source)

    total_elapsed = (time.perf_counter() - t0) * 1000
    synth_count = sum(1 for s in sources if "synthesis" in s)

    return {
        "success": True,
        "query": query,
        "sources": sources,
        "pages_fetched": len(to_synthesize),
        "pages_synthesized": synth_count,
        "pages_irrelevant": len(irrelevant_pages),
        "irrelevant_rate": round(irrelevant_rate, 3),
        "dedup_paragraphs_removed": dedup_stats["paragraphs_removed"],
        "dedup_chars_saved": dedup_stats["chars_saved"],
        "total_elapsed_ms": total_elapsed,
        "search_backend": search_backend,
    }


def web_research(
    query: str,
    max_results: int = 5,
    max_pages: int = 3,
    domain_filter: str | None = None,
) -> dict[str, Any]:
    """Deep web research: search, fetch, and synthesize with worker models.

    Performs a web search, fetches the top pages in parallel, then uses
    worker models to extract and synthesize query-relevant information
    from each page. Returns dense summaries instead of bare snippets.

    Args:
        query: Search query.
        max_results: Maximum search results to retrieve (default 5).
        max_pages: Maximum pages to fetch and synthesize (default 3).
        domain_filter: Optional domain filter (e.g., "docs.python.org").

    Returns:
        Dict with synthesized sources and metadata.
    """
    result = safe_execute(
        _web_research_impl,
        query,
        max_results=max_results,
        max_pages=max_pages,
        domain_filter=domain_filter,
        timeout_seconds=90,
        max_output=32768,
    )

    if not result.success:
        return {
            "success": False,
            "error": result.error,
            "query": query,
        }

    return result.data


def register_research_tool(registry: ToolRegistry) -> int:
    """Register the web_research tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="web_research",
        description=(
            "Deep web research: searches the web, fetches top pages in parallel, "
            "and uses worker models to synthesize relevant information from each "
            "page. Returns dense summaries instead of bare search snippets. "
            "Use this instead of web_search when you need actual content, not just URLs."
        ),
        category=ToolCategory.WEB,
        parameters={
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum search results to retrieve (default 5)",
                "required": False,
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum pages to fetch and synthesize (default 3)",
                "required": False,
            },
            "domain_filter": {
                "type": "string",
                "description": "Filter results to specific domain",
                "required": False,
            },
        },
        handler=web_research,
        side_effects=["network_access", "calls_llm", "read_only"],
    )

    registry.register_tool(tool)
    return 1
