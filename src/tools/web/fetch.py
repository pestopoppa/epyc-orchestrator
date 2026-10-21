#!/usr/bin/env python3
"""Web fetch tool for retrieving content from URLs.

Fetches web pages and extracts readable content, with support for:
- HTML to markdown conversion
- Content selectors for documentation sites
- Trust level checking via source registry
- Caching for repeated fetches
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from src.config import _registry_timeout
from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import safe_execute

logger = logging.getLogger(__name__)

# Default timeout from registry
_WEB_FETCH_TIMEOUT = int(_registry_timeout("tools", "web_fetch", 30))

# Simple cache for fetched content (URL -> content)
_fetch_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 900  # 15 minutes


def _get_project_root() -> Path:
    """Get project root from config with fallback."""
    import os

    try:
        from src.config import get_config

        root = get_config().paths.project_root
        if root.exists():
            return root
    except Exception:
        pass

    # Fallback: cwd
    return Path(os.getcwd())


def _load_source_registry() -> dict[str, Any]:
    """Load the source registry configuration."""
    registry_path = _get_project_root() / "orchestration" / "source_registry.yaml"
    if not registry_path.exists():
        return {}

    with open(registry_path) as f:
        return yaml.safe_load(f) or {}


def _get_domain_trust(url: str, source_registry: dict[str, Any]) -> int | None:
    """Get trust level for a domain from source registry.

    Args:
        url: URL to check.
        source_registry: Loaded source registry.

    Returns:
        Trust level (1-3) or None if not in registry.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # Remove www. prefix for matching
    if domain.startswith("www."):
        domain = domain[4:]

    # Search through all categories
    for category in ["coding", "research", "llm_specific", "general"]:
        cat_data = source_registry.get(category, {})
        for section in cat_data.values():
            if isinstance(section, list):
                for entry in section:
                    if isinstance(entry, dict) and entry.get("domain", "").lower() == domain:
                        return entry.get("trust")

    return None


def _extract_content(html: str, url: str) -> str:
    """Extract readable content from HTML.

    Args:
        html: Raw HTML content.
        url: Source URL (for site-specific extraction).

    Returns:
        Extracted text content.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: strip HTML tags with regex
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Try to find main content
    main_content = None
    for selector in ["article", "main", ".content", ".documentation", "#content", ".markdown-body"]:
        found = soup.select_one(selector)
        if found:
            main_content = found
            break

    if main_content is None:
        main_content = soup.body if soup.body else soup

    # Extract text
    text = main_content.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def _legacy_curl(url: str, timeout: int) -> str:
    """Pre-SEC-EXEC-3 fetch: curl handles gzip/deflate decompression and avoids bot detection."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "curl", "-s", "-L", "--max-time", str(timeout), "--compressed",
                "-H", "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "-H", "Accept: text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
                "-H", "Accept-Language: en-US,en;q=0.5",
                "-w", "\n__CONTENT_TYPE__:%{content_type}",
                url,
            ],
            capture_output=True,
            timeout=timeout + 5,
        )
        if result.returncode != 0:
            raise Exception(f"curl failed: {result.stderr.decode()[:200]}")
        return result.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        raise Exception(f"Fetch timed out after {timeout}s")
    except FileNotFoundError:
        raise Exception("curl not found")


_CURL_TRAILER = re.compile(r"\n__CONTENT_TYPE__:([^\n]*)\n__CODE__:(\d*)\n__REDIRECT__:([^\n]*)")


def _guarded_curl(url: str, timeout: int) -> str:
    """curl with every hop validated by ``web_guard`` and pinned via ``--resolve``.

    Returns the same ``body + "\\n__CONTENT_TYPE__:<type>"`` shape as the legacy path. Raises
    ``Exception("WEB_FETCH DENIED: ...")`` for a refused destination.
    """
    import subprocess

    from src.repl_environment import web_guard

    current = url
    for hop in range(web_guard.MAX_REDIRECTS + 1):
        try:
            host, port, ip = web_guard.resolve_pinned(current)
        except web_guard.WebGuardRefusal as refusal:
            reason = str(refusal) if hop == 0 else f"redirect hop {hop}: {refusal}"
            web_guard.log_refusal("fetch_docs", current, reason)
            raise Exception(f"WEB_FETCH DENIED: {reason}") from None
        addr = f"[{ip}]" if ":" in ip else ip
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "--max-time", str(timeout), "--compressed",
                    "--proto", "=http,https", "--noproxy", "*", "--max-redirs", "0",
                    "--resolve", f"{host}:{port}:{addr}",
                    "-H", "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                    "-H", "Accept: text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
                    "-H", "Accept-Language: en-US,en;q=0.5",
                    "-w", "\n__CONTENT_TYPE__:%{content_type}\n__CODE__:%{http_code}\n__REDIRECT__:%{redirect_url}",
                    current,
                ],
                capture_output=True,
                timeout=timeout + 5,
            )
        except subprocess.TimeoutExpired:
            raise Exception(f"Fetch timed out after {timeout}s") from None
        except FileNotFoundError:
            raise Exception("curl not found") from None
        if result.returncode != 0:
            raise Exception(f"curl failed: {result.stderr.decode()[:200]}")
        raw = result.stdout.decode("utf-8", errors="replace")
        cut = raw.rfind("\n__CONTENT_TYPE__:")
        m = _CURL_TRAILER.fullmatch(raw, cut) if cut >= 0 else None
        if m is None:
            raise Exception("curl output carried no status trailer")
        body, ctype, code, redirect = raw[:cut], m.group(1), m.group(2), m.group(3).strip()
        if code.startswith("3") and redirect:
            current = redirect
            continue
        return body + "\n__CONTENT_TYPE__:" + ctype
    reason = f"more than {web_guard.MAX_REDIRECTS} redirects"
    web_guard.log_refusal("fetch_docs", url, reason)
    raise Exception(f"WEB_FETCH DENIED: {reason}")


def _fetch_url(url: str, max_length: int = 8000, timeout: int = _WEB_FETCH_TIMEOUT) -> str:
    """Fetch content from a URL.

    Args:
        url: URL to fetch.
        max_length: Maximum content length to return.
        timeout: Request timeout in seconds.

    Returns:
        Extracted content from the URL.

    Raises:
        Exception: If fetch fails.
    """
    import time

    # Check cache
    cache_key = url
    if cache_key in _fetch_cache:
        content, cached_at = _fetch_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL_SECONDS:
            logger.debug(f"Cache hit for {url}")
            return content

    from src.repl_environment.exec_containment import flag_enabled

    if flag_enabled("web_fetch_guard"):
        # SEC-EXEC-3: private/loopback/link-local/metadata deny; redirects handled here, one
        # validated hop at a time, with curl pinned to the address that was validated.
        raw = _guarded_curl(url, timeout)
    else:
        raw = _legacy_curl(url, timeout)

    # Split content type from body (appended by -w flag)
    if "\n__CONTENT_TYPE__:" in raw:
        body, ct_line = raw.rsplit("\n__CONTENT_TYPE__:", 1)
        content_type = ct_line.strip()
    else:
        body = raw
        content_type = "text/html"

    if "text/html" in content_type:
        content = _extract_content(body, url)
    else:
        content = body

    # Truncate if needed
    if len(content) > max_length:
        content = content[:max_length] + f"\n\n[... truncated at {max_length} chars]"

    # Cache result
    _fetch_cache[cache_key] = (content, time.time())

    return content


def _audit_fetch(url: str, started: float, result: Any) -> None:
    """SEC-EXEC-4/7: one durable audit record per fetch_docs call, refused and failed included."""
    from src.repl_environment.exec_containment import audit_tool_call

    error = None if result.success else str(result.error or "")
    refused = error is not None and "WEB_FETCH DENIED" in error
    audit_tool_call(
        "fetch_docs", text=url, cwd=None, started=started,
        outcome="ok" if result.success else ("refused" if refused else "error"),
        refusal=error if refused else None,
    )


def fetch_docs(
    url: str,
    max_length: int = 8000,
    check_trust: bool = True,
) -> dict[str, Any]:
    """Fetch documentation from a URL.

    Args:
        url: URL to fetch.
        max_length: Maximum content length.
        check_trust: Whether to check source registry trust level.

    Returns:
        Dict with content, trust_level, and metadata.
    """
    # Load source registry for trust checking
    source_registry = _load_source_registry() if check_trust else {}
    trust_level = _get_domain_trust(url, source_registry)

    # Fetch content
    import time as _time

    _started = _time.time()
    result = safe_execute(
        _fetch_url,
        url,
        max_length=max_length,
        timeout_seconds=30,
    )
    _audit_fetch(url, _started, result)

    if not result.success:
        return {
            "success": False,
            "error": result.error,
            "url": url,
            "trust_level": trust_level,
        }

    return {
        "success": True,
        "content": result.data,
        "url": url,
        "trust_level": trust_level,
        "elapsed_ms": result.elapsed_ms,
        "truncated": result.truncated,
    }


def register_fetch_tool(registry: ToolRegistry) -> int:
    """Register the fetch_docs tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="fetch_docs",
        description="Fetch documentation or content from a URL",
        category=ToolCategory.WEB,
        parameters={
            "url": {
                "type": "string",
                "description": "URL to fetch content from",
                "required": True,
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum content length (default 8000)",
                "required": False,
            },
            "check_trust": {
                "type": "boolean",
                "description": "Check source registry for trust level",
                "required": False,
            },
        },
        handler=fetch_docs,
    )

    registry.register_tool(tool)
    return 1
