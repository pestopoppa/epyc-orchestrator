"""Thin model-facing registration for the existing read-only trace navigation API."""

from __future__ import annotations

import json
from typing import Any

from src.trace import scoped


def trace_navigation_tools_enabled() -> bool:
    """The ``trace_navigation_tools`` flag (default off); unreadable == off."""
    try:
        from src.features import features

        return bool(features().trace_navigation_tools)
    except Exception:  # noqa: BLE001
        return False


def register_trace_navigation_tools_if_enabled(server: Any, *, navigation: Any | None = None) -> bool:
    """Register the tools only while ``trace_navigation_tools`` is on. Returns whether it did.

    The stdio MCP server reads the flag once at startup; a runtime toggle reaches new servers and
    the REPL (which reads it per request), not an already-running MCP process.
    """
    if not trace_navigation_tools_enabled():
        return False
    register_trace_navigation_tools(server, navigation=navigation)
    return True


def register_trace_navigation_tools(server: Any, *, navigation: Any | None = None) -> None:
    """Register the two UTM-B1 lookup tools over the existing navigation API.

    ``session_id`` is the CALLER's session (the OpenCode plugin stamps it on every call) and is
    the default filter: rows of other sessions are never returned. ``scope="all"`` is refused
    unless the ``trace_navigation_cross_session`` setting is on. No database path, write
    operation, embedding source, or retrieval policy is a model-controlled argument.
    """

    @server.tool(name="ms.search")
    def ms_search(text: str, session_id: str = "", scope: str = "session") -> str:
        """Search this session's trace FTS store and return ranked event rows (scope="all" needs the cross-session setting)."""
        out = scoped.search(text, session_id, scope, navigation=navigation)
        return json.dumps(out, ensure_ascii=False, sort_keys=True)

    @server.tool(name="ms.expand")
    def ms_expand(event_ids: list[int], session_id: str = "", scope: str = "session") -> str:
        """Expand exact trace event IDs of this session in caller-supplied order (scope="all" needs the cross-session setting)."""
        out = scoped.expand(event_ids, session_id, scope, navigation=navigation)
        return json.dumps(out, ensure_ascii=False, sort_keys=True)
