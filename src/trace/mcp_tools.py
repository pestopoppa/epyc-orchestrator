"""Thin model-facing registration for the existing read-only trace navigation API."""

from __future__ import annotations

import json
import os
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


BOUND_SESSION_ENV = "ORCHESTRATOR_MCP_SESSION_ID"


def bound_session_from_env() -> str:
    """The server-side bound session: set by whoever LAUNCHES the server, never by a client call."""
    return os.environ.get(BOUND_SESSION_ENV, "").strip()


def _check_session(bound: str, supplied: Any) -> dict[str, Any] | None:
    """Fail closed: refusal when nothing is bound or the client names a different session."""
    if not bound:
        return {
            "error": f"no server-bound session ({BOUND_SESSION_ENV} is unset); trace navigation is unavailable",
            "records": [],
        }
    if isinstance(supplied, str) and supplied.strip() and supplied.strip() != bound:
        return {
            "error": "session_id does not match the session bound to this server; "
            "a client cannot choose its session",
            "records": [],
        }
    return None


def register_trace_navigation_tools(
    server: Any, *, navigation: Any | None = None, bound_session_id: str | None = None
) -> None:
    """Register the two UTM-B1 lookup tools over the existing navigation API.

    The session is bound SERVER-side (``bound_session_id``, default the launcher-set
    ``ORCHESTRATOR_MCP_SESSION_ID``, read once at registration). The client-supplied
    ``session_id`` argument is only a consistency check: a mismatch is refused, and with no bound
    session both tools return an error and no rows. ``scope="all"`` is additionally refused
    unless the ``trace_navigation_cross_session`` setting is on. No database path, write
    operation, embedding source, or retrieval policy is a model-controlled argument.
    """
    bound = (bound_session_id if bound_session_id is not None else bound_session_from_env()).strip()

    @server.tool(name="ms.search")
    def ms_search(text: str, session_id: str = "", scope: str = "session") -> str:
        """Search this server-bound session's trace FTS store and return ranked event rows (scope="all" needs the cross-session setting)."""
        out = _check_session(bound, session_id) or scoped.search(text, bound, scope, navigation=navigation)
        return json.dumps(out, ensure_ascii=False, sort_keys=True)

    @server.tool(name="ms.expand")
    def ms_expand(event_ids: list[int], session_id: str = "", scope: str = "session") -> str:
        """Expand exact trace event IDs of this server-bound session in caller-supplied order (scope="all" needs the cross-session setting)."""
        out = _check_session(bound, session_id) or scoped.expand(event_ids, bound, scope, navigation=navigation)
        return json.dumps(out, ensure_ascii=False, sort_keys=True)
