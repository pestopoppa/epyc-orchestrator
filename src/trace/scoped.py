"""Session-scoped read-only trace navigation shared by the MCP and REPL exposures (UTM-B1).

The trace store holds rows from every session. A model-facing tool must never read another
session's rows by default, so both operations take the CALLER's session id and refuse to run
without one. ``scope="all"`` (every session) is explicit and is honoured only while the
``trace_navigation_cross_session`` setting is on (default off). Expansion is filtered by the row's
own ``session_id`` as well, so exact event ids from another session do not resolve.

No database path, write, or embedding source is exposed; ``navigation`` is injected only by tests.
"""

from __future__ import annotations

from typing import Any, Iterable

SCOPES = ("session", "all")


def default_navigation() -> Any:
    from src.trace import navigation

    return navigation


def cross_session_allowed() -> bool:
    """The separate privacy setting; any failure to read it means off."""
    try:
        from src.features import features

        return bool(features().trace_navigation_cross_session)
    except Exception:  # noqa: BLE001 - unreadable == off
        return False


def _refuse(message: str) -> dict[str, Any]:
    return {"error": message, "records": []}


def _resolve(session_id: Any, scope: Any) -> tuple[str | None, dict[str, Any] | None]:
    """Return (session filter or None for all-scope, refusal or None)."""
    if scope not in SCOPES:
        return None, _refuse(f"scope must be one of {list(SCOPES)}")
    if scope == "all":
        if not cross_session_allowed():
            return None, _refuse("cross-session scope is disabled (trace_navigation_cross_session is off)")
        return None, None
    sid = session_id.strip() if isinstance(session_id, str) else ""
    if not sid:
        return None, _refuse("no caller session id; session-scoped trace navigation needs one")
    return sid, None


def search(text: str, session_id: Any, scope: Any = "session", *, navigation: Any | None = None) -> dict[str, Any]:
    sid, refusal = _resolve(session_id, scope)
    if refusal is not None:
        return refusal
    nav = navigation if navigation is not None else default_navigation()
    if sid is None:
        rows = nav.search_records(text)
    else:
        rows = [r for r in nav.search_records(text, session_id=sid) if r.get("session_id") == sid]
    return {"scope": scope, "records": rows}


def expand(event_ids: Iterable[int], session_id: Any, scope: Any = "session", *,
           navigation: Any | None = None) -> dict[str, Any]:
    sid, refusal = _resolve(session_id, scope)
    if refusal is not None:
        return refusal
    nav = navigation if navigation is not None else default_navigation()
    rows = nav.get_records(event_ids)
    if sid is not None:
        rows = [r for r in rows if r.get("session_id") == sid]
    return {"scope": scope, "records": rows}
