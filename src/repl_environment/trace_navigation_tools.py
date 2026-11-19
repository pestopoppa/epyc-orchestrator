"""UTM-B1: session-scoped read-only trace navigation as REPL tools (``ms_search`` / ``ms_expand``).

Gated by the ``trace_navigation_tools`` feature flag (default off). With it off nothing here is
installed and the REPL globals and tool registry are byte-identical to before.

Two surfaces, one implementation (``src.trace.scoped``):

* REPL globals ``ms_search(text, scope="session")`` / ``ms_expand(event_ids, scope="session")``,
  bound to bound methods of the environment (``_ms_search`` / ``_ms_expand``) so the per-snippet
  tool budget charges them and ``tool_broker.bridged_tools`` bridges them to the contained child
  lane automatically (the child never touches the trace store).
* Two ``Tool`` entries in a per-session clone of the tool registry, so ``list_tools()``,
  ``TOOL("ms_search", ...)``, the tool picker (``tool_selector``) and the typed codemode stubs
  (``codemode_stubs``) see them like any registry tool.

The caller's session is this REPL's own ``_session_id``; it is never a model argument.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from src.trace import scoped

NAMES = ("ms_search", "ms_expand")


def enabled() -> bool:
    try:
        from src.features import features

        return bool(features().trace_navigation_tools)
    except Exception:  # noqa: BLE001 - unreadable == off
        return False


def ms_search(env: Any, text: str, scope: str = "session") -> dict[str, Any]:
    """Search this session's trace store (FTS, ranked). ``scope="all"`` needs the cross-session setting."""
    return scoped.search(text, getattr(env, "_session_id", ""), scope, navigation=_navigation())


def ms_expand(env: Any, event_ids: list[int], scope: str = "session") -> dict[str, Any]:
    """Expand exact trace event ids of this session, in the order given."""
    return scoped.expand(event_ids, getattr(env, "_session_id", ""), scope, navigation=_navigation())


def _navigation() -> Any:
    return scoped.default_navigation()


def registry_with_trace_navigation(registry: Any, env: Any) -> Any:
    """Clone the tool map for this session and add the two tools (the shared registry is not mutated)."""
    from src.registry.tool_registry import Tool, ToolCategory, ToolRegistry

    if not isinstance(registry, ToolRegistry):
        return registry
    local = copy.copy(registry)
    local._tools = registry._tools.copy()
    local.register_tool(Tool(
        name="ms_search",
        description="Search THIS session's trace store (ranked event rows with ids); expand a row with ms_expand.",
        category=ToolCategory.DATA,
        parameters={"text": {"type": "string", "required": True}, "scope": {"type": "string"}},
        handler=lambda text, scope="session": json.dumps(ms_search(env, text, scope), ensure_ascii=False, sort_keys=True),
        side_effects=[],
        allowed_callers=["direct", "chain"],
    ), update=True)
    local.register_tool(Tool(
        name="ms_expand",
        description="Expand exact trace event ids (from ms_search) of THIS session, in the order given.",
        category=ToolCategory.DATA,
        parameters={"event_ids": {"type": "array", "required": True}, "scope": {"type": "string"}},
        handler=lambda event_ids, scope="session": json.dumps(ms_expand(env, event_ids, scope), ensure_ascii=False, sort_keys=True),
        side_effects=[],
        allowed_callers=["direct", "chain"],
    ), update=True)
    return local
