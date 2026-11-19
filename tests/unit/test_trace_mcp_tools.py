"""Off-host FastMCP controls over a temporary synthetic trace database (UTM-B1).

Pins: the tools exist only behind the ``trace_navigation_tools`` flag, every call is scoped to the
caller's own session, and ``scope="all"`` works only while ``trace_navigation_cross_session`` is on.
"""

import asyncio
import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client, FastMCP

from src.features import reset_features
from src.trace import navigation
from src.trace.store import Event, ensure_schema, upsert_events
from src.trace.mcp_tools import (
    register_trace_navigation_tools,
    register_trace_navigation_tools_if_enabled,
)

FLAG = "ORCHESTRATOR_TRACE_NAVIGATION_TOOLS"
CROSS = "ORCHESTRATOR_TRACE_NAVIGATION_CROSS_SESSION"


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(tmp_path / "no_runtime_flags.json"))
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(CROSS, raising=False)
    reset_features()
    yield
    reset_features()


def _on(monkeypatch, *names):
    for n in names:
        monkeypatch.setenv(n, "1")
    reset_features()


def _run(coro):
    return asyncio.run(coro)


def _event(session, line, summary):
    return Event(
        ts_utc=f"2026-10-07T00:00:{line:02d}+00:00",
        source="synthetic",
        source_path="synthetic-source.jsonl",
        source_line=line,
        session_id=session,
        category="observe",
        summary=summary,
        detail_json=json.dumps({"private": f"payload {line}"}),
    )


def _synthetic_navigation(tmp_path):
    db_path = tmp_path / "synthetic-trace.sqlite"
    conn = ensure_schema(db_path)
    try:
        upsert_events(conn, [
            _event("sess-a", 1, "needle first synthetic event"),
            _event("sess-a", 2, "needle second synthetic event"),
            _event("sess-b", 3, "needle secret of the other session"),
        ])
    finally:
        conn.close()
    return SimpleNamespace(
        search_records=lambda text, **filters: navigation.search_records(text, db_path=db_path, **filters),
        get_records=lambda ids: navigation.get_records(ids, db_path=db_path),
    )


def _server(tmp_path):
    server = FastMCP("synthetic-trace-navigation")
    register_trace_navigation_tools(server, navigation=_synthetic_navigation(tmp_path))
    return server


def _call(server, tool, args):
    async def _go():
        async with Client(server) as client:
            res = await client.call_tool(tool, args)
            return json.loads(res.content[0].text)

    return _run(_go())


async def _tool_names(server):
    async with Client(server) as client:
        return {t.name for t in await client.list_tools()}


def test_default_orchestrator_catalog_registers_only_through_the_flag_gate():
    repository = Path(__file__).resolve().parents[2]
    source = ast.parse((repository / "src/mcp_server.py").read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(source)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "register_trace_navigation_tools_if_enabled" in called
    assert "register_trace_navigation_tools" not in called  # never unconditionally


def test_flag_off_means_not_registered(tmp_path):
    server = FastMCP("flag-off")
    assert register_trace_navigation_tools_if_enabled(server, navigation=_synthetic_navigation(tmp_path)) is False
    assert _run(_tool_names(server)) == set()


def test_flag_on_means_registered(monkeypatch, tmp_path):
    _on(monkeypatch, FLAG)
    server = FastMCP("flag-on")
    assert register_trace_navigation_tools_if_enabled(server, navigation=_synthetic_navigation(tmp_path)) is True
    assert _run(_tool_names(server)) == {"ms.search", "ms.expand"}


def test_orchestrator_server_catalog_follows_the_flag(monkeypatch):
    import src.mcp_server as mcp_server

    try:
        reset_features()
        mcp_server = importlib.reload(mcp_server)
        assert not ({"ms.search", "ms.expand"} & _run(_tool_names(mcp_server.mcp)))
        _on(monkeypatch, FLAG)
        mcp_server = importlib.reload(mcp_server)
        assert {"ms.search", "ms.expand"} <= _run(_tool_names(mcp_server.mcp))
    finally:
        monkeypatch.delenv(FLAG, raising=False)
        reset_features()
        importlib.reload(mcp_server)


def test_registration_exposes_dotted_names_and_exact_optional_schemas(tmp_path):
    server = _server(tmp_path)

    async def _schemas():
        async with Client(server) as client:
            return {tool.name: tool.inputSchema for tool in await client.list_tools()}

    schemas = _run(_schemas())

    assert set(schemas) == {"ms.search", "ms.expand"}
    search = schemas["ms.search"]
    assert search["required"] == ["text"]
    assert search["properties"]["text"]["type"] == "string"
    assert search["properties"]["session_id"]["default"] == ""
    assert search["properties"]["scope"]["default"] == "session"
    assert search["additionalProperties"] is False
    expand = schemas["ms.expand"]
    assert expand["required"] == ["event_ids"]
    assert expand["properties"]["event_ids"] == {"items": {"type": "integer"}, "type": "array"}
    assert expand["properties"]["session_id"]["default"] == ""
    assert expand["properties"]["scope"]["default"] == "session"
    assert expand["additionalProperties"] is False


def test_search_and_expand_return_only_the_callers_session(tmp_path):
    server = _server(tmp_path)
    a = _call(server, "ms.search", {"text": "needle", "session_id": "sess-a"})["records"]
    b = _call(server, "ms.search", {"text": "needle", "session_id": "sess-b"})["records"]
    assert len(a) == 2 and {r["session_id"] for r in a} == {"sess-a"}
    assert len(b) == 1 and b[0]["session_id"] == "sess-b"
    ids_a = [r["id"] for r in a]
    expanded = _call(server, "ms.expand", {"event_ids": ids_a[::-1], "session_id": "sess-a"})["records"]
    assert [r["id"] for r in expanded] == ids_a[::-1]
    # exact ids of ANOTHER session do not resolve for this caller
    leak = _call(server, "ms.expand", {"event_ids": [b[0]["id"]] + ids_a, "session_id": "sess-a"})["records"]
    assert b[0]["id"] not in [r["id"] for r in leak]
    assert "secret" not in json.dumps(leak)


def test_missing_session_id_returns_nothing_and_does_not_leak(tmp_path):
    server = _server(tmp_path)
    out = _call(server, "ms.search", {"text": "needle"})
    assert out["records"] == [] and "session" in out["error"]
    out = _call(server, "ms.expand", {"event_ids": [1, 2, 3]})
    assert out["records"] == [] and "error" in out


def test_all_scope_is_refused_while_the_setting_is_off(monkeypatch, tmp_path):
    _on(monkeypatch, FLAG)  # the tool flag alone does not open cross-session reads
    server = _server(tmp_path)
    out = _call(server, "ms.search", {"text": "needle", "session_id": "sess-a", "scope": "all"})
    assert out["records"] == [] and "cross-session" in out["error"]
    out = _call(server, "ms.expand", {"event_ids": [1, 2, 3], "session_id": "sess-a", "scope": "all"})
    assert out["records"] == [] and "cross-session" in out["error"]


def test_all_scope_works_only_with_the_setting_on_and_is_explicit(monkeypatch, tmp_path):
    _on(monkeypatch, FLAG, CROSS)
    server = _server(tmp_path)
    allrows = _call(server, "ms.search", {"text": "needle", "session_id": "sess-a", "scope": "all"})["records"]
    assert {r["session_id"] for r in allrows} == {"sess-a", "sess-b"}
    expanded = _call(server, "ms.expand", {"event_ids": [r["id"] for r in allrows], "scope": "all"})["records"]
    assert len(expanded) == 3
    # the default stays session-scoped even with the setting on
    default = _call(server, "ms.search", {"text": "needle", "session_id": "sess-a"})["records"]
    assert {r["session_id"] for r in default} == {"sess-a"}


def test_unknown_scope_is_refused(tmp_path):
    out = _call(_server(tmp_path), "ms.search", {"text": "needle", "session_id": "sess-a", "scope": "everything"})
    assert out["records"] == [] and "scope" in out["error"]
