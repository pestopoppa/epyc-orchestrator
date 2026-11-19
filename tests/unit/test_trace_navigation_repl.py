"""UTM-B1 REPL exposure: ``ms_search`` / ``ms_expand`` behind ``trace_navigation_tools``.

Covers flag off/on, session isolation (the caller's session is the REPL's own ``_session_id``, never
a model argument), the cross-session setting, the shared tool broker / contained child lane, the
typed stub and the tool picker index.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from src.features import reset_features
from src.prompt_builders import codemode_stubs
from src.prompt_builders.tool_selector import lexical_scores
from src.registry.tool_registry import ToolCategory, ToolPermissions, ToolRegistry
from src.repl_environment import exec_sandbox, tool_broker
from src.repl_environment.environment import REPLEnvironment
from src.repl_environment.types import REPLConfig
from src.trace import navigation, scoped
from src.trace.store import Event, ensure_schema, upsert_events

FLAG = "ORCHESTRATOR_TRACE_NAVIGATION_TOOLS"
CROSS = "ORCHESTRATOR_TRACE_NAVIGATION_CROSS_SESSION"
_CAPS = exec_sandbox.capabilities()
needs_kernel = pytest.mark.skipif(
    not (_CAPS["landlock"] and _CAPS["netdeny"]), reason="host cannot enforce Landlock+seccomp (the lane refuses to run)"
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(tmp_path / "no_runtime_flags.json"))
    for k in list(os.environ):
        if k.startswith("ORCHESTRATOR_REPL_CHILD"):
            monkeypatch.delenv(k, raising=False)
    for k in (FLAG, CROSS, "ORCHESTRATOR_REPL_CHILD_LANE", "ORCHESTRATOR_CODEMODE_STUBS", "ORCHESTRATOR_QUERY_MEMORY"):
        monkeypatch.delenv(k, raising=False)
    reset_features()
    yield
    reset_features()


def _on(monkeypatch, *names):
    for n in names:
        monkeypatch.setenv(n, "1")
    reset_features()


def _event(session, line, summary):
    return Event(
        ts_utc=f"2026-10-07T00:00:{line:02d}+00:00", source="synthetic", source_path="s.jsonl", source_line=line,
        session_id=session, category="observe", summary=summary, detail_json=json.dumps({"n": line}),
    )


@pytest.fixture
def trace_db(monkeypatch, tmp_path):
    db = tmp_path / "trace.sqlite"
    conn = ensure_schema(db)
    try:
        upsert_events(conn, [
            _event("sess-a", 1, "needle alpha"), _event("sess-a", 2, "needle beta"),
            _event("sess-b", 3, "needle secret of b"),
        ])
    finally:
        conn.close()
    nav = SimpleNamespace(
        search_records=lambda text, **f: navigation.search_records(text, db_path=db, **f),
        get_records=lambda ids: navigation.get_records(ids, db_path=db),
    )
    monkeypatch.setattr(scoped, "default_navigation", lambda: nav)
    return nav


@pytest.fixture
def envs():
    made = []

    def make(session="sess-a", registry=True, **kw):
        reg = ToolRegistry()
        # production roles carry the ``data`` category (orchestration/model_registry.yaml tool_permissions)
        reg._permissions["worker_general"] = ToolPermissions(allowed_categories=[ToolCategory.DATA])
        e = REPLEnvironment(context="ctx", config=REPLConfig(timeout_seconds=20), task_id=session,
                            tool_registry=reg if registry else None, **kw)
        made.append(e)
        return e

    yield make
    for e in made:
        e.close_child_lane()


def test_flag_off_installs_nothing(envs, trace_db):
    env = envs()
    assert "ms_search" not in env._globals and "ms_expand" not in env._globals
    assert "ms_search" not in {t["name"] for t in env._list_tools()}


def test_flag_on_installs_globals_and_registry_tools(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    env = envs()
    assert {"ms_search", "ms_expand"} <= set(env._globals)
    assert {"ms_search", "ms_expand"} <= {t["name"] for t in env._list_tools()}
    assert {"ms_search", "ms_expand"} <= tool_broker.exposed_surface(env)


def test_shared_registry_is_not_mutated(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    shared = ToolRegistry()
    env = REPLEnvironment(context="ctx", config=REPLConfig(timeout_seconds=20), task_id="sess-a", tool_registry=shared)
    try:
        assert "ms_search" in env.tool_registry._tools and "ms_search" not in shared._tools
    finally:
        env.close_child_lane()


def test_session_isolation_in_process(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    a, b = envs("sess-a"), envs("sess-b")
    ra = a.execute("r = ms_search('needle')\nprint(sorted({x['session_id'] for x in r['records']}), len(r['records']))")
    rb = b.execute("r = ms_search('needle')\nprint(sorted({x['session_id'] for x in r['records']}), len(r['records']))")
    assert ra.error is None and "['sess-a'] 2" in ra.output
    assert rb.error is None and "['sess-b'] 1" in rb.output
    # sess-b's exact id does not resolve for sess-a
    b_id = b._ms_search("needle")["records"][0]["id"]
    assert a._ms_expand([b_id])["records"] == []
    assert [r["id"] for r in b._ms_expand([b_id])["records"]] == [b_id]


def test_session_is_not_a_model_argument(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    env = envs("sess-a")
    r = env.execute("ms_search('needle', session_id='sess-b')")
    assert r.error is not None or "sess-b" not in (r.output or "").replace("session_id='sess-b'", "")


def test_all_scope_needs_the_separate_setting(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    env = envs("sess-a")
    refused = env._ms_search("needle", "all")
    assert refused["records"] == [] and "cross-session" in refused["error"]
    assert env._ms_expand([1, 2, 3], "all")["records"] == []
    _on(monkeypatch, FLAG, CROSS)
    rows = env._ms_search("needle", "all")["records"]
    assert {r["session_id"] for r in rows} == {"sess-a", "sess-b"}
    assert {r["session_id"] for r in env._ms_search("needle")["records"]} == {"sess-a"}


def test_registry_route_is_session_scoped(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    env = envs("sess-b")
    out = json.loads(env._invoke_tool("ms_search", text="needle"))
    assert {r["session_id"] for r in out["records"]} == {"sess-b"}


def test_tool_picker_can_index_and_stubs_type_them(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    env = envs()
    tools = env._list_tools()
    scores = lexical_scores("search the trace history for an earlier event", tools)
    assert scores["ms_search"] > 0
    stub = codemode_stubs.render_stub(next(t for t in tools if t["name"] == "ms_expand"))
    assert stub.startswith("def ms_expand(event_ids: list") and "scope: str" in stub


def test_snippet_tool_budget_charges_the_tools(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "1")
    env = envs()
    r = env.execute("ms_search('needle')\nms_search('needle')\nms_search('needle')")
    assert r.error is not None or "budget" in (r.output or "").lower() or "limit" in (r.output or "").lower()


@needs_kernel
def test_child_lane_gets_the_tools_and_stays_session_scoped(monkeypatch, envs, trace_db):
    _on(monkeypatch, FLAG)
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_LANE", "1")
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP", "off")
    reset_features()
    a, b = envs("sess-a"), envs("sess-b")
    assert {"ms_search", "ms_expand"} <= set(tool_broker.bridged_tools(a))
    code = ("r = ms_search('needle')\n"
            "ids = [x['id'] for x in r['records']]\n"
            "e = ms_expand(ids)\n"
            "print(sorted({x['session_id'] for x in e['records']}), len(e['records']))\n"
            "print(ms_search('needle', 'all')['error'][:15])")
    ra, rb = a.execute(code), b.execute(code)
    assert ra.error is None, ra.error
    assert "['sess-a'] 2" in ra.output and "cross-session" in ra.output
    assert rb.error is None and "['sess-b'] 1" in rb.output


@needs_kernel
def test_child_lane_flag_off_has_no_tools(monkeypatch, envs, trace_db):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_LANE", "1")
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP", "off")
    reset_features()
    env = envs()
    assert "ms_search" not in tool_broker.bridged_tools(env)
    r = env.execute("print(ms_search('needle'))")
    assert r.error is not None
