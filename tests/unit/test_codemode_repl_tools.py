"""codemode CM-2..CM-5: all default OFF; each flagged function behaves as specified."""

from __future__ import annotations

import pytest

from src.repl_environment import codemode_tools as cm

TOOLS = [
    {"name": "syn_sku_price", "description": "Look up unit price for a product SKU", "parameters": {"sku": {"type": "string"}}},
    {"name": "read_file", "description": "Read a file from disk", "parameters": {"path": {"type": "string"}}},
    {"name": "syn_weather", "description": "Weather forecast for a city", "parameters": {"city": {"type": "string"}}},
]
FLAGS = (cm.ENV_SEARCH, cm.ENV_CLASSIFY, cm.ENV_HANDLES, "ORCHESTRATOR_STRUCTURED_READ_ONLY")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    # CM-5 makes replay recording default-on with handles; these tests must not write the real trace store / CAS
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_RECORDING", "0")


# CM-2
def test_search_tools_detail_levels():
    assert cm.search_tools(TOOLS, "price of a sku")[0] == "syn_sku_price"
    brief = cm.search_tools(TOOLS, "price of a sku", "brief")
    assert set(brief[0]) == {"name", "description"}
    assert "parameters" in cm.search_tools(TOOLS, "price of a sku", "full")[0]
    assert cm.search_tools(TOOLS, "zzzz") == []
    with pytest.raises(ValueError):
        cm.search_tools(TOOLS, "x", "everything")


def test_describe_tool():
    assert cm.describe_tool(TOOLS, "read_file")["name"] == "read_file"
    with pytest.raises(KeyError):
        cm.describe_tool(TOOLS, "nope")


# CM-5
def test_handle_store_caps_and_rollback():
    h = cm.HandleStore(max_value=50, max_total=80)
    h.begin_cell()
    assert h.store("a", [1, 2, 3]) == "handle:a"
    assert h.load("handle:a") == [1, 2, 3]  # visible inside the cell
    h.rollback()
    with pytest.raises(KeyError):
        h.load("a")  # failed cell leaves nothing
    h.begin_cell()
    h.store("a", "x" * 30)
    h.commit()
    assert h.handles() == {"a": 32}
    with pytest.raises(ValueError):
        h.store("b", "y" * 60)  # per-value cap
    h.begin_cell()
    h.store("b", "y" * 40)
    with pytest.raises(ValueError):
        h.store("c", "z" * 30)  # total cap
    with pytest.raises(ValueError):
        h.store("bad name", 1)
    with pytest.raises(ValueError):
        h.store("d", {1, 2})  # not JSON


def test_handle_store_snapshot_replays():
    h = cm.HandleStore()
    h.begin_cell(); h.store("k", {"v": 1}); h.commit()
    g = cm.HandleStore()
    g.restore(h.snapshot())
    assert g.load("k") == {"v": 1}


class _Env:
    tool_registry = object()
    llm_primitives = None

    def _list_tools(self):
        return TOOLS


def test_install_default_off_adds_nothing():
    g: dict = {}
    assert cm.install(_Env(), g) == [] and g == {}


def test_install_flags(monkeypatch):
    monkeypatch.setenv(cm.ENV_SEARCH, "1")
    monkeypatch.setenv(cm.ENV_HANDLES, "1")
    g: dict = {}
    env = _Env()
    assert set(cm.install(env, g)) == {"search_tools", "describe_tool", "store", "load", "handles"}
    assert g["search_tools"]("weather forecast")[0] == "syn_weather"
    assert g["describe_tool"]("read_file")["name"] == "read_file"


@pytest.mark.repl_in_process
def test_environment_integration(monkeypatch):
    from src.repl_environment.environment import REPLEnvironment

    monkeypatch.setenv(cm.ENV_HANDLES, "1")
    env = REPLEnvironment(context="c")
    ok = env.execute("h = store('a', [1,2])\nprint(h)")
    assert not ok.error and env._handle_store.load("a") == [1, 2]
    bad = env.execute("store('b', 1)\nraise ValueError('boom')")
    assert bad.error and "b" not in env._handle_store.handles()
    cp = env.checkpoint()
    assert cp["handle_store"] == {"a": "[1, 2]"}
    env2 = REPLEnvironment(context="c")
    env2.restore(cp)  # flag on => globals expose load
    assert env2.execute("print(load('a'))").output.strip() == "[1, 2]"


def test_environment_default_off_has_no_new_globals():
    from src.repl_environment.environment import REPLEnvironment

    env = REPLEnvironment(context="c")
    assert not ({"store", "load", "handles", "search_tools", "describe_tool", "classify"} & set(env._globals))
    assert "handle_store" not in env.checkpoint()


# CM-4
class _Prim:
    usage_total: dict = {}

    def llm_call(self, prompt, **kw):
        lab = "spam" if "buy now" in prompt else "ham"
        return '{"answers": {"label": {"choice": "%s"}}}' % lab


def test_classify_items():
    out = cm.classify_items(_Prim(), ["buy now!!", "lunch at noon", "buy now"], ["spam", "ham"])
    assert out == ["spam", "ham", "spam"]
    with pytest.raises(ValueError):
        cm.classify_items(_Prim(), ["x"], ["only"])
    with pytest.raises(ValueError):
        cm.classify_items(_Prim(), ["x"] * 3, ["a", "b"], max_items=2)


# CM-3
def test_read_only_tools_never_return_bare_string(monkeypatch):
    from src.registry.tool_registry import SideEffect, Tool, ToolCategory, ToolRegistry

    def mk(name, side):
        return Tool(name=name, description="d", category=ToolCategory.DATA, parameters={"n": {"type": "integer"}}, handler=lambda n=1: "x" * n,
                    side_effects=side)

    reg = ToolRegistry.__new__(ToolRegistry)
    reg._tools = {"ro": mk("ro", [SideEffect.READ_ONLY]), "rw": mk("rw", [SideEffect.MODIFIES_FILES])}
    reg._invocations = []
    called = {}
    monkeypatch.setattr(reg, "can_use_tool", lambda *a, **k: True)
    monkeypatch.setattr(reg, "_record_invocation", lambda inv: called.setdefault("n", 0))
    r = reg.invoke("ro", "worker_general", n=3)
    assert isinstance(r, str)  # flag off: today's behaviour
    monkeypatch.setenv("ORCHESTRATOR_STRUCTURED_READ_ONLY", "1")
    r = reg.invoke("ro", "worker_general", n=3)
    assert r.ok and r.output == "xxx"
    assert isinstance(reg.invoke("rw", "worker_general", n=3), str)  # write tools unchanged
    rep = cm.check_shape_consistency(lambda n=1: reg.invoke("ro", "worker_general", n=n), lambda n: {"n": n}, 5, 1000)
    assert rep["consistent"] and not rep["bare_string"]


def test_shape_check_catches_bare_string_flip():
    rep = cm.check_shape_consistency(lambda n: [1] * n if n < 100 else "too long: see file", lambda n: {"n": n})
    assert not rep["consistent"] and rep["bare_string"]
