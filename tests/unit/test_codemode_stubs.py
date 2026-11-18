"""Codemode reliability fix 4: ORCHESTRATOR_CODEMODE_STUBS=1 typed stubs + object-returning bound tools (default off)."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from src.prompt_builders import tool_selector as ts
from src.prompt_builders.codemode_stubs import ENV_STUBS, render_stub, render_stubs_block, stubs_enabled
from src.registry.tool_registry import ToolOutput
from src.repl_environment import codemode_tools as cm

TOOLS = [
    {"name": "syn_sku_price", "description": "Look up the current unit price in cents for a product SKU.",
     "parameters": {"sku": {"type": "string", "required": True}}, "returns": "dict", "returns_doc": '{"sku", "price_cents"}'},
    {"name": "list_dir", "description": "List directory contents.", "parameters": {"path": {"type": "string", "required": True}},
     "returns": "list[str]"},
    {"name": "peek", "description": "First n chars.", "parameters": {"n": {"type": "integer", "required": True},
                                                                  "file_path": {"type": "string", "required": False}}, "returns": "str"},
    {"name": "statistics", "description": "Stats.", "parameters": {"data": {"type": "array", "required": True},
                                                                  "metrics": {"type": "array", "required": False}}},
]


@pytest.fixture(autouse=True)
def _off(monkeypatch):
    monkeypatch.delenv(ENV_STUBS, raising=False)


def test_flag_parse(monkeypatch):
    assert not stubs_enabled({}) and stubs_enabled({ENV_STUBS: "1"}) and not stubs_enabled({ENV_STUBS: "0"})


def test_render_stub_types_returns_and_docstring():
    s = render_stub(TOOLS[0])
    assert s.startswith('def syn_sku_price(sku: str) -> dict:  # {"sku", "price_cents"}')
    assert '"""Look up the current unit price in cents for a product SKU."""' in s
    assert render_stub(TOOLS[1]).splitlines()[0] == "def list_dir(path: str) -> list[str]:"
    assert render_stub(TOOLS[2]).splitlines()[0] == "def peek(n: int, file_path: str = None) -> str:"
    assert render_stub(TOOLS[3]).splitlines()[0] == "def statistics(data: list, metrics: list = None) -> Any:"


def test_rendered_stub_block_is_valid_python_and_keeps_list_tools():
    block = render_stubs_block(TOOLS, ["syn_sku_price", "list_dir", "nope"])
    code = block.split("```python\n")[1].split("```")[0]
    import ast

    ast.parse(code)
    assert "nope" not in block and "list_tools()" in block and "FINAL(" in block and "no json.loads" in block


def test_selector_block_is_unchanged_with_flag_off_and_stubs_with_flag_on(monkeypatch):
    off = ts.render_tools_block(TOOLS, ["syn_sku_price"])
    assert "syn_sku_price(sku: string)" in off and "def " not in off
    monkeypatch.setenv(ENV_STUBS, "1")
    assert "def syn_sku_price(sku: str) -> dict" in ts.render_tools_block(TOOLS, ["syn_sku_price"])


def test_objectify_unwraps_cm3_envelope_and_parses_only_promised_objects():
    assert cm.objectify(ToolOutput(ok=True, output=[1, 2])) == [1, 2]
    with pytest.raises(cm.ToolCallError, match="denied"):
        cm.objectify(ToolOutput(ok=False, status="error", output="denied"))
    assert cm.objectify('["a", "b"]', "list[str]") == ["a", "b"]
    assert cm.objectify('["a", "b"]', "str") == '["a", "b"]'  # read_file style: text stays text
    assert cm.objectify('["a"]', None) == '["a"]'
    assert cm.objectify("not json [", "list[str]") == "not json ["
    assert cm.objectify(ToolOutput(ok=True, output='{"k": 1}'), "dict") == {"k": 1}


def test_bound_callable_maps_positionals_and_raises_clear_errors():
    seen = []

    def invoke(name, **kw):
        seen.append((name, kw))
        return json.dumps(["x.json"])

    g: dict = {"statistics": object()}
    bound = cm.bind_stub_tools(g, invoke, TOOLS, names={"list_dir", "peek"})
    assert sorted(bound) == ["list_dir", "peek"]
    assert g["list_dir"]("d") == ["x.json"] and seen[-1] == ("list_dir", {"path": "d"})
    g["peek"](5, "f")
    assert seen[-1] == ("peek", {"n": 5, "file_path": "f"})
    with pytest.raises(TypeError, match="missing required"):
        g["list_dir"]()
    with pytest.raises(TypeError, match="at most 1"):
        g["list_dir"]("a", "b")
    # existing globals are kept unless override=True
    assert cm.bind_stub_tools(g, invoke, TOOLS, names={"statistics"}) == []
    assert cm.bind_stub_tools(g, invoke, TOOLS, names={"statistics"}, override=True) == ["statistics"]


def test_install_binds_only_with_flag(monkeypatch):
    from types import SimpleNamespace

    env = SimpleNamespace(tool_registry=object(), _list_tools=lambda: TOOLS, _invoke_tool=lambda n, **k: "[]")
    g: dict = {}
    assert cm.install(env, g, {}) == [] and g == {}
    assert set(cm.install(env, g, {ENV_STUBS: "1"})) == {"syn_sku_price", "list_dir", "peek", "statistics"}


@pytest.fixture
def world():
    from scripts.benchmark.codemode_ab.world import World

    root = Path("/mnt/raid0/llm/tmp/codemode-rel-test") / uuid.uuid4().hex
    w = World(root, Path(__file__).resolve().parents[2])
    (w.root / "data").mkdir(parents=True)
    (w.root / "data" / "a.json").write_text('{"qty": 3}')
    yield w
    shutil.rmtree(root, ignore_errors=True)


def test_harness_repl_flag_off_keeps_v1_behaviour(world):
    from scripts.benchmark.codemode_ab.arms import make_repl

    env = make_repl(world, bind_names={"list_dir"})
    res = env.execute('x = list_dir("data")\nprint(type(x).__name__, peek(5))')
    assert res.error is None and "str" in res.output  # JSON text, and peek() with no file is silently ''


@pytest.mark.repl_in_process
def test_harness_repl_stubs_return_objects_and_peek_raises(monkeypatch, world):
    from scripts.benchmark.codemode_ab.arms import PEEK_NO_FILE_ERROR, make_repl

    monkeypatch.setenv(ENV_STUBS, "1")
    env = make_repl(world, bind_names={"list_dir", "syn_sku_price", "read_file"})
    res = env.execute('x = list_dir("data")\nprint(type(x).__name__, x)\nprint(syn_sku_price("SKU-1")["currency"])\n'
                      'print(json.loads(read_file("data/a.json"))["qty"])')
    assert res.error is None, res.error
    assert "list ['a.json']" in res.output and "USD" in res.output and "3" in res.output
    bad = env.execute("peek(10)")
    assert bad.error and "needs a file_path" in bad.error and PEEK_NO_FILE_ERROR.split(":")[0] in bad.error
    assert env.execute('print(peek(3, file_path="data/a.json"))').output.strip() == '{"q'
