"""UFH14-LB3: the tightened turn-1 grammar shape and the replay classifier."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from pathlib import Path as _P  # noqa: F401
from src.registry.tool_registry import ToolRegistry

_spec = importlib.util.spec_from_file_location(
    "lb3", Path(__file__).resolve().parents[2] / "scripts/analysis/ufh14_lb3_turn1_grammar_replay.py"
)
lb3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb3)


def _registry_with_tools():
    from src.registry.tool_registry import Tool, ToolCategory

    reg = ToolRegistry()
    reg.register_tool(Tool(name="web_search", description="d", category=ToolCategory.WEB,
                           parameters={"query": {"type": "string"}}))
    return reg


def test_legacy_shape_is_unchanged_and_admits_any_line():
    g = _registry_with_tools().generate_gbnf_grammar(None)
    assert g == _registry_with_tools().generate_gbnf_grammar(None, shape="legacy")
    assert "tool-name" in g
    assert "action ::= tool-call | final-call | code-line" in g
    assert 'code-line ::= [^\\n]+ "\\n"' in g


def test_call_shape_drops_the_any_line_escape_and_uses_call():
    g = _registry_with_tools().generate_gbnf_grammar(None, shape="call")
    assert "tool-name" in g
    assert "code-line" not in g and "TOOL(" not in g
    assert "action ::= call-stmt | final-call" in g
    assert 'CALL(' in g


def test_classifier():
    c = lb3.classify_turn1
    assert c('# plan\nresult = CALL("web_search", query="x")\n') == "tool-call"
    assert c('# plan\nTOOL("web_search")\n') == "tool-call"
    assert c('# done\nFINAL("42")\n') == "final-call"
    assert c("# thinking\nx = 1 + 2\n") == "code-line"
    assert c("# only a comment\n") == "empty"


# ---- GRAM-LAZY-1 / UFH14-LB3.1b: lazy-legacy payload shape ----------------------------------
def test_lazy_legacy_grammar_starts_at_the_call_and_has_no_escape():
    g = _registry_with_tools().generate_gbnf_grammar(None, shape="lazy-legacy")
    assert g.startswith("root ::= tool-call | final-call\n")
    assert "thought" not in g and "code-line" not in g
    assert 'tool-call ::= "TOOL(\\"" tool-name' in g and 'final-call ::= "FINAL("' in g
    # legacy stays byte-identical (production default untouched)
    legacy = _registry_with_tools().generate_gbnf_grammar(None)
    assert "action ::= tool-call | final-call | code-line" in legacy


def test_lazy_grammar_wire_fields():
    import re

    from src.backends.grammar_guard import LazyGrammar, apply_grammar_fields
    from src.registry.tool_registry import LAZY_LEGACY_TRIGGERS

    g = LazyGrammar("root ::= x", LAZY_LEGACY_TRIGGERS)
    assert isinstance(g, str) and g == "root ::= x"
    p: dict = {}
    apply_grammar_fields(p, g)
    assert p["grammar"] == "root ::= x" and type(p["grammar"]) is str
    assert p["grammar_lazy"] is True
    trig = p["grammar_triggers"]
    assert trig and all(set(t) == {"type", "value"} and t["type"] == 2 for t in trig)
    rx = re.compile(trig[0]["value"])
    assert rx.search("# plan\nTOOL(\"web_search\")").group(1) == "TOOL("
    assert rx.search("FINAL('a')").group(1) == "FINAL("
    assert rx.search("# I will FINAL( later\nx = 1") is None  # mid-line mention does not fire
    # plain str grammar: eager, no lazy fields (legacy/call/off wire unchanged)
    q: dict = {}
    apply_grammar_fields(q, "root ::= x")
    assert q == {"grammar": "root ::= x"}
    apply_grammar_fields(q, None)
    assert q == {"grammar": "root ::= x"}


def test_lazy_grammar_still_passes_the_llguidance_guard():
    import pytest
    from fastapi import HTTPException

    from src.backends.grammar_guard import LazyGrammar, apply_grammar_fields

    with pytest.raises(HTTPException):
        apply_grammar_fields({}, LazyGrammar("%llguidance {}", [{"type": 2, "value": "x"}]))


def test_helpers_selector_records_lazy_legacy(monkeypatch):
    import src.graph.helpers as h

    src = Path(h.__file__).read_text()
    assert '"lazy-legacy"' in src and "state.turn1_grammar_shape = _turn1_shape" in src


def test_execute_turn_has_no_function_local_import_os():
    """Regression: a local ``import os`` inside ``_execute_turn`` (since 2026-03-03) made ``os`` a function local, so the turn-1
    grammar block's ``os.environ.get`` raised UnboundLocalError on every first turn and the blanket ``except`` swallowed it: the
    turn-1 GBNF (and therefore every ORCHESTRATOR_TURN1_GRAMMAR arm) never applied after UFH14-LB3 (517e24d4)."""
    import ast

    import src.graph.helpers as h

    tree = ast.parse(Path(h.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_execute_turn")
    local = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Import) and any(a.name == "os" for a in n.names)]
    assert not local, f"function-local `import os` in _execute_turn at lines {local}"
