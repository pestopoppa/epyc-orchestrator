"""UFH14-LB3: the tightened turn-1 grammar shape and the replay classifier."""
from __future__ import annotations

import importlib.util
from pathlib import Path

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
