"""Regression for the turn-1 grammar being silently disabled (UFH14-LB3 follow-up, 2026-10-22)."""

from __future__ import annotations

import ast
from pathlib import Path


def test_execute_turn_has_no_function_local_import_os():
    """Regression: a local ``import os`` inside ``_execute_turn`` (since 2026-03-03) made ``os`` a function local, so the turn-1
    grammar block's ``os.environ.get`` raised UnboundLocalError on every first turn and the blanket ``except`` swallowed it: the
    turn-1 GBNF (and therefore every ORCHESTRATOR_TURN1_GRAMMAR arm) never applied after UFH14-LB3 (517e24d4)."""
    import src.graph.helpers as h

    tree = ast.parse(Path(h.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_execute_turn")
    local = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Import) and any(a.name == "os" for a in n.names)]
    assert not local, f"function-local `import os` in _execute_turn at lines {local}"


def test_turn1_grammar_default_is_off_and_opt_in_shapes_remain():
    """Operator decision 2026-10-23 (turn1-grammar-4arm: off 66.7% vs legacy 40.0% pass): with
    ORCHESTRATOR_TURN1_GRAMMAR unset, no grammar is sent on turn 1; legacy/call stay opt-in."""
    import src.graph.helpers as h

    src = Path(h.__file__).read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_execute_turn")
    defaults = [
        c.args[1].value
        for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "get"
        and len(c.args) > 1
        and isinstance(c.args[0], ast.Constant)
        and c.args[0].value == "ORCHESTRATOR_TURN1_GRAMMAR"
    ]
    assert defaults == ["off"]
    # grammar is attached only when the resolved shape is not "off"; both opt-in shapes remain wired
    assert '_turn1_shape != "off"' in src
    assert 'shape="call" if _turn1_shape == "call" else "legacy"' in src
