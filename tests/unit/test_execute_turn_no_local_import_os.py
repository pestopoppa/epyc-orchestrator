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
