"""No default-off feature may detach a pre-existing production guard."""

from __future__ import annotations

import ast
from datetime import datetime
import functools
import json
from pathlib import Path
import subprocess
from typing import Any

from src.registry.artifact_integrity import TemplatePinError

ROOT = Path(__file__).resolve().parents[2]


def signatures(text):
    result = {}

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[name] = [
                        ast.unparse(x) for x in child.decorator_list
                    ]
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(ast.parse(text))
    return result


def test_every_preexisting_function_keeps_exact_decorators():
    baseline = json.loads((ROOT / "tests/fixtures/ev5_decorator_baseline.json").read_text())
    for filename, functions in baseline.items():
        actual = signatures((ROOT / filename).read_text())
        for name, decorators in functions.items():
            assert name in actual, (filename, name)
            assert actual[name] == decorators, (filename, name)
    stack = signatures((ROOT / "scripts/server/orchestrator_stack.py").read_text())
    assert stack["thinkprm_eval_config"] == []


def test_real_start_server_converts_artifact_refusal_without_spawning(
    tmp_path, monkeypatch, capsys
):
    # Execute the actual two function definitions, including the actual AST
    # decorator attachment, without importing fleet configuration or services.
    tree = ast.parse((ROOT / "scripts/server/orchestrator_stack.py").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_refuse_on_artifact_integrity_error", "start_server"}
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    calls = []

    def reject(*_args, **_kwargs):
        raise TemplatePinError("frontdoor", "file:/invalid-template", "a" * 64, "b" * 64)

    def forbidden_spawn(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("native spawn reached despite bad artifact")

    monkeypatch.setattr(subprocess, "Popen", forbidden_spawn)
    namespace = {
        "functools": functools,
        "json": json,
        "datetime": datetime,
        "subprocess": subprocess,
        "Any": Any,
        "LOG_DIR": tmp_path / "logs",
        "EMBEDDING_PLACEMENT": {},
        "validate_serving_shape_capacity": lambda: None,
        "_bench_guarded_numa_prefix": lambda *a, **k: [],
        "build_server_command": reject,
    }
    exec(compile(module, "actual_orchestrator_stack_guard", "exec"), namespace)
    start_server = namespace["start_server"]
    assert start_server.__wrapped__ is not None
    assert start_server(8070, ["frontdoor"], None, eval_batch_frontdoor_mode=True) is None
    assert not calls
    assert "[REFUSED] frontdoor" in capsys.readouterr().out
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs/artifact_integrity_refusals.jsonl").read_text().splitlines()
    ]
    assert rows[0]["kind"] == "TemplatePinError" and rows[0]["roles"] == ["frontdoor"]
