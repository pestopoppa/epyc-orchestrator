"""Structural guards for the E8 r2 successor executor."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_successor.py"
FINALIZER_PATH = ROOT / "scripts/benchmark/finalize_e8_quality_baseline_v5_recovery_r2.py"
SPEC = importlib.util.spec_from_file_location("e8_r2_successor_test", PATH)
assert SPEC and SPEC.loader
SUCCESSOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUCCESSOR)


def test_existing_successor_namespace_refuses_before_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    monkeypatch.setattr(
        SUCCESSOR, "build_plan", lambda _source: pytest.fail("must not plan existing output")
    )
    with pytest.raises(FileExistsError, match="already exists"):
        SUCCESSOR.execute(SimpleNamespace(source_dir=source, output_dir=output))
    assert list(output.iterdir()) == []


def test_generation_is_not_nested_under_scorer_active_load() -> None:
    tree = ast.parse(PATH.read_text())
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    nested_calls = [
        call
        for with_node in ast.walk(execute)
        if isinstance(with_node, ast.With)
        and any(isinstance(item.context_expr, ast.Call) and getattr(item.context_expr.func, "attr", None) == "active_load" for item in with_node.items)
        for call in ast.walk(with_node)
        if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "_generate_with_watcher"
    ]
    assert nested_calls == []


def test_finalizer_dispatches_successor_before_legacy_watcher_requirement() -> None:
    source = FINALIZER_PATH.read_text()
    assert source.index('if plan.get("schema") == SUCCESSOR.PLAN_SCHEMA:') < source.index(
        'required["watcher"] = intermediate / "runtime_watch.r2.jsonl"'
    )


def test_finalizer_hashes_nested_failed_source_bindings() -> None:
    source = FINALIZER_PATH.read_text()
    assert 'item != required["failed_binding"]' in source


def test_successor_admission_uses_frontdoor_capacity_not_observer_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health probes may span all regions; only generated frontdoor traffic contends."""
    monkeypatch.setattr(
        SUCCESSOR.RECOVERY,
        "compatible_frontdoor_capacity",
        lambda _instances, _live, held: (3, [{"regions": ["q0"]}, {"regions": ["q1"]}, {"regions": ["q2"]}])
        if held == {"q3"}
        else pytest.fail("claim was not passed to frontdoor admission"),
    )
    from src.runtime import instance_topology

    monkeypatch.setattr(
        instance_topology,
        "get_instance_regions",
        lambda: {
            ("frontdoor", 1): frozenset({"q0"}),
            ("frontdoor", 2): frozenset({"q1"}),
            ("frontdoor", 3): frozenset({"q2"}),
            ("worker_general", 1): frozenset({"q0", "q1", "q2", "q3"}),
        },
    )
    monkeypatch.setattr(
        instance_topology,
        "topology_idx_for_port",
        lambda role, port: {("frontdoor", 8070): 1, ("frontdoor", 8080): 2, ("frontdoor", 8180): 3}.get((role, port)),
    )
    monkeypatch.setattr(SUCCESSOR.RECOVERY, "_locked_global_regions", lambda _regions: {"q3"})
    proof = SUCCESSOR.RECOVERY.preflight_frontdoor_capacity(
        {"runtime_topology": [{"port": port, "roles": ["frontdoor"]} for port in (8070, 8080, 8180)]},
        required=3,
        claim={"claims": [{"payload": {"region": "q3"}}]},
    )
    assert proof["capacity"] == 3
    assert proof["held_global_regions"] == ["q3"]
