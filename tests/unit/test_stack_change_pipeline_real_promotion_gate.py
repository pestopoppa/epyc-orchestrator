"""End-to-end subprocess controls for the promotion gate over temp stack worlds.

This module is intentionally outside ``PROMOTION_GATE_TARGETS``: its positive control launches
that exact suite, so including itself would recurse.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

from scripts.registry import stack_change_pipeline as pipeline
from scripts.registry.stack_change_pipeline import (
    StackChangePipelineConfig,
    run_stack_change_pipeline,
)


def _scenario_helpers():
    return importlib.import_module("tests.unit.test_stack_change_pipeline_simulated_fixtures")


def test_real_promotion_gate_accepts_each_approved_swapped_temporary_world(tmp_path: Path) -> None:
    scenario = _scenario_helpers()
    worlds = (
        ("frontdoor", {"frontdoor", "worker_summarize"},
         scenario._base_frontdoor_registry, scenario._swapped_frontdoor_registry),
        ("worker", scenario.WORKER_LANE_ROLES,
         scenario._worker_alias_registry, scenario._swapped_worker_registry),
        ("vision", {"worker_vision", "vision_escalation"},
         scenario._vision_registry, scenario._swapped_vision_registry),
        ("ingest", scenario.INGEST_PROCESS_ROLES,
         scenario._ingest_registry, scenario._swapped_ingest_registry),
    )
    for name, roles, prepare_base, prepare_swap in worlds:
        world = tmp_path / name
        world.mkdir()
        config = scenario._config(world, mode="update", roles=set(roles))
        prepare_base(config.lean_registry)
        assert run_stack_change_pipeline(config).ok

        prepare_swap(config.lean_registry)
        approved = StackChangePipelineConfig(
            **{**config.__dict__, "allow_descriptor_model_removal": True}
        )
        assert run_stack_change_pipeline(approved).ok

        checked = StackChangePipelineConfig(
            **{**approved.__dict__, "mode": "check", "run_promotion_gate": True}
        )
        report = run_stack_change_pipeline(checked)
        gate = next(step for step in report.steps if step.name == "promotion_gate")
        assert report.ok, name
        assert gate.status == "ok", name
        command = pipeline._promotion_gate_command()
        assert command[:3] == [sys.executable, "-m", "pytest"]
        assert command[3:8] == ["-q", "-c", str(pipeline.REPO_ROOT / "pyproject.toml"),
                               f"--rootdir={pipeline.REPO_ROOT}", "-o"]
        assert command[8:10] == ["addopts=", "-p"]
        assert command[10] == "no:cacheprovider"
        assert str(pipeline.REPO_ROOT / scenario.SIMULATED_FIXTURE_TARGET) in command


def test_real_promotion_gate_does_not_run_after_a_bad_temporary_world(tmp_path: Path) -> None:
    scenario = _scenario_helpers()
    config = scenario._config(tmp_path, mode="update", roles={"frontdoor", "worker_summarize"})
    scenario._base_frontdoor_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    # The registry remains parseable, but the already-generated descriptor and
    # priors no longer describe it. The gate must be skipped on that real error.
    scenario._base_frontdoor_registry(config.lean_registry, throughput=99.0)
    checked = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    report = run_stack_change_pipeline(checked)
    gate = next(step for step in report.steps if step.name == "promotion_gate")
    assert not report.ok
    assert gate.status == "skipped"
