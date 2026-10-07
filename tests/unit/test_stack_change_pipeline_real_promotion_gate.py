"""End-to-end subprocess controls for the promotion gate over temp stack worlds.

This module is intentionally outside ``PROMOTION_GATE_TARGETS``: its positive control launches
that exact suite, so including itself would recurse.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from scripts.registry import stack_change_pipeline as pipeline
from scripts.registry.stack_change_pipeline import (
    StackChangePipelineConfig,
    run_stack_change_pipeline,
)


# Imported pytest fixtures are enrolled in this module's outer cases too.
# Merely importing the scenario helpers inside a test does not apply their
# hermetic runtime-attestation and realized-NUMA fixtures.
from tests.unit.test_stack_change_pipeline_simulated_fixtures import (
    _clean_runtime_attestation,
    _pin_realized_compile_mode,
)

from tests.unit.test_stack_change_guard import _synthetic_stack_manifest_inputs


def _capture_report(report, capture_root: Path, label: str):
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / f"{label}.pipeline.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _scenario_helpers():
    return importlib.import_module("tests.unit.test_stack_change_pipeline_simulated_fixtures")


def test_real_promotion_gate_accepts_each_approved_swapped_temporary_world(
    tmp_path: Path, _synthetic_stack_manifest_inputs
) -> None:
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
    capture_root = Path(os.environ.get("W4_PROMOTION_GATE_CAPTURE_DIR", tmp_path / "capture"))
    capture_root.mkdir(parents=True, exist_ok=True)
    worlds_summary = []
    for name, roles, prepare_base, prepare_swap in worlds:
        world = tmp_path / name
        world.mkdir()
        config = scenario._config(world, mode="update", roles=set(roles))
        prepare_base(config.lean_registry)
        assert _capture_report(run_stack_change_pipeline(config), capture_root, f"{name}-base").ok

        prepare_swap(config.lean_registry)
        approved = StackChangePipelineConfig(
            **{**config.__dict__, "allow_descriptor_model_removal": True}
        )
        assert _capture_report(run_stack_change_pipeline(approved), capture_root, f"{name}-swap").ok

        checked = StackChangePipelineConfig(
            **{**approved.__dict__, "mode": "check", "run_promotion_gate": True}
        )
        inner_junit = capture_root / f"{name}.junit.xml"
        prior_addopts = os.environ.get("PYTEST_ADDOPTS")
        os.environ["PYTEST_ADDOPTS"] = f"--junitxml={inner_junit}"
        # The production gate clips subprocess output for normal callers. For this
        # conformance fixture, retain the actual subprocess output in its report.
        clip_output = pipeline._clip_output
        pipeline._clip_output = lambda text, *, max_chars=2000: text
        try:
            report = run_stack_change_pipeline(checked)
        finally:
            pipeline._clip_output = clip_output
            if prior_addopts is None:
                os.environ.pop("PYTEST_ADDOPTS", None)
            else:
                os.environ["PYTEST_ADDOPTS"] = prior_addopts
        _capture_report(report, capture_root, f"{name}-check")
        gate = next(step for step in report.steps if step.name == "promotion_gate")
        assert report.ok, name
        assert gate.status == "ok", name
        assert inner_junit.is_file(), name
        suite = ET.parse(inner_junit).getroot()
        suite_rows = list(suite.iter("testsuite"))
        assert suite_rows, name
        nodes = list(suite.iter("testcase"))
        assert nodes, name
        assert all(not any(row.find(tag) is not None for tag in ("failure", "error", "skipped"))
                   for row in nodes), name
        assert sum(int(row.attrib.get("tests", "0")) for row in suite_rows) == len(nodes), name
        cases = [{"classname": row.attrib.get("classname"), "name": row.attrib.get("name")}
                 for row in nodes]
        assert cases, name
        expected_modules = {
            ".".join(Path(target).with_suffix("").parts)
            for target in pipeline.PROMOTION_GATE_TARGETS
        }
        observed_modules = {
            module for module in expected_modules
            if any(case["classname"] == module or case["classname"].startswith(module + ".")
                   for case in cases)
        }
        assert observed_modules == expected_modules, name
        worlds_summary.append(
            {
                "world": name,
                "gate_status": gate.status,
                "gate_details": gate.details,
                "inner_junit": str(inner_junit),
                "tests": cases,
                "test_count": len(cases),
            }
        )
        command = pipeline._promotion_gate_command()
        assert command[:3] == [sys.executable, "-m", "pytest"]
        assert command[3:8] == ["-q", "-c", str(pipeline.REPO_ROOT / "pyproject.toml"),
                               f"--rootdir={pipeline.REPO_ROOT}", "-o"]
        assert command[8:10] == ["addopts=", "-p"]
        assert command[10] == "no:cacheprovider"
        assert str(pipeline.REPO_ROOT / scenario.SIMULATED_FIXTURE_TARGET) in command
    identity_sets = [
        {(case["classname"], case["name"]) for case in world["tests"]}
        for world in worlds_summary
    ]
    assert all(rows == identity_sets[0] for rows in identity_sets[1:])
    (capture_root / "worlds.json").write_text(
        json.dumps({"worlds": worlds_summary}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_real_promotion_gate_does_not_run_after_a_bad_temporary_world(
    tmp_path: Path, _synthetic_stack_manifest_inputs
) -> None:
    scenario = _scenario_helpers()
    config = scenario._config(tmp_path, mode="update", roles={"frontdoor", "worker_summarize"})
    scenario._base_frontdoor_registry(config.lean_registry)
    capture_root = Path(os.environ.get("W4_PROMOTION_GATE_CAPTURE_DIR", tmp_path / "capture"))
    assert _capture_report(run_stack_change_pipeline(config), capture_root, "refusal-base").ok
    # The registry remains parseable, but the already-generated descriptor and
    # priors no longer describe it. The gate must be skipped on that real error.
    scenario._base_frontdoor_registry(config.lean_registry, throughput=99.0)
    checked = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    report = run_stack_change_pipeline(checked)
    _capture_report(report, capture_root, "refusal-check")
    gate = next(step for step in report.steps if step.name == "promotion_gate")
    assert not report.ok
    assert gate.status == "skipped"
    capture_root = Path(os.environ.get("W4_PROMOTION_GATE_CAPTURE_DIR", tmp_path / "capture"))
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / "refusal.json").write_text(
        json.dumps({"invalid_world": True, "report_ok": report.ok,
                    "gate_status": gate.status, "gate_warnings": gate.warnings,
                    "gate_errors": gate.errors}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
