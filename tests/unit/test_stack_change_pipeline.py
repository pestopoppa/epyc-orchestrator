"""Tests for the canonical stack-change pipeline command."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.registry.stack_change_pipeline import (
    StackChangePipelineConfig,
    run_stack_change_pipeline,
)


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _registry(path: Path, *, throughput: float = 24.3) -> Path:
    return _write_yaml(
        path,
        {
            "server_mode": {
                "frontdoor": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "slots": 2,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_role": "frontdoor",
                    "memory_gb": 37,
                    "throughput": throughput,
                    "benchmark_score": "170/183 (92.9%)",
                    "benchmark_date": "2026-05-04",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8070],
                }
            },
            "roles": {
                "frontdoor": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q8_0",
                        "quant": "Q8_0",
                        "architecture": "qwen35moe",
                        "size_gb": 37,
                        "ctx_max": 131072,
                    },
                    "performance": {
                        "quality_pct": 93,
                        "baseline_tps": throughput,
                        "benchmark_date": "2026-05-04",
                    },
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                }
            },
        },
    )


def _procedure(path: Path) -> Path:
    return _write_yaml(
        path,
        {
            "inputs": [
                {
                    "name": "role",
                    "type": "string",
                    "description": "Role assignment",
                    "validation": {"enum": ["frontdoor"]},
                }
            ]
        },
    )


def _schema(path: Path) -> Path:
    return _write_json(
        path,
        {
            "properties": {
                "permissions": {
                    "properties": {
                        "roles": {
                            "items": {
                                "enum": ["frontdoor", "admin"],
                            }
                        }
                    }
                }
            }
        },
    )


def _config(tmp_path: Path, *, mode: str) -> StackChangePipelineConfig:
    repo_root = tmp_path
    registry = _registry(tmp_path / "orchestration" / "model_registry.yaml")
    descriptors = tmp_path / "orchestration" / "model_descriptors.yaml"
    priors = tmp_path / "orchestration" / "derived" / "stack_priors.yaml"
    procedure = _procedure(tmp_path / "orchestration" / "procedures" / "add_model.yaml")
    schema = _schema(tmp_path / "orchestration" / "procedure.schema.json")
    return StackChangePipelineConfig(
        mode=mode,  # type: ignore[arg-type]
        repo_root=repo_root,
        lean_registry=registry,
        research_registry=None,
        descriptors=descriptors,
        stack_priors=priors,
        procedure=procedure,
        schema=schema,
        surface_exceptions=tmp_path / "missing_exceptions.yaml",
        roles={"frontdoor"},
        allow_known_gaps=True,
    )


def test_update_then_check_succeeds_with_known_gaps_allowed(tmp_path: Path) -> None:
    update_report = run_stack_change_pipeline(_config(tmp_path, mode="update"))
    check_report = run_stack_change_pipeline(_config(tmp_path, mode="check"))

    assert update_report.ok
    assert check_report.ok
    assert {step.name for step in check_report.steps} == {
        "descriptors",
        "stack_priors",
        "procedure_enums",
        "guard",
        "guard_all_surfaces",
        "guard_strict",
    }


def test_check_reports_stale_generated_artifact_without_writing(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(config).ok
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _registry(config.lean_registry, throughput=42.0)

    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check"}
    )
    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    assert any("artifact is stale" in error for error in report.errors)
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before


def test_check_fails_on_stale_procedure_enums(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(config).ok
    retired_role = "architect" + "_coding"
    _procedure(config.procedure).write_text(
        f"inputs:\n- name: role\n  validation:\n    enum: [{retired_role}]\n",
        encoding="utf-8",
    )

    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check"}
    )
    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    assert any("procedure role enums are stale" in error for error in report.errors)
