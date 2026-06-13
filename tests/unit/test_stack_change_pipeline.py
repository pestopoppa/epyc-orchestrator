"""Tests for the canonical stack-change pipeline command."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from scripts.registry import stack_change_pipeline as pipeline
from scripts.registry.stack_change_pipeline import (
    PipelineReport,
    PipelineStep,
    PROMOTION_GATE_TARGETS,
    SIMULATED_FIXTURE_TARGET,
    StackChangePipelineConfig,
    _print_report,
    run_stack_change_pipeline,
)

PROMOTION_GATE_COMMAND = "promotion_gate: run uv run pytest -q " + " ".join(
    PROMOTION_GATE_TARGETS
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


def test_update_merges_shared_alias_mismatch_into_runtime_descriptor(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update")
    _write_yaml(
        config.lean_registry,
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "slots": 1,
                    "model": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "model_role": "worker_general",
                    "shared_with": ["worker_math"],
                    "memory_gb": 16,
                    "throughput": 60.7,
                    "benchmark_score": "90%",
                    "runtime_requirements": {
                        "binary_dir": "/mnt/raid0/llm/ik_llama.cpp/build/bin",
                        "ld_library_path": [
                            "/mnt/raid0/llm/ik_llama.cpp/build/src",
                            "/mnt/raid0/llm/ik_llama.cpp/build/ggml/src",
                            "/mnt/raid0/llm/ik_llama.cpp/build/examples/mtmd",
                        ],
                    },
                    "numa_instances": 4,
                    "numa_ports": [8082, 8182, 8282, 8382],
                }
            },
            "roles": {
                "worker_general": {
                    "model": {
                        "name": "gemma-4-26B-A4B-it-Q4_K_M",
                        "quant": "Q4_K_M",
                        "architecture": "gemma4",
                        "size_gb": 16,
                        "ctx_max": 16384,
                    },
                    "performance": {"quality_pct": 90, "baseline_tps": 44.7},
                    "acceleration": {"type": "speculative_decoding", "spec_type": "mtp"},
                    "memory": {"pinned": True, "residency": "hot"},
                },
                "worker_math": {
                    "model": {
                        "name": "Qwen2.5-Math-7B-Instruct",
                        "quant": "Q4_K_M",
                        "architecture": "dense",
                        "size_gb": 4.4,
                        "ctx_max": 32768,
                    },
                    "performance": {"quality_pct": 88, "baseline_tps": 12.4},
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                }
            },
        },
    )
    conflict_config = StackChangePipelineConfig(
        **{**config.__dict__, "roles": {"worker_general", "worker_math"}}
    )
    _write_yaml(
        conflict_config.procedure,
        {
            "inputs": [
                {
                    "name": "role",
                    "type": "string",
                    "validation": {"enum": ["worker_general", "worker_math"]},
                }
            ]
        },
    )
    _write_json(
        conflict_config.schema,
        {
            "properties": {
                "permissions": {
                    "properties": {
                        "roles": {
                            "items": {
                                "enum": ["worker_general", "worker_math", "admin"],
                            }
                        }
                    }
                }
            }
        },
    )

    report = run_stack_change_pipeline(conflict_config)

    assert report.ok
    descriptors = yaml.safe_load(conflict_config.descriptors.read_text(encoding="utf-8"))
    assert [model["model_id"] for model in descriptors["models"]] == [
        "gemma4-26b-a4b-q4_k_m"
    ]
    model = descriptors["models"][0]
    assert model["role_bindings"]["roles"] == ["worker_general", "worker_math"]
    assert model["role_bindings"]["alias_overrides"] == [
        {
            "role": "worker_math",
            "served_by": "worker_general",
            "ignored_model_id": "qwen2.5-math-7b-q4_k_m",
            "reason": "server_mode.shared_with runtime takes precedence",
        }
    ]
    assert not any("ignored non-live role model metadata" in gap for gap in model["known_gaps"])
    assert not any(
        gap.startswith("Role-server conflict:")
        for gap in model["known_gaps"]
    )
    assert conflict_config.stack_priors.exists()


def test_check_reports_shared_alias_mismatch_without_conflict_error(
    tmp_path: Path,
) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    config = StackChangePipelineConfig(
        **{**update_config.__dict__, "mode": "check"}
    )
    _write_yaml(
        config.lean_registry,
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "slots": 1,
                    "model": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "model_role": "worker_general",
                    "shared_with": ["worker_math"],
                    "memory_gb": 16,
                    "throughput": 60.7,
                    "benchmark_score": "90%",
                    "runtime_requirements": {
                        "binary_dir": "/mnt/raid0/llm/ik_llama.cpp/build/bin",
                        "ld_library_path": [
                            "/mnt/raid0/llm/ik_llama.cpp/build/src",
                            "/mnt/raid0/llm/ik_llama.cpp/build/ggml/src",
                            "/mnt/raid0/llm/ik_llama.cpp/build/examples/mtmd",
                        ],
                    },
                    "numa_instances": 4,
                    "numa_ports": [8082, 8182, 8282, 8382],
                }
            },
            "roles": {
                "worker_general": {
                    "model": {
                        "name": "gemma-4-26B-A4B-it-Q4_K_M",
                        "quant": "Q4_K_M",
                        "architecture": "gemma4",
                        "size_gb": 16,
                        "ctx_max": 16384,
                    },
                    "performance": {"quality_pct": 90, "baseline_tps": 44.7},
                    "acceleration": {"type": "speculative_decoding", "spec_type": "mtp"},
                    "memory": {"pinned": True, "residency": "hot"},
                },
                "worker_math": {
                    "model": {
                        "name": "Qwen2.5-Math-7B-Instruct",
                        "quant": "Q4_K_M",
                        "architecture": "dense",
                        "size_gb": 4.4,
                        "ctx_max": 32768,
                    },
                    "performance": {"quality_pct": 88, "baseline_tps": 12.4},
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                }
            },
        },
    )
    conflict_config = StackChangePipelineConfig(
        **{**config.__dict__, "roles": {"worker_general", "worker_math"}}
    )

    report = run_stack_change_pipeline(conflict_config)

    descriptor_step = next(step for step in report.steps if step.name == "descriptors")
    assert not report.ok
    assert any("descriptor artifact is stale" in error for error in descriptor_step.errors)
    assert any("descriptor update would remove existing model_id" in error for error in descriptor_step.errors)
    assert not any("descriptor generated role/server conflict" in error for error in descriptor_step.errors)


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
        "simulated_fixtures",
        "promotion_gate",
    }
    promotion_step = next(step for step in check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "reference"
    assert any(PROMOTION_GATE_COMMAND.removeprefix("promotion_gate: run ") in detail for detail in promotion_step.details)
    assert check_report.acceptance_lines() == [
        "acceptance: no-inference checks passed",
        PROMOTION_GATE_COMMAND,
    ]


def test_acceptance_lines_summarize_unique_hardcoded_surface_warnings() -> None:
    duplicate_warning = (
        "hardcoded_surface.production_blocker.retired_role_in_active_code: "
        "src/example.py:1: retired_role"
    )
    report = PipelineReport(
        steps=[
            PipelineStep(
                name="guard",
                status="warnings",
                warnings=[
                    duplicate_warning,
                    duplicate_warning,
                    "hardcoded_surface.waived.production_blocker.retired_role_in_active_code: src/example.py:2",
                    "hardcoded_surface.legacy_test.retired_role_in_tests: tests/example.py:3",
                    "hardcoded_surface.historical_doc.retired_role_in_operator_docs: docs/example.md:4",
                    "role 'example' has 1 known gap(s)",
                ],
            )
        ]
    )

    assert report.hardcoded_surface_warning_counts() == {
        "production_blocker": 1,
        "waived_production_blocker": 1,
        "legacy_test": 1,
        "historical_doc": 1,
    }
    assert report.acceptance_lines() == [
        "acceptance: no-inference checks passed",
        "warnings: 5 unique (6 total)",
        "surface_warnings: production_blocker=1, waived_production_blocker=1, legacy_test=1, historical_doc=1",
        PROMOTION_GATE_COMMAND,
    ]


def test_print_report_includes_promotion_gate_for_passing_check(
    tmp_path: Path,
    capsys,
) -> None:
    assert run_stack_change_pipeline(_config(tmp_path, mode="update")).ok
    report = run_stack_change_pipeline(_config(tmp_path, mode="check"))

    _print_report(report)

    output = capsys.readouterr().out
    assert "summary: ok" in output
    assert "acceptance: no-inference checks passed" in output
    assert PROMOTION_GATE_COMMAND in output
    assert SIMULATED_FIXTURE_TARGET in output


def test_print_report_blocks_promotion_for_failed_check(
    tmp_path: Path,
    capsys,
) -> None:
    config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(config).ok
    _registry(config.lean_registry, throughput=42.0)
    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check"}
    )
    report = run_stack_change_pipeline(check_config)

    _print_report(report)

    output = capsys.readouterr().out
    assert not report.ok
    assert "summary: failed" in output
    assert "acceptance: blocked" in output
    assert "promotion_gate: fix " in output


def test_run_promotion_gate_executes_combined_no_inference_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        if cmd[:4] != ["uv", "run", "pytest", "-q"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(cmd, 0, stdout="53 passed\n", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    config = StackChangePipelineConfig(
        **{
            **update_config.__dict__,
            "run_promotion_gate": True,
            "mode": "check",
        }
    )
    report = run_stack_change_pipeline(config)
    promotion_step = next(step for step in report.steps if step.name == "promotion_gate")

    assert report.ok
    assert captured["cmd"] == ["uv", "run", "pytest", "-q", *PROMOTION_GATE_TARGETS]
    assert captured["cwd"] == tmp_path
    assert promotion_step.status == "ok"
    assert any("53 passed" in detail for detail in promotion_step.details)


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
    descriptor_step = next(step for step in report.steps if step.name == "descriptors")
    assert any("artifact is stale" in error for error in report.errors)
    assert any("descriptor changed qwen3.6-35b-a3b-q8_0" in detail for detail in descriptor_step.details)
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before


def test_update_refuses_to_remove_existing_descriptor_model_ids(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update")
    _write_yaml(
        config.descriptors,
        {
            "models": [
                {
                    "model_id": "qwen3.6-35b-a3b-q8_0",
                },
                {
                    "model_id": "benchmark-only-reap",
                },
            ]
        },
    )
    descriptor_before = config.descriptors.read_text(encoding="utf-8")

    report = run_stack_change_pipeline(config)

    assert not report.ok
    assert any("benchmark-only-reap" in error for error in report.errors)
    assert config.descriptors.read_text(encoding="utf-8") == descriptor_before
    assert not config.stack_priors.exists()
    assert {step.name: step.status for step in report.steps}["stack_priors"] == "skipped"


def test_check_reports_descriptor_model_removal_blocker(tmp_path: Path) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    loaded = yaml.safe_load(update_config.descriptors.read_text(encoding="utf-8"))
    loaded["models"].append({"model_id": "benchmark-only-reap"})
    _write_yaml(update_config.descriptors, loaded)

    config = StackChangePipelineConfig(
        **{**update_config.__dict__, "mode": "check"}
    )

    report = run_stack_change_pipeline(config)

    assert not report.ok
    descriptor_step = next(step for step in report.steps if step.name == "descriptors")
    assert any("descriptor artifact is stale:" in error for error in descriptor_step.errors)
    assert any(
        "descriptor generated removes model_id(s): benchmark-only-reap" in detail
        for detail in descriptor_step.details
    )
    assert any("benchmark-only-reap" in error for error in descriptor_step.errors)
    assert not any(
        error.endswith("stack_change_pipeline.py update") for error in descriptor_step.errors
    )


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
