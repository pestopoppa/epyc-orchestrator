"""Scenario-level simulated stack-change fixtures.

These tests use only temporary registries/artifacts. They exercise the
stack-change pipeline against realistic data-only edits without touching live
generated files or running inference.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "autopilot"))

from orchestration.repl_memory.q_scorer import (  # noqa: E402
    PRIOR_SOURCE_STACK_PRIORS,
    stack_prior_q_scorer_priors_by_role,
    validate_live_q_scorer_prior_sources,
)
from scripts.registry.stack_change_pipeline import (  # noqa: E402
    SIMULATED_FIXTURE_TARGET,
    StackChangePipelineConfig,
    run_stack_change_pipeline,
)
from scripts.registry import stack_change_pipeline as pipeline  # noqa: E402

gen_system_card = importlib.import_module("gen_system_card")


@pytest.fixture(autouse=True)
def _clean_runtime_attestation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_runtime_attestation_warnings", lambda: [])


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_role_enum_files(config: StackChangePipelineConfig, roles: set[str]) -> None:
    ordered = sorted(roles)
    permission_roles = [*ordered, "admin"]
    _write_yaml(
        config.procedure,
        {
            "inputs": [
                {
                    "name": "role",
                    "type": "string",
                    "validation": {"enum": ordered},
                }
            ]
        },
    )
    config.schema.parent.mkdir(parents=True, exist_ok=True)
    config.schema.write_text(
        """{
  "properties": {
    "permissions": {
      "properties": {
        "roles": {
          "items": {
            "enum": %s
          }
        }
      }
    }
  }
}
"""
        % json.dumps(permission_roles),
        encoding="utf-8",
    )


def _base_frontdoor_registry(path: Path, *, throughput: float = 24.3) -> Path:
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
                },
                "coder_escalation": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "slots": 2,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_role": "coder_escalation",
                    "memory_gb": 37,
                    "throughput": throughput,
                    "benchmark_score": "170/183 (92.9%)",
                    "benchmark_date": "2026-05-04",
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8070],
                },
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
                },
                "coder_escalation": {
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
                },
            },
        },
    )


def _swapped_frontdoor_registry(path: Path, *, throughput: float = 18.5) -> Path:
    return _write_yaml(
        path,
        {
            "server_mode": {
                "frontdoor": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "slots": 2,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "model_role": "frontdoor",
                    "memory_gb": 20,
                    "throughput": throughput,
                    "benchmark_score": "160/183 (87.4%)",
                    "benchmark_date": "2026-06-13",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8070],
                },
                "coder_escalation": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "slots": 2,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "model_role": "coder_escalation",
                    "memory_gb": 20,
                    "throughput": throughput,
                    "benchmark_score": "160/183 (87.4%)",
                    "benchmark_date": "2026-06-13",
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8070],
                },
            },
            "roles": {
                "frontdoor": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q4_K_M",
                        "quant": "Q4_K_M",
                        "architecture": "qwen35moe",
                        "size_gb": 20,
                        "ctx_max": 131072,
                    },
                    "performance": {
                        "quality_pct": 87.4,
                        "baseline_tps": throughput,
                        "benchmark_date": "2026-06-13",
                    },
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                },
                "coder_escalation": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q4_K_M",
                        "quant": "Q4_K_M",
                        "architecture": "qwen35moe",
                        "size_gb": 20,
                        "ctx_max": 131072,
                    },
                    "performance": {
                        "quality_pct": 87.4,
                        "baseline_tps": throughput,
                        "benchmark_date": "2026-06-13",
                    },
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                },
            },
        },
    )


def _worker_alias_registry(
    path: Path,
    *,
    binary_dir: str = "/mnt/raid0/llm/ik_llama.cpp/build/bin",
    ld_library_path: list[str] | None = None,
) -> Path:
    if ld_library_path is None:
        ld_library_path = [
            "/mnt/raid0/llm/ik_llama.cpp/build/src",
            "/mnt/raid0/llm/ik_llama.cpp/build/ggml/src",
            "/mnt/raid0/llm/ik_llama.cpp/build/examples/mtmd",
        ]
    return _write_yaml(
        path,
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "slots": 1,
                    "model": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "model_role": "worker_general",
                    "shared_with": ["worker_math", "toolrunner"],
                    "memory_gb": 16,
                    "throughput": 60.7,
                    "benchmark_score": "90%",
                    "runtime_requirements": {
                        "binary_dir": binary_dir,
                        "ld_library_path": ld_library_path,
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
                },
                "toolrunner": {
                    "model": {
                        "name": "Qwen3-Coder-30B-A3B-Instruct",
                        "quant": "Q4_K_M",
                        "architecture": "qwen3coder",
                        "size_gb": 16,
                        "ctx_max": 32768,
                    },
                    "performance": {"quality_pct": 84, "baseline_tps": 39.1},
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                },
            },
        },
    )


def _config(tmp_path: Path, *, mode: str, roles: set[str]) -> StackChangePipelineConfig:
    registry = tmp_path / "orchestration" / "model_registry.yaml"
    descriptors = tmp_path / "orchestration" / "model_descriptors.yaml"
    priors = tmp_path / "orchestration" / "derived" / "stack_priors.yaml"
    operator_summary = tmp_path / "docs" / "generated" / "current_stack_summary.md"
    procedure = tmp_path / "orchestration" / "procedures" / "add_model.yaml"
    schema = tmp_path / "orchestration" / "procedure.schema.json"
    _write_role_enum_files(
        StackChangePipelineConfig(
            mode=mode,  # type: ignore[arg-type]
            repo_root=tmp_path,
            lean_registry=registry,
            research_registry=None,
            descriptors=descriptors,
            stack_priors=priors,
            operator_summary=operator_summary,
            procedure=procedure,
            schema=schema,
            surface_exceptions=tmp_path / "missing_exceptions.yaml",
            roles=roles,
            allow_known_gaps=True,
        ),
        roles,
    )
    return StackChangePipelineConfig(
        mode=mode,  # type: ignore[arg-type]
        repo_root=tmp_path,
        lean_registry=registry,
        research_registry=None,
        descriptors=descriptors,
        stack_priors=priors,
        operator_summary=operator_summary,
        procedure=procedure,
        schema=schema,
        surface_exceptions=tmp_path / "missing_exceptions.yaml",
        roles=roles,
        allow_known_gaps=True,
    )


def test_pipeline_report_names_simulated_fixture_target(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "coder_escalation"})
    _base_frontdoor_registry(config.lean_registry)

    report = run_stack_change_pipeline(config)

    step = next(step for step in report.steps if step.name == "simulated_fixtures")
    assert step.name == "simulated_fixtures"
    assert step.status == "reference"
    assert SIMULATED_FIXTURE_TARGET in step.details[0]


def test_simulated_update_does_not_write_real_operator_summary(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "coder_escalation"})
    _base_frontdoor_registry(config.lean_registry)
    real_summary = StackChangePipelineConfig(mode="check").operator_summary
    before = real_summary.read_text(encoding="utf-8")

    assert run_stack_change_pipeline(config).ok

    assert config.operator_summary.exists()
    assert config.operator_summary != real_summary
    assert real_summary.read_text(encoding="utf-8") == before


def test_simulated_frontdoor_swap_updates_generated_consumers_with_approval(
    tmp_path: Path,
) -> None:
    roles = {"frontdoor", "coder_escalation"}
    config = _config(tmp_path, mode="update", roles=roles)
    _base_frontdoor_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    descriptors_before = config.descriptors.read_text(encoding="utf-8")
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _swapped_frontdoor_registry(config.lean_registry)

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    descriptor_step = next(step for step in check_report.steps if step.name == "descriptors")
    assert any("descriptor artifact is stale:" in error for error in descriptor_step.errors)
    assert any("descriptor update would remove existing model_id" in error for error in descriptor_step.errors)
    assert config.descriptors.read_text(encoding="utf-8") == descriptors_before
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before

    approved_config = StackChangePipelineConfig(
        **{**config.__dict__, "allow_descriptor_model_removal": True}
    )
    update_report = run_stack_change_pipeline(approved_config)

    assert update_report.ok
    descriptors = yaml.safe_load(config.descriptors.read_text(encoding="utf-8"))
    assert [model["model_id"] for model in descriptors["models"]] == [
        "qwen3.6-35b-a3b-q4_k_m"
    ]
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    assert set(priors["roles"]) == roles
    for role in roles:
        assert priors["roles"][role]["model_id"] == "qwen3.6-35b-a3b-q4_k_m"
        assert priors["roles"][role]["priors"]["throughput_tps"] == 18.5
        assert priors["roles"][role]["priors"]["quality_overall"] == pytest.approx(0.874)

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    assert q_priors.baseline_tps_by_role["frontdoor"] == 18.5
    assert q_priors.baseline_tps_by_role["coder_escalation"] == 18.5
    assert q_priors.baseline_tps_source_by_role["frontdoor"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["frontdoor"] == pytest.approx(0.874)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []


def test_simulated_shared_runtime_aliases_compile_as_one_runtime_descriptor(
    tmp_path: Path,
) -> None:
    roles = {"worker_general", "worker_math", "toolrunner"}
    config = _config(tmp_path, mode="update", roles=roles)
    _worker_alias_registry(config.lean_registry)

    report = run_stack_change_pipeline(config)

    assert report.ok
    descriptors = yaml.safe_load(config.descriptors.read_text(encoding="utf-8"))
    assert [model["model_id"] for model in descriptors["models"]] == [
        "gemma4-26b-a4b-q4_k_m"
    ]
    model = descriptors["models"][0]
    assert model["role_bindings"]["roles"] == ["toolrunner", "worker_general", "worker_math"]
    assert not any(gap.startswith("Role-server conflict:") for gap in model["known_gaps"])
    assert not any("ignored non-live role model metadata" in gap for gap in model["known_gaps"])
    alias_overrides = model["role_bindings"]["alias_overrides"]
    ignored_models = {override["ignored_model_id"] for override in alias_overrides}
    assert ignored_models == {"qwen2.5-math-7b-q4_k_m", "qwen3-coder-30b-a3b-q4_k_m"}

    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    primary_runtime = priors["roles"]["worker_general"]["serving"]["launch"]["runtime"]
    primary_requirements = priors["roles"]["worker_general"]["serving"]["launch"]["requirements"]
    for alias in ("worker_math", "toolrunner"):
        assert priors["roles"][alias]["serving"]["binding"] == "server_mode.shared_with"
        assert priors["roles"][alias]["serving"]["launch"]["runtime"] == primary_runtime
        assert priors["roles"][alias]["serving"]["launch"]["requirements"] == primary_requirements


def test_simulated_retired_role_enum_is_removed_by_update(tmp_path: Path) -> None:
    roles = {"frontdoor", "coder_escalation"}
    config = _config(tmp_path, mode="update", roles=roles)
    _base_frontdoor_registry(config.lean_registry)
    retired_role = "architect" + "_coding"
    assert run_stack_change_pipeline(config).ok
    _write_role_enum_files(config, roles | {retired_role})

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    assert any("procedure role enums are stale" in error for error in check_report.errors)

    update_report = run_stack_change_pipeline(config)

    assert update_report.ok
    procedure = yaml.safe_load(config.procedure.read_text(encoding="utf-8"))
    role_enum = procedure["inputs"][0]["validation"]["enum"]
    assert retired_role not in role_enum
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    assert retired_role not in priors["roles"]
    card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert f"| {retired_role} |" not in card
    assert f"{retired_role} is not an active server role" in card


def test_simulated_runtime_requirement_drift_fails_until_regenerated(tmp_path: Path) -> None:
    roles = {"worker_general", "worker_math", "toolrunner"}
    config = _config(tmp_path, mode="update", roles=roles)
    _worker_alias_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _worker_alias_registry(
        config.lean_registry,
        binary_dir="/tmp/simulated-ik-build/bin",
        ld_library_path=["/tmp/simulated-ik-build/lib"],
    )

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    assert any("stack-prior artifact is stale" in error for error in check_report.errors)
    assert any("serving.launch.runtime does not match" in error for error in check_report.errors)
    assert any("/tmp/simulated-ik-build/bin" in error for error in check_report.errors)
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before

    update_report = run_stack_change_pipeline(config)

    assert update_report.ok
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    runtime = priors["roles"]["worker_general"]["serving"]["launch"]["runtime"]
    assert runtime["binary_dir"] == "/tmp/simulated-ik-build/bin"
    assert runtime["binary_path"] == "/tmp/simulated-ik-build/bin/llama-server"
    assert runtime["ld_library_path"] == ["/tmp/simulated-ik-build/lib"]
    assert runtime["env_policy"] == "binary_override_strip_ggml"


def test_simulated_context_kv_and_acceleration_drift_are_rejected(
    tmp_path: Path,
) -> None:
    roles = {
        "frontdoor",
        "worker_general",
        "worker_vision",
        "vision_escalation",
        "architect_general",
    }
    config = _config(tmp_path, mode="update", roles=roles)
    _write_yaml(
        config.lean_registry,
        {
            "server_mode": {
                "frontdoor": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "model_role": "frontdoor",
                    "model": "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "throughput": 24.3,
                    "memory_gb": 37,
                },
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "model_role": "worker_general",
                    "model": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "throughput": 60.7,
                    "memory_gb": 16,
                    "runtime_requirements": {
                        "binary_dir": "/mnt/raid0/llm/ik_llama.cpp/build/bin",
                        "ld_library_path": ["/mnt/raid0/llm/ik_llama.cpp/build/src"],
                    },
                },
                "worker_vision": {
                    "url": "http://localhost:8086",
                    "port": 8086,
                    "tier": "hot",
                    "model_role": "worker_vision",
                    "model": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                    "throughput": 20.0,
                    "memory_gb": 7,
                },
                "vision_escalation": {
                    "url": "http://localhost:8087",
                    "port": 8087,
                    "tier": "hot",
                    "model_role": "vision_escalation",
                    "model": "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf",
                    "throughput": 8.0,
                    "memory_gb": 20,
                },
                "architect_general": {
                    "url": "http://localhost:8083",
                    "port": 8083,
                    "tier": "hot",
                    "model_role": "architect_general",
                    "model": "Qwen3.5-122B-A3B-Instruct-Q4_K_M.gguf",
                    "throughput": 12.19,
                    "memory_gb": 133,
                },
            },
            "roles": {
                "frontdoor": {
                    "model": {"name": "Qwen3.6-35B-A3B-Q8_0", "ctx_max": 131072},
                    "performance": {"quality_pct": 93, "baseline_tps": 24.3},
                    "memory": {"residency": "hot"},
                },
                "worker_general": {
                    "model": {"name": "gemma-4-26B-A4B-it-Q4_K_M", "ctx_max": 16384},
                    "performance": {"quality_pct": 90, "baseline_tps": 60.7},
                    "acceleration": {"type": "speculative_decoding", "spec_type": "mtp"},
                    "memory": {"residency": "hot"},
                },
                "worker_vision": {
                    "model": {"name": "Qwen2.5-VL-7B-Instruct-Q4_K_M", "ctx_max": 8192},
                    "performance": {"quality_pct": 81, "baseline_tps": 20.0},
                    "memory": {"residency": "hot"},
                },
                "vision_escalation": {
                    "model": {"name": "Qwen3-VL-30B-A3B-Instruct-Q4_K_M", "ctx_max": 16384},
                    "performance": {"quality_pct": 90, "baseline_tps": 8.0},
                    "memory": {"residency": "hot"},
                },
                "architect_general": {
                    "model": {"name": "Qwen3.5-122B-A3B-Instruct-Q4_K_M", "ctx_max": 16384},
                    "performance": {"quality_pct": 94, "baseline_tps": 12.19},
                    "acceleration": {
                        "type": "moe_expert_reduction",
                        "override_key": "qwen35moe.expert_used_count",
                        "experts": 8,
                        "draft_max": 4,
                    },
                    "memory": {"residency": "hot"},
                },
            },
        },
    )
    assert run_stack_change_pipeline(config).ok
    payload = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    payload["roles"]["frontdoor"]["serving"]["launch"]["runtime"]["cache"]["kv_type_k"] = "f16"
    payload["roles"]["worker_general"]["serving"]["launch"]["runtime"]["flags"]["spec"][
        "enabled"
    ] = False
    payload["roles"]["worker_general"]["serving"]["effective_context_tokens"] = 8192
    payload["roles"]["architect_general"]["serving"]["launch"]["runtime"]["flags"][
        "override_kv"
    ] = []
    payload["roles"]["worker_vision"]["serving"]["launch"]["requirements"][
        "mmproj_path"
    ] = "/tmp/stale-mmproj.gguf"
    payload["roles"]["vision_escalation"]["serving"]["launch"]["runtime"]["flags"][
        "override_kv"
    ] = []
    _write_yaml(config.stack_priors, payload)

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    assert any("serving.effective_context_tokens 8192" in error for error in report.errors)
    assert any("serving.launch.runtime does not match" in error for error in report.errors)
    assert any('"kv_type_k": "f16"' in error for error in report.errors)
    assert any('"enabled": false' in error for error in report.errors)
    assert any("qwen35moe.expert_used_count=int:8" in error for error in report.errors)
    assert any("qwen3vlmoe.expert_used_count=int:4" in error for error in report.errors)
    assert any("mmproj_path" in error and "stale-mmproj" in error for error in report.errors)
