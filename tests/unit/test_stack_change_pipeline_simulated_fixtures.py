"""Scenario-level simulated stack-change fixtures.

These tests use only temporary registries/artifacts. They exercise the
stack-change pipeline against realistic data-only edits without touching live
generated files or running inference.
"""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "autopilot"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

from orchestration.repl_memory.q_scorer import (  # noqa: E402
    PRIOR_SOURCE_STACK_PRIORS,
    stack_prior_q_scorer_priors_by_role,
    validate_live_q_scorer_prior_sources,
)
from seeding_rewards import (  # noqa: E402
    ALLOW_DEGRADED_CONFIG_KEY,
    MODEL_DESCRIPTORS_CONFIG_KEY,
    PRIOR_SOURCE_MODEL_DESCRIPTORS,
    STACK_PRIORS_CONFIG_KEY,
    RoleResult,
    compute_comparative_rewards,
    descriptor_throughput_by_role,
    throughput_prior_provenance,
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


def _swapped_worker_registry(path: Path, *, throughput: float = 66.2) -> Path:
    return _write_yaml(
        path,
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "slots": 1,
                    "model": "gemma-4-26B-A4B-it-Q8_0.gguf",
                    "model_path": "/models/gemma-4-26B-A4B-it-Q8_0.gguf",
                    "model_role": "worker_general",
                    "shared_with": ["worker_math", "toolrunner"],
                    "memory_gb": 30,
                    "throughput": throughput,
                    "benchmark_score": "92%",
                    # 2026-06-26 v6 cutover: worker consolidated onto canonical
                    # llama.cpp (v6); ik_llama.cpp deprecated.
                    "runtime_requirements": {
                        "binary_dir": "/mnt/raid0/llm/llama.cpp/build/bin",
                        "ld_library_path": [
                            "/mnt/raid0/llm/llama.cpp/build/src",
                            "/mnt/raid0/llm/llama.cpp/build/ggml/src",
                            "/mnt/raid0/llm/llama.cpp/build/examples/mtmd",
                        ],
                    },
                    "numa_instances": 4,
                    "numa_ports": [8082, 8182, 8282, 8382],
                }
            },
            "roles": {
                "worker_general": {
                    "model": {
                        "name": "gemma-4-26B-A4B-it-Q8_0",
                        "quant": "Q8_0",
                        "architecture": "gemma4",
                        "size_gb": 30,
                        "ctx_max": 16384,
                    },
                    "performance": {"quality_pct": 90, "baseline_tps": throughput},
                    # 2026-06-26 v6 cutover: MTP spec token is now 'draft-mtp'.
                    "acceleration": {"type": "speculative_decoding", "spec_type": "draft-mtp"},
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


def _vision_registry(
    path: Path,
    *,
    worker_model: str = "Qwen2.5-VL-7B-Instruct-Q4_K_M",
    worker_memory_gb: int = 7,
    worker_throughput: float = 20.0,
    worker_quality_pct: float = 81.0,
    worker_quant: str = "Q4_K_M",
    escalation_model: str = "Qwen3-VL-30B-A3B-Instruct-Q4_K_M",
    escalation_memory_gb: int = 20,
    escalation_throughput: float = 8.0,
    escalation_quality_pct: float = 90.0,
    escalation_quant: str = "Q4_K_M",
    benchmark_date: str = "2026-05-04",
) -> Path:
    worker_gguf = f"{worker_model}.gguf"
    escalation_gguf = f"{escalation_model}.gguf"
    return _write_yaml(
        path,
        {
            "server_mode": {
                "worker_vision": {
                    "url": "http://localhost:8086",
                    "port": 8086,
                    "tier": "hot",
                    "slots": 2,
                    "model": worker_gguf,
                    "model_path": f"/models/{worker_gguf}",
                    "model_role": "worker_vision",
                    "memory_gb": worker_memory_gb,
                    "throughput": worker_throughput,
                    "benchmark_score": f"{worker_quality_pct:g}%",
                    "benchmark_date": benchmark_date,
                    "mmproj": f"/models/{worker_model}-mmproj.gguf",
                },
                "vision_escalation": {
                    "url": "http://localhost:8087",
                    "port": 8087,
                    "tier": "hot",
                    "slots": 1,
                    "model": escalation_gguf,
                    "model_path": f"/models/{escalation_gguf}",
                    "model_role": "vision_escalation",
                    "memory_gb": escalation_memory_gb,
                    "throughput": escalation_throughput,
                    "benchmark_score": f"{escalation_quality_pct:g}%",
                    "benchmark_date": benchmark_date,
                    "mmproj": f"/models/{escalation_model}-mmproj.gguf",
                },
            },
            "roles": {
                "worker_vision": {
                    "model": {
                        "name": worker_model,
                        "quant": worker_quant,
                        "architecture": "qwen2vl",
                        "size_gb": worker_memory_gb,
                        "ctx_max": 8192,
                    },
                    "performance": {
                        "quality_pct": worker_quality_pct,
                        "baseline_tps": worker_throughput,
                        "benchmark_date": benchmark_date,
                    },
                    "memory": {"pinned": True, "residency": "hot"},
                },
                "vision_escalation": {
                    "model": {
                        "name": escalation_model,
                        "quant": escalation_quant,
                        "architecture": "qwen3vlmoe",
                        "size_gb": escalation_memory_gb,
                        "ctx_max": 16384,
                    },
                    "performance": {
                        "quality_pct": escalation_quality_pct,
                        "baseline_tps": escalation_throughput,
                        "benchmark_date": benchmark_date,
                    },
                    "acceleration": {
                        "type": "moe_expert_reduction",
                        "override_key": "qwen3vlmoe.expert_used_count",
                        "experts": 4,
                    },
                    "memory": {"pinned": True, "residency": "hot"},
                },
            },
        },
    )


def _swapped_vision_registry(path: Path) -> Path:
    return _vision_registry(
        path,
        worker_model="Qwen2.5-VL-7B-Instruct-Q8_0",
        worker_memory_gb=13,
        worker_throughput=14.5,
        worker_quality_pct=84.0,
        worker_quant="Q8_0",
        escalation_model="Qwen3-VL-30B-A3B-Instruct-Q8_0",
        escalation_memory_gb=38,
        escalation_throughput=5.9,
        escalation_quality_pct=92.0,
        escalation_quant="Q8_0",
        benchmark_date="2026-06-13",
    )


def _ingest_registry(
    path: Path,
    *,
    model: str = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M",
    memory_gb: int = 46,
    throughput: float = 20.8,
    quality_pct: float = 92.59,
    quant: str = "Q4_K_M",
    ctx_max: int = 262144,
    benchmark_date: str = "2026-05-04",
) -> Path:
    gguf = f"{model}.gguf"
    return _write_yaml(
        path,
        {
            "server_mode": {
                "ingest_long_context": {
                    "url": "http://localhost:8085",
                    "port": 8085,
                    "tier": "hot",
                    "slots": 1,
                    "model": gguf,
                    "model_path": f"/models/{gguf}",
                    "model_role": "ingest_long_context",
                    "memory_gb": memory_gb,
                    "throughput": throughput,
                    "benchmark_score": f"{quality_pct:g}%",
                    "benchmark_date": benchmark_date,
                }
            },
            "roles": {
                "ingest_long_context": {
                    "model": {
                        "name": model,
                        "quant": quant,
                        "architecture": "qwen3next",
                        "size_gb": memory_gb,
                        "ctx_max": ctx_max,
                    },
                    "performance": {
                        "quality_pct": quality_pct,
                        "baseline_tps": throughput,
                        "long_context_quality": f"{quality_pct:g}%",
                        "benchmark_date": benchmark_date,
                    },
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                }
            },
        },
    )


def _swapped_ingest_registry(path: Path) -> Path:
    return _ingest_registry(
        path,
        model="Qwen3-Next-80B-A3B-Instruct-Q8_0",
        memory_gb=82,
        throughput=14.2,
        quality_pct=96.3,
        quant="Q8_0",
        ctx_max=262144,
        benchmark_date="2026-06-13",
    )


def _worker_alias_registry(
    path: Path,
    *,
    # 2026-06-26 v6 cutover: worker consolidated onto canonical llama.cpp (v6);
    # ik_llama.cpp deprecated. Default runtime points at the canonical build tree.
    binary_dir: str = "/mnt/raid0/llm/llama.cpp/build/bin",
    ld_library_path: list[str] | None = None,
) -> Path:
    if ld_library_path is None:
        ld_library_path = [
            "/mnt/raid0/llm/llama.cpp/build/src",
            "/mnt/raid0/llm/llama.cpp/build/ggml/src",
            "/mnt/raid0/llm/llama.cpp/build/examples/mtmd",
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
                    # 2026-06-26 v6 cutover: MTP spec token is now 'draft-mtp'.
                    "acceleration": {"type": "speculative_decoding", "spec_type": "draft-mtp"},
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


def _assert_text_stack_primary_port_consumers(
    stack_priors_path: Path,
    *,
    expected_roles: set[str],
    expected_port: int,
) -> None:
    from scripts.autopilot.preflight_audit import _model_server_target_groups
    from scripts.benchmark import corpus_quality_gate
    from scripts.graph_router.train_graph_router import load_model_fleet
    from src.api.routes.openai_compat import _ordered_live_role_ids
    from src.cli_orch import _stack_status_targets

    artifact = yaml.safe_load(stack_priors_path.read_text(encoding="utf-8"))
    records = artifact["roles"]
    grouped_role_name = "/".join(sorted(expected_roles))

    assert (grouped_role_name, expected_port) in _stack_status_targets(stack_priors_path)

    live_models = corpus_quality_gate._load_live_models(stack_priors_path)
    assert {role: live_models[role]["port"] for role in expected_roles} == {
        role: expected_port for role in expected_roles
    }

    fleet = {record["role_id"]: record for record in load_model_fleet(stack_priors_path)}
    assert {role: fleet[role]["port"] for role in expected_roles} == {
        role: expected_port for role in expected_roles
    }

    _, names_by_health_url = _model_server_target_groups(records, "http://localhost:8000")
    assert sorted(names_by_health_url[f"http://localhost:{expected_port}/health"]) == sorted(
        expected_roles
    )

    sentinel_records = {
        **records,
        "_sentinel_before": {"serving": {"ports": [expected_port - 1]}},
        "_sentinel_after": {"serving": {"ports": [expected_port + 1]}},
    }
    ordered_roles = _ordered_live_role_ids(sentinel_records)
    assert ordered_roles.index("_sentinel_before") < min(
        ordered_roles.index(role) for role in expected_roles
    )
    assert max(ordered_roles.index(role) for role in expected_roles) < ordered_roles.index(
        "_sentinel_after"
    )


def _primary_port_for_roles(priors: dict[str, Any], roles: set[str]) -> int:
    ports = {
        port
        for role in roles
        for port in ((priors["roles"][role].get("serving") or {}).get("ports") or [])
        if isinstance(port, int)
    }
    assert ports
    return min(ports)


def _assert_worker_pool_stack_prior_consumer(
    records: dict[str, dict[str, Any]],
    *,
    expected_port: int,
    expected_model_path_fragment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.registry import stack_priors
    from src.services.worker_pool import WorkerPoolManager, WorkerTier

    monkeypatch.setattr(stack_priors, "live_stack_role_records", lambda _path=None: records)
    config = WorkerPoolManager().config

    assert list(config.workers) == ["worker_general"]
    worker = config.workers["worker_general"]
    assert worker.port == expected_port
    assert expected_model_path_fragment in worker.model_path
    assert worker.tier is WorkerTier.HOT
    assert worker.managed_process is False


def _assert_factual_risk_stack_prior_consumer(
    stack_priors_path: Path,
    *,
    expected_tiers: dict[str, str],
) -> None:
    from src.classifiers.factual_risk import _role_adjustment, _role_tier_for_role

    adjustment_by_tier = {"tier_1": 0.11, "tier_2": 0.22, "tier_3": 0.33}
    config = {"role_adjustments": adjustment_by_tier}

    for role, expected_tier in expected_tiers.items():
        assert _role_tier_for_role(role, stack_priors_path=stack_priors_path) == expected_tier
        assert _role_adjustment(
            role,
            config=config,
            stack_priors_path=stack_priors_path,
        ) == pytest.approx(adjustment_by_tier[expected_tier])


def _assert_seeding_descriptor_fallback_consumer(
    descriptors_path: Path,
    *,
    expected_roles: set[str],
    expected_tps: float,
) -> None:
    throughput = descriptor_throughput_by_role(descriptors_path)
    assert {role: throughput[role] for role in expected_roles} == {
        role: expected_tps for role in expected_roles
    }

    missing_stack_priors = descriptors_path.with_name("missing_stack_priors.yaml")
    cost_config = {
        STACK_PRIORS_CONFIG_KEY: missing_stack_priors,
        MODEL_DESCRIPTORS_CONFIG_KEY: descriptors_path,
        ALLOW_DEGRADED_CONFIG_KEY: True,
    }
    provenance = throughput_prior_provenance(cost_config)
    assert provenance["source"] == PRIOR_SOURCE_MODEL_DESCRIPTORS
    assert set(expected_roles) <= set(provenance["roles"])
    assert provenance["model_descriptors_path"] == str(descriptors_path)

    rewards = compute_comparative_rewards(
        {
            "frontdoor:direct": RoleResult(
                role="frontdoor",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=1.0,
            ),
            "worker_general:direct": RoleResult(
                role="worker_general",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=2.0,
                generation_ms=2000,
                tokens_generated=100,
            ),
        },
        cost_config=cost_config,
    )
    expected_elapsed = 100 / expected_tps
    expected_reward = 0.5 - 0.15 * max(0.0, (2.0 / expected_elapsed) - 1.0)
    assert math.isclose(rewards["worker_general:direct"], expected_reward)


def test_pipeline_report_names_simulated_fixture_target(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "coder_escalation"})
    _base_frontdoor_registry(config.lean_registry)

    report = run_stack_change_pipeline(config)

    step = next(step for step in report.steps if step.name == "simulated_fixtures")
    assert step.name == "simulated_fixtures"
    assert step.status == "reference"
    assert SIMULATED_FIXTURE_TARGET in step.details[0]


def test_simulated_check_runs_promotion_gate_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "coder_escalation"})
    _base_frontdoor_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check", "run_promotion_gate": True}
    )

    report = run_stack_change_pipeline(check_config)

    assert report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    step = next(step for step in report.steps if step.name == "promotion_gate")
    assert step.status == "ok"
    assert any("promotion gate ok" in detail for detail in step.details)


def test_simulated_check_fails_when_promotion_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "coder_escalation"})
    _base_frontdoor_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **_)
        return subprocess.CompletedProcess(
            args=command,
            returncode=7,
            stdout="partial output\n",
            stderr="promotion gate failed\n",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check", "run_promotion_gate": True}
    )

    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    step = next(step for step in report.steps if step.name == "promotion_gate")
    assert step.status == "failed"
    assert step.errors == ["promotion gate exited 7"]
    assert any("partial output" in detail for detail in step.details)
    assert any("promotion gate failed" in detail for detail in step.details)


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
    monkeypatch: pytest.MonkeyPatch,
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

    operator_summary = config.operator_summary.read_text(encoding="utf-8")
    assert "Source: `orchestration/derived/stack_priors.yaml`" in operator_summary
    for role in roles:
        assert f"| {role}" in operator_summary
        assert priors["roles"][role]["display_name"] in operator_summary
    assert "Qwen_Qwen3.6-35B-A3B-Q8_0" not in operator_summary

    system_card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert "Source: orchestration/derived/stack_priors.yaml" in system_card
    for role in roles:
        assert f"| {role} |" in system_card
        assert priors["roles"][role]["display_name"] in system_card
    assert "Qwen_Qwen3.6-35B-A3B-Q8_0" not in system_card

    from src.api.routes.dashboard_topology import _stack_prior_port_hints
    from src.api.routes.health import _stack_prior_backend_urls

    expected_port = _primary_port_for_roles(priors, roles)
    assert _stack_prior_backend_urls(config.stack_priors) == {
        "coder_escalation/frontdoor": f"http://localhost:{expected_port}"
    }
    port_hints = _stack_prior_port_hints(config.stack_priors)
    assert port_hints[expected_port].split(".", 1)[0] in roles

    from src.api.routes import chat_routing, openai_compat

    monkeypatch.setattr(openai_compat, "live_stack_role_records", lambda: priors["roles"])
    openai_roles = openai_compat.available_roles()
    assert openai_roles[:3] == ["orchestrator", "architect", "worker"]
    assert set(roles) <= set(openai_roles)
    retired_architect_role = "architect" + "_coding"
    assert retired_architect_role not in openai_roles

    monkeypatch.setattr(chat_routing, "live_stack_role_records", lambda: priors["roles"])
    assert set(chat_routing._live_heuristic_prior_roles()) == roles

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    assert q_priors.baseline_tps_by_role["frontdoor"] == 18.5
    assert q_priors.baseline_tps_by_role["coder_escalation"] == 18.5
    assert q_priors.baseline_tps_source_by_role["frontdoor"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["frontdoor"] == pytest.approx(0.874)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []

    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    swap_check_config = StackChangePipelineConfig(
        **{**approved_config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    swap_check_report = run_stack_change_pipeline(swap_check_config)

    assert swap_check_report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    promotion_step = next(step for step in swap_check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "ok"
    assert any("promotion gate ok" in detail for detail in promotion_step.details)
    assert any(step.name == "operator_summary" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "q_scorer_priors" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "runtime_attestation" and step.status == "ok" for step in swap_check_report.steps)


def test_simulated_worker_swap_updates_generated_consumers_with_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = {"worker_general", "worker_math", "toolrunner"}
    config = _config(tmp_path, mode="update", roles=roles)
    _worker_alias_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    descriptors_before = config.descriptors.read_text(encoding="utf-8")
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _swapped_worker_registry(config.lean_registry)

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
        "gemma4-26b-a4b-q8_0"
    ]
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    assert set(priors["roles"]) == roles
    for role in roles:
        assert priors["roles"][role]["model_id"] == "gemma4-26b-a4b-q8_0"
        assert priors["roles"][role]["priors"]["throughput_tps"] == 66.2
        assert priors["roles"][role]["priors"]["quality_overall"] == pytest.approx(0.9)

    operator_summary = config.operator_summary.read_text(encoding="utf-8")
    assert "Source: `orchestration/derived/stack_priors.yaml`" in operator_summary
    for role in roles:
        assert f"| {role}" in operator_summary
        assert priors["roles"][role]["display_name"] in operator_summary
    assert "gemma-4-26B-A4B-it-Q4_K_M" not in operator_summary

    system_card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert "Source: orchestration/derived/stack_priors.yaml" in system_card
    for role in roles:
        assert f"| {role} |" in system_card
        assert priors["roles"][role]["display_name"] in system_card
    assert "gemma-4-26B-A4B-it-Q4_K_M" not in system_card

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    assert q_priors.baseline_tps_by_role["worker_general"] == 66.2
    assert q_priors.baseline_tps_source_by_role["worker_general"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["worker_general"] == pytest.approx(0.9)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []
    _assert_seeding_descriptor_fallback_consumer(
        config.descriptors,
        expected_roles=roles,
        expected_tps=66.2,
    )
    expected_port = _primary_port_for_roles(priors, roles)
    _assert_text_stack_primary_port_consumers(
        config.stack_priors,
        expected_roles=roles,
        expected_port=expected_port,
    )
    _assert_worker_pool_stack_prior_consumer(
        priors["roles"],
        expected_port=expected_port,
        expected_model_path_fragment="gemma-4-26B-A4B-it-Q8_0.gguf",
        monkeypatch=monkeypatch,
    )
    _assert_factual_risk_stack_prior_consumer(
        config.stack_priors,
        expected_tiers={"worker_general": "tier_2"},
    )

    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    swap_check_config = StackChangePipelineConfig(
        **{**approved_config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    swap_check_report = run_stack_change_pipeline(swap_check_config)

    assert swap_check_report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    promotion_step = next(step for step in swap_check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "ok"
    assert any("promotion gate ok" in detail for detail in promotion_step.details)
    assert any(step.name == "operator_summary" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "q_scorer_priors" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "runtime_attestation" and step.status == "ok" for step in swap_check_report.steps)


def test_simulated_vision_swap_updates_generated_consumers_with_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = {"worker_vision", "vision_escalation"}
    config = _config(tmp_path, mode="update", roles=roles)
    _vision_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    descriptors_before = config.descriptors.read_text(encoding="utf-8")
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _swapped_vision_registry(config.lean_registry)

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
    assert {model["model_id"] for model in descriptors["models"]} == {
        "qwen2.5-vl-7b-q8_0",
        "qwen3-vl-30b-a3b-q8_0",
    }
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    assert set(priors["roles"]) == roles
    assert priors["roles"]["worker_vision"]["model_id"] == "qwen2.5-vl-7b-q8_0"
    assert priors["roles"]["worker_vision"]["priors"]["throughput_tps"] == 14.5
    assert priors["roles"]["worker_vision"]["priors"]["quality_overall"] == pytest.approx(0.84)
    assert priors["roles"]["vision_escalation"]["model_id"] == "qwen3-vl-30b-a3b-q8_0"
    assert priors["roles"]["vision_escalation"]["priors"]["throughput_tps"] == 5.9
    assert priors["roles"]["vision_escalation"]["priors"]["quality_overall"] == pytest.approx(0.92)

    operator_summary = config.operator_summary.read_text(encoding="utf-8")
    assert "Source: `orchestration/derived/stack_priors.yaml`" in operator_summary
    for role in roles:
        assert f"| {role}" in operator_summary
        assert priors["roles"][role]["display_name"] in operator_summary
    assert "Qwen2.5-VL-7B-Instruct-Q4_K_M" not in operator_summary
    assert "Qwen3-VL-30B-A3B-Instruct-Q4_K_M" not in operator_summary

    system_card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert "Source: orchestration/derived/stack_priors.yaml" in system_card
    for role in roles:
        assert f"| {role} |" in system_card
        assert priors["roles"][role]["display_name"] in system_card
    assert "Qwen2.5-VL-7B-Instruct-Q4_K_M" not in system_card
    assert "Qwen3-VL-30B-A3B-Instruct-Q4_K_M" not in system_card

    from src.api.routes.chat_pipeline.vision_stage import _vl_port_for_role
    from src.api.routes.chat_vision import _vl_url_for_port, _vl_url_for_role
    from src.api.routes.vision_serving import stack_prior_vl_ports

    assert stack_prior_vl_ports(config.stack_priors) == {
        "worker_vision": 8086,
        "vision_escalation": 8087,
    }
    assert _vl_port_for_role("worker_vision", config.stack_priors) == 8086
    assert _vl_port_for_role("vision_escalation", config.stack_priors) == 8087
    assert _vl_url_for_role("worker_vision", config.stack_priors) == "http://localhost:8086"
    assert _vl_url_for_role("vision_escalation", config.stack_priors) == "http://localhost:8087"
    assert _vl_url_for_port(8086, config.stack_priors) == "http://localhost:8086"
    assert _vl_url_for_port(8087, config.stack_priors) == "http://localhost:8087"

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    assert q_priors.baseline_tps_by_role["worker_vision"] == 14.5
    assert q_priors.baseline_tps_by_role["vision_escalation"] == 5.9
    assert q_priors.baseline_tps_source_by_role["worker_vision"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["vision_escalation"] == pytest.approx(0.92)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []
    _assert_factual_risk_stack_prior_consumer(
        config.stack_priors,
        expected_tiers={
            "worker_vision": "tier_3",
            "vision_escalation": "tier_2",
        },
    )

    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    swap_check_config = StackChangePipelineConfig(
        **{**approved_config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    swap_check_report = run_stack_change_pipeline(swap_check_config)

    assert swap_check_report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    promotion_step = next(step for step in swap_check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "ok"
    assert any("promotion gate ok" in detail for detail in promotion_step.details)
    assert any(step.name == "operator_summary" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "q_scorer_priors" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "runtime_attestation" and step.status == "ok" for step in swap_check_report.steps)


def test_simulated_ingest_swap_updates_generated_consumers_with_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = {"ingest_long_context"}
    config = _config(tmp_path, mode="update", roles=roles)
    _ingest_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    descriptors_before = config.descriptors.read_text(encoding="utf-8")
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _swapped_ingest_registry(config.lean_registry)

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
        "qwen3-next-80b-a3b-q8_0"
    ]
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    role = priors["roles"]["ingest_long_context"]
    assert set(priors["roles"]) == roles
    assert role["model_id"] == "qwen3-next-80b-a3b-q8_0"
    assert role["priors"]["throughput_tps"] == 14.2
    assert role["priors"]["quality_overall"] == pytest.approx(0.963)
    assert role["model"]["ctx_max"] == 262144
    assert role["serving"]["effective_context_tokens"] == 32768
    assert role["serving"]["launch"]["runtime"]["cache"]["context_tokens"] == 32768

    operator_summary = config.operator_summary.read_text(encoding="utf-8")
    assert "Source: `orchestration/derived/stack_priors.yaml`" in operator_summary
    assert "| ingest_long_context" in operator_summary
    assert role["display_name"] in operator_summary
    assert "Qwen3-Next-80B-A3B-Instruct-Q4_K_M" not in operator_summary

    system_card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert "Source: orchestration/derived/stack_priors.yaml" in system_card
    assert "| ingest_long_context |" in system_card
    assert role["display_name"] in system_card
    assert "Qwen3-Next-80B-A3B-Instruct-Q4_K_M" not in system_card

    from src.api.routes.dashboard_topology import _stack_prior_port_hints
    from src.api.routes.health import _stack_prior_backend_urls

    expected_port = _primary_port_for_roles(priors, roles)
    assert _stack_prior_backend_urls(config.stack_priors) == {
        "ingest_long_context": f"http://localhost:{expected_port}"
    }
    port_hints = _stack_prior_port_hints(config.stack_priors)
    assert port_hints[expected_port].split(".", 1)[0] == "ingest_long_context"
    if expected_port == 8085:
        assert not ({8185, 8285, 8385, 8485} & set(port_hints))

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    assert q_priors.baseline_tps_by_role["ingest_long_context"] == 14.2
    assert q_priors.baseline_tps_source_by_role["ingest_long_context"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["ingest_long_context"] == pytest.approx(0.963)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []
    _assert_factual_risk_stack_prior_consumer(
        config.stack_priors,
        expected_tiers={"ingest_long_context": "tier_1"},
    )

    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    swap_check_config = StackChangePipelineConfig(
        **{**approved_config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    swap_check_report = run_stack_change_pipeline(swap_check_config)

    assert swap_check_report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    promotion_step = next(step for step in swap_check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "ok"
    assert any("promotion gate ok" in detail for detail in promotion_step.details)
    assert any(step.name == "operator_summary" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "q_scorer_priors" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "runtime_attestation" and step.status == "ok" for step in swap_check_report.steps)


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
                    # 2026-06-26 v6 cutover: worker on canonical llama.cpp (v6); ik deprecated.
                    "runtime_requirements": {
                        "binary_dir": "/mnt/raid0/llm/llama.cpp/build/bin",
                        "ld_library_path": ["/mnt/raid0/llm/llama.cpp/build/src"],
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
                    # 2026-06-26 v6 cutover: MTP spec token is now 'draft-mtp'.
                    "acceleration": {"type": "speculative_decoding", "spec_type": "draft-mtp"},
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
    ] = ["stale.vision_override=int:1"]
    _write_yaml(config.stack_priors, payload)

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    assert any("serving.effective_context_tokens 8192" in error for error in report.errors)
    assert any("serving.launch.runtime does not match" in error for error in report.errors)
    assert any('"kv_type_k": "f16"' in error for error in report.errors)
    assert any('"enabled": false' in error for error in report.errors)
    assert any("qwen35moe.expert_used_count=int:8" in error for error in report.errors)
    assert any("stale.vision_override=int:1" in error for error in report.errors)
    assert any("mmproj_path" in error and "stale-mmproj" in error for error in report.errors)
