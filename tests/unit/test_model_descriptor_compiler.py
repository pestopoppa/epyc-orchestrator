"""Tests for compiling model-capability descriptors from registries."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.registry.model_descriptors import (
    DescriptorCompileError,
    _canonical_model_id,
    compile_model_descriptors,
    write_model_descriptors,
)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_canonical_model_id_matches_stable_descriptor_policy() -> None:
    assert _canonical_model_id("Qwen3.6-35B-A3B-Q8_0", "Q8_0") == "qwen3.6-35b-a3b-q8_0"
    assert (
        _canonical_model_id("gemma-4-26B-A4B-it-Q4_K_M", "Q4_K_M")
        == "gemma4-26b-a4b-q4_k_m"
    )
    assert (
        _canonical_model_id("Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf", "Q4_K_M")
        == "qwen3-next-80b-a3b-q4_k_m"
    )
    assert (
        _canonical_model_id("Qwen2.5-VL-7B-Instruct", "Q4_K_M")
        == "qwen2.5-vl-7b-q4_k_m"
    )


def test_compile_merges_same_model_roles(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "model_registry.yaml",
        {
            "server_mode": {
                "frontdoor": {
                    "port": 8070,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_role": "frontdoor",
                    "memory_gb": 37,
                    "throughput": 24.3,
                    "benchmark_score": "170/183 (92.9%)",
                    "benchmark_date": "2026-05-04",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8080],
                },
                "coder_escalation": {
                    "port": 8070,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_role": "coder_escalation",
                    "memory_gb": 37,
                    "throughput": 24.3,
                    "benchmark_score": "29/30 (97%)",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
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
                    "performance": {"quality_pct": 93, "baseline_tps": 24.3},
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True},
                },
                "coder_escalation": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q8_0",
                        "quant": "Q8_0",
                        "architecture": "qwen35moe",
                        "size_gb": 37,
                        "ctx_max": 131072,
                    },
                    "performance": {"coder_suite": "29/30 (97%)"},
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True},
                },
                "worker_summarize": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q8_0",
                        "quant": "Q8_0",
                        "architecture": "qwen35moe",
                        "size_gb": 37,
                        "ctx_max": 131072,
                    },
                    "performance": {"long_context": "27/27 (100%)"},
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True},
                },
            },
        },
    )

    compiled = compile_model_descriptors(
        lean_registry_path=registry_path,
        research_registry_path=None,
        active_roles={"frontdoor", "coder_escalation", "worker_summarize"},
    )

    assert compiled["status"] == "compiled"
    assert len(compiled["models"]) == 1
    model = compiled["models"][0]
    assert model["model_id"] == "qwen3.6-35b-a3b-q8_0"
    assert model["role_bindings"]["roles"] == [
        "coder_escalation",
        "frontdoor",
        "worker_summarize",
    ]
    assert model["role_bindings"]["server_roles"] == ["coder_escalation", "frontdoor"]
    assert model["quality"]["suite_vector"]["overall"] == 0.93
    assert model["quality"]["suite_vector"]["coder"] == 0.9667
    assert not any("server" in gap or "port" in gap for gap in model["known_gaps"])


def test_compile_refuses_missing_load_bearing_fields(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "model_registry.yaml",
        {
            "server_mode": {},
            "roles": {
                "worker_math": {
                    "model": {
                        "name": "Qwen2.5-Math-7B-Instruct",
                        "quant": "Q4_K_M",
                        "architecture": "dense",
                        "size_gb": 4.4,
                    },
                    "acceleration": {"type": "speculative_decoding", "draft_role": "draft"},
                    "memory": {"pinned": True},
                }
            },
        },
    )

    with pytest.raises(DescriptorCompileError) as exc:
        compile_model_descriptors(
            lean_registry_path=registry_path,
            research_registry_path=None,
            active_roles={"worker_math"},
        )

    assert "Missing quality suite_vector evidence" in str(exc.value)
    assert "Missing server_mode binding" in str(exc.value)


def test_allow_incomplete_records_known_gaps_and_writes(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "model_registry.yaml",
        {
            "server_mode": {},
            "roles": {
                "worker_math": {
                    "model": {
                        "name": "Qwen2.5-Math-7B-Instruct",
                        "quant": "Q4_K_M",
                        "architecture": "dense",
                        "size_gb": 4.4,
                    },
                    "acceleration": {"type": "speculative_decoding", "draft_role": "draft"},
                    "memory": {"pinned": True},
                }
            },
        },
    )
    output = tmp_path / "model_descriptors.yaml"

    compiled = write_model_descriptors(
        output,
        lean_registry_path=registry_path,
        research_registry_path=None,
        active_roles={"worker_math"},
        allow_incomplete=True,
    )

    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert compiled["status"] == "compiled_with_gaps"
    assert loaded["models"][0]["known_gaps"]
    assert "worker_math" in loaded["models"][0]["role_bindings"]["roles"]


def test_compile_enriches_context_and_thinking_from_research_registry(tmp_path: Path) -> None:
    lean_path = _write_yaml(
        tmp_path / "lean.yaml",
        {
            "server_mode": {
                "frontdoor": {
                    "port": 8070,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_role": "frontdoor",
                    "throughput": 24.3,
                    "benchmark_score": "170/183 (93%)",
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            },
            "roles": {
                "frontdoor": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q8_0",
                        "quant": "Q8_0",
                        "architecture": "qwen35moe",
                        "size_gb": 37,
                    },
                    "performance": {
                        "quality_pct": 93,
                        "benchmark_date": "2026-05-04",
                    },
                }
            },
        },
    )
    research_path = _write_yaml(
        tmp_path / "research.yaml",
        {
            "roles": {
                "qwen36_q8_0": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B",
                        "path": "/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                        "quant": "Q8_0",
                        "max_context": 262144,
                        "disable_thinking": True,
                    }
                }
            }
        },
    )

    compiled = compile_model_descriptors(
        lean_registry_path=lean_path,
        research_registry_path=research_path,
        active_roles={"frontdoor"},
    )

    model = compiled["models"][0]
    assert model["ctx_max"] == 262144
    assert model["acceleration"]["enable_thinking"] is False
    assert model["quality"]["measured"][0]["date"] == "2026-05-04"
    assert compiled["status"] == "compiled"


def test_compile_uses_role_endpoint_for_dedicated_vision_role(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "model_registry.yaml",
        {
            "server_mode": {},
            "roles": {
                "worker_vision": {
                    "port": 8086,
                    "model": {
                        "name": "Qwen2.5-VL-7B-Instruct",
                        "quant": "Q4_K_M",
                        "architecture": "dense",
                        "size_gb": 4.4,
                        "ctx_max": 32768,
                        "mmproj_path": "/models/mmproj.gguf",
                    },
                    "candidate_roles": ["vision"],
                    "server": {"endpoint": "http://localhost:8086"},
                    "acceleration": {
                        "type": "baseline",
                        "disallowed": ["speculative_decoding"],
                    },
                    "performance": {
                        "vl_score": "11/12 (92%)",
                        "benchmark_date": "2026-03-04",
                    },
                }
            },
        },
    )

    compiled = compile_model_descriptors(
        lean_registry_path=registry_path,
        research_registry_path=None,
        active_roles={"worker_vision"},
        allow_incomplete=True,
    )

    model = compiled["models"][0]
    assert model["serving"]["ports"] == [8086]
    assert model["serving"]["numa_policy"] == "role_endpoint_binding"
    assert model["modalities"] == ["text", "vision"]
    assert model["quality"]["suite_vector"]["overall"] == 0.9167
    assert model["quality"]["suite_vector"]["vision_language"] == 0.9167
    assert "Missing serving port binding" not in model["known_gaps"]
    assert "Missing server_mode binding" not in model["known_gaps"]


def test_compile_preserves_benchmark_only_server_model_role(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "model_registry.yaml",
        {
            "server_mode": {
                "reap_25b": {
                    "url": "http://localhost:8196",
                    "port": 8196,
                    "slots": 1,
                    "model_role": "reap_25b_frontdoor",
                    "model": "cerebras_Qwen3-Coder-REAP-25B-A3B-Q4_K_M.gguf",
                    "model_path": "/models/cerebras_Qwen3-Coder-REAP-25B-A3B-Q4_K_M.gguf",
                    "memory_gb": 15,
                    "tier": "warm",
                    "throughput": 39.6,
                    "benchmark_date": "2026-03-24",
                    "acceleration": {
                        "type": "speculative_decoding",
                        "draft_max": 24,
                        "lookup": True,
                    },
                }
            },
            "roles": {
                "reap_25b_frontdoor": {
                    "model": {
                        "name": "REAP-Qwen3-Coder-25B-A3B-Q4_K_M",
                        "quant": "Q4_K_M",
                        "architecture": "moe",
                        "size_gb": 15,
                    },
                    "candidate_roles": ["frontdoor", "coder", "worker"],
                    "acceleration": {
                        "type": "speculative_decoding",
                        "draft_role": "qwen3-coder-0.75b-q4_0",
                        "draft_max": 24,
                        "lookup": True,
                    },
                    "performance": {
                        "baseline_tps": 39.6,
                        "optimized_tps": 39.6,
                        "benchmark_date": "2026-03-24",
                    },
                    "memory": {"residency": "warm", "pinned": False},
                }
            },
        },
    )

    compiled = compile_model_descriptors(
        lean_registry_path=registry_path,
        research_registry_path=None,
        active_roles={"frontdoor"},
        allow_incomplete=True,
    )

    model = compiled["models"][0]
    assert model["model_id"] == "reap-qwen3-coder-25b-a3b-q4_k_m"
    assert model["role_bindings"]["roles"] == ["reap_25b_frontdoor"]
    assert model["role_bindings"]["server_roles"] == ["reap_25b"]
    assert model["serving"]["ports"] == [8196]
    assert model["speed"]["solo_96t_tps"] == 39.6
    assert model["known_gaps"] == [
        "Missing enable_thinking compatibility evidence",
        "Missing quality suite_vector evidence",
        "Missing structured ctx_max",
    ]
