"""Tests for derived stack-prior compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.registry.stack_priors import (
    STACK_PRIORS_VERSION,
    StackPriorsCompileError,
    compile_stack_priors,
    validate_stack_priors_contract,
)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_compile_prefers_server_mode_for_shared_role_memory_and_serving(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "frontdoor": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "throughput": 24.3,
                },
                "coder_escalation": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "throughput": 24.3,
                },
            },
            "roles": {
                "frontdoor": {"memory": {"residency": "warm"}},
                "coder_escalation": {"memory": {"residency": "warm"}},
            },
        },
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "qwen3.6-35b-a3b-q8",
                    "display_name": "Qwen3.6 Q8",
                    "family": "qwen3.6",
                    "arch": "moe",
                    "params_b": 35,
                    "active_b": 3,
                    "quant": "Q8_0",
                    "mem_gb": 37,
                    "ctx_max": 131072,
                    "modalities": ["text"],
                    "role_bindings": {
                        "roles": ["frontdoor", "coder_escalation"],
                        "server_roles": ["frontdoor", "coder_escalation"],
                        "shared_mmap": True,
                    },
                    "quality": {
                        "suite_vector": {"overall": 0.929},
                        "measured": [{"date": "2026-05-04"}],
                    },
                    "speed": {"solo_96t_tps": 24.3, "measured": [{"value_tps": 24.3}]},
                    "acceleration": {"spec_type": "none"},
                    "serving": {"binary": "llama.cpp", "ports": [8070]},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"frontdoor", "coder_escalation"},
    )

    frontdoor = priors["roles"]["frontdoor"]
    coder = priors["roles"]["coder_escalation"]
    assert priors["status"] == "compiled"
    assert priors["stack_priors_version"] == STACK_PRIORS_VERSION
    assert priors["contract"]["schema"] == "epyc.stack_priors"
    assert sorted(priors["source_artifacts"]) == [
        "descriptors",
        "registry",
        "stack_manifest",
        "stack_numa",
    ]
    assert validate_stack_priors_contract(priors) == []
    assert frontdoor["priors"]["memory_cost"] == 1.0
    assert frontdoor["evidence"]["precedence"]["memory_cost"] == "server_mode.tier"
    assert coder["serving"]["endpoint"] == "http://localhost:8070"
    assert coder["serving"]["shared_mmap"] is True


def test_compile_maps_model_role_server_binding(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "model_role": "worker_general",
                    "throughput": "60.7",
                }
            },
            "roles": {"worker_general": {"memory": {"residency": "warm"}}},
        },
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "gemma4-26b-a4b-q4",
                    "role_bindings": {"roles": ["worker_general"], "server_roles": ["worker"]},
                    "quality": {"suite_vector": {"overall": 0.9}, "measured": []},
                    "speed": {"quarter_48t_tps": 60.7, "measured": []},
                    "acceleration": {"spec_type": "mtp"},
                    "serving": {"ports": [8072], "binary": "ik-pr1744"},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_general"},
    )

    worker = priors["roles"]["worker_general"]
    assert worker["serving"]["server_role"] == "worker"
    assert worker["serving"]["binding"] == "server_mode.model_role"
    assert worker["priors"]["throughput_tps"] == 60.7
    assert worker["priors"]["memory_cost"] == 1.0


def test_compile_preserves_conflicts_as_gaps_when_allowed(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {"server_mode": {}, "roles": {"worker_math": {"memory": {"residency": "warm"}}}},
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "qwen2.5-math-7b-q4",
                    "role_bindings": {"roles": ["worker_math"], "server_roles": []},
                    "quality": {"suite_vector": {}, "measured": []},
                    "speed": {"measured": []},
                    "acceleration": {},
                    "serving": {"ports": []},
                    "known_gaps": ["Role-server conflict: stale worker server binding"],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_math"},
        allow_incomplete=True,
    )

    role = priors["roles"]["worker_math"]
    assert priors["status"] == "compiled_with_gaps"
    assert role["status"] == "compiled_with_gaps"
    assert "Role-server conflict: stale worker server binding" in role["known_gaps"]
    assert role["serving"]["binding"] == "stack_manifest.alias->stack_manifest.role"


def test_compile_uses_stack_manifest_when_server_mode_is_absent(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {"server_mode": {}, "roles": {"worker_vision": {"memory": {"residency": "warm"}}}},
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "qwen2.5-vl-7b-q4",
                    "role_bindings": {"roles": ["worker_vision"], "server_roles": ["worker_vision"]},
                    "quality": {"suite_vector": {"overall": 0.81}, "measured": []},
                    "speed": {"solo_96t_tps": 20.0, "measured": []},
                    "acceleration": {},
                    "serving": {"ports": [8086]},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_vision"},
    )

    role = priors["roles"]["worker_vision"]
    assert role["deployment_status"] == "live_stack"
    assert role["serving"]["binding"] == "stack_manifest.role"
    assert role["serving"]["endpoint"] == "http://localhost:8086"
    assert role["priors"]["memory_cost"] == 1.0


def test_compile_refuses_missing_descriptor_without_allow_incomplete(tmp_path: Path) -> None:
    registry_path = _write_yaml(tmp_path / "registry.yaml", {"server_mode": {}, "roles": {}})
    descriptor_path = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})

    with pytest.raises(StackPriorsCompileError) as exc:
        compile_stack_priors(
            registry_path=registry_path,
            descriptor_path=descriptor_path,
            active_roles={"architect_coding"},
        )

    assert "architect_coding: Missing model descriptor binding" in str(exc.value)
