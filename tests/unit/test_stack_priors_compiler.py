"""Tests for derived stack-prior compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.registry.stack_priors import (
    STACK_PRIORS_VERSION,
    StackPriorsCompileError,
    canonical_stack_role_id,
    compile_stack_priors,
    live_role_primary_ports,
    _launch_runtime_record,
    live_stack_lock_role_sets,
    live_stack_role_ids,
    live_stack_role_records,
    live_stack_safe_non_stream_roles,
    live_stack_serving_slot_limits,
    live_stack_serving_url_values,
    live_warm_worker_slots,
    load_stack_priors_artifact,
    stack_prior_endpoint_port,
    stack_prior_launch_entries,
    stack_prior_launch_modes,
    stack_prior_model_mem_gb,
    stack_prior_serving_url_value,
    stack_prior_serving_ports,
    stack_prior_uses_shared_worker_launch,
    validate_stack_priors_contract,
)

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_runtime_stack_prior_helpers_fail_closed_on_missing_artifact(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    assert load_stack_priors_artifact(missing) is None
    assert live_stack_role_records(missing) == {}
    assert live_stack_role_ids(missing) == []
    assert live_warm_worker_slots(missing) == {}
    assert live_role_primary_ports(frozenset({"worker_vision"}), missing) == {}
    assert live_stack_lock_role_sets(missing) is None
    assert live_stack_safe_non_stream_roles(missing, min_mem_gb=64.0) is None
    assert live_stack_serving_url_values(missing) == {}
    assert live_stack_serving_slot_limits(missing) == {}


def test_runtime_stack_prior_helpers_project_live_roles(tmp_path: Path) -> None:
    priors = _write_yaml(
        tmp_path / "stack_priors.yaml",
        {
            "roles": {
                "worker_batch": {
                    "deployment_status": "live_stack",
                    "serving": {"tier": "warm", "slots": 4, "ports": [9123]},
                },
                "worker_general": {
                    "deployment_status": "live_stack",
                    "serving": {"tier": "hot", "slots": 4, "ports": [8072]},
                },
                "worker_vision": {
                    "deployment_status": "live_stack",
                    "serving": {"endpoint": "http://127.0.0.1:9101", "ports": [9999]},
                },
                "vision_escalation": {
                    "deployment_status": "live_stack",
                    "serving": {"ports": [9107]},
                },
                "candidate_worker": {
                    "deployment_status": "benchmark_or_candidate",
                    "serving": {"tier": "warm", "slots": 8, "ports": [9000]},
                },
            }
        },
    )

    assert sorted(live_stack_role_records(priors)) == [
        "vision_escalation",
        "worker_batch",
        "worker_general",
        "worker_vision",
    ]
    assert canonical_stack_role_id("worker_explore") == "worker_general"
    assert canonical_stack_role_id(_RETIRED_ARCHITECT_ROLE) == "architect_general"
    assert canonical_stack_role_id("unknown_role") is None
    assert live_stack_role_ids(
        priors,
        preferred_order=["frontdoor", "worker_general", "vision_escalation"],
    ) == [
        "worker_general",
        "vision_escalation",
        "worker_batch",
        "worker_vision",
    ]
    assert live_warm_worker_slots(priors) == {"worker_batch": 4}
    assert live_role_primary_ports(
        frozenset({"worker_vision", "vision_escalation", "candidate_worker"}),
        priors,
    ) == {"worker_vision": 9101, "vision_escalation": 9107}
    assert stack_prior_endpoint_port({"endpoint": "http://localhost:1234/v1"}) == 1234
    assert stack_prior_serving_ports({"ports": [1, "2", 3, None]}) == [1, 3]
    assert stack_prior_serving_url_value({"ports": [9100, 9200]}) == (
        "full:http://localhost:9100,http://localhost:9200"
    )
    assert stack_prior_serving_url_value({"endpoint": "http://localhost:9300"}) == (
        "http://localhost:9300"
    )
    assert live_stack_serving_url_values(priors) == {
        "worker_batch": "http://localhost:9123",
        "worker_general": "http://localhost:8072",
        "worker_vision": "http://localhost:9999",
        "vision_escalation": "http://localhost:9107",
    }
    assert live_stack_serving_slot_limits(priors) == {
        "http://localhost:9123": 4,
        "http://localhost:8072": 4,
    }


def test_runtime_stack_prior_policy_helpers_project_launch_and_memory(tmp_path: Path) -> None:
    priors = _write_yaml(
        tmp_path / "stack_priors.yaml",
        {
            "roles": {
                "frontdoor": {
                    "deployment_status": "live_stack",
                    "serving": {"launch": {"modes": ["default"], "entries": []}},
                    "model": {"mem_gb": 37.0},
                },
                "worker_general": {
                    "deployment_status": "live_stack",
                    "serving": {"launch": {"modes": ["worker_pool"], "entries": []}},
                    "model": {"mem_gb": 16.0},
                },
                "worker_vision": {
                    "deployment_status": "live_stack",
                    "serving": {"launch": {"entries": [{"vision_type": "worker"}]}},
                    "model": {"mem_gb": 22.0},
                },
                "architect_general": {
                    "deployment_status": "live_stack",
                    "serving": {"launch": {"modes": ["default"], "entries": []}},
                    "model": {"mem_gb": 69.0},
                },
                "candidate_large": {
                    "deployment_status": "benchmark_or_candidate",
                    "serving": {"launch": {"modes": ["worker_pool"], "entries": []}},
                    "model": {"mem_gb": 120.0},
                },
            }
        },
    )
    records = live_stack_role_records(priors)

    assert stack_prior_launch_modes(records["worker_general"]) == {"worker_pool"}
    assert stack_prior_launch_entries(records["worker_vision"]) == [{"vision_type": "worker"}]
    assert stack_prior_uses_shared_worker_launch(records["worker_general"]) is True
    assert stack_prior_uses_shared_worker_launch(records["worker_vision"]) is True
    assert stack_prior_uses_shared_worker_launch(records["frontdoor"]) is False
    assert stack_prior_model_mem_gb(records["architect_general"]) == 69.0

    lock_roles = live_stack_lock_role_sets(priors)
    assert lock_roles is not None
    heavy, light = lock_roles
    assert {"frontdoor", "architect_general"} <= heavy
    assert {"worker_general", "worker_vision"} <= light
    assert "candidate_large" not in heavy
    assert "candidate_large" not in light
    assert live_stack_safe_non_stream_roles(priors, min_mem_gb=64.0) == frozenset(
        {"architect_general"}
    )


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
                    "architecture": {"n_layers": 64, "attention_layers": 16},
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
        "orchestrator_stack",
        "registry",
        "stack_manifest",
        "stack_numa",
        "stack_paths",
        "stack_runtime",
    ]
    assert validate_stack_priors_contract(priors) == []
    assert frontdoor["priors"]["memory_cost"] == 1.0
    assert frontdoor["evidence"]["precedence"]["memory_cost"] == "server_mode.tier"
    assert frontdoor["model"]["n_layers"] == 64
    assert frontdoor["model"]["attention_layers"] == 16
    frontdoor_runtime = frontdoor["serving"]["launch"]["runtime"]
    assert frontdoor_runtime["binary_family"] == "llama.cpp"
    assert frontdoor_runtime["cache"]["slots"] == 1
    assert frontdoor_runtime["cache"]["ubatch"] == 8192
    assert frontdoor_runtime["cache"]["kv_type_k"] == "q8_0"
    assert frontdoor_runtime["cache"]["kv_type_v"] == "q8_0"
    assert frontdoor_runtime["cache"]["mlock"] is True
    assert frontdoor_runtime["cache"]["slot_save_path"].endswith("/kv_slots/frontdoor")
    assert frontdoor_runtime["flags"]["jinja"] is True
    assert frontdoor_runtime["flags"]["spec"]["enabled"] is False
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
                    "runtime_requirements": {
                        "binary_dir": "/mnt/raid0/llm/ik_llama.cpp/build/bin",
                        "ld_library_path": [
                            "/mnt/raid0/llm/ik_llama.cpp/build/src",
                            "/mnt/raid0/llm/ik_llama.cpp/build/ggml/src",
                        ],
                    },
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
    assert worker["serving"]["ports"] == [8072, 8082, 8182, 8282, 8382]
    assert worker["serving"]["effective_context_tokens"] == 16384
    assert worker["serving"]["launch"]["primary_roles"] == ["worker_general"]
    assert worker["serving"]["launch"]["modes"] == ["worker_pool"]
    assert worker["serving"]["launch"]["requirements"]["model_path"].endswith(
        "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    )
    assert worker["serving"]["launch"]["requirements"]["draft_model_path"].endswith(
        "gemma-4-26B-A4B-it-assistant-Q8_0.gguf"
    )
    runtime = worker["serving"]["launch"]["runtime"]
    assert runtime["binary_family"] == "ik-pr1744"
    assert runtime["binary_path"].endswith("/ik_llama.cpp/build/bin/llama-server")
    assert runtime["binary_dir"] == "/mnt/raid0/llm/ik_llama.cpp/build/bin"
    assert runtime["ld_library_path"] == [
        "/mnt/raid0/llm/ik_llama.cpp/build/src",
        "/mnt/raid0/llm/ik_llama.cpp/build/ggml/src",
    ]
    assert runtime["env_policy"] == "binary_override_strip_ggml"
    assert runtime["kmp_blocktime"] == 10
    assert runtime["cache"]["context_tokens"] == 16384
    assert runtime["cache"]["slots"] == 1
    assert runtime["cache"]["ubatch"] == 512
    assert runtime["cache"]["kv_type_k"] == "q8_0"
    assert runtime["cache"]["kv_type_v"] == "q8_0"
    assert runtime["cache"]["no_mmap"] is True
    assert runtime["cache"]["mlock"] is False
    assert runtime["flags"]["jinja"] is True
    assert runtime["flags"]["reasoning"] == "off"
    assert runtime["flags"]["spec"]["enabled"] is True
    assert runtime["flags"]["spec"]["type"] == "mtp"
    assert runtime["flags"]["spec"]["draft_max"] == 2
    assert runtime["flags"]["spec"]["draft_p_min"] == 0.0
    assert runtime["flags"]["spec"]["threads_draft"] == 16
    assert worker["priors"]["throughput_tps"] == 60.7
    assert worker["priors"]["memory_cost"] == 1.0


def test_compile_shared_aliases_use_runtime_descriptor(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "slots": 1,
                    "model_role": "worker_general",
                    "shared_with": ["worker_math", "toolrunner"],
                    "throughput": 60.7,
                    "numa_ports": [8082],
                }
            },
            "roles": {
                "worker_general": {"memory": {"residency": "hot"}},
                "worker_math": {"memory": {"residency": "hot"}},
                "toolrunner": {"memory": {"residency": "hot"}},
            },
        },
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "gemma4-26b-a4b-q4",
                    "role_bindings": {
                        "roles": ["worker_general", "worker_math", "toolrunner"],
                        "server_roles": ["worker"],
                        "shared_mmap": True,
                        "alias_overrides": [
                            {
                                "role": "worker_math",
                                "served_by": "worker_general",
                                "ignored_model_id": "qwen2.5-math-7b-q4_k_m",
                                "reason": "server_mode.shared_with runtime takes precedence",
                            }
                        ],
                    },
                    "quality": {"suite_vector": {"overall": 0.9}, "measured": []},
                    "speed": {"quarter_48t_tps": 60.7, "measured": []},
                    "acceleration": {
                        "spec_type": "mtp",
                        "draft_compat": ["gemma4-26b-a4b-assistant-q8"],
                    },
                    "serving": {"ports": [8072, 8082], "binary": "ik-pr1744"},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_general", "worker_math", "toolrunner"},
        allow_incomplete=True,
    )

    for role in ("worker_general", "worker_math", "toolrunner"):
        record = priors["roles"][role]
        assert record["model_id"] == "gemma4-26b-a4b-q4"
        assert record["acceleration"]["spec_type"] == "mtp"
        assert record["priors"]["throughput_tps"] == 60.7
        assert not any(gap.startswith("Role-server conflict:") for gap in record["known_gaps"])
        assert record["known_gaps"] == []
        assert record["evidence"]["alias_overrides"] == [
            {
                "role": "worker_math",
                "served_by": "worker_general",
                "ignored_model_id": "qwen2.5-math-7b-q4_k_m",
                "reason": "server_mode.shared_with runtime takes precedence",
            }
        ]

    assert priors["roles"]["worker_general"]["serving"]["ports"] == [
        8072,
        8082,
        8182,
        8282,
        8382,
    ]
    assert priors["roles"]["worker_math"]["serving"]["ports"] == [8072, 8082]
    assert priors["roles"]["toolrunner"]["serving"]["ports"] == [8072, 8082]
    assert priors["roles"]["worker_math"]["serving"]["effective_context_tokens"] == 16384
    assert priors["roles"]["toolrunner"]["serving"]["effective_context_tokens"] == 16384
    assert priors["roles"]["worker_math"]["serving"]["binding"] == "server_mode.shared_with"
    assert priors["roles"]["toolrunner"]["serving"]["binding"] == "server_mode.shared_with"
    assert priors["roles"]["worker_math"]["serving"]["launch"]["requirements"] == (
        priors["roles"]["worker_general"]["serving"]["launch"]["requirements"]
    )
    assert priors["roles"]["worker_math"]["serving"]["launch"]["runtime"] == (
        priors["roles"]["worker_general"]["serving"]["launch"]["runtime"]
    )


def test_launch_runtime_record_canonicalizes_worker_explore_kv_types() -> None:
    runtime = _launch_runtime_record(
        role="worker_explore",
        descriptor={},
        server_cfg=None,
        role_cfg=None,
        launch_cfg={
            "launch": {
                "primary_roles": ["worker_explore"],
                "modes": ["worker_pool"],
                "requirements": {"model_path": "/models/gemma.gguf"},
                "runtime": {},
            }
        },
    )

    assert runtime["cache"]["kv_type_k"] == "q8_0"
    assert runtime["cache"]["kv_type_v"] == "q8_0"
    assert runtime["cache"]["slots"] == 1


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
    assert role["serving"]["ports"] == [8072, 8082]
    assert role["serving"]["launch"]["entries"][0]["alias"] is True
    assert role["serving"]["launch"]["entries"][0]["primary_role"] == "worker_general"


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
    assert role["serving"]["slots"] == 2
    assert role["serving"]["effective_context_tokens"] == 8192
    assert role["serving"]["launch"]["entries"] == [
        {
            "port": 8086,
            "primary_role": "worker_vision",
            "mode": "vision",
            "alias": False,
            "vision_type": "worker",
        }
    ]
    assert role["serving"]["launch"]["requirements"]["model_path"].endswith(
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    )
    assert role["serving"]["launch"]["requirements"]["mmproj_path"].endswith(
        "mmproj-model-f16.gguf"
    )
    runtime = role["serving"]["launch"]["runtime"]
    assert runtime["binary_family"] == "llama.cpp"
    assert runtime["cache"]["slots"] == 2
    assert runtime["cache"]["ubatch"] is None
    assert runtime["cache"]["mlock"] is False
    assert runtime["flags"]["flash_attn"] is True
    assert runtime["flags"]["jinja"] is False
    assert runtime["flags"]["spec"]["enabled"] is False
    assert role["priors"]["memory_cost"] == 1.0


def test_compile_refuses_missing_descriptor_without_allow_incomplete(tmp_path: Path) -> None:
    registry_path = _write_yaml(tmp_path / "registry.yaml", {"server_mode": {}, "roles": {}})
    descriptor_path = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})

    with pytest.raises(StackPriorsCompileError) as exc:
        compile_stack_priors(
            registry_path=registry_path,
            descriptor_path=descriptor_path,
            active_roles={_RETIRED_ARCHITECT_ROLE},
        )

    assert f"{_RETIRED_ARCHITECT_ROLE}: Missing model descriptor binding" in str(exc.value)
