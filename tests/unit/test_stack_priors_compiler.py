"""Tests for derived stack-prior compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.registry.stack_priors import (
    STACK_PRIORS_VERSION,
    StackPriorsCompileError,
    StackPriorsModeError,
    _realized_compile_numa_mode,
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
    live_stack_slot_query_ports,
    live_warm_worker_slots,
    load_stack_priors_artifact,
    _policy_hints,
    stack_prior_endpoint_port,
    stack_prior_launch_entries,
    stack_prior_launch_modes,
    stack_prior_model_mem_gb,
    stack_prior_primary_port,
    stack_prior_serving_url_value,
    stack_prior_serving_ports,
    stack_prior_uses_shared_worker_launch,
    _launch_record,
    _serving_record,
    _stack_manifest_info,
    _server_mode_launch_requirement_overrides,
    validate_stack_priors_contract,
)

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_stack_manifest_info_defaults_to_launcher_full_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)

    _aliases, roles = _stack_manifest_info()

    assert roles["frontdoor"]["url"] == "http://localhost:8070"
    assert roles["frontdoor"]["ports"] == [8070]
    # 2026-08-01 W1 cutover: coder_escalation was http://localhost:8070 (alias on
    # frontdoor's 35B CPU process); it is now an alias on architect_general's
    # :8083 MI210 Qwen3.6-27B process.
    assert roles["coder_escalation"]["url"] == "http://localhost:8083"
    assert roles["worker_summarize"]["url"] == "http://localhost:8070"
    assert roles["worker_general"]["url"] == "http://localhost:8072"
    assert roles["worker_general"]["ports"] == [8072]
    assert roles["ingest_long_context"]["url"] == "http://localhost:8085"
    # 2026-08-01 W1 cutover: vision_escalation was its own server on
    # http://localhost:8087; it is now an alias on worker_vision's :8086 process
    # (port 8087 retired). The requirements-equality assertion below is what makes
    # it an alias rather than a second VL model, so it is kept verbatim.
    assert roles["vision_escalation"]["url"] == "http://localhost:8086"
    assert (
        roles["vision_escalation"]["launch"]["requirements"]
        == roles["worker_vision"]["launch"]["requirements"]
    )
    # 2026-08-01 W1 cutover: was "Qwen2.5-VL-7B-Instruct".
    assert "Qwen3-VL-30B-A3B-Instruct" in roles["vision_escalation"]["launch"][
        "requirements"
    ]["model_path"]


def test_stack_manifest_info_can_compile_explicit_both_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "both")

    _aliases, roles = _stack_manifest_info()

    assert roles["frontdoor"]["url"] == "http://localhost:8070"
    # Half fleet: full + 2 halves. Was [8070, 8080, 8180, 8280, 8380] (full + 4
    # quarters) before the 2026-07-30 quarter retirement in stack_numa.py; asserted
    # against the current lineup as of the 2026-08-01 W1 cutover sweep.
    assert roles["frontdoor"]["ports"] == [8070, 8080, 8180]
    assert roles["worker_general"]["url"] == "http://localhost:8072"
    # Was [8072, 8082, 8182, 8282, 8382] (full + 4 quarters); now full + 2 halves.
    assert roles["worker_general"]["ports"] == [8072, 8082, 8182]


def test_alias_roles_inherit_host_full_fleet_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    """WP-13: alias roles (shared_with_first_n) ride the host's llama-server(s) so
    they must inherit the host's FULL serving fleet, not just the instances they
    were tagged onto (shared_with_first_n_count). Host roles are unchanged."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "both")

    _aliases, roles = _stack_manifest_info()

    # Host roles keep their own fleet + primary port/url (unchanged by the fix).
    # Fleets were full + 4 quarters ([...8282, 8382] / [...8280, 8380]) until the
    # 2026-07-30 quarter retirement; they are now full + 2 halves.
    assert roles["worker_general"]["ports"] == [8072, 8082, 8182]
    assert roles["worker_general"]["port"] == 8072
    assert roles["worker_general"]["url"] == "http://localhost:8072"
    assert roles["frontdoor"]["ports"] == [8070, 8080, 8180]
    assert roles["frontdoor"]["port"] == 8070

    # Aliases now inherit the FULL host fleet (previously a single quarter).
    assert roles["worker_math"]["ports"] == roles["worker_general"]["ports"]
    assert roles["toolrunner"]["ports"] == roles["worker_general"]["ports"]
    assert roles["worker_explore"]["ports"] == roles["worker_general"]["ports"]
    # 2026-08-01 W1 cutover: coder_escalation's host changed frontdoor -> architect_general.
    assert roles["coder_escalation"]["ports"] == roles["architect_general"]["ports"]
    assert roles["worker_summarize"]["ports"] == roles["frontdoor"]["ports"]

    # Alias primary port/url still resolves to the host's primary (first) port.
    assert roles["worker_math"]["port"] == 8072
    assert roles["worker_math"]["url"] == "http://localhost:8072"
    # 2026-08-01 W1 cutover: was http://localhost:8070 / host "frontdoor".
    assert roles["coder_escalation"]["url"] == "http://localhost:8083"
    assert _aliases["worker_math"] == "worker_general"
    assert _aliases["coder_escalation"] == "architect_general"


def test_serving_record_projects_alias_host_fleet_full_url() -> None:
    """WP-13: _serving_record over an alias launch record emits the full host
    fleet as serving.ports, and stack_prior_serving_url_value emits the
    ``full:``-prefixed fleet URL."""
    host_fleet = [8072, 8082, 8182, 8282, 8382]
    alias_launch_cfg = {
        "tier": "hot",
        "port": host_fleet[0],
        "ports": list(host_fleet),
        "url": "http://localhost:8072",
        "effective_context_tokens": 16384,
        "launch": _launch_record([]),
    }

    serving = _serving_record(
        "worker_math",
        {},
        None,
        None,
        None,
        "stack_manifest.alias->worker_general",
        alias_launch_cfg,
    )

    assert serving["ports"] == host_fleet
    assert stack_prior_serving_url_value(serving) == (
        "full:http://localhost:8072,http://localhost:8082,"
        "http://localhost:8182,http://localhost:8282,http://localhost:8382"
    )


def test_regenerated_worker_math_url_byte_equals_fix_a_delegated_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CRITICAL ACCEPTANCE (WP-13 fleet convergence): a FUTURE regeneration of the
    stack priors yields a worker_math serving URL byte-identical to the operative
    Fix-A delegated field value (models.ServerConfig.worker_math ->
    _server_url_default('worker_general') -> the full worker fleet). Proves the
    generator and the operative layer converge on the same wire value, so a
    deploy of the regenerated artifact is a no-op on worker_math's URL."""
    from src.config.models import _LEGACY_SERVER_URL_FALLBACKS

    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "both")

    _aliases, roles = _stack_manifest_info()
    serving = _serving_record(
        "worker_math",
        {},
        None,
        None,
        None,
        "stack_manifest.alias->worker_general",
        roles["worker_math"],
    )
    regenerated = stack_prior_serving_url_value(serving)

    fix_a_worker_math = _LEGACY_SERVER_URL_FALLBACKS["worker_math"]
    assert regenerated == fix_a_worker_math
    # worker_math delegates its URL default to worker_general's fleet; the two
    # legacy literals are byte-identical (commit 89748805), so the generated
    # serving URL matches whichever the operative layer resolves.
    assert fix_a_worker_math == _LEGACY_SERVER_URL_FALLBACKS["worker_general"]
    # Fleet literal tracks the live lineup: was
    # "...8182,http://localhost:8282,http://localhost:8382" (full + 4 quarters)
    # before the 2026-07-30 quarter retirement; now full + 2 halves.
    assert regenerated == (
        "full:http://localhost:8072,http://localhost:8082,http://localhost:8182"
    )


def test_alias_without_host_fleet_falls_back_to_own_launch_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP-13: when the primary/host has no resolved port fleet, the alias falls
    back to its own launch ports (prior behavior preserved)."""
    from scripts.server import stack_manifest

    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    monkeypatch.setattr(
        stack_manifest,
        "ROLE_LAUNCH_META",
        {
            "synth_primary": {
                "tier": "hot",
                "mode": "default",
                "no_numa": True,
                "port": None,
                "shared_with_first_n": ["synth_alias"],
            }
        },
    )
    # synth_primary launches no server (no ports resolvable, PORT_MAP empty);
    # synth_alias appears on its own launch server at 9191.
    monkeypatch.setattr(
        stack_manifest, "HOT_SERVERS", [{"port": 9191, "roles": ["synth_alias"]}]
    )
    monkeypatch.setattr(stack_manifest, "WARM_SERVERS", [])
    monkeypatch.setattr(stack_manifest, "PORT_MAP", {})

    _aliases, roles = _stack_manifest_info()

    assert roles["synth_primary"]["ports"] == []
    # Host fleet absent -> alias keeps its own launch port (fallback preserved).
    assert roles["synth_alias"]["ports"] == [9191]
    assert roles["synth_alias"]["port"] == 9191
    assert roles["synth_alias"]["url"] == "http://localhost:9191"
    assert _aliases["synth_alias"] == "synth_primary"


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
    assert stack_prior_primary_port(
        {"endpoint": "http://localhost:1234/v1", "ports": [5678]}
    ) == 1234
    assert stack_prior_primary_port({"endpoint": "http://localhost:notaport", "ports": [5678]}) == 5678
    assert stack_prior_primary_port({"endpoint": "http://localhost:notaport"}) is None
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


def test_live_stack_slot_query_ports_filters_non_llama_and_aliases(tmp_path: Path) -> None:
    priors = _write_yaml(
        tmp_path / "stack_priors.yaml",
        {
            "roles": {
                "frontdoor": {
                    "deployment_status": "live_stack",
                    "serving": {
                        "binary": "llama.cpp",
                        "launch": {
                            "entries": [
                                {"port": 8070, "alias": False},
                                {"port": 8080, "alias": False},
                            ]
                        },
                    },
                },
                "coder_escalation": {
                    "deployment_status": "live_stack",
                    "serving": {
                        "binary": "llama.cpp",
                        "launch": {"entries": [{"port": 8070, "alias": True}]},
                    },
                },
                "worker_general": {
                    "deployment_status": "live_stack",
                    "serving": {
                        "binary": "ik-pr1744",
                        "launch": {
                            "runtime": {
                                "binary_path": "/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server"
                            },
                            "entries": [{"port": 8072, "alias": False}],
                        },
                    },
                },
                "reap_candidate": {
                    "deployment_status": "benchmark_only",
                    "serving": {
                        "binary": "llama.cpp",
                        "launch": {"entries": [{"port": 8099, "alias": False}]},
                    },
                },
                "embedder": {
                    "deployment_status": "live_stack",
                    "serving": {
                        "binary": "embedding-server",
                        "launch": {"entries": [{"port": 8090, "alias": False}]},
                    },
                },
            }
        },
    )

    assert live_stack_slot_query_ports(priors) == {
        "frontdoor": [8070, 8080],
        "worker_general": [8072],
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
    # 2026-08-01 W1 cutover: was http://localhost:8070. The endpoint is resolved
    # from the REAL launcher manifest (stack_manifest.PORT_MAP), not from this
    # synthetic registry — which is exactly the drift this assertion detects.
    # coder_escalation is now an alias on architect_general's :8083 process.
    assert coder["serving"]["endpoint"] == "http://localhost:8083"
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
                    "serving": {"ports": [8072], "binary": "llama.cpp"},
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
    assert worker["serving"]["ports"] == [8072]
    # 16384 -> 262144 (2026-08-02, operator-ratified). DERIVED from
    # server_mode.worker.serving_shape.n_ctx, which reaches worker_general through
    # the model_role binding this test is about. The 16384 it used to read was the
    # stale `roles.worker_general.model.max_context`; the gemma4 GGUF's own
    # context_length is 262144.
    assert worker["serving"]["effective_context_tokens"] == 262144
    assert worker["serving"]["launch"]["primary_roles"] == ["worker_general"]
    assert worker["serving"]["launch"]["modes"] == ["worker_pool"]
    assert worker["serving"]["launch"]["requirements"]["model_path"].endswith(
        "gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf"
    )
    assert worker["serving"]["launch"]["requirements"]["draft_model_path"].endswith(
        "gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf"
    )
    runtime = worker["serving"]["launch"]["runtime"]
    assert runtime["binary_family"] == "llama.cpp"
    assert runtime["binary_path"].endswith("/llama.cpp/build/bin/llama-server")
    # 2026-07-31: binary_dir is now ALWAYS resolved, from the role's declared device
    # via the stable kernel layer, instead of being left None and letting a CPU-only
    # literal decide. A CPU role resolves to the cpu backend; a role declaring
    # device: ROCm0 resolves to the gpu backend, which is what makes that
    # declaration reach the launcher at all.
    assert runtime["binary_dir"] is not None
    assert runtime["binary_dir"].endswith("/llama.cpp/build/bin")
    assert runtime["ld_library_path"] == []
    # A DERIVED backend must not carry the env consequences of an EXPLICIT
    # registry binary_dir override — otherwise every role silently changes policy.
    assert runtime["env_policy"] == "canonical"
    assert runtime["kmp_blocktime"] is None
    # Same 16384 -> 262144 move as serving.effective_context_tokens above; this is
    # the runtime half of the same number, and the two must not diverge.
    assert runtime["cache"]["context_tokens"] == 262144
    assert runtime["cache"]["slots"] == 1
    assert runtime["cache"]["ubatch"] == 512
    assert runtime["cache"]["kv_type_k"] == "q8_0"
    assert runtime["cache"]["kv_type_v"] == "q8_0"
    assert runtime["cache"]["no_mmap"] is True
    assert runtime["cache"]["mlock"] is False
    assert runtime["flags"]["jinja"] is True
    assert runtime["flags"]["reasoning"] == "off"
    assert runtime["flags"]["spec"]["enabled"] is True
    assert runtime["flags"]["spec"]["type"] == "draft-mtp"
    assert runtime["flags"]["spec"]["draft_max"] == 2
    assert runtime["flags"]["spec"]["draft_p_min"] == 0.0
    assert runtime["flags"]["spec"]["threads_draft"] == 16
    assert worker["priors"]["throughput_tps"] == 60.7
    assert worker["priors"]["memory_cost"] == 1.0


def test_compile_prefers_server_mode_launch_requirement_paths(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "model_role": "worker_general",
                    "model_path": "/models/gemma-4-26B-A4B-it-Q8_0.gguf",
                    "draft_model_path": "/models/gemma-4-26B-A4B-it-draft-Q8_0.gguf",
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
                    "model_id": "gemma4-26b-a4b-q8",
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

    launch = priors["roles"]["worker_general"]["serving"]["launch"]
    assert launch["requirements"]["model_path"] == "/models/gemma-4-26B-A4B-it-Q8_0.gguf"
    assert (
        launch["requirements"]["draft_model_path"]
        == "/models/gemma-4-26B-A4B-it-draft-Q8_0.gguf"
    )
    assert (
        launch["runtime"]["flags"]["spec"]["draft_model_path"]
        == "/models/gemma-4-26B-A4B-it-draft-Q8_0.gguf"
    )


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

    assert priors["roles"]["worker_general"]["serving"]["ports"] == [8072]
    assert priors["roles"]["worker_math"]["serving"]["ports"] == [8072]
    assert priors["roles"]["toolrunner"]["serving"]["ports"] == [8072]
    # 16384 -> 262144 (2026-08-02): both aliases ride worker's process and inherit
    # its serving_shape.n_ctx, which is the property this test asserts.
    assert priors["roles"]["worker_math"]["serving"]["effective_context_tokens"] == 262144
    assert priors["roles"]["toolrunner"]["serving"]["effective_context_tokens"] == 262144
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


def test_launch_runtime_record_derives_reasoning_off_from_thinking_prior() -> None:
    runtime = _launch_runtime_record(
        role="frontdoor",
        descriptor={
            "acceleration": {
                "enable_thinking": False,
                "thinking_control": {
                    "mode": "toggle_off",
                    "source": "model.disable_thinking",
                },
            },
        },
        server_cfg={},
        role_cfg={},
        launch_cfg={
            "effective_context_tokens": 8192,
            "launch": {
                "primary_roles": ["frontdoor"],
                "modes": ["default"],
                "requirements": {},
                "runtime": {},
            },
        },
    )

    assert runtime["flags"]["reasoning"] == "off"


def test_launch_runtime_record_does_not_force_reasoning_when_template_ignores_toggle() -> None:
    runtime = _launch_runtime_record(
        role="ingest_long_context",
        descriptor={
            "acceleration": {
                "enable_thinking": None,
                "thinking_control": {
                    "mode": "template_ignores_enable_thinking",
                    "source": "registry note",
                },
            },
        },
        server_cfg={},
        role_cfg={},
        launch_cfg={
            "effective_context_tokens": 131072,
            "launch": {
                "primary_roles": ["ingest_long_context"],
                "modes": ["default"],
                "requirements": {},
                "runtime": {},
            },
        },
    )

    assert runtime["flags"]["reasoning"] is None
    # 1 -> 2. This fixture passes an empty server_cfg, so `slots` comes from the
    # manifest's declared `launch_shape.fallback_slots.default` (2). It read 1
    # only because the SERIAL_ROLES clamp — an ADMISSION policy applied to a
    # SERVING number — used to rewrite it here; that clamp was removed on
    # 2026-08-02 when the operator ratified explicit per-instance slot counts.
    assert runtime["cache"]["slots"] == 2


def test_launch_runtime_record_projects_ap3b_spec_numeric_controls() -> None:
    runtime = _launch_runtime_record(
        role="worker_general",
        descriptor={},
        server_cfg={
            "acceleration": {
                "type": "speculative_decoding",
                "spec_type": "ngram-mod,draft-mtp",
                "draft_max": 5,
                "draft_min": 0,
                "draft_p_min": 0.125,
                "draft_p_split": 0.5,
                "threads_draft": 12,
                "ngram_mod_n_min": 0,
                "ngram_mod_n_max": 96,
                "ngram_mod_n_match": 16,
            }
        },
        role_cfg=None,
        launch_cfg={
            "launch": {
                "primary_roles": ["worker_general"],
                "modes": ["worker_pool"],
                "entries": [{"worker_type": "explore"}],
                "requirements": {
                    "model_path": "/models/gemma.gguf",
                    "draft_model_path": "/models/draft.gguf",
                },
                "runtime": {},
            }
        },
    )

    spec = runtime["flags"]["spec"]
    assert spec["enabled"] is True
    assert spec["type"] == "ngram-mod,draft-mtp"
    assert spec["draft_model_path"] == "/models/draft.gguf"
    assert spec["draft_max"] == 5
    assert spec["draft_min"] == 0
    assert spec["draft_p_min"] == 0.125
    assert spec["draft_p_split"] == 0.5
    assert spec["threads_draft"] == 12
    assert spec["ngram_mod_n_min"] == 0
    assert spec["ngram_mod_n_max"] == 96
    assert spec["ngram_mod_n_match"] == 16


def test_launch_runtime_record_does_not_inject_vision_escalation_override() -> None:
    launch_cfg = {
        "launch": {
            "primary_roles": ["vision_escalation"],
            "modes": ["vision"],
            "entries": [{"vision_type": "escalation"}],
            "requirements": {
                "model_path": "/models/qwen2.5-vl.gguf",
                "mmproj_path": "/models/qwen2.5-vl-mmproj.gguf",
            },
            "runtime": {},
        }
    }

    runtime = _launch_runtime_record(
        role="vision_escalation",
        descriptor={},
        server_cfg=None,
        role_cfg={
            "server": {"device": "ROCm0", "reasoning": "off"},
            "acceleration": {"type": "baseline"},
        },
        launch_cfg=launch_cfg,
    )

    assert runtime["flags"]["device"] == "ROCm0"
    assert runtime["flags"]["reasoning"] == "off"
    assert runtime["flags"]["override_kv"] == []

    runtime = _launch_runtime_record(
        role="vision_escalation",
        descriptor={},
        server_cfg=None,
        role_cfg={
            "acceleration": {
                "type": "moe_expert_reduction",
                "override_key": "qwen3vlmoe.expert_used_count",
                "experts": 4,
            }
        },
        launch_cfg=launch_cfg,
    )

    assert runtime["flags"]["override_kv"] == ["qwen3vlmoe.expert_used_count=int:4"]


def test_launch_runtime_record_accepts_role_level_runtime_requirements() -> None:
    runtime = _launch_runtime_record(
        role="vision_escalation",
        descriptor={},
        server_cfg=None,
        role_cfg={
            "server": {
                "device": "ROCm0",
                "reasoning": "off",
                "runtime_requirements": {
                    "binary_dir": "/tmp/v7-hip/bin",
                    "ld_library_path": ["/tmp/v7-hip/bin"],
                },
            },
            "acceleration": {"type": "baseline"},
        },
        launch_cfg={
            "launch": {
                "primary_roles": ["vision_escalation"],
                "modes": ["vision"],
                "entries": [{"vision_type": "escalation"}],
                "requirements": {
                    "model_path": "/models/minicpm.gguf",
                    "mmproj_path": "/models/minicpm-mmproj.gguf",
                },
                "runtime": {},
            }
        },
    )

    assert runtime["binary_dir"] == "/tmp/v7-hip/bin"
    assert runtime["binary_path"] == "/tmp/v7-hip/bin/llama-server"
    assert runtime["ld_library_path"] == ["/tmp/v7-hip/bin"]
    assert runtime["env_policy"] == "binary_override_strip_ggml"


def test_server_mode_requirement_overrides_keep_shared_alias_on_served_model() -> None:
    requirements = _server_mode_launch_requirement_overrides(
        "worker_math",
        {
            "model_role": "worker_general",
            "model": "gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf",
            "draft_model": "gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf",
        },
        {
            "model": {
                "path": (
                    "lmstudio-community/Qwen2.5-Math-7B-Instruct-GGUF/"
                    "Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf"
                )
            }
        },
    )

    assert requirements["model_path"] == (
        "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf"
    )
    assert requirements["draft_model_path"] == (
        "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf"
    )


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
    assert role["serving"]["ports"] == [8072]
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
    # 2026-08-02 phase 2: was 2, from the `mode == "vision"` literal
    # `1 if vision_type == "escalation" else 2`. That literal is the exhibit for
    # this refactor — it shadowed `server_mode.worker_vision.slots: 1` and made
    # the launcher emit `-np 2` for a role declaring 1. With `server_mode` empty,
    # as here, the value now comes from the DECLARED
    # `launch_shape.fallback_slots.vision` (1) instead of a literal in the
    # compiler. A registry that declares slots still wins over it — that is the
    # `_runtime_flag_int_prior` path, covered by the server_mode tests above.
    assert role["serving"]["slots"] == 1
    # 8192 -> 16384 (2026-08-01 W1) -> 65536 (2026-08-02). LAUNCH_CONTEXT_TOKENS is
    # now DERIVED from server_mode.worker_vision.serving_shape.n_ctx rather than
    # declared launcher-side, so this reads the ratified value even though this
    # fixture's own `server_mode` is empty.
    assert role["serving"]["effective_context_tokens"] == 65536
    # `cpu_shape_class` and per-entry `slots` are new (2026-08-02): the entry is
    # now self-describing — where it runs, on what shape, with how many slots —
    # which is what lets one role's full and halves carry different `-np`.
    # worker_vision has a single GPU_HOST_LANE instance, so its class is that.
    # `slots` is 1 here because this fixture declares no registry slots at all and
    # falls through to launch_shape.fallback_slots.vision, as asserted above.
    assert role["serving"]["launch"]["entries"] == [
        {
            "port": 8086,
            "primary_role": "worker_vision",
            "mode": "vision",
            "alias": False,
            "cpu_shape_class": "gpu_host_lane",
            "vision_type": "worker",
            "slots": 1,
        }
    ]
    # 2026-08-01 W1 cutover: was Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf +
    # mmproj-model-f16.gguf; the VL lane moved to Qwen3-VL-30B-A3B on MI210.
    assert role["serving"]["launch"]["requirements"]["model_path"].endswith(
        "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
    )
    assert role["serving"]["launch"]["requirements"]["mmproj_path"].endswith(
        "mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"
    )
    runtime = role["serving"]["launch"]["runtime"]
    assert runtime["binary_family"] == "llama.cpp"
    # Same phase-2 move as serving.slots above: 2 was the `mode == "vision"`
    # literal, 1 is launch_shape.fallback_slots.vision.
    assert runtime["cache"]["slots"] == 1
    assert runtime["cache"]["ubatch"] is None
    assert runtime["cache"]["mlock"] is False
    # Still None, and deliberately so: `device` is a server_mode field and this
    # test's registry has an EMPTY server_mode. The live compile does emit ROCm0
    # for worker_vision after the 2026-08-01 W1 cutover (from
    # server_mode.worker_vision.device); that path is witnessed by
    # tests/unit/test_build_server_command_helpers.py, not here. Asserting ROCm0
    # here would assert the launcher can invent a device the registry never
    # declared, which is the opposite of the contract.
    assert runtime["flags"]["device"] is None
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


# --------------------------------------------------------------------------- #
# ESC-8 Fix 6: priors compile must not read the ambient default-full env.       #
# --------------------------------------------------------------------------- #

_FIX6_FULL_HOST_PORTS = {8070, 8072, 8085}


def _quarters_connect(_host: str, port: int) -> bool:
    """Quarters-only fleet: the full host ports are dead, everything else live."""
    return port not in _FIX6_FULL_HOST_PORTS


def _fulls_connect(_host: str, _port: int) -> bool:
    return True


def _all_dead_connect(_host: str, _port: int) -> bool:
    return False


def test_realized_compile_numa_mode_env_unset_uses_realized() -> None:
    assert _realized_compile_numa_mode(environ={}, connect=_quarters_connect) == "quarter"
    assert _realized_compile_numa_mode(environ={}, connect=_fulls_connect) in {"full", "both"}


def test_realized_compile_numa_mode_env_contradiction_prefers_realized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        mode = _realized_compile_numa_mode(
            environ={"ORCHESTRATOR_STACK_NUMA_MODE": "full"},
            connect=_quarters_connect,
        )
    assert mode == "quarter"
    assert any("contradicts the realized fleet" in rec.message for rec in caplog.records)


def test_realized_compile_numa_mode_env_agreement_kept() -> None:
    assert (
        _realized_compile_numa_mode(
            environ={"ORCHESTRATOR_STACK_NUMA_MODE": "quarter"},
            connect=_quarters_connect,
        )
        == "quarter"
    )


def test_realized_compile_numa_mode_refuses_without_signal() -> None:
    with pytest.raises(StackPriorsModeError):
        _realized_compile_numa_mode(environ={}, connect=_all_dead_connect)
    # An explicit env does NOT rescue a no-signal probe (unverifiable fleet).
    with pytest.raises(StackPriorsModeError):
        _realized_compile_numa_mode(
            environ={"ORCHESTRATOR_STACK_NUMA_MODE": "full"},
            connect=_all_dead_connect,
        )


def _worker_math_conflict_paths(tmp_path: Path) -> tuple[Path, Path]:
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
    return registry_path, descriptor_path


def test_compile_require_realized_mode_derives_quarter_lineup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V4: a clean-shell compile with a sub-full-live probe derives the sub-full
    lineup (alias inherits the host's sub-full fleet), never the dead full port.

    The sub-full shape is whatever the role declares in NUMA_CONFIG: quarters when
    this test was written, halves since the 2026-07-30 quarter retirement. The
    invariant under test — the dead full host port must never appear — is
    shape-independent."""
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    registry_path, descriptor_path = _worker_math_conflict_paths(tmp_path)

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_math"},
        allow_incomplete=True,
        require_realized_mode=True,
        connect=_quarters_connect,
    )

    ports = priors["roles"]["worker_math"]["serving"]["ports"]
    assert 8072 not in ports  # the dead full host port must not appear
    # Was [8082, 8182, 8282, 8382] (4 quarters) before the 2026-07-30 quarter
    # retirement; worker_general's sub-full lineup is now 2 halves.
    assert ports == [8082, 8182]


def test_compile_require_realized_mode_refuses_without_signal(tmp_path: Path) -> None:
    """V5: a compile with no realized signal refuses rather than defaulting full."""
    registry_path, descriptor_path = _worker_math_conflict_paths(tmp_path)
    with pytest.raises(StackPriorsModeError):
        compile_stack_priors(
            registry_path=registry_path,
            descriptor_path=descriptor_path,
            active_roles={"worker_math"},
            allow_incomplete=True,
            require_realized_mode=True,
            connect=_all_dead_connect,
        )


def test_compile_default_does_not_probe_realized_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realized-fleet resolution is strictly opt-in: a plain compile never
    calls it (so it can neither refuse nor probe sockets by default)."""
    import src.registry.stack_priors as sp

    def _boom(**_kw):
        raise AssertionError("resolver must not run without require_realized_mode")

    monkeypatch.setattr(sp, "_realized_compile_numa_mode", _boom)
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    registry_path, descriptor_path = _worker_math_conflict_paths(tmp_path)

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_math"},
        allow_incomplete=True,
    )
    # Legacy full-mode default is unchanged.
    assert priors["roles"]["worker_math"]["serving"]["ports"] == [8072]


def test_policy_hints_returns_none_thresholds_when_model_memory_unknown() -> None:
    hints = _policy_hints({"launch": {"modes": ["default"], "entries": []}}, {})
    assert hints["lock_class"] == "exclusive"
    assert hints["contention_class"] == "heavy"
    assert hints["tap_safe_non_stream"] is None
    assert hints["high_cost"] is None
    assert hints["model_mem_gb"] is None
    assert hints["source"] == "stack_priors.compile"
    assert hints["thresholds"] == {
        "tap_safe_non_stream_min_mem_gb": 64.0,
        "high_cost_min_mem_gb": 60.0,
    }


def test_policy_hints_classify_shared_worker_as_light_and_low_cost() -> None:
    serving = {"launch": {"modes": ["worker_pool"], "entries": []}}
    hints = _policy_hints(serving, {"mem_gb": 37})
    assert hints["lock_class"] == "shared"
    assert hints["contention_class"] == "light"
    assert hints["tap_safe_non_stream"] is False
    assert hints["high_cost"] is False
    assert hints["model_mem_gb"] == 37.0


def test_policy_hints_flag_heavy_high_cost_role() -> None:
    serving = {"launch": {"modes": ["default"], "entries": []}}
    hints = _policy_hints(serving, {"mem_gb": 238})
    assert hints["lock_class"] == "exclusive"
    assert hints["contention_class"] == "heavy"
    assert hints["tap_safe_non_stream"] is True
    assert hints["high_cost"] is True


def test_compile_projects_ctx_model_max_and_policy_hints(tmp_path: Path) -> None:
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
                    "mem_gb": 37,
                    "ctx_max": 16384,
                    "ctx_model_max": 131072,
                    "role_bindings": {
                        "roles": ["worker_general"],
                        "server_roles": ["worker"],
                    },
                    "quality": {"suite_vector": {"overall": 0.9}, "measured": []},
                    "speed": {"quarter_48t_tps": 60.7, "measured": []},
                    "acceleration": {"spec_type": "mtp"},
                    "serving": {"ports": [8072], "binary": "llama.cpp"},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_general"},
        allow_incomplete=True,
    )

    worker = priors["roles"]["worker_general"]
    # 620: model-native context is projected alongside the effective ctx_max.
    assert worker["model"]["ctx_model_max"] == 131072
    assert worker["model"]["ctx_max"] == 16384
    # Additive fields must not break the generated contract shape.
    assert validate_stack_priors_contract(priors) == []
    # 622: policy hints projected. The worker rides a shared worker_pool launch,
    # so it is a light/shared role; 37 GB is below both memory thresholds.
    policy = worker["policy"]
    assert policy["lock_class"] == "shared"
    assert policy["contention_class"] == "light"
    assert policy["tap_safe_non_stream"] is False
    assert policy["high_cost"] is False
    assert policy["model_mem_gb"] == 37.0
