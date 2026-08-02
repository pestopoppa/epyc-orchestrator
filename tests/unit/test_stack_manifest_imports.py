"""Tests for stack_manifest + stack_paths + stack_commands extraction.

Per the Tranche-7 handoff: "Move constants only after adding tests that import
both `scripts.server.orchestrator_stack` and top-level `orchestrator_stack`."

These tests guard that contract — they verify the registry compiler's fallback
import path (`from orchestrator_stack import ROLE_LAUNCH_META`) keeps working
after the manifest extraction, and that the same names resolve via both
import surfaces.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "server"))


# ----- top-level orchestrator_stack import (registry_compiler fallback path) -----


def test_top_level_orchestrator_stack_import_resolves_role_launch_meta() -> None:
    """src/registry/registry_compiler.py:266 does `from orchestrator_stack import ROLE_LAUNCH_META`
    after `sys.path.insert(0, '/.../scripts/server')`. Must keep working."""
    top_level = importlib.import_module("orchestrator_stack")
    assert hasattr(top_level, "ROLE_LAUNCH_META")
    assert isinstance(top_level.ROLE_LAUNCH_META, dict)
    assert "frontdoor" in top_level.ROLE_LAUNCH_META


def test_top_level_and_package_path_return_same_role_launch_meta() -> None:
    """Both `import orchestrator_stack` and `import scripts.server.orchestrator_stack`
    are valid; they should expose the SAME constant by value."""
    top = importlib.import_module("orchestrator_stack")
    pkg = importlib.import_module("scripts.server.orchestrator_stack")
    assert top.ROLE_LAUNCH_META == pkg.ROLE_LAUNCH_META


# ----- core manifest constants are re-exported from orchestrator_stack -----


@pytest.mark.parametrize(
    "name",
    [
        "PORT_MAP",
        "ROLE_LAUNCH_META",
        "HOT_ROLES",
        "SERIAL_ROLES",
        "NUMA_REPLICA_PORTS",
        "HOT_SERVERS",
        "WARM_SERVERS",
        "EMBEDDING_MODEL_PATH",
        "EMBEDDER_PORTS",
        "WORKER_POOL_MODELS",
        "EXPLORE_DRAFT_MODEL",
        "VISION_WORKER_MODEL",
        "VISION_WORKER_MMPROJ",
        "VISION_ESCALATION_MODEL",
        "VISION_ESCALATION_MMPROJ",
        "DEV_MODEL",
        "DEV_MODEL_PATH",
        "ORCHESTRATOR_PROFILES",
        "DOCKER_SERVICES",
        "validate_model_paths",
        "validate_against_registry",
        "_build_servers_from_classification",
        "_validate_role_classification",
        "_filter_by_numa_mode",
    ],
)
def test_manifest_name_reexported(name: str) -> None:
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    assert hasattr(stack, name), f"{name} not re-exported from orchestrator_stack"


def test_gate3_tool_telemetry_profile_sets_required_api_env() -> None:
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    profile = stack.ORCHESTRATOR_PROFILES["gate3-tool-telemetry"]

    assert profile["AUTOPILOT_TOOL_SENTINELS"] == "1"
    assert profile["ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT"] == "1"


# ----- path/binary constants re-exported from stack_paths via orchestrator_stack -----


@pytest.mark.parametrize(
    "name",
    [
        "_PATHS",
        "STATE_FILE",
        "LLAMA_SERVER",
        "LLAMA_MATH_TOOLS",
        "LLAMA_SERVER_V2",
        "_V2_ROLES",
        "LOG_DIR",
        "SLOT_SAVE_DIR",
        "_get_paths",
        "_HEALTH_SERVER_STARTUP",
        "_HEALTH_VISION_SERVER",
        "_HEALTH_WORKER_SERVER",
    ],
)
def test_path_name_reexported(name: str) -> None:
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    assert hasattr(stack, name), f"{name} not re-exported from orchestrator_stack"


# ----- cmd_* lazy __getattr__ resolution -----


@pytest.mark.parametrize("cmd_name", ["cmd_start", "cmd_stop", "cmd_reload", "cmd_status"])
def test_cmd_name_resolves_lazily_via_getattr(cmd_name: str) -> None:
    """orchestrator_stack.cmd_X resolves via module __getattr__ to stack_commands."""
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    fn = getattr(stack, cmd_name)
    assert callable(fn)


@pytest.mark.parametrize("cmd_name", ["cmd_start", "cmd_stop", "cmd_reload", "cmd_status"])
def test_cmd_name_resolves_to_stack_commands_function(cmd_name: str) -> None:
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    commands = importlib.import_module("scripts.server.stack_commands")

    assert getattr(stack, cmd_name) is getattr(commands, cmd_name)


def test_orchestrator_stack_unknown_attr_raises() -> None:
    """__getattr__ must reject unknown names (not return cmd_* for everything)."""
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    with pytest.raises(AttributeError, match="does_not_exist_xyz"):
        stack.does_not_exist_xyz


# ----- HOT_SERVERS / WARM_SERVERS computed correctly -----


def test_hot_servers_computed_at_module_load() -> None:
    """_build_servers_from_classification is run at stack_manifest module load.
    HOT_SERVERS should be a non-empty list of server dicts."""
    from scripts.server.stack_manifest import HOT_SERVERS

    assert isinstance(HOT_SERVERS, list)
    assert len(HOT_SERVERS) > 0
    for srv in HOT_SERVERS:
        assert "port" in srv
        assert "roles" in srv


def test_hot_servers_includes_frontdoor() -> None:
    from scripts.server.stack_manifest import HOT_SERVERS

    ports = {s["port"] for s in HOT_SERVERS}
    # frontdoor primary on 8070
    assert 8070 in ports


def test_port_map_aliases_match_computed_launch_servers() -> None:
    from scripts.server.stack_manifest import HOT_SERVERS, PORT_MAP, WARM_SERVERS

    computed_role_ports: dict[str, int] = {}
    for server in HOT_SERVERS + WARM_SERVERS:
        for role in server.get("roles", []):
            computed_role_ports.setdefault(role, server["port"])

    for role in ("coder_escalation", "worker_summarize", "worker_math", "toolrunner"):
        assert PORT_MAP[role] == computed_role_ports[role]


def test_dense_retriever_candidates_are_warm_embedding_roles() -> None:
    from scripts.server.stack_manifest import (
        EMBEDDING_SERVER_RECIPES,
        HOT_SERVERS,
        ROLE_LAUNCH_META,
        WARM_SERVERS,
    )

    candidate_ports = {
        "embedder_granite_97m_r2": 8096,
        "embedder_multilingual_e5_base": 8097,
        "embedder_bge_m3": 8098,
    }
    hot_roles = {role for server in HOT_SERVERS for role in server.get("roles", [])}
    warm_role_ports = {
        role: server["port"] for server in WARM_SERVERS for role in server.get("roles", [])
    }

    for role, port in candidate_ports.items():
        assert role not in hot_roles
        assert warm_role_ports[role] == port
        assert ROLE_LAUNCH_META[role]["mode"] == "embedding"
        assert EMBEDDING_SERVER_RECIPES[port]["model_path"].endswith(".gguf")


def test_eval_batch_frontdoor_is_warm_launcher_only() -> None:
    from scripts.server.stack_manifest import HOT_SERVERS, PORT_MAP, ROLE_LAUNCH_META, WARM_SERVERS

    hot_roles = {role for server in HOT_SERVERS for role in server.get("roles", [])}
    warm_servers = {
        role: server for server in WARM_SERVERS for role in server.get("roles", [])
    }

    assert PORT_MAP["eval_batch_frontdoor"] == 18070
    assert "eval_batch_frontdoor" not in hot_roles
    assert warm_servers["eval_batch_frontdoor"]["port"] == 18070
    assert warm_servers["eval_batch_frontdoor"]["eval_batch_frontdoor"] is True
    assert ROLE_LAUNCH_META["eval_batch_frontdoor"]["launcher_only"] is True


def test_launch_kv_quant_configs_keep_canonical_worker_roles_only() -> None:
    """2026-08-02: the table is DERIVED from master, so its keys follow master.

    `worker_explore` used to be absent because the hand-written
    `launch_shape.kv_quant_configs` simply did not list it. That table is deleted;
    the entries now come from `server_mode.<role>.serving_shape.kv_quant` resolved
    over every role name the launcher can be asked about, and worker_explore is a
    declared `shared_with` alias of `worker` — it rides that process and therefore
    genuinely has that process's KV types. Its presence is the derivation working,
    not drift. What the test still pins is that no alias contradicts its host.
    """
    from scripts.server.stack_manifest import LAUNCH_KV_QUANT_CONFIGS

    assert LAUNCH_KV_QUANT_CONFIGS["worker_general"] == ("q8_0", "q8_0")
    for alias in ("worker_explore", "worker_math", "toolrunner"):
        assert LAUNCH_KV_QUANT_CONFIGS[alias] == LAUNCH_KV_QUANT_CONFIGS["worker_general"]
    assert LAUNCH_KV_QUANT_CONFIGS["worker_summarize"] == LAUNCH_KV_QUANT_CONFIGS["frontdoor"]
    # worker_vision stays ABSENT: it declares no kv_quant, so no -ctk/-ctv is
    # emitted and llama-server's f16 default stands. A q8_0 VL quality check is
    # running separately; do not pre-empt it by adding a row.
    assert "worker_vision" not in LAUNCH_KV_QUANT_CONFIGS


def test_validate_against_registry_checks_port_map_alias_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.server.stack_manifest as manifest

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "process_layout": {
                    "hot_resident": [
                        "frontdoor",
                        "coder_escalation",
                        "worker_summarize",
                        "worker_general",
                        "worker_math",
                        "toolrunner",
                        "architect_general",
                        "ingest_long_context",
                        "worker_vision",
                        "vision_escalation",
                    ],
                    "warm_mmap": [],
                },
                "server_mode": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    drifted = dict(manifest.PORT_MAP)
    drifted["coder_escalation"] = 8071
    monkeypatch.setattr(manifest, "PORT_MAP", drifted)

    warnings = manifest.validate_against_registry(str(registry))

    assert any(
        "role 'coder_escalation': PORT_MAP says port 8071" in warning
        for warning in warnings
    )


def test_validate_against_registry_checks_server_mode_shared_alias_drift(
    tmp_path: Path,
) -> None:
    from scripts.server.stack_manifest import validate_against_registry

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "process_layout": {
                    "hot_resident": [
                        "frontdoor",
                        "coder_escalation",
                        "worker_summarize",
                        "worker_general",
                        "worker_math",
                        "toolrunner",
                        "architect_general",
                        "ingest_long_context",
                        "worker_vision",
                        "vision_escalation",
                    ],
                    "warm_mmap": [],
                },
                "server_mode": {
                    "worker": {
                        "port": 8082,
                        "model_role": "worker_general",
                        "shared_with": ["worker_math", "toolrunner"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    warnings = validate_against_registry(str(registry))

    assert any(
        "role 'worker': registry server_mode says port 8082 for launch role 'worker_math'"
        in warning
        for warning in warnings
    )


# ----- validate_model_paths is a callable that returns list[str] -----


def test_validate_model_paths_returns_list() -> None:
    from scripts.server.stack_manifest import validate_model_paths

    errors = validate_model_paths()
    assert isinstance(errors, list)
    # all entries should be strings (file paths formatted into error labels)
    for e in errors:
        assert isinstance(e, str)


# ----- numa-mode filter -----


def test_filter_by_numa_mode_both_returns_input_unchanged() -> None:
    from scripts.server.stack_manifest import HOT_SERVERS, _filter_by_numa_mode

    assert _filter_by_numa_mode(HOT_SERVERS, "both") == HOT_SERVERS


def test_filter_by_numa_mode_full_strips_quarters() -> None:
    """For roles with full_instance_idx, mode=full keeps only the full one."""
    from scripts.server.stack_manifest import HOT_SERVERS, _filter_by_numa_mode

    full_only = _filter_by_numa_mode(HOT_SERVERS, "full")
    # frontdoor has 5 instances in NUMA_CONFIG (0=full, 1-4=quarters). With
    # mode=full we keep only the full one — count of frontdoor servers should drop.
    frontdoor_in_full = [s for s in full_only if "frontdoor" in s.get("roles", [])]
    frontdoor_in_all = [s for s in HOT_SERVERS if "frontdoor" in s.get("roles", [])]
    assert len(frontdoor_in_full) < len(frontdoor_in_all)
    assert len(frontdoor_in_full) == 1


# ----- stack_paths is self-contained (no cycle) -----


def test_stack_paths_imports_cleanly() -> None:
    """stack_paths should import without pulling in orchestrator_stack."""
    # Clear cached modules to force a fresh import
    for k in list(sys.modules):
        if k.startswith("scripts.server.stack_paths"):
            del sys.modules[k]
    import scripts.server.stack_paths as sp  # noqa: F401
    # No assertion needed — import success is the test


def test_math_tools_path_supports_subproject_build() -> None:
    from scripts.server.stack_paths import LLAMA_MATH_TOOLS

    assert LLAMA_MATH_TOOLS.name == "llama-math-tools"
    assert "build/bin/llama-math-tools" in str(
        LLAMA_MATH_TOOLS
    ) or "tools/math-tools/build/llama-math-tools" in str(LLAMA_MATH_TOOLS)


# =============================================================================
# Phase 2 — declaration derivation + parity guard
# =============================================================================
# `slots`, `device` and the role alias lists are DERIVED from the master
# registry instead of restated in launch_manifest.yaml. These tests pin both
# halves of that: the derivation produces the declared value, and the guard
# that keeps a second copy from reappearing actually raises.


def test_resolve_slots_returns_the_master_declaration_not_the_serial_policy() -> None:
    """`-np` must come from server_mode.<role>.slots, never from SERIAL_ROLES.

    frontdoor and architect_critic are IN SERIAL_ROLES and DECLARE their slots;
    the old launcher formula (`1 if role in SERIAL_ROLES else 2`) answered the
    admission question with a serving number and returned 1 for both.

    The declared values moved on 2026-08-02 (frontdoor 2 -> 16, architect_critic
    2 -> 4, operator-ratified), so this compares against the DECLARATION rather
    than a literal — the property under test is "resolve_slots returns what master
    says", which must survive master saying something different.
    """
    from scripts.server.stack_manifest import DECLARED_SLOTS, SERIAL_ROLES, resolve_slots

    for role in ("frontdoor", "architect_critic"):
        assert role in SERIAL_ROLES
        decision = resolve_slots(role, "default")
        assert decision.slots == DECLARED_SLOTS[role]
        assert decision.slots > 1, "a serial role must still not be clamped to 1 slot"
        assert decision.declared
        assert decision.source.startswith("master:")


def test_resolve_slots_is_per_instance_for_a_split_role() -> None:
    """One role, two shapes, two answers — what a per-role `slots` cannot express."""
    from scripts.server.stack_manifest import declared_slots_by_port, resolve_slots

    for role, full_port, half_ports in (
        ("frontdoor", 8070, (8080, 8180)),
        ("worker_general", 8072, (8082, 8182)),
    ):
        full = resolve_slots(role, numa_instance=0)
        assert full.slots == 16
        assert full.source.endswith(".full")
        for idx in (1, 2):
            half = resolve_slots(role, numa_instance=idx)
            assert half.slots == 4
            assert half.source.endswith(".half")
        assert declared_slots_by_port(role) == {full_port: 16, half_ports[0]: 4, half_ports[1]: 4}

    # A single-shape role has no split and resolves through the flat compat scalar.
    gpu = resolve_slots("architect_general", numa_instance=0)
    assert (gpu.slots, gpu.source) == (8, "master:architect_general/direct")


def test_resolve_slots_matches_master_for_every_declared_role() -> None:
    from scripts.server.stack_manifest import DECLARED_SLOTS, resolve_slots

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]

    assert DECLARED_SLOTS, "master declares slots for at least the serving roles"
    for role, slots in DECLARED_SLOTS.items():
        decision = resolve_slots(role)
        assert decision.slots == slots
        master_role = decision.source.split(":", 1)[1].split("/", 1)[0]
        assert registry[master_role]["slots"] == slots


def test_resolve_slots_labels_its_fallback_source() -> None:
    """An undeclared role gets the manifest fallback AND says so.

    A fallback that is indistinguishable from a declaration is how "master says
    1 and we launched 2" stayed invisible.
    """
    from scripts.server.stack_manifest import FALLBACK_SLOTS, resolve_slots

    decision = resolve_slots("embedder", "embedding")
    assert not decision.declared
    assert decision.source == "manifest:fallback_slots.embedding"
    assert decision.slots == FALLBACK_SLOTS["embedding"]

    fast = resolve_slots("worker_fast", "worker_pool", worker_type="fast")
    assert fast.source == "manifest:fallback_slots.worker_pool_fast"
    assert fast.slots == 4


def test_vision_device_is_derived_from_master() -> None:
    from scripts.server.stack_manifest import VISION_WORKER_DEVICE

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]
    assert VISION_WORKER_DEVICE == registry["worker_vision"]["device"]

    manifest = yaml.safe_load(
        (ROOT / "orchestration" / "launch_manifest.yaml").read_text()
    )
    assert "device" not in manifest["vision"]["worker"], (
        "vision.worker must not re-declare a device — it is derived from master"
    )


def test_role_aliases_are_derived_from_master_plus_declared_extras() -> None:
    from scripts.server.stack_manifest import ROLE_LAUNCH_META

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]
    manifest = yaml.safe_load(
        (ROOT / "orchestration" / "launch_manifest.yaml").read_text()
    )["role_launch_meta"]

    def master_shared(launcher_role: str) -> list[str]:
        row = registry.get(launcher_role)
        if not isinstance(row, dict):
            row = next(
                (cfg for cfg in registry.values() if cfg.get("model_role") == launcher_role),
                None,
            )
        return list((row or {}).get("shared_with") or [])

    for role, meta in ROLE_LAUNCH_META.items():
        extras = list(manifest[role].get("launcher_only_aliases") or [])
        expected = extras + [a for a in master_shared(role) if a not in extras]
        assert meta.get("shared_with_first_n", []) == expected, role
        assert "shared_with_first_n" not in manifest[role], (
            f"{role} re-declares shared_with_first_n; it is derived from master"
        )


def test_parity_guard_passes_on_the_shipped_configuration() -> None:
    from scripts.server.stack_manifest import validate_declaration_parity

    validate_declaration_parity()


@pytest.mark.parametrize(
    "label,target,key,value",
    [
        ("launcher port drifts", "PORT_MAP", "frontdoor", 8071),
        ("launcher tier drifts", "ROLE_LAUNCH_META.frontdoor", "tier", "warm"),
    ],
)
def test_parity_guard_fails_when_a_launcher_row_drifts(
    label: str, target: str, key: str, value: object, monkeypatch
) -> None:
    from scripts.server import stack_manifest as sm

    if "." in target:
        attr, sub = target.split(".", 1)
        container = dict(getattr(sm, attr))
        container[sub] = {**container[sub], key: value}
        monkeypatch.setattr(sm, attr, container)
    else:
        container = dict(getattr(sm, target))
        container[key] = value
        monkeypatch.setattr(sm, target, container)

    with pytest.raises(ValueError, match="parity violated"):
        sm.validate_declaration_parity()


def test_parity_guard_fails_when_master_drifts(monkeypatch) -> None:
    from scripts.server import stack_manifest as sm

    mutated = {
        name: ({**cfg, "port": 9999} if name == "frontdoor" else cfg)
        for name, cfg in sm.MASTER_SERVER_MODE.items()
    }
    monkeypatch.setattr(sm, "MASTER_SERVER_MODE", mutated)

    with pytest.raises(ValueError, match="parity violated"):
        sm.validate_declaration_parity()


def test_parity_guard_fails_when_a_derived_field_is_redeclared(monkeypatch) -> None:
    from scripts.server import stack_manifest as sm

    manifest = {**sm._MANIFEST}
    manifest["vision"] = {
        **manifest["vision"],
        "worker": {**manifest["vision"]["worker"], "device": "ROCm0"},
    }
    monkeypatch.setattr(sm, "_MANIFEST", manifest)

    with pytest.raises(ValueError, match="DERIVED from master"):
        sm.validate_declaration_parity()


def test_parity_guard_fails_on_an_unargued_launcher_only_alias(monkeypatch) -> None:
    from scripts.server import stack_manifest as sm

    monkeypatch.setattr(sm, "_PARITY", {**sm._PARITY, "exceptions": []})

    with pytest.raises(ValueError, match="written reason"):
        sm.validate_declaration_parity()


def test_parity_guard_fails_on_a_stale_exception(monkeypatch) -> None:
    """When master starts declaring the alias, the exception must be deleted."""
    from scripts.server import stack_manifest as sm

    mutated = {
        name: (
            {**cfg, "shared_with": ["worker_explore", *cfg.get("shared_with", [])]}
            if name == "worker"
            else cfg
        )
        for name, cfg in sm.MASTER_SERVER_MODE.items()
    }
    monkeypatch.setattr(sm, "MASTER_SERVER_MODE", mutated)

    with pytest.raises(ValueError, match="no longer launcher-only"):
        sm.validate_declaration_parity()
