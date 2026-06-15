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
