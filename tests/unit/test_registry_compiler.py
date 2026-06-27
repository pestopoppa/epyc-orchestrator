from __future__ import annotations

from pathlib import Path

import yaml

from src.registry.registry_compiler import (
    active_roles_from_launch_meta,
    compile_lean,
)


def test_active_roles_from_launch_meta_includes_shared_aliases() -> None:
    launch_meta = {
        "frontdoor": {
            "tier": "hot",
            "shared_with_first_n": ["coder_escalation", "worker_summarize"],
        },
        "worker_general": {
            "tier": "hot",
            "shared_with_first_n": ["worker_math", "toolrunner"],
        },
        "embedder": {"tier": "hot", "mode": "embedding"},
    }

    assert active_roles_from_launch_meta(launch_meta) == {
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
        "worker_general",
        "worker_math",
        "toolrunner",
        "embedder",
    }


def test_compile_lean_keeps_alias_records_when_active_roles_include_aliases(
    tmp_path: Path,
) -> None:
    master = {
        "runtime_defaults": {"temperature": 0.2},
        "server_mode": {
            "frontdoor": {
                "port": 8070,
                "shared_with": ["coder_escalation", "worker_summarize"],
            },
            "coder_escalation": {"port": 8070},
            "worker_summarize": {"port": 8070},
            "cold_candidate": {"port": 9000},
        },
        "roles": {
            "frontdoor": {"model": {"path": "frontdoor.gguf"}},
            "coder_escalation": {"model": {"path": "frontdoor.gguf"}},
            "worker_summarize": {"model": {"path": "frontdoor.gguf"}},
            "cold_candidate": {"model": {"path": "cold.gguf"}},
        },
    }
    master_path = tmp_path / "model_registry.yaml"
    master_path.write_text(yaml.safe_dump(master), encoding="utf-8")

    active = active_roles_from_launch_meta(
        {
            "frontdoor": {
                "shared_with_first_n": ["coder_escalation", "worker_summarize"]
            }
        }
    )
    lean = compile_lean(master_path, active)

    assert set(lean["server_mode"]) == {
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
    }
    assert set(lean["roles"]) == {
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
    }


def test_compile_lean_keeps_server_mode_backing_process_for_model_role(
    tmp_path: Path,
) -> None:
    master = {
        "server_mode": {
            "worker": {
                "port": 8082,
                "model_role": "worker_general",
                "shared_with": ["worker_math", "toolrunner"],
            },
            "cold_candidate": {"port": 9000, "model_role": "cold_candidate"},
        },
        "roles": {
            "worker_general": {"model": {"path": "worker.gguf"}},
            "worker_math": {"model": {"path": "math.gguf"}},
            "toolrunner": {"model": {"path": "toolrunner.gguf"}},
            "cold_candidate": {"model": {"path": "cold.gguf"}},
        },
    }
    master_path = tmp_path / "model_registry.yaml"
    master_path.write_text(yaml.safe_dump(master), encoding="utf-8")

    active = active_roles_from_launch_meta(
        {
            "worker_general": {
                "shared_with_first_n": ["worker_math", "toolrunner"]
            }
        }
    )
    lean = compile_lean(master_path, active)

    assert set(lean["server_mode"]) == {"worker"}
    assert set(lean["roles"]) == {
        "worker_general",
        "worker_math",
        "toolrunner",
    }


def test_compile_lean_drops_retired_runtime_timeout_aliases(tmp_path: Path) -> None:
    master = {
        "runtime_defaults": {
            "timeouts": {
                "default": 600,
                "roles": {
                    "worker_coder": 30,
                    "worker_code": 60,
                    "worker_general": 60,
                },
            }
        },
        "roles": {
            "worker_general": {"model": {"path": "worker.gguf"}},
        },
    }
    master_path = tmp_path / "model_registry.yaml"
    master_path.write_text(yaml.safe_dump(master), encoding="utf-8")

    lean = compile_lean(master_path, {"worker_general"})

    role_timeouts = lean["runtime_defaults"]["timeouts"]["roles"]
    assert role_timeouts == {
        "worker_coder": 30,
        "worker_general": 60,
    }
