"""Tests for runtime-facts manifest generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.server.fleet_markers import (
    LAUNCH_SOURCE_STACK_COMMANDS,
    write_llama_marker,
    write_orchestrator_marker,
)
from scripts.server.runtime_facts_manifest import (
    RUNTIME_FACTS_MANIFEST_NAME,
    build_runtime_facts_manifest,
    read_runtime_stack_selected_servers,
    runtime_facts_manifest_path,
    write_runtime_facts_manifest,
)
from scripts.server.stack_state import ProcessInfo


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_build_runtime_facts_manifest_serializes_process_info_and_markers(
    tmp_path: Path,
) -> None:
    stack_priors_path = _write_yaml(
        tmp_path / "stack_priors.yaml",
        """
stack_priors_version: 4
compiled_at: "2026-07-11T00:00:00Z"
status: live
source_artifacts: {}
roles:
  frontdoor:
    deployment_status: live_stack
    serving: {}
""".strip()
        + "\n",
    )
    orchestrator_marker = write_orchestrator_marker(tmp_dir=tmp_path, git_sha="cafebabe")
    llama_marker = write_llama_marker(
        port=8070,
        roles=["frontdoor", "coder_escalation"],
        source=LAUNCH_SOURCE_STACK_COMMANDS,
        tmp_dir=tmp_path,
    )

    process = ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="2026-07-11T00:00:00Z",
        model_path="/models/frontdoor.gguf",
        log_file="/logs/frontdoor.log",
    )
    manifest = build_runtime_facts_manifest(
        state={"frontdoor": process},
        launch_contracts={
            "frontdoor": {
                "ports": [8070],
                "binary": "llama.cpp",
                "launch": {"entries": [{"port": 8070, "alias": False}]},
            }
        },
        stack_priors_path=stack_priors_path,
        stack_numa_mode="quarter",
        tmp_dir=tmp_path,
        repo_short_sha="abc1234",
        source="unit-test",
    )

    assert manifest["schema"] == "epyc.orchestrator.runtime_facts"
    assert manifest["schema_version"] == 1
    assert manifest["source"] == "unit-test"
    assert manifest["repo"] == {"short_sha": "abc1234"}
    assert manifest["runtime_stack"]["stack_numa_mode"] == "quarter"
    assert 8080 in manifest["runtime_stack"]["selected_ports"]
    assert 8070 not in manifest["runtime_stack"]["selected_ports"]
    assert manifest["runtime_stack"]["paths"] == {
        "tmp_dir": str(tmp_path),
        "state_file": str(manifest["runtime_stack"]["paths"]["state_file"]),
        "stack_priors_path": str(stack_priors_path),
        "log_dir": str(manifest["runtime_stack"]["paths"]["log_dir"]),
        "llama_server": str(manifest["runtime_stack"]["paths"]["llama_server"]),
    }
    assert manifest["launch_contracts"]["frontdoor"]["ports"] == [8070]
    assert manifest["state"]["frontdoor"] == asdict(process)
    assert manifest["stack_priors"] == {
        "path": str(stack_priors_path),
        "available": True,
        "stack_priors_version": 4,
        "compiled_at": "2026-07-11T00:00:00Z",
        "status": "live",
        "source_artifacts": {},
        "live_role_count": 1,
        "live_roles": ["frontdoor"],
    }

    orchestrator = manifest["fleet_markers"]["orchestrator"]
    assert orchestrator is not None
    assert orchestrator["git_sha"] == "cafebabe"
    assert isinstance(orchestrator["started_at"], float)
    assert manifest["fleet_markers"]["llama"] == {
        "8070": {
            "started_at": manifest["fleet_markers"]["llama"]["8070"]["started_at"],
            "source": LAUNCH_SOURCE_STACK_COMMANDS,
            "roles": ["frontdoor", "coder_escalation"],
        }
    }

    assert orchestrator_marker.exists()
    assert llama_marker.exists()


def test_write_runtime_facts_manifest_writes_valid_json_and_handles_missing_inputs(
    tmp_path: Path,
) -> None:
    malformed_stack_priors_path = _write_yaml(
        tmp_path / "stack_priors.yaml",
        "{not: valid: yaml",
    )

    manifest_path = write_runtime_facts_manifest(
        state={
            "frontdoor": ProcessInfo(
                role="frontdoor",
                pid=123,
                port=8070,
                started_at="2026-07-11T00:00:00Z",
                model_path="/models/frontdoor.gguf",
                log_file="/logs/frontdoor.log",
            )
        },
        launch_contracts={
            "frontdoor": {"ports": [8070], "launch": {"entries": []}}
        },
        stack_priors_path=malformed_stack_priors_path,
        stack_numa_mode="full",
        tmp_dir=tmp_path,
        repo_short_sha=None,
        source="unit-test",
    )

    assert manifest_path == runtime_facts_manifest_path(tmp_path)
    assert manifest_path.name == RUNTIME_FACTS_MANIFEST_NAME
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "epyc.orchestrator.runtime_facts"
    assert manifest["schema_version"] == 1
    assert manifest["source"] == "unit-test"
    assert manifest["repo"] == {"short_sha": None}
    assert manifest["runtime_stack"]["stack_numa_mode"] == "full"
    assert manifest["launch_contracts"]["frontdoor"]["ports"] == [8070]
    assert manifest["state"]["frontdoor"]["pid"] == 123
    assert manifest["stack_priors"] == {
        "path": str(malformed_stack_priors_path),
        "available": False,
        "live_role_count": 0,
        "live_roles": [],
    }
    assert manifest["fleet_markers"]["orchestrator"] is None
    assert manifest["fleet_markers"]["llama"] == {}


def test_write_runtime_facts_manifest_uses_atomic_replace_no_tmp_left_behind(
    tmp_path: Path,
) -> None:
    stack_priors_path = _write_yaml(
        tmp_path / "stack_priors.yaml",
        """
stack_priors_version: 4
compiled_at: "2026-07-11T00:00:00Z"
status: live
source_artifacts: {}
roles: {}
""".strip()
        + "\n",
    )

    manifest_path = write_runtime_facts_manifest(
        state={},
        launch_contracts={},
        stack_priors_path=stack_priors_path,
        stack_numa_mode="both",
        tmp_dir=tmp_path,
        repo_short_sha="abc1234",
        source="unit-test",
    )

    assert manifest_path.exists()
    leftover = [entry for entry in tmp_path.iterdir() if ".tmp." in entry.name]
    assert leftover == []


def test_read_runtime_stack_selected_servers_rejects_malformed_or_stale_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "facts.json"
    state_file = tmp_path / "orchestrator_state.json"

    manifest_path.write_text("{}", encoding="utf-8")
    assert read_runtime_stack_selected_servers(
        manifest_path=manifest_path,
        state_file=state_file,
    ) is None

    manifest_path.write_text(
        json.dumps({
            "schema": "epyc.orchestrator.runtime_facts",
            "schema_version": 1,
            "runtime_stack": {
                "stack_numa_mode": "full",
                "selected_ports": [8070],
                "selected_servers": [
                    {"port": 8070, "roles": ["frontdoor"]},
                    {"port": 8070, "roles": ["duplicate"]},
                ],
            },
        }),
        encoding="utf-8",
    )
    assert read_runtime_stack_selected_servers(
        manifest_path=manifest_path,
        state_file=state_file,
    ) is None

    manifest_path.write_text(
        json.dumps({
            "schema": "epyc.orchestrator.runtime_facts",
            "schema_version": 1,
            "runtime_stack": {
                "stack_numa_mode": "full",
                "selected_ports": [8070],
                "selected_servers": [{"port": 8070, "roles": ["frontdoor"]}],
            },
        }),
        encoding="utf-8",
    )
    state_file.write_text("{}", encoding="utf-8")
    assert read_runtime_stack_selected_servers(
        manifest_path=manifest_path,
        state_file=state_file,
    ) is None


def test_read_runtime_stack_selected_servers_accepts_valid_manifest(tmp_path: Path) -> None:
    stack_priors_path = _write_yaml(
        tmp_path / "stack_priors.yaml",
        """
stack_priors_version: 4
compiled_at: "2026-07-11T00:00:00Z"
status: live
source_artifacts: {}
roles: {}
""".strip()
        + "\n",
    )
    manifest_path = write_runtime_facts_manifest(
        state={},
        launch_contracts={},
        stack_priors_path=stack_priors_path,
        stack_numa_mode="full",
        tmp_dir=tmp_path,
        repo_short_sha="abc1234",
        source="unit-test",
    )

    servers = read_runtime_stack_selected_servers(
        manifest_path=manifest_path,
        state_file=tmp_path / "missing-state.json",
    )

    assert servers is not None
    ports = {server["port"] for server in servers}
    assert 8070 in ports
    assert 8080 not in ports
