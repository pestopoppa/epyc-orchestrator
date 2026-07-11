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
        tmp_dir=tmp_path,
        repo_short_sha="abc1234",
        source="unit-test",
    )

    assert manifest["schema"] == "epyc.orchestrator.runtime_facts"
    assert manifest["schema_version"] == 1
    assert manifest["source"] == "unit-test"
    assert manifest["repo"] == {"short_sha": "abc1234"}
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
        tmp_dir=tmp_path,
        repo_short_sha="abc1234",
        source="unit-test",
    )

    assert manifest_path.exists()
    leftover = [entry for entry in tmp_path.iterdir() if ".tmp." in entry.name]
    assert leftover == []

