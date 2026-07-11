"""Runtime-only stack facts manifest.

This module writes a derived JSON cache under the runtime tmp directory. It is
not a source of truth: launch/state facts still come from stack priors, the
launcher state file, and fleet marker files.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.server.fleet_markers import (
    discover_llama_markers,
    read_orchestrator_marker_metadata,
)
from scripts.server.stack_paths import _PATHS
from src.registry.stack_priors import (
    load_stack_priors_artifact,
    live_stack_role_records,
)


RUNTIME_FACTS_MANIFEST_NAME = "orchestrator_runtime_facts.json"
RUNTIME_FACTS_SCHEMA = "epyc.orchestrator.runtime_facts"
RUNTIME_FACTS_MANIFEST_VERSION = 1


def runtime_facts_manifest_path(tmp_dir: Path | None = None) -> Path:
    """Return the runtime facts manifest path."""
    return (tmp_dir or _PATHS["tmp_dir"]) / RUNTIME_FACTS_MANIFEST_NAME


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _stack_priors_summary(stack_priors_path: Path) -> dict[str, Any]:
    artifact = load_stack_priors_artifact(stack_priors_path)
    if artifact is None:
        return {
            "path": str(stack_priors_path),
            "available": False,
            "live_role_count": 0,
            "live_roles": [],
        }

    live_roles = sorted(live_stack_role_records(stack_priors_path))
    source_artifacts = artifact.get("source_artifacts")
    return {
        "path": str(stack_priors_path),
        "available": True,
        "stack_priors_version": artifact.get("stack_priors_version"),
        "compiled_at": artifact.get("compiled_at"),
        "status": artifact.get("status"),
        "source_artifacts": source_artifacts if isinstance(source_artifacts, dict) else {},
        "live_role_count": len(live_roles),
        "live_roles": live_roles,
    }


def build_runtime_facts_manifest(
    *,
    state: Mapping[str, Any],
    launch_contracts: Mapping[str, Any],
    stack_priors_path: Path,
    tmp_dir: Path | None = None,
    repo_short_sha: str | None = None,
    source: str,
) -> dict[str, Any]:
    """Build a serializable runtime-facts payload from existing sources."""
    marker_dir = tmp_dir or _PATHS["tmp_dir"]
    llama_markers = {
        str(port): metadata
        for port, metadata in sorted(discover_llama_markers(tmp_dir=marker_dir).items())
    }
    return {
        "schema": RUNTIME_FACTS_SCHEMA,
        "schema_version": RUNTIME_FACTS_MANIFEST_VERSION,
        "generated_at": _timestamp(),
        "source": str(source),
        "repo": {
            "short_sha": repo_short_sha,
        },
        "stack_priors": _stack_priors_summary(stack_priors_path),
        "launch_contracts": _jsonable(launch_contracts),
        "state": _jsonable(state),
        "fleet_markers": {
            "orchestrator": read_orchestrator_marker_metadata(tmp_dir=marker_dir),
            "llama": llama_markers,
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_runtime_facts_manifest(
    *,
    state: Mapping[str, Any],
    launch_contracts: Mapping[str, Any],
    stack_priors_path: Path,
    tmp_dir: Path | None = None,
    repo_short_sha: str | None = None,
    source: str,
) -> Path:
    """Write the runtime facts manifest atomically and return its path."""
    path = runtime_facts_manifest_path(tmp_dir)
    payload = build_runtime_facts_manifest(
        state=state,
        launch_contracts=launch_contracts,
        stack_priors_path=stack_priors_path,
        tmp_dir=tmp_dir,
        repo_short_sha=repo_short_sha,
        source=source,
    )
    _atomic_write_json(path, payload)
    return path
