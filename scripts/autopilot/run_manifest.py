"""Deterministic, source-bound provenance for an in-flight AutoPilot trial.

The manifest is intentionally a control-plane receipt, not an instrument-era
record.  It binds the exact controller/evaluator sources and selected task
before dispatch, then rejects crash recovery if either source or evaluator
drifted.  Historical journal rows remain append-only and are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


RUN_MANIFEST_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_descriptor(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256_bytes(resolved.read_bytes())}


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def build_run_manifest(
    *,
    source_paths: Mapping[str, Path],
    task: Mapping[str, Any],
    evaluator: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic manifest over sources, task, and evaluator."""
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "sources": {
            name: _file_descriptor(path)
            for name, path in sorted(source_paths.items())
        },
        "task": dict(task),
        "evaluator": dict(evaluator),
    }
    return {**payload, "manifest_sha256": _manifest_digest(payload)}


def manifest_drift_reasons(
    manifest: Mapping[str, Any],
    *,
    source_paths: Mapping[str, Path],
    evaluator: Mapping[str, Any],
) -> list[str]:
    """Return deterministic incompatibilities for an in-flight manifest.

    The task is deliberately not recomputed at resume: the persisted in-flight
    action is the task being recovered.  Code and evaluator identity must still
    agree before recovery can classify or terminalize it.
    """
    reasons: list[str] = []
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        reasons.append("unsupported-schema")
    if manifest.get("manifest_sha256") != _manifest_digest(payload):
        reasons.append("manifest-digest-mismatch")
    expected_sources = {
        name: _file_descriptor(path) for name, path in sorted(source_paths.items())
    }
    if manifest.get("sources") != expected_sources:
        reasons.append("source-drift")
    if manifest.get("evaluator") != dict(evaluator):
        reasons.append("evaluator-drift")
    if not isinstance(manifest.get("task"), Mapping):
        reasons.append("missing-task")
    return reasons
