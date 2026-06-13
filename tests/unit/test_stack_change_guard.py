"""Tests for stack-prior drift validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from scripts.validate.stack_change_guard import validate_stack_priors


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _priors(registry: Path, descriptors: Path, *, memory_cost: float = 1.0) -> dict:
    return {
        "source_artifacts": {
            "registry": {"path": str(registry), "sha256": _sha(registry)},
            "descriptors": {"path": str(descriptors), "sha256": _sha(descriptors)},
        },
        "roles": {
            "frontdoor": {
                "deployment_status": "live_stack",
                "model_id": "qwen",
                "serving": {"endpoint": "http://localhost:8070", "tier": "hot"},
                "priors": {"memory_cost": memory_cost},
                "known_gaps": [],
            }
        },
        "known_global_gaps": {},
    }


def test_validate_stack_priors_accepts_fresh_live_artifact(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    result = validate_stack_priors(priors)

    assert result.ok
    assert result.warnings == []


def test_validate_stack_priors_rejects_stale_source_hash(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    registry.write_text("roles:\n  changed: true\n", encoding="utf-8")

    result = validate_stack_priors(priors)

    assert not result.ok
    assert any("hash mismatch" in error for error in result.errors)


def test_validate_stack_priors_rejects_retired_live_role(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["architect_coding"] = {
        "deployment_status": "live_stack",
        "model_id": "retired",
        "serving": {"endpoint": "http://localhost:8084", "tier": "hot"},
        "priors": {"memory_cost": 1.0},
        "known_gaps": [],
    }
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(priors)

    assert not result.ok
    assert "retired role 'architect_coding' appears as live_stack" in result.errors


def test_validate_stack_priors_rejects_hot_memory_penalty(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(
        tmp_path / "stack_priors.yaml",
        _priors(registry, descriptors, memory_cost=2.0),
    )

    result = validate_stack_priors(priors)

    assert not result.ok
    assert any("live HOT role 'frontdoor'" in error for error in result.errors)


def test_validate_stack_priors_strict_fails_on_known_gaps(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["known_gaps"] = ["missing ctx"]
    payload["known_global_gaps"] = {"frontdoor": ["missing ctx"]}
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    loose = validate_stack_priors(priors)
    strict = validate_stack_priors(priors, strict=True)

    assert loose.ok
    assert loose.warnings
    assert not strict.ok
    assert any(error.startswith("strict:") for error in strict.errors)
