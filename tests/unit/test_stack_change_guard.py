"""Tests for stack-prior drift validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.validate.stack_change_guard import (
    scan_hardcoded_surfaces,
    validate_stack_priors,
)
from src.registry.stack_priors import STACK_PRIORS_VERSION, stack_priors_contract


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _priors(registry: Path, descriptors: Path, *, memory_cost: float = 1.0) -> dict:
    return {
        "stack_priors_version": STACK_PRIORS_VERSION,
        "contract": stack_priors_contract(),
        "compiled_at": "2026-06-13T00:00:00Z",
        "status": "compiled",
        "coverage_scope": "test",
        "precedence_spec": "docs/reference/stack-truth-precedence.md",
        "source_artifacts": {
            "registry": {"path": str(registry), "sha256": _sha(registry)},
            "descriptors": {"path": str(descriptors), "sha256": _sha(descriptors)},
        },
        "roles": {
            "frontdoor": {
                "role": "frontdoor",
                "deployment_status": "live_stack",
                "status": "compiled",
                "model_id": "qwen",
                "display_name": "Qwen",
                "serving": {
                    "endpoint": "http://localhost:8070",
                    "server_role": "frontdoor",
                    "binding": "server_mode.direct",
                    "ports": [8070],
                    "slots": 1,
                    "tier": "hot",
                    "binary": None,
                    "binary_dir": None,
                    "numa_policy": None,
                    "shared_mmap": False,
                },
                "priors": {
                    "throughput_tps": 24.3,
                    "quality_overall": 0.9,
                    "memory_cost": memory_cost,
                },
                "acceleration": {},
                "model": {},
                "evidence": {},
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


def test_validate_stack_priors_rejects_missing_contract(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload.pop("contract")
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(priors)

    assert not result.ok
    assert "missing top-level stack-prior field: contract" in result.errors


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


def test_scan_hardcoded_surfaces_classifies_curated_surfaces(tmp_path: Path) -> None:
    active = tmp_path / "scripts" / "benchmark"
    tests = tmp_path / "tests" / "unit"
    docs = tmp_path / "docs"
    active.mkdir(parents=True)
    tests.mkdir(parents=True)
    docs.mkdir(parents=True)
    (active / "seeding_types.py").write_text(
        'DEFAULT_ROLES = ["architect_coding"]\n',
        encoding="utf-8",
    )
    (tests / "test_legacy.py").write_text(
        'def test_legacy():\n    assert "architect_coding"\n',
        encoding="utf-8",
    )
    (docs / "ARCHITECTURE.md").write_text(
        "Historical chain: architect_coding\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    categories = {finding.category for finding in findings}
    assert "production_blocker" in categories
    assert "legacy_test" in categories
    assert "historical_doc" in categories
    assert any(finding.path.as_posix() == "scripts/benchmark/seeding_types.py" for finding in findings)


def test_validate_stack_priors_can_include_surface_warnings(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    active = tmp_path / "scripts" / "benchmark"
    active.mkdir(parents=True)
    (active / "seeding_rewards.py").write_text(
        'DEFAULT_BASELINE_TPS = {"frontdoor": 10.3}\n',
        encoding="utf-8",
    )

    loose = validate_stack_priors(priors, scan_surfaces=True, repo_root=tmp_path)
    strict = validate_stack_priors(
        priors,
        strict=True,
        scan_surfaces=True,
        repo_root=tmp_path,
    )

    assert loose.ok
    assert any("hardcoded_surface.production_blocker" in warning for warning in loose.warnings)
    assert not strict.ok
    assert any("hardcoded_surface.production_blocker" in error for error in strict.errors)


def test_validate_stack_priors_keeps_waived_surface_visible_in_strict_mode(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    active = tmp_path / "scripts" / "benchmark"
    active.mkdir(parents=True)
    (active / "seeding_rewards.py").write_text(
        'DEFAULT_BASELINE_TPS = {"frontdoor": 10.3}\n',
        encoding="utf-8",
    )
    exceptions = _write_yaml(
        tmp_path / "exceptions.yaml",
        {
            "exceptions": [
                {
                    "rule_id": "seeding_baseline_tps_table",
                    "category": "production_blocker",
                    "path_glob": "scripts/benchmark/seeding_rewards.py",
                    "classification": "degraded_fallback",
                    "owner": "stack-change-governance",
                    "rationale": "temporary fixture for strict-mode waiver behavior",
                    "expires": "2099-01-01",
                }
            ]
        },
    )

    result = validate_stack_priors(
        priors,
        strict=True,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_exceptions_path=exceptions,
    )

    assert result.ok
    assert any("hardcoded_surface.waived.production_blocker" in warning for warning in result.warnings)


def test_validate_stack_priors_rejects_invalid_surface_exception(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    exceptions = _write_yaml(
        tmp_path / "exceptions.yaml",
        {
            "exceptions": [
                {
                    "rule_id": "seeding_baseline_tps_table",
                    "category": "production_blocker",
                    "path_glob": "scripts/benchmark/seeding_rewards.py",
                    "classification": "degraded_fallback",
                    "rationale": "missing owner should fail",
                    "expires": "2099-01-01",
                }
            ]
        },
    )

    result = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_exceptions_path=exceptions,
    )

    assert not result.ok
    assert any("missing non-empty 'owner'" in error for error in result.errors)


def test_validate_stack_priors_rejects_stale_procedure_role_enum(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    procedure_dir = tmp_path / "orchestration" / "procedures"
    procedure_dir.mkdir(parents=True)
    _write_yaml(
        procedure_dir / "add_model_to_registry.yaml",
        {
            "inputs": [
                {
                    "name": "role",
                    "type": "string",
                    "description": "Role assignment",
                    "validation": {"enum": ["frontdoor", "architect_coding"]},
                }
            ]
        },
    )

    result = validate_stack_priors(priors, scan_surfaces=True, repo_root=tmp_path)

    assert not result.ok
    assert any("procedure role enum drift" in error for error in result.errors)


def test_validate_stack_priors_rejects_stale_procedure_schema_role_enum(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    orchestration_dir = tmp_path / "orchestration"
    orchestration_dir.mkdir()
    schema = {
        "properties": {
            "permissions": {
                "properties": {
                    "roles": {
                        "items": {
                            "enum": ["frontdoor", "architect_coding", "admin"],
                        }
                    }
                }
            }
        }
    }
    (orchestration_dir / "procedure.schema.json").write_text(
        json.dumps(schema),
        encoding="utf-8",
    )

    result = validate_stack_priors(priors, scan_surfaces=True, repo_root=tmp_path)

    assert not result.ok
    assert any("procedure schema permission enum drift" in error for error in result.errors)
