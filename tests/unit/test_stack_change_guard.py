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
    stack_manifest = registry.parent / "stack_manifest.py"
    stack_manifest.write_text("ROLE_LAUNCH_META = {}\n", encoding="utf-8")
    stack_numa = registry.parent / "stack_numa.py"
    stack_numa.write_text("NUMA_CONFIG = {}\n", encoding="utf-8")
    orchestrator_stack = registry.parent / "orchestrator_stack.py"
    orchestrator_stack.write_text("# launcher\n", encoding="utf-8")
    stack_paths = registry.parent / "stack_paths.py"
    stack_paths.write_text("# paths\n", encoding="utf-8")
    stack_runtime = registry.parent / "stack_runtime.py"
    stack_runtime.write_text("# runtime\n", encoding="utf-8")
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
            "stack_manifest": {"path": str(stack_manifest), "sha256": _sha(stack_manifest)},
            "stack_numa": {"path": str(stack_numa), "sha256": _sha(stack_numa)},
            "orchestrator_stack": {
                "path": str(orchestrator_stack),
                "sha256": _sha(orchestrator_stack),
            },
            "stack_paths": {"path": str(stack_paths), "sha256": _sha(stack_paths)},
            "stack_runtime": {"path": str(stack_runtime), "sha256": _sha(stack_runtime)},
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
                    "ports": [8070, 8080, 8180, 8280, 8380],
                    "slots": 1,
                    "tier": "hot",
                    "effective_context_tokens": 32768,
                    "binary": None,
                    "binary_dir": None,
                    "numa_policy": None,
                    "shared_mmap": False,
                    "launch": {
                        "entries": [
                            {
                                "port": 8070,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 0,
                            },
                            {
                                "port": 8080,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 1,
                            },
                            {
                                "port": 8180,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 2,
                            },
                            {
                                "port": 8280,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 3,
                            },
                            {
                                "port": 8380,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 4,
                            },
                        ],
                        "primary_roles": ["frontdoor"],
                        "modes": ["default"],
                        "requirements": {},
                        "runtime": {},
                    },
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

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

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


def test_validate_stack_priors_rejects_stale_stack_manifest_hash(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    (tmp_path / "stack_manifest.py").write_text(
        "ROLE_LAUNCH_META = {'frontdoor': {'tier': 'warm'}}\n",
        encoding="utf-8",
    )

    result = validate_stack_priors(priors)

    assert not result.ok
    assert any("source_artifacts.stack_manifest hash mismatch" in error for error in result.errors)


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


def test_validate_stack_priors_rejects_launch_manifest_endpoint_drift(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["endpoint"] = "http://localhost:9999"
    payload["roles"]["frontdoor"]["serving"]["ports"] = [9999]
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

    assert not result.ok
    assert any("serving.endpoint port 9999" in error for error in result.errors)
    assert any("does not include launch manifest port 8070" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_tier_drift(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "warm"}},
    )

    assert not result.ok
    assert any("serving.tier 'hot'" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_context_drift(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["effective_context_tokens"] = 8192
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {
                "port": 8070,
                "tier": "hot",
                "effective_context_tokens": 32768,
            }
        },
    )

    assert not result.ok
    assert any("serving.effective_context_tokens 8192" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_port_set_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["ports"] = [8070, 9999]
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "ports": [8070, 8080], "tier": "hot"}
        },
    )

    assert not result.ok
    assert any("missing launch manifest port(s) [8080]" in error for error in result.errors)
    assert any("include non-launch port(s) [9999]" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_entry_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["launch"]["entries"] = [
        {"port": 8070, "primary_role": "frontdoor", "mode": "worker_pool", "alias": False}
    ]
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {
                "port": 8070,
                "ports": [8070],
                "tier": "hot",
                "launch_entries": [
                    {
                        "port": 8070,
                        "primary_role": "frontdoor",
                        "mode": "default",
                        "alias": False,
                    }
                ],
            }
        },
    )

    assert not result.ok
    assert any("serving.launch.entries do not match" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_requirement_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["launch"]["requirements"] = {
        "model_path": "/stale/model.gguf"
    }
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {
                "port": 8070,
                "ports": [8070],
                "tier": "hot",
                "launch_requirements": {
                    "model_path": "/current/model.gguf",
                    "mmproj_path": "/current/mmproj.gguf",
                },
            }
        },
    )

    assert not result.ok
    assert any("serving.launch.requirements do not match" in error for error in result.errors)
    assert any("mmproj_path" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_runtime_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["launch"]["runtime"] = {
        "binary_family": "llama.cpp",
        "binary_path": "/stale/llama-server",
        "cache": {"kv_type_k": "q8_0", "kv_type_v": "q8_0", "slots": 1},
        "flags": {"spec": {"enabled": False}},
    }
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {
                "port": 8070,
                "ports": [8070],
                "tier": "hot",
                "launch_runtime": {
                    "binary_family": "llama.cpp",
                    "binary_path": "/current/llama-server",
                    "cache": {"kv_type_k": "q8_0", "kv_type_v": "q8_0", "slots": 1},
                    "flags": {"spec": {"enabled": False}},
                },
            }
        },
    )

    assert not result.ok
    assert any("serving.launch.runtime does not match" in error for error in result.errors)
    assert any("/current/llama-server" in error for error in result.errors)


def test_validate_stack_priors_strict_fails_on_known_gaps(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["known_gaps"] = ["missing ctx"]
    payload["known_global_gaps"] = {"frontdoor": ["missing ctx"]}
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    launch_targets = {"frontdoor": {"port": 8070, "tier": "hot"}}
    loose = validate_stack_priors(priors, launch_manifest_targets=launch_targets)
    strict = validate_stack_priors(priors, strict=True, launch_manifest_targets=launch_targets)

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


def test_scan_hardcoded_surfaces_flags_retired_launch_env_var(tmp_path: Path) -> None:
    server = tmp_path / "scripts" / "server"
    server.mkdir(parents=True)
    (server / "orchestrator_stack.py").write_text(
        'env["ORCHESTRATOR_LANGGRAPH_ARCHITECT_CODING"] = "1"\n',
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "retired_role_env_flag"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/server/orchestrator_stack.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_stale_autopilot_program_guidance(tmp_path: Path) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "program.md").write_text(
        """
**Target ports**:
- coder: 8071
- architect_coding: 8084
- NOTE: Entire stack fits in HOT tier with mlock on 512GB RAM. WARM tier demotion is unnecessary.
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "stale_autopilot_program_stack_guidance"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/program.md"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_stack_prior_autopilot_program_guidance(tmp_path: Path) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "program.md").write_text(
        """
**Target endpoints**: derive live primary endpoints from `orchestration/derived/stack_priors.yaml`.
Tier demotion is not an open exploration surface by default.
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "stale_autopilot_program_stack_guidance"
        for finding in findings
    )


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

    launch_targets = {"frontdoor": {"port": 8070, "tier": "hot"}}
    loose = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        launch_manifest_targets=launch_targets,
    )
    strict = validate_stack_priors(
        priors,
        strict=True,
        scan_surfaces=True,
        repo_root=tmp_path,
        launch_manifest_targets=launch_targets,
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
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
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
