"""Tests for stack-prior drift validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

import scripts.validate.stack_change_guard as stack_change_guard
from scripts.validate.stack_change_guard import (
    HARDCODED_SURFACE_RULES,
    REQUIRED_CONSUMER_SURFACE_IDS,
    GuardResult,
    HardcodedSurfaceRule,
    hardcoded_surface_rule_inventory,
    hardcoded_surface_warning_counts,
    main as stack_change_guard_main,
    scan_hardcoded_surfaces,
    validate_surface_manifest,
    validate_stack_priors,
)
from src.registry.stack_priors import STACK_PRIORS_VERSION, stack_priors_contract


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _consumer_surface(surface_id: str = "unit_consumer") -> dict:
    return {
        "surface_id": surface_id,
        "classification": "typed_consumer",
        "owner": "stack-change-governance",
        "consumer_scope": "unit consumer surface",
        "source_of_truth": "generated stack priors",
        "promotion_blocker": True,
        "review_cadence": "every stack change",
        "validation_command": "uv run pytest -q tests/unit/test_stack_change_guard.py",
        "implementation_refs": ["src/unit.py"],
        "drift_response": "derive from generated truth",
    }


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


def test_hardcoded_surface_rule_inventory_is_machine_readable() -> None:
    inventory = hardcoded_surface_rule_inventory()
    rule_ids = [rule["rule_id"] for rule in inventory["rules"]]

    assert inventory["version"] == 1
    assert inventory["rule_count"] == len(HARDCODED_SURFACE_RULES)
    assert inventory["consumer_surface_count"] == 0
    assert len(rule_ids) == len(set(rule_ids))
    assert "production_blocker" in inventory["categories"]
    assert any(
        rule["rule_id"] == "seeding_baseline_tps_table"
        and rule["category"] == "production_blocker"
        and "scripts/benchmark/seeding_rewards.py" in rule["path_globs"]
        for rule in inventory["rules"]
    )
    assert all("ownership" in rule for rule in inventory["rules"])


def test_hardcoded_surface_rule_inventory_includes_consumer_surfaces() -> None:
    inventory = hardcoded_surface_rule_inventory(
        ownership_manifest={
            "consumer_surfaces": [
                _consumer_surface("q_scorer_priors"),
                _consumer_surface("runtime_attestation"),
            ]
        }
    )

    assert inventory["consumer_surface_count"] == 2
    assert [surface["surface_id"] for surface in inventory["consumer_surfaces"]] == [
        "q_scorer_priors",
        "runtime_attestation",
    ]
    assert all(
        surface["classification"] == "typed_consumer"
        for surface in inventory["consumer_surfaces"]
    )


def test_validate_surface_manifest_accepts_complete_rule_ownership(tmp_path: Path) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit test surface",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "uv run python scripts/validate/stack_change_guard.py",
                    "drift_response": "remove stale hardcode",
                }
            ],
        },
    )

    assert validate_surface_manifest(manifest, rules=(rule,)) == []


def test_validate_surface_manifest_accepts_consumer_surface_ownership(tmp_path: Path) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit scanner surface",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "uv run python scripts/validate/stack_change_guard.py",
                    "drift_response": "remove stale hardcode",
                }
            ],
            "consumer_surfaces": [_consumer_surface()],
        },
    )

    assert validate_surface_manifest(
        manifest,
        rules=(rule,),
        required_consumer_surface_ids=frozenset({"unit_consumer"}),
    ) == []


def test_validate_surface_manifest_rejects_missing_consumer_surface(
    tmp_path: Path,
) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit scanner surface",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "guard",
                    "drift_response": "remove stale hardcode",
                }
            ],
        },
    )

    errors = validate_surface_manifest(
        manifest,
        rules=(rule,),
        required_consumer_surface_ids=frozenset({"unit_consumer"}),
    )

    assert any("'consumer_surfaces'" in error for error in errors)


def test_validate_surface_manifest_rejects_consumer_surface_metadata_drift(
    tmp_path: Path,
) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    bad_surface = {
        **_consumer_surface(),
        "classification": "unknown",
        "promotion_blocker": "yes",
        "implementation_refs": [],
    }
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit scanner surface",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "guard",
                    "drift_response": "remove stale hardcode",
                }
            ],
            "consumer_surfaces": [bad_surface, _consumer_surface()],
        },
    )

    errors = validate_surface_manifest(
        manifest,
        rules=(rule,),
        required_consumer_surface_ids=frozenset({"unit_consumer"}),
    )

    assert any("classification 'unknown'" in error for error in errors)
    assert any("missing boolean 'promotion_blocker'" in error for error in errors)
    assert any("implementation_refs" in error for error in errors)
    assert any("surface_id 'unit_consumer' is duplicated" in error for error in errors)


def test_validate_surface_manifest_rejects_missing_rule_ownership(tmp_path: Path) -> None:
    rules = (
        HardcodedSurfaceRule(
            rule_id="covered_rule",
            category="historical_doc",
            pattern=r"\bold\b",
            path_globs=("docs/**/*.md",),
            remediation="label historical",
        ),
        HardcodedSurfaceRule(
            rule_id="missing_rule",
            category="legacy_test",
            pattern=r"\bold\b",
            path_globs=("tests/**/*.py",),
            remediation="migrate fixture",
        ),
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "covered_rule",
                    "category": "historical_doc",
                    "owner": "documentation-governance",
                    "consumer_scope": "docs",
                    "promotion_blocker": False,
                    "review_cadence": "every stack change",
                    "evidence_command": "guard --all-hardcoded-surfaces",
                    "drift_response": "label historical",
                }
            ],
        },
    )

    errors = validate_surface_manifest(manifest, rules=rules)

    assert any("missing_rule" in error for error in errors)


def test_validate_surface_manifest_rejects_category_policy_drift(tmp_path: Path) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "legacy_test",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit test surface",
                    "promotion_blocker": False,
                    "review_cadence": "every stack change",
                    "evidence_command": "guard",
                    "drift_response": "remove stale hardcode",
                }
            ],
        },
    )

    errors = validate_surface_manifest(manifest, rules=(rule,))

    assert any("category 'legacy_test'" in error for error in errors)
    assert any("promotion_blocker=False" in error for error in errors)


def test_hardcoded_surface_warning_counts_bucket_unique_warnings() -> None:
    warnings = [
        "hardcoded_surface.production_blocker.seeding_baseline_tps_table: file:1",
        "hardcoded_surface.production_blocker.seeding_baseline_tps_table: file:1",
        "hardcoded_surface.waived.production_blocker.retired_role_in_active_code: file:2",
        "hardcoded_surface.legacy_test.retired_role_in_tests: file:3",
        "non-surface warning",
    ]

    assert hardcoded_surface_warning_counts(warnings) == {
        "production_blocker": 1,
        "waived_production_blocker": 1,
        "legacy_test": 1,
    }


def test_stack_change_guard_can_print_surface_rule_inventory_json(capsys) -> None:
    rc = stack_change_guard_main(
        ["--list-hardcoded-surface-rules", "--surface-inventory-format", "json"]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rule_count"] == len(HARDCODED_SURFACE_RULES)
    assert set(surface["surface_id"] for surface in payload["consumer_surfaces"]) == set(
        REQUIRED_CONSUMER_SURFACE_IDS
    )
    assert any(
        rule["rule_id"] == "stale_autopilot_program_stack_guidance"
        and rule["ownership"]["owner"] == "autopilot-instrumentation"
        for rule in payload["rules"]
    )


def test_stack_change_guard_can_print_summary_only(monkeypatch, capsys) -> None:
    warnings = [
        "hardcoded_surface.production_blocker.seeding_baseline_tps_table: file:1",
        "hardcoded_surface.production_blocker.seeding_baseline_tps_table: file:1",
        "hardcoded_surface.waived.production_blocker.retired_role_in_active_code: file:2",
        "hardcoded_surface.historical_doc.retired_role_in_operator_docs: docs:1",
        "non-surface warning",
    ]

    def fake_validate_stack_priors(*args, **kwargs) -> GuardResult:
        return GuardResult(errors=[], warnings=warnings)

    monkeypatch.setattr(
        stack_change_guard,
        "validate_stack_priors",
        fake_validate_stack_priors,
    )

    rc = stack_change_guard_main(["--surface-summary-only"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "WARN: 4 unique stack-prior warning(s) (5 total)" in output
    assert (
        "surface_warnings: production_blocker=1, "
        "waived_production_blocker=1, historical_doc=1"
    ) in output
    assert "other_warnings: 1" in output
    assert "file:1" not in output


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


def test_scan_hardcoded_surfaces_flags_retired_lean_registry_role(tmp_path: Path) -> None:
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    (orchestration / "model_registry_lean.yaml").write_text(
        "roles:\n  architect_coding:\n    tier: B\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "retired_role_in_lean_registry"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "orchestration/model_registry_lean.yaml"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_retired_source_access_role(tmp_path: Path) -> None:
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    (orchestration / "source_registry.yaml").write_text(
        "role_access:\n  architect_coding:\n    enabled: true\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "retired_role_in_source_access"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "orchestration/source_registry.yaml"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_retired_quality_signature_role(tmp_path: Path) -> None:
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    (orchestration / "model_quality_signatures.yaml").write_text(
        "models:\n  retired:\n    role: architect_coding\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "retired_role_in_quality_signature"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "orchestration/model_quality_signatures.yaml"
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


def test_scan_hardcoded_surfaces_flags_stale_autopilot_program_model_claims(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "program.md").write_text(
        """
- **Q-scorer frontdoor throughput**: Currently uses 19.6 t/s but actual is 12.7 t/s.
try-cheap-first (Qwen3-Coder-30B-A3B, fastest)
  -> frontdoor (Qwen3.5-35B-A3B, quality gate)
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


def test_scan_hardcoded_surfaces_flags_autopilot_program_local_yaml_reader(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "program.md").write_text(
        """
```python
import yaml
data = yaml.safe_load(open("orchestration/derived/stack_priors.yaml")) or {}
```
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


def test_scan_hardcoded_surfaces_flags_static_cli_degraded_status_targets(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "cli_orch.py").write_text(
        "FALLBACK_STATUS_TARGETS = [('frontdoor/coder_escalation', 8070)]\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_cli_degraded_status_targets"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/cli_orch.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_autopilot_preflight_targets(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "preflight_audit.py").write_text(
        "FALLBACK_MODEL_SERVER_TARGETS = [('frontdoor', 'http://localhost:8070/health')]\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_autopilot_preflight_targets"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/preflight_audit.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_stale_corpus_quality_gate_models(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "scripts" / "benchmark"
    benchmark.mkdir(parents=True)
    (benchmark / "corpus_quality_gate.py").write_text(
        """
FALLBACK_MODELS = {
    "frontdoor": {"port": 8070, "name": "Qwen3.6-35B-A3B Q8_0"},
}
parser.add_argument("--models", default=["7b", "32b"])
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "stale_corpus_quality_gate_models"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/benchmark/corpus_quality_gate.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_local_config_stack_prior_reader(
    tmp_path: Path,
) -> None:
    config = tmp_path / "src" / "config"
    config.mkdir(parents=True)
    (config / "models.py").write_text(
        """
import yaml

payload = yaml.safe_load(priors_path.read_text(encoding="utf-8")) or {}
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "local_config_stack_prior_yaml_reader"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/config/models.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_local_q_scorer_stack_prior_reader(
    tmp_path: Path,
) -> None:
    repl_memory = tmp_path / "orchestration" / "repl_memory"
    repl_memory.mkdir(parents=True)
    (repl_memory / "q_scorer.py").write_text(
        """
import yaml

data = yaml.safe_load(stack_priors_path.read_text(encoding="utf-8")) or {}
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "local_q_scorer_stack_prior_yaml_reader"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "orchestration/repl_memory/q_scorer.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_local_generated_docs_stack_prior_reader(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    registry = tmp_path / "scripts" / "registry"
    autopilot.mkdir(parents=True)
    registry.mkdir(parents=True)
    (autopilot / "gen_system_card.py").write_text(
        'stack_priors = _load_yaml(root_path / "orchestration" / "derived" / "stack_priors.yaml")\n',
        encoding="utf-8",
    )
    (registry / "render_stack_summary.py").write_text(
        "stack_priors = load_yaml(stack_priors_path)\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    finding_paths = {
        finding.path.as_posix()
        for finding in findings
        if finding.rule_id == "local_generated_docs_stack_prior_yaml_reader"
        and finding.category == "production_blocker"
    }
    assert finding_paths == {
        "scripts/autopilot/gen_system_card.py",
        "scripts/registry/render_stack_summary.py",
    }


def test_scan_hardcoded_surfaces_flags_static_factual_risk_role_tiers(
    tmp_path: Path,
) -> None:
    classifier = tmp_path / "src" / "classifiers"
    classifier.mkdir(parents=True)
    (classifier / "factual_risk.py").write_text(
        "_ROLE_TO_TIER: dict[str, str] = {'frontdoor': 'tier_3'}\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_factual_risk_role_tiers"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/classifiers/factual_risk.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_openai_model_role_order(
    tmp_path: Path,
) -> None:
    route = tmp_path / "src" / "api" / "routes"
    route.mkdir(parents=True)
    (route / "openai_compat.py").write_text(
        "PREFERRED_ROLE_ORDER = ('frontdoor', 'coder_escalation')\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_openai_model_role_order"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/api/routes/openai_compat.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_inference_lock_role_policy(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inference_lock.py").write_text(
        """
HEAVY_ROLES: frozenset[str] = frozenset({"frontdoor", "architect_general"})
LIGHT_ROLES = frozenset({"worker_general"})
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_inference_lock_role_policy"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/runtime/inference_lock.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_derived_inference_lock_policy(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inference_lock.py").write_text(
        """
_LEGACY_HEAVY_ROLES = frozenset({"frontdoor", "architect_general"})
_LEGACY_LIGHT_ROLES = frozenset({"worker_general"})
_DERIVED_LOCK_ROLES = _lock_roles_from_stack_priors()
if _DERIVED_LOCK_ROLES is None:
    HEAVY_ROLES = _LEGACY_HEAVY_ROLES
    LIGHT_ROLES = _LEGACY_LIGHT_ROLES
else:
    HEAVY_ROLES, LIGHT_ROLES = _DERIVED_LOCK_ROLES
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "static_inference_lock_role_policy"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_inference_tap_stream_policy(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inference_tap.py").write_text(
        'SAFE_NON_STREAM_ROLES: frozenset[str] = frozenset({"architect_general"})\n',
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_inference_tap_stream_policy"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/runtime/inference_tap.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_derived_inference_tap_stream_policy(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inference_tap.py").write_text(
        """
_LEGACY_SAFE_NON_STREAM_ROLES: frozenset[str] = frozenset({"architect_general"})
_DERIVED_SAFE_NON_STREAM_ROLES = _safe_non_stream_roles_from_stack_priors()
SAFE_NON_STREAM_ROLES: frozenset[str] = (
    _LEGACY_SAFE_NON_STREAM_ROLES
    if _DERIVED_SAFE_NON_STREAM_ROLES is None
    else _DERIVED_SAFE_NON_STREAM_ROLES
)
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "static_inference_tap_stream_policy"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_autopilot_kv_port_map(tmp_path: Path) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "kv_compress.py").write_text(
        "PRODUCTION_PORTS = {'frontdoor': 8070}\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_autopilot_kv_production_ports"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/kv_compress.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_kv_degraded_fallback_port_map(tmp_path: Path) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "kv_compress.py").write_text(
        "_FALLBACK_PRODUCTION_PORTS = {'frontdoor': 8070}\n"
        "PRODUCTION_PORTS = _FALLBACK_PRODUCTION_PORTS\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "static_autopilot_kv_production_ports"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_stale_launch_wrapper_inventory(tmp_path: Path) -> None:
    server = tmp_path / "scripts" / "server"
    server.mkdir(parents=True)
    (server / "launch_production.sh").write_text(
        """
echo "  - architect_coding (8084): Qwen3-Coder-480B-A35B"
echo "RAM breakdown (full mode):"
echo "  - Total: ~535GB"
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "stale_launch_wrapper_static_inventory"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/server/launch_production.sh"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_manifest_derived_launch_wrapper(tmp_path: Path) -> None:
    server = tmp_path / "scripts" / "server"
    server.mkdir(parents=True)
    (server / "launch_production.sh").write_text(
        """
echo "Mode: FULL production stack (manifest-defined HOT tier)"
echo "Launch inventory comes from scripts/server/stack_manifest.py."
echo "Use --status after launch for live processes, ports, and residency."
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "stale_launch_wrapper_static_inventory"
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
