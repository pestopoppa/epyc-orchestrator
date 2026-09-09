from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from scripts.server import orchestrator_stack as launcher
from scripts.server import stack_manifest
from scripts.server import autokernel_enrollment as enrollment
from scripts.server.autokernel_enrollment import (
    ArtifactPin,
    EnrollmentExportError,
    ExportContext,
    SourcePin,
    _loaded_builder_identity,
    _verify_artifact_pins,
    capture_current_context,
    export_production_enrollment,
    seal_export_bundle,
)
from src.registry_loader import RegistryLoader


ROOT = Path(__file__).resolve().parents[2]


def _context(tmp_path: Path, monkeypatch=None, *, roles=("frontdoor",), mode="full", artifacts=()):
    # A lean registry is itself a fixed point of the compiler.  Keeping two pinned
    # byte copies exercises the master->lean drift guard without reaching another repo.
    lean = ROOT / "orchestration/model_registry.yaml"
    source_paths = {
        "lean_registry": lean,
        "descriptors": ROOT / "orchestration/model_descriptors.yaml",
        "launch_manifest": ROOT / "orchestration/launch_manifest.yaml",
        "topology": ROOT / "orchestration/stack_topology.yaml",
        "stack_runtime": ROOT / "scripts/server/stack_runtime.py",
        "stack_paths": ROOT / "scripts/server/stack_paths.py",
    }
    prior_path = ROOT / "orchestration/derived/stack_priors.yaml"
    if monkeypatch is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        prior = yaml.safe_load(prior_path.read_text())
        names = {"registry": "lean_registry", "descriptors": "descriptors",
                 "launch_manifest": "launch_manifest", "stack_topology": "topology",
                 "stack_runtime": "stack_runtime", "stack_paths": "stack_paths"}
        for prior_name, source_name in names.items():
            prior["source_artifacts"][prior_name]["sha256"] = hashlib.sha256(
                source_paths[source_name].read_bytes()).hexdigest()
        prior_path = tmp_path / "stack_priors.yaml"
        prior_path.write_text(yaml.safe_dump(prior, sort_keys=False))
        monkeypatch.setattr(launcher, "STACK_PRIORS_PATH", prior_path)
    paths = {
        "master_registry": lean,
        "lean_registry": lean,
        "launcher": ROOT / "scripts/server/orchestrator_stack.py",
        "launch_manifest": ROOT / "orchestration/launch_manifest.yaml",
        "topology": ROOT / "orchestration/stack_topology.yaml",
        "stack_env": ROOT / "scripts/server/stack_env.py",
        "stack_runtime": ROOT / "scripts/server/stack_runtime.py",
        "stack_priors": prior_path,
        "stack_manifest": ROOT / "scripts/server/stack_manifest.py",
        "stack_numa": ROOT / "scripts/server/stack_numa.py",
        "stack_paths": ROOT / "scripts/server/stack_paths.py",
        "descriptors": ROOT / "orchestration/model_descriptors.yaml",
    }
    sources = tuple(SourcePin(name, str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                              "test-revision") for name, path in paths.items())
    return ExportContext("test-export", "2026-09-09T00:00:00Z", mode, sources,
                         tuple(artifacts), (), tuple(roles), _loaded_builder_identity())


def test_export_uses_actual_builder_without_runtime_side_effects(tmp_path: Path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    with patch.object(Path, "mkdir", side_effect=AssertionError("runtime mkdir")), \
         patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("launch")), \
         patch.object(launcher, "pre_evict_nodes", side_effect=AssertionError("evict")):
        export = export_production_enrollment(context)
    row = next(item for item in export["targets"] if item["primary_role"] == "frontdoor")
    registry = RegistryLoader(ROOT / "orchestration/model_registry.yaml", validate_paths=False)
    expected = launcher.build_server_command(
        registry.get_role("frontdoor"), row["port"], numa_instance=row["numa_instance"],
        prepare_runtime_dirs=False)
    assert row["command_argv"] == expected
    assert row["argv"] == row["topology"]["argv_prefix"] + expected
    assert row["topology"]["argv_prefix"]
    assert row["status"] == "waiting_artifact"
    assert export["disposition"]["unsupported"] >= 2  # speech is retained, never omitted


def test_known_artifacts_make_row_ready_and_alias_deduplicates(tmp_path: Path, monkeypatch):
    first = export_production_enrollment(
        _context(tmp_path, monkeypatch, roles=("frontdoor", "worker_summarize")))
    row = next(item for item in first["targets"] if item["primary_role"] == "frontdoor")
    pins = [ArtifactPin("executable", row["command_argv"][0], "1" * 64),
            ArtifactPin("model", row["command_argv"][2], "2" * 64)]
    ld_dir = row["environment"]["LD_LIBRARY_PATH"].split(":")[0]
    pins.append(ArtifactPin("dso", str(Path(ld_dir) / "libggml.so"), "3" * 64))
    second = export_production_enrollment(
        _context(tmp_path, monkeypatch, roles=("frontdoor", "worker_summarize"), artifacts=pins))
    rows = [item for item in second["targets"] if item["primary_role"] == "frontdoor"]
    assert len(rows) == 1
    assert rows[0]["status"] == "ready"
    assert rows[0]["obligations"] == ["frontdoor", "worker_summarize"]
    assert rows[0]["optional_seed"] is False


def test_source_drift_refuses_before_builder(tmp_path: Path):
    context = _context(tmp_path)
    sources = tuple(replace(item, sha256="0" * 64) if item.name == "master_registry" else item
                    for item in context.sources)
    with patch.object(launcher.subprocess, "Popen") as popen:
        with pytest.raises(EnrollmentExportError, match="source drift"):
            export_production_enrollment(replace(context, sources=sources))
    popen.assert_not_called()


def test_instance_mode_changes_frozen_target_set(tmp_path: Path, monkeypatch):
    full = export_production_enrollment(
        _context(tmp_path / "a", monkeypatch, mode="full"))
    half = export_production_enrollment(
        _context(tmp_path / "b", monkeypatch, mode="quarter"))
    full_ports = {x.get("port") for x in full["targets"] if x["primary_role"] == "frontdoor"}
    half_ports = {x.get("port") for x in half["targets"] if x["primary_role"] == "frontdoor"}
    assert full_ports and half_ports and full_ports.isdisjoint(half_ports)


def test_missing_requested_role_is_independent_of_valid_target(tmp_path: Path, monkeypatch):
    export = export_production_enrollment(
        _context(tmp_path, monkeypatch, roles=("frontdoor", "does_not_exist")))
    assert any(row["primary_role"] == "frontdoor" for row in export["targets"])
    missing = next(row for row in export["targets"] if row["target_id"] == "does_not_exist")
    assert missing["status"] == "unsupported"


def test_context_direct_instance_is_revalidated(tmp_path: Path):
    context = _context(tmp_path)
    with pytest.raises(EnrollmentExportError, match="instance_mode"):
        export_production_enrollment(replace(context, instance_mode="moved"))


def test_unrelated_parent_environment_cannot_be_exported(tmp_path: Path):
    context = replace(_context(tmp_path), base_environment=(("SECRET_SENTINEL", "do-not-copy"),))
    with pytest.raises(EnrollmentExportError, match="unsupported keys"):
        export_production_enrollment(context)


def test_credential_like_allowed_prefix_cannot_be_exported(tmp_path: Path):
    context = replace(_context(tmp_path), base_environment=(("GGML_API_TOKEN", "secret"),))
    with pytest.raises(EnrollmentExportError, match="credential-like"):
        export_production_enrollment(context)


def test_loaded_configuration_must_match_current_pinned_file(tmp_path: Path):
    context = _context(tmp_path)
    moved = dict(stack_manifest._MANIFEST)
    moved["parity"] = {"loaded-only": True}
    with patch.object(stack_manifest, "_MANIFEST", moved):
        with pytest.raises(EnrollmentExportError, match="loaded launcher configuration"):
            export_production_enrollment(context)


def test_launcher_only_seed_remains_optional_but_production_alias_does_not(
        tmp_path: Path, monkeypatch):
    original = stack_manifest._filter_by_numa_mode

    def without_explicit_seed(entries, mode):
        return [entry for entry in original(entries, mode)
                if entry["roles"][0] != "eval_batch_frontdoor"]

    monkeypatch.setattr(stack_manifest, "_filter_by_numa_mode", without_explicit_seed)
    context = replace(
        _context(tmp_path, monkeypatch, roles=()),
        seed_roles=("eval_batch_frontdoor", "worker_summarize"))
    export = export_production_enrollment(context)
    explicit = next(row for row in export["targets"]
                    if row["primary_role"] == "eval_batch_frontdoor")
    production = next(row for row in export["targets"]
                      if row["primary_role"] == "frontdoor")
    assert explicit["optional_seed"] is True
    assert production["optional_seed"] is False


def test_factory_captures_loaded_sources_without_artifact_hashing(tmp_path: Path, monkeypatch):
    expected = _context(tmp_path, monkeypatch)
    captured = capture_current_context(
        master_registry=ROOT / "orchestration/model_registry.yaml",
        revision="fixture-revision", instance_mode="full", requested_roles=("frontdoor",))
    assert {pin.name for pin in captured.sources} == {pin.name for pin in expected.sources}
    assert captured.artifacts == ()
    export = export_production_enrollment(captured)
    assert any(row["primary_role"] == "frontdoor" for row in export["targets"])


def test_stale_compiled_prior_dependency_refuses(tmp_path: Path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    prior_path = Path(next(pin.path for pin in context.sources if pin.name == "stack_priors"))
    prior = yaml.safe_load(prior_path.read_text())
    prior["source_artifacts"]["descriptors"]["sha256"] = "0" * 64
    prior_path.write_text(yaml.safe_dump(prior, sort_keys=False))
    sources = tuple(replace(pin, sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest())
                    if pin.name == "stack_priors" else pin for pin in context.sources)
    with pytest.raises(EnrollmentExportError, match="stale for descriptors"):
        export_production_enrollment(replace(context, sources=sources))


def test_artifact_verification_is_opt_in_and_cached_per_path(tmp_path: Path):
    artifact = tmp_path / "small.bin"
    artifact.write_bytes(b"small")
    digest = hashlib.sha256(b"small").hexdigest()
    pins = (ArtifactPin("model", str(artifact), digest),
            ArtifactPin("dso", str(artifact), digest))
    original_open = Path.open
    opens = 0
    def counted_open(path, *args, **kwargs):
        nonlocal opens
        if path == artifact:
            opens += 1
        return original_open(path, *args, **kwargs)
    with patch.object(Path, "open", counted_open):
        assert _verify_artifact_pins(pins) == {}
    assert opens == 1
    artifact.write_bytes(b"moved")
    failures = _verify_artifact_pins(pins)
    assert failures == {("model", str(artifact)): "sha256_mismatch",
                        ("dso", str(artifact)): "sha256_mismatch"}


def test_prepare_runtime_dirs_false_only_skips_mkdir(monkeypatch):
    role = Mock()
    role.name = "frontdoor"
    role.model.full_path = "/models/example.gguf"
    role.acceleration.type = "none"
    monkeypatch.setattr(launcher, "_stack_prior_launch", lambda role: ({}, {}))
    monkeypatch.setattr(launcher, "_resolve_parallel_slots", lambda *a: "1")
    monkeypatch.setattr(launcher, "_resolve_thread_count", lambda *a: "8")
    with patch.object(Path, "mkdir") as mkdir:
        pure = launcher._build_role_command(role, 8070, prepare_runtime_dirs=False)
        mkdir.assert_not_called()
    with patch.object(Path, "mkdir") as mkdir:
        normal = launcher._build_role_command(role, 8070)
        mkdir.assert_called_once()
    assert pure == normal
    with pytest.raises(TypeError, match="must be bool"):
        launcher._build_role_command(role, 8070, prepare_runtime_dirs=1)


def test_default_dispatch_preserves_private_builder_call_shape(monkeypatch):
    role = Mock(name="role")
    role.name = "frontdoor"
    builder = Mock(return_value=["server"])
    monkeypatch.setattr(launcher, "_build_role_command", builder)
    launcher.build_server_command(role, 8070)
    builder.assert_called_once_with(role, 8070, 0)
    with pytest.raises(TypeError, match="must be bool"):
        launcher.build_server_command(role, 8070, prepare_runtime_dirs="false")


def test_bundle_seals_actual_per_target_recipe_bytes_and_aliases_are_cosmetic(
        tmp_path: Path, monkeypatch):
    first = export_production_enrollment(_context(tmp_path, monkeypatch))
    frontdoor = next(row for row in first["targets"] if row["primary_role"] == "frontdoor")
    pins = [ArtifactPin("executable", frontdoor["command_argv"][0], "1" * 64),
            ArtifactPin("model", frontdoor["command_argv"][2], "2" * 64)]
    ld_dir = frontdoor["environment"]["LD_LIBRARY_PATH"].split(":")[0]
    pins.append(ArtifactPin("dso", str(Path(ld_dir) / "libggml.so"), "3" * 64))
    export = export_production_enrollment(
        _context(tmp_path, monkeypatch, artifacts=pins))
    out = tmp_path / "enrollment.json"
    sealed = seal_export_bundle(export, out)
    row = next(item for item in sealed["targets"] if item["primary_role"] == "frontdoor")
    recipe = next(item for item in row["artifacts"] if item["use"] == "recipe")
    assert hashlib.sha256(Path(recipe["path"]).read_bytes()).hexdigest() == recipe["sha256"]
    alias_changed = copy.deepcopy(row)
    alias_changed["aliases"].append("display-only")
    alias_changed["obligations"].append("display-only")
    assert enrollment._recipe_body(alias_changed) == enrollment._recipe_body(row)


def test_bundle_retry_after_output_failure_and_tamper_refusal(
        tmp_path: Path, monkeypatch):
    export = export_production_enrollment(_context(tmp_path, monkeypatch))
    out = tmp_path / "enrollment.json"
    original = enrollment._publish_exact_at
    failed = False

    def fail_output(directory_fd, name, body):
        nonlocal failed
        if name == out.name and not failed:
            failed = True
            raise OSError("injected output failure")
        return original(directory_fd, name, body)

    with patch.object(enrollment, "_publish_exact_at", side_effect=fail_output):
        with pytest.raises(OSError, match="injected"):
            seal_export_bundle(export, out)
    assert not out.exists()
    sealed = seal_export_bundle(export, out)
    recipe = next(item for row in sealed["targets"] for item in row.get("artifacts", [])
                  if item.get("use") == "recipe")
    Path(recipe["path"]).write_bytes(b"tampered")
    with pytest.raises(EnrollmentExportError, match="differs"):
        seal_export_bundle(export, out)


def test_actual_both_mode_full_and_quarter_launches_have_distinct_recipe_bytes(
        tmp_path: Path, monkeypatch):
    export = export_production_enrollment(
        _context(tmp_path, monkeypatch, mode="both"))
    sealed = seal_export_bundle(export, tmp_path / "both.json")
    frontdoors = [row for row in sealed["targets"] if row["primary_role"] == "frontdoor"]
    assert len(frontdoors) == 3
    recipe_digests = {
        next(item["sha256"] for item in row["artifacts"] if item["use"] == "recipe")
        for row in frontdoors
    }
    assert len(recipe_digests) == 3
