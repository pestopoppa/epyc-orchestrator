"""Model preflight validates the files the active generated launcher will use."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.server import stack_commands, stack_manifest as manifest


def live(requirements, *, family="llama.cpp", status="live_stack"):
    return {
        "deployment_status": status,
        "serving": {
            "launch": {
                "requirements": requirements,
                "runtime": {"binary_family": family},
            }
        },
    }


@pytest.fixture
def declared(tmp_path, monkeypatch):
    model = tmp_path / "actual.gguf"
    model.touch()
    models = tmp_path / "models"
    models.mkdir()
    (models / "LightOnOCR-2-1B-bbox-Q4_K_M.gguf").touch()
    (tmp_path / "orchestration/derived").mkdir(parents=True)
    (tmp_path / "orchestration/tool_registry.yaml").touch()
    monkeypatch.setitem(manifest._PATHS, "project_root", tmp_path)
    monkeypatch.setitem(manifest._PATHS, "models_dir", models)
    monkeypatch.setattr(manifest, "EMBEDDING_MODEL_PATH", str(model))
    # These are real absent files; a stale constant must not veto active inputs.
    monkeypatch.setattr(manifest, "WORKER_POOL_MODELS", {"old": str(tmp_path / "retired.gguf")})
    monkeypatch.setattr(manifest, "EXPLORE_DRAFT_MODEL", str(tmp_path / "retired-draft.gguf"))
    monkeypatch.setattr(manifest, "VISION_WORKER_MODEL", str(tmp_path / "retired-vl.gguf"))
    monkeypatch.setattr(manifest, "VISION_WORKER_MMPROJ", str(tmp_path / "retired-mmproj.gguf"))
    return tmp_path, model


def write_roles(root, roles):
    (root / "orchestration/derived/stack_priors.yaml").write_text(yaml.safe_dump({"roles": roles}))


def test_actual_active_drafter_missing_is_not_a_false_green(declared):
    root, model = declared
    missing = root / "actual-dflash.gguf"
    write_roles(
        root,
        {"architect_critic": live({"model_path": str(model), "draft_model_path": str(missing)})},
    )
    errors = manifest.validate_model_paths()
    assert errors == [f"[HOT] architect_critic draft_model_path: {missing}"]
    missing.touch()
    assert manifest.validate_model_paths() == []


@pytest.mark.parametrize("kind", ["model_path", "draft_model_path", "mmproj_path"])
def test_each_active_file_requirement_is_checked(declared, kind):
    root, model = declared
    missing = root / "missing.gguf"
    requirements = {"model_path": str(model), kind: str(missing)}
    write_roles(root, {"worker_vision": live(requirements)})
    assert manifest.validate_model_paths() == [f"[HOT] worker_vision {kind}: {missing}"]


def test_nullable_optional_and_non_file_backend_needs_are_preserved(declared):
    root, model = declared
    write_roles(
        root,
        {
            "frontdoor": live(
                {"model_path": str(model), "draft_model_path": None, "mmproj_path": None}
            ),
            "remote": live({}, family="remote-http"),
        },
    )
    assert manifest.validate_model_paths() == []


def test_retired_paths_and_inactive_or_tenant_records_do_not_bind(declared):
    root, model = declared
    write_roles(
        root,
        {
            "frontdoor": live({"model_path": str(model)}),
            "retired": live({"model_path": str(root / "retired")}, status="retired"),
            "tenant": live({"model_path": str(root / "tenant")}, status="launcher_tenant"),
        },
    )
    assert manifest.validate_model_paths() == []


def test_aliases_and_repeated_requirements_check_each_active_file_once(declared):
    root, model = declared
    requirements = {"model_path": str(model), "draft_model_path": str(model)}
    write_roles(root, {"worker": live(requirements), "worker_general": live(requirements)})
    real = Path.is_file
    calls = []

    def counted(path):
        calls.append(str(path))
        return real(path)

    with patch.object(Path, "is_file", counted):
        assert manifest.validate_model_paths() == []
    assert calls.count(str(model)) == 1


@pytest.mark.parametrize(
    "requirements",
    [
        None,
        [],
        {"model_path": None},
        {"model_path": ""},
        {"model_path": 3},
        {"model_path": "ok", "draft_model_path": ""},
    ],
)
def test_malformed_active_requirements_refuse(declared, requirements):
    root, _ = declared
    write_roles(root, {"frontdoor": live(requirements)})
    assert any(
        "frontdoor" in error and ("invalid generated" in error or "malformed" in error)
        for error in manifest.validate_model_paths()
    )


@pytest.mark.parametrize("content", [None, "broken: [", "roles: {}", "roles: []"])
def test_missing_malformed_or_empty_generated_artifact_refuses(declared, content):
    root, _ = declared
    path = root / "orchestration/derived/stack_priors.yaml"
    if content is not None:
        path.write_text(content)
    assert any("requirements unavailable" in error for error in manifest.validate_model_paths())


def test_existing_auxiliary_and_tool_requirements_remain_required(declared):
    root, model = declared
    write_roles(root, {"frontdoor": live({"model_path": str(model)})})
    (root / "models/LightOnOCR-2-1B-bbox-Q4_K_M.gguf").unlink()
    (root / "orchestration/tool_registry.yaml").unlink()
    errors = manifest.validate_model_paths()
    assert len(errors) == 2
    assert errors[0].startswith("[AUX] document_formalizer:")
    assert errors[1].startswith("[TOOL] tool_registry.yaml:")


def test_actual_start_preflight_refuses_before_host_or_service_actions(
    declared, monkeypatch, capsys
):
    root, model = declared
    missing = root / "actual-drafter.gguf"
    write_roles(
        root,
        {"architect_critic": live({"model_path": str(model), "draft_model_path": str(missing)})},
    )
    monkeypatch.setattr(stack_commands, "_capacity_launch_gate", lambda: True)
    monkeypatch.setattr(stack_commands, "_start_registry_prelude", lambda *a, **kw: 0)
    monkeypatch.setattr(stack_commands, "_run_stack_change_launch_gate", lambda args: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("host/service action before model preflight")

    for name in (
        "apply_host_prerequisites",
        "RegistryLoader",
        "load_state",
        "start_server",
        "start_orchestrator",
        "start_aux_service",
        "start_docker_container",
    ):
        monkeypatch.setattr(stack_commands, name, forbidden)
    assert stack_commands.cmd_start(Namespace(dev=False)) == 1
    assert str(missing) in capsys.readouterr().out
