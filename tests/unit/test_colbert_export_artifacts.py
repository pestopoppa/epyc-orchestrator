"""Tests for ColBERT export/download artifact profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.benchmark.colbert import export_lateon_onnx_int8 as export


def test_lateon_profile_declares_prebuilt_onnx_runtime_files():
    profile = export.resolve_profile("lateon")

    assert profile.repo_id == "lightonai/LateOn"
    assert profile.ships_prebuilt_onnx is True
    assert "model_int8.onnx" in profile.download_files
    assert profile.default_out == Path("/mnt/raid0/llm/models/lateon-onnx-int8")


def test_reason_mxbai_profile_is_source_only():
    profile = export.resolve_profile("reason-mxbai")

    assert profile.repo_id == "DataScience-UIBK/Reason-mxbai-colbert-v0-32m"
    assert profile.ships_prebuilt_onnx is False
    assert "model.safetensors" in profile.download_files
    assert "model_int8.onnx" not in profile.download_files
    assert profile.model_slot_env == "REASON_MXBAI_MODEL_PATH"


def test_known_model_id_selects_matching_profile():
    profile = export.resolve_profile(
        "lateon",
        "DataScience-UIBK/Reason-mxbai-colbert-v0-32m",
    )

    assert profile.name == "reason-mxbai"


def test_custom_model_id_reuses_selected_manifest():
    profile = export.resolve_profile("reason-mxbai", "example/custom-colbert")

    assert profile.name == "custom:example/custom-colbert"
    assert profile.repo_id == "example/custom-colbert"
    assert profile.ships_prebuilt_onnx is False
    assert profile.download_files == export.MODEL_PROFILES["reason-mxbai"].download_files


def test_artifact_plan_is_serializable():
    profile = export.resolve_profile("reason-mxbai")
    plan = export.artifact_plan(profile, Path("/tmp/reason"))

    assert plan["profile"] == "reason-mxbai"
    assert plan["out"] == "/tmp/reason"
    assert plan["runtime_files"] == ["model_int8.onnx", "tokenizer.json"]


def test_source_only_profile_fails_fast_without_onnx(tmp_path):
    profile = export.resolve_profile("reason-mxbai")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    with pytest.raises(export.OnnxArtifactsMissingError, match="does not ship"):
        export.ensure_onnx_artifacts(tmp_path, profile)


def test_print_plan_json_does_not_download(capsys):
    rc = export.main(["--profile", "reason-mxbai", "--print-plan", "--json"])

    assert rc == 0
    out = capsys.readouterr().out
    assert '"profile": "reason-mxbai"' in out
    assert '"ships_prebuilt_onnx": false' in out


def test_main_returns_export_required_for_source_only_parity(tmp_path):
    rc = export.main([
        "--profile",
        "reason-mxbai",
        "--out",
        str(tmp_path),
        "--no-download",
    ])

    assert rc == 3
