"""Unit coverage for scripts.lib.onboard shared onboarding helper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from scripts.lib.onboard import (
    HealthCheckResult,
    ModelInfo,
    build_registry_entry,
    collect_model_info,
    detect_architecture,
    detect_family,
    detect_quantization,
    estimate_tier,
    extract_short_name,
    find_compatible_drafts,
    generate_compatible_targets_patterns,
    generate_optimization_options,
    generate_role_name,
    onboard_model,
    run_health_check,
    suggest_candidate_roles,
)


def _model_info(**overrides) -> ModelInfo:
    base = dict(
        path="/models/model.gguf",
        relative_path="model.gguf",
        filename="model.gguf",
        name="Qwen3-Coder-30B-A3B",
        architecture="qwen3moe",
        family="Qwen3-Coder",
        quantization="Q4_K_M",
        size_gb=30.0,
        tier="B",
        is_draft=False,
    )
    base.update(overrides)
    return ModelInfo(**base)


def test_detect_helpers_cover_key_patterns():
    assert detect_architecture("Qwen3-Coder-30B-A3B-Q4_K_M.gguf") == "qwen3moe"
    assert detect_architecture("Qwen3-VL-30B-A3B-Q4_K_M.gguf") == "qwen3vlmoe"
    assert detect_architecture("Qwen3-Next-80B-Q4_K_M.gguf") == "ssm_moe_hybrid"
    assert detect_architecture("Mixtral-8x7B-Q4_K_M.gguf") == "mixtral"
    assert detect_architecture("DenseModel-Q8_0.gguf") == "dense"

    assert detect_family("Qwen2.5-Coder-7B-Q4_K_M.gguf") == "Qwen2.5-Coder"
    assert detect_family("Qwen3-VL-30B.gguf") == "Qwen3-VL"
    assert detect_family("DeepSeek-R1-Distill-Llama-8B.gguf") == "DeepSeek-R1-Distill-Llama"
    assert detect_family("Meta-Llama-3.1-8B-Instruct.gguf") == "Llama-3.1"
    assert detect_family("Unknown-Model.gguf") == "Unknown"

    assert detect_quantization("foo-Q4_K_M.gguf") == "Q4_K_M"
    assert detect_quantization("foo-Q8_0.gguf") == "Q8_0"
    assert detect_quantization("foo-bf16.gguf") == "BF16"
    assert detect_quantization("foo.gguf") == "Unknown"

    assert extract_short_name("Qwen3-Coder-30B-A3B-Q4_K_M.gguf") == "Qwen3-Coder-30B-A3B"
    assert estimate_tier(1.0, is_draft=False) == "D"
    assert estimate_tier(8.0, is_draft=False) == "C"
    assert estimate_tier(40.0, is_draft=False) == "B"
    assert estimate_tier(80.0, is_draft=False) == "A"


def test_collect_model_info_relative_and_absolute_paths(tmp_path: Path):
    model_base = tmp_path / "models"
    model_base.mkdir(parents=True)
    rel_model = model_base / "Qwen3-Coder-30B-A3B-Q4_K_M.gguf"
    rel_model.write_bytes(b"x")

    info_rel = collect_model_info(str(rel_model.name), model_base=str(model_base))
    assert info_rel.path == str(rel_model)
    assert info_rel.relative_path == rel_model.name
    assert info_rel.architecture == "qwen3moe"
    assert info_rel.quantization == "Q4_K_M"
    assert info_rel.is_draft is True  # tiny file -> draft heuristic

    abs_model = tmp_path / "outside.gguf"
    abs_model.write_bytes(b"abc")
    info_abs = collect_model_info(str(abs_model), model_base=str(model_base))
    assert info_abs.path == str(abs_model)
    assert info_abs.relative_path == str(abs_model)

    with pytest.raises(FileNotFoundError):
        collect_model_info("does-not-exist.gguf", model_base=str(model_base))


def test_generate_optimization_options_for_dense_and_moe_architectures():
    reg = MagicMock()

    dense = _model_info(architecture="dense", is_draft=False)
    with patch("scripts.lib.onboard.Executor"):
        dense_cfg = generate_optimization_options(dense, reg)
    dense_types = [c.config_type for c in dense_cfg]
    assert dense_types.count("baseline") == 1
    assert "lookup" in dense_types

    moe = _model_info(architecture="qwen3moe")
    with patch("scripts.lib.onboard.Executor"):
        moe_cfg = generate_optimization_options(moe, reg)
    moe_types = [c.config_type for c in moe_cfg]
    assert "moe" in moe_types
    assert "moe_lookup" in moe_types

    ssm = _model_info(architecture="qwen3next")
    with patch("scripts.lib.onboard.Executor"):
        ssm_cfg = generate_optimization_options(ssm, reg)
    assert all(c.config_type in {"baseline", "moe"} for c in ssm_cfg)


def test_draft_target_compatibility_helpers():
    draft = _model_info(
        architecture="dense",
        is_draft=True,
        family="Qwen2.5-Coder",
        name="Qwen2.5-Coder-0.5B",
    )
    assert generate_compatible_targets_patterns(draft) == ["Qwen2.5", "Qwen2"]

    reg = MagicMock()
    reg.get_all_roles.return_value = ["draft_1", "not_draft"]
    reg.get_role_config.side_effect = [
        {"tier": "D", "compatible_targets": ["qwen3-coder"]},
        {"tier": "B", "compatible_targets": ["qwen3-coder"]},
    ]
    target = _model_info(architecture="dense", is_draft=False, name="qwen3-coder-30b")
    assert find_compatible_drafts(target, reg) == ["draft_1"]

    blocked = _model_info(architecture="qwen3moe")
    assert find_compatible_drafts(blocked, reg) == []
    assert find_compatible_drafts(_model_info(is_draft=True), reg) == []


def test_run_health_check_handles_binary_missing():
    info = _model_info(path="/tmp/model.gguf")
    reg = MagicMock()
    with patch("scripts.lib.onboard.validate_binaries", side_effect=FileNotFoundError("missing bin")):
        result = run_health_check(info, reg)
    assert result.success is False
    assert "missing bin" in (result.error_message or "")


def test_run_health_check_success_first_combo(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    info = _model_info(path=str(model), architecture="dense")
    reg = MagicMock()

    fake_parse = SimpleNamespace(tokens_per_second=77.7)
    completed = subprocess.CompletedProcess(
        args=["x"],
        returncode=0,
        stdout="tokens per second",
        stderr="",
    )
    with (
        patch("scripts.lib.onboard.validate_binaries"),
        patch("scripts.lib.onboard.get_binary", return_value="/bin/llama-completion"),
        patch("scripts.lib.onboard.parse_output", return_value=fake_parse),
        patch("scripts.lib.onboard.shutil.which", return_value=None),
        patch("scripts.lib.onboard.subprocess.run", return_value=completed),
    ):
        result = run_health_check(info, reg)

    assert result.success is True
    assert result.tokens_per_second == 77.7
    assert result.flags_used == ["--no-conversation"]


def test_run_health_check_timeout_then_failure(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    info = _model_info(path=str(model), architecture="qwen3moe")
    reg = MagicMock()

    # timeout, non-success output, generic exception, non-success output
    side_effects = [
        subprocess.TimeoutExpired(cmd=["x"], timeout=1),
        subprocess.CompletedProcess(args=["x"], returncode=0, stdout="", stderr=""),
        RuntimeError("boom"),
        subprocess.CompletedProcess(args=["x"], returncode=0, stdout="", stderr=""),
    ]
    with (
        patch("scripts.lib.onboard.validate_binaries"),
        patch("scripts.lib.onboard.get_binary", return_value="/bin/llama-completion"),
        patch("scripts.lib.onboard.shutil.which", return_value="numactl"),
        patch("scripts.lib.onboard.subprocess.run", side_effect=side_effects),
    ):
        result = run_health_check(info, reg)

    assert result.success is False
    assert "All flag combinations failed" in (result.error_message or "")


def test_role_suggestion_and_role_name_generation():
    draft = _model_info(is_draft=True, tier="D", name="tiny")
    assert suggest_candidate_roles(draft) == ["draft"]

    heavy = _model_info(
        name="Qwen3-Coder-Thinking-Math-VL",
        size_gb=60.0,
        tier="A",
        is_draft=False,
    )
    roles = suggest_candidate_roles(heavy)
    assert {"coder", "thinking", "math", "vision", "architect", "ingest", "general"} <= set(roles)

    small = _model_info(name="PlainModel", size_gb=5.0, tier="C", is_draft=False)
    small_roles = suggest_candidate_roles(small)
    assert "general" in small_roles
    assert "worker" in small_roles

    role_name = generate_role_name(heavy, roles)
    assert role_name.startswith("thinking_")
    assert len(role_name) > 10


def test_build_registry_entry_for_moe_and_draft_paths():
    info_moe = _model_info(architecture="qwen3moe", is_draft=False)
    health_ok = HealthCheckResult(success=True, tokens_per_second=12.3, flags_used=["--jinja"])
    entry = build_registry_entry(info_moe, ["coder", "general"], health_ok, compatible_targets=[])
    assert entry["tier"] == "B"
    assert entry["model"]["architecture"] == "qwen3moe"
    assert entry["acceleration"]["type"] == "moe_expert_reduction"
    assert entry["performance"]["baseline_tps"] == 12.3
    assert entry["launch_flags"] == ["--jinja"]

    info_ssm = _model_info(architecture="qwen3next")
    entry_ssm = build_registry_entry(info_ssm, ["general"], HealthCheckResult(success=False), [])
    assert entry_ssm["constraints"]["forbid"] == ["speculative_decoding", "prompt_lookup"]

    info_draft = _model_info(is_draft=True, architecture="dense")
    entry_draft = build_registry_entry(
        info_draft,
        ["draft"],
        HealthCheckResult(success=False),
        compatible_targets=["Qwen3"],
    )
    assert entry_draft["compatible_targets"] == ["Qwen3"]


def test_onboard_model_rejects_existing_path_and_handles_draft_and_target(tmp_path: Path):
    model = tmp_path / "target.gguf"
    model.write_bytes(b"x")
    draft_file = tmp_path / "draft.gguf"
    draft_file.write_bytes(b"x")

    registry = MagicMock()
    registry.path_exists_in_registry.return_value = "existing_role"
    with patch("scripts.lib.onboard.collect_model_info", return_value=_model_info(path=str(model))):
        with pytest.raises(ValueError, match="already in registry"):
            onboard_model(str(model), registry=registry, model_base=str(tmp_path))

    registry.path_exists_in_registry.return_value = None
    registry.get_model_path.return_value = str(draft_file)

    with (
        patch("scripts.lib.onboard.collect_model_info", return_value=_model_info(path=str(model), is_draft=False)),
        patch("scripts.lib.onboard.generate_optimization_options", return_value=[]),
        patch("scripts.lib.onboard.find_compatible_drafts", return_value=["draft_role"]),
        patch("scripts.lib.onboard.run_health_check", return_value=HealthCheckResult(success=True, tokens_per_second=11.0)),
        patch("scripts.lib.onboard.suggest_candidate_roles", return_value=["general"]),
        patch("scripts.lib.onboard.generate_role_name", return_value="general_target"),
        patch("scripts.lib.onboard.build_registry_entry", return_value={"ok": True}),
        patch("scripts.lib.onboard.os.path.exists", return_value=True),
    ):
        result = onboard_model(str(model), registry=registry, model_base=str(tmp_path))

    spec_cfgs = [c for c in result.configs if c.config_type == "spec"]
    assert len(spec_cfgs) == 6
    assert result.compatible_drafts == ["draft_role"]
    assert result.suggested_role_name == "general_target"

    with (
        patch("scripts.lib.onboard.collect_model_info", return_value=_model_info(path=str(model), is_draft=True)),
        patch("scripts.lib.onboard.generate_optimization_options", return_value=[]),
        patch("scripts.lib.onboard.generate_compatible_targets_patterns", return_value=["Qwen3"]),
        patch("scripts.lib.onboard.run_health_check", return_value=HealthCheckResult(success=False)),
        patch("scripts.lib.onboard.suggest_candidate_roles", return_value=["draft"]),
        patch("scripts.lib.onboard.generate_role_name", return_value="draft_model"),
        patch("scripts.lib.onboard.build_registry_entry", return_value={"ok": True}),
    ):
        result_draft = onboard_model(str(model), registry=registry, model_base=str(tmp_path))

    assert result_draft.compatible_drafts == []
    assert result_draft.compatible_targets == ["Qwen3"]
