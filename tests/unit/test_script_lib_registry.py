"""Unit coverage for scripts.lib.registry shared runtime helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.lib.registry import ModelRegistry, get_all_roles, resolve_model_path


def _write_registry(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.fixture
def sample_registry(tmp_path: Path) -> ModelRegistry:
    payload = {
        "runtime_defaults": {
            "model_base_path": str(tmp_path / "models"),
            "context_limits": {
                "llama3_instruct": 131072,
                "llama3": 8192,
                "qwen3": 65536,
                "default": 4096,
            },
            "server_defaults": {
                "context_length": 32768,
                "flash_attention": True,
                "ubatch_size": 1536,
            },
        },
        "runtime_quirks": {
            "qwen3_vl_30b": {
                "quirks": [
                    {
                        "issue": "needs mmproj",
                        "workaround": "set --mmproj",
                        "discovered": "2026-01-01",
                    }
                ]
            }
        },
        "command_templates": {
            "baseline": "llama-cli -m {model_path}",
            "speculative_decoding": "llama-spec -m {model_path} -md {draft_path}",
        },
        "roles": {
            "target_main": {
                "tier": "B",
                "deprecated": False,
                "model": {
                    "name": "Qwen3-VL-30B-Instruct",
                    "path": "target.gguf",
                    "mmproj_path": "mmproj.gguf",
                    "architecture": "qwen3vlmoe",
                },
                "acceleration": {
                    "type": "moe_expert_reduction",
                    "override_key": "qwen3moe.expert_used_count",
                    "baseline_experts": 12,
                    "disallowed": ["prompt_lookup"],
                },
                "constraints": {
                    "forbid": ["legacy_mode"],
                    "flash_attention": False,
                    "ubatch_size": 2048,
                },
                "forbidden_configs": ["prompt_lookup", "legacy_mode"],
                "performance": {"baseline_tps": 5.0},
            },
            "target_blocked": {
                "tier": "C",
                "model": {
                    "name": "Qwen3-VL-30B-Blocked",
                    "path": "blocked.gguf",
                    "architecture": "dense",
                },
                "constraints": {"forbid": ["speculative_decoding"]},
            },
            "draft_small": {
                "tier": "D",
                "model": {
                    "name": "Qwen2.5-0.5B-Instruct",
                    "path": "draft.gguf",
                    "architecture": "dense",
                },
                "compatible_targets": ["qwen3-vl-30b"],
            },
            "legacy_abs": {
                "tier": "A",
                "deprecated": True,
                "model": {
                    "name": "Meta-Llama-3-8B-Instruct",
                    "path": "/abs/legacy.gguf",
                    "architecture": "dense",
                    "max_context": 77777,
                },
                "performance": {"baseline_tps": 40.0},
            },
        },
    }
    return ModelRegistry(str(_write_registry(tmp_path, payload)))


def test_roles_and_path_helpers(sample_registry: ModelRegistry):
    assert set(sample_registry.get_all_roles()) == {"target_main", "target_blocked", "draft_small"}
    assert "legacy_abs" in sample_registry.get_all_roles(include_deprecated=True)

    path = sample_registry.get_model_path("target_main")
    assert path and path.endswith("/target.gguf")
    assert sample_registry.get_model_path("legacy_abs") == "/abs/legacy.gguf"
    mmproj = sample_registry.get_mmproj_path("target_main")
    assert mmproj and mmproj.endswith("/mmproj.gguf")
    assert sample_registry.get_mmproj_path("missing") is None


def test_arch_tier_accel_and_constraints(sample_registry: ModelRegistry):
    assert sample_registry.get_architecture("target_main") == "qwen3vlmoe"
    assert sample_registry.get_architecture("missing") == "dense"
    assert sample_registry.get_tier("target_main") == "B"
    assert sample_registry.get_tier("missing") is None
    assert sample_registry.get_acceleration("missing") == {"type": "none"}
    assert sample_registry.get_constraints("missing") == {}
    assert sample_registry.get_command_template("baseline") == "llama-cli -m {model_path}"


def test_forbidden_and_quirks_lookup(sample_registry: ModelRegistry):
    forbidden = sample_registry.get_forbidden_configs("target_main")
    assert set(forbidden) == {"prompt_lookup", "legacy_mode"}

    quirks = sample_registry.get_quirks("target_main")
    assert len(quirks) == 1
    assert quirks[0]["issue"] == "needs mmproj"
    assert sample_registry.get_quirks("missing") == []


def test_draft_and_target_resolution(sample_registry: ModelRegistry):
    drafts = sample_registry.get_drafts_for_model("target_main")
    assert drafts == ["draft_small"]
    assert sample_registry.get_drafts_for_model("target_blocked") == []
    assert sample_registry.get_drafts_for_model("draft_small") == []

    targets = sample_registry.get_targets_for_draft("draft_small")
    assert targets == ["target_main"]
    assert sample_registry.get_targets_for_draft("target_main") == []
    assert sample_registry.get_targets_for_draft("missing") == []


def test_context_flash_ubatch_and_timeouts(sample_registry: ModelRegistry):
    # Explicit max_context
    assert sample_registry.get_max_context("legacy_abs") == 77777
    # Qwen family fallback to context_limits
    assert sample_registry.get_max_context("target_main") == 65536
    # Server default fallback
    assert sample_registry.get_max_context("missing") == 32768

    assert sample_registry.get_flash_attention("target_main") is False
    assert sample_registry.get_flash_attention("missing") is True
    assert sample_registry.get_ubatch_size("target_main") == 2048
    assert sample_registry.get_ubatch_size("missing") == 1536

    assert sample_registry.get_baseline_tps("target_main") == 5.0
    assert sample_registry.get_baseline_tps("missing") is None
    assert sample_registry.get_timeout_multiplier("target_main", reference_tps=20.0) == 4.0
    # Fast models never get below 1.0.
    assert sample_registry.get_timeout_multiplier("legacy_abs", reference_tps=20.0) == 1.0
    # No baseline data uses conservative fallback.
    assert sample_registry.get_timeout_multiplier("missing") == 2.0


def test_add_entry_and_path_exists(tmp_path: Path):
    payload = {
        "runtime_defaults": {"model_base_path": str(tmp_path / "models")},
        "roles": {
            "existing": {
                "tier": "C",
                "model": {"name": "x", "path": "x.gguf", "architecture": "dense"},
            }
        },
    }
    registry = ModelRegistry(str(_write_registry(tmp_path, payload)))

    with pytest.raises(ValueError, match="already exists"):
        registry.add_model_entry("existing", {"tier": "C", "model": {"path": "y.gguf", "architecture": "dense"}})
    with pytest.raises(ValueError, match="Missing required field: tier"):
        registry.add_model_entry("new1", {"model": {"path": "y.gguf", "architecture": "dense"}})
    with pytest.raises(ValueError, match="Missing required field: model.path"):
        registry.add_model_entry("new2", {"tier": "C", "model": {"architecture": "dense"}})
    with pytest.raises(ValueError, match="Missing required field: model.architecture"):
        registry.add_model_entry("new3", {"tier": "C", "model": {"path": "z.gguf"}})

    registry.add_model_entry(
        "new_ok",
        {
            "tier": "D",
            "model": {"name": "n", "path": "new.gguf", "architecture": "dense"},
        },
    )
    registry.reload()
    assert registry.role_exists("new_ok") is True
    assert registry.path_exists_in_registry("new.gguf") == "new_ok"
    assert registry.path_exists_in_registry("/does/not/exist.gguf") is None


def test_module_level_helpers_use_load_registry(monkeypatch):
    class _FakeRegistry:
        def get_all_roles(self, include_deprecated: bool = False):
            return ["a", "b"] if include_deprecated else ["a"]

        def get_model_path(self, role: str):
            return f"/models/{role}.gguf"

    monkeypatch.setattr("scripts.lib.registry.load_registry", lambda path=None: _FakeRegistry())

    assert get_all_roles() == ["a"]
    assert get_all_roles(include_deprecated=True) == ["a", "b"]
    assert resolve_model_path("worker") == "/models/worker.gguf"
