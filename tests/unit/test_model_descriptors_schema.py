"""Schema checks for the model-capability descriptor seed."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR_PATH = REPO_ROOT / "orchestration" / "model_descriptors.yaml"
REGISTRY_PATH = REPO_ROOT / "orchestration" / "model_registry.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        loaded = yaml.safe_load(fh)
    assert isinstance(loaded, dict)
    return loaded


def test_descriptor_metadata_is_versioned_and_timestamped() -> None:
    descriptors = _load_yaml(DESCRIPTOR_PATH)

    assert descriptors["descriptor_version"] == 3
    compiled_at = descriptors["compiled_at"]
    assert isinstance(compiled_at, str)
    datetime.strptime(compiled_at, "%Y-%m-%dT%H:%M:%SZ")
    assert descriptors["model_id_policy"]["invariant"].endswith("never by role name")


def test_model_records_are_unique_and_not_role_keys() -> None:
    descriptors = _load_yaml(DESCRIPTOR_PATH)
    registry = _load_yaml(REGISTRY_PATH)
    role_names = set((registry.get("roles") or {}).keys())
    server_role_names = set(
        key for key, value in (registry.get("server_mode") or {}).items() if isinstance(value, dict)
    )

    models = descriptors["models"]
    model_ids = [model["model_id"] for model in models]

    assert len(model_ids) == len(set(model_ids))
    assert not set(model_ids) & role_names
    assert not set(model_ids) & server_role_names


def test_every_model_has_consumer_ready_sections() -> None:
    descriptors = _load_yaml(DESCRIPTOR_PATH)

    required_model_fields = {
        "model_id",
        "family",
        "arch",
        "params_b",
        "active_b",
        "quant",
        "mem_gb",
        "ctx_max",
        "modalities",
        "role_bindings",
        "quality",
        "speed",
        "acceleration",
        "serving",
        "known_gaps",
    }
    required_quality_fields = {"suite_vector", "source", "eval_protocol", "measured"}
    required_speed_fields = {"solo_96t_tps", "quarter_48t_tps", "prefill_tps", "source"}
    required_accel_fields = {
        "spec_type",
        "draft_compat",
        "enable_thinking",
        "thinking_control",
        "kv",
    }
    required_serving_fields = {"binary", "numa_policy", "mlock", "ports"}
    required_serving_fields.add("requirements")

    for model in descriptors["models"]:
        assert required_model_fields <= set(model), model["model_id"]
        assert required_quality_fields <= set(model["quality"]), model["model_id"]
        assert required_speed_fields <= set(model["speed"]), model["model_id"]
        assert required_accel_fields <= set(model["acceleration"]), model["model_id"]
        assert required_serving_fields <= set(model["serving"]), model["model_id"]
        assert isinstance(model["serving"]["requirements"], dict)
        assert isinstance(model["known_gaps"], list)
        assert isinstance(model["role_bindings"].get("roles"), list)


def test_vision_descriptors_expose_projector_requirements() -> None:
    descriptors = _load_yaml(DESCRIPTOR_PATH)
    by_id = {model["model_id"]: model for model in descriptors["models"]}

    worker = by_id["qwen2.5-vl-7b-q4_k_m"]
    escalation = by_id["qwen3-vl-30b-a3b-q4_k_m"]

    assert worker["serving"]["requirements"]["mmproj_path"].endswith(
        "Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf"
    )
    assert escalation["serving"]["requirements"]["mmproj_path"].endswith(
        "Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"
    )


def test_shared_runtime_aliases_do_not_emit_role_server_conflicts() -> None:
    descriptors = _load_yaml(DESCRIPTOR_PATH)

    conflicts = [
        model
        for model in descriptors["models"]
        if any("server" in gap and "conflict" in gap for gap in model["known_gaps"])
    ]

    assert conflicts == []
    worker = next(
        model
        for model in descriptors["models"]
        if model["model_id"] == "gemma4-26b-a4b-q4_k_m"
    )
    assert {"worker_general", "worker_math", "toolrunner"} <= set(
        worker["role_bindings"]["roles"]
    )
    assert not any("ignored non-live role model metadata" in gap for gap in worker["known_gaps"])
    alias_overrides = worker["role_bindings"].get("alias_overrides") or []
    ignored_models = {override.get("ignored_model_id") for override in alias_overrides}
    assert "qwen2.5-math-7b-q4_k_m" in ignored_models
    assert "qwen3-coder-30b-a3b-q4_k_m" in ignored_models
