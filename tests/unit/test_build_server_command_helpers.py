"""Tests for the per-mode helpers that back build_server_command()."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from scripts.server import orchestrator_stack as oss
from scripts.server import stack_commands


def _stack_prior_role(role: str) -> dict[str, Any]:
    priors_path = oss._PATHS["project_root"] / "orchestration/derived/stack_priors.yaml"
    payload = yaml.safe_load(priors_path.read_text(encoding="utf-8"))
    role_record = payload["roles"][role]
    assert isinstance(role_record, dict)
    return role_record


def _write_launch_prior(
    tmp_path: Path,
    role: str,
    *,
    requirements: dict[str, Any],
    runtime: dict[str, Any],
) -> Path:
    path = tmp_path / "stack_priors.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "roles": {
                    role: {
                        "deployment_status": "live_stack",
                        "serving": {
                            "launch": {
                                "requirements": requirements,
                                "runtime": runtime,
                            }
                        },
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _flag_value(cmd: list[str], flag: str) -> str | None:
    if flag not in cmd:
        return None
    idx = cmd.index(flag)
    if idx + 1 >= len(cmd):
        return None
    return cmd[idx + 1]


def _all_flag_values(cmd: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for idx, value in enumerate(cmd):
        if value == flag and idx + 1 < len(cmd):
            values.append(cmd[idx + 1])
    return values


def _optional_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: str | None) -> float | None:
    return float(value) if value is not None else None


def _command_runtime_signature(cmd: list[str]) -> dict[str, Any]:
    spec_enabled = "-md" in cmd and "--spec-type" in cmd
    return {
        "binary_path": cmd[0],
        "cache": {
            "context_tokens": _optional_int(_flag_value(cmd, "-c")),
            "slots": _optional_int(_flag_value(cmd, "-np")),
            "ubatch": _optional_int(_flag_value(cmd, "-ub")),
            "kv_type_k": _flag_value(cmd, "-ctk"),
            "kv_type_v": _flag_value(cmd, "-ctv"),
            "no_mmap": "--no-mmap" in cmd,
            "mlock": "--mlock" in cmd,
            "slot_save_path": _flag_value(cmd, "--slot-save-path"),
        },
        "flags": {
            "flash_attn": _flag_value(cmd, "--flash-attn") == "on",
            "jinja": "--jinja" in cmd,
            "reasoning": _flag_value(cmd, "--reasoning"),
            "override_kv": sorted(_all_flag_values(cmd, "--override-kv")),
            "spec": {
                "enabled": spec_enabled,
                "type": _flag_value(cmd, "--spec-type") if spec_enabled else None,
                "draft_model_path": _flag_value(cmd, "-md") if spec_enabled else None,
                "draft_max": (
                    _optional_int(_flag_value(cmd, "--draft-max")) if spec_enabled else None
                ),
                "draft_p_min": (
                    _optional_float(_flag_value(cmd, "--draft-p-min"))
                    if spec_enabled
                    else None
                ),
                "threads_draft": (
                    _optional_int(_flag_value(cmd, "--threads-draft"))
                    if spec_enabled
                    else None
                ),
            },
        },
    }


def _stack_prior_runtime_signature(runtime: dict[str, Any]) -> dict[str, Any]:
    cache = runtime["cache"]
    flags = runtime["flags"]
    spec = flags["spec"]
    return {
        "binary_path": runtime["binary_path"],
        "cache": {
            "context_tokens": cache["context_tokens"],
            "slots": cache["slots"],
            "ubatch": cache["ubatch"],
            "kv_type_k": cache["kv_type_k"],
            "kv_type_v": cache["kv_type_v"],
            "no_mmap": cache["no_mmap"],
            "mlock": cache["mlock"],
            "slot_save_path": cache["slot_save_path"],
        },
        "flags": {
            "flash_attn": flags["flash_attn"],
            "jinja": flags["jinja"],
            "reasoning": flags["reasoning"],
            "override_kv": sorted(flags["override_kv"]),
            "spec": {
                "enabled": spec["enabled"],
                "type": spec["type"],
                "draft_model_path": spec["draft_model_path"],
                "draft_max": spec["draft_max"],
                "draft_p_min": spec["draft_p_min"],
                "threads_draft": spec["threads_draft"],
            },
        },
    }


def _assert_detached_popen(popen) -> None:
    kwargs = popen.call_args.kwargs
    assert kwargs["stdin"] is oss.subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True


def test_descriptor_active_roles_are_canonical_launch_roles() -> None:
    active_roles = stack_commands._descriptor_active_roles()

    assert "worker_general" in active_roles
    assert "architect_general" in active_roles
    assert "worker_explore" not in active_roles
    assert "architect_coding" not in active_roles


# -----------------------------------------------------------------------------
# Vision / embedding / dev / worker-mode shape regressions
# -----------------------------------------------------------------------------


def test_build_vision_command_escalation_emits_expert_reduction() -> None:
    cmd = oss._build_vision_command(port=8087, vision_type="escalation")
    assert "--mmproj" in cmd
    assert "--override-kv" in cmd
    assert "qwen3vlmoe.expert_used_count=int:4" in cmd
    assert cmd[cmd.index("-c") + 1] == "16384"
    assert cmd[cmd.index("-t") + 1] == "96"


def test_build_vision_command_worker_uses_small_model() -> None:
    cmd = oss._build_vision_command(port=8086, vision_type="worker")
    assert oss.VISION_WORKER_MODEL in cmd
    assert oss.VISION_WORKER_MMPROJ in cmd
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert cmd[cmd.index("-t") + 1] == "24"


@pytest.mark.parametrize(
    ("role", "port", "vision_type"),
    [
        ("worker_vision", 8086, "worker"),
        ("vision_escalation", 8087, "escalation"),
    ],
)
def test_build_vision_command_matches_stack_prior_launch_witness(
    role: str,
    port: int,
    vision_type: str,
) -> None:
    role_record = _stack_prior_role(role)
    launch = role_record["serving"]["launch"]
    requirements = launch["requirements"]
    runtime = launch["runtime"]

    cmd = oss._build_vision_command(port=port, vision_type=vision_type)

    assert _flag_value(cmd, "-m") == requirements["model_path"]
    assert _flag_value(cmd, "--mmproj") == requirements["mmproj_path"]
    assert _command_runtime_signature(cmd) == _stack_prior_runtime_signature(runtime)


def test_build_vision_command_prefers_stack_prior_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    priors = _write_launch_prior(
        tmp_path,
        "vision_escalation",
        requirements={
            "model_path": "/prior/vision.gguf",
            "mmproj_path": "/prior/mmproj.gguf",
        },
        runtime={
            "binary_path": "/prior/llama-server",
            "cache": {"context_tokens": 12000, "slots": 1},
            "flags": {
                "flash_attn": False,
                "override_kv": ["qwen3vlmoe.expert_used_count=int:2"],
                "spec": {"enabled": False},
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)

    cmd = oss._build_vision_command(port=9087, vision_type="escalation")

    assert cmd[0] == "/prior/llama-server"
    assert _flag_value(cmd, "-m") == "/prior/vision.gguf"
    assert _flag_value(cmd, "--mmproj") == "/prior/mmproj.gguf"
    assert _all_flag_values(cmd, "--override-kv") == [
        "qwen3vlmoe.expert_used_count=int:2"
    ]
    assert _flag_value(cmd, "-np") == "1"
    assert _flag_value(cmd, "-c") == "12000"
    assert "--flash-attn" not in cmd


def test_build_embedding_command_enables_embeddings_and_cls_pool() -> None:
    cmd = oss._build_embedding_command(port=8090)
    assert "--embeddings" in cmd
    assert "--pooling" in cmd
    assert cmd[cmd.index("--pooling") + 1] == "cls"
    assert cmd[cmd.index("-np") + 1] == "4"
    assert cmd[cmd.index("-t") + 1] == "4"


def test_build_dev_command_short_context_small_threads() -> None:
    cmd = oss._build_dev_command(port=9999)
    assert cmd[cmd.index("-c") + 1] == "4096"
    assert cmd[cmd.index("-t") + 1] == "16"
    assert "--flash-attn" in cmd


def test_build_worker_fast_command_uses_4_slots() -> None:
    cmd = oss._build_worker_fast_command(port=8102, model_path="/m/fast.gguf")
    assert cmd[cmd.index("-np") + 1] == "4"
    assert cmd[cmd.index("-m") + 1] == "/m/fast.gguf"
    assert cmd[cmd.index("-c") + 1] == "16384"


def test_build_worker_general_command_engages_mtp_path() -> None:
    cmd = oss._build_worker_general_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )
    # MTP-specific flags must all be present
    assert cmd[cmd.index("--spec-type") + 1] == "mtp"
    assert cmd[cmd.index("--draft-max") + 1] == "2"
    assert cmd[cmd.index("--draft-p-min") + 1] == "0.0"
    assert cmd[cmd.index("--threads-draft") + 1] == "16"
    assert cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("-ctv") + 1] == "q8_0"
    assert "--no-mmap" in cmd
    assert "--jinja" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"


def test_build_worker_general_command_matches_stack_prior_launch_witness() -> None:
    role_record = _stack_prior_role("worker_general")
    launch = role_record["serving"]["launch"]
    requirements = launch["requirements"]
    runtime = launch["runtime"]

    cmd = oss._build_worker_general_command(
        port=8072,
        model_path=requirements["model_path"],
        binary_override=runtime["binary_path"],
    )

    assert _flag_value(cmd, "-m") == requirements["model_path"]
    assert _flag_value(cmd, "-md") == requirements["draft_model_path"]
    assert _command_runtime_signature(cmd) == _stack_prior_runtime_signature(runtime)


def test_build_worker_general_command_prefers_stack_prior_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    priors = _write_launch_prior(
        tmp_path,
        "worker_general",
        requirements={
            "model_path": "/prior/gemma.gguf",
            "draft_model_path": "/prior/draft.gguf",
        },
        runtime={
            "binary_path": "/prior/ik/llama-server",
            "cache": {
                "context_tokens": 12288,
                "slots": 1,
                "ubatch": 256,
                "kv_type_k": "q5_0",
                "kv_type_v": "q6_0",
                "no_mmap": False,
            },
            "flags": {
                "flash_attn": False,
                "jinja": False,
                "reasoning": "off",
                "override_kv": [],
                "spec": {
                    "enabled": True,
                    "type": "mtp",
                    "draft_model_path": "/prior/draft.gguf",
                    "draft_max": 4,
                    "draft_p_min": 0.25,
                    "threads_draft": 8,
                },
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)

    cmd = oss._build_worker_general_command(
        port=8072,
        model_path="/fallback/gemma.gguf",
        binary_override=None,
    )

    assert cmd[0] == "/prior/ik/llama-server"
    assert _flag_value(cmd, "-m") == "/prior/gemma.gguf"
    assert _flag_value(cmd, "-md") == "/prior/draft.gguf"
    assert _flag_value(cmd, "--draft-max") == "4"
    assert _flag_value(cmd, "--draft-p-min") == "0.25"
    assert _flag_value(cmd, "--threads-draft") == "8"
    assert _flag_value(cmd, "-ub") == "256"
    assert _flag_value(cmd, "-c") == "12288"
    assert _flag_value(cmd, "-ctk") == "q5_0"
    assert _flag_value(cmd, "-ctv") == "q6_0"
    assert "--no-mmap" not in cmd
    assert "--jinja" not in cmd
    assert "--flash-attn" not in cmd


def test_build_worker_general_command_rejects_boolean_runtime_numbers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    priors = _write_launch_prior(
        tmp_path,
        "worker_general",
        requirements={
            "model_path": "/prior/gemma.gguf",
            "draft_model_path": "/prior/draft.gguf",
        },
        runtime={
            "binary_path": "/prior/ik/llama-server",
            "cache": {
                "context_tokens": True,
                "slots": True,
                "ubatch": True,
                "kv_type_k": "q8_0",
                "kv_type_v": "q8_0",
            },
            "flags": {
                "flash_attn": True,
                "jinja": True,
                "reasoning": "off",
                "spec": {
                    "enabled": True,
                    "type": "mtp",
                    "draft_model_path": "/prior/draft.gguf",
                    "draft_max": True,
                    "draft_p_min": True,
                    "threads_draft": True,
                },
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)

    cmd = oss._build_worker_general_command(
        port=8072,
        model_path="/fallback/gemma.gguf",
        binary_override=None,
    )

    assert _flag_value(cmd, "-np") == "1"
    assert _flag_value(cmd, "-c") == str(oss.LAUNCH_CONTEXT_TOKENS["worker_general"])
    assert _flag_value(cmd, "-ub") == str(oss.WORKER_MTP_UBATCH_TOKENS)
    assert _flag_value(cmd, "--draft-max") == str(oss.WORKER_MTP_DRAFT_MAX)
    assert _flag_value(cmd, "--draft-p-min") == str(oss.WORKER_MTP_DRAFT_P_MIN)
    assert _flag_value(cmd, "--threads-draft") == str(oss.WORKER_MTP_THREADS_DRAFT)


def test_build_worker_general_command_uses_binary_override_when_set() -> None:
    cmd = oss._build_worker_general_command(
        port=8072, model_path="/m/gemma4.gguf",
        binary_override="/opt/ik_llama.cpp/build/bin/llama-server",
    )
    assert cmd[0] == "/opt/ik_llama.cpp/build/bin/llama-server"


def test_build_worker_general_command_prefers_live_stack_prior_binary() -> None:
    cmd = oss._build_worker_general_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )
    runtime = _stack_prior_role("worker_general")["serving"]["launch"]["runtime"]
    assert cmd[0] == runtime["binary_path"]


def test_build_worker_general_command_falls_back_to_llama_server_without_priors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", tmp_path / "missing.yaml")

    cmd = oss._build_worker_general_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )

    assert cmd[0] == str(oss.LLAMA_SERVER)


def test_build_worker_general_command_uses_numa_thread_count_for_port() -> None:
    """Quarter instances must get the per-instance thread count from NUMA_CONFIG, not 96.

    Post-da1aed6 the thread count is resolved by ``numa_instance`` *index* (not by
    port): the start loop / dispatcher pass the index of the instance being
    launched, and _resolve_thread_count picks instances[numa_instance].threads.
    This is what makes gemma4 quarters get -t 48 (idx 1..4) while the full
    instance (idx 0) gets -t 96.
    """
    instances = oss.NUMA_CONFIG["worker_general"]["instances"]
    for idx, inst in enumerate(instances):
        port, expected_threads = inst[1], inst[2]
        cmd = oss._build_worker_general_command(
            port=port, model_path="/m/gemma4.gguf", binary_override=None,
            numa_instance=idx,
        )
        assert cmd[cmd.index("-t") + 1] == str(expected_threads), (
            f"instance {idx} (port {port}) expected -t {expected_threads}"
        )
    # Full instance (idx 0) is -t 96 — the over-subscription the launcher bug
    # wrongly applied to quarters too (now fixed by forwarding numa_instance).
    assert instances[0][2] == 96


def test_build_worker_general_command_unknown_port_uses_fallback_96() -> None:
    cmd = oss._build_worker_general_command(
        port=9999, model_path="/m/gemma4.gguf", binary_override=None,
    )
    assert cmd[cmd.index("-t") + 1] == "96"


def test_build_worker_explore_command_keeps_compatibility_wrapper() -> None:
    cmd = oss._build_worker_explore_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )
    assert cmd[cmd.index("--spec-type") + 1] == "mtp"


# -----------------------------------------------------------------------------
# Default-role builder sub-helpers
# -----------------------------------------------------------------------------


def test_resolve_thread_count_from_numa_config() -> None:
    # frontdoor[0] is NUMA_NODE0 = 96 threads
    assert oss._resolve_thread_count("frontdoor") == "96"


def test_resolve_thread_count_fallback_for_unknown_role() -> None:
    assert oss._resolve_thread_count("nonexistent_role") == "96"


def test_resolve_binary_for_role_defaults_to_llama_server() -> None:
    # _V2_ROLES is currently empty, so every role gets LLAMA_SERVER
    assert oss._resolve_binary_for_role("frontdoor") == oss.LLAMA_SERVER
    assert oss._resolve_binary_for_role("architect_general") == oss.LLAMA_SERVER


def test_append_kv_quant_args_emits_q8_for_frontdoor() -> None:
    cmd: list[str] = []
    oss._append_kv_quant_args(cmd, "frontdoor")
    assert cmd == ["-ctk", "q8_0", "-ctv", "q8_0"]


def test_append_kv_quant_args_emits_q4_f16_for_architect_general() -> None:
    cmd: list[str] = []
    oss._append_kv_quant_args(cmd, "architect_general")
    assert cmd == ["-ctk", "q4_0", "-ctv", "f16"]


def test_append_kv_quant_args_noop_for_role_without_config() -> None:
    cmd = ["pre-existing"]
    oss._append_kv_quant_args(cmd, "worker_vision")  # not in _KV_QUANT_CONFIGS
    assert cmd == ["pre-existing"]


def test_apply_numa_spec_overrides_rewrites_draft_max() -> None:
    cmd = ["--draft-max", "16", "--other", "thing"]
    numa_cfg = {"spec_overrides": {"draft_max": 2}}
    oss._apply_numa_spec_overrides(cmd, numa_cfg)
    assert cmd == ["--draft-max", "2", "--other", "thing"]


def test_apply_numa_spec_overrides_noop_without_overrides() -> None:
    cmd = ["--draft-max", "16"]
    oss._apply_numa_spec_overrides(cmd, {"instances": []})
    assert cmd == ["--draft-max", "16"]


def test_apply_numa_spec_overrides_noop_when_numa_cfg_none() -> None:
    cmd = ["--draft-max", "16"]
    oss._apply_numa_spec_overrides(cmd, None)
    assert cmd == ["--draft-max", "16"]


def test_append_acceleration_args_moe_expert_reduction() -> None:
    accel = SimpleNamespace(
        type="moe_expert_reduction",
        experts=4,
        override_key="qwen3vlmoe.expert_used_count",
        draft_role=None,
    )
    cmd: list[str] = []
    oss._append_acceleration_args(cmd, "vision_escalation", accel, "/m/x.gguf")
    assert cmd == ["--override-kv", "qwen3vlmoe.expert_used_count=int:4"]


def test_append_acceleration_args_skips_spec_decode_for_architect_general() -> None:
    """architect_general is gated out of speculative_decoding per _NO_SPEC_DECODE."""
    accel = SimpleNamespace(
        type="speculative_decoding",
        draft_role="some_drafter",
        k=12,
        experts=None,
        n_layer_exit_draft=None,
    )
    cmd: list[str] = []
    oss._append_acceleration_args(cmd, "architect_general", accel, "/m/x.gguf")
    assert cmd == []


def test_append_acceleration_args_self_speculation_emits_md_and_n_layer() -> None:
    accel = SimpleNamespace(
        type="self_speculation",
        n_layer_exit_draft=4,
        k=8,
        draft_role=None,
        experts=None,
        n_layer_exit_intermediate=None,
    )
    cmd: list[str] = []
    oss._append_acceleration_args(cmd, "some_role", accel, "/m/target.gguf")
    assert "-md" in cmd
    assert cmd[cmd.index("-md") + 1] == "/m/target.gguf"
    assert cmd[cmd.index("--n-layer-exit-draft") + 1] == "4"
    assert cmd[cmd.index("--draft-max") + 1] == "8"


def test_append_acceleration_args_hierarchical_speculation_with_intermediate() -> None:
    accel = SimpleNamespace(
        type="hierarchical_speculation",
        n_layer_exit_draft=3,
        n_layer_exit_intermediate=7,
        k=12,
        draft_role=None,
        experts=None,
    )
    cmd: list[str] = []
    oss._append_acceleration_args(cmd, "some_role", accel, "/m/x.gguf")
    assert "--hierarchical-spec" in cmd
    assert cmd[cmd.index("--n-layer-exit-intermediate") + 1] == "7"


def test_build_role_command_prefers_stack_prior_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    slot_dir = tmp_path / "slot-cache" / "frontdoor"
    priors = _write_launch_prior(
        tmp_path,
        "frontdoor",
        requirements={},
        runtime={
            "binary_path": "/prior/llama-server",
            "cache": {
                "context_tokens": 24576,
                "slots": 1,
                "ubatch": 2048,
                "kv_type_k": "q5_0",
                "kv_type_v": "q6_0",
                "mlock": False,
                "slot_save_path": str(slot_dir),
            },
            "flags": {
                "flash_attn": False,
                "jinja": False,
                "reasoning": None,
                "override_kv": ["qwen36.expert_used_count=int:3"],
                "spec": {"enabled": False},
            },
        },
    )
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", priors)
    role = SimpleNamespace(
        name="frontdoor",
        model=SimpleNamespace(full_path="/fallback/frontdoor.gguf"),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )

    cmd = oss._build_role_command(role, port=9070)

    assert cmd[0] == "/prior/llama-server"
    assert _flag_value(cmd, "-m") == "/fallback/frontdoor.gguf"
    assert _flag_value(cmd, "-np") == "1"
    assert _flag_value(cmd, "-c") == "24576"
    assert _flag_value(cmd, "-ub") == "2048"
    assert _flag_value(cmd, "-ctk") == "q5_0"
    assert _flag_value(cmd, "-ctv") == "q6_0"
    assert _all_flag_values(cmd, "--override-kv") == [
        "qwen36.expert_used_count=int:3"
    ]
    assert _flag_value(cmd, "--slot-save-path") == str(slot_dir)
    assert "--mlock" not in cmd
    assert "--jinja" not in cmd
    assert "--flash-attn" not in cmd


# -----------------------------------------------------------------------------
# Dispatcher routing
# -----------------------------------------------------------------------------


def test_dispatcher_routes_vision_mode() -> None:
    with patch.object(oss, "_build_vision_command", return_value=["VISION"]) as m:
        out = oss.build_server_command(None, 8087, vision_mode=True, vision_type="escalation")
    assert out == ["VISION"]
    # numa_instance defaults to 0 (full) and is forwarded post-da1aed6 so quarters
    # get NUMA_CONFIG -t (was always -t 96).
    m.assert_called_once_with(8087, "escalation", 0)


def test_dispatcher_routes_embedding_mode() -> None:
    with patch.object(oss, "_build_embedding_command", return_value=["EMB"]) as m:
        out = oss.build_server_command(None, 8090, embedding_mode=True)
    assert out == ["EMB"]
    m.assert_called_once_with(8090)


def test_dispatcher_routes_worker_fast() -> None:
    with patch.object(oss, "_build_worker_fast_command", return_value=["FAST"]) as m:
        out = oss.build_server_command(
            None, 8102, worker_pool_mode=True, worker_type="fast",
        )
    assert out == ["FAST"]
    m.assert_called_once()
    assert m.call_args.args[0] == 8102


def test_dispatcher_routes_worker_general_with_binary_override() -> None:
    with patch.object(oss, "_build_worker_general_command", return_value=["GEMMA"]) as m:
        out = oss.build_server_command(
            None, 8072, worker_pool_mode=True, worker_type="explore",
            binary_override="/opt/ik/llama-server",
        )
    assert out == ["GEMMA"]
    m.assert_called_once()
    # signature: (port, model_path, binary_override)
    assert m.call_args.args[0] == 8072
    assert m.call_args.args[2] == "/opt/ik/llama-server"


def test_dispatcher_raises_on_unknown_worker_type() -> None:
    with pytest.raises(ValueError, match="Unknown worker type"):
        oss.build_server_command(
            None, 8000, worker_pool_mode=True, worker_type="ghost",
        )


def test_dispatcher_routes_dev_mode() -> None:
    with patch.object(oss, "_build_dev_command", return_value=["DEV"]) as m:
        out = oss.build_server_command(None, 9999, dev_mode=True)
    assert out == ["DEV"]
    m.assert_called_once_with(9999)


def test_dispatcher_routes_default_to_role_builder() -> None:
    fake_role = SimpleNamespace(name="frontdoor")
    with patch.object(oss, "_build_role_command", return_value=["ROLE"]) as m:
        out = oss.build_server_command(fake_role, 8070)
    assert out == ["ROLE"]
    # numa_instance (default 0) forwarded post-da1aed6.
    m.assert_called_once_with(fake_role, 8070, 0)


def test_start_server_vision_forwards_numa_instance_to_prefix(tmp_path, monkeypatch) -> None:
    """Vision quarter launches must inherit their NUMA_CONFIG CPU mask, not idx0."""
    calls: list[tuple[str, int]] = []

    def fake_prefix(role: str, instance_idx: int = 0) -> list[str]:
        calls.append((role, instance_idx))
        return ["taskset", "-c", f"{role}:{instance_idx}"]

    fake_proc = SimpleNamespace(pid=4242)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "_write_llama_marker", lambda *a, **kw: None)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(oss, "build_launch_env", lambda *a, **kw: {})
    with (
        patch.object(oss, "_numa_prefix", side_effect=fake_prefix),
        patch.object(oss, "build_server_command", return_value=["llama-server"]),
        patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen,
    ):
        info = oss.start_server(
            port=8187,
            roles=["vision_escalation"],
            registry=SimpleNamespace(),
            vision_mode=True,
            vision_type="escalation",
            numa_instance=1,
        )

    assert info is not None
    assert calls == [("vision_escalation", 1)]
    assert popen.call_args.args[0][:3] == ["taskset", "-c", "vision_escalation:1"]
    _assert_detached_popen(popen)


def test_start_server_worker_pool_forwards_numa_instance_to_prefix(tmp_path, monkeypatch) -> None:
    """Worker quarter launches must use the quarter CPU mask as well as -t 48."""
    calls: list[tuple[str, int]] = []

    def fake_prefix(role: str, instance_idx: int = 0) -> list[str]:
        calls.append((role, instance_idx))
        return ["taskset", "-c", f"{role}:{instance_idx}"]

    fake_proc = SimpleNamespace(pid=4343)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "_write_llama_marker", lambda *a, **kw: None)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(oss, "build_launch_env", lambda *a, **kw: {})
    monkeypatch.setattr(oss, "_runtime_requirements_for_role", lambda *a, **kw: (None, None))
    with (
        patch.object(oss, "_numa_prefix", side_effect=fake_prefix),
        patch.object(oss, "build_server_command", return_value=["llama-server"]),
        patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen,
    ):
        info = oss.start_server(
            port=8282,
            roles=["worker_general"],
            registry=SimpleNamespace(),
            worker_pool_mode=True,
            worker_type="explore",
            numa_instance=3,
        )

    assert info is not None
    assert calls == [("worker_general", 3)]
    assert popen.call_args.args[0][:3] == ["taskset", "-c", "worker_general:3"]
    _assert_detached_popen(popen)


def test_start_server_default_detaches_child_stdio(tmp_path, monkeypatch) -> None:
    """Stack-managed llama-server children must outlive non-interactive launchers."""
    fake_proc = SimpleNamespace(pid=4444)
    fake_role = SimpleNamespace(
        model=SimpleNamespace(name="frontdoor-model", full_path="/m/frontdoor.gguf"),
    )
    registry = SimpleNamespace(get_role=lambda _role: fake_role)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "_write_llama_marker", lambda *a, **kw: None)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(oss, "build_launch_env", lambda *a, **kw: {})
    with (
        patch.object(oss, "_numa_prefix", return_value=["taskset", "-c", "frontdoor:0"]),
        patch.object(oss, "build_server_command", return_value=["llama-server"]),
        patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen,
    ):
        info = oss.start_server(
            port=8070,
            roles=["frontdoor"],
            registry=registry,
        )

    assert info is not None
    assert popen.call_args.args[0][:3] == ["taskset", "-c", "frontdoor:0"]
    _assert_detached_popen(popen)


def test_start_document_formalizer_detaches_child_stdio(tmp_path, monkeypatch) -> None:
    fake_proc = SimpleNamespace(pid=4545)
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path)
    monkeypatch.setattr(oss, "wait_for_health", lambda *a, **kw: True)
    with patch.object(oss.subprocess, "Popen", return_value=fake_proc) as popen:
        info = oss.start_document_formalizer()

    assert info is not None
    _assert_detached_popen(popen)
