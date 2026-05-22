"""Tests for the per-mode helpers that back build_server_command()."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.server import orchestrator_stack as oss


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


def test_build_worker_explore_command_engages_mtp_path() -> None:
    cmd = oss._build_worker_explore_command(
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


def test_build_worker_explore_command_uses_binary_override_when_set() -> None:
    cmd = oss._build_worker_explore_command(
        port=8072, model_path="/m/gemma4.gguf",
        binary_override="/opt/ik_llama.cpp/build/bin/llama-server",
    )
    assert cmd[0] == "/opt/ik_llama.cpp/build/bin/llama-server"


def test_build_worker_explore_command_falls_back_to_llama_server() -> None:
    cmd = oss._build_worker_explore_command(
        port=8072, model_path="/m/gemma4.gguf", binary_override=None,
    )
    assert cmd[0] == str(oss.LLAMA_SERVER)


def test_build_worker_explore_command_uses_numa_thread_count_for_port() -> None:
    """Quarter instances must get the per-instance thread count from NUMA_CONFIG, not 96."""
    # Find a quarter port for worker_general (not the full instance)
    full_port, full_threads, *_ = (
        (inst[1], inst[2], None) for inst in oss.NUMA_CONFIG["worker_general"]["instances"]
        if inst[1] != 8072
    )
    quarter_port = full_port[0]  # generator returns first
    expected_threads = full_port[1]

    cmd = oss._build_worker_explore_command(
        port=quarter_port, model_path="/m/gemma4.gguf", binary_override=None,
    )
    assert cmd[cmd.index("-t") + 1] == str(expected_threads)


def test_build_worker_explore_command_unknown_port_uses_fallback_96() -> None:
    cmd = oss._build_worker_explore_command(
        port=9999, model_path="/m/gemma4.gguf", binary_override=None,
    )
    assert cmd[cmd.index("-t") + 1] == "96"


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


# -----------------------------------------------------------------------------
# Dispatcher routing
# -----------------------------------------------------------------------------


def test_dispatcher_routes_vision_mode() -> None:
    with patch.object(oss, "_build_vision_command", return_value=["VISION"]) as m:
        out = oss.build_server_command(None, 8087, vision_mode=True, vision_type="escalation")
    assert out == ["VISION"]
    m.assert_called_once_with(8087, "escalation")


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


def test_dispatcher_routes_worker_explore_with_binary_override() -> None:
    with patch.object(oss, "_build_worker_explore_command", return_value=["GEMMA"]) as m:
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
    m.assert_called_once_with(fake_role, 8070)
