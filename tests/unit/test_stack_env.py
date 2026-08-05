"""Tests for orchestrator stack launch-env composition."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.server.stack_env import (
    _CANONICAL_OMP_ENV,
    _LLVM20_LIBDIR,
    _ROLE_ENV_BLOCKS,
    _role_env_overrides,
    build_launch_env,
)


def test_canonical_omp_env_applied_to_every_role() -> None:
    env = build_launch_env("frontdoor", base_env={})
    for key, value in _CANONICAL_OMP_ENV.items():
        assert env[key] == value


def test_llvm20_libdir_prepended_when_missing() -> None:
    env = build_launch_env("worker", base_env={"LD_LIBRARY_PATH": "/opt/foo/lib"})
    assert env["LD_LIBRARY_PATH"] == f"{_LLVM20_LIBDIR}:/opt/foo/lib"


def test_llvm20_libdir_set_when_ld_library_path_unset() -> None:
    env = build_launch_env("worker", base_env={})
    assert env["LD_LIBRARY_PATH"] == _LLVM20_LIBDIR


def test_llvm20_libdir_idempotent_when_already_present() -> None:
    initial = f"{_LLVM20_LIBDIR}:/opt/foo/lib"
    env = build_launch_env("worker", base_env={"LD_LIBRARY_PATH": initial})
    assert env["LD_LIBRARY_PATH"] == initial


def test_worker_role_gets_v6_iqk_without_vestigial_ccd_stack() -> None:
    env = build_launch_env("worker", base_env={})
    assert env["GGML_IQK"] == "1"
    assert "GGML_CCD_POOLS" not in env
    assert "GGML_CCD_WORK_DIST" not in env
    assert "GGML_BARRIER_LOCAL_BETWEEN_OPS" not in env


def test_frontdoor_role_gets_no_ggml_env() -> None:
    env = build_launch_env("frontdoor", base_env={})
    for key in env:
        if key == "GGML_IQK":
            continue
        assert not key.startswith("GGML_"), f"frontdoor must not set {key}"


def test_explicit_iqk_override_survives_canonical_default() -> None:
    env = build_launch_env("worker_general", base_env={"GGML_IQK": "0"})
    assert env["GGML_IQK"] == "0"


def test_arch_alias_fallthrough_worker_summarize_inherits_frontdoor() -> None:
    direct = _role_env_overrides("frontdoor")
    aliased = _role_env_overrides("worker_summarize")
    assert direct == aliased == {}


def test_arch_alias_fallthrough_toolrunner_inherits_worker() -> None:
    aliased = _role_env_overrides("toolrunner")
    expected = _ROLE_ENV_BLOCKS["worker_general"]
    assert aliased == expected


def test_worker_explore_alias_inherits_worker() -> None:
    aliased = _role_env_overrides("worker_explore")
    expected = _ROLE_ENV_BLOCKS["worker_general"]
    assert aliased == expected


def test_worker_general_role_gets_v6_iqk_directly() -> None:
    env = build_launch_env("worker_general", base_env={})
    assert env["GGML_IQK"] == "1"
    assert "GGML_CCD_POOLS" not in env
    assert "GGML_CCD_WORK_DIST" not in env
    assert "GGML_BARRIER_LOCAL_BETWEEN_OPS" not in env


def test_unknown_role_returns_empty_overrides() -> None:
    assert _role_env_overrides("nonexistent_role_xyz") == {}


def test_unknown_role_still_gets_canonical_omp_and_libdir() -> None:
    env = build_launch_env("nonexistent_role_xyz", base_env={})
    for key, value in _CANONICAL_OMP_ENV.items():
        assert env[key] == value
    assert env["LD_LIBRARY_PATH"] == _LLVM20_LIBDIR


def test_base_env_preserved() -> None:
    env = build_launch_env("worker", base_env={"USER": "tester", "HOME": "/home/tester"})
    assert env["USER"] == "tester"
    assert env["HOME"] == "/home/tester"


def test_role_overrides_returns_independent_dict_copies() -> None:
    first = _role_env_overrides("worker")
    first["MUTATED_KEY"] = "MUTATED"
    second = _role_env_overrides("worker")
    assert "MUTATED_KEY" not in second


def test_architect_general_overrides_repack_interleave() -> None:
    """The 122B's Probe-B repack override follows the MODEL, not the role name.

    Retargeted 2026-08-04. `GGML_NUMA_REPACK_INTERLEAVE=0` is the Qwen3.5-122B-A10B
    Probe-B tuning (c2, +1.28%, σ ~0.4%, z ~3; bundle
    data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe/). The 2026-07-31 W1
    cutover moved that GGUF from architect_general to architect_critic, and
    stack_env moved the block with it — architect_general is now Qwen3.6-27B
    dense Q8 on MI210 (ROCm0), where a CPU NUMA repack setting is at best inert
    and at worst misleading provenance. Nothing was dropped, so nothing is
    deleted here: the assertion is retargeted to the role that serves the model,
    and the complementary guard below is ADDED so the setting cannot silently
    come back on a ROCm process.
    """
    registry = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml")
        .read_text()
    )["server_mode"]

    # The premise is read from master, not asserted here: whichever role serves
    # the 122B on the CPU is the one that must carry the override.
    assert "Qwen3.5-122B" in str(registry["architect_critic"]["model"])
    assert registry["architect_critic"].get("device") in (None, "cpu", "CPU")
    env = build_launch_env("architect_critic", base_env={})
    assert env["GGML_NUMA_REPACK_INTERLEAVE"] == "0"

    # ...and a ROCm role must not inherit a CPU NUMA repack setting.
    assert registry["architect_general"]["device"] == "ROCm0"
    gpu_env = build_launch_env("architect_general", base_env={})
    assert "GGML_NUMA_REPACK_INTERLEAVE" not in gpu_env
