"""Tests for orchestrator stack launch-env composition."""

from __future__ import annotations

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


def test_worker_role_gets_ccd_stack() -> None:
    env = build_launch_env("worker", base_env={})
    assert env["GGML_CCD_POOLS"] == "1"
    assert env["GGML_CCD_WORK_DIST"] == "1"
    assert env["GGML_BARRIER_LOCAL_BETWEEN_OPS"] == "1"


def test_frontdoor_role_gets_no_ggml_env() -> None:
    env = build_launch_env("frontdoor", base_env={})
    for key in env:
        assert not key.startswith("GGML_"), f"frontdoor must not set {key}"


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


def test_worker_general_role_gets_ccd_stack_directly() -> None:
    env = build_launch_env("worker_general", base_env={})
    assert env["GGML_CCD_POOLS"] == "1"
    assert env["GGML_CCD_WORK_DIST"] == "1"
    assert env["GGML_BARRIER_LOCAL_BETWEEN_OPS"] == "1"


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
    first["GGML_CCD_POOLS"] = "MUTATED"
    second = _role_env_overrides("worker")
    assert second["GGML_CCD_POOLS"] == "1"


def test_architect_general_overrides_repack_interleave() -> None:
    env = build_launch_env("architect_general", base_env={})
    assert env["GGML_NUMA_REPACK_INTERLEAVE"] == "0"
