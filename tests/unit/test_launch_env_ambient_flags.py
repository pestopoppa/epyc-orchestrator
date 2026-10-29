"""Managed launch recipes reject undeclared parent process controls."""

from types import SimpleNamespace

import pytest

from scripts.server.stack_env import build_launch_env, build_service_env


@pytest.mark.parametrize("role", ["frontdoor", "worker", "architect_critic", "embedder"])
def test_parent_llama_defaults_do_not_change_declared_launch(role):
    ambient = {
        "PATH": "/usr/bin",
        "LLAMA_ARG_N_PARALLEL": "64",
        "LLAMA_ARG_KV_UNIFIED": "0",
        "LLAMA_ARG_FUTURE_FLAG": "unexpected",
    }
    env = build_launch_env(role, ambient)
    assert not any(key.startswith("LLAMA_ARG_") for key in env)
    assert env["PATH"] == ambient["PATH"]
    assert ambient["LLAMA_ARG_N_PARALLEL"] == "64"


@pytest.mark.parametrize(
    "declared", [{}, {"LD_PRELOAD": "/declared/shim.so"}, {"SHIM_NPROCS": "32"}]
)
def test_aux_only_receives_declared_preload_and_thread_shim(declared):
    ambient = {
        "PATH": "/usr/bin",
        "LLAMA_ARG_N_PARALLEL": "64",
        "LD_PRELOAD": "/ambient/shim.so",
        "SHIM_NPROCS": "192",
    }
    spec = SimpleNamespace(env=declared, ld_library_path=None)
    env = build_service_env(spec, ambient)
    assert env.get("LD_PRELOAD") == declared.get("LD_PRELOAD")
    assert env.get("SHIM_NPROCS") == declared.get("SHIM_NPROCS")
    assert "LLAMA_ARG_N_PARALLEL" not in env
    assert env["PATH"] == ambient["PATH"]
    assert ambient["LD_PRELOAD"] == "/ambient/shim.so"
