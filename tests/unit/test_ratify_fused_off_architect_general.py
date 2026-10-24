"""RATIFY-FUSED-OFF-ARCHGEN-20261024: GGML_FUSED_DECODE_OFF=1 is DECLARED for architect_general only.

The qwen4exp fused decode is default-ON in v10 code. Before this change nothing declared it
OFF for the :8074 Flash-Next process: stack_env's architect_general block was empty, the launcher
strips every undeclared GGML_* key, and the registry's `optimal_cpu_serving.env` is descriptive
(no code reads it). Every measured Q38FN number ran with OFF=1.

Guards:
  1. the block is exactly {GGML_FUSED_DECODE_OFF: "1"} and only architect_general (plus the
     unlaunched legacy enum alias architect_coding) resolves to it;
  2. the REAL launcher transform keeps it under a binary override and overrides an ambient value;
  3. live env attestation declares it for the :8074 process;
  4. the offline export (autokernel_enrollment) recomputes the launcher's own env transform and
     matches it for every launch-manifest role. Before this change the export still carried
     the pre-5fd6bbdb strip and would have hidden the declared key.
"""

from __future__ import annotations

import yaml
from pathlib import Path

import pytest

from scripts.server import autokernel_enrollment as enrollment
from scripts.server import orchestrator_stack as osk
from scripts.server import stack_env
from scripts.server.env_attestation import declared_env_for
from src.roles import Role

KEY, VALUE = "GGML_FUSED_DECODE_OFF", "1"
REPO = Path(__file__).resolve().parents[2]


def _known_role_names() -> set[str]:
    lm = yaml.safe_load((REPO / "orchestration/launch_manifest.yaml").read_text())
    reg = yaml.safe_load((REPO / "orchestration/model_registry.yaml").read_text())
    names = set(stack_env._ROLE_ENV_BLOCKS) | set(stack_env._STACK_ENV_CANONICAL_ALIASES)
    names |= set((lm.get("port_map") or {})) | set((lm.get("role_launch_meta") or {}))
    names |= set((reg.get("roles") or {})) | set((reg.get("server_mode") or {}))
    names |= {m.value for m in Role} | {m.name.lower() for m in Role}
    return names


def _launched_role_names() -> set[str]:
    lm = yaml.safe_load((REPO / "orchestration/launch_manifest.yaml").read_text())
    return set((lm.get("port_map") or {})) | set((lm.get("role_launch_meta") or {}))


def test_architect_general_block_is_exactly_fused_off():
    assert stack_env._ROLE_ENV_BLOCKS["architect_general"] == {KEY: VALUE}
    assert stack_env._role_env_overrides("architect_general") == {KEY: VALUE}


def test_only_architect_general_resolves_to_the_block():
    hits = {n for n in _known_role_names() if stack_env._role_env_overrides(n).get(KEY) is not None}
    # architect_coding is a legacy Role enum alias of architect_general (role removed 2026-05-06)
    assert hits <= {"architect_general", "architect_coding"}
    assert "architect_general" in hits
    assert "architect_coding" not in _launched_role_names(), "architect_coding is launched again: re-check"
    for role in ("architect_critic", "coder_escalation", "ingest_long_context", "frontdoor",
                 "worker_general", "worker_vision", "vision_escalation", "embedder", "worker_fast"):
        assert KEY not in stack_env._role_env_overrides(role), role


def test_real_launcher_transform_keeps_it_and_overrides_ambient(capsys):
    base = {"GGML_FUSED_DECODE_OFF": "0", "GGML_AMBIENT_LEAK": "1", "PATH": "/usr/bin"}
    env = stack_env.build_launch_env("architect_general", dict(base))
    osk._apply_runtime_requirements_env(env, binary_override="/k/bin/llama-server", ld_paths=["/k/lib"],
                                        preserve=stack_env._role_env_overrides("architect_general"))
    assert env[KEY] == VALUE
    assert "GGML_AMBIENT_LEAK" not in env
    assert env["GGML_IQK"] == "1"
    other = stack_env.build_launch_env("architect_critic", {"GGML_FUSED_DECODE_OFF": "1"})
    osk._apply_runtime_requirements_env(other, binary_override="/k/bin/llama-server", ld_paths=None,
                                        preserve=stack_env._role_env_overrides("architect_critic"))
    assert KEY not in other, "an ambient OFF must not reach a role that does not declare it"


def test_env_attestation_declares_it_for_the_8074_process():
    declared, is_llama = declared_env_for("architect_general", "architect_general",
                                          "/k/bin/llama-server --port 8074", {})
    assert is_llama and declared[KEY] == VALUE
    critic, _ = declared_env_for("architect_critic", "architect_critic", "/k/bin/llama-server", {})
    assert KEY not in critic


@pytest.mark.parametrize("binary_override", ["/k/bin/llama-server", None])
@pytest.mark.parametrize("ld_paths", [["/k/lib"], None])
def test_export_env_transform_recomputes_the_launcher(binary_override, ld_paths, capsys):
    """RESTATED DERIVATION guard: the export's copy must equal the launcher's own transform."""
    base = {"GGML_AMBIENT_LEAK": "1", "LD_LIBRARY_PATH": "/amb", "PATH": "/usr/bin"}
    for role in sorted(_launched_role_names() | {"architect_general", "worker"}):
        preserve = stack_env._role_env_overrides(role)
        real = stack_env.build_launch_env(role, dict(base))
        osk._apply_runtime_requirements_env(real, binary_override=binary_override, ld_paths=ld_paths,
                                            preserve=preserve)
        export = stack_env.build_launch_env(role, dict(base))
        enrollment._apply_declared_runtime_env(export, binary_override=binary_override,
                                               ld_paths=ld_paths, env_role=role)
        assert export == real, role
    capsys.readouterr()
