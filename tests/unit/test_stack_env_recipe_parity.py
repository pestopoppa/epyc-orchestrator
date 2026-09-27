"""The critic's launch env must equal what its registry recipe declares as serving env.

`server_mode.architect_critic.recipe.env` in the master registry was declared and never
read (SSU-F11). This test composes the env the way start_server does, including the
ambient strip, and diffs it against the compiled lean registry's declaration in both
directions for the knobs the recipe governs.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.server import orchestrator_stack as launcher
from scripts.server.stack_env import _role_env_overrides, build_launch_env

LEAN = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"


def _launch_env(role: str) -> dict[str, str]:
    env = build_launch_env(role, {})
    binary_override, ld_paths = launcher._stack_prior_runtime_overrides(role)
    launcher._apply_runtime_requirements_env(
        env, binary_override=binary_override, ld_paths=ld_paths, preserve=_role_env_overrides(role)
    )
    return env


def test_critic_launch_env_matches_the_registry_recipe_env() -> None:
    recipe = yaml.safe_load(LEAN.read_text())["server_mode"]["architect_critic"]["recipe"]
    declared = {k: str(v) for k, v in (recipe.get("env") or {}).items()}
    not_serving = set(recipe.get("env_not_serving") or {})
    emitted = _launch_env("architect_critic")
    missing = {k: v for k, v in declared.items() if emitted.get(k) != v}
    assert not missing, f"registry recipe.env not emitted for architect_critic: {missing}"
    leaked = not_serving & set(emitted)
    assert not leaked, f"recipe.env_not_serving knobs reached the launch env: {sorted(leaked)}"
