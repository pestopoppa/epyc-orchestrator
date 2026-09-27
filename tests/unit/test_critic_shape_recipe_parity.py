"""architect_critic's declared cpu_shape must be the one its registry recipe names.

The master registry's `recipe.cpu_shape: NUMA_FULL_T48` was a pointer nothing read, so
stack_topology.yaml kept NUMA_FULL and the live critic ran -t 96 against a recipe of 48
for four days (DAR-LAT-3h). This recomputes both sides and diffs them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.server.stack_numa import _CPU_SHAPES, NUMA_CONFIG, NUMA_INSTANCE_SHAPES

LEAN = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"


def test_critic_topology_shape_matches_registry_recipe() -> None:
    recipe = yaml.safe_load(LEAN.read_text())["server_mode"]["architect_critic"]["recipe"]
    assert NUMA_INSTANCE_SHAPES["architect_critic"] == (recipe["cpu_shape"],)
    threads = NUMA_CONFIG["architect_critic"]["instances"][0][2]
    assert threads == _CPU_SHAPES[recipe["cpu_shape"]][1] == int(recipe["threads"])
