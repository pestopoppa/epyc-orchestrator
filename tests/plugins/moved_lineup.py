"""SSU-F18 opt-in mutation: move the worker lane onto architect_critic.

Only compiler-test processes selecting this plugin receive the mutation. Files
are never overwritten; pytest's monkeypatch restores every module attribute.
"""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parents[1] / "fixtures" / "stack_priors_moved_lineup"
PINNED_TEST = "test_regenerated_worker_math_url_byte_equals_fix_a_delegated_value"


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name == "test_stack_priors_compiler.py" and item.name == PINNED_TEST:
            item.add_marker(pytest.mark.xfail(strict=True, reason="SSU-F18 deliberate multi-instance full: wire-prefix pin"))


@pytest.fixture(autouse=True)
def moved_lineup(request, monkeypatch, tmp_path):
    if request.node.path.name != "test_stack_priors_compiler.py":
        return
    from scripts.server import stack_manifest as manifest
    from scripts.validate import check_shared_with_derivations as shared

    registry_overlay = yaml.safe_load((FIXTURES / "model_registry.yaml").read_text())
    launch_overlay = yaml.safe_load((FIXTURES / "launch_manifest.yaml").read_text())
    host = registry_overlay["host"]
    aliases = registry_overlay["shared_roles"]
    assert launch_overlay["inherit_port_from"] == host
    registry = deepcopy(manifest._MASTER)
    for row in registry["server_mode"].values():
        if isinstance(row, dict) and isinstance(row.get("shared_with"), list):
            row["shared_with"] = [name for name in row["shared_with"] if name not in aliases]
    destination = registry["server_mode"][host]
    destination["shared_with"] = list(dict.fromkeys(destination.get("shared_with", []) + aliases))
    declarations = deepcopy(manifest._MANIFEST)
    port = manifest.PORT_MAP[host]
    for alias in aliases:
        declarations["port_map"][alias] = port
    monkeypatch.setattr(manifest, "_MASTER", registry)
    monkeypatch.setattr(manifest, "MASTER_SERVER_MODE", registry["server_mode"])
    monkeypatch.setattr(manifest, "_MANIFEST", declarations)
    monkeypatch.setattr(manifest, "PORT_MAP", dict(declarations["port_map"]))
    monkeypatch.setattr(manifest, "ROLE_LAUNCH_META", manifest._build_role_launch_meta())
    hot, warm = manifest._build_servers_from_classification()
    monkeypatch.setattr(manifest, "HOT_SERVERS", hot)
    monkeypatch.setattr(manifest, "WARM_SERVERS", warm)

    # The cross-source alias check must see the SAME moved declarations, rather
    # than comparing the mutated launcher to the unchanged on-disk registry.
    moved_registry = tmp_path / "model_registry.yaml"
    moved_manifest = tmp_path / "launch_manifest.yaml"
    moved_registry.write_text(yaml.safe_dump(registry))
    moved_manifest.write_text(yaml.safe_dump(declarations))
    original_load = shared.load_sources
    original_registry = manifest._LEAN_REGISTRY_PATH
    original_manifest = shared.DEFAULT_LAUNCH_MANIFEST

    def load_moved_sources(master, launch, topology, models_py=None):
        return original_load(
            moved_registry if master == original_registry else master,
            moved_manifest if launch == original_manifest else launch,
            topology,
            models_py,
        )

    monkeypatch.setattr(shared, "load_sources", load_moved_sources)
