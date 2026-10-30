"""Capacity is a launch/check boundary, while oversized lineups stay inspectable."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.fixtures.ni63_capacity_child.source_clone import probe_environment

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def capacity_clone(tmp_path_factory):
    # Copy current source, not NI63's independently pinned historical closure.
    # Inputs stay private; no model, live state, host policy or source pin changes.
    root = tmp_path_factory.mktemp("explicit-capacity")
    for name in ("scripts", "src", "orchestration"):
        shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def child(root, body):
    result = subprocess.run(
        [sys.executable, "-B", "-c", body],
        cwd=root,
        env=probe_environment(root),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def set_oversized(root, oversized):
    path = root / "orchestration/launch_manifest.yaml"
    data = yaml.safe_load(path.read_text())
    # Native host capacity arithmetic against unchanged /proc/meminfo. A reserve
    # greater than all MemTotal leaves a negative budget and must refuse.
    data["capacity"]["host_os_reserve_gib"] = 1e12 if oversized else 0
    path.write_text(yaml.safe_dump(data))


def test_oversized_import_status_and_explicit_pipeline_check(capacity_clone):
    set_oversized(capacity_clone, True)
    result = child(
        capacity_clone,
        """
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path
from scripts.server import stack_manifest, orchestrator_stack, stack_commands
from scripts.registry import stack_change_pipeline as pipeline
# Import must succeed without weakening the actual validator.
try:
    stack_manifest.validate_serving_shape_capacity()
except ValueError as exc:
    assert "device host (CPU RAM) OVERSUBSCRIBED" in str(exc)
else:
    raise AssertionError("oversized lineup accepted")
# Run the actual status handler with private empty state and no network probes.
with patch.object(stack_commands, "load_state", return_value={}), \\
     patch.object(stack_commands, "_declared_non_optional_services", return_value=[]), \\
     patch.object(stack_commands, "OPTIONAL_AUXILIARY_ROLES", set()):
    assert stack_commands.cmd_status(Namespace()) == 0
cfg = pipeline.StackChangePipelineConfig(mode="check", repo_root=Path.cwd(),
    lean_registry=Path.cwd()/"orchestration/model_registry.yaml")
step = pipeline._serving_shape_capacity_step(cfg)
assert step.status == "failed", step
assert any("OVERSUBSCRIBED" in error for error in step.errors), step
print("CHECK_REFUSED")
""",
    )
    assert "No components running" in result.stdout
    assert "CHECK_REFUSED" in result.stdout


@pytest.mark.parametrize("command", ["start", "reload"])
def test_oversized_actual_command_refuses_before_service_actions(capacity_clone, command):
    set_oversized(capacity_clone, True)
    result = child(
        capacity_clone,
        f"""
from argparse import Namespace
from unittest.mock import patch
from scripts.server import orchestrator_stack as stack, stack_commands as commands
# Trap even preparatory mutation/state paths, not merely the eventual spawn.
with patch.object(commands, "_start_registry_prelude", side_effect=AssertionError("compile")), \\
     patch.object(commands, "load_state", side_effect=AssertionError("state access")), \\
     patch.object(stack.subprocess, "Popen", side_effect=AssertionError("spawn")):
    assert commands.cmd_{command}(Namespace()) == 2
""",
    )
    assert "[REFUSED]" in result.stdout and "OVERSUBSCRIBED" in result.stdout


@pytest.mark.parametrize("entry", ["server", "api", "aux"])
def test_direct_serving_entries_refuse_before_spawn(capacity_clone, entry):
    set_oversized(capacity_clone, True)
    child(
        capacity_clone,
        f"""
from unittest.mock import patch
from scripts.server import orchestrator_stack as stack
with patch.object(stack.subprocess, "Popen", side_effect=AssertionError("spawn")):
    try:
        if {entry!r} == "server":
            stack.start_server(0, [], None)
        elif {entry!r} == "api":
            stack.start_orchestrator()
        else:
            stack.start_aux_service("whisper")
    except ValueError as exc:
        assert "OVERSUBSCRIBED" in str(exc)
    else:
        raise AssertionError("oversized launch was not refused")
""",
    )


def test_valid_capacity_keeps_pipeline_and_serving_dispatch(capacity_clone):
    set_oversized(capacity_clone, False)
    child(
        capacity_clone,
        """
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from scripts.server import stack_manifest, stack_commands as commands
from scripts.registry import stack_change_pipeline as pipeline
stack_manifest.validate_serving_shape_capacity()
cfg = pipeline.StackChangePipelineConfig(mode="check", repo_root=Path.cwd(),
    lean_registry=Path.cwd()/"orchestration/model_registry.yaml")
assert pipeline._serving_shape_capacity_step(cfg).status == "ok"
# Reach the existing next stage on valid capacity; never launch a real service.
with patch.object(commands, "_start_registry_prelude", return_value=93) as start:
    assert commands.cmd_start(Namespace()) == 93
    start.assert_called_once()
with patch.object(commands, "load_state", side_effect=RuntimeError("VALID_RELOAD_DISPATCH")):
    try:
        commands.cmd_reload(Namespace())
    except RuntimeError as exc:
        assert str(exc) == "VALID_RELOAD_DISPATCH"
    else:
        raise AssertionError("reload did not reach state stage")
""",
    )
