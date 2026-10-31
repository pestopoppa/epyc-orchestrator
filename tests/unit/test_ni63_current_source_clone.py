"""Current-source regression clones record source drift while pinning workloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from tests.fixtures.ni63_capacity_child import source_clone

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fixture_root(tmp_path):
    root = tmp_path / "current-source"
    for relative in [*source_clone.SOURCE_PINS, source_clone.CAPACITY_ARTIFACT_PIN[0]]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


def materialize(root, tmp_path):
    parent = tmp_path / "children"
    parent.mkdir(exist_ok=True)
    return source_clone.materialize_source_clone(
        root, parent, variant="fit", project_venv_python=Path(sys.executable)
    )


def test_current_source_drift_is_copied_and_recorded_not_rejected(fixture_root, tmp_path):
    relative = "scripts/server/orchestrator_stack.py"
    current = fixture_root / relative
    current.write_bytes(current.read_bytes() + b"\n# current-source regression fixture change\n")
    clone = materialize(fixture_root, tmp_path)
    manifest = json.loads((clone / "source_identity.json").read_text())
    assert manifest["historical_fixture_baseline"] == source_clone.PINNED_APP_COMMIT
    assert manifest["variant"] == "fit"
    assert set(manifest["sources"]) == set(source_clone.SOURCE_PINS)
    assert manifest["sources"][relative]["matches_historical_baseline"] is False
    for path, identity in manifest["sources"].items():
        copied = (clone / path).read_bytes()
        assert copied == (fixture_root / path).read_bytes()
        assert identity["sha256"] == hashlib.sha256(copied).hexdigest()
    assert (clone / "source_identity.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("damage", ["missing", "linked"])
def test_absent_or_linked_current_source_still_refuses(fixture_root, tmp_path, damage):
    source = fixture_root / "scripts/server/orchestrator_stack.py"
    source.unlink()
    if damage == "linked":
        source.symlink_to(ROOT / "scripts/server/orchestrator_stack.py")
    with pytest.raises(RuntimeError, match="pinned source is absent or linked"):
        materialize(fixture_root, tmp_path)


def test_capacity_policy_change_still_refuses(fixture_root, tmp_path):
    policy = fixture_root / source_clone.CAPACITY_ARTIFACT_PIN[0]
    policy.write_bytes(policy.read_bytes() + b"\n# forbidden workload policy drift\n")
    with pytest.raises(RuntimeError, match="declared GPU-capacity source differs"):
        materialize(fixture_root, tmp_path)
    assert not list((tmp_path / "children").rglob("source_identity.json"))


def test_workload_fixture_change_still_refuses(fixture_root, tmp_path, monkeypatch):
    monkeypatch.setattr(
        source_clone,
        "FIXTURE_PINS",
        {**source_clone.FIXTURE_PINS, "launch_manifest.yaml": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="fixture bytes changed"):
        materialize(fixture_root, tmp_path)
    assert not list((tmp_path / "children").rglob("source_identity.json"))
