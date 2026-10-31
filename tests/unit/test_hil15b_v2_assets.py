"""Tracked frozen-asset/native-driver integration; no model or regeneration."""
import hashlib
import json
from pathlib import Path

from scripts.trace.hil15a_flagon_run import grade, load_tasks

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "artifacts/hil/hil15b-v2-20261025"


def test_frozen_asset_bytes_and_original_anchor():
    raw = (ASSETS / "hil15b_v2_tasks.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "0089c2a840bc40367433de754424b17ad9a5edca954b312513e6ecaa0abd4312"
    )
    warrants = json.loads((ASSETS / "hil15b_v2_warrants.json").read_text())
    anchor = next(w for w in warrants if w["task_key"] == "lcg5-hard")
    assert anchor["inputs"] == dict(
        x0=12345, a=1103515245, c=12345, m=2147483648, n=5
    )
    assert anchor["answer"] == "1109335178"


def test_native_loader_grade_and_exported_asset_partition():
    tasks = load_tasks(str(ASSETS / "hil15b_v2_tasks.json"))
    calibration = json.loads((ASSETS / "calibration_tasks.json").read_text())
    heldout = json.loads((ASSETS / "heldout_tasks.json").read_text())
    assert len(tasks) == 20 and len(calibration) == 16 and len(heldout) == 4
    assert {t["task_key"] for t in calibration}.isdisjoint(
        {t["task_key"] for t in heldout}
    )
    assert sorted(t["task_key"] for t in tasks) == sorted(
        t["task_key"] for t in calibration + heldout
    )
    for task in tasks:
        assert grade(task, task["mock_answer"])
        assert not grade(task, "answer: " + task["mock_answer"])
        assert not grade(task, task["mock_answer"] + "x")
        assert not grade(task, None)
        assert task["tool_required"] is True
        assert "file_write_safe(" in task["prompt"]
        assert "peek(n=10000, file_path=" in task["prompt"]
        assert "/codex-ni-20261025-ufh08/generation-a/task-files/" in task["prompt"]


def test_canonical_driver_not_changed():
    driver = ROOT / "scripts/trace/hil15a_flagon_run.py"
    assert hashlib.sha256(driver.read_bytes()).hexdigest() == (
        "75abff0842a550051b3a25f3bf079a505044120a9ee57e80cc08ef2661bb4c63"
    )
