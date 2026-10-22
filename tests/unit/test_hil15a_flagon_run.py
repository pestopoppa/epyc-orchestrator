"""HIL-15a driver: guards and a mocked-LLM end-to-end run (plumbing only; the mock AUC is not a finding)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "trace" / "hil15a_flagon_run.py"
sys.path.insert(0, str(ROOT / "scripts" / "trace"))
import hil15a_flagon_run as drv  # noqa: E402


def _run(args, env_extra=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("WINDOW_CONFIRMED", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=300)


def test_live_refuses_without_window(tmp_path):
    r = _run(["--out-dir", str(tmp_path)])
    assert r.returncode != 0 and "WINDOW_CONFIRMED=1" in (r.stderr + r.stdout)


def test_live_needs_private_server_proof(tmp_path):
    r = _run(["--out-dir", str(tmp_path)], {"WINDOW_CONFIRMED": "1"})
    assert r.returncode != 0 and "--server-pid" in (r.stderr + r.stdout)


def test_too_few_task_keys(tmp_path):
    t = tmp_path / "t.json"
    t.write_text(json.dumps(drv.DEFAULT_TASKS[:3]))
    r = _run(["--out-dir", str(tmp_path / "o"), "--mock-llm", "--no-reexec", "--tasks", str(t)])
    assert r.returncode != 0 and ">= 5 task keys" in (r.stderr + r.stdout)


def test_port_ownership(monkeypatch):
    monkeypatch.setattr(drv, "listening_pid", lambda port: 4242)
    with pytest.raises(SystemExit, match="production stack port"):
        drv.prove_port_ownership("http://127.0.0.1:8070", 4242, {8070})
    with pytest.raises(SystemExit, match="owned by pid 4242"):
        drv.prove_port_ownership("http://127.0.0.1:9311", 1, {8070})
    with pytest.raises(SystemExit, match="loopback"):
        drv.prove_port_ownership("http://10.0.0.5:9311", 4242, set())
    assert drv.prove_port_ownership("http://127.0.0.1:9311", 4242, {8070}) == 9311


def test_grade():
    assert drv.grade(drv.DEFAULT_TASKS[0], "the answer is 391") and not drv.grade(drv.DEFAULT_TASKS[0], None)


def test_mock_end_to_end(tmp_path):
    root = os.environ.get("EPYC_ROOT", "/workspace")
    r = _run(["--out-dir", str(tmp_path), "--mock-llm", "--no-reexec", "--epyc-root", root])
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["mock"] is True and set(m["flags"]) == set(drv.FLAGS), r.stderr[-800:]
    assert m["join"]["step_bearing_runs"] == m["join"]["joined_outcome"] == m["join"]["task_runs"] == 24
    assert all(v["rc"] == 0 for v in m["replay_verify"].values()) and len(m["replay_verify"]) == 24
    assert m["auc_table"]["n_tasks"] == 6 and "auc_out_of_fold" in m["auc_table"]
    assert sorted(m["label_classes"]) == ["False", "True"]


def test_missing_sidecar_adapter_is_not_a_problem(tmp_path):
    empty_root = tmp_path / "root"
    empty_root.mkdir()
    r = _run(["--out-dir", str(tmp_path / "o"), "--mock-llm", "--no-reexec", "--epyc-root", str(empty_root)])
    m = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert m["l2_scan"]["rc"] == 0, m["l2_scan"]["stderr"]
    assert m["l2_scan"]["sidecar"] is None and "adapter not found" in m["l2_scan"]["sidecar_skipped"]
    assert m["verdict"] == "OK", r.stderr[-500:]
    assert m["dep_coverage"]["task_keys"] == 6


def test_wall_cap_and_dependency_coverage_warning(tmp_path):
    r = _run(["--out-dir", str(tmp_path / "o"), "--mock-llm", "--no-reexec", "--max-wall-s", "0.0001", "--repeats", "3"])
    m = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert len(m["runs"]) <= 6, r.stderr[-300:]   # repeat-major: at most one pass over the 6 keys got started


def test_new_task_set_mock_auc(tmp_path):
    tasks = Path("/mnt/raid0/llm/tmp/fillers-3c4ca599/jobs/hil15a_tasks.json")
    if not tasks.exists():
        pytest.skip("filler task set not present")
    _run(["--out-dir", str(tmp_path / "o"), "--mock-llm", "--no-reexec", "--tasks", str(tasks)])
    m = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert len(m["runs"]) == 56 and m["dep_coverage"]["task_keys"] == 14 and not m["warnings"]
    assert "status" not in m["auc_table"] and "auc_out_of_fold" in m["auc_table"]
