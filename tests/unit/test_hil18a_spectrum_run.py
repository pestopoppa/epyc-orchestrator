"""HIL-18a in-process spectrum driver: guards, selection, resume, and a mocked-LLM end to end (plumbing only)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "trace" / "hil18a_spectrum_run.py"
sys.path.insert(0, str(ROOT / "scripts" / "trace"))
import hil18a_spectrum_run as drv  # noqa: E402


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


def test_select_prefers_variance_and_drops_all_pass(tmp_path):
    tasks = [{"task_key": k} for k in ("a", "b", "c", "d")]
    m = tmp_path / "m.json"
    runs = [{"task_key": "a", "pass": True}, {"task_key": "a", "pass": True},
            {"task_key": "b", "pass": False}, {"task_key": "b", "pass": False},
            {"task_key": "c", "pass": True}, {"task_key": "c", "pass": False}]
    m.write_text(json.dumps({"runs": runs}))
    assert [t["task_key"] for t in drv.select_tasks(tasks, str(m), 10)] == ["c", "b"]
    assert [t["task_key"] for t in drv.select_tasks(tasks, str(m), 1)] == ["c"]
    assert len(drv.select_tasks(tasks, None, 3)) == 3


def test_done_counts(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text('{"task_key":"a"}\n{"task_key":"a"}\n{"task_key":"b"}\n')
    assert drv.done_counts(f) == {"a": 2, "b": 1} and drv.done_counts(tmp_path / "none") == {}


def test_mock_end_to_end_and_resume(tmp_path):
    tasks = tmp_path / "t.json"
    tasks.write_text(json.dumps([{"task_key": f"k{i}", "prompt": f"say {i}", "expect": r"\bok\b", "mock_answer": "ok"} for i in range(5)]))
    base = ["--out-dir", str(tmp_path / "o"), "--mock-llm", "--no-reexec", "--tasks", str(tasks), "--max-tasks", "2",
            "--n", "6", "--min-replays", "6", "--tag", "t1", "--epyc-root", str(tmp_path / "noroot")]
    # stop file present from the start: nothing runs, partial exit
    stop = tmp_path / "STOP"
    stop.write_text("")
    r0 = _run(base + ["--stop-file", str(stop)])
    assert r0.returncode == 4, r0.stderr[-600:]
    stop.unlink()
    r = _run(base)
    m = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert r.returncode == 0, r.stderr[-800:]
    assert m["mock"] is True and m["tasks_complete"] == ["k0", "k1"] and m["join"]["replays_started"] == 12
    assert m["join"]["with_steps"] == 12 and m["report"]["tasks_reported"] == 2
    assert "sidecar_skipped" in m
    rep = json.loads((tmp_path / "o" / "spectrum_report.json").read_text())
    assert rep["schema"] == "spectrum_report/v1" and rep["label"] == "mock"
    # rerun resumes: nothing new to replay
    r2 = _run(base)
    m2 = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert r2.returncode == 0 and m2["replays_this_call"] == 0
