"""Regression coverage for the standalone region-lock status command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_status_json_is_clean_when_stack_is_stopped(tmp_path: Path) -> None:
    """Status must not bootstrap config/runtime facts just to inspect locks."""
    env = os.environ.copy()
    env.pop("ORCHESTRATOR_TMP_DIR", None)
    env["ORCHESTRATOR_PATHS_TMP_DIR"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "src.runtime.region_lock_cli", "status", "--json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == [
        {"region": "q0", "global_held": False, "holders": []},
        {"region": "q1", "global_held": False, "holders": []},
        {"region": "q2", "global_held": False, "holders": []},
        {"region": "q3", "global_held": False, "holders": []},
    ]


def _rl_run(tmp_path: Path, cpus: str, marker: str, extra: list[str] | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env.pop("ORCHESTRATOR_PATHS_TMP_DIR", None)
    env.pop("FILLER_SMOKE", None)
    env.pop("FILLER_DRYRUN", None)
    env["ORCHESTRATOR_TMP_DIR"] = str(tmp_path)
    cmd = [sys.executable, "-m", "src.runtime.region_lock_cli", "run", "--no-preflight", "--role", "build",
           "--timeout-s", "3", *(extra or []), "--cpu-list", cpus, "--",
           "bash", "-c", f"touch {tmp_path / marker}; sleep 6"]
    return subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_disjoint_subquarter_cpu_lists_share_one_quarter(tmp_path: Path) -> None:
    """ws58 sub-quarter packing: two holders on disjoint cpu lists inside q2 both run; an overlapping one and a
    whole-quarter one time out (75) while they hold."""
    a = _rl_run(tmp_path, "48-51,144-147", "a")
    b = _rl_run(tmp_path, "52-55", "b")
    import time as _t
    for _ in range(100):
        if (tmp_path / "a").exists() and (tmp_path / "b").exists():
            break
        _t.sleep(0.1)
    c = _rl_run(tmp_path, "51", "c")
    d = _rl_run(tmp_path, "48-71", "d")
    assert c.wait(20) == 75 and d.wait(20) == 75
    assert a.wait(20) == 0, a.stderr.read()
    assert b.wait(20) == 0, b.stderr.read()
    assert (tmp_path / "a").exists() and (tmp_path / "b").exists()
    assert not (tmp_path / "c").exists() and not (tmp_path / "d").exists()
