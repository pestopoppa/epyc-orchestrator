"""Review fixes of 2026-11-03 for the region-lock CLI / gpu-quiet lock.

Model-free: temp lock dirs only, injected clocks/samplers, no subprocess to rocm-smi.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def lock_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_PATHS_TMP_DIR", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    monkeypatch.setenv("GPU_QUIET_HW_CHECK", "0")
    return tmp_path


def _cli(lock_dir: Path, args: list[str], extra_env: dict[str, str] | None = None, timeout: int = 30):
    env = os.environ.copy()
    env.pop("ORCHESTRATOR_PATHS_TMP_DIR", None)
    env.pop("FILLER_SMOKE", None)
    env.pop("FILLER_DRYRUN", None)
    env["ORCHESTRATOR_TMP_DIR"] = str(lock_dir)
    env["GPU_QUIET_HW_CHECK"] = "0"
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "src.runtime.region_lock_cli", *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False,
    )


# ---- change 1: finite default timeout + holder printed on every wait -----------------------------

def test_timeout_default_is_600_and_explicit_zero_means_forever():
    from src.runtime.region_lock_cli import DEFAULT_TIMEOUT_S, build_parser

    assert DEFAULT_TIMEOUT_S == 600.0
    parser = build_parser()
    assert parser.parse_args(["run", "--regions", "q0", "--", "true"]).timeout_s == 600.0
    assert parser.parse_args(["run", "--regions", "q0", "--timeout-s", "0", "--", "true"]).timeout_s == 0.0


def test_wait_observer_prints_every_poll_when_interval_zero():
    from src.runtime.region_lock_cli import make_wait_observer

    out = io.StringIO()
    now = [100.0]
    obs = make_wait_observer(print_every_s=0, clock=lambda: now[0], out=out,
                             describe=lambda path, region: "pid 7 role=x tag=y age=3s")
    for i in range(3):
        now[0] += 0.05
        obs("q0", "bench", Path("/tmp/cpu_region.GLOBAL.q0.lock"), i * 0.05, "tagA")
    lines = out.getvalue().splitlines()
    assert len(lines) == 3
    assert all("pid 7 role=x tag=y age=3s" in line and "q0" in line for line in lines)


def test_wait_observer_first_block_immediate_then_rate_limited():
    from src.runtime.region_lock_cli import make_wait_observer

    out = io.StringIO()
    now = [0.0]
    obs = make_wait_observer(print_every_s=5.0, clock=lambda: now[0], out=out,
                             describe=lambda path, region: "holder")
    p = Path("/x/lock")
    obs("q1", "r", p, 0.0, None)      # first block: printed at once
    now[0] = 1.0
    obs("q1", "r", p, 1.0, None)      # inside the interval: silent
    now[0] = 5.5
    obs("q1", "r", p, 5.5, None)      # interval elapsed: printed
    assert len(out.getvalue().splitlines()) == 2


def test_describe_holders_reports_pid_tag_age_from_proc_locks(lock_dir):
    from src.runtime.cpu_region_lock import cpu_region_lock, global_region_lock_path
    from src.runtime.region_lock_cli import describe_holders

    with cpu_region_lock("holderrole", frozenset({"q0"}), timeout_s=2, request_tag="held-by-test"):
        text = describe_holders(global_region_lock_path("q0"), "q0")
    assert f"pid {os.getpid()}" in text
    assert "role=holderrole" in text and "tag=held-by-test" in text and "age=" in text


def test_describe_holders_gpu_quiet(lock_dir):
    from src.runtime.gpu_quiet_lock import gpu_quiet_lock, gpu_quiet_lock_path
    from src.runtime.region_lock_cli import describe_holders

    with gpu_quiet_lock("exclusive", role="gq", timeout_s=2, request_tag="yarn"):
        text = describe_holders(gpu_quiet_lock_path(), "gpu-quiet")
    assert f"pid {os.getpid()}" in text and "mode=exclusive" in text and "tag=yarn" in text


def test_cli_wait_prints_holder_and_exits_75(lock_dir):
    from src.runtime.cpu_region_lock import cpu_region_lock

    with cpu_region_lock("holderrole", frozenset({"q0"}), timeout_s=2, request_tag="blocker"):
        res = _cli(lock_dir, ["run", "--regions", "q0", "--no-preflight", "--timeout-s", "1.0", "--", "true"],
                   {"REGION_LOCK_WAIT_PRINT_EVERY_S": "0"})
    assert res.returncode == 75, res.stderr
    assert res.stderr.count("held by pid") >= 2          # a line per wait iteration, not once
    assert f"pid {os.getpid()}" in res.stderr and "blocker" in res.stderr

