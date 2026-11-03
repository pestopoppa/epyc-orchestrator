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


# ---- change 2: gpu-quiet is opt-in (default none) ---------------------------------------------------

def test_gpu_quiet_defaults_to_none_and_help_says_decision_grade_only():
    from src.runtime import region_lock_cli as cli

    parser = cli.build_parser()
    assert parser.parse_args(["run", "--regions", "q0", "--", "true"]).gpu_quiet == "none"
    assert parser.parse_args(["run", "--gpu-quiet", "shared", "--", "true"]).gpu_quiet == "shared"
    assert parser.parse_args(["run", "--gpu-quiet", "none", "--", "true"]).gpu_quiet == "none"
    run_help = next(a for a in parser._subparsers._group_actions[0].choices["run"]._actions
                    if "--gpu-quiet" in a.option_strings).help
    assert "OPT-IN" in run_help and "decision-grade" in run_help
    assert "co-residency" in (cli.__doc__ or "") and "OPT-IN" in (cli.__doc__ or "")


def test_cli_default_run_takes_no_gpu_quiet_lock(lock_dir):
    probe = ("import fcntl,sys\nfh=open(sys.argv[1],'a+b')\n"
             "try:\n    fcntl.flock(fh.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept OSError:\n    sys.exit(7)\n")
    for extra in ([], ["--gpu-quiet", "none"]):
        res = _cli(lock_dir, ["run", "--regions", "q1", "--no-preflight", *extra, "--",
                              sys.executable, "-c", probe, str(lock_dir / "gpu_quiet.lock")])
        assert res.returncode == 0, res.stderr          # nobody holds gpu-quiet while the child runs
        assert "gpu-quiet" not in res.stderr.split("acquiring", 1)[1].split("\n", 1)[0]


def test_cli_gpu_quiet_none_alone_is_a_usage_error(lock_dir):
    res = _cli(lock_dir, ["run", "--gpu-quiet", "none", "--", "true"])
    assert res.returncode == 64 and "--gpu-quiet" in res.stderr


# ---- change 3: a gpu-quiet wait cites a hardware sample ----------------------------------------------

def _fake_sysfs(root: Path, *, busy: int, used: int, total: int, kfd_pids=(11, 12)) -> tuple[Path, Path]:
    drm = root / "drm"
    dev = drm / "card2" / "device"
    dev.mkdir(parents=True)
    (dev / "gpu_busy_percent").write_text(f"{busy}\n")
    (dev / "mem_info_vram_used").write_text(f"{used}\n")
    (dev / "mem_info_vram_total").write_text(f"{total}\n")
    (drm / "card2-DP-1").mkdir()            # a connector: ignored
    (drm / "card0" / "device").mkdir(parents=True)  # no VRAM files: ignored
    kfd = root / "kfd"
    kfd.mkdir()
    for pid in kfd_pids:
        (kfd / str(pid)).mkdir()
    return drm, kfd


def test_gpu_hw_sample_reads_sysfs_only(tmp_path, monkeypatch):
    from src.runtime.region_lock_cli import gpu_hw_sample

    monkeypatch.setenv("GPU_QUIET_HW_CHECK", "1")
    drm, kfd = _fake_sysfs(tmp_path, busy=87, used=8 * 2**30, total=64 * 2**30)

    def no_rocm_smi():
        raise AssertionError("rocm-smi must not run when sysfs answers")

    text = gpu_hw_sample(drm_dir=drm, kfd_dir=kfd, rocm_smi=no_rocm_smi)
    assert text == "hw: card2 busy=87% vram=8.0/64.0GiB; kfd_procs=2"


def test_gpu_hw_sample_falls_back_to_rocm_smi_and_can_be_disabled(tmp_path, monkeypatch):
    from src.runtime.region_lock_cli import gpu_hw_sample

    monkeypatch.setenv("GPU_QUIET_HW_CHECK", "1")
    (tmp_path / "drm").mkdir()
    text = gpu_hw_sample(drm_dir=tmp_path / "drm", kfd_dir=tmp_path / "nokfd", rocm_smi=lambda: "rocm-smi X")
    assert text == "hw: rocm-smi X; kfd_procs=?"
    monkeypatch.setenv("GPU_QUIET_HW_CHECK", "0")
    assert "disabled" in gpu_hw_sample(drm_dir=tmp_path / "drm", kfd_dir=tmp_path / "nokfd")


def test_wait_observer_adds_hw_sample_for_gpu_quiet_only():
    from src.runtime.region_lock_cli import make_wait_observer

    out = io.StringIO()
    obs = make_wait_observer(print_every_s=0, clock=lambda: 0.0, out=out,
                             describe=lambda path, region: "pid 9", hw_sample=lambda: "hw: SAMPLE")
    obs("gpu-quiet", "r", Path("/x/gpu_quiet.lock"), 0.0, "t")
    obs("q0", "r", Path("/x/cpu_region.GLOBAL.q0.lock"), 0.0, "t")
    gq_line, q_line = out.getvalue().splitlines()
    assert "held by pid 9 | hw: SAMPLE" in gq_line
    assert "hw:" not in q_line


def test_cli_gpu_quiet_wait_and_timeout_cite_hardware(lock_dir):
    from src.runtime.gpu_quiet_lock import gpu_quiet_lock

    drm, kfd = _fake_sysfs(lock_dir / "fakesys", busy=0, used=1 * 2**30, total=64 * 2**30, kfd_pids=())
    env = {"GPU_QUIET_HW_CHECK": "1", "ORCHESTRATOR_DRM_CLASS_DIR": str(drm),
           "ORCHESTRATOR_KFD_PROC_DIR": str(kfd), "REGION_LOCK_WAIT_PRINT_EVERY_S": "0"}
    with gpu_quiet_lock("exclusive", role="gpu-bench", timeout_s=2, request_tag="yarn-like"):
        res = _cli(lock_dir, ["run", "--gpu-quiet", "shared", "--timeout-s", "0.5", "--", "true"], env)
    assert res.returncode == 75, res.stderr
    waits = [ln for ln in res.stderr.splitlines() if ln.startswith("[region-lock] waiting")]
    assert waits and all("hw: card2 busy=0% vram=1.0/64.0GiB; kfd_procs=0" in ln for ln in waits)
    assert all(f"pid {os.getpid()}" in ln and "tag=yarn-like" in ln for ln in waits)
    assert "gpu-quiet at timeout: held by" in res.stderr and "kfd_procs=0" in res.stderr


# ---- change 4: cross-role preflight reads the stack's record, never a name-pattern scan ---------------

def test_region_lock_cli_has_no_pgrep():
    src = (REPO_ROOT / "src/runtime/region_lock_cli.py").read_text()
    assert '"pgrep"' not in src and "pgrep -f" not in src


def _state_file(path: Path, pid: int) -> Path:
    import json

    path.write_text(json.dumps({"orchestrator": {"role": "orchestrator", "pid": pid, "port": 8000,
                                                 "started_at": "x", "model_path": "", "log_file": ""}}))
    return path


def test_dispatch_pids_reads_state_file_without_subprocess(tmp_path, monkeypatch):
    from src.runtime import region_lock_cli as cli

    def boom(*a, **k):
        raise AssertionError("preflight must not spawn a process scan")

    monkeypatch.setattr(cli.subprocess, "run", boom)
    # Our own pid is live but its cmdline does not name the API module: a reused/stale record.
    assert cli._dispatch_pids(_state_file(tmp_path / "s.json", os.getpid())) == []
    assert cli._dispatch_pids(tmp_path / "missing.json") == []
    (tmp_path / "bad.json").write_text("{not json")
    assert cli._dispatch_pids(tmp_path / "bad.json") == []
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    assert cli._stack_state_file() == tmp_path / "orchestrator_state.json"
    assert cli._preflight(strict=True) is None      # no state file there: nothing dispatching


@pytest.mark.parametrize("flag, refused", [("0", True), ("1", False)])
def test_preflight_judges_the_recorded_api_pid(tmp_path, monkeypatch, flag, refused):
    from src.runtime import region_lock_cli as cli

    env = {k: v for k, v in os.environ.items() if k != "ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"}
    env["ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"] = flag
    # A stand-in whose cmdline names the API module (argv after -c), started and reaped by this test only.
    child = subprocess.Popen([sys.executable, "-c", "import sys,time; time.sleep(60)", "uvicorn", "src.api:app"],
                             env=env)
    try:
        monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
        _state_file(tmp_path / "orchestrator_state.json", child.pid)
        assert cli._dispatch_pids() == [child.pid]
        err = cli._preflight(strict=True)
        assert (err is not None) is refused
        if refused:
            assert str(child.pid) in err
            assert cli._preflight(strict=False) is None
    finally:
        child.kill()
        child.wait(timeout=10)
