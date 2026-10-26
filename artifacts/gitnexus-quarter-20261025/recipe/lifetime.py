#!/usr/bin/python3
"""GRANT ONLY: exercise real Node lifetime through the production supervisor.

All substitutions exist in this imported test process. Production exposes no
environment or CLI target override. This does not replace real guard proof.
"""
import ctypes
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def proc_identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return int(fields[19]), fields[0]


def load_writer(scripts):
    spec = importlib.util.spec_from_file_location("fixture_writer", scripts / "gitnexus-writer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(scripts, directory):
    writer = load_writer(scripts)
    guard = writer.cpu_guard()
    cpus = frozenset(os.sched_getaffinity(0))
    current = {"owner": os.getppid(), "region": "fixture-only", "cpus": cpus}
    original_popen = subprocess.Popen
    captured = {}

    def launch(_production_command, **kwargs):
        # Keep the exact production env, pass_fds and preexec_fn. Only this test
        # process substitutes its non-CLI Node fixture command.
        child = original_popen([guard.NODE, "--v8-pool-size=1", str(HERE / "node-hold.cjs"), str(directory)], **kwargs)
        captured["child"] = child
        start, _ = proc_identity(child.pid)
        temporary = directory / "CAPTURED.tmp"
        temporary.write_text(json.dumps({"pid": child.pid, "start": start}))
        temporary.replace(directory / "CAPTURED.json")
        return child

    def readback(_current, pid):
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
        if len(tasks) > len(cpus) or any(frozenset(os.sched_getaffinity(int(tid.name))) != cpus for tid in tasks):
            raise RuntimeError("fixture Node thread mask/count escaped the grant")

    admission = SimpleNamespace(NODE=guard.NODE, GNX=guard.GNX,
        installed_source=guard.installed_source, claim=lambda: current, node_admission=readback,
        native_artifact=guard.native_artifact,
        DB_BUFFER_BYTES=guard.DB_BUFFER_BYTES, DB_MAX_BYTES=guard.DB_MAX_BYTES)
    with (directory / "writer.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with patch.object(writer.subprocess, "Popen", side_effect=launch):
                writer.quarter_run("app", [], lock.fileno(), admission, current)
        except writer.Termination as exc:
            return 128 + exc.signum
    return 1


def main(scripts, output):
    # Adopt and reap ONLY the known captured Node when its test writer is killed.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot become fixture subreaper")
    receipts = []
    protection = load_writer(scripts)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGKILL):
        directory = output / signal.Signals(sig).name
        directory.mkdir(parents=True)
        driver_pid = os.getpid()
        writer = subprocess.Popen([sys.executable, str(HERE / "lifetime.py"), "--worker", str(scripts), str(directory)],
            preexec_fn=lambda: protection.parent_death_guard(driver_pid, set()))
        captured = None
        try:
            limit = time.monotonic() + 15
            while time.monotonic() < limit:
                if writer.poll() is not None:
                    raise RuntimeError(f"fixture writer exited early: {writer.returncode}")
                if (directory / "CAPTURED.json").exists() and (directory / "NODE-READY.json").exists():
                    captured = json.loads((directory / "CAPTURED.json").read_text())
                    break
                time.sleep(0.05)
            if captured is None:
                raise RuntimeError("no captured Node readiness")
            pid = captured["pid"]
            if proc_identity(pid)[0] != captured["start"]:
                raise RuntimeError("captured Node identity changed")
            masks = {tid.name: sorted(os.sched_getaffinity(int(tid.name)))
                     for tid in Path(f"/proc/{pid}/task").iterdir()}
            if any(mask != sorted(os.sched_getaffinity(0)) for mask in masks.values()):
                raise RuntimeError("actual Node fixture affinity mismatch")
            lock_stat = (directory / "writer.lock").stat()
            inherited = False
            for entry in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    state = entry.stat()
                    inherited |= (state.st_dev, state.st_ino) == (lock_stat.st_dev, lock_stat.st_ino)
                except FileNotFoundError:
                    pass
            if not inherited:
                raise RuntimeError("actual Node did not inherit the writer lock FD")
            # Prove the inherited single-writer FD excludes a second owner.
            with (directory / "writer.lock").open("a+b") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise RuntimeError("live writer lock was not held")
            writer.send_signal(sig)  # Captured own writer PID only.
            rc = writer.wait(timeout=10)
            limit = time.monotonic() + 8
            reaped = False
            while time.monotonic() < limit:
                try:
                    reaped = os.waitpid(pid, os.WNOHANG)[0] == pid
                except ChildProcessError:
                    # Graceful writer already reaped it.
                    reaped = not Path(f"/proc/{pid}").exists()
                if reaped and not Path(f"/proc/{pid}").exists():
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("captured Node survived writer termination")
            expected_rc = -sig if sig == signal.SIGKILL else 128 + sig
            if rc != expected_rc:
                raise RuntimeError(f"unexpected writer rc {rc}, expected {expected_rc}")
            with (directory / "writer.lock").open("a+b") as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipts.append({"signal": sig, "writer_pid": writer.pid, "writer_rc": rc,
                "captured_node_pid": pid, "node_reaped": reaped, "proc_absent": True,
                "node_task_masks": masks, "node_inherited_lock_fd": inherited, "writer_lock_released": True})
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.wait(timeout=10)
            if captured is not None:
                pid = captured["pid"]
                try:
                    start, state = proc_identity(pid)
                    if start == captured["start"] and state != "Z":
                        os.kill(pid, signal.SIGKILL)
                except FileNotFoundError:
                    pass
                limit = time.monotonic() + 5
                while time.monotonic() < limit:
                    try:
                        if os.waitpid(pid, os.WNOHANG)[0] == pid:
                            break
                    except ChildProcessError:
                        break
                    time.sleep(0.05)
                else:
                    raise RuntimeError("captured Node could not be reaped during cleanup")
    (output / "lifetime-proof.json").write_text(json.dumps({"status": "PASS", "receipts": receipts}, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        raise SystemExit(worker(Path(sys.argv[2]), Path(sys.argv[3])))
    main(Path(sys.argv[1]), Path(sys.argv[2]))
