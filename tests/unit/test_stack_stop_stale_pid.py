"""`stop <role>` must not signal a state-file PID that is no longer that server."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from scripts.server import stack_commands, stack_processes
from scripts.server.stack_state import ProcessInfo

MODEL = "/mnt/raid0/llm/models/m.gguf"
RECORDED = datetime(2026, 10, 27, 12, 0, 0)


def _info(pid=2055785, port=8080) -> ProcessInfo:
    return ProcessInfo(
        role="worker", pid=pid, port=port, started_at=RECORDED.isoformat(),
        model_path=MODEL, log_file="x.log",
    )


def _patch_proc(monkeypatch, *, alive=True, start=None, argv=(), listeners=()):
    monkeypatch.setattr(stack_processes, "pid_alive", lambda pid: alive)
    monkeypatch.setattr(
        stack_processes, "_proc_start_epoch",
        lambda pid: (start or RECORDED - timedelta(minutes=5)).timestamp(),
    )
    monkeypatch.setattr(stack_processes, "_proc_cmdline", lambda pid: list(argv))
    monkeypatch.setattr(stack_processes, "collect_descendants", lambda pid: [])
    monkeypatch.setattr(stack_processes, "pids_on_port", lambda port: list(listeners))


def _run_stop(monkeypatch, info):
    state = {"worker": info}
    saved = {}
    killed = []
    monkeypatch.setattr(stack_commands, "load_state", lambda: state)
    monkeypatch.setattr(stack_commands, "save_state", lambda s: saved.update(s) or saved.setdefault("_", 1))
    monkeypatch.setattr(stack_commands, "kill_process", lambda pid, *a, **k: killed.append(pid) or True)
    monkeypatch.setattr(stack_commands, "_pids_on_port", lambda p: stack_processes.pids_on_port(p))
    monkeypatch.setattr(stack_commands, "_refresh_runtime_facts_manifest", lambda *a, **k: None)
    monkeypatch.setattr(stack_commands, "realized_stack_numa_mode_from_state", lambda s: None)
    stack_commands.cmd_stop(argparse.Namespace(all=False, components=["worker"]))
    return state, killed


def test_reused_pid_different_cmdline_no_port_is_stale(monkeypatch, capsys):
    _patch_proc(monkeypatch, argv=["/usr/bin/python3", "other.py"])
    state, killed = _run_stop(monkeypatch, _info())
    assert killed == []
    assert "worker" not in state
    assert "[STALE] worker PID 2055785 is not this server; state entry dropped" in capsys.readouterr().out


def test_reused_pid_started_after_record_is_stale(monkeypatch):
    argv = ["/x/llama-server", "-m", MODEL, "--port", "8080"]
    _patch_proc(monkeypatch, argv=argv, start=RECORDED + timedelta(days=3))
    ok, why = stack_processes.verify_state_pid(_info())
    assert not ok and "reused" in why


def test_genuine_server_by_port_is_signalled(monkeypatch, capsys):
    _patch_proc(monkeypatch, listeners=[2055785], argv=["x"])
    state, killed = _run_stop(monkeypatch, _info())
    assert killed == [2055785]
    assert "worker" not in state
    assert "[STALE]" not in capsys.readouterr().out


def test_genuine_server_by_cmdline_is_signalled(monkeypatch):
    argv = ["/x/llama-server", "-m", MODEL, "--port", "8080"]
    _patch_proc(monkeypatch, argv=argv)
    _, killed = _run_stop(monkeypatch, _info())
    assert killed == [2055785]


def test_dead_pid_not_signalled(monkeypatch, capsys):
    _patch_proc(monkeypatch, alive=False)
    state, killed = _run_stop(monkeypatch, _info())
    assert killed == []
    assert "worker" not in state
    assert "[STALE]" in capsys.readouterr().out
