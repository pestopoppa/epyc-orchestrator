"""Tests for safe stack reload behavior."""

from __future__ import annotations

from argparse import Namespace

from scripts.server import orchestrator_stack as stack


def test_reload_embedders_uses_listener_pid_helper(monkeypatch) -> None:
    original_info = stack.ProcessInfo(
        role="embedder",
        pid=111,
        port=8090,
        started_at="before",
        model_path="old",
        log_file="old.log",
    )
    replacement_info = stack.ProcessInfo(
        role="embedder",
        pid=222,
        port=8090,
        started_at="after",
        model_path="new",
        log_file="new.log",
    )
    state = {"server_8090": original_info, "embedder": original_info}
    killed: list[int] = []
    pid_helper_calls: list[int] = []
    saved: list[dict[str, stack.ProcessInfo]] = []

    monkeypatch.setattr(stack, "EMBEDDER_PORTS", [8090])
    monkeypatch.setattr(stack, "load_state", lambda: state)
    monkeypatch.setattr(stack, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(stack, "RegistryLoader", lambda: object())
    monkeypatch.setattr(stack, "kill_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(stack, "is_port_in_use", lambda port: port == 8090)
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)

    def fake_pids_on_port(port: int) -> list[int]:
        pid_helper_calls.append(port)
        return [333]

    monkeypatch.setattr(stack, "_pids_on_port", fake_pids_on_port)
    monkeypatch.setattr(
        stack,
        "start_server",
        lambda *args, **kwargs: replacement_info,
    )

    rc = stack.cmd_reload(Namespace(components=["embedders"]))

    assert rc == 0
    assert killed == [111, 333]
    assert pid_helper_calls == [8090]
    assert saved[-1]["server_8090"] == replacement_info
    assert saved[-1]["embedder"] == replacement_info
