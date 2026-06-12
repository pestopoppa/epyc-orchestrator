"""Tests for safe stack reload behavior."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from scripts.server import orchestrator_stack as stack
from scripts.server import stack_commands


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


def test_reload_document_formalizer_uses_auxiliary_starter(monkeypatch) -> None:
    old_info = stack.ProcessInfo(
        role="document_formalizer",
        pid=111,
        port=9001,
        started_at="before",
        model_path="old",
        log_file="old.log",
    )
    new_info = stack.ProcessInfo(
        role="document_formalizer",
        pid=222,
        port=9001,
        started_at="after",
        model_path="LightOnOCR-2-1B-bbox",
        log_file="document_formalizer.log",
    )
    state = {"document_formalizer": old_info}
    killed: list[int] = []
    pid_helper_calls: list[int] = []
    saved: list[dict[str, stack.ProcessInfo]] = []

    monkeypatch.setattr(stack, "load_state", lambda: state)
    monkeypatch.setattr(stack, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(
        stack,
        "RegistryLoader",
        lambda: (_ for _ in ()).throw(AssertionError("registry should not load")),
    )
    monkeypatch.setattr(stack, "kill_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)

    def fake_pids_on_port(port: int) -> list[int]:
        pid_helper_calls.append(port)
        return [333]

    monkeypatch.setattr(stack, "_pids_on_port", fake_pids_on_port)
    monkeypatch.setattr(stack, "start_document_formalizer", lambda: new_info)

    rc = stack.cmd_reload(Namespace(components=["document_formalizer"]))

    assert rc == 0
    assert killed == [333]
    assert pid_helper_calls == [9001]
    assert saved[-1] == {"document_formalizer": new_info}


def test_preserved_process_info_records_listener_pid(monkeypatch) -> None:
    monkeypatch.setattr(stack_commands, "_pids_on_port", lambda port: [777])

    info = stack_commands._preserved_process_info(
        "frontdoor",
        8070,
        "preserved:frontdoor",
        "llama-server-8070.log",
    )

    assert info is not None
    assert info == stack_commands.ProcessInfo(
        role="frontdoor",
        pid=777,
        port=8070,
        started_at=info.started_at,
        model_path="preserved:frontdoor",
        log_file="llama-server-8070.log",
    )


def test_scan_known_ports_includes_warm_aux_and_docker(monkeypatch) -> None:
    captured: list[list[int]] = []

    monkeypatch.setattr(stack_commands, "HOT_SERVERS", [{"port": 8070}])
    monkeypatch.setattr(stack_commands, "WARM_SERVERS", [{"port": 8085}])
    monkeypatch.setattr(stack_commands, "NUMA_REPLICA_PORTS", {8180})
    monkeypatch.setattr(stack_commands, "DOCKER_SERVICES", [{"port": 8088}])

    def fake_scan(ports):
        captured.append(list(ports))
        return {}

    monkeypatch.setattr(stack_commands._stack_processes, "scan_known_ports", fake_scan)

    assert stack_commands._scan_known_ports() == {}
    assert captured == [[8000, 8070, 8085, 8088, 8180, 8190, 9000, 9001]]


def test_run_attestation_snapshot_treats_issue_exit_as_written(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    project_root = tmp_path
    script = project_root / "scripts" / "attest" / "generate_attestation.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('stub')\n", encoding="utf-8")
    (project_root / "orchestration").mkdir()
    calls = []

    monkeypatch.setattr(stack_commands, "_PATHS", {"project_root": project_root})
    monkeypatch.setattr(
        stack_commands.subprocess,
        "run",
        lambda cmd, **kwargs: (
            calls.append((cmd, kwargs))
            or SimpleNamespace(returncode=1, stdout="wrote latest.json\n", stderr="")
        ),
    )

    stack_commands._run_attestation_snapshot("unit_reload")

    out = capsys.readouterr().out
    assert "snapshot written for unit_reload (rc=1)" in out
    assert "--trigger" in calls[0][0]
    assert "unit_reload" in calls[0][0]
