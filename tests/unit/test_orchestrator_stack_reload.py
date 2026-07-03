"""Tests for safe stack reload behavior."""

from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import yaml

from scripts.server import orchestrator_stack as stack
from scripts.server import stack_commands


def _stack_gate_args(**overrides) -> Namespace:
    defaults = {
        "dev": False,
        "validate_only": False,
        "migrate_to": None,
        "dry_run": False,
        "skip_stack_change_gate": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_stack_change_launch_gate_runs_canonical_command(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(cmd, 0, stdout="summary: ok\n", stderr="")

    monkeypatch.setattr(stack_commands.subprocess, "run", fake_run)

    assert stack_commands._run_stack_change_launch_gate(_stack_gate_args())

    assert captured["cmd"] == list(stack_commands.STACK_CHANGE_LAUNCH_GATE_COMMAND)
    assert str(captured["cwd"]).endswith("epyc-orchestrator")
    out = capsys.readouterr().out
    assert "stack-change-gate" in out
    assert "summary: ok" in out


def test_repo_short_sha_uses_repo_root(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(cmd, 0, stdout="abcdef1\n", stderr="")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)

    assert stack._repo_short_sha(tmp_path) == "abcdef1"
    assert captured["cmd"] == [
        "git",
        "-C",
        str(tmp_path),
        "rev-parse",
        "--short",
        "HEAD",
    ]
    assert captured["timeout"] == 5


def test_repo_short_sha_returns_none_on_git_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="not a repo")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)

    assert stack._repo_short_sha(tmp_path) is None


def test_start_parser_compiles_registry_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_cmd_start(args: Namespace) -> int:
        captured["compile_registry"] = args.compile_registry
        return 0

    monkeypatch.setattr(stack_commands, "cmd_start", fake_cmd_start)
    monkeypatch.setattr(sys, "argv", ["orchestrator_stack.py", "start"])

    assert stack.main() == 0
    assert captured["compile_registry"] is True


def test_start_parser_accepts_no_compile_registry(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_cmd_start(args: Namespace) -> int:
        captured["compile_registry"] = args.compile_registry
        return 0

    monkeypatch.setattr(stack_commands, "cmd_start", fake_cmd_start)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator_stack.py", "start", "--no-compile-registry"],
    )

    assert stack.main() == 0
    assert captured["compile_registry"] is False


def test_cmd_start_compiles_registry_by_default_arg_absence(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_or_compile(**kwargs):
        calls.append(kwargs)
        return {"roles": {}}

    import src.registry.registry_compiler as compiler

    monkeypatch.setattr(compiler, "load_or_compile", fake_load_or_compile)

    args = _stack_gate_args(
        stack_profile="default",
        validate_only=True,
        compile_descriptors=False,
        allow_incomplete_descriptors=False,
    )

    assert stack_commands.cmd_start(args) == 0
    assert len(calls) == 1
    assert calls[0]["output_path"].name == "model_registry.yaml"
    assert calls[0]["cache_key_path"].name == ".lean_cache_key"


def test_stack_change_launch_gate_failure_blocks_launch(monkeypatch, capsys) -> None:
    def fake_run(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="summary: failed\n", stderr="boom\n")

    monkeypatch.setattr(stack_commands.subprocess, "run", fake_run)

    assert not stack_commands._run_stack_change_launch_gate(_stack_gate_args())
    out = capsys.readouterr().out
    assert "summary: failed" in out
    assert "boom" in out
    assert "refusing launch" in out


def test_stack_change_launch_gate_has_explicit_skip(monkeypatch, capsys) -> None:
    def fake_run(*_args, **_kwargs):
        raise AssertionError("gate subprocess should not run")

    monkeypatch.setattr(stack_commands.subprocess, "run", fake_run)

    assert stack_commands._run_stack_change_launch_gate(
        _stack_gate_args(skip_stack_change_gate=True)
    )
    assert "SKIPPED (--skip-stack-change-gate)" in capsys.readouterr().out


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


def test_scan_known_ports_derives_manifest_aux_ports(monkeypatch) -> None:
    captured: list[list[int]] = []

    monkeypatch.setattr(stack_commands, "HOT_SERVERS", [{"port": 8070}])
    monkeypatch.setattr(stack_commands, "WARM_SERVERS", [{"port": 8085}])
    monkeypatch.setattr(stack_commands, "NUMA_REPLICA_PORTS", {8180})
    monkeypatch.setattr(stack_commands, "DOCKER_SERVICES", [{"port": 8088}])
    monkeypatch.setattr(stack_commands, "_stack_prior_serving_ports", lambda: set())
    monkeypatch.setattr(
        stack_commands,
        "PORT_MAP",
        {
            "orchestrator": 8000,
            "sd_server": 8190,
            "whisper": 9000,
            "document_formalizer": 9001,
            "manifest_only_aux": 9010,
        },
    )

    def fake_scan(ports):
        captured.append(list(ports))
        return {}

    monkeypatch.setattr(stack_commands._stack_processes, "scan_known_ports", fake_scan)

    assert stack_commands._scan_known_ports() == {}
    assert captured == [[8000, 8070, 8085, 8088, 8180, 8190, 9000, 9001, 9010]]


def test_stack_prior_serving_ports_only_reads_live_roles(tmp_path: Path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        yaml.safe_dump(
            {
                "roles": {
                    "frontdoor": {
                        "deployment_status": "live_stack",
                        "serving": {"ports": [8070, 8080, 8180]},
                    },
                    "candidate": {
                        "deployment_status": "benchmark_or_candidate",
                        "serving": {"ports": [9999]},
                    },
                    "malformed": {
                        "deployment_status": "live_stack",
                        "serving": {"ports": ["9000", None]},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert stack_commands._stack_prior_serving_ports(priors) == {8070, 8080, 8180}


def test_scan_known_ports_includes_stack_prior_live_ports(monkeypatch) -> None:
    captured: list[list[int]] = []

    monkeypatch.setattr(stack_commands, "HOT_SERVERS", [{"port": 8070}])
    monkeypatch.setattr(stack_commands, "WARM_SERVERS", [])
    monkeypatch.setattr(stack_commands, "NUMA_REPLICA_PORTS", set())
    monkeypatch.setattr(stack_commands, "DOCKER_SERVICES", [])
    monkeypatch.setattr(stack_commands, "PORT_MAP", {"orchestrator": 8000})
    monkeypatch.setattr(stack_commands, "_stack_prior_serving_ports", lambda: {9123})

    def fake_scan(ports):
        captured.append(list(ports))
        return {}

    monkeypatch.setattr(stack_commands._stack_processes, "scan_known_ports", fake_scan)

    assert stack_commands._scan_known_ports() == {}
    assert captured == [[8000, 8070, 9123]]


def test_status_attestation_detects_expected_model_basename() -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )

    assert stack_commands._status_attestation(
        info,
        alive=True,
        cmdline=["llama-server", "-m", "/canonical/current.gguf"],
    ) == "ok"


def test_status_attestation_detects_model_drift() -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )

    assert stack_commands._status_attestation(
        info,
        alive=True,
        cmdline=["llama-server", "-m", "/models/stale.gguf"],
    ) == "model-drift"


def test_status_attestation_detects_expected_mmproj_basename() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_vision",
        pid=123,
        port=8086,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="vision.log",
    )

    assert stack_commands._status_attestation(
        info,
        alive=True,
        cmdline=[
            "llama-server",
            "-m",
            "/models/current.gguf",
            "--mmproj",
            "/canonical/mmproj-model-f16.gguf",
        ],
        launch_requirements={"mmproj_path": "/models/mmproj-model-f16.gguf"},
    ) == "ok"


def test_status_attestation_detects_mmproj_drift() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_vision",
        pid=123,
        port=8086,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="vision.log",
    )

    assert stack_commands._status_attestation(
        info,
        alive=True,
        cmdline=[
            "llama-server",
            "-m",
            "/models/current.gguf",
            "--mmproj",
            "/models/stale-mmproj.gguf",
        ],
        launch_requirements={"mmproj_path": "/models/current-mmproj.gguf"},
    ) == "mmproj-drift"


def test_runtime_attestation_warnings_report_model_drift(monkeypatch) -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"frontdoor": info})
    monkeypatch.setattr(stack_commands, "_scan_known_ports", lambda: {})
    monkeypatch.setattr(stack_commands, "_stack_prior_launch_contracts", lambda: {})
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: ["llama-server", "-m", "/models/stale.gguf"],
    )

    assert stack_commands.runtime_attestation_warnings() == [
        "frontdoor pid 123 expected current.gguf; live cmdline has stale.gguf"
    ]


def test_runtime_attestation_warnings_report_unmanaged_listener(monkeypatch) -> None:
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(stack_commands, "_scan_known_ports", lambda: {8070: [123, 456]})

    assert stack_commands.runtime_attestation_warnings() == [
        "known stack port 8070 has unmanaged listener pid(s) 123,456"
    ]


def test_runtime_attestation_warnings_report_runtime_flag_drift(monkeypatch) -> None:
    info = stack_commands.ProcessInfo(
        role="worker_general",
        pid=123,
        port=8072,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="worker.log",
    )

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"worker_general": info})
    monkeypatch.setattr(stack_commands, "_scan_known_ports", lambda: {8072: [123]})
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        stack_commands,
        "_stack_prior_launch_contracts",
        lambda: {
            "worker_general": {
                "requirements": {
                    "model_path": "/models/current.gguf",
                    "draft_model_path": "/models/draft.gguf",
                },
                "runtime": {
                    "binary_path": "/opt/llama/bin/llama-server",
                    "cache": {
                        "context_tokens": 16384,
                        "slots": 1,
                        "ubatch": 512,
                        "kv_type_k": "q8_0",
                        "kv_type_v": "q8_0",
                        "no_mmap": False,
                        "mlock": True,
                    },
                    "flags": {
                        "flash_attn": True,
                        "jinja": True,
                        "spec": {
                            "enabled": True,
                            "type": "draft-mtp",
                            "draft_max": 2,
                            "draft_p_min": 0.0,
                            "threads_draft": 1,
                        },
                    },
                },
                "ports": [8072],
            }
        },
    )
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: [
            "/opt/llama/bin/llama-server",
            "-m",
            "/models/current.gguf",
            "-md",
            "/models/draft.gguf",
            "-c",
            "8192",
            "-np",
            "1",
            "-ub",
            "512",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--mlock",
            "--flash-attn",
            "off",
            "--jinja",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
            "--draft-p-min",
            "0.0",
            "--threads-draft",
            "1",
        ],
    )

    assert stack_commands.runtime_attestation_warnings() == [
        "worker_general pid 123 runtime context_tokens expected 16384; live cmdline has 8192",
        "worker_general pid 123 runtime flash_attn expected True; live cmdline has False",
    ]


def test_runtime_attestation_accepts_embedded_nextn_without_md() -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/qwen-mtp.gguf",
        log_file="frontdoor.log",
    )

    warnings = stack_commands._runtime_attestation_warnings(
        "frontdoor",
        info,
        [
            "llama-server",
            "-m",
            "/models/qwen-mtp.gguf",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "4",
        ],
        {
            "requirements": {
                "model_path": "/models/qwen-mtp.gguf",
                "draft_model_path": "/models/qwen-mtp.gguf",
            },
            "runtime": {
                "flags": {
                    "spec": {
                        "enabled": True,
                        "type": "draft-mtp",
                        "draft_max": 4,
                    }
                }
            },
        },
    )

    assert warnings == []


def test_runtime_attestation_still_warns_when_separate_draft_missing() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_general",
        pid=123,
        port=8072,
        started_at="now",
        model_path="/models/gemma.gguf",
        log_file="worker.log",
    )

    warnings = stack_commands._runtime_attestation_warnings(
        "worker_general",
        info,
        [
            "llama-server",
            "-m",
            "/models/gemma.gguf",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
        ],
        {
            "requirements": {
                "model_path": "/models/gemma.gguf",
                "draft_model_path": "/models/gemma-assistant.gguf",
            },
            "runtime": {
                "flags": {
                    "spec": {
                        "enabled": True,
                        "type": "draft-mtp",
                        "draft_max": 2,
                    }
                }
            },
        },
    )

    assert warnings == [
        (
            "worker_general pid 123 runtime draft_model_path expected "
            "gemma-assistant.gguf; live cmdline has no -md"
        )
    ]


def test_launch_contract_for_process_canonicalizes_alias_role() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_explore",
        pid=123,
        port=9999,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="worker.log",
    )
    contracts = {
        "worker_general": {
            "requirements": {"model_path": "/models/current.gguf"},
            "runtime": {"cache": {"context_tokens": 16384}},
            "ports": [8072],
        }
    }

    assert stack_commands._launch_contract_for_process(
        "worker_explore",
        info,
        contracts,
    ) == contracts["worker_general"]


def test_runtime_attestation_warnings_match_replica_by_port(monkeypatch) -> None:
    info = stack_commands.ProcessInfo(
        role="server",
        pid=123,
        port=8282,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="worker-8282.log",
    )

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"server_8282": info})
    monkeypatch.setattr(stack_commands, "_scan_known_ports", lambda: {8282: [123]})
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        stack_commands,
        "_stack_prior_launch_contracts",
        lambda: {
            "worker_general": {
                "requirements": {
                    "model_path": "/models/current.gguf",
                    "draft_model_path": "/models/draft.gguf",
                },
                "runtime": {
                    "flags": {
                        "spec": {
                            "enabled": True,
                            "type": "draft-mtp",
                            "draft_max": 2,
                        },
                    },
                },
                "ports": [8282],
            }
        },
    )
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: [
            "llama-server",
            "-m",
            "/models/current.gguf",
            "-md",
            "/models/draft.gguf",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
        ],
    )

    assert stack_commands.runtime_attestation_warnings() == []


def test_cmd_status_prints_model_attestation_warning(monkeypatch, capsys) -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )
    saved: list[dict[str, stack_commands.ProcessInfo]] = []

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"frontdoor": info})
    monkeypatch.setattr(stack_commands, "save_state", lambda state: saved.append(dict(state)))
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(stack_commands, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: ["llama-server", "-m", "/models/stale.gguf"],
    )

    rc = stack_commands.cmd_status(Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "ATTEST" in out
    assert "model-drift" in out
    assert "expected current.gguf" in out
    assert "live cmdline has stale.gguf" in out
    assert saved[-1] == {"frontdoor": info}


def test_cmd_status_prints_mmproj_attestation_warning(monkeypatch, capsys) -> None:
    info = stack_commands.ProcessInfo(
        role="worker_vision",
        pid=123,
        port=8086,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="vision.log",
    )
    saved: list[dict[str, stack_commands.ProcessInfo]] = []

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"server_8086": info})
    monkeypatch.setattr(stack_commands, "save_state", lambda state: saved.append(dict(state)))
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(stack_commands, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: [
            "llama-server",
            "-m",
            "/models/current.gguf",
            "--mmproj",
            "/models/stale-mmproj.gguf",
        ],
    )
    monkeypatch.setattr(
        stack_commands,
        "_stack_prior_launch_contracts",
        lambda: {
            "worker_vision": {
                "requirements": {"mmproj_path": "/models/current-mmproj.gguf"},
                "runtime": {},
                "ports": [8086],
            }
        },
    )

    rc = stack_commands.cmd_status(Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "mmproj-drift" in out
    assert "expected mmproj current-mmproj.gguf" in out
    assert "live cmdline has stale-mmproj.gguf" in out
    assert saved[-1] == {"server_8086": info}


def test_cmd_status_prints_runtime_contract_warning(monkeypatch, capsys) -> None:
    info = stack_commands.ProcessInfo(
        role="worker_general",
        pid=123,
        port=8072,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="worker.log",
    )
    saved: list[dict[str, stack_commands.ProcessInfo]] = []

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"worker_general": info})
    monkeypatch.setattr(stack_commands, "save_state", lambda state: saved.append(dict(state)))
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(stack_commands, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(
        stack_commands,
        "_stack_prior_launch_contracts",
        lambda: {
            "worker_general": {
                "requirements": {"model_path": "/models/current.gguf"},
                "runtime": {
                    "cache": {"context_tokens": 16384},
                    "flags": {"flash_attn": True},
                },
                "ports": [8072],
            }
        },
    )
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: [
            "llama-server",
            "-m",
            "/models/current.gguf",
            "-c",
            "8192",
            "--flash-attn",
            "off",
        ],
    )

    rc = stack_commands.cmd_status(Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "runtime context_tokens expected 16384; live cmdline has 8192" in out
    assert "runtime flash_attn expected True; live cmdline has False" in out
    assert saved[-1] == {"worker_general": info}
