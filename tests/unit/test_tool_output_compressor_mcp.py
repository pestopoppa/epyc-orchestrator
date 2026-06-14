"""Tests for the bash-compressor MCP server skeleton."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass

from src import tool_output_compressor_mcp as mod


@dataclass(frozen=True)
class _CompressionResult:
    text: str
    strategy: str
    original_chars: int
    compressed_chars: int


def _call_tool(command: str, **kwargs):
    args = {"command": command, **kwargs}
    return asyncio.run(mod.mcp.call_tool("run_bash_compressed", args))


def _result_text(result) -> str:
    return result.content[0].text


def test_run_bash_compressed_runs_allowlisted_command(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="raw output", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setenv("TOOL_COMPRESSION_MONITOR_PATH", str(tmp_path / "telemetry.jsonl"))
    monkeypatch.setattr(
        mod,
        "_compress_output",
        lambda output, command: _CompressionResult(
            text=f"compressed {command}: {output}",
            strategy="test_strategy",
            original_chars=len(output),
            compressed_chars=len(f"compressed {command}: {output}"),
        ),
    )

    result = _call_tool("git status", timeout_s=999)

    assert _result_text(result) == "compressed git status: raw output"
    assert calls == [
        (
            ["git", "status"],
            {
                "capture_output": True,
                "text": True,
                "timeout": mod.MAX_TIMEOUT_S,
                "cwd": mod.PROJECT_ROOT,
            },
        )
    ]
    assert result.meta["tool_compression"]["compressor_strategy"] == "test_strategy"


def test_run_bash_compressed_rejects_blocked_command(monkeypatch) -> None:
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess called")),
    )

    result = mod.run_bash_compressed("rm -rf .")

    assert "blocked for security" in result


def test_run_bash_compressed_rejects_unknown_git_subcommand(monkeypatch) -> None:
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess called")),
    )

    result = mod.run_bash_compressed("git reset --hard")

    assert "git reset not allowed" in result


def test_run_bash_compressed_rejects_working_dir_outside_project(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess called")),
    )

    result = mod.run_bash_compressed("pwd", working_dir=str(tmp_path))

    assert "working_dir must stay under" in result


def test_run_bash_compressed_includes_stderr_and_exit_code(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="out", stderr="err")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "_compress_output", lambda output, command: output)

    result = mod.run_bash_compressed("python -m pytest")

    assert result == "[exit code 2]\nout\n[STDERR]\nerr"


def test_run_bash_compressed_emits_telemetry(monkeypatch, tmp_path) -> None:
    telemetry = tmp_path / "tool_compression_monitor.jsonl"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="raw output", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setenv("TOOL_COMPRESSION_MONITOR_PATH", str(telemetry))
    monkeypatch.setattr(
        mod,
        "_compress_output",
        lambda output, command: _CompressionResult(
            text="short",
            strategy="unit",
            original_chars=len(output),
            compressed_chars=5,
        ),
    )

    result = _call_tool("git status")

    assert _result_text(result) == "short"
    [record] = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert record["tool"] == "run_bash_compressed"
    assert record["command"] == "git status"
    assert record["pre_bytes"] == len("raw output")
    assert record["post_bytes"] == len("short")
    assert record["compressor_strategy"] == "unit"
