"""Tests for the bash-compressor MCP server skeleton."""

from __future__ import annotations

import subprocess

from src import tool_output_compressor_mcp as mod


def test_run_bash_compressed_runs_allowlisted_command(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="raw output", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "_compress_output", lambda output, command: f"compressed {command}: {output}")

    result = mod.run_bash_compressed("git status", timeout_s=999)

    assert result == "compressed git status: raw output"
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
