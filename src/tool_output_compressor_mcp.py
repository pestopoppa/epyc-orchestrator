#!/usr/bin/env python3
"""MCP surface for safe, compressed bash-style command output."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_COMPRESSOR_DIR = Path("/mnt/raid0/llm/epyc-root/scripts/utils")

MAX_TIMEOUT_S = 120
MAX_RAW_OUTPUT_CHARS = 200_000
MAX_RETURN_CHARS = 80_000
TELEMETRY_PATH = Path("/mnt/raid0/llm/epyc-root/logs/tool_compression_monitor.jsonl")

SAFE_COMMANDS = {
    "ls",
    "find",
    "wc",
    "du",
    "file",
    "head",
    "tail",
    "cat",
    "grep",
    "awk",
    "sed",
    "sort",
    "uniq",
    "tr",
    "cut",
    "git",
    "pwd",
    "whoami",
    "date",
    "echo",
    "printf",
    "python",
    "python3",
}

BLOCKED_COMMANDS = {
    "rm",
    "mv",
    "cp",
    "chmod",
    "chown",
    "chgrp",
    "dd",
    "mkfs",
    "mount",
    "umount",
    "kill",
    "pkill",
    "killall",
    "sudo",
    "su",
    "bash",
    "sh",
    "zsh",
    "csh",
    "wget",
    "curl",
    "nc",
    "netcat",
    "ncat",
}

SAFE_GIT_SUBCOMMANDS = {"status", "log", "diff", "branch", "show", "ls-files", "rev-parse"}

mcp = FastMCP("bash-compressor")


def _error(message: str) -> str:
    return f"[ERROR: {message}]"


def _parse_command(command: str) -> tuple[list[str] | None, str | None]:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return None, f"Invalid command syntax: {exc}"
    if not parts:
        return None, "Empty command"

    base_cmd = Path(parts[0]).name
    if base_cmd in BLOCKED_COMMANDS:
        return None, f"Command '{base_cmd}' is blocked for security"
    if base_cmd not in SAFE_COMMANDS:
        return None, f"Command '{base_cmd}' not in allowlist: {sorted(SAFE_COMMANDS)}"
    if base_cmd == "git" and len(parts) > 1 and parts[1] not in SAFE_GIT_SUBCOMMANDS:
        return None, f"git {parts[1]} not allowed. Safe: {sorted(SAFE_GIT_SUBCOMMANDS)}"
    return parts, None


def _resolve_working_dir(working_dir: str) -> Path:
    if not working_dir.strip():
        return PROJECT_ROOT
    resolved = Path(working_dir).expanduser().resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"working_dir must stay under {root}")
    if not resolved.is_dir():
        raise ValueError(f"working_dir is not a directory: {resolved}")
    return resolved


def _compress_output(output: str, command: str):
    if str(ROOT_COMPRESSOR_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_COMPRESSOR_DIR))
    try:
        from compress_tool_output import compress_tool_output_with_metadata
    except Exception:
        return None
    return compress_tool_output_with_metadata(output, command)


def _truncate(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[... truncated {label} at {max_chars} chars]"


def _text_content(result: ToolResult) -> str | None:
    if len(result.content) != 1:
        return None
    block = result.content[0]
    text = getattr(block, "text", None)
    return text if isinstance(text, str) else None


def _with_text_content(result: ToolResult, text: str, metadata: dict) -> ToolResult:
    block = result.content[0]
    content = [block.model_copy(update={"text": text})]
    structured = result.structured_content
    if isinstance(structured, dict) and isinstance(structured.get("result"), str):
        structured = {**structured, "result": text}
    meta = dict(result.meta or {})
    meta["tool_compression"] = metadata
    return ToolResult(content=content, structured_content=structured, meta=meta)


def _telemetry_path() -> Path:
    import os

    return Path(os.environ.get("TOOL_COMPRESSION_MONITOR_PATH", str(TELEMETRY_PATH)))


def _write_telemetry(record: dict) -> None:
    try:
        path = _telemetry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        # Telemetry must never make a tool result fail.
        return


class CompressorMiddleware(Middleware):
    """Compress bash tool results after command execution."""

    async def on_call_tool(self, context, call_next):
        result = await call_next(context)
        message = context.message
        if getattr(message, "name", "") != "run_bash_compressed":
            return result
        if not isinstance(result, ToolResult):
            return result

        arguments = getattr(message, "arguments", None) or {}
        command = str(arguments.get("command") or "")
        raw_text = _text_content(result)
        if raw_text is None:
            return result

        compressed = _compress_output(raw_text, command)
        if compressed is None:
            post_text = _truncate(raw_text, MAX_RETURN_CHARS, "after compression")
            strategy = "compressor_unavailable"
        else:
            post_text = _truncate(compressed.text, MAX_RETURN_CHARS, "after compression")
            strategy = compressed.strategy
        post_bytes = len(post_text.encode("utf-8"))
        pre_bytes = len(raw_text.encode("utf-8"))
        ratio = round(post_bytes / max(pre_bytes, 1), 4)
        metadata = {
            "command": command[:200],
            "pre_bytes": pre_bytes,
            "post_bytes": post_bytes,
            "compression_ratio": ratio,
            "compressor_strategy": strategy,
        }
        _write_telemetry({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": "run_bash_compressed",
            **metadata,
        })
        return _with_text_content(result, post_text, metadata)


@mcp.tool()
def run_bash_compressed(command: str, timeout_s: int = 60, working_dir: str = "") -> str:
    """Run an allowlisted command; middleware compresses stdout/stderr on MCP calls."""
    parts, parse_error = _parse_command(command)
    if parse_error or parts is None:
        return _error(parse_error or "Invalid command")

    try:
        timeout = min(max(int(timeout_s), 1), MAX_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout = 60

    try:
        cwd = _resolve_working_dir(working_dir)
    except ValueError as exc:
        return _error(str(exc))

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return _error(f"Command timed out after {timeout}s")
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}")

    output = result.stdout
    if result.stderr:
        output += "\n[STDERR]\n" + result.stderr
    if result.returncode:
        output = f"[exit code {result.returncode}]\n{output}"

    output = _truncate(output, MAX_RAW_OUTPUT_CHARS, "before compression")
    return output


mcp.add_middleware(CompressorMiddleware())


if __name__ == "__main__":
    mcp.run(transport="stdio")
