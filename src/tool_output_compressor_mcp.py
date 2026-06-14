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
TOP_UP_LOOKBACK_TURNS = 3
TOP_UP_REASON_REPEAT = "repeat_command_within_3_turns"
TOP_UP_REASON_LIST_VIEW = "file_view_after_listing"

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


def _session_journal_path() -> Path:
    import os

    return Path(os.environ.get("TOOL_COMPRESSION_SESSION_PATH", str(_telemetry_path())))


def _current_session_id() -> str | None:
    import os

    return os.environ.get("TOOL_COMPRESSION_SESSION_ID")


def _normalize_for_followup(command: str) -> str:
    tokens = _tokenize_command(command)
    if not tokens:
        return ""
    base = Path(tokens[0]).name
    return " ".join([base] + tokens[1:])


def _tokenize_command(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command.split()
    return tokens


def _iter_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _session_records(path: Path, session_id: str | None) -> list[dict]:
    records = [
        rec
        for rec in _iter_jsonl_records(path)
        if rec.get("tool") == "run_bash_compressed" and isinstance(rec.get("command"), str)
    ]

    if session_id is None:
        return records
    return [rec for rec in records if rec.get("session_id") == session_id]


def _infer_top_up_followup(command: str, session_history: list[dict]) -> dict[str, object]:
    normalized = _normalize_for_followup(command)
    if not normalized:
        return {
            "top_up_candidate": False,
            "followup_distance": None,
            "followup_reason": None,
            "next_turn_followup_command": None,
            "followup_source_command": None,
        }

    current_tokens = _tokenize_command(normalized)
    for distance, previous in enumerate(
        reversed(session_history[-TOP_UP_LOOKBACK_TURNS:]),
        start=1,
    ):
        previous_command = str(previous.get("command", ""))
        previous_normalized = _normalize_for_followup(previous_command)
        if not previous_normalized:
            continue

        previous_tokens = _tokenize_command(previous_normalized)

        reason: str | None = None
        if normalized == previous_normalized:
            reason = TOP_UP_REASON_REPEAT
        elif _file_followup_after_ls(current_tokens, previous_tokens):
            reason = TOP_UP_REASON_LIST_VIEW

        if reason is not None:
            return {
                "top_up_candidate": True,
                "followup_distance": distance,
                "followup_reason": reason,
                "next_turn_followup_command": command,
                "followup_source_command": previous_command,
            }

    return {
        "top_up_candidate": False,
        "followup_distance": None,
        "followup_reason": None,
        "next_turn_followup_command": None,
        "followup_source_command": None,
    }


def _file_followup_after_ls(current_tokens: list[str], previous_tokens: list[str]) -> bool:
    if len(current_tokens) < 2 or current_tokens[0] not in {"cat", "head", "tail"}:
        return False
    if not previous_tokens or previous_tokens[0] != "ls":
        return False

    scope = _ls_scope(previous_tokens[1:])
    if not scope:
        return False

    target = _file_view_target(current_tokens)
    if target is None:
        return False
    if scope == ".":
        return "/" not in target and not target.startswith("-")

    return target == scope or target.startswith(f"{scope}/")


def _file_view_target(tokens: list[str]) -> str | None:
    args = tokens[1:]
    skip_next = False
    options_with_values = {"-n", "-c", "--lines", "--bytes"}
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            continue
        if arg in options_with_values:
            skip_next = True
            continue
        if arg.startswith(("--lines=", "--bytes=")):
            continue
        if arg.startswith("-"):
            continue
        return arg
    return None


def _ls_scope(args: list[str]) -> str:
    non_flag_args = [arg for arg in args if not arg.startswith("-") and arg != "--"]
    if not non_flag_args:
        return "."
    return non_flag_args[0]


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


def _extract_session_id(message) -> str | None:
    session_from_env = _current_session_id()
    if session_from_env:
        return session_from_env

    message_meta = getattr(message, "meta", None) or {}
    if isinstance(message_meta, dict):
        return message_meta.get("session_id") or message_meta.get("sessionId")
    return None


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

        session_id = _extract_session_id(message)
        journal_path = _session_journal_path()
        session_history = _session_records(journal_path, session_id)
        followup = _infer_top_up_followup(command, session_history)

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
            **followup,
            "session_id": session_id,
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
