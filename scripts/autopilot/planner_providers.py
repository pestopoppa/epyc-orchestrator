"""Planner provider adapters for AutoPilot.

The coordinator treats provider calls uniformly while preserving the existing
Claude CLI controller path. Codex is intentionally read-only: it may draft or
critique plans, but it cannot edit files during planning.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from controller_io import (
    _append_planner_archive,
    _open_planner_tap,
    invoke_controller as _invoke_claude_controller,
)

log = logging.getLogger("autopilot")


@dataclass
class PlannerProviderResult:
    provider: str
    role: str
    text: str = ""
    session_id: str | None = None
    ok: bool = False
    error: str = ""
    duration_s: float = 0.0
    raw_events: list[str] = field(default_factory=list)


class PlannerProvider(Protocol):
    name: str
    supports_resume: bool

    def invoke(
        self,
        prompt: str,
        *,
        role: str,
        session_id: str | None = None,
        timeout: int = 300,
        cwd: Path | str | None = None,
    ) -> PlannerProviderResult: ...


class ClaudePlannerProvider:
    """Claude Code CLI planner provider.

    This delegates to the existing streaming implementation so the dashboard
    tap and planner archive retain their current behavior.
    """

    name = "claude"
    supports_resume = True

    def __init__(self, invoke_fn: Any | None = None) -> None:
        self._invoke_fn = invoke_fn or _invoke_claude_controller

    def invoke(
        self,
        prompt: str,
        *,
        role: str,
        session_id: str | None = None,
        timeout: int = 300,
        cwd: Path | str | None = None,
    ) -> PlannerProviderResult:
        start = time.time()
        try:
            resume_id = session_id if role == "draft" else None
            text, next_session_id = self._invoke_fn(
                prompt,
                session_id=resume_id,
                timeout=timeout,
                cwd=cwd,
            )
            ok = bool((text or "").strip())
            return PlannerProviderResult(
                provider=self.name,
                role=role,
                text=text or "",
                session_id=next_session_id,
                ok=ok,
                error="" if ok else "empty response",
                duration_s=time.time() - start,
            )
        except Exception as exc:
            log.exception("Claude planner provider failed")
            return PlannerProviderResult(
                provider=self.name,
                role=role,
                session_id=session_id,
                ok=False,
                error=str(exc),
                duration_s=time.time() - start,
            )


class CodexPlannerProvider:
    """OpenAI Codex CLI planner provider.

    Uses ``codex exec --json`` and parses JSONL assistant-message events. The
    sandbox is read-only so the provider can inspect context and produce
    structured planner output without mutating the worktree.
    """

    name = "codex"
    supports_resume = False

    def __init__(
        self,
        binary_path: str | None = None,
        model: str | None = None,
    ) -> None:
        self._binary = binary_path or os.environ.get("AUTOPILOT_CODEX_BINARY", "codex")
        self._model = model or os.environ.get(
            "AUTOPILOT_CODEX_MODEL",
            "gpt-5.3-codex",
        )

    def invoke(
        self,
        prompt: str,
        *,
        role: str,
        session_id: str | None = None,
        timeout: int = 300,
        cwd: Path | str | None = None,
    ) -> PlannerProviderResult:
        del session_id
        start = time.time()
        tap = _open_planner_tap()
        raw_events: list[str] = []

        try:
            # Prompt is piped to `codex exec` via STDIN (positional `-`), NOT
            # handed off through a temp file. The read-only sandbox cannot open
            # an arbitrary /mnt/raid0/llm/tmp/*.txt, so the previous "Read the
            # file …" approach made Codex emit a file-read error that the
            # provider then treated as a successful (but unparseable) critique →
            # fail-open approve. Stdin is sandbox-safe and removes that failure
            # mode entirely. (codex exec: "[PROMPT] … if `-` is used,
            # instructions are read from stdin".)
            cmd = [
                self._binary,
                "exec",
                "--json",
                "-m",
                self._model,
                "-s",
                "read-only",
                "-",
            ]
            if tap is not None:
                _tap_write(
                    tap,
                    f"\n{'=' * 72}\n"
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"PLANNER provider={self.name} role={role} start\n"
                    f"prompt_chars: {len(prompt)}\n"
                    f"{'-' * 72}\n",
                )

            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd) if cwd else None,
                env=env,
            )
            try:
                stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                err = f"timeout after {timeout}s"
                _tap_write(tap, f"[TIMEOUT provider={self.name} role={role}] {err}\n")
                result = PlannerProviderResult(
                    provider=self.name,
                    role=role,
                    ok=False,
                    error=err,
                    duration_s=time.time() - start,
                    raw_events=raw_events,
                )
                _archive_codex_call(prompt, result, raw_events)
                return result

            raw_events = [line for line in (stdout or "").splitlines() if line.strip()]
            for line in raw_events[-200:]:
                _tap_write(tap, _summarize_codex_event(line) + "\n")

            text = parse_codex_jsonl(stdout or "")
            ok = proc.returncode == 0 and bool(text.strip())
            error = ""
            if proc.returncode != 0:
                error = f"codex rc={proc.returncode}: {(stderr or '')[:500]}"
            elif not ok:
                error = "empty response"

            if ok:
                _tap_write(
                    tap,
                    f"[END provider={self.name} role={role}] "
                    f"result_chars={len(text)}\n{'=' * 72}\n",
                )
            else:
                _tap_write(
                    tap,
                    f"[FAIL provider={self.name} role={role}] {error[:400]}\n{'=' * 72}\n",
                )

            result = PlannerProviderResult(
                provider=self.name,
                role=role,
                text=text,
                ok=ok,
                error=error,
                duration_s=time.time() - start,
                raw_events=raw_events,
            )
            _archive_codex_call(prompt, result, raw_events)
            return result
        except FileNotFoundError:
            err = f"Codex CLI not found: {self._binary}"
            log.error(err)
            result = PlannerProviderResult(
                provider=self.name,
                role=role,
                ok=False,
                error=err,
                duration_s=time.time() - start,
                raw_events=raw_events,
            )
            _archive_codex_call(prompt, result, raw_events)
            return result
        except Exception as exc:
            log.exception("Codex planner provider failed")
            result = PlannerProviderResult(
                provider=self.name,
                role=role,
                ok=False,
                error=str(exc),
                duration_s=time.time() - start,
                raw_events=raw_events,
            )
            _archive_codex_call(prompt, result, raw_events)
            return result
        finally:
            # Prompt is piped via stdin now (no temp file to clean up).
            if tap is not None:
                try:
                    tap.close()
                except Exception:
                    pass


def parse_codex_jsonl(output: str) -> str:
    """Extract assistant text from Codex ``--json`` JSONL output."""
    parts: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")
        if event_type == "item.completed":
            item = event.get("item", {})
            parts.extend(_message_text_parts(item))
        elif event_type in {"agent_message", "message"}:
            parts.extend(_message_text_parts(event))
        elif event_type == "response.completed":
            response = event.get("response", {})
            parts.extend(_message_text_parts(response))

    return "".join(parts) if parts else output


def get_planner_provider(name: str) -> PlannerProvider:
    normalized = (name or "").strip().lower()
    if normalized == "claude":
        return ClaudePlannerProvider()
    if normalized == "codex":
        return CodexPlannerProvider()
    raise ValueError(f"Unknown planner provider: {name}")


def _message_text_parts(obj: dict[str, Any]) -> list[str]:
    if obj.get("type") == "agent_message" and isinstance(obj.get("text"), str):
        return [obj["text"]]
    if isinstance(obj.get("text"), str):
        return [obj["text"]]

    content = obj.get("content")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    out.append(text)
        return out
    return []


def _summarize_codex_event(line: str) -> str:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return f"[codex] {line[:500]}"

    event_type = event.get("type", "?")
    if event_type == "item.completed":
        item = event.get("item", {})
        item_type = item.get("type", "?")
        text = "".join(_message_text_parts(item)).strip()
        if text:
            return f"[codex:item.completed:{item_type}] {text[:1200]}"
        return f"[codex:item.completed:{item_type}]"
    return f"[codex:{event_type}] {json.dumps(event, separators=(',', ':'))[:500]}"


def _tap_write(tap: Any, text: str) -> None:
    if tap is None:
        return
    try:
        tap.write(text)
        tap.flush()
    except Exception:
        pass


def _archive_codex_call(
    prompt: str,
    result: PlannerProviderResult,
    raw_events: list[str],
) -> None:
    import hashlib

    _append_planner_archive(
        {
            "ts": time.time(),
            "ts_iso": datetime.now().isoformat(timespec="seconds"),
            "provider": result.provider,
            "role": result.role,
            "duration_s": result.duration_s,
            "ok": result.ok,
            "error": result.error,
            "prompt_chars": len(prompt),
            "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "result_chars": len(result.text),
            "result_preview": result.text[:500],
            "n_events": len(raw_events),
            "events": raw_events[-200:],
        }
    )
