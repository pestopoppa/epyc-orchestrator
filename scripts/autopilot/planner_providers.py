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
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from controller_io import (
    _append_planner_archive,
    _open_planner_tap,
    invoke_controller as _invoke_claude_controller,
)

log = logging.getLogger("autopilot")

_LOCAL_TRANSIENT_HTTP_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)

_LOCAL_ACTION_OUTPUT_CONTRACT = """\
CRITICAL OUTPUT CONTRACT FOR THIS LOCAL AUTOPILOT DRAFT:
- Return exactly one fenced ```json:autopilot_actions block followed by one fenced ```json:autopilot_rationale block.
- Do not write analysis, caveats, summaries, or prose before the first fence.
- Use only action types and fields explicitly allowed in the prompt.
- Do not invent flags, surfaces, suites, dependencies, or evidence.
- If no high-confidence action is justified, emit the safest valid fallback action from the prompt.
"""


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
    supports_resume = False

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
            resume_id = session_id if role == "draft" and self.supports_resume else None
            text, next_session_id = self._invoke_fn(
                prompt,
                session_id=resume_id,
                timeout=timeout,
                cwd=cwd,
            )
            ok = bool((text or "").strip())
            stored_session_id = (
                next_session_id if role == "draft" and self.supports_resume else None
            )
            return PlannerProviderResult(
                provider=self.name,
                role=role,
                text=text or "",
                session_id=stored_session_id,
                ok=ok,
                error="" if ok else "empty response",
                duration_s=time.time() - start,
            )
        except Exception as exc:
            log.exception("Claude planner provider failed")
            return PlannerProviderResult(
                provider=self.name,
                role=role,
                session_id=session_id if self.supports_resume else None,
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
        name: str | None = None,
    ) -> None:
        self._binary = binary_path or os.environ.get("AUTOPILOT_CODEX_BINARY", "codex")
        self._model = model or os.environ.get("AUTOPILOT_CODEX_MODEL")
        self.name = name or self.name

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
                "-s",
                "read-only",
                "-",
            ]
            if self._model:
                cmd[3:3] = ["-m", self._model]
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


class LocalPlannerProvider:
    """OpenAI-compatible local planner drafter.

    This is intended for routine draft generation against the live local stack
    while the coordinator keeps a cloud critic as the binding reviewer. It calls
    the orchestrator's OpenAI-compatible endpoint with REPL disabled so planner
    drafting remains a text-only inference request, not an execution surface.
    """

    name = "local"
    supports_resume = False

    def __init__(
        self,
        *,
        url: str | None = None,
        role: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        name: str | None = None,
    ) -> None:
        self._url = (
            url
            or os.environ.get("AUTOPILOT_LOCAL_PLANNER_URL")
            or "http://127.0.0.1:8000/v1/chat/completions"
        )
        self._role = role or os.environ.get("AUTOPILOT_LOCAL_PLANNER_ROLE") or "frontdoor"
        self._model = model or os.environ.get("AUTOPILOT_LOCAL_PLANNER_MODEL") or self._role
        self._temperature = (
            float(temperature)
            if temperature is not None
            else _env_float("AUTOPILOT_LOCAL_PLANNER_TEMPERATURE", 0.0)
        )
        self._top_p = (
            float(top_p)
            if top_p is not None
            else _env_optional_float("AUTOPILOT_LOCAL_PLANNER_TOP_P")
        )
        self._top_k = (
            int(top_k) if top_k is not None else _env_optional_int("AUTOPILOT_LOCAL_PLANNER_TOP_K")
        )
        self._seed = (
            int(seed) if seed is not None else _env_optional_int("AUTOPILOT_LOCAL_PLANNER_SEED")
        )
        self._max_tokens = (
            int(max_tokens)
            if max_tokens is not None
            else _env_int("AUTOPILOT_LOCAL_PLANNER_MAX_TOKENS", 2048)
        )
        self.name = name or self.name

    def invoke(
        self,
        prompt: str,
        *,
        role: str,
        session_id: str | None = None,
        timeout: int = 300,
        cwd: Path | str | None = None,
    ) -> PlannerProviderResult:
        del session_id, cwd
        start = time.time()
        tap = _open_planner_tap()
        payload = self._payload(prompt, planner_role=role)
        try:
            if tap is not None:
                _tap_write(
                    tap,
                    f"\n{'=' * 72}\n"
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"PLANNER provider={self.name} role={role} start\n"
                    f"url: {self._url}\n"
                    f"local_role: {self._role}\n"
                    f"prompt_chars: {len(prompt)}\n"
                    f"{'-' * 72}\n",
                )
            with httpx.Client(timeout=timeout) as client:
                data = _post_local_json_with_retries(
                    client,
                    self._url,
                    payload,
                    provider=self.name,
                    role=role,
                    tap=tap,
                )
            text = parse_openai_chat_response(data)
            ok = bool(text.strip())
            error = "" if ok else "empty response"
            if ok:
                _tap_write(
                    tap,
                    f"[local:result:{self.name}:{role}] {text[:4000]}\n"
                    f"[END provider={self.name} role={role}] "
                    f"result_chars={len(text)}\n{'=' * 72}\n",
                )
            else:
                _tap_write(
                    tap,
                    f"[FAIL provider={self.name} role={role}] {error}\n{'=' * 72}\n",
                )
            result = PlannerProviderResult(
                provider=self.name,
                role=role,
                text=text,
                ok=ok,
                error=error,
                duration_s=time.time() - start,
                raw_events=[json.dumps(data, default=str)[:4000]],
            )
            _archive_local_call(prompt, payload, result, data, url=self._url)
            return result
        except Exception as exc:
            log.exception("Local planner provider failed")
            result = PlannerProviderResult(
                provider=self.name,
                role=role,
                ok=False,
                error=str(exc),
                duration_s=time.time() - start,
                raw_events=[],
            )
            _tap_write(
                tap,
                f"[FAIL provider={self.name} role={role}] {str(exc)[:400]}\n{'=' * 72}\n",
            )
            _archive_local_call(prompt, payload, result, None, url=self._url)
            return result
        finally:
            if tap is not None:
                try:
                    tap.close()
                except Exception:
                    pass

    def _payload(self, prompt: str, *, planner_role: str | None = None) -> dict[str, Any]:
        content = _local_planner_prompt(prompt, planner_role=planner_role)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
            "x_orchestrator_role": self._role,
            "x_disable_repl": True,
        }
        if self._top_p is not None:
            payload["top_p"] = self._top_p
        if self._top_k is not None:
            payload["top_k"] = self._top_k
        if self._seed is not None:
            payload["seed"] = self._seed
        return payload


class LocalChatPlannerProvider:
    """OpenAI-compatible local planner via the orchestrator ``/chat`` endpoint.

    Draft calls can force the ingest lane to keep controller prompts out of the
    chat router while preserving the regular /chat transport.
    """

    name = "local_chat"
    supports_resume = False

    def __init__(
        self,
        *,
        url: str | None = None,
        max_tokens: int | None = None,
        name: str | None = None,
    ) -> None:
        self._url = (
            url
            or os.environ.get("AUTOPILOT_LOCAL_CHAT_PLANNER_URL")
            or "http://127.0.0.1:8000/chat"
        )
        self._max_tokens = (
            int(max_tokens)
            if max_tokens is not None
            else _env_int("AUTOPILOT_LOCAL_PLANNER_MAX_TOKENS", 2048)
        )
        self._force_role = _env_optional_force_role(
            "AUTOPILOT_LOCAL_CHAT_PLANNER_FORCE_ROLE",
            default="ingest_long_context",
        )
        self.name = name or self.name

    def invoke(
        self,
        prompt: str,
        *,
        role: str,
        session_id: str | None = None,
        timeout: int = 300,
        cwd: Path | str | None = None,
    ) -> PlannerProviderResult:
        del session_id, cwd
        start = time.time()
        tap = _open_planner_tap()
        payload = self._payload(prompt, planner_role=role)

        try:
            if tap is not None:
                _tap_write(
                    tap,
                    f"\n{'=' * 72}\n"
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"PLANNER provider={self.name} role={role} start\n"
                    f"url: {self._url}\n"
                    f"request_id: {payload['request_id']}\n"
                    f"prompt_chars: {len(prompt)}\n"
                    f"{'-' * 72}\n",
                )
            with httpx.Client(timeout=timeout) as client:
                data = _post_local_json_with_retries(
                    client,
                    self._url,
                    payload,
                    provider=self.name,
                    role=role,
                    tap=tap,
                )
            text = str(data.get("answer") or "")
            ok = bool(text.strip())
            error = "" if ok else "empty response"
            if ok:
                _tap_write(
                    tap,
                    f"[local:result:{self.name}:{role}] {text[:4000]}\n"
                    f"[END provider={self.name} role={role}] "
                    f"result_chars={len(text)}\n{'=' * 72}\n",
                )
            else:
                _tap_write(
                    tap,
                    f"[FAIL provider={self.name} role={role}] {error}\n{'=' * 72}\n",
                )
            result = PlannerProviderResult(
                provider=self.name,
                role=role,
                text=text,
                ok=ok,
                error=error,
                duration_s=time.time() - start,
                raw_events=[json.dumps(data, default=str)[:4000]],
            )
            _archive_local_chat_call(prompt, payload, result, data, url=self._url)
            return result
        except Exception as exc:
            log.exception("Local chat planner provider failed")
            result = PlannerProviderResult(
                provider=self.name,
                role=role,
                ok=False,
                error=str(exc),
                duration_s=time.time() - start,
                raw_events=[],
            )
            _tap_write(
                tap,
                f"[FAIL provider={self.name} role={role}] {str(exc)[:400]}\n{'=' * 72}\n",
            )
            _archive_local_chat_call(prompt, payload, result, None, url=self._url)
            return result
        finally:
            if tap is not None:
                try:
                    tap.close()
                except Exception:
                    pass

    def _payload(self, prompt: str, *, planner_role: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "mock_mode": False,
            "real_mode": True,
            "max_turns": 1,
            "max_tokens": self._max_tokens,
            "request_priority": "background",
            "workload_class": "campaign",
            "request_id": f"planner-local-chat-{uuid.uuid4().hex[:8]}",
        }
        if planner_role == "draft" and self._force_role:
            payload["force_role"] = self._force_role
        return payload


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


def parse_openai_chat_response(data: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible chat response."""
    choices = data.get("choices")
    if isinstance(choices, list):
        parts: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
                    continue
                parts.extend(_message_text_parts(message))
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    parts.append(content)
        if parts:
            return "".join(parts)
    for key in ("content", "text", "output_text", "response"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return ""


def _local_planner_prompt(prompt: str, *, planner_role: str | None = None) -> str:
    if planner_role != "draft":
        return prompt
    if not _env_bool("AUTOPILOT_LOCAL_PLANNER_ACTION_CONTRACT", True):
        return prompt
    return f"{_LOCAL_ACTION_OUTPUT_CONTRACT}\n{prompt}\n\n{_LOCAL_ACTION_OUTPUT_CONTRACT}"


def get_planner_provider(name: str) -> PlannerProvider:
    normalized = (name or "").strip().lower()
    if normalized == "claude":
        return ClaudePlannerProvider()
    if normalized == "codex":
        return CodexPlannerProvider()
    if normalized in {"codex_critic", "codex-critic", "codex_reviewer", "codex-reviewer"}:
        return CodexPlannerProvider(name="codex_critic")
    if normalized in {"local", "local_frontdoor", "frontdoor_local"}:
        return LocalPlannerProvider(name="local")
    if normalized in {"local_worker", "local_worker_general", "worker_general_local"}:
        return LocalPlannerProvider(
            role="worker_general",
            model="worker_general",
            name="local_worker",
        )
    if normalized in {"local_ingest", "local_ingest_long_context", "ingest_local"}:
        return LocalPlannerProvider(
            role="ingest_long_context",
            model="ingest_long_context",
            name="local_ingest",
        )
    if normalized in {"local_chat", "local_chat_planner", "chat_local"}:
        return LocalChatPlannerProvider(name="local_chat")
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


def _post_local_json_with_retries(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    provider: str,
    role: str,
    tap: Any,
) -> dict[str, Any]:
    attempts = max(1, _env_int("AUTOPILOT_LOCAL_PLANNER_HTTP_ATTEMPTS", 4))
    base_sleep_s = max(0.0, _env_float("AUTOPILOT_LOCAL_PLANNER_RETRY_SLEEP_S", 2.0))
    max_sleep_s = max(base_sleep_s, _env_float("AUTOPILOT_LOCAL_PLANNER_RETRY_MAX_SLEEP_S", 8.0))
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            payload_error = _local_planner_payload_error(data)
            if not payload_error:
                return data
            if not _local_planner_payload_error_retryable(payload_error):
                raise RuntimeError(payload_error)
            if attempt >= attempts:
                raise RuntimeError(payload_error)
            sleep_s = min(max_sleep_s, base_sleep_s * attempt)
            _tap_write(
                tap,
                f"[RETRY provider={provider} role={role}] "
                f"attempt={attempt}/{attempts} local_error={payload_error[:180]}; "
                f"sleep={sleep_s:.1f}s\n",
            )
            if sleep_s > 0:
                time.sleep(sleep_s)
        except _LOCAL_TRANSIENT_HTTP_ERRORS as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            sleep_s = min(max_sleep_s, base_sleep_s * attempt)
            _tap_write(
                tap,
                f"[RETRY provider={provider} role={role}] "
                f"attempt={attempt}/{attempts} transient={type(exc).__name__}: "
                f"{str(exc)[:180]}; sleep={sleep_s:.1f}s\n",
            )
            if sleep_s > 0:
                time.sleep(sleep_s)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status not in {502, 503, 504} or attempt >= attempts:
                raise
            last_exc = exc
            sleep_s = min(max_sleep_s, base_sleep_s * attempt)
            _tap_write(
                tap,
                f"[RETRY provider={provider} role={role}] "
                f"attempt={attempt}/{attempts} status={status}; sleep={sleep_s:.1f}s\n",
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("local planner HTTP retry loop exited without response")


def _local_planner_payload_error(data: dict[str, Any]) -> str:
    text = parse_openai_chat_response(data)
    if not text and isinstance(data.get("answer"), str):
        text = str(data["answer"])
    stripped = text.strip()
    if stripped.startswith("[ERROR"):
        return stripped
    if stripped.startswith("[MOCK]"):
        return stripped
    return ""


def _local_planner_payload_error_retryable(error: str) -> bool:
    """Return whether a local planner error payload can clear by waiting.

    Admission pressure and mock-mode races are transient. Backend 400s from the
    role server are deterministic for the same prompt/role pair, so retrying
    only burns the planner window before falling back.
    """
    lowered = str(error or "").lower()
    if "400 bad request" in lowered:
        return False
    if "context" in lowered and ("exceed" in lowered or "too long" in lowered):
        return False
    return True


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


def _archive_local_call(
    prompt: str,
    payload: dict[str, Any],
    result: PlannerProviderResult,
    response_data: dict[str, Any] | None,
    *,
    url: str,
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
            "local_planner": {
                "url": url,
                "model": payload.get("model"),
                "x_orchestrator_role": payload.get("x_orchestrator_role"),
                "x_disable_repl": payload.get("x_disable_repl"),
                "temperature": payload.get("temperature"),
                "top_p": payload.get("top_p"),
                "top_k": payload.get("top_k"),
                "seed": payload.get("seed"),
                "max_tokens": payload.get("max_tokens"),
            },
            "response_preview": (
                json.dumps(response_data, default=str)[:1000] if response_data is not None else ""
            ),
        }
    )


def _archive_local_chat_call(
    prompt: str,
    payload: dict[str, Any],
    result: PlannerProviderResult,
    response_data: dict[str, Any] | None,
    *,
    url: str,
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
            "local_chat_planner": {
                "url": url,
                "request_id": payload.get("request_id"),
                "request_priority": payload.get("request_priority"),
                "workload_class": payload.get("workload_class"),
                "mock_mode": payload.get("mock_mode"),
                "real_mode": payload.get("real_mode"),
                "max_turns": payload.get("max_turns"),
                "max_tokens": payload.get("max_tokens"),
            },
            "response_preview": (
                json.dumps(response_data, default=str)[:1000] if response_data is not None else ""
            ),
        }
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_optional_force_role(name: str, *, default: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip()
    if not normalized or normalized.lower() in {"0", "false", "off"}:
        return None
    return normalized
