"""Detect llama-server context-size failures in any shape they arrive in.

Source of truth: the frozen production tree ``/mnt/raid0/llm/llama.cpp`` @
``ffc1bac82`` (production-consolidated-v10), read-only:

* ``tools/server/server-common.cpp:49-51`` — ``ERROR_TYPE_EXCEED_CONTEXT_SIZE``
  formats as ``{"code": 400, "message": ..., "type": "exceed_context_size_error"}``.
* ``tools/server/server-task.cpp:1531-1538`` — for that type the error object also
  carries ``n_prompt_tokens`` and ``n_ctx``.
* ``tools/server/server-context.cpp:3292-3310`` — the two per-slot admission
  messages ("input (N tokens) is larger than the max context size (M tokens).
  skipping" and "request (N tokens) exceeds the available context size (M
  tokens), try increasing it").
* ``tools/server/server-context.cpp:3759-3764`` — pool exhaustion: "Context size
  has been exceeded." sent with the DEFAULT type (``server_error``, HTTP 500) to
  every processing slot, after idle-slot purge and batch halving failed.
* ``tools/server/server-context.cpp:4362-4365`` — a non-streaming error body is
  ``{"error": <error object>}`` with the object's ``code`` as the HTTP status.
* ``tools/server/server-context.cpp:4506-4519, 4540-4550`` — in streaming mode
  the FIRST error is returned as a non-streaming response (same body, same
  status); a later error arrives in-band as ``data: {"error": <error object>}``.

Pure functions only — no I/O — so both the backend and the tests use them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.exceptions import ContextOverflowError

EXCEED_CONTEXT_SIZE_TYPE = "exceed_context_size_error"

# Message fragments, lower-cased. The typed ``exceed_context_size_error`` is
# authoritative when present; the fragments cover bodies that lost their type
# (proxies, older builds, the text/plain shape).
_TOO_LARGE_FRAGMENTS = (
    "exceeds the available context size",
    "is larger than the max context size",
)
_POOL_FRAGMENTS = ("context size has been exceeded",)
# MTP/draft speculative decoding under a FULL KV pool: once llama_decode starts
# halving the batch ("failed to find free space in the KV cache, retrying with
# smaller batch size"), post_decode meets a draft index outside the shrunken
# view and throws (server-context.cpp:4003-4009, upstream issue 24840); the
# slot fails with "got exception: speculative batch index 8 is not inside the
# current sub-batch [0, 8)" (:2798-2799, server_error/500) instead of the clean
# "Context size has been exceeded.". Measured 2026-09-24 on the v10 27B with
# --kv-unified -np 2, draft n_max 8 (np_context_kvu_study_20260924). Same cause,
# same recovery → pool_exhausted. Matched loosely: both fragments required.
_SPEC_SUBBATCH_FRAGMENTS = ("speculative batch index", "sub-batch")

_TOKENS_RE = re.compile(r"\((\d+)\s+tokens\)")


@dataclass(frozen=True)
class ContextOverflowInfo:
    """Server-reported context overflow, normalised."""

    kind: str
    message: str
    http_status: int | None = None
    n_prompt_tokens: int | None = None
    n_ctx: int | None = None
    error_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "http_status": self.http_status,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_ctx": self.n_ctx,
            "error_type": self.error_type,
        }

    def to_error(self, *, role: str = "", backend_url: str = "") -> ContextOverflowError:
        return context_overflow_error_from_info(self, role=role, backend_url=backend_url)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _classify_message(message: str, error_type: str) -> str | None:
    low = message.lower()
    if error_type == EXCEED_CONTEXT_SIZE_TYPE:
        return ContextOverflowError.REQUEST_TOO_LARGE
    if any(fragment in low for fragment in _TOO_LARGE_FRAGMENTS):
        return ContextOverflowError.REQUEST_TOO_LARGE
    if any(fragment in low for fragment in _POOL_FRAGMENTS):
        return ContextOverflowError.POOL_EXHAUSTED
    if all(fragment in low for fragment in _SPEC_SUBBATCH_FRAGMENTS):
        return ContextOverflowError.POOL_EXHAUSTED
    return None


def _from_error_object(obj: dict[str, Any], http_status: int | None) -> ContextOverflowInfo | None:
    message = obj.get("message")
    message = message if isinstance(message, str) else ""
    error_type = obj.get("type")
    error_type = error_type if isinstance(error_type, str) else ""
    kind = _classify_message(message, error_type)
    if kind is None:
        return None
    n_prompt = _as_int(obj.get("n_prompt_tokens"))
    n_ctx = _as_int(obj.get("n_ctx"))
    if kind == ContextOverflowError.REQUEST_TOO_LARGE and (n_prompt is None or n_ctx is None):
        # Recover the numbers from the message: "(N tokens) ... (M tokens)".
        numbers = [int(x) for x in _TOKENS_RE.findall(message)]
        if n_prompt is None and len(numbers) >= 1:
            n_prompt = numbers[0]
        if n_ctx is None and len(numbers) >= 2:
            n_ctx = numbers[1]
    status = http_status if http_status is not None else _as_int(obj.get("code"))
    return ContextOverflowInfo(
        kind=kind,
        message=message,
        http_status=status,
        n_prompt_tokens=n_prompt,
        n_ctx=n_ctx,
        error_type=error_type,
    )


def classify_error_payload(payload: Any, http_status: int | None = None) -> ContextOverflowInfo | None:
    """Classify a decoded error payload: ``{"error": {...}}``, a bare error object,
    an OpenAI-stream chunk carrying ``error``, or a plain string."""
    if isinstance(payload, str):
        kind = _classify_message(payload, "")
        if kind is None:
            return None
        return _from_error_object({"message": payload}, http_status)
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if isinstance(err, dict):
        return _from_error_object(err, http_status)
    if isinstance(err, str):
        return _from_error_object(
            {"message": err, "type": payload.get("type"), **{
                k: payload.get(k) for k in ("n_prompt_tokens", "n_ctx", "code") if k in payload
            }},
            http_status,
        )
    if "message" in payload:
        return _from_error_object(payload, http_status)
    return None


def classify_error_body(body: str | bytes | None, http_status: int | None = None) -> ContextOverflowInfo | None:
    """Classify a raw HTTP error body (JSON, SSE ``data:`` line(s), or text)."""
    if body is None:
        return None
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    text = body.strip()
    if not text:
        return None
    try:
        return classify_error_payload(json.loads(text), http_status)
    except (json.JSONDecodeError, ValueError):
        pass
    for line in text.splitlines():
        info = classify_sse_line(line, http_status)
        if info is not None:
            return info
    return classify_error_payload(text, http_status)


def classify_sse_line(line: str | bytes, http_status: int | None = None) -> ContextOverflowInfo | None:
    """Classify one SSE line (``data: {...}`` / ``error: {...}``); None if not an overflow."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    for prefix in ("data:", "error:"):
        if line.startswith(prefix):
            raw = line[len(prefix):].strip()
            break
    else:
        return None
    if not raw or raw == "[DONE]":
        return None
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return classify_error_payload(raw, http_status)
    if isinstance(decoded, dict) and "error" not in decoded and "message" not in decoded:
        return None
    return classify_error_payload(decoded, http_status)


def context_overflow_error_from_info(
    info: ContextOverflowInfo | dict[str, Any],
    *,
    role: str = "",
    backend_url: str = "",
) -> ContextOverflowError:
    """Build the typed error from a classified server failure."""
    if isinstance(info, dict):
        info = ContextOverflowInfo(
            kind=str(info.get("kind") or ContextOverflowError.REQUEST_TOO_LARGE),
            message=str(info.get("message") or ""),
            http_status=_as_int(info.get("http_status")),
            n_prompt_tokens=_as_int(info.get("n_prompt_tokens")),
            n_ctx=_as_int(info.get("n_ctx")),
            error_type=str(info.get("error_type") or ""),
        )
    where = f" on role {role}" if role else ""
    where += f" ({backend_url})" if backend_url else ""
    if info.kind == ContextOverflowError.REQUEST_TOO_LARGE:
        sizes = ""
        if info.n_prompt_tokens and info.n_ctx:
            sizes = f": {info.n_prompt_tokens} prompt tokens > per-request n_ctx {info.n_ctx}"
        text = f"context overflow (request too large){where}{sizes}. Server: {info.message}"
    else:
        text = (
            f"context overflow (shared KV pool exhausted){where}: the server ran out of "
            f"free KV cells under concurrent load. Server: {info.message}"
        )
    return ContextOverflowError(
        text,
        kind=info.kind,
        role=role,
        backend_url=backend_url,
        n_prompt_tokens=info.n_prompt_tokens,
        n_ctx=info.n_ctx,
        http_status=info.http_status,
        server_message=info.message,
    )
