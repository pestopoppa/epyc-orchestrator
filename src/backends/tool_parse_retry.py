"""Retry-once on a SERVER tool-call parse failure (flag ``tool_call_parse_error_retry``).

Operator ruling: fix at the source. The server owns tool-call parsing (autokernel box
PFXC-TOOLPARSE: a strict Qwen XML parser that rejects unrecoverable malformed tool calls). The
orchestrator only reacts to the server's own failure signal; it never inspects argument values.

Contract (autokernel):

* Non-stream: HTTP 500 with body
  ``{"error": {"code": 500, "type": "tool_call_parse_error", "message": "...",
  "retryable": true, "raw_excerpt": "<=256 chars"}}``.
* Stream: one SSE ``data: {"error": {...same...}}`` and the stream closes WITHOUT ``data: [DONE]``;
  no partial tool_calls delta precedes it.
* Key on ``error.type == "tool_call_parse_error"``, NOT on the HTTP status (500 is generic).
  Every other 5xx, and any error without that type, stays on the existing infra path.

Second trigger (current servers, strict parser not deployed): native tool parser loss, i.e. the
orchestrator's own ``detect_loss`` raised ``NativeToolParseLoss``. ``truncated_tool_call_opener``
is a length-cap stop, not a model format error, so a re-prompt cannot help; it is not retried.

``raw_excerpt`` goes to telemetry only (truncated); it is never put in the correction prompt or
in the failure message (which must keep classifying as ``ErrorCategory.FORMAT``).

The retry lives in ``LlamaServerBackend._with_tool_parse_retry``: one re-prompt in the same
message history with ``CORRECTION``; a second failure raises ``ToolCallParseFailed``, a
``NativeToolParseLoss`` subclass (the fallback layer re-raises it, no model swap) whose message
classifies as FORMAT (in ``no_escalate_categories``), so the tier never escalates. On the /v1
client-tool path this runs before anything is relayed: the client gets a clean call or an error.
"""

from __future__ import annotations

import json
from typing import Any

from src.backends.native_tool_parse import NativeToolParseLoss

ERROR_TYPE = "tool_call_parse_error"
FAILURE_REASON = "tool_call_parse_error"
EXCERPT_LIMIT = 256

CORRECTION = (
    "Your previous tool call could not be parsed; "
    "re-emit it as a single well-formed tool call."
)

# Loss reasons that are a length-cap stop, not a model format error: never re-prompted.
NOT_RETRYABLE_LOSS_REASONS = frozenset({"truncated_tool_call_opener"})


class ToolCallParseFailed(NativeToolParseLoss):
    """The server's tool-call parse failure persisted after the one corrective retry."""


def parse_error_from_payload(payload: Any) -> dict[str, Any] | None:
    """Return the ``error`` object when ``payload`` is a tool_call_parse_error body/event."""
    err = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(err, dict) and err.get("type") == ERROR_TYPE:
        return err
    return None


def parse_error_from_text(text: str | None) -> dict[str, Any] | None:
    """Same as ``parse_error_from_payload`` for a raw HTTP body (any status)."""
    if not text:
        return None
    try:
        return parse_error_from_payload(json.loads(text))
    except (ValueError, TypeError):
        return None


def describe(err: dict[str, Any]) -> str:
    """Message for the failed result: type, message and a truncated raw_excerpt (telemetry/log)."""
    excerpt = str(err.get("raw_excerpt") or "")[:EXCERPT_LIMIT]
    return f"{ERROR_TYPE}: {str(err.get('message') or '')[:200]} | raw_excerpt={excerpt!r}"
