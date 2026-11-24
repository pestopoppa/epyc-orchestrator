"""AEI-CHAT-FINISH-REASON: carry the backend completion_reason onto the /chat response.

A length-cut answer used to be indistinguishable from a complete one. When the
``chat_finish_reason`` flag is on, ``finish_reason_fields`` returns the
``finish_reason`` / ``truncated`` keyword arguments for ``ChatResponse``; flag off, or no
inference call observed in this context, it returns ``{}`` so the response is byte-identical
(both fields are omitted when unset).
"""

from __future__ import annotations

from typing import Any

# Backend completion reasons that mean the answer was cut off, not finished.
TRUNCATING_REASONS = frozenset({"length", "context_limit", "context_overflow"})


def is_truncating(reason: str | None) -> bool:
    return str(reason or "") in TRUNCATING_REASONS


def finish_reason_fields(primitives: Any) -> dict[str, Any]:
    """``{"finish_reason", "truncated"}`` for the last call of this context, or ``{}``."""
    from src.features import features

    if not features().chat_finish_reason:
        return {}
    getter = getattr(primitives, "get_last_inference_meta", None)
    meta = getter() if callable(getter) else None
    if not isinstance(meta, dict):
        return {}
    reason = str(meta.get("completion_reason") or "")
    if not reason:
        return {}
    return {"finish_reason": reason, "truncated": is_truncating(reason)}
