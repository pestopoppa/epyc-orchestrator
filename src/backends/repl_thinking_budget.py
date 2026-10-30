"""Opt-in request-local thinking budget for REPL chat-lane turns."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from src.backends.serving_params import thinking_budget_fields

_BUDGET: ContextVar[int | None] = ContextVar("repl_thinking_budget", default=None)


def enabled() -> bool:
    from src.features import features

    return bool(getattr(features(), "repl_thinking_budget", False))


@contextmanager
def budget_scope(tokens: int | None) -> Iterator[None]:
    """Carry the caller's per-turn value; always restore the previous context."""
    if not enabled():
        yield
        return
    if tokens is not None:
        thinking_budget_fields(tokens)  # Existing contract: zero emits no budget fields.
    token = _BUDGET.set(tokens or None)
    try:
        yield
    finally:
        _BUDGET.reset(token)


def current_budget() -> int | None:
    return _BUDGET.get() if enabled() else None


def request_fields(request: Any, template_kwargs: dict | None) -> dict[str, Any]:
    """Fields for this chat body only; thinking-off calls remain thinking-off."""
    if not enabled() or (template_kwargs or {}).get("enable_thinking") is False:
        return {}
    tokens = getattr(request, "thinking_budget", None)
    return thinking_budget_fields(tokens) if tokens is not None else {}
