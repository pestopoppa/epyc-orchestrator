"""Opt-in request-local thinking budget for REPL chat-lane turns."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from src.backends.serving_params import thinking_budget_fields

_BUDGET: ContextVar[int | None] = ContextVar("repl_thinking_budget", default=None)
_GENERATION_CAP: ContextVar[int | None] = ContextVar("repl_generation_cap", default=None)


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


def current_generation_cap() -> int | None:
    return _GENERATION_CAP.get() if enabled() else None


def cap_generation_tokens(n_tokens: int) -> int:
    """Apply the optional routed cap while preserving zero-token prefill requests."""
    cap = current_generation_cap()
    if cap is None or n_tokens == 0:
        return n_tokens
    return cap if n_tokens < 0 else min(n_tokens, cap)


def effective_request_budget(request: Any, routing: Any = None) -> int | None:
    """Resolve explicit caller budget before the optional router-selected budget.

    Zero retains the established inherit semantics; it does not disable a configured
    route budget. The feature gate remains enforced by ``budget_scope``.
    """
    explicit = getattr(request, "thinking_budget", 0)
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return explicit
    routed = getattr(routing, "reasoning_budget", None)
    return routed if isinstance(routed, int) and not isinstance(routed, bool) and routed > 0 else None


def routed_budget_for_band(
    difficulty_band: str | None,
    *,
    easy: int | None,
    hard: int | None,
) -> int | None:
    """Apply only explicitly configured values for known existing difficulty bands."""
    band = str(difficulty_band or "").strip().lower()
    if band == "easy":
        return easy
    if band == "hard":
        return hard
    return None


def routed_generation_cap_for_band(
    difficulty_band: str | None,
    *,
    easy: int | None,
    hard: int | None,
) -> int | None:
    """Return the configured cap after limiting it to routing's admitted decode reserve."""
    requested = routed_budget_for_band(difficulty_band, easy=easy, hard=hard)
    if requested is None:
        return None
    from src.api.routes.chat_pipeline.routing_decision import DEFAULT_ROUTING_DECODE_RESERVE_TOKENS

    return min(int(requested), DEFAULT_ROUTING_DECODE_RESERVE_TOKENS)


def routing_budget_outputs(
    request: Any, difficulty_band: str | None, config: Any
) -> tuple[int | None, int | None]:
    """Resolve the two optional policy outputs after routing assessment."""
    reasoning = routed_budget_for_band(
        difficulty_band,
        easy=getattr(config, "reasoning_budget_easy", None),
        hard=getattr(config, "reasoning_budget_hard", None),
    )
    if isinstance(reasoning, int) and not isinstance(reasoning, bool) and reasoning > 0:
        from src.api.routes.chat_pipeline.routing_decision import DEFAULT_ROUTING_DECODE_RESERVE_TOKENS

        request_cap = getattr(request, "max_tokens", None)
        admitted_cap = (
            int(request_cap)
            if isinstance(request_cap, int) and not isinstance(request_cap, bool) and request_cap > 0
            else DEFAULT_ROUTING_DECODE_RESERVE_TOKENS
        )
        reasoning = min(reasoning, admitted_cap)
    generation = None
    if getattr(request, "max_tokens", None) is None:
        generation = routed_generation_cap_for_band(
            difficulty_band,
            easy=getattr(config, "generation_max_tokens_easy", None),
            hard=getattr(config, "generation_max_tokens_hard", None),
        )
    return reasoning, generation


def active_routed_policy_applies(request: Any, config: Any) -> bool:
    """Whether this request can use configured policy without overriding caller caps."""
    if not enabled():
        return False
    caller_thinking = getattr(request, "thinking_budget", 0)
    needs_reasoning = not (isinstance(caller_thinking, int) and caller_thinking > 0) and any(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (
            getattr(config, "reasoning_budget_easy", None),
            getattr(config, "reasoning_budget_hard", None),
        )
    )
    needs_generation = getattr(request, "max_tokens", None) is None and any(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (
            getattr(config, "generation_max_tokens_easy", None),
            getattr(config, "generation_max_tokens_hard", None),
        )
    )
    return needs_reasoning or needs_generation


def assess_request_budgets(
    request: Any, role: str, task_id: str, config: Any
) -> tuple[int | None, int | None]:
    """Assess difficulty only when an enabled configured policy can affect the request."""
    if not active_routed_policy_applies(request, config):
        return None, None
    from src.api.routes.chat_pipeline.routing_decision import assess_difficulty

    band = assess_difficulty(request.prompt, role, task_id)[1]
    return routing_budget_outputs(request, band, config)


def effective_generation_cap(request: Any, routing: Any = None) -> int | None:
    """Caller max_tokens is authoritative; otherwise return the optional routed cap."""
    if getattr(request, "max_tokens", None) is not None:
        return None
    routed = getattr(routing, "generation_max_tokens", None)
    return routed if isinstance(routed, int) and not isinstance(routed, bool) and routed > 0 else None


@contextmanager
def request_budget_scope(request: Any, routing: Any = None) -> Iterator[None]:
    """Carry both resolved budgets in request-local context across async/thread hops."""
    reasoning = effective_request_budget(request, routing)
    generation_cap = effective_generation_cap(request, routing)
    with budget_scope(reasoning):
        if not enabled():
            yield
            return
        token = _GENERATION_CAP.set(generation_cap)
        try:
            yield
        finally:
            _GENERATION_CAP.reset(token)


def request_fields(request: Any, template_kwargs: dict | None) -> dict[str, Any]:
    """Fields for this chat body only; thinking-off calls remain thinking-off."""
    if not enabled() or (template_kwargs or {}).get("enable_thinking") is False:
        return {}
    tokens = getattr(request, "thinking_budget", None)
    return thinking_budget_fields(tokens) if tokens is not None else {}


# -- AEI-THINK-HEADROOM (flag ``thinking_budget_headroom``, default off) -----------------
DEFAULT_ANSWER_FLOOR_TOKENS = 256
ANSWER_FLOOR_ENV = "ORCHESTRATOR_THINKING_ANSWER_FLOOR_TOKENS"


def headroom_enabled() -> bool:
    from src.features import features

    return bool(getattr(features(), "thinking_budget_headroom", False))


def answer_floor_tokens() -> int:
    """Answer tokens kept free of thinking; autopilot sweeps ``thinking.answer_floor_tokens``."""
    from src.env_parsing import env_int

    return max(0, env_int(ANSWER_FLOOR_ENV, DEFAULT_ANSWER_FLOOR_TOKENS))


def clamp_payload_thinking_budget(payload: dict[str, Any]) -> None:
    """Clamp ``thinking_budget_tokens`` to ``max_tokens - floor`` in a chat body, in place.

    No-op with the flag off or when the body carries no budget / no cap. When max_tokens <= floor
    there is no room for any budget that preserves the floor; the body is left unchanged (warned).
    """
    if not headroom_enabled():
        return
    budget = payload.get("thinking_budget_tokens")
    cap = payload.get("max_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        return
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        return
    room = cap - answer_floor_tokens()
    if room <= 0:
        import logging

        logging.getLogger(__name__).warning(
            "thinking_budget_headroom: max_tokens=%d leaves no room above the %d-token answer floor; "
            "thinking budget %d left unclamped", cap, answer_floor_tokens(), budget,
        )
        return
    if budget > room:
        payload["thinking_budget_tokens"] = room


def is_thinking_exhausted(
    output: str | None, reasoning_content: str | None, finish_reason: str | None, tool_calls: Any = None
) -> bool:
    """Empty content, non-empty reasoning, finish_reason=length: the cap was spent thinking."""
    if not headroom_enabled() or tool_calls:
        return False
    return (
        not (output or "").strip()
        and bool((reasoning_content or "").strip())
        and finish_reason == "length"
    )
