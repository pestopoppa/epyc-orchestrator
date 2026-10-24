"""Per-request state of a shared LLMPrimitives (A1-02/A2-01).

`_init_primitives` reuses one instance for every concurrent /chat request on a
worker, so per-request counters, the call log, cache_prompt and the budget
diagnostics live on a RequestScope bound in a ContextVar by `begin_request()`.
`asyncio.to_thread` and the batch paths copy the context, so every thread of one
request shares (by reference) the same scope; `lock` guards its accumulators.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


def _initial_budget_diagnostics() -> dict[str, Any]:
    return {
        "deadline_present": False,
        "budget_applied": False,
        "deadline_remaining_ms_start": None,
        "deadline_remaining_ms_end": None,
        "timeout_clamp_events": 0,
        "depth_override_enabled": False,
        "depth_override_events": 0,
        "depth_override_roles": [],
        "depth_override_skip_events": 0,
        "depth_override_skip_reasons": [],
    }


@dataclass
class RequestScope:
    call_log: list = field(default_factory=list)
    total_calls: int = 0
    total_batch_calls: int = 0
    total_tokens_generated: int = 0
    # Sum of server-REPORTED prompt tokens (never estimated; see inference.py).
    total_prompt_tokens_reported: int = 0
    # Clean timing accumulators from llama.cpp timings.
    total_prompt_eval_ms: float = 0.0
    total_generation_ms: float = 0.0
    # Server-side overhead not captured in inference timings.
    total_http_overhead_ms: float = 0.0
    last_predicted_tps: float = 0.0  # Most recent call's clean t/s
    max_recursion_depth_reached: int = 0
    cache_prompt: bool | None = None  # None = backend default
    current_query: Any = None
    completed_queries: list = field(default_factory=list)
    budget_diagnostics: dict[str, Any] = field(default_factory=_initial_budget_diagnostics)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class ScopeField:
    """An LLMPrimitives attribute stored on the context's current RequestScope."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.field = name.lstrip("_")

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return getattr(obj._request_scope(), self.field)

    def __set__(self, obj: Any, value: Any) -> None:
        setattr(obj._request_scope(), self.field, value)
