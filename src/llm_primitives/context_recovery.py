"""Recovery policy for llama-server context overflow (ContextOverflowError).

Order, each step bounded:

(a) POOL EXHAUSTED and the request fits one slot on its own → it is
    concurrency (a unified KV pool filled by other in-flight requests), so back
    off and retry the same role: at most ``ORCHESTRATOR_CTX_POOL_RETRIES``
    (default 2) retries, exponential backoff from
    ``ORCHESTRATOR_CTX_POOL_BACKOFF_S`` (default 2s, cap 15s), never past the
    request deadline, abandoned on cancellation.
(b) REQUEST TOO LARGE for the role (or pool-exhausted but it would not fit even
    alone) → reroute ONCE to the first role in ``context_overflow_roles()``
    whose live per-request limit fits it. Semantic compaction of a
    conversation happens one level up, in the graph (``_execute_turn`` forces
    ``_maybe_compact_context`` on this error), because only the graph owns the
    context it can safely externalise. Nothing here truncates a prompt.
(c) Otherwise re-raise the typed error, annotated with every attempt made. The
    caller gets a clear ContextOverflowError — never a truncated prompt and
    never an empty answer.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable

from src.backends.context_limits import (
    ContextLimit,
    context_overflow_roles,
    estimate_tokens_conservative,
    get_context_limit_resolver,
)
from src.exceptions import ContextOverflowError

log = logging.getLogger(__name__)

POOL_RETRIES_ENV = "ORCHESTRATOR_CTX_POOL_RETRIES"
POOL_BACKOFF_ENV = "ORCHESTRATOR_CTX_POOL_BACKOFF_S"
DEFAULT_POOL_RETRIES = 2
DEFAULT_POOL_BACKOFF_S = 2.0
MAX_POOL_BACKOFF_S = 15.0


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return default


def _needed_tokens(exc: ContextOverflowError, prompt: str) -> int:
    """Prompt tokens: the server's exact count when it gave one, else a conservative estimate."""
    if exc.n_prompt_tokens:
        return int(exc.n_prompt_tokens)
    return estimate_tokens_conservative(prompt)


def recover_context_overflow(
    exc: ContextOverflowError,
    *,
    prompt: str,
    role: str,
    call: Callable[[str], Any],
    urls_for_role: Callable[[str], list[str] | str | None] | None = None,
    deadline_s: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    resolver: Any | None = None,
    max_pool_retries: int | None = None,
    backoff_s: float | None = None,
    reroute_candidates: list[str] | None = None,
) -> Any:
    """Apply the (a)/(b)/(c) policy. ``call(role)`` re-issues the request on ``role``.

    ``deadline_s`` is on the same clock as ``clock`` (the primitives layer uses
    ``time.perf_counter`` deadlines).
    """
    resolver = resolver or get_context_limit_resolver()
    max_pool_retries = (
        _env_int(POOL_RETRIES_ENV, DEFAULT_POOL_RETRIES)
        if max_pool_retries is None else max(0, int(max_pool_retries))
    )
    backoff_s = _env_float(POOL_BACKOFF_ENV, DEFAULT_POOL_BACKOFF_S) if backoff_s is None else backoff_s
    candidates = list(reroute_candidates) if reroute_candidates is not None else context_overflow_roles()

    attempts: list[dict[str, Any]] = list(getattr(exc, "recovery", []) or [])
    current_exc = exc
    current_role = role
    pool_retries = 0
    rerouted = False

    def _limit(r: str) -> ContextLimit | None:
        urls = urls_for_role(r) if urls_for_role else None
        try:
            return resolver.limit_for_role(r, urls)
        except Exception:
            log.debug("context recovery: limit lookup failed for %s", r, exc_info=True)
            return None

    while True:
        if current_exc.backend_url and current_exc.n_ctx:
            try:
                resolver.observe(current_exc.backend_url, n_ctx=current_exc.n_ctx)
            except Exception:
                pass
        needed = _needed_tokens(current_exc, prompt)
        limit = _limit(current_role)
        if current_exc.n_ctx:
            fits_alone = needed < int(current_exc.n_ctx)
        elif limit is not None:
            fits_alone = limit.fits(needed)
        else:
            # Unknown limit: a pool failure is presumed to be concurrency; the
            # retries are bounded either way.
            fits_alone = current_exc.kind == ContextOverflowError.POOL_EXHAUSTED

        # (a) concurrency on a shared pool → bounded backoff + retry.
        if (
            current_exc.kind == ContextOverflowError.POOL_EXHAUSTED
            and current_exc.source != "admission"
            and fits_alone
            and pool_retries < max_pool_retries
        ):
            delay = min(MAX_POOL_BACKOFF_S, backoff_s * (2 ** pool_retries))
            delay *= 1.0 + random.uniform(0.0, 0.25)
            if deadline_s is not None and clock() + delay >= deadline_s:
                attempts.append({"step": "pool_backoff", "role": current_role,
                                 "outcome": "skipped_deadline", "delay_s": round(delay, 2)})
            elif cancel_check is not None and _cancelled(cancel_check):
                attempts.append({"step": "pool_backoff", "role": current_role, "outcome": "cancelled"})
            else:
                pool_retries += 1
                log.warning(
                    "Context pool exhausted on %s (est %d prompt tokens, fits alone): "
                    "backoff %.1fs then retry %d/%d",
                    current_role, needed, delay, pool_retries, max_pool_retries,
                )
                sleep(delay)
                try:
                    out = call(current_role)
                except ContextOverflowError as retry_exc:
                    attempts.append({"step": "pool_backoff", "role": current_role,
                                     "outcome": f"overflow:{retry_exc.kind}", "delay_s": round(delay, 2)})
                    current_exc = retry_exc
                    continue
                log.info("Context pool retry succeeded on %s after %d retr%s",
                         current_role, pool_retries, "y" if pool_retries == 1 else "ies")
                return out

        # (b) too large for this role → one reroute to a larger-context role.
        if not rerouted and (
            current_exc.kind == ContextOverflowError.REQUEST_TOO_LARGE or not fits_alone
        ):
            rerouted = True
            target = None
            try:
                target = resolver.larger_context_role(
                    needed,
                    candidates=candidates,
                    exclude={current_role, role},
                    url_for_role=urls_for_role,
                )
            except Exception:
                log.debug("context recovery: larger-role lookup failed", exc_info=True)
            if target is not None:
                target_role, target_limit = target
                log.warning(
                    "Context overflow on %s (%d prompt tokens > %s): rerouting to %s "
                    "(per-request n_ctx %d, source %s)",
                    current_role, needed, current_exc.n_ctx or (limit.per_request_n_ctx if limit else "?"),
                    target_role, target_limit.per_request_n_ctx, target_limit.source,
                )
                try:
                    out = call(target_role)
                except ContextOverflowError as reroute_exc:
                    attempts.append({"step": "reroute", "from": current_role, "role": target_role,
                                     "outcome": f"overflow:{reroute_exc.kind}"})
                    current_exc = reroute_exc
                    current_role = target_role
                    continue
                return out
            attempts.append({"step": "reroute", "from": current_role, "role": None,
                             "outcome": "no_larger_role", "needed_tokens": needed})

        # (c) nothing left that can help → the typed error, with its history.
        current_exc.recovery = attempts
        if not current_exc.role:
            current_exc.role = current_role
        raise current_exc


def _cancelled(cancel_check: Callable[[], bool]) -> bool:
    try:
        return bool(cancel_check())
    except Exception:
        return False
