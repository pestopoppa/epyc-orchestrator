"""Shared (unified) KV pool admission — queue, never oversubscribe.

See the block comment below for the failure it prevents and the design.
"""

from __future__ import annotations

import logging
import os
import math
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Shared (unified) KV pool admission ────────────────────────────────────
#
# The per-backend semaphores in src/api/admission.py count REQUESTS (== -np).
# Under --kv-unified that is not the binding constraint: every slot may take a
# request as large as the whole -c, and the server admits each one against slot
# n_ctx only
# (server-context.cpp:3303-3310) — it never checks the pool's occupancy. Two
# long requests (e.g. 120k + 90k on a 196608 pool) are both admitted, and when
# llama_decode runs out of cells the server purges idle slots, halves the batch
# to 1 and then fails EVERY in-flight request with "Context size has been
# exceeded." (:3759-3764, :2949-2951).
#
# So this is the PRIMARY mechanism, modelled on how vLLM/SGLang schedule a
# shared KV pool: never oversubscribe it. A request whose token reservation
# (prompt + generation budget) does not fit alongside the in-flight ones is
# QUEUED, not dispatched, strictly first-come-first-served per server (a
# stream of small requests cannot starve a waiting long one), until it fits or
# its own deadline / cancellation ends the wait. Retry-on-exhaustion in the
# inference layer (src/llm_primitives/context_recovery.py) is only the
# FALLBACK, for load this process cannot see (clients that bypass the
# orchestrator, e.g. opencode straight to :8083) and estimate error.
#
# A lone request is always admitted whatever its size — the server decides
# whether it fits one slot (HTTP 400 → reroute/typed error).

KV_POOL_WAIT_ENV = "ORCHESTRATOR_KV_POOL_WAIT_S"
# Used only when the request carries no deadline of its own. Long on purpose:
# queueing is the correct behaviour under pool pressure, and a long in-flight
# request can legitimately hold the pool for many minutes.
DEFAULT_KV_POOL_WAIT_S = 1800.0

# Bounded queue (vLLM `max_num_queued_reqs`, SGLang `--max-queued-requests`,
# TGI `--max-concurrent-requests`): past this many waiters per server a new
# request is refused at once (503 + Retry-After) instead of waiting silently.
# <= 0 disables the bound.
KV_POOL_MAX_QUEUED_ENV = "ORCHESTRATOR_KV_POOL_MAX_QUEUED"
DEFAULT_KV_POOL_MAX_QUEUED = 8

# Adaptive decode reservation (SGLang `new_token_ratio`): reserve
# prompt + ceil(max_new_tokens * ratio). The ratio starts at INIT, decays by
# DECAY per request that completes without pool trouble, never below MIN, and
# snaps back to INIT on any POOL_EXHAUSTED for that server. Reserving the full
# max_tokens over-serialises thinking-mode calls (max_tokens 32768 that finish
# far below it); a ratio that is too low shows up as POOL_EXHAUSTED, which
# resets it and is absorbed by the retry fallback.
KV_POOL_RATIO_INIT_ENV = "ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO"
KV_POOL_RATIO_MIN_ENV = "ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO_MIN"
KV_POOL_RATIO_DECAY_ENV = "ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO_DECAY"
DEFAULT_NEW_TOKEN_RATIO = 1.0
DEFAULT_MIN_NEW_TOKEN_RATIO = 0.3
DEFAULT_NEW_TOKEN_RATIO_DECAY = 0.05


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class KVPoolQueueFull(RuntimeError):
    """The per-server admission queue is at its bound; retry later (503)."""

    def __init__(self, url: str, queued: int, limit: int) -> None:
        super().__init__(
            f"KV pool admission queue for {url} is full ({queued} waiting, limit {limit})"
        )
        self.url = url
        self.queued = queued
        self.limit = limit


def _default_occupancy(url: str) -> Any:
    from src.backends.context_limits import get_context_limit_resolver

    return get_context_limit_resolver().pool_occupancy(url)


class SharedKVPoolAdmission:
    """FCFS token reservations against each server's shared KV pool.

    ``_fits`` counts the larger of (a) this process's own reservations and
    (b) the server's REAL occupancy from ``GET /slots`` (in-flight cells plus
    their ratio-weighted remaining decode), so load that bypasses the
    orchestrator is seen. When /slots is unavailable it degrades to (a).
    """

    def __init__(self, occupancy: Callable[[str], Any] | None = None) -> None:
        self._cond = threading.Condition()
        self._inflight: dict[str, dict[int, int]] = {}
        self._queue: dict[str, list[int]] = {}
        self._ratio: dict[str, float] = {}
        self._next_ticket = 0
        self._occupancy_fn = occupancy if occupancy is not None else _default_occupancy

    # -- adaptive decode reservation -----------------------------------------
    def new_token_ratio(self, url: str) -> float:
        with self._cond:
            return self._ratio.get(url, self._ratio_init())

    @staticmethod
    def _ratio_init() -> float:
        return max(0.0, _env_float(KV_POOL_RATIO_INIT_ENV, DEFAULT_NEW_TOKEN_RATIO))

    def _decay(self, url: str) -> None:
        floor = max(0.0, _env_float(KV_POOL_RATIO_MIN_ENV, DEFAULT_MIN_NEW_TOKEN_RATIO))
        step = max(0.0, _env_float(KV_POOL_RATIO_DECAY_ENV, DEFAULT_NEW_TOKEN_RATIO_DECAY))
        current = self._ratio.get(url, self._ratio_init())
        self._ratio[url] = max(min(floor, current), current - step)

    def report_pool_exhausted(self, url: str) -> None:
        """The server ran out of KV cells: stop being optimistic about decode."""
        with self._cond:
            before = self._ratio.get(url, self._ratio_init())
            self._ratio[url] = self._ratio_init()
        logger.warning(
            "KV pool exhausted on %s: new_token_ratio %.2f -> %.2f", url, before, self._ratio_init()
        )

    def reservation_tokens(self, url: str, prompt_tokens: int, max_new_tokens: int) -> int:
        ratio = self.new_token_ratio(url)
        return int(prompt_tokens) + int(math.ceil(max(0, int(max_new_tokens)) * ratio))

    # -- accounting ------------------------------------------------------------
    def in_flight_tokens(self, url: str) -> int:
        with self._cond:
            return sum(self._inflight.get(url, {}).values())

    def queued(self, url: str) -> int:
        with self._cond:
            return len(self._queue.get(url, []))

    def _observed(self, url: str) -> tuple[int, int] | None:
        """(projected in-flight tokens, processing slots) from /slots, or None."""
        try:
            occ = self._occupancy_fn(url)
        except Exception:
            logger.debug("KV pool admission: occupancy read failed for %s", url, exc_info=True)
            return None
        if occ is None:
            return None
        ratio = self.new_token_ratio(url)
        return int(occ.projected_tokens(ratio)), int(occ.processing)

    def _fits(self, url: str, tokens: int, pool_tokens: int, observed: tuple[int, int] | None) -> bool:
        reserved = self._inflight.get(url, {})
        own = sum(reserved.values())
        seen, processing = observed if observed is not None else (0, 0)
        if not reserved and processing == 0:
            return True  # a lone request is always admitted; the server decides
        return max(own, seen) + tokens <= pool_tokens

    def acquire(
        self,
        url: str,
        tokens: int,
        pool_tokens: int,
        *,
        max_new_tokens: int = 0,
        deadline_s: float | None = None,
        timeout_s: float | None = None,
        cancel_check=None,
        poll_s: float = 0.25,
        max_queued: int | None = None,
    ) -> int | None:
        """Queue for ``tokens`` (+ ratio-weighted ``max_new_tokens``) of ``url``'s
        pool; return a ticket, or None when the wait ends (deadline, timeout,
        cancellation) before it fits. Raises KVPoolQueueFull when ``max_queued``
        (default ``ORCHESTRATOR_KV_POOL_MAX_QUEUED``) requests already wait.

        ``deadline_s`` is a ``time.perf_counter`` deadline (the primitives clock).
        ``timeout_s`` defaults to ``ORCHESTRATOR_KV_POOL_WAIT_S`` only when there
        is no deadline; with a deadline the deadline alone bounds the wait.
        """
        pool_tokens = max(1, int(pool_tokens))
        want = self.reservation_tokens(url, tokens, max_new_tokens)
        want = max(1, min(want, pool_tokens))
        if timeout_s is None and deadline_s is None:
            timeout_s = _env_float(KV_POOL_WAIT_ENV, DEFAULT_KV_POOL_WAIT_S)
        if max_queued is None:
            max_queued = _env_int(KV_POOL_MAX_QUEUED_ENV, DEFAULT_KV_POOL_MAX_QUEUED)
        start = time.perf_counter()
        with self._cond:
            queue = self._queue.setdefault(url, [])
            if max_queued > 0 and len(queue) >= max_queued:
                raise KVPoolQueueFull(url, len(queue), max_queued)
            self._next_ticket += 1
            ticket = self._next_ticket
            queue.append(ticket)
        logged = False
        try:
            while True:
                # Read the server outside the lock (an HTTP GET, cached ~1.5 s).
                observed = self._observed(url)
                with self._cond:
                    if queue[0] == ticket and self._fits(url, want, pool_tokens, observed):
                        self._inflight.setdefault(url, {})[ticket] = want
                        if logged:
                            logger.info(
                                "KV pool admission: %s request of %d tokens admitted after %.1fs",
                                url, want, time.perf_counter() - start,
                            )
                        return ticket
                    now = time.perf_counter()
                    if deadline_s is not None and now >= deadline_s:
                        return None
                    if timeout_s is not None and now - start >= max(0.0, timeout_s):
                        return None
                    if cancel_check is not None:
                        try:
                            if cancel_check():
                                return None
                        except Exception:
                            pass
                    if not logged:
                        logger.warning(
                            "KV pool admission: %s own=%d observed=%s pool=%d, %d queued ahead; "
                            "request of %d tokens queued (not dispatched)",
                            url, sum(self._inflight.get(url, {}).values()),
                            observed[0] if observed else "n/a", pool_tokens,
                            queue.index(ticket), want,
                        )
                        logged = True
                    self._cond.wait(timeout=poll_s)
        finally:
            # Leave the queue whether admitted or abandoned, and wake the next
            # waiter (the head may have changed).
            with self._cond:
                try:
                    queue.remove(ticket)
                except ValueError:
                    pass
                if not queue and self._queue.get(url) is queue:
                    self._queue.pop(url, None)
                self._cond.notify_all()

    def release(self, url: str, ticket: int | None, *, success: bool = True) -> None:
        """Return a reservation. ``success`` = the request finished without pool
        trouble, which decays the decode ratio toward its floor."""
        if ticket is None:
            return
        with self._cond:
            reserved = self._inflight.get(url)
            if reserved is not None:
                reserved.pop(ticket, None)
                if not reserved:
                    self._inflight.pop(url, None)
            if success:
                self._decay(url)
            self._cond.notify_all()

    def get_status(self) -> dict[str, dict[str, Any]]:
        with self._cond:
            urls = set(self._inflight) | set(self._queue) | set(self._ratio)
            return {
                url: {
                    "reserved_tokens": sum(self._inflight.get(url, {}).values()),
                    "in_flight": len(self._inflight.get(url, {})),
                    "queued": len(self._queue.get(url, [])),
                    "new_token_ratio": round(self._ratio.get(url, self._ratio_init()), 4),
                }
                for url in urls
            }


_shared_pool_admission = SharedKVPoolAdmission()


def get_shared_pool_admission() -> SharedKVPoolAdmission:
    return _shared_pool_admission
