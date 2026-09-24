"""Shared (unified) KV pool admission — queue, never oversubscribe.

See the block comment below for the failure it prevents and the design.
"""

from __future__ import annotations

import logging
import os
import threading
import time

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


class SharedKVPoolAdmission:
    """FCFS token reservations against each server's shared KV pool."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._inflight: dict[str, dict[int, int]] = {}
        self._queue: dict[str, list[int]] = {}
        self._next_ticket = 0

    def in_flight_tokens(self, url: str) -> int:
        with self._cond:
            return sum(self._inflight.get(url, {}).values())

    def queued(self, url: str) -> int:
        with self._cond:
            return len(self._queue.get(url, []))

    def _fits(self, url: str, tokens: int, pool_tokens: int) -> bool:
        reserved = self._inflight.get(url, {})
        return not reserved or sum(reserved.values()) + tokens <= pool_tokens

    def acquire(
        self,
        url: str,
        tokens: int,
        pool_tokens: int,
        *,
        deadline_s: float | None = None,
        timeout_s: float | None = None,
        cancel_check=None,
        poll_s: float = 0.25,
    ) -> int | None:
        """Queue for ``tokens`` of ``url``'s pool; return a ticket, or None when
        the wait ends (deadline, timeout, cancellation) before it fits.

        ``deadline_s`` is a ``time.perf_counter`` deadline (the primitives clock).
        ``timeout_s`` defaults to ``ORCHESTRATOR_KV_POOL_WAIT_S`` only when there
        is no deadline; with a deadline the deadline alone bounds the wait.
        """
        tokens = max(1, min(int(tokens), int(pool_tokens)))
        if timeout_s is None and deadline_s is None:
            try:
                timeout_s = float(os.environ.get(KV_POOL_WAIT_ENV, DEFAULT_KV_POOL_WAIT_S))
            except ValueError:
                timeout_s = DEFAULT_KV_POOL_WAIT_S
        start = time.perf_counter()
        with self._cond:
            self._next_ticket += 1
            ticket = self._next_ticket
            queue = self._queue.setdefault(url, [])
            queue.append(ticket)
            logged = False
            try:
                while not (queue[0] == ticket and self._fits(url, tokens, pool_tokens)):
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
                            "KV pool admission: %s holds %d/%d reserved tokens, %d queued ahead; "
                            "request of %d tokens queued (not dispatched)",
                            url, sum(self._inflight.get(url, {}).values()), pool_tokens,
                            queue.index(ticket), tokens,
                        )
                        logged = True
                    self._cond.wait(timeout=poll_s)
                self._inflight.setdefault(url, {})[ticket] = tokens
                if logged:
                    logger.info("KV pool admission: %s request of %d tokens admitted after %.1fs",
                                url, tokens, time.perf_counter() - start)
                return ticket
            finally:
                # Leave the queue whether admitted or abandoned, and wake the
                # next waiter (the head may have changed).
                try:
                    queue.remove(ticket)
                except ValueError:
                    pass
                if not queue:
                    self._queue.pop(url, None)
                self._cond.notify_all()

    def release(self, url: str, ticket: int | None) -> None:
        if ticket is None:
            return
        with self._cond:
            reserved = self._inflight.get(url)
            if reserved is not None:
                reserved.pop(ticket, None)
                if not reserved:
                    self._inflight.pop(url, None)
            self._cond.notify_all()

    def get_status(self) -> dict[str, dict[str, int]]:
        with self._cond:
            urls = set(self._inflight) | set(self._queue)
            return {
                url: {
                    "reserved_tokens": sum(self._inflight.get(url, {}).values()),
                    "in_flight": len(self._inflight.get(url, {})),
                    "queued": len(self._queue.get(url, [])),
                }
                for url in urls
            }


_shared_pool_admission = SharedKVPoolAdmission()


def get_shared_pool_admission() -> SharedKVPoolAdmission:
    return _shared_pool_admission
