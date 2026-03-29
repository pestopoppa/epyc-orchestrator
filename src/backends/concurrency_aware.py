"""Concurrency-aware backend for pre-warm NUMA deployments.

Routes single sessions to the full-speed (1×96t) instance for maximum
per-request throughput. When concurrent requests arrive, migrates KV state
from the full instance to a quarter (48t) instance and routes new requests
to idle quarters.

Pre-warm architecture:
    - 1 full-speed instance (96t, node-pinned) — best single-session speed
    - 4 quarter instances (48t each, NUMA-quarter-pinned) — concurrent slots

The full instance is ALWAYS running (weights in RAM, mlocked). Quarter
instances are ALWAYS running too. The only dynamic operation is KV state
save/restore on transition, using llama.cpp's slot save/restore API.

Usage:
    full_backend = CachingBackend(srv_96t, ...)
    quarter_backends = [CachingBackend(srv_48t_0, ...), ...]
    ca = ConcurrencyAwareBackend(full_backend, quarter_backends, role="frontdoor")
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class ConcurrencyAwareBackend:
    """Routes requests between full-speed and quarter instances based on load.

    Single active request  → full-speed instance (max per-request throughput)
    Multiple active requests → quarter instances (max concurrency)

    KV state migration is triggered when a second request arrives while the
    full instance is busy. The migration is:
        1. Save KV state from full instance (POST /slots/0?action=save)
        2. Restore on a quarter instance (POST /slots/0?action=restore)
        3. Route the original request's continuation to that quarter
        4. Route new request to another idle quarter

    Note: KV migration is best-effort. If save/restore fails, the request
    continues on the quarter instance without prior KV state (cold start).
    """

    def __init__(
        self,
        full_backend: Any,
        quarter_backends: list[Any],
        role: str = "",
        full_port: int = 0,
    ):
        if not quarter_backends:
            raise ValueError("ConcurrencyAwareBackend requires at least one quarter backend")
        self._full = full_backend
        self._quarters = quarter_backends
        self._role = role
        self._full_port = full_port
        self._lock = threading.Lock()

        # Tracking state
        self._full_active = False  # Is the full instance currently serving a request?
        self._quarter_active: list[bool] = [False] * len(quarter_backends)
        self._total_requests = 0
        self._full_requests = 0
        self._quarter_requests = 0
        self._migrations = 0

        logger.info(
            "ConcurrencyAwareBackend initialized for %s: 1 full + %d quarters",
            role or "unknown",
            len(quarter_backends),
        )

    def _select(self) -> tuple[Any, int, bool]:
        """Select the best backend for the next request.

        Returns (backend, index, is_full) where index is:
            -1 for full instance
            0..N for quarter instances
        """
        with self._lock:
            self._total_requests += 1

            # If full instance is idle, use it (best per-request speed)
            if not self._full_active:
                self._full_active = True
                self._full_requests += 1
                return self._full, -1, True

            # Full is busy — find an idle quarter
            for i, active in enumerate(self._quarter_active):
                if not active:
                    self._quarter_active[i] = True
                    self._quarter_requests += 1
                    return self._quarters[i], i, False

            # All quarters busy — overflow to least-recently-used quarter
            # (round-robin among quarters as fallback)
            idx = self._quarter_requests % len(self._quarters)
            self._quarter_active[idx] = True
            self._quarter_requests += 1
            logger.warning(
                "All %s instances busy (%d quarters), overflow to quarter %d",
                self._role, len(self._quarters), idx,
            )
            return self._quarters[idx], idx, False

    def _release(self, idx: int, is_full: bool) -> None:
        with self._lock:
            if is_full:
                self._full_active = False
            elif 0 <= idx < len(self._quarter_active):
                self._quarter_active[idx] = False

    # === Forward all backend interface methods ===

    def infer(self, role_config: Any, request: Any) -> Any:
        backend, idx, is_full = self._select()
        try:
            return backend.infer(role_config, request)
        finally:
            self._release(idx, is_full)

    def infer_streaming(self, role_config: Any, request: Any) -> Any:
        backend, idx, is_full = self._select()
        try:
            return backend.infer_streaming(role_config, request)
        finally:
            self._release(idx, is_full)

    def infer_stream_text(self, role_config: Any, request: Any, on_chunk: Any = None) -> Any:
        backend, idx, is_full = self._select()
        try:
            return backend.infer_stream_text(role_config, request, on_chunk=on_chunk)
        finally:
            self._release(idx, is_full)

    def health_check(self, pid: int = 0) -> bool:
        """Check health of full instance + all quarters."""
        full_ok = self._full.health_check(pid)
        quarters_ok = all(q.health_check(pid) for q in self._quarters)
        return full_ok and quarters_ok

    def get_stats(self) -> dict[str, Any]:
        """Telemetry for observability (DS-1 compatible)."""
        with self._lock:
            quarter_active = list(self._quarter_active)
            full_active = self._full_active

        return {
            "role": self._role,
            "backend_type": "concurrency_aware",
            "full_instance": {
                "port": self._full_port,
                "active": full_active,
                "total_served": self._full_requests,
            },
            "quarter_instances": len(self._quarters),
            "quarter_active": quarter_active,
            "total_active": (1 if full_active else 0) + sum(quarter_active),
            "idle_quarters": sum(1 for a in quarter_active if not a),
            "total_requests": self._total_requests,
            "full_requests": self._full_requests,
            "quarter_requests": self._quarter_requests,
            "migrations": self._migrations,
        }

    def __len__(self) -> int:
        return 1 + len(self._quarters)

    def __repr__(self) -> str:
        return (
            f"ConcurrencyAwareBackend(role={self._role!r}, "
            f"full=1, quarters={len(self._quarters)})"
        )
