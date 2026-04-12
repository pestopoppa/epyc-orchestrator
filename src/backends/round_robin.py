"""Round-robin backend for distributing requests across NUMA replica instances.

Wraps multiple LlamaServerBackend (or CachingBackend) instances and cycles
through them using an atomic counter. Provides transparent load distribution
for multi-instance roles (frontdoor, coder_escalation).

Usage:
    backends = [CachingBackend(srv1, ...), CachingBackend(srv2, ...)]
    rr = RoundRobinBackend(backends)
    result = rr.infer(role_config, request)  # routes to next instance
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class RoundRobinBackend:
    """Distributes inference requests across multiple backend instances.

    Thread-safe via a simple lock on the counter. For our workload
    (serial roles with admission control), contention is negligible.
    """

    def __init__(self, backends: list[Any], role: str = ""):
        if not backends:
            raise ValueError("RoundRobinBackend requires at least one backend")
        self._backends = backends
        self._role = role
        self._counter = 0
        self._lock = threading.Lock()
        # DS-1: Queue depth telemetry — per-instance active request tracking
        self._active_per_instance: list[int] = [0] * len(backends)
        self._total_per_instance: list[int] = [0] * len(backends)
        self._last_request_ts: float = 0.0
        logger.info(
            "RoundRobinBackend initialized for %s with %d instances",
            role or "unknown",
            len(backends),
        )

    def _next(self) -> tuple[Any, int]:
        with self._lock:
            idx = self._counter % len(self._backends)
            backend = self._backends[idx]
            self._counter += 1
            self._active_per_instance[idx] += 1
            self._total_per_instance[idx] += 1
            self._last_request_ts = time.monotonic()
        return backend, idx

    def _release(self, idx: int) -> None:
        with self._lock:
            self._active_per_instance[idx] = max(0, self._active_per_instance[idx] - 1)

    # === Forward all CachingBackend / LlamaServerBackend interface methods ===

    def infer(self, role_config: Any, request: Any) -> Any:
        backend, idx = self._next()
        try:
            return backend.infer(role_config, request)
        finally:
            self._release(idx)

    def infer_streaming(self, role_config: Any, request: Any) -> Any:
        backend, idx = self._next()
        try:
            return backend.infer_streaming(role_config, request)
        finally:
            self._release(idx)

    def infer_stream_text(self, role_config: Any, request: Any, on_chunk: Any = None) -> Any:
        backend, idx = self._next()
        try:
            return backend.infer_stream_text(role_config, request, on_chunk=on_chunk)
        finally:
            self._release(idx)

    def health_check(self, pid: int = 0) -> bool:
        return all(b.health_check(pid) for b in self._backends)

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats across all instances including queue depth telemetry."""
        with self._lock:
            active = list(self._active_per_instance)
            totals = list(self._total_per_instance)
            last_ts = self._last_request_ts

        combined: dict[str, Any] = {
            "role": self._role,
            "round_robin_instances": len(self._backends),
            "round_robin_requests": self._counter,
            # DS-1: Queue depth telemetry
            "active_per_instance": active,
            "total_active": sum(active),
            "idle_instances": sum(1 for a in active if a == 0),
            "total_per_instance": totals,
            "seconds_since_last_request": (
                round(time.monotonic() - last_ts, 1) if last_ts > 0 else None
            ),
            "per_instance": [],
        }
        for i, b in enumerate(self._backends):
            inst_stats: dict[str, Any] = {
                "instance": i,
                "active": active[i] if i < len(active) else 0,
                "total_served": totals[i] if i < len(totals) else 0,
            }
            if hasattr(b, "get_stats"):
                inst_stats.update(b.get_stats())
            combined["per_instance"].append(inst_stats)
        return combined

    # DS-6: Dynamic instance management for QuarterScheduler

    def add_instance(self, backend: Any) -> None:
        """Add a new backend instance to the round-robin pool.

        Thread-safe. The new instance starts receiving requests immediately.
        """
        with self._lock:
            self._backends.append(backend)
            self._active_per_instance.append(0)
            self._total_per_instance.append(0)
        logger.info(
            "RoundRobin[%s]: added instance (now %d total)",
            self._role, len(self._backends),
        )

    def remove_instance(self, idx: int) -> bool:
        """Remove a backend instance by index. Returns False if index invalid.

        Thread-safe. Refuses removal if the instance has active requests.
        Caller must drain traffic before calling this.
        """
        with self._lock:
            if idx < 0 or idx >= len(self._backends):
                return False
            if self._active_per_instance[idx] > 0:
                logger.warning(
                    "RoundRobin[%s]: refusing to remove instance %d with %d active requests",
                    self._role, idx, self._active_per_instance[idx],
                )
                return False
            self._backends.pop(idx)
            self._active_per_instance.pop(idx)
            self._total_per_instance.pop(idx)
        logger.info(
            "RoundRobin[%s]: removed instance %d (now %d total)",
            self._role, idx, len(self._backends),
        )
        return True

    def instance_count(self) -> int:
        """Return current number of backend instances."""
        with self._lock:
            return len(self._backends)

    def __len__(self) -> int:
        return len(self._backends)

    def __repr__(self) -> str:
        return f"RoundRobinBackend(role={self._role!r}, instances={len(self._backends)})"
