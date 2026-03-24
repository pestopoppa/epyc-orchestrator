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
        logger.info(
            "RoundRobinBackend initialized for %s with %d instances",
            role or "unknown",
            len(backends),
        )

    def _next(self) -> Any:
        with self._lock:
            backend = self._backends[self._counter % len(self._backends)]
            self._counter += 1
        return backend

    # === Forward all CachingBackend / LlamaServerBackend interface methods ===

    def infer(self, role_config: Any, request: Any) -> Any:
        return self._next().infer(role_config, request)

    def infer_streaming(self, role_config: Any, request: Any) -> Any:
        return self._next().infer_streaming(role_config, request)

    def infer_stream_text(self, role_config: Any, request: Any, on_chunk: Any = None) -> Any:
        return self._next().infer_stream_text(role_config, request, on_chunk=on_chunk)

    def health_check(self, pid: int = 0) -> bool:
        return all(b.health_check(pid) for b in self._backends)

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats across all instances."""
        combined: dict[str, Any] = {
            "round_robin_instances": len(self._backends),
            "round_robin_requests": self._counter,
            "per_instance": [],
        }
        for i, b in enumerate(self._backends):
            if hasattr(b, "get_stats"):
                combined["per_instance"].append(b.get_stats())
        return combined

    def __len__(self) -> int:
        return len(self._backends)

    def __repr__(self) -> str:
        return f"RoundRobinBackend(role={self._role!r}, instances={len(self._backends)})"
