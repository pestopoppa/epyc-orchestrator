"""Per-backend admission control using asyncio semaphores.

Prevents request pile-up on slow backends (especially architects) by limiting
concurrent in-flight requests per backend URL. When a backend's queue is full,
requests are rejected immediately with a clear error rather than blocking
until timeout.

OBSERVATION (2026-02-20): Slot/Admission Alignment
===================================================
Key finding from backend saturation investigation (handoffs/active/backend-saturation-504-429.md):
Every backend had 2x more llama-server slots than admission controller allowed. llama-server
partitions KV cache evenly across all slots, so 50% of KV cache was wasted on idle slots.

Concurrency sweep results (concurrent_sweep_20260219_144159.summary.json) showed:
- frontdoor: optimal at concurrency=2, p95 latency 1.33x (acceptable)
- coder_escalation: p95 latency 1.98x at concurrency=2 (degradation) — reduced to serial
- worker: ALL concurrent levels (2,3,4) rejected on p95 threshold — reduced to serial
- architects: already serial

Changes applied:
  - Aligned admission limits with llama-server slot counts (no idle slots)
  - Reduced coder_escalation from 2→1 and worker from 4→1 based on sweep data
  - Also updated orchestration/model_registry.yaml server_mode slot counts
  - Increased frontdoor timeout from 90→180s in model_registry.yaml

Expected impact: 50% reduction in idle KV cache pressure across all backends.

See DEFAULT_LIMITS comments for per-backend rationale.

Usage:
    controller = AdmissionController.from_defaults()
    if not controller.try_acquire(backend_url):
        raise RuntimeError("Backend queue full")
    try:
        result = do_inference(backend_url, ...)
    finally:
        controller.release(backend_url)
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time

import yaml

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STACK_PRIORS_PATH = PROJECT_ROOT / "orchestration" / "derived" / "stack_priors.yaml"

# Degraded fallback only. Normal defaults are generated from stack priors.
FALLBACK_LIMITS: dict[str, int] = {
    "http://localhost:8070": 2,
    "http://localhost:8072": 1,
    "http://localhost:8080": 2,
    "http://localhost:8082": 1,
    "http://localhost:8083": 2,
    "http://localhost:8085": 1,
    "http://localhost:8086": 2,
    "http://localhost:8087": 1,
}

# Embedding servers (8090-8095) are not gated — they're lightweight.


def _merge_limit(limits: dict[str, int], url: str, slots: int) -> None:
    if not url or slots <= 0:
        return
    limits[url] = max(slots, limits.get(url, 0))


def _limits_from_stack_priors(path: Path = STACK_PRIORS_PATH) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    roles = loaded.get("roles") if isinstance(loaded, dict) else None
    if not isinstance(roles, dict):
        return {}

    limits: dict[str, int] = {}
    for record in roles.values():
        if not isinstance(record, dict) or record.get("deployment_status") != "live_stack":
            continue
        serving = record.get("serving")
        if not isinstance(serving, dict):
            continue
        slots = serving.get("slots")
        if not isinstance(slots, int) or slots <= 0:
            continue
        endpoint = serving.get("endpoint")
        if isinstance(endpoint, str):
            _merge_limit(limits, endpoint, slots)
        ports = serving.get("ports")
        if isinstance(ports, list):
            for port in ports:
                if isinstance(port, int):
                    _merge_limit(limits, f"http://localhost:{port}", slots)
    return limits


def _load_default_limits(path: Path = STACK_PRIORS_PATH) -> dict[str, int]:
    defaults = dict(FALLBACK_LIMITS)
    try:
        limits = _limits_from_stack_priors(path)
    except (OSError, yaml.YAMLError, AttributeError, TypeError):
        logger.warning("Falling back to static admission limits", exc_info=True)
        return defaults
    for url, slots in limits.items():
        _merge_limit(defaults, url, slots)
    return defaults


DEFAULT_LIMITS: dict[str, int] = _load_default_limits()


class AdmissionController:
    """Thread-safe per-backend concurrency limiter.

    Uses threading.Semaphore (not asyncio.Semaphore) because inference calls
    happen in thread pool via asyncio.to_thread / inference_lock (fcntl).
    """

    def __init__(self, limits: dict[str, int] | None = None):
        self._limits = limits or DEFAULT_LIMITS
        self._semaphores: dict[str, threading.Semaphore] = {
            url: threading.Semaphore(n) for url, n in self._limits.items()
        }
        self._lock = threading.Lock()
        self._waiting: dict[str, dict[str, int]] = {
            url: {"interactive": 0, "background": 0}
            for url in self._limits
        }

    @classmethod
    def from_defaults(cls) -> AdmissionController:
        return cls(DEFAULT_LIMITS)

    def _norm_priority(self, priority: str | None) -> str:
        p = str(priority or "interactive").strip().lower()
        return "background" if p == "background" else "interactive"

    def acquire(
        self,
        backend_url: str,
        *,
        priority: str = "interactive",
        wait: bool = False,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        cancel_check=None,
        poll_s: float = 0.02,
    ) -> bool:
        """Acquire admission slot with optional bounded wait.

        Interactive requests are prioritized over background requests when
        waiters exist on the same backend.
        """
        sem = self._semaphores.get(backend_url)
        if sem is None:
            return True

        prio = self._norm_priority(priority)
        start = time.perf_counter()
        waited = False
        with self._lock:
            q = self._waiting.setdefault(
                backend_url, {"interactive": 0, "background": 0}
            )
            q[prio] = q.get(prio, 0) + 1
            waited = True
        try:
            while True:
                if cancel_check is not None:
                    try:
                        if cancel_check():
                            return False
                    except Exception:
                        pass
                now = time.perf_counter()
                if deadline_s is not None and now >= deadline_s:
                    return False
                if timeout_s is not None and (now - start) >= max(0.0, timeout_s):
                    return False

                with self._lock:
                    waiting_interactive = self._waiting.get(backend_url, {}).get(
                        "interactive", 0
                    )
                # Keep one slot path open for interactive arrivals while background waits.
                if prio == "background" and waiting_interactive > 0:
                    if not wait:
                        return False
                    time.sleep(poll_s)
                    continue

                if sem.acquire(blocking=False):
                    return True

                if not wait:
                    return False
                time.sleep(poll_s)
        finally:
            if waited:
                with self._lock:
                    q = self._waiting.setdefault(
                        backend_url, {"interactive": 0, "background": 0}
                    )
                    q[prio] = max(0, q.get(prio, 0) - 1)

    def try_acquire(self, backend_url: str) -> bool:
        """Non-blocking acquire. Returns True if admitted, False if queue full."""
        acquired = self.acquire(backend_url, wait=False)
        if not acquired:
            logger.warning(
                "Admission rejected for %s (limit=%d)",
                backend_url,
                self._limits.get(backend_url, 0),
            )
        return acquired

    def release(self, backend_url: str) -> None:
        """Release a slot for the backend."""
        sem = self._semaphores.get(backend_url)
        if sem is not None:
            sem.release()

    def get_status(self) -> dict[str, dict[str, int]]:
        """Return current admission status per backend."""
        status = {}
        for url, sem in self._semaphores.items():
            limit = self._limits.get(url, 0)
            # Semaphore._value is the current counter (available slots)
            available = sem._value  # type: ignore[attr-defined]
            with self._lock:
                waiting = self._waiting.get(url, {})
            status[url] = {
                "limit": limit,
                "available": available,
                "in_flight": limit - available,
                "waiting_interactive": int(waiting.get("interactive", 0)),
                "waiting_background": int(waiting.get("background", 0)),
            }
        return status
