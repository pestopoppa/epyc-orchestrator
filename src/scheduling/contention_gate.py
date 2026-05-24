"""Cross-role admission gate.

Phase B of `handoffs/active/cross-role-bw-aware-routing.md`.

Wraps the central model-call boundary (`src.llm_primitives.inference._real_call_impl`
or `_real_call_single`) so a request for role R is admitted only when the
active set of decoding roles is compatible with R per the contention matrix.

Key design points (per handoff):

- Authoritative active-decode source under `ORCHESTRATOR_PER_REGION_LOCKS=1`:
  `src.runtime.cpu_region_lock.active_region_holders()`. The in-process
  `_full_active`/`_quarter_active` flags on `ConcurrencyAwareBackend` are
  NOT cross-process and must not be used here.
- The gate runs BEFORE `LLMPrimitives._acquire_role()` — otherwise the
  per-role semaphore wait briefly hides an admitted request from the
  active-decode snapshot.
- Background traffic queues on `block`/`borderline`/`unknown` pairs.
  Foreground traffic queues on hard `block`, may DEGRADED_ALLOW on
  borderline-below-floor when SLO is tight.
- Metrics: `contention_blocked_count`, `contention_wait_seconds`,
  `active_decodes_by_role`, `contention_unknown_pair_count`.

Threading model: gate decisions are inexpensive (one /proc/locks scan +
matrix lookup) — single global lock is fine. Queue management uses a
simple wait-and-retry with backoff; we don't need a priority queue today.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from src.scheduling.contention import (
    ContentionMatrix,
    MatrixStatus,
    PairDecision,
    TrafficClass,
    load_contention_matrix,
    matrix_status,
    pair_policy,
)

log = logging.getLogger("scheduling.contention_gate")

# Polling interval for queued requests waiting on a contention release.
# Short enough to feel responsive, long enough not to thrash /proc/locks.
_GATE_POLL_S = 0.150

# Default max queue wait if the request doesn't specify one.
_DEFAULT_FOREGROUND_WAIT_MS = 5_000   # 5 s
_DEFAULT_BACKGROUND_WAIT_MS = 90_000  # 90 s (autopilot can wait quite a while)


@dataclass
class GateDecision:
    """Outcome of a single gate evaluation."""

    admitted: bool
    decision: PairDecision
    waited_s: float = 0.0
    blocking_roles: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class GateMetrics:
    """Counters exposed to dashboard / Prometheus.

    Aggregated across the process; the orchestrator's existing telemetry
    exporter can pick them up via `gate.metrics_snapshot()`.
    """

    contention_blocked_count: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    contention_wait_seconds: float = 0.0
    contention_unknown_pair_count: int = 0
    contention_degraded_allow_count: int = 0
    contention_admitted_count: int = 0
    contention_timeout_count: int = 0
    # active_decodes_by_role: {role: count} for back-compat;
    # active_instances_by_role: {role: [instance_idx, ...]} for richer dashboard rendering
    active_decodes_by_role: dict[str, int] = field(default_factory=dict)
    active_instances_by_role: dict[str, list[int]] = field(default_factory=dict)


class ContentionGate:
    """Singleton-ish gate. Use `get_gate()` to access the process instance."""

    def __init__(
        self,
        matrix: ContentionMatrix | None = None,
        active_holders_fn: Callable[[], dict[str, list[int]]] | None = None,
    ) -> None:
        self._matrix = matrix
        self._active_holders_fn = active_holders_fn  # injectable for tests
        self._metrics = GateMetrics()
        self._lock = threading.Lock()
        self._matrix_status_cache: MatrixStatus | None = None
        self._matrix_status_checked_at: float = 0.0

    # ── matrix access (lazy load + status caching) ──────────────────

    def _get_matrix(self) -> ContentionMatrix | None:
        if self._matrix is None:
            try:
                self._matrix = load_contention_matrix()
            except FileNotFoundError:
                # Fail-open per handoff line 113 — return None; pair_policy()
                # consumers handle None matrix gracefully (background queues,
                # foreground allows).
                return None
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to load contention matrix: %s", exc)
                return None
        return self._matrix

    def matrix_health(self) -> MatrixStatus:
        """Cheap, cached status check (refreshed every 60 s)."""
        now = time.time()
        if self._matrix_status_cache is None or (now - self._matrix_status_checked_at) > 60.0:
            self._matrix_status_cache = matrix_status()
            self._matrix_status_checked_at = now
        return self._matrix_status_cache

    # ── active-decode snapshot ──────────────────────────────────────

    def _active_holders(self) -> dict[str, list[int]]:
        if self._active_holders_fn is not None:
            return self._active_holders_fn()
        # Only consult region locks when the feature flag is on; otherwise
        # there's no cross-process active-decode source and we degrade to
        # "no actives" (which means everything is allowed).
        if os.environ.get("ORCHESTRATOR_PER_REGION_LOCKS", "0").strip() not in {"1", "true", "yes", "on"}:
            return {}
        try:
            from src.runtime.cpu_region_lock import active_region_holders
            return active_region_holders()
        except Exception as exc:  # noqa: BLE001
            log.warning("active_region_holders failed: %s", exc)
            return {}

    # ── admission core ──────────────────────────────────────────────

    def evaluate(
        self,
        role: str,
        traffic_class: TrafficClass | str = TrafficClass.FOREGROUND_INTERACTIVE,
    ) -> GateDecision:
        """One-shot evaluation: would role be admitted right now?

        Does NOT wait; just returns the decision based on a current snapshot.
        Useful for testing + when the caller has its own queue/retry logic.
        """
        if isinstance(traffic_class, str):
            try:
                traffic_class = TrafficClass(traffic_class)
            except ValueError:
                traffic_class = TrafficClass.BACKGROUND
        matrix = self._get_matrix()
        holders = self._active_holders()

        # Update active-by-role snapshot for metrics (both shapes)
        with self._lock:
            self._metrics.active_decodes_by_role = {r: len(idxs) for r, idxs in holders.items()}
            self._metrics.active_instances_by_role = {r: list(idxs) for r, idxs in holders.items()}

        if not holders:
            return GateDecision(admitted=True, decision=PairDecision.ALLOW, reason="no active decodes")

        # Evaluate against each active role (including same-role for multi-instance pairs).
        worst: PairDecision = PairDecision.ALLOW
        blocking: list[str] = []
        for active_role in holders.keys():
            decision = pair_policy(role, active_role, traffic_class, matrix=matrix)
            if decision == PairDecision.ALLOW:
                continue
            # QUEUE > DEGRADED_ALLOW > BLOCK ordering for "worst"; we
            # bias toward the most-restrictive decision so the caller's
            # wait logic always picks a safe path.
            if worst == PairDecision.ALLOW:
                worst = decision
            elif decision == PairDecision.BLOCK:
                worst = PairDecision.BLOCK
            elif decision == PairDecision.QUEUE and worst != PairDecision.BLOCK:
                worst = PairDecision.QUEUE
            blocking.append(active_role)

        if worst == PairDecision.ALLOW:
            return GateDecision(admitted=True, decision=PairDecision.ALLOW, reason="all pairs allow")

        if worst == PairDecision.DEGRADED_ALLOW:
            # Foreground SLO override — let the request through but tag it
            with self._lock:
                self._metrics.contention_degraded_allow_count += 1
            return GateDecision(
                admitted=True,
                decision=PairDecision.DEGRADED_ALLOW,
                blocking_roles=sorted(set(blocking)),
                reason=f"degraded_allow (active: {','.join(sorted(set(blocking)))})",
            )

        return GateDecision(
            admitted=False,
            decision=worst,
            blocking_roles=sorted(set(blocking)),
            reason=f"{worst.value} (active: {','.join(sorted(set(blocking)))})",
        )

    def admit(
        self,
        role: str,
        traffic_class: TrafficClass | str = TrafficClass.FOREGROUND_INTERACTIVE,
        max_queue_wait_ms: int | None = None,
    ) -> GateDecision:
        """Wait-with-budget version of `evaluate`. Polls every ~150 ms.

        Returns a `GateDecision` with `admitted=True` and `waited_s` set
        on success, or `admitted=False` with `reason="timeout"` on budget
        exhaustion.
        """
        if isinstance(traffic_class, str):
            try:
                traffic_class = TrafficClass(traffic_class)
            except ValueError:
                traffic_class = TrafficClass.BACKGROUND
        if max_queue_wait_ms is None:
            if traffic_class == TrafficClass.BACKGROUND:
                max_queue_wait_ms = _DEFAULT_BACKGROUND_WAIT_MS
            else:
                max_queue_wait_ms = _DEFAULT_FOREGROUND_WAIT_MS

        deadline = time.monotonic() + max_queue_wait_ms / 1000.0
        wait_start = time.monotonic()
        first_decision: GateDecision | None = None
        while True:
            decision = self.evaluate(role, traffic_class)
            if decision.admitted:
                decision.waited_s = time.monotonic() - wait_start
                with self._lock:
                    self._metrics.contention_wait_seconds += decision.waited_s
                    self._metrics.contention_admitted_count += 1
                return decision
            first_decision = first_decision or decision
            # Record the block in metrics on first hit
            for blocker in decision.blocking_roles:
                key = tuple(sorted([role, blocker]))
                with self._lock:
                    self._metrics.contention_blocked_count[key] += 1
            if time.monotonic() >= deadline:
                with self._lock:
                    self._metrics.contention_timeout_count += 1
                first_decision.waited_s = time.monotonic() - wait_start
                first_decision.reason = f"timeout after {first_decision.waited_s:.1f}s ({first_decision.reason})"
                return first_decision
            time.sleep(_GATE_POLL_S)

    @contextmanager
    def gate_decode(
        self,
        role: str,
        traffic_class: TrafficClass | str = TrafficClass.FOREGROUND_INTERACTIVE,
        max_queue_wait_ms: int | None = None,
    ) -> Iterator[GateDecision]:
        """Context manager — the cleanest call-site shape.

        Usage in `_real_call_impl`:
            with gate.gate_decode(role, traffic_class, max_queue_wait_ms) as d:
                if not d.admitted:
                    raise ContentionDenied(d.reason)
                # … invoke backend.infer*() …
        """
        decision = self.admit(role, traffic_class, max_queue_wait_ms)
        try:
            yield decision
        finally:
            # No release needed — the gate doesn't hold cross-process state.
            # Active-decode counts are sourced from region locks which the
            # backend.infer*() call already manages.
            pass

    # ── metrics ─────────────────────────────────────────────────────

    def metrics_snapshot(self) -> dict[str, Any]:
        """Read-only snapshot of current counters (for /dashboard exposure).

        Per-role scheduling state (quarter preference, migration counts) is
        gathered separately by the dashboard endpoint since it lives on
        `app.state.llm_primitives._backends` (per-request injection), not
        on a module-level singleton this gate can reach.
        """
        with self._lock:
            return {
                "contention_blocked_count": {
                    f"{a}+{b}": n for (a, b), n in self._metrics.contention_blocked_count.items()
                },
                "contention_wait_seconds": self._metrics.contention_wait_seconds,
                "contention_unknown_pair_count": self._metrics.contention_unknown_pair_count,
                "contention_degraded_allow_count": self._metrics.contention_degraded_allow_count,
                "contention_admitted_count": self._metrics.contention_admitted_count,
                "contention_timeout_count": self._metrics.contention_timeout_count,
                "active_decodes_by_role": dict(self._metrics.active_decodes_by_role),
                "active_instances_by_role": {r: list(idxs) for r, idxs in self._metrics.active_instances_by_role.items()},
                "matrix_status": self.matrix_health().value,
            }


# ── singleton accessor ──────────────────────────────────────────────

_GATE_SINGLETON: ContentionGate | None = None
_GATE_SINGLETON_LOCK = threading.Lock()


def get_gate() -> ContentionGate:
    """Process-wide ContentionGate. Lazy-init so import is cheap."""
    global _GATE_SINGLETON
    if _GATE_SINGLETON is None:
        with _GATE_SINGLETON_LOCK:
            if _GATE_SINGLETON is None:
                _GATE_SINGLETON = ContentionGate()
    return _GATE_SINGLETON


def reset_gate() -> None:
    """Test helper: reset the singleton."""
    global _GATE_SINGLETON
    with _GATE_SINGLETON_LOCK:
        _GATE_SINGLETON = None


class ContentionDenied(RuntimeError):
    """Raised by `_real_call_impl` when the gate denies admission past the deadline.

    Callers (chat route) should surface this as a 503 with `Retry-After`
    rather than propagating to the user as an unhandled 500.
    """
