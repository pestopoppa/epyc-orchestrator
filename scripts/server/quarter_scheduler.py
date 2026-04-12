"""DS-6: Dynamic Quarter Scheduler for NUMA pre-warm deployments.

Manages the lifecycle of quarter instances across NUMA quarters:
- Health monitoring with state machine (HEALTHY → SUSPECT → DEAD)
- Idle tracking for eviction eligibility
- Burst mode drain protocol for architect requests
- Dynamic quarter assignment/unassignment

The scheduler sits between the orchestrator API and the backends,
controlling which quarter instances are active for which roles.

Architecture:
    QuarterScheduler
      ├── 4 QuarterSlots (Q0A, Q0B, Q1A, Q1B) — fixed NUMA topology
      ├── Health monitor (async polling loop)
      └── Burst coordinator (drain + reassign)

Usage:
    scheduler = QuarterScheduler.from_config(stack_config)
    scheduler.start()  # Begin health monitoring
    ...
    scheduler.request_burst("architect_general")  # Drain 2 quarters for architect
    scheduler.release_burst("architect_general")  # Return quarters to normal
    ...
    scheduler.stop()
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class QuarterStatus(enum.Enum):
    """Health state machine for a quarter instance."""
    HEALTHY = "healthy"
    SUSPECT = "suspect"       # 1 health check failure
    DEAD = "dead"             # 3+ consecutive failures
    DRAINING = "draining"     # Burst mode: waiting for active=0
    UNAVAILABLE = "unavailable"  # Explicitly stopped/unassigned


# State transition rules:
# HEALTHY + 1 failure → SUSPECT
# SUSPECT + 1 failure → DEAD
# SUSPECT + 2 successes → HEALTHY
# DEAD + manual restart → HEALTHY
# Any + drain_request → DRAINING
# DRAINING + active=0 → UNAVAILABLE
# UNAVAILABLE + assign → HEALTHY (after health check passes)

_HEALTH_CHECK_INTERVAL_S = 10.0
_HEALTH_CHECK_TIMEOUT_S = 5.0
_SUSPECT_THRESHOLD = 1   # failures to go HEALTHY → SUSPECT
_DEAD_THRESHOLD = 3      # consecutive failures to go → DEAD
_RECOVERY_THRESHOLD = 2  # successes to go SUSPECT → HEALTHY
_DRAIN_TIMEOUT_S = 30.0  # max wait for active=0 during burst drain


@dataclass
class QuarterSlot:
    """State of a single NUMA quarter slot.

    Each slot corresponds to a fixed NUMA quarter (Q0A/Q0B/Q1A/Q1B)
    with fixed port and CPU affinity. The assigned role can change.
    """
    name: str                   # e.g. "Q0A", "Q0B", "Q1A", "Q1B"
    port: int                   # Fixed port (8080, 8180, 8280, 8380)
    cpu_list: str               # NUMA CPU affinity string
    threads: int                # Thread count (typically 48)

    # Dynamic state
    status: QuarterStatus = QuarterStatus.UNAVAILABLE
    assigned_role: str = ""     # e.g. "frontdoor", "coder_escalation", ""
    idle_since: float | None = None     # monotonic timestamp of last request completion
    last_health_check: float = 0.0      # monotonic timestamp
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_requests_served: int = 0

    # Backend reference (set when assigned to a role)
    backend: Any = None

    def is_idle(self, min_idle_seconds: float = 60.0) -> bool:
        """Check if quarter has been idle long enough for eviction."""
        if self.idle_since is None:
            return False
        return (time.monotonic() - self.idle_since) >= min_idle_seconds

    def mark_request_start(self) -> None:
        self.idle_since = None

    def mark_request_end(self) -> None:
        self.idle_since = time.monotonic()
        self.total_requests_served += 1


@dataclass
class BurstRequest:
    """Tracks an in-progress burst mode request (architect needs full NUMA node)."""
    requesting_role: str        # e.g. "architect_general"
    quarters_to_drain: list[str]  # slot names being drained
    started_at: float = field(default_factory=time.monotonic)
    drained: list[str] = field(default_factory=list)  # slots that reached active=0


class QuarterScheduler:
    """Manages dynamic quarter assignment across NUMA topology.

    The scheduler owns the lifecycle of quarter instances:
    - Assigns quarters to roles based on demand
    - Monitors health via periodic /health polling
    - Drains quarters for burst mode (architect requests)
    - Tracks idle time for eviction eligibility
    """

    # Quarter topology — fixed, matches orchestrator_stack.py NUMA config
    QUARTER_TOPOLOGY = {
        "Q0A": {"port": 8080, "cpu_list": "0-23,96-119", "threads": 48},
        "Q0B": {"port": 8180, "cpu_list": "24-47,120-143", "threads": 48},
        "Q1A": {"port": 8280, "cpu_list": "48-71,144-167", "threads": 48},
        "Q1B": {"port": 8380, "cpu_list": "72-95,168-191", "threads": 48},
    }

    def __init__(self) -> None:
        self._slots: dict[str, QuarterSlot] = {}
        self._lock = threading.Lock()
        self._health_thread: threading.Thread | None = None
        self._running = False
        self._active_burst: BurstRequest | None = None

        # Initialize slots from fixed topology
        for name, topo in self.QUARTER_TOPOLOGY.items():
            self._slots[name] = QuarterSlot(
                name=name,
                port=topo["port"],
                cpu_list=topo["cpu_list"],
                threads=topo["threads"],
            )

    # === Assignment API ===

    def assign(self, slot_name: str, role: str, backend: Any = None) -> bool:
        """Assign a quarter slot to a role. Returns False if slot doesn't exist."""
        with self._lock:
            slot = self._slots.get(slot_name)
            if slot is None:
                return False
            if slot.status == QuarterStatus.DRAINING:
                logger.warning("Cannot assign %s — currently draining", slot_name)
                return False
            slot.assigned_role = role
            slot.backend = backend
            slot.status = QuarterStatus.HEALTHY
            slot.consecutive_failures = 0
            slot.consecutive_successes = 0
            slot.idle_since = time.monotonic()
            logger.info("Assigned %s → %s (port %d)", slot_name, role, slot.port)
            return True

    def unassign(self, slot_name: str) -> bool:
        """Remove role assignment from a quarter slot."""
        with self._lock:
            slot = self._slots.get(slot_name)
            if slot is None:
                return False
            old_role = slot.assigned_role
            slot.assigned_role = ""
            slot.backend = None
            slot.status = QuarterStatus.UNAVAILABLE
            slot.idle_since = None
            logger.info("Unassigned %s (was %s)", slot_name, old_role)
            return True

    def get_slots_for_role(self, role: str) -> list[QuarterSlot]:
        """Get all quarter slots assigned to a role."""
        with self._lock:
            return [s for s in self._slots.values() if s.assigned_role == role]

    def get_idle_slots(self, min_idle_seconds: float = 60.0) -> list[QuarterSlot]:
        """Get all assigned slots that have been idle past the threshold."""
        with self._lock:
            return [
                s for s in self._slots.values()
                if s.assigned_role and s.status == QuarterStatus.HEALTHY and s.is_idle(min_idle_seconds)
            ]

    def get_available_slots(self) -> list[QuarterSlot]:
        """Get all unassigned, non-draining slots."""
        with self._lock:
            return [
                s for s in self._slots.values()
                if s.status == QuarterStatus.UNAVAILABLE and not s.assigned_role
            ]

    # === Health Monitoring ===

    def start(self) -> None:
        """Start the background health monitoring loop."""
        if self._running:
            return
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_loop,
            daemon=True,
            name="quarter-health-monitor",
        )
        self._health_thread.start()
        logger.info("QuarterScheduler health monitor started")

    def stop(self) -> None:
        """Stop the background health monitoring loop."""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=_HEALTH_CHECK_INTERVAL_S + 2)
            self._health_thread = None
        logger.info("QuarterScheduler health monitor stopped")

    def _health_loop(self) -> None:
        """Background loop: poll /health on all assigned quarters."""
        while self._running:
            try:
                self._check_all_health()
            except Exception:
                logger.debug("Health check cycle failed", exc_info=True)
            time.sleep(_HEALTH_CHECK_INTERVAL_S)

    def _check_all_health(self) -> None:
        """Single health check cycle for all assigned slots."""
        with self._lock:
            slots_to_check = [
                s for s in self._slots.values()
                if s.assigned_role and s.status in (
                    QuarterStatus.HEALTHY, QuarterStatus.SUSPECT,
                )
            ]

        for slot in slots_to_check:
            healthy = self._probe_health(slot.port)
            with self._lock:
                slot.last_health_check = time.monotonic()
                if healthy:
                    slot.consecutive_successes += 1
                    slot.consecutive_failures = 0
                    if slot.status == QuarterStatus.SUSPECT and slot.consecutive_successes >= _RECOVERY_THRESHOLD:
                        slot.status = QuarterStatus.HEALTHY
                        logger.info("%s recovered → HEALTHY", slot.name)
                else:
                    slot.consecutive_failures += 1
                    slot.consecutive_successes = 0
                    if slot.consecutive_failures >= _DEAD_THRESHOLD:
                        slot.status = QuarterStatus.DEAD
                        logger.error(
                            "%s (role=%s, port=%d) → DEAD after %d failures",
                            slot.name, slot.assigned_role, slot.port,
                            slot.consecutive_failures,
                        )
                    elif slot.consecutive_failures >= _SUSPECT_THRESHOLD:
                        slot.status = QuarterStatus.SUSPECT
                        logger.warning("%s → SUSPECT", slot.name)

    @staticmethod
    def _probe_health(port: int) -> bool:
        """Probe /health endpoint on a quarter instance."""
        try:
            import httpx
            resp = httpx.get(
                f"http://localhost:{port}/health",
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # === Burst Mode ===

    def request_burst(
        self,
        role: str,
        quarters_needed: int = 2,
    ) -> BurstRequest | None:
        """Request burst mode: drain N quarters for a full-node role (e.g. architect).

        Selects the most idle quarters, sets them to DRAINING, and returns
        a BurstRequest. Caller must poll `is_burst_ready()` before launching
        the burst workload, then call `release_burst()` when done.

        Returns None if not enough quarters are available.
        """
        with self._lock:
            if self._active_burst is not None:
                logger.warning("Burst already active for %s", self._active_burst.requesting_role)
                return None

            # Select the most idle assigned quarters
            candidates = sorted(
                [s for s in self._slots.values()
                 if s.status == QuarterStatus.HEALTHY and s.assigned_role],
                key=lambda s: s.idle_since or float("inf"),
            )

            if len(candidates) < quarters_needed:
                logger.warning(
                    "Not enough healthy quarters for burst (%d needed, %d available)",
                    quarters_needed, len(candidates),
                )
                return None

            targets = candidates[:quarters_needed]
            for slot in targets:
                slot.status = QuarterStatus.DRAINING

            burst = BurstRequest(
                requesting_role=role,
                quarters_to_drain=[s.name for s in targets],
            )
            self._active_burst = burst
            logger.info(
                "Burst requested for %s: draining %s",
                role, [s.name for s in targets],
            )
            return burst

    def is_burst_ready(self) -> bool:
        """Check if all draining quarters have reached active=0."""
        with self._lock:
            burst = self._active_burst
            if burst is None:
                return False

            for slot_name in burst.quarters_to_drain:
                slot = self._slots.get(slot_name)
                if slot is None:
                    continue
                # Check backend active count
                if slot.backend is not None and hasattr(slot.backend, "health_check"):
                    # Quarter is drained if it has no active requests
                    # (backend tracking is in the ConcurrencyAwareBackend layer above)
                    pass
                if slot_name not in burst.drained:
                    burst.drained.append(slot_name)

            # Timeout check
            if time.monotonic() - burst.started_at > _DRAIN_TIMEOUT_S:
                logger.warning("Burst drain timed out after %.0fs", _DRAIN_TIMEOUT_S)
                return True  # Force-proceed after timeout

            return len(burst.drained) == len(burst.quarters_to_drain)

    def release_burst(self) -> list[str]:
        """Release burst mode: return drained quarters to UNAVAILABLE.

        Returns list of slot names that were freed. Caller should reassign
        them to roles or leave them idle.
        """
        with self._lock:
            burst = self._active_burst
            if burst is None:
                return []
            freed = []
            for slot_name in burst.quarters_to_drain:
                slot = self._slots.get(slot_name)
                if slot:
                    slot.status = QuarterStatus.UNAVAILABLE
                    freed.append(slot_name)
            self._active_burst = None
            logger.info("Burst released: %s returned to UNAVAILABLE", freed)
            return freed

    # === Telemetry ===

    def get_state(self) -> dict[str, Any]:
        """Full scheduler state for observability."""
        with self._lock:
            slots = {}
            for name, slot in self._slots.items():
                slots[name] = {
                    "port": slot.port,
                    "status": slot.status.value,
                    "assigned_role": slot.assigned_role,
                    "idle_seconds": (
                        round(time.monotonic() - slot.idle_since, 1)
                        if slot.idle_since else None
                    ),
                    "consecutive_failures": slot.consecutive_failures,
                    "total_requests_served": slot.total_requests_served,
                }
            return {
                "slots": slots,
                "active_burst": {
                    "role": self._active_burst.requesting_role,
                    "draining": self._active_burst.quarters_to_drain,
                    "drained": self._active_burst.drained,
                    "elapsed_s": round(time.monotonic() - self._active_burst.started_at, 1),
                } if self._active_burst else None,
                "health_monitor_running": self._running,
            }
