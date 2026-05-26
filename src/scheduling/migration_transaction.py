"""WP-3: transactional KV migration state model.

Wraps the existing _slot_save → _slot_restore → _slot_erase sequence with
explicit transaction states so callers can introspect status, wait on
completion, distinguish failure modes, and honor a per-request deadline.

The state machine matches the 2026-05-25 audit refinement in
handoffs/active/within-role-placement-state-machine.md § Phase 3:

  PLANNED   → SAVING       → RESTORING      → VERIFIED      → SOURCE_ERASED → COMMITTED
                ↓               ↓                ↓                                 ↓
              ABORTED         ABORTED          ABORTED                          (terminal)

The incoming request must not be placed on the newly-freed full/quarter
topology until the transaction reaches VERIFIED; SOURCE_ERASED runs only
after restore verification. Telemetry consumers see the transaction ID +
per-state timestamps + reason for any ABORTED transitions.

This module is pure state — the actual HTTP calls live in
src/backends/concurrency_aware.py:_slot_save/_slot_restore/_slot_erase.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


class MigrationState(str, enum.Enum):
    """Per the audit refinement, states must be explicit + observable.

    Transitions (only these are legal):
      PLANNED → SAVING | ABORTED
      SAVING → RESTORING | ABORTED
      RESTORING → VERIFIED | ABORTED
      VERIFIED → SOURCE_ERASED | ABORTED      (incoming request may now be placed)
      SOURCE_ERASED → COMMITTED               (terminal success)
      * → ABORTED                              (terminal failure)
    """

    PLANNED = "planned"
    SAVING = "saving"
    RESTORING = "restoring"
    VERIFIED = "verified"            # restore confirmed; safe to release waiting placement
    SOURCE_ERASED = "source_erased"  # source KV cleared from full
    COMMITTED = "committed"          # terminal success
    ABORTED = "aborted"              # terminal failure (with reason in transaction.detail)


TERMINAL_STATES = {MigrationState.COMMITTED, MigrationState.ABORTED}


@dataclass
class MigrationTransaction:
    """Single KV migration's lifecycle. Thread-safe via internal lock.

    Public-readable fields are updated atomically with internal lock held;
    callers waiting on `event` are released exactly once when state enters
    VERIFIED (i.e. the new placement may proceed) or ABORTED (placement
    falls through to Phase 2 queue).
    """

    role: str
    session_id: str
    source_url: str
    target_quarter: int
    target_url: str
    migration_budget_ms: int = 30_000  # default 30s; per-request overrides honored
    txn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    state: MigrationState = MigrationState.PLANNED
    detail: str = ""
    started_at: float = field(default_factory=time.monotonic)
    state_history: list[tuple[MigrationState, float]] = field(default_factory=list)

    # Signaled when state enters VERIFIED *or* ABORTED — placement waiters
    # check `is_safe_to_proceed` to decide whether to dispatch.
    event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        # Record the initial PLANNED entry so state_history is never empty.
        self.state_history.append((self.state, self.started_at))

    # ── state transitions (caller must use these, not assign .state) ──

    def advance(self, new_state: MigrationState, detail: str = "") -> bool:
        """Attempt to transition to `new_state`. Returns True if the
        transition was legal and applied. False if it was illegal
        (caller should treat as no-op + log)."""
        with self._lock:
            if not self._is_legal_transition(self.state, new_state):
                return False
            self.state = new_state
            if detail:
                self.detail = detail
            self.state_history.append((new_state, time.monotonic()))
            if new_state is MigrationState.VERIFIED or new_state is MigrationState.ABORTED:
                # Wake any placement waiters.
                self.event.set()
            return True

    @staticmethod
    def _is_legal_transition(current: MigrationState, target: MigrationState) -> bool:
        if target is MigrationState.ABORTED:
            return current not in TERMINAL_STATES
        legal: dict[MigrationState, set[MigrationState]] = {
            MigrationState.PLANNED: {MigrationState.SAVING},
            MigrationState.SAVING: {MigrationState.RESTORING},
            MigrationState.RESTORING: {MigrationState.VERIFIED},
            MigrationState.VERIFIED: {MigrationState.SOURCE_ERASED},
            MigrationState.SOURCE_ERASED: {MigrationState.COMMITTED},
        }
        return target in legal.get(current, set())

    # ── observers ────────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self.state in TERMINAL_STATES

    @property
    def is_safe_to_proceed(self) -> bool:
        """True iff the migrated session's KV is verified on the quarter
        (so the source full instance can be claimed by the waiting request)."""
        with self._lock:
            return self.state in (
                MigrationState.VERIFIED,
                MigrationState.SOURCE_ERASED,
                MigrationState.COMMITTED,
            )

    @property
    def elapsed_ms(self) -> float:
        with self._lock:
            return (time.monotonic() - self.started_at) * 1000

    def wait_for_completion(self, deadline_s: Optional[float] = None) -> MigrationState:
        """Block until the transaction reaches VERIFIED or a terminal state,
        or until `deadline_s` (monotonic seconds) elapses. Returns the
        final observed state — callers compare against ABORTED to detect
        failure. `deadline_s=None` honors `migration_budget_ms`.
        """
        if deadline_s is None:
            deadline_s = self.started_at + (self.migration_budget_ms / 1000.0)
        timeout_s = max(0.0, deadline_s - time.monotonic())
        self.event.wait(timeout=timeout_s)
        with self._lock:
            return self.state

    # ── telemetry snapshot ────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "txn_id": self.txn_id,
                "role": self.role,
                "session_id": self.session_id,
                "source_url": self.source_url,
                "target_quarter": self.target_quarter,
                "target_url": self.target_url,
                "state": self.state.value,
                "detail": self.detail,
                "elapsed_ms": (time.monotonic() - self.started_at) * 1000,
                "state_history": [
                    {"state": s.value, "ts_offset_ms": (ts - self.started_at) * 1000}
                    for s, ts in self.state_history
                ],
                "is_terminal": self.state in TERMINAL_STATES,
                "is_safe_to_proceed": self.state in (
                    MigrationState.VERIFIED,
                    MigrationState.SOURCE_ERASED,
                    MigrationState.COMMITTED,
                ),
            }
