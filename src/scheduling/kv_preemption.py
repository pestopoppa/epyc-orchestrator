"""Owner-bound, in-process KV preemption *selection*; no runtime wiring.

This registry never cancels work, releases reservations, requeues requests or
establishes that a server has stopped. An owner must use the same registry lock
boundary before exposing output. Owner IDs are correlation tokens supplied by
trusted callers, not a cross-process authentication mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
from threading import Lock
from typing import Callable


@dataclass(frozen=True, slots=True)
class Attempt:
    """Immutable attempt identity and facts certified by its owner.

    Larger ``priority`` means higher priority. ``queue_sequence`` is the
    original arrival order, retained across attempts; smaller is older.
    """

    attempt_id: str
    request_id: str
    owner_id: str
    server_url: str
    priority: int
    queue_sequence: int
    preemptions: int = 0
    cancellable: bool = False
    replayable: bool = False
    output_uncommitted_verified: bool = False
    server_task_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("attempt_id", "request_id", "owner_id", "server_url"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("priority", "queue_sequence", "preemptions"):
            value = getattr(self, name)
            if type(value) is not int or (name != "priority" and value < 0):
                raise ValueError(f"{name} must be an integer" + (" >= 0" if name != "priority" else ""))
        if any(type(flag) is not bool for flag in
               (self.cancellable, self.replayable, self.output_uncommitted_verified)):
            raise ValueError("eligibility flags must be booleans")
        if self.server_task_id is not None and (
            not isinstance(self.server_task_id, str) or not self.server_task_id.strip()
        ):
            raise ValueError("server_task_id must be a nonempty string when present")


@dataclass(frozen=True, slots=True)
class Victim:
    """A selection, not a cancellation or evidence of freed capacity."""

    waiter_id: str
    attempt_id: str
    request_id: str
    owner_id: str
    server_url: str


@dataclass(frozen=True, slots=True)
class CancelCommand:
    """Owner-facing cancellation intent; issuing it does not cancel a task."""

    victim: Victim
    server_task_id: str
    nonce: str


@dataclass(frozen=True, slots=True)
class NativeDrainWitness:
    """A task-bound receipt requiring independent verification."""

    attempt_id: str
    server_url: str
    server_task_id: str
    cancel_nonce: str
    kind: str
    receipt_sha256: str


@dataclass(slots=True)
class _Entry:
    attempt: Attempt
    output_committed: bool = False
    selected_by: str | None = None
    finished: bool = False
    cancel_command: CancelCommand | None = None
    cancel_acknowledged: bool = False
    drain_witness: NativeDrainWitness | None = None


class AttemptRegistry:
    """Serialize owner updates and select at most one victim per waiter.

    A returned decision is retained even after the victim finishes: repeated
    calls by the same waiter return the same victim, never a second one. A
    different waiter cannot select an already selected attempt. Registry scope
    is one process; cross-worker ownership must be implemented separately.
    """

    def __init__(self, *, max_preemptions: int = 1,
                 verify_drain: Callable[[NativeDrainWitness], bool] | None = None) -> None:
        if type(max_preemptions) is not int or max_preemptions < 0:
            raise ValueError("max_preemptions must be an integer >= 0")
        self._max_preemptions = max_preemptions
        self._verify_drain = verify_drain
        self._lock = Lock()
        self._entries: dict[str, _Entry] = {}
        self._decisions: dict[str, Victim] = {}
        self._waiter_facts: dict[str, tuple[str, int, int]] = {}
        self._request_preemption_floor: dict[str, int] = {}

    def register(self, attempt: Attempt) -> None:
        with self._lock:
            if attempt.attempt_id in self._entries:
                raise ValueError("attempt_id already registered")
            if any(not entry.finished and entry.attempt.request_id == attempt.request_id
                   for entry in self._entries.values()):
                raise ValueError("request already has an active attempt")
            if any(entry.attempt.request_id == attempt.request_id
                   and entry.selected_by is not None and entry.drain_witness is None
                   for entry in self._entries.values()):
                raise ValueError("selected predecessor has no verified native drain")
            if attempt.preemptions < self._request_preemption_floor.get(attempt.request_id, 0):
                raise ValueError("preemption count fell below request history")
            self._request_preemption_floor[attempt.request_id] = attempt.preemptions
            self._entries[attempt.attempt_id] = _Entry(attempt)

    def _owned(self, attempt_id: str, owner_id: str) -> _Entry:
        entry = self._entries.get(attempt_id)
        if entry is None or entry.attempt.owner_id != owner_id:
            raise ValueError("unknown attempt or owner")
        return entry

    def commit_output(self, attempt_id: str, owner_id: str) -> bool:
        """Claim the response-commit boundary; false means do not expose output.

        Owners must call this *before* external headers or content are exposed.
        A selected or finished attempt cannot subsequently commit output.
        """
        with self._lock:
            entry = self._owned(attempt_id, owner_id)
            if entry.selected_by is not None or entry.finished:
                return False
            entry.output_committed = True
            return True

    def finish(self, attempt_id: str, owner_id: str) -> None:
        """Record owner-side terminal state; this does not free server capacity."""
        with self._lock:
            self._owned(attempt_id, owner_id).finished = True

    def select(self, *, waiter_id: str, server_url: str, priority: int,
               queue_sequence: int) -> Victim | None:
        """Select a lower-priority or same-priority younger eligible attempt.

        Lowest priority wins; within a priority, the youngest attempt wins.
        Selection is advisory. The caller still needs separate owner cancellation,
        server-drain proof, reservation release and idempotent retry protocols.
        """
        if not isinstance(waiter_id, str) or not waiter_id.strip():
            raise ValueError("waiter_id must be a nonempty string")
        if not isinstance(server_url, str) or not server_url.strip():
            raise ValueError("server_url must be a nonempty string")
        if type(priority) is not int or type(queue_sequence) is not int or queue_sequence < 0:
            raise ValueError("invalid waiter priority or sequence")
        facts = (server_url, priority, queue_sequence)
        with self._lock:
            previous = self._waiter_facts.get(waiter_id)
            if previous is not None and previous != facts:
                raise ValueError("waiter facts changed")
            self._waiter_facts[waiter_id] = facts
            if waiter_id in self._decisions:
                return self._decisions[waiter_id]
            candidates = [entry for entry in self._entries.values()
                          if not entry.finished and not entry.output_committed
                          and entry.selected_by is None
                          and entry.attempt.server_url == server_url
                          and entry.attempt.request_id != waiter_id
                          and entry.attempt.cancellable and entry.attempt.replayable
                          and entry.attempt.output_uncommitted_verified
                          and entry.attempt.server_task_id is not None
                          and entry.attempt.preemptions < self._max_preemptions
                          and (entry.attempt.priority < priority or (
                              entry.attempt.priority == priority
                              and entry.attempt.queue_sequence > queue_sequence))]
            if not candidates:
                return None
            entry = min(candidates, key=lambda e: (e.attempt.priority,
                                                    -e.attempt.queue_sequence,
                                                    e.attempt.attempt_id))
            entry.selected_by = waiter_id
            attempt = entry.attempt
            self._request_preemption_floor[attempt.request_id] = attempt.preemptions + 1
            victim = Victim(waiter_id, attempt.attempt_id, attempt.request_id,
                            attempt.owner_id, attempt.server_url)
            self._decisions[waiter_id] = victim
            return victim

    def begin_cancel(self, victim: Victim, owner_id: str) -> CancelCommand:
        """Issue one owner-bound intent for an existing task-bound selection.

        The owner, not this class, performs any cancellation. A finished task
        without a previously issued intent cannot begin a new cancellation.
        """
        with self._lock:
            entry = self._owned(victim.attempt_id, owner_id)
            if self._decisions.get(victim.waiter_id) != victim:
                raise ValueError("unknown victim selection")
            if entry.cancel_command is not None:
                return entry.cancel_command
            if entry.finished or entry.attempt.server_task_id is None:
                raise ValueError("attempt finished or server task identity missing")
            entry.cancel_command = CancelCommand(victim, entry.attempt.server_task_id,
                                                  secrets.token_hex(16))
            return entry.cancel_command

    def acknowledge_cancel(self, command: CancelCommand, owner_id: str) -> None:
        """Record owner acknowledgement, not proof of server-side drain."""
        with self._lock:
            entry = self._owned(command.victim.attempt_id, owner_id)
            if entry.cancel_command != command:
                raise ValueError("unknown cancellation intent")
            entry.cancel_acknowledged = True

    def confirm_native_drain(self, command: CancelCommand, owner_id: str,
                             witness: NativeDrainWitness) -> bool:
        """Verify a matching native receipt; never release a reservation.

        No verifier, a verifier error, an unrecognized result or missing receipt
        fails closed. Verify outside the lock, then recheck state before recording.
        The verifier must validate native source, receipt custody and that the
        observation follows this cancellation intent; a supplied digest alone
        is not proof. This registry does not infer drain from a socket close.
        """
        with self._lock:
            entry = self._owned(command.victim.attempt_id, owner_id)
            if entry.cancel_command != command or not entry.cancel_acknowledged:
                return False
            if (witness.attempt_id != command.victim.attempt_id
                    or witness.server_url != command.victim.server_url
                    or witness.server_task_id != command.server_task_id
                    or witness.cancel_nonce != command.nonce
                    or witness.kind not in {"terminal", "absent"}
                    or not isinstance(witness.receipt_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", witness.receipt_sha256) is None):
                return False
            if entry.drain_witness is not None:
                return entry.drain_witness == witness
        if self._verify_drain is None:
            return False
        try:
            if self._verify_drain(witness) is not True:
                return False
        except Exception:
            return False
        with self._lock:
            entry = self._owned(command.victim.attempt_id, owner_id)
            if entry.cancel_command != command or not entry.cancel_acknowledged:
                return False
            if entry.drain_witness is not None:
                return entry.drain_witness == witness
            entry.drain_witness = witness
            return True


class OutputCommitObservation:
    """An owner-bound passthrough observation, deliberately never preemptible.

    The current native API does not provide a request-bound compare-and-cancel
    or a verifiable drain receipt. This adapter records the response boundary
    without advertising cancellation, replay, or released KV capacity.
    """

    def __init__(self, registry: AttemptRegistry, *, attempt_id: str, request_id: str,
                 owner_id: str, server_url: str, priority: int, queue_sequence: int) -> None:
        self._registry = registry
        self._attempt_id = attempt_id
        self._owner_id = owner_id
        registry.register(Attempt(attempt_id=attempt_id, request_id=request_id,
                                  owner_id=owner_id, server_url=server_url,
                                  priority=priority, queue_sequence=queue_sequence))

    def commit_before_output(self) -> bool:
        """Mark the boundary before handing headers or bytes to the response."""
        return self._registry.commit_output(self._attempt_id, self._owner_id)

    def finish(self) -> None:
        """Mark the local owner finished; do not infer native drain."""
        self._registry.finish(self._attempt_id, self._owner_id)
