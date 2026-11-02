"""Owner-bound, in-process KV preemption *selection*; no runtime wiring.

This registry never cancels work, releases reservations, requeues requests or
establishes that a server has stopped. An owner must use the same registry lock
boundary before exposing output. Owner IDs are correlation tokens supplied by
trusted callers, not a cross-process authentication mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


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


@dataclass(frozen=True, slots=True)
class Victim:
    """A selection, not a cancellation or evidence of freed capacity."""

    waiter_id: str
    attempt_id: str
    request_id: str
    owner_id: str
    server_url: str


@dataclass(slots=True)
class _Entry:
    attempt: Attempt
    output_committed: bool = False
    selected_by: str | None = None
    finished: bool = False


class AttemptRegistry:
    """Serialize owner updates and select at most one victim per waiter.

    A returned decision is retained even after the victim finishes: repeated
    calls by the same waiter return the same victim, never a second one. A
    different waiter cannot select an already selected attempt. Registry scope
    is one process; cross-worker ownership must be implemented separately.
    """

    def __init__(self, *, max_preemptions: int = 1) -> None:
        if type(max_preemptions) is not int or max_preemptions < 0:
            raise ValueError("max_preemptions must be an integer >= 0")
        self._max_preemptions = max_preemptions
        self._lock = Lock()
        self._entries: dict[str, _Entry] = {}
        self._decisions: dict[str, Victim] = {}
        self._waiter_facts: dict[str, tuple[str, int, int]] = {}

    def register(self, attempt: Attempt) -> None:
        with self._lock:
            if attempt.attempt_id in self._entries:
                raise ValueError("attempt_id already registered")
            if any(not entry.finished and entry.attempt.request_id == attempt.request_id
                   for entry in self._entries.values()):
                raise ValueError("request already has an active attempt")
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
            victim = Victim(waiter_id, attempt.attempt_id, attempt.request_id,
                            attempt.owner_id, attempt.server_url)
            self._decisions[waiter_id] = victim
            return victim
