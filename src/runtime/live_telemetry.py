"""Bounded, non-blocking lifecycle telemetry for the live dashboard.

This is deliberately an in-process observability reducer, not a durable event
store.  Producers take one short mutex, update the authoritative request state,
and return without filesystem or network I/O.  Dashboard SSE consumers sample
the reduced state and independently reconcile it with llama-server ``/slots``.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from typing import Any, Iterator

LIFECYCLE_TRANSITIONS = frozenset(
    {
        "queued",
        "route_selected",
        "placement_lease_acquired",
        "placement_lease_released",
        "backend_dispatched",
        "first_output",
        "completed",
        "failed",
        "rerouted",
        "escalated",
        "api_request_started",
        "api_request_finished",
    }
)
_OWNERSHIP_TRANSITIONS = frozenset({"placement_lease_acquired", "placement_lease_released"})
_TERMINAL_TRANSITIONS = frozenset({"completed", "failed"})
_NON_EVICTABLE_TRANSITIONS = (
    _OWNERSHIP_TRANSITIONS | _TERMINAL_TRANSITIONS | {"api_request_started", "api_request_finished"}
)
_IDENTITY_FIELDS = (
    "request_id",
    "task_id",
    "batch_id",
    "role",
    "model",
    "port",
    "lease_token",
)

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "live_telemetry_context",
    default={},
)


def _clean_identity(values: dict[str, Any]) -> dict[str, Any]:
    return {
        field: values.get(field)
        for field in _IDENTITY_FIELDS
        if values.get(field) not in (None, "")
    }


def bind_lifecycle_context(**identity: Any) -> contextvars.Token:
    """Bind request identity until ``reset_lifecycle_context`` is called."""
    merged = dict(_context.get())
    merged.update(_clean_identity(identity))
    return _context.set(merged)


def reset_lifecycle_context(token: contextvars.Token) -> None:
    _context.reset(token)


@contextmanager
def lifecycle_context(**identity: Any) -> Iterator[None]:
    token = bind_lifecycle_context(**identity)
    try:
        yield
    finally:
        reset_lifecycle_context(token)


def current_lifecycle_context() -> dict[str, Any]:
    return dict(_context.get())


class LiveTelemetryReducer:
    """Thread-safe bounded transition queue plus authoritative state reducer."""

    def __init__(
        self,
        *,
        queue_capacity: int = 1024,
        history_capacity: int = 512,
        request_capacity: int = 512,
    ) -> None:
        self.queue_capacity = max(1, int(queue_capacity))
        self.history_capacity = max(1, int(history_capacity))
        self.request_capacity = max(1, int(request_capacity))
        self._lock = threading.Lock()
        self._pending: deque[dict[str, Any]] = deque()
        self._history: deque[dict[str, Any]] = deque()
        self._requests: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._active_leases: dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self._frame_sequence = 0
        self._overflow_total = 0
        self._overflow_critical = 0
        self._history_evictions = 0
        self._history_critical_evictions = 0
        self._request_evictions = 0
        self._request_retirements = 0

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    @staticmethod
    def _request_key(event: dict[str, Any]) -> str:
        for field in ("request_id", "task_id", "batch_id"):
            value = event.get(field)
            if value not in (None, ""):
                return f"{field}:{value}"
        return f"anonymous:{event['sequence']}"

    def _append_pending(self, event: dict[str, Any]) -> None:
        if len(self._pending) >= self.queue_capacity:
            evict_at = next(
                (
                    idx
                    for idx, queued in enumerate(self._pending)
                    if queued.get("transition") not in _NON_EVICTABLE_TRANSITIONS
                ),
                None,
            )
            if evict_at is None:
                evicted = self._pending.popleft()
                if evicted.get("transition") in _NON_EVICTABLE_TRANSITIONS:
                    self._overflow_critical += 1
            else:
                self._pending.rotate(-evict_at)
                self._pending.popleft()
                self._pending.rotate(evict_at)
            self._overflow_total += 1
        self._pending.append(event)

    def _append_history(self, event: dict[str, Any]) -> None:
        if len(self._history) >= self.history_capacity:
            evict_at = next(
                (
                    idx
                    for idx, retained in enumerate(self._history)
                    if retained.get("transition") not in _NON_EVICTABLE_TRANSITIONS
                ),
                None,
            )
            if evict_at is None:
                evicted = self._history.popleft()
                if evicted.get("transition") in _NON_EVICTABLE_TRANSITIONS:
                    self._history_critical_evictions += 1
            else:
                self._history.rotate(-evict_at)
                self._history.popleft()
                self._history.rotate(evict_at)
            self._history_evictions += 1
        self._history.append(event)

    def _reduce_request(self, event: dict[str, Any]) -> None:
        key = self._request_key(event)
        state = self._requests.pop(key, None) or {
            "request_id": event.get("request_id"),
            "task_id": event.get("task_id"),
            "batch_id": event.get("batch_id"),
            "role": event.get("role"),
            "model": event.get("model"),
            "port": event.get("port"),
            "lease_tokens": [],
            "transitions": [],
            "terminal": False,
            "api_active": False,
            "api_seen": False,
            "backend_dispatched": False,
        }
        for field in _IDENTITY_FIELDS:
            value = event.get(field)
            if field != "lease_token" and value not in (None, ""):
                state[field] = value
        transition = str(event["transition"])
        lease_token = event.get("lease_token")
        if transition == "placement_lease_acquired" and lease_token:
            if lease_token not in state["lease_tokens"]:
                state["lease_tokens"].append(lease_token)
            self._active_leases[str(lease_token)] = {
                field: event.get(field) for field in _IDENTITY_FIELDS
            } | {
                "sequence": event["sequence"],
                "source_ts": event["source_ts"],
                "details": event.get("details") or {},
            }
        elif transition == "placement_lease_released" and lease_token:
            state["lease_tokens"] = [
                token for token in state["lease_tokens"] if token != lease_token
            ]
            self._active_leases.pop(str(lease_token), None)
        if transition == "api_request_started":
            state["api_active"] = True
            state["api_seen"] = True
            state["api_started_ts"] = event["source_ts"]
        elif transition == "api_request_finished":
            state["api_active"] = False
            state["api_finished_ts"] = event["source_ts"]
            state["api_outcome"] = (event.get("details") or {}).get("outcome")
        elif transition == "backend_dispatched":
            state["backend_dispatched"] = True
        if transition in _TERMINAL_TRANSITIONS:
            state["terminal"] = True
            state["outcome"] = transition
        state["stage"] = transition
        state["last_sequence"] = event["sequence"]
        state["source_ts"] = event["source_ts"]
        state["transitions"] = (state["transitions"] + [event["sequence"]])[-32:]
        self._requests[key] = state
        while len(self._requests) > self.request_capacity:
            retire_key = next(
                (
                    request_key
                    for request_key, candidate in self._requests.items()
                    if not candidate.get("api_seen")
                    and not candidate.get("lease_tokens")
                    and candidate.get("stage") == "placement_lease_released"
                ),
                None,
            )
            if retire_key is not None:
                self._requests.pop(retire_key, None)
                self._request_retirements += 1
            else:
                self._requests.popitem(last=False)
                self._request_evictions += 1

    def emit(self, transition: str, **fields: Any) -> dict[str, Any]:
        if transition not in LIFECYCLE_TRANSITIONS:
            raise ValueError(f"unknown lifecycle transition: {transition}")
        source_ts = float(fields.pop("source_ts", time.time()))
        source_monotonic_ns = int(fields.pop("source_monotonic_ns", time.monotonic_ns()))
        merged = current_lifecycle_context()
        merged.update({key: value for key, value in fields.items() if value is not None})
        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "transition": transition,
                "source_ts": source_ts,
                "source_monotonic_ns": source_monotonic_ns,
                "process_id": os.getpid(),
                **{field: merged.get(field) for field in _IDENTITY_FIELDS},
                "details": dict(merged.get("details") or {}),
            }
            self._reduce_request(event)
            self._append_pending(event)
            self._append_history(event)
            return dict(event)

    def frame(self) -> dict[str, Any]:
        """Return one sequenced immutable snapshot of the reduced state."""
        with self._lock:
            self._frame_sequence += 1
            pending = list(self._pending)
            self._pending.clear()
            history = [dict(event) for event in self._history]
            requests = [dict(state) for state in self._requests.values()]
            active_leases = [dict(lease) for lease in self._active_leases.values()]
            degraded = bool(
                self._overflow_total or self._history_critical_evictions or self._request_evictions
            )
            batch_activity = self._batch_activity_unlocked(requests)
            return {
                "schema_version": "live_telemetry.v1",
                "process_id": os.getpid(),
                "frame_sequence": self._frame_sequence,
                "event_sequence": self._sequence,
                "source_ts": time.time(),
                "requests": requests,
                "active_leases": active_leases,
                "batch_activity": batch_activity,
                "transitions": history,
                "pending_transitions": pending,
                "degraded": degraded,
                "overflow": {
                    "total": self._overflow_total,
                    "critical": self._overflow_critical,
                    "history_evictions": self._history_evictions,
                    "history_critical_evictions": self._history_critical_evictions,
                    "request_evictions": self._request_evictions,
                    "request_retirements": self._request_retirements,
                    "state_preserved": True,
                },
            }

    def _batch_activity_unlocked(
        self, requests: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        request_states = requests if requests is not None else list(self._requests.values())
        batches: dict[str, dict[str, Any]] = {}
        for state in request_states:
            batch_id = state.get("batch_id")
            if batch_id in (None, ""):
                continue
            key = str(batch_id)
            bucket = batches.setdefault(
                key,
                {
                    "batch_id": batch_id,
                    "active_unresolved": 0,
                    "queued_unresolved": 0,
                    "backend_dispatched_unresolved": 0,
                    "request_ids": [],
                    "task_ids": [],
                },
            )
            if not state.get("api_active"):
                continue
            bucket["active_unresolved"] += 1
            if state.get("backend_dispatched"):
                bucket["backend_dispatched_unresolved"] += 1
            else:
                bucket["queued_unresolved"] += 1
            if state.get("request_id") not in (None, ""):
                bucket["request_ids"].append(state["request_id"])
            if state.get("task_id") not in (None, ""):
                bucket["task_ids"].append(state["task_id"])
        degraded = bool(
            self._overflow_total or self._history_critical_evictions or self._request_evictions
        )
        return {
            "process_id": os.getpid(),
            "event_sequence": self._sequence,
            "source_ts": time.time(),
            "certificate_valid": not degraded,
            "degraded": degraded,
            "overflow_total": self._overflow_total,
            "overflow_critical": self._overflow_critical + self._history_critical_evictions,
            "history_evictions": self._history_evictions,
            "history_critical_evictions": self._history_critical_evictions,
            "request_evictions": self._request_evictions,
            "request_retirements": self._request_retirements,
            "active_unresolved": sum(item["active_unresolved"] for item in batches.values()),
            "queued_unresolved": sum(item["queued_unresolved"] for item in batches.values()),
            "batches": batches,
        }

    def batch_activity(self) -> dict[str, Any]:
        with self._lock:
            return self._batch_activity_unlocked()

    def reset_for_tests(self) -> None:
        with self._lock:
            self._pending.clear()
            self._history.clear()
            self._requests.clear()
            self._active_leases.clear()
            self._sequence = 0
            self._frame_sequence = 0
            self._overflow_total = 0
            self._overflow_critical = 0
            self._history_evictions = 0
            self._history_critical_evictions = 0
            self._request_evictions = 0
            self._request_retirements = 0


_REDUCER = LiveTelemetryReducer()


def emit_lifecycle_transition(transition: str, **fields: Any) -> dict[str, Any] | None:
    """Best-effort producer API; telemetry can never break request handling."""
    try:
        return _REDUCER.emit(transition, **fields)
    except Exception:
        return None


def live_telemetry_sequence() -> int:
    return _REDUCER.sequence


def live_telemetry_frame() -> dict[str, Any]:
    return _REDUCER.frame()


def live_batch_activity_summary() -> dict[str, Any]:
    """Read-only API lifecycle counts for drain/reconciliation consumers."""
    return _REDUCER.batch_activity()


def reset_live_telemetry_for_tests() -> None:
    _REDUCER.reset_for_tests()
