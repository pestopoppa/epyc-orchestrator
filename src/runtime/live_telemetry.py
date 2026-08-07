"""Bounded, non-blocking lifecycle telemetry for the live dashboard.

Each producer uses an in-process reducer: it takes one short mutex and returns
without filesystem or network I/O. A coalesced background publisher exports
bounded worker snapshots so a multi-worker API can build one host-authoritative
view. This remains ephemeral observability, not a durable event store. Dashboard
SSE consumers independently reconcile it with llama-server ``/slots``.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from pathlib import Path
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
            details = event.get("details") or {}
            if details.get("topology_role"):
                state["topology_role"] = details["topology_role"]
            if details.get("instance_shape"):
                state["instance_shape"] = details["instance_shape"]
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

    def _snapshot_unlocked(
        self,
        *,
        drain_pending: bool,
        advance_frame: bool,
    ) -> dict[str, Any]:
        if advance_frame:
            self._frame_sequence += 1
        pending = list(self._pending)
        if drain_pending:
            self._pending.clear()
        history = [dict(event) for event in self._history]
        requests = [dict(state) for state in self._requests.values()]
        active_leases = [dict(lease) for lease in self._active_leases.values()]
        degraded = bool(
            self._overflow_total or self._history_critical_evictions or self._request_evictions
        )
        batch_activity = self._batch_activity_unlocked(requests)
        pid = os.getpid()
        return {
            "schema_version": "live_telemetry.v1",
            "process_id": pid,
            "process_start_id": _process_start_id(pid),
            "frame_sequence": self._frame_sequence,
            "event_sequence": self._sequence,
            "source_sequences": {str(pid): self._sequence},
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

    def frame(self) -> dict[str, Any]:
        """Return one sequenced immutable snapshot of the reduced state."""
        with self._lock:
            return self._snapshot_unlocked(drain_pending=True, advance_frame=True)

    def publication_frame(self) -> dict[str, Any]:
        """Return observational state without consuming pending transitions."""
        with self._lock:
            return self._snapshot_unlocked(drain_pending=False, advance_frame=False)

    def publisher_frame(self) -> dict[str, Any]:
        """Export and acknowledge pending notifications for this worker."""
        with self._lock:
            return self._snapshot_unlocked(drain_pending=True, advance_frame=False)

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

_PUBLISH_INTERVAL_S = 0.5
_WORKER_STALE_S = 3.0
_PUBLISH_WAKE = threading.Event()
_PUBLISH_THREAD: threading.Thread | None = None
_PUBLISH_LOCK = threading.Lock()


def _telemetry_root() -> Path:
    explicit = os.environ.get("ORCHESTRATOR_LIVE_TELEMETRY_DIR")
    if explicit:
        return Path(explicit)
    tmp_root = (
        os.environ.get("ORCHESTRATOR_TMP_DIR")
        or os.environ.get("ORCHESTRATOR_PATHS_TMP_DIR")
        or "/mnt/raid0/llm/tmp"
    )
    return Path(tmp_root) / "live_telemetry_workers"


def _process_start_id(pid: int) -> str:
    """Linux process start ticks, preventing stale-file acceptance after PID reuse."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().split()
        return str(fields[21])
    except (OSError, ValueError, IndexError):
        return ""


def _process_cmdline(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{int(pid)}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
        )
    except OSError:
        return ""


def _expected_api_worker_pids() -> list[int]:
    """Return the live uvicorn spawn-worker roster visible to this worker."""
    parent_pid = os.getppid()
    if "uvicorn" not in _process_cmdline(parent_pid):
        return []
    try:
        raw = Path(f"/proc/{parent_pid}/task/{parent_pid}/children").read_text()
    except OSError:
        return []
    workers: list[int] = []
    for value in raw.split():
        try:
            pid = int(value)
        except ValueError:
            continue
        cmdline = _process_cmdline(pid)
        if "multiprocessing.spawn" in cmdline and _process_start_id(pid):
            workers.append(pid)
    return sorted(workers)


def _configured_api_worker_count() -> int:
    parent_cmdline = _process_cmdline(os.getppid())
    if "uvicorn" not in parent_cmdline:
        return 1
    match = re.search(r"(?:^|\s)--workers(?:=|\s+)(\d+)(?:\s|$)", parent_cmdline)
    if match:
        return max(1, int(match.group(1)))
    try:
        return max(1, int(os.environ.get("WEB_CONCURRENCY", "1")))
    except ValueError:
        return 1


def _is_api_spawn_worker() -> bool:
    return os.getpid() in _expected_api_worker_pids()


def _worker_state_path(pid: int, *, root: Path | None = None) -> Path:
    return (root or _telemetry_root()) / f"worker-{int(pid)}.json"


def _write_worker_state(
    frame: dict[str, Any],
    *,
    root: Path | None = None,
) -> None:
    """Atomically publish one worker frame; called only by the background thread."""
    target = _worker_state_path(int(frame["process_id"]), root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    payload = json.dumps(frame, separators=(",", ":"), sort_keys=True).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _publisher_loop() -> None:
    next_publish = 0.0
    while True:
        delay = max(0.0, next_publish - time.monotonic())
        _PUBLISH_WAKE.wait(timeout=delay if delay else _PUBLISH_INTERVAL_S)
        _PUBLISH_WAKE.clear()
        now = time.monotonic()
        if now < next_publish:
            continue
        try:
            _write_worker_state(_REDUCER.publisher_frame())
        except Exception:
            # A missing/stale worker publication invalidates the host certificate.
            # Telemetry must never perturb request execution.
            pass
        next_publish = time.monotonic() + _PUBLISH_INTERVAL_S


def _ensure_worker_publisher() -> None:
    global _PUBLISH_THREAD
    if not _is_api_spawn_worker() or (_PUBLISH_THREAD and _PUBLISH_THREAD.is_alive()):
        return
    with _PUBLISH_LOCK:
        if _PUBLISH_THREAD and _PUBLISH_THREAD.is_alive():
            return
        _PUBLISH_THREAD = threading.Thread(
            target=_publisher_loop,
            daemon=True,
            name="live-telemetry-publisher",
        )
        _PUBLISH_THREAD.start()
        _PUBLISH_WAKE.set()


def _read_worker_frame(pid: int, *, root: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(_worker_state_path(pid, root=root).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or int(payload.get("process_id") or -1) != int(pid):
        return None
    if str(payload.get("process_start_id") or "") != _process_start_id(pid):
        return None
    return payload


def _merge_worker_frames(
    frames: dict[int, dict[str, Any]],
    *,
    expected_pids: list[int],
    now: float,
    stale_after_s: float,
    local_frame_sequence: int = 0,
    expected_worker_count: int | None = None,
) -> dict[str, Any]:
    def retain_events(
        events: list[dict[str, Any]], capacity: int
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Bound a merged stream while preferring ownership/terminal events."""
        evictions = max(0, len(events) - capacity)
        critical_evictions = 0
        retained = list(events)
        while len(retained) > capacity:
            evict_at = next(
                (
                    idx
                    for idx, event in enumerate(retained)
                    if event.get("transition") not in _NON_EVICTABLE_TRANSITIONS
                ),
                None,
            )
            if evict_at is None:
                retained.pop(0)
                critical_evictions += 1
            else:
                retained.pop(evict_at)
        return retained, evictions, critical_evictions

    stale_pids = sorted(
        pid
        for pid, frame in frames.items()
        if now - float(frame.get("source_ts") or 0.0) > stale_after_s
    )
    missing_pids = sorted(set(expected_pids) - set(frames))
    observed_pids = sorted(frames)
    expected_count = max(len(expected_pids), int(expected_worker_count or 0))
    roster_incomplete = len(expected_pids) < expected_count
    coverage_complete = not missing_pids and not stale_pids and not roster_incomplete
    ordered_frames = [frames[pid] for pid in observed_pids]
    requests = [dict(item) for frame in ordered_frames for item in frame.get("requests") or []]
    leases = [dict(item) for frame in ordered_frames for item in frame.get("active_leases") or []]
    transitions = [
        dict(item) for frame in ordered_frames for item in frame.get("transitions") or []
    ]
    transitions.sort(
        key=lambda item: (
            float(item.get("source_ts") or 0.0),
            int(item.get("process_id") or 0),
            int(item.get("sequence") or 0),
        )
    )
    pending = [
        dict(item) for frame in ordered_frames for item in frame.get("pending_transitions") or []
    ]
    pending.sort(
        key=lambda item: (
            float(item.get("source_ts") or 0.0),
            int(item.get("process_id") or 0),
            int(item.get("sequence") or 0),
        )
    )
    source_sequences = {
        str(pid): int(frames[pid].get("event_sequence") or 0) for pid in observed_pids
    }
    overflow_keys = (
        "total",
        "critical",
        "history_evictions",
        "history_critical_evictions",
        "request_evictions",
        "request_retirements",
    )
    overflow = {
        key: sum(int((frame.get("overflow") or {}).get(key) or 0) for frame in ordered_frames)
        for key in overflow_keys
    }
    transitions, merged_history_evictions, merged_history_critical = retain_events(transitions, 512)
    pending, merged_pending_evictions, merged_pending_critical = retain_events(pending, 1024)
    overflow["total"] += merged_pending_evictions
    overflow["critical"] += merged_pending_critical
    overflow["history_evictions"] += merged_history_evictions
    overflow["history_critical_evictions"] += merged_history_critical
    overflow["state_preserved"] = all(
        bool((frame.get("overflow") or {}).get("state_preserved", True)) for frame in ordered_frames
    )
    overflow["state_preserved"] = bool(overflow["state_preserved"]) and not bool(
        merged_history_critical or merged_pending_critical
    )
    worker_degraded = any(bool(frame.get("degraded")) for frame in ordered_frames) or bool(
        merged_history_critical or merged_pending_critical
    )
    degraded = worker_degraded or not coverage_complete

    batches: dict[str, dict[str, Any]] = {}
    for frame in ordered_frames:
        summary = frame.get("batch_activity") or {}
        for raw_batch_id, raw_bucket in (summary.get("batches") or {}).items():
            if not isinstance(raw_bucket, dict):
                continue
            batch_id = str(raw_batch_id)
            bucket = batches.setdefault(
                batch_id,
                {
                    "batch_id": raw_bucket.get("batch_id", raw_batch_id),
                    "active_unresolved": 0,
                    "queued_unresolved": 0,
                    "backend_dispatched_unresolved": 0,
                    "request_ids": [],
                    "task_ids": [],
                    "worker_pids": [],
                },
            )
            for key in (
                "active_unresolved",
                "queued_unresolved",
                "backend_dispatched_unresolved",
            ):
                bucket[key] += int(raw_bucket.get(key) or 0)
            bucket["request_ids"] = sorted(
                set(bucket["request_ids"]) | {str(v) for v in raw_bucket.get("request_ids") or []}
            )
            bucket["task_ids"] = sorted(
                set(bucket["task_ids"]) | {str(v) for v in raw_bucket.get("task_ids") or []}
            )
            bucket["worker_pids"].append(int(frame.get("process_id") or 0))
    batch_activity = {
        "process_id": os.getpid(),
        "process_ids": observed_pids,
        "event_sequence": sum(source_sequences.values()),
        "source_sequences": source_sequences,
        "source_ts": now,
        "certificate_valid": not degraded,
        "degraded": degraded,
        "overflow_total": overflow["total"],
        "overflow_critical": overflow["critical"] + overflow["history_critical_evictions"],
        "history_evictions": overflow["history_evictions"],
        "history_critical_evictions": overflow["history_critical_evictions"],
        "request_evictions": overflow["request_evictions"],
        "request_retirements": overflow["request_retirements"],
        "active_unresolved": sum(item["active_unresolved"] for item in batches.values()),
        "queued_unresolved": sum(item["queued_unresolved"] for item in batches.values()),
        "batches": batches,
        "worker_coverage": {
            "complete": coverage_complete,
            "expected_pids": sorted(expected_pids),
            "expected_count": expected_count,
            "observed_pids": observed_pids,
            "missing_pids": missing_pids,
            "stale_pids": stale_pids,
            "roster_incomplete": roster_incomplete,
        },
    }
    return {
        "schema_version": "live_telemetry.host.v1",
        "process_id": os.getpid(),
        "process_ids": observed_pids,
        "frame_sequence": local_frame_sequence,
        "event_sequence": sum(source_sequences.values()),
        "source_sequences": source_sequences,
        "source_ts": now,
        "requests": requests,
        "active_leases": leases,
        "batch_activity": batch_activity,
        "transitions": transitions,
        "pending_transitions": pending,
        "degraded": degraded,
        "overflow": overflow,
        "worker_coverage": batch_activity["worker_coverage"],
    }


def _host_frame(local_frame: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_api_worker_pids()
    expected_count = _configured_api_worker_count()
    if expected_count <= 1:
        return local_frame
    now = time.time()
    root = _telemetry_root()
    frames: dict[int, dict[str, Any]] = {}
    for pid in expected:
        frame = local_frame if pid == os.getpid() else _read_worker_frame(pid, root=root)
        if frame is not None:
            frames[pid] = frame
    return _merge_worker_frames(
        frames,
        expected_pids=expected,
        now=now,
        stale_after_s=_WORKER_STALE_S,
        local_frame_sequence=int(local_frame.get("frame_sequence") or 0),
        expected_worker_count=expected_count,
    )


def emit_lifecycle_transition(transition: str, **fields: Any) -> dict[str, Any] | None:
    """Best-effort producer API; telemetry can never break request handling."""
    try:
        event = _REDUCER.emit(transition, **fields)
        _ensure_worker_publisher()
        _PUBLISH_WAKE.set()
        return event
    except Exception:
        return None


def live_telemetry_sequence() -> tuple[tuple[str, int], ...]:
    frame = _host_frame(_REDUCER.publication_frame())
    return tuple(sorted((frame.get("source_sequences") or {}).items()))


def live_telemetry_frame() -> dict[str, Any]:
    return _host_frame(_REDUCER.frame())


def live_batch_activity_summary() -> dict[str, Any]:
    """Read-only API lifecycle counts for drain/reconciliation consumers."""
    return _host_frame(_REDUCER.publication_frame())["batch_activity"]


def reset_live_telemetry_for_tests() -> None:
    _REDUCER.reset_for_tests()


_ensure_worker_publisher()
