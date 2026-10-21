"""Fleet dispatch log: one JSONL record per ``ConcurrencyAwareBackend`` dispatch.

Write side for two filed, measurement-blocked tasks (neither could be scored
because the record did not exist):

* **WP-15.1** (``within-role-placement-state-machine.md``): per-request ARRIVAL
  time, queue wait, the instance chosen, and the fleet state (in-flight requests
  and reserved length per instance) at arrival, so a KV-length-aware selection
  rule can be replay-scored on a fit window and a holdout window.
* **HSF-3.1a** (``heterogeneous-slot-fabric-residency.md``): session-keyed
  completion -> next-enqueue gaps with a client class. A gap is DERIVED by the
  reader as ``next.ts_arrival - prev.ts_end`` over records sharing a
  ``session.key`` (workers write to one file under a flock, so the pairing is
  cross-process; nothing is paired in-process here).

Not a second serving log. ``serving_calls`` stays the per-HTTP-call record
(timings, tokens, outcome); this one is the dispatch decision and the arrival
clock. Join on ``request_id`` (``serving_call.v1`` ``caller.request_id``) for the
real prompt/cached token counts: lengths here are ``prompt_chars`` and the
``n_tokens`` budget only, never an estimate dressed up as tokens.

Belief-kernel write side: ``schema`` is versioned, every record carries
``config_sha256`` and a ``kind="config"`` record (the config echo) is written
once per backend before its first dispatch record, records are self-hashed like
``serving_call.v1``, and provenance (commit, run id) rides every record.

Behaviour: additive. Nothing here changes routing, locking, timing or results;
every failure is swallowed. Kill switch / path: ``ORCHESTRATOR_DISPATCH_LOG``
(``off``/``0``/``none`` disables). Unset, it follows ``serving_calls`` (so the
suite's ``ORCHESTRATOR_SERVING_CALLS_LOG=off`` also silences it). Takes effect at
the next orchestrator start.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.backends import serving_calls

SCHEMA = "epyc.orchestrator.dispatch.v1"
LOG_ENV = "ORCHESTRATOR_DISPATCH_LOG"
_OFF = {"off", "0", "none", "false", "disabled"}

_write_lock = threading.Lock()


def log_path() -> Path | None:
    """Where records go, or None when disabled."""
    override = os.environ.get(LOG_ENV, "").strip()
    if override.lower() in _OFF:
        return None
    if override:
        return Path(override)
    if serving_calls.log_path() is None:
        return None
    return serving_calls._log_dir() / "dispatch" / "dispatch.jsonl"


def session_key(session_id: str | None) -> str | None:
    """Stable non-reversible key for a session id (the id itself is not logged)."""
    if not session_id:
        return None
    return hashlib.sha256(str(session_id).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def config_digest(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        serving_calls._json_safe(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_record(record: dict[str, Any]) -> bool:
    """Append one self-hashed record under a cross-process flock. Never raises."""
    path = log_path()
    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = serving_calls._json_safe(dict(record))
        record["record_sha256"] = serving_calls.record_digest(record)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        lock_path = path.with_name(path.name + ".lock")
        with _write_lock, open(lock_path, "a") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                serving_calls._maybe_rotate(path)
                with open(path, "a") as fh:
                    fh.write(line)
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return True
    except Exception:
        return False


def _request_lengths(request: Any) -> dict[str, Any]:
    prompt = getattr(request, "prompt", None)
    chars = len(prompt) if isinstance(prompt, str) else None
    payload = getattr(request, "chat_payload", None)
    if chars is None and isinstance(payload, dict):
        try:
            chars = len(json.dumps(payload.get("messages") or [], default=str))
        except Exception:
            chars = None
    n_tokens = getattr(request, "n_tokens", None)
    return {
        "prompt_chars": chars,
        "n_tokens_budget": n_tokens if isinstance(n_tokens, int) else None,
    }


class DispatchLog:
    """Per-backend recorder: in-flight accounting plus the record writer.

    One instance per ``ConcurrencyAwareBackend``. ``begin`` is called at arrival
    (before placement), ``dispatched`` once an instance is held, ``end`` when the
    call returns. All methods swallow their own failures and return a handle that
    is safe to pass back even if an earlier step failed.
    """

    def __init__(self, backend_id: str, config: dict[str, Any]) -> None:
        self._id = backend_id
        self._config = serving_calls._json_safe(dict(config))
        self._config_sha = config_digest(self._config)
        self._config_written = False
        self._lock = threading.Lock()
        # key -> {"requests": int, "prompt_chars": int, "n_tokens_budget": int}
        self._inflight: dict[str, dict[str, int]] = {}
        self._waiting = 0

    # -- state ------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "waiting": self._waiting,
                "instances": {k: dict(v) for k, v in self._inflight.items()},
            }

    def _ensure_config(self) -> None:
        if self._config_written:
            return
        self._config_written = True
        write_record(
            {
                "schema": SCHEMA,
                "kind": "config",
                "record_id": uuid.uuid4().hex,
                "ts": serving_calls._iso(time.time()),
                "backend_id": self._id,
                "config": self._config,
                "config_sha256": self._config_sha,
                "provenance": serving_calls.process_provenance(),
            }
        )

    # -- lifecycle --------------------------------------------------------

    def begin(self, request: Any, *, method: str, role: str | None) -> dict[str, Any] | None:
        try:
            if log_path() is None:
                return None
            staged = serving_calls.peek_staged() or {}
            now = time.time()
            # The primitives layer starts its wait clock before the region lock;
            # when staged, that is the true arrival, else this call's own entry.
            arrival = staged.get("_ts0") or now
            session_id = staged.get("session_id") or getattr(request, "session_id", None)
            lengths = _request_lengths(request)
            with self._lock:
                self._waiting += 1
            return {
                "method": method,
                "role": role,
                "ts_arrival": arrival,
                "ts_enter": now,
                "request_id": staged.get("request_id"),
                "task_id": staged.get("task_id") or getattr(request, "task_id", None),
                "session_id": session_id,
                "session_source": (
                    "staged" if staged.get("session_id")
                    else ("request" if session_id else None)
                ),
                "client_class": staged.get("client_class"),
                "client_class_provenance": staged.get("client_class_provenance"),
                "client": staged.get("client"),
                "workload_class": staged.get("workload_class")
                or getattr(request, "workload_class", None),
                "priority": staged.get("priority"),
                "lengths": lengths,
                "fleet_at_arrival": self._snapshot(),
                "_held": False,
            }
        except Exception:
            return None

    def dispatched(self, handle: dict[str, Any] | None, *, meta: dict[str, Any]) -> None:
        if handle is None:
            return
        try:
            key = str(meta.get("port") or meta.get("instance_idx"))
            lengths = handle["lengths"]
            with self._lock:
                self._waiting = max(0, self._waiting - 1)
                slot = self._inflight.setdefault(
                    key, {"requests": 0, "prompt_chars": 0, "n_tokens_budget": 0}
                )
                slot["requests"] += 1
                slot["prompt_chars"] += lengths.get("prompt_chars") or 0
                slot["n_tokens_budget"] += lengths.get("n_tokens_budget") or 0
                handle["_held"] = True
            handle["ts_dispatch"] = time.time()
            handle["instance"] = {
                "key": key,
                "port": meta.get("port"),
                "backend_url": meta.get("backend_url"),
                "topology_idx": meta.get("instance_idx"),
                "concurrency_idx": meta.get("concurrency_idx"),
                "shape": meta.get("instance_shape"),
                "is_full": meta.get("instance_idx") == 0,
            }
            handle["fleet_at_dispatch"] = self._snapshot()
        except Exception:
            pass

    def end(
        self, handle: dict[str, Any] | None, *, result: Any = None, error: BaseException | None = None,
        completion_observed: bool = True,
    ) -> None:
        if handle is None:
            return
        try:
            ts_end = time.time()
            lengths = handle["lengths"]
            instance = handle.get("instance")
            with self._lock:
                if handle.get("_held") and instance:
                    slot = self._inflight.get(instance["key"])
                    if slot:
                        slot["requests"] = max(0, slot["requests"] - 1)
                        slot["prompt_chars"] = max(0, slot["prompt_chars"] - (lengths.get("prompt_chars") or 0))
                        slot["n_tokens_budget"] = max(
                            0, slot["n_tokens_budget"] - (lengths.get("n_tokens_budget") or 0)
                        )
                        if slot["requests"] == 0:
                            self._inflight.pop(instance["key"], None)
                else:
                    self._waiting = max(0, self._waiting - 1)
            self._ensure_config()
            dispatched_ts = handle.get("ts_dispatch")
            outcome = "exception" if error is not None else (
                "ok" if getattr(result, "success", True) else "failed"
            )
            if not completion_observed and error is None:
                outcome = "handed_off"
            write_record(
                {
                    "schema": SCHEMA,
                    "kind": "dispatch",
                    "record_id": uuid.uuid4().hex,
                    "backend_id": self._id,
                    "config_sha256": self._config_sha,
                    "method": handle["method"],
                    "role": handle["role"],
                    "request_id": handle["request_id"],
                    "task_id": handle["task_id"],
                    "ts_arrival": serving_calls._iso(handle["ts_arrival"]),
                    "ts_dispatch": serving_calls._iso(dispatched_ts) if dispatched_ts else None,
                    "ts_end": serving_calls._iso(ts_end),
                    "queue_wait_ms": (
                        round(max(0.0, (dispatched_ts - handle["ts_arrival"]) * 1000.0), 3)
                        if dispatched_ts else None
                    ),
                    "service_ms": (
                        round(max(0.0, (ts_end - dispatched_ts) * 1000.0), 3)
                        if dispatched_ts else None
                    ),
                    "completion_observed": completion_observed,
                    "outcome": outcome,
                    "error_type": type(error).__name__ if error is not None else None,
                    "session": {
                        "key": session_key(handle["session_id"]),
                        "source": handle["session_source"],
                    },
                    "client": {
                        "class": handle["client_class"],
                        "class_provenance": handle["client_class_provenance"],
                        "id": handle["client"],
                    },
                    "workload_class": handle["workload_class"],
                    "priority": handle["priority"],
                    "request": lengths,
                    "instance": instance,
                    "fleet_at_arrival": handle["fleet_at_arrival"],
                    "fleet_at_dispatch": handle.get("fleet_at_dispatch"),
                    "provenance": serving_calls.process_provenance(),
                }
            )
        except Exception:
            pass
