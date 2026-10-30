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

from src.backends import prompt_layout, serving_calls

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


def resolve_client_class(
    explicit: str | None,
    explicit_provenance: str | None = None,
    *,
    lane: str,
    workload_class: str | None = None,
    priority: str | None = None,
    trace_keys: Any = None,
    client: str | None = None,
    session_id: str | None = None,
) -> tuple[str | None, str, str | None]:
    """``(class, class_source, provenance)``; ``class_source`` is ``header`` or ``derived``.

    A caller-supplied ``x_client_class`` always wins (``header``). Otherwise a
    SERVER-SIDE label is derived from what the server itself knows about the call:
    the lane, the scheduler workload class, request priority, whether the call
    arrived on the typed /v1 surface (trace keys), the process-wide client id and
    whether the call carries a session. A derived class is a label for analysis, not
    an identity claim - readers that want only self-reported classes filter on
    ``class_source == "header"``. Vocabulary: ``eval_harness`` (workload eval_batch),
    ``campaign`` (workload campaign, e.g. autopilot /chat planner calls),
    ``external:<client id>`` / ``external_passthrough`` (passthrough lane),
    ``v1_client`` (typed /v1 request without a class tag), ``orchestrator_internal``
    (everything else through the primitives layer or a direct backend call).
    """
    if explicit:
        return explicit, "header", explicit_provenance
    try:
        if workload_class == "eval_batch":
            return "eval_harness", "derived", "derived:workload_class=eval_batch"
        if workload_class == "campaign":
            return "campaign", "derived", "derived:workload_class=campaign"
        if lane == "passthrough":
            if client:
                return f"external:{client}"[:64], "derived", "derived:lane=passthrough+x-client-id"
            return "external_passthrough", "derived", "derived:lane=passthrough"
        if isinstance(trace_keys, dict) and trace_keys:
            return "v1_client", "derived", "derived:v1_trace_keys"
        if priority == "background":
            return "background", "derived", "derived:priority=background"
        return "orchestrator_internal", "derived", f"derived:lane={lane}"
    except Exception:
        return None, "derived", None


_MODEL_KEYS = ("model_path", "launch_id", "launched_at", "argv_sha256", "binary_realpath", "identity_source")


def model_identity(port: int | None) -> dict[str, Any]:
    """Served-model identity for ``port`` from the stack's launch sidecar.

    WP-15.2: which GGUF served a port on a date. No network call: the sidecar
    (``logs/server_launches/<port>.json``, written by the stack at launch) is
    stat-ed and re-read only when its mtime changes, so a server reload shows up
    as a new ``launch_id`` / ``model_path`` without a per-request probe. Absence is
    reported (``identity_source = "absent"``), never filled. Never raises.
    """
    try:
        ident = serving_calls.server_identity(port)
        return {k: ident[k] for k in _MODEL_KEYS if k in ident}
    except Exception:
        return {"identity_source": "error"}


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
            cls, cls_source, cls_prov = resolve_client_class(
                staged.get("client_class"), staged.get("client_class_provenance"),
                lane="fleet", workload_class=staged.get("workload_class")
                or getattr(request, "workload_class", None),
                priority=staged.get("priority"), trace_keys=staged.get("trace_keys"),
                client=staged.get("client"), session_id=session_id,
            )
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
                "client_class": cls,
                "client_class_source": cls_source,
                "client_class_provenance": cls_prov,
                "client": staged.get("client"),
                "workload_class": staged.get("workload_class")
                or getattr(request, "workload_class", None),
                "priority": staged.get("priority"),
                # HSF-LB1: the tool whose result opened this turn, when the caller staged it.
                "preceding_tool": staged.get("preceding_tool"),
                "turn_trigger": staged.get("turn_trigger"),
                "lengths": lengths,
                "layout": prompt_layout.fields_for_request(request, staged=staged),
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
                "model": model_identity(meta.get("port")),
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
                        "class_source": handle["client_class_source"],
                        "class_provenance": handle["client_class_provenance"],
                        "id": handle["client"],
                    },
                    "workload_class": handle["workload_class"],
                    "priority": handle["priority"],
                    # HSF-LB1: key session gaps by the tool that preceded the request and
                    # by how the previous generation ended (the backend's own stop signal).
                    "preceding_tool": handle.get("preceding_tool"),
                    "turn_trigger": handle.get("turn_trigger"),
                    "finish_reason": getattr(result, "completion_reason", None) or None,
                    "request": lengths,
                    # RTE-LOG-1: hashes and lengths only; absent when no layout was staged.
                    **(handle.get("layout") or {}),
                    "prompt_tokens": getattr(result, "prompt_tokens", None),
                    "cached_prompt_tokens": serving_calls._cached_prompt_tokens(result, None),
                    "instance": instance,
                    "fleet_at_arrival": handle["fleet_at_arrival"],
                    "fleet_at_dispatch": handle.get("fleet_at_dispatch"),
                    "provenance": serving_calls.process_provenance(),
                }
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Non-fleet lanes (passthrough, direct single-server backend calls)
# ---------------------------------------------------------------------------

_standalone_config_written: set[str] = set()


def record_standalone(
    *,
    lane: str,
    role: str | None,
    base_url: str | None,
    port: int | None,
    ts_arrival: float,
    ts_dispatch: float | None,
    ts_end: float,
    request_id: str | None = None,
    session_id: str | None = None,
    client_class: str | None = None,
    client_class_provenance: str | None = None,
    client: str | None = None,
    workload_class: str | None = None,
    priority: str | None = None,
    prompt_chars: int | None = None,
    n_tokens_budget: int | None = None,
    outcome: str = "ok",
    error_type: str | None = None,
    method: str | None = None,
    trace_keys: Any = None,
    completion_observed: bool = True,
    preceding_tool: str | None = None,
    turn_trigger: str | None = None,
    finish_reason: str | None = None,
    layout: dict[str, Any] | None = None,
    prompt_tokens: int | None = None,
    cached_prompt_tokens: int | None = None,
) -> None:
    """One ``dispatch.v1`` record for a lane with no fleet placement.

    ``lane`` is ``passthrough`` or ``direct`` (a single-server backend call that
    did not go through ``ConcurrencyAwareBackend``). ``fleet_*`` are null: there is
    no instance choice and no fleet accounting on these lanes. Never raises.
    """
    try:
        if log_path() is None:
            return
        client_class, class_source, client_class_provenance = resolve_client_class(
            client_class, client_class_provenance, lane=lane, workload_class=workload_class,
            priority=priority, trace_keys=trace_keys, client=client, session_id=session_id,
        )
        config = {"lane": lane, "dispatch_log_env": LOG_ENV}
        cfg_sha = config_digest(config)
        if lane not in _standalone_config_written:
            _standalone_config_written.add(lane)
            write_record(
                {
                    "schema": SCHEMA,
                    "kind": "config",
                    "record_id": uuid.uuid4().hex,
                    "ts": serving_calls._iso(time.time()),
                    "backend_id": f"lane:{lane}",
                    "config": config,
                    "config_sha256": cfg_sha,
                    "provenance": serving_calls.process_provenance(),
                }
            )
        write_record(
            {
                "schema": SCHEMA,
                "kind": "dispatch",
                "record_id": uuid.uuid4().hex,
                "backend_id": f"lane:{lane}",
                "config_sha256": cfg_sha,
                "method": method or lane,
                "role": role,
                "request_id": request_id,
                "task_id": None,
                "ts_arrival": serving_calls._iso(ts_arrival),
                "ts_dispatch": serving_calls._iso(ts_dispatch) if ts_dispatch else None,
                "ts_end": serving_calls._iso(ts_end),
                "queue_wait_ms": (
                    round(max(0.0, (ts_dispatch - ts_arrival) * 1000.0), 3) if ts_dispatch else None
                ),
                "service_ms": (
                    round(max(0.0, (ts_end - ts_dispatch) * 1000.0), 3) if ts_dispatch else None
                ),
                "completion_observed": completion_observed,
                "outcome": outcome,
                "error_type": error_type,
                "session": {
                    "key": session_key(session_id),
                    "source": lane if session_id else None,
                },
                "client": {
                    "class": client_class,
                    "class_source": class_source,
                    "class_provenance": client_class_provenance,
                    "id": client,
                },
                "workload_class": workload_class,
                "priority": priority,
                "preceding_tool": preceding_tool,
                "turn_trigger": turn_trigger,
                "finish_reason": finish_reason,
                "request": {"prompt_chars": prompt_chars, "n_tokens_budget": n_tokens_budget},
                "instance": {
                    "key": str(port) if port else None,
                    "port": port,
                    "model": model_identity(port),
                    "backend_url": base_url,
                },
                # RTE-LOG-1: forward only the existing hashes/lengths projection.
                # Native counts retain None when the backend did not report them.
                **({key: layout[key] for key in (
                    "blocks", "layout_chars", "layout_match", "layout_schema",
                    "unlocated_blocks", "turn_index", "source", "tail_chars", "rebased",
                ) if key in layout} if layout and prompt_layout.enabled() else {}),
                "prompt_tokens": prompt_tokens,
                "cached_prompt_tokens": cached_prompt_tokens,
                "fleet_at_arrival": None,
                "fleet_at_dispatch": None,
                "provenance": serving_calls.process_provenance(),
            }
        )
    except Exception:
        pass
