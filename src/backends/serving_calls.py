"""Per-call serving telemetry: one JSONL record per HTTP call to a llama-server.

Why this exists (workspace-89 prefill-share analysis, 2026-10-03): the llama-server
logs carry no wall clock and no caller, ``progress/*.jsonl`` carried no prompt timing,
and the only way to attribute a week of model-server traffic to a role or client was
to reconstruct it from slot ids and timestamps relative to process start. This module
is the write side that makes that reconstruction unnecessary.

Choke point
    ``LlamaServerBackend.infer`` and ``LlamaServerBackend.infer_stream_text`` are
    wrapped with :func:`recorded_call`. Every HTTP call the orchestrator makes to a
    llama-server through the backend layer — direct, via ``CachingBackend``, via
    ``ConcurrencyAwareBackend`` or via ``model_server`` — passes through one of the
    two, so one record is written per backend call whatever the caller.

Layers above contribute what only they know, through context variables that never
change a call's behaviour:

* :func:`stage_caller` — the primitives layer stages WHO is calling (role, request /
  task / session ids, workload class) and WHEN it started waiting, before it takes
  the inference / region lock. The wrapped call consumes the stage, so the gap
  between staging and dispatch is the queue / lock wait. A call that never reaches
  the backend (lock timeout, cancellation while queued) is recorded by
  :func:`abandon_staged`, with what is known.
* :func:`note` — anything below the wrapper adds facts it parsed (llama.cpp's
  ``timings`` object, the endpoint, the instance a ``ConcurrencyAwareBackend``
  placed the call on).

Provenance needed for a belief-kernel ``ClaimTuple`` projection rides every record:
the orchestrator commit the process started on, a per-process run id, and the
server launch identity (argv sha256, binary, served model) that the stack writes to
``logs/server_launches/<port>.json`` at launch (``scripts/server/stack_log_banner.py``).
A record whose server predates that sidecar says so (``server.identity_source =
"absent"``) rather than guessing.

Writing never affects inference: every failure in this module is swallowed.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import logging
import math
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from src.runtime.git_head import resolve_git_head

try:  # pragma: no cover - fcntl exists on every supported host
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

SCHEMA = "epyc.orchestrator.serving_call.v1"

#: Path override. ``off`` / ``0`` / ``none`` disables the log (the test suite does).
LOG_ENV = "ORCHESTRATOR_SERVING_CALLS_LOG"
MAX_MB_ENV = "ORCHESTRATOR_SERVING_CALLS_MAX_MB"
KEEP_ENV = "ORCHESTRATOR_SERVING_CALLS_KEEP"
_DEFAULT_MAX_MB = 64
_DEFAULT_KEEP = 8
_DISABLED_VALUES = {"off", "0", "none", "false", "disabled"}

#: The subset of llama.cpp's ``timings`` object every record carries verbatim.
TIMING_KEYS = (
    "cache_n",
    "prompt_n",
    "prompt_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_second",
    "draft_n",
    "draft_n_accepted",
    # KPF-27e / KPF-24: the RTG-58 P1 server-fork fields. A champion (or
    # fork-off) server never sends them, so its records are unchanged; with the
    # fork on they carry the fork credit (``n_fork_tokens``, included in
    # ``cache_n``) and the slot/task that served the call on EVERY endpoint
    # (the chat lane had neither before), which the prefix index binds to.
    "n_fork_tokens",
    "fork_src_slot",
    "fork_src_kind",
    "id_slot",
    "id_task",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_write_lock = threading.Lock()

# Context: the caller staged by the primitives layer, the facts noted during a call,
# and a re-entrancy flag so `infer_stream_text -> infer` writes ONE record.
_STAGED: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "serving_calls_staged", default=None
)
_NOTES: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "serving_calls_notes", default=None
)
_IN_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "serving_calls_in_call", default=False
)
#: UFH14-LB2: the summary of the last recorded call made in THIS context (set by
#: ``recorded_call`` after the call returns), so a parse event can say which
#: generation it is parsing and how that generation ended (``finish_reason``).
_LAST_CALL: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "serving_calls_last_call", default=None
)

# UFH14-LB6: the writer used to swallow every failure silently, so a live process whose
# records stopped arriving (19 records in 17 days against ~9k logged ``llama POST``
# lines) could not tell "disabled", "nested" and "write failed" apart. These counters
# are per process and are surfaced by :func:`health` and a one-time WARNING.
_STATS: dict[str, int] = {
    "recorded": 0,
    "skipped_disabled": 0,
    "skipped_nested": 0,
    "build_failed": 0,
    "write_failed": 0,
}
_stats_lock = threading.Lock()
_logged_failure_kinds: set[str] = set()
_log = logging.getLogger(__name__)


def _bump(key: str, exc: BaseException | None = None) -> None:
    """Count a recorder event; the FIRST failure of each kind is logged with its
    exception (fail-open must stay visible: a swallowed failure is invisible)."""
    with _stats_lock:
        _STATS[key] = _STATS.get(key, 0) + 1
        first = key.endswith("_failed") and key not in _logged_failure_kinds
        if first:
            _logged_failure_kinds.add(key)
    if first:
        _log.warning(
            "serving_calls %s (first occurrence this process): %s: %s",
            key, type(exc).__name__ if exc else "-", exc,
        )


def health() -> dict[str, Any]:
    """Per-process recorder state: where records go and what happened to the calls.

    ``skipped_disabled`` > 0 with ``enabled`` false means the kill switch is set in
    this process's environment; ``build_failed`` / ``write_failed`` > 0 means records
    were dropped by an exception (the first of each kind is logged at WARNING).
    """
    path = log_path()
    with _stats_lock:
        stats = dict(_STATS)
    return {"enabled": path is not None, "path": str(path) if path else None, **stats,
            "pid": os.getpid()}


def last_call() -> dict[str, Any] | None:
    """Summary of the last recorded call made in this context (copy), or None."""
    value = _LAST_CALL.get()
    return dict(value) if value else None


# ---------------------------------------------------------------------------
# Process provenance (computed once, at import: the code that is LOADED)
# ---------------------------------------------------------------------------


_PROCESS = {
    "orch_commit": os.environ.get("ORCHESTRATOR_GIT_SHA") or resolve_git_head(_REPO_ROOT),
    "run_id": uuid.uuid4().hex,
    "pid": os.getpid(),
    "host": socket.gethostname(),
    "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
}


def process_provenance() -> dict[str, Any]:
    """The per-process provenance block (commit, run id, pid, host, start)."""
    prov = dict(_PROCESS)
    if prov["pid"] != os.getpid():  # forked worker: own run id, inherited commit
        _PROCESS.update(run_id=uuid.uuid4().hex, pid=os.getpid())
        prov = dict(_PROCESS)
    return prov


# ---------------------------------------------------------------------------
# Paths and the server-launch sidecar
# ---------------------------------------------------------------------------


def _log_dir() -> Path:
    return Path(os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", str(_REPO_ROOT / "logs")))


def log_path() -> Path | None:
    """Where records go, or None when disabled."""
    override = os.environ.get(LOG_ENV, "").strip()
    if override.lower() in _DISABLED_VALUES:
        return None
    if override:
        return Path(override)
    return _log_dir() / "serving_calls" / "serving_calls.jsonl"


def launch_sidecar_dir() -> Path:
    """Directory the stack writes per-port launch identity into (see stack_log_banner)."""
    return _log_dir() / "server_launches"


_SIDECAR_CACHE: dict[int, tuple[float, dict[str, Any] | None]] = {}


def server_identity(port: int | None) -> dict[str, Any]:
    """Launch identity for the server on ``port``, from the stack's sidecar.

    Re-read whenever the sidecar's mtime changes (a reload rewrites it). Absence is
    reported, never filled: ``identity_source`` is ``"absent"`` when no sidecar
    exists — e.g. a server launched before the sidecar was introduced.
    """
    if not port:
        return {"identity_source": "absent"}
    path = launch_sidecar_dir() / f"{int(port)}.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"identity_source": "absent"}
    cached = _SIDECAR_CACHE.get(int(port))
    if cached is None or cached[0] != mtime:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = None
        _SIDECAR_CACHE[int(port)] = (mtime, data if isinstance(data, dict) else None)
        cached = _SIDECAR_CACHE[int(port)]
    data = cached[1]
    if not data:
        return {"identity_source": "unreadable"}
    keep = (
        "launch_id",
        "launched_at",
        "pid",
        "roles",
        "argv_sha256",
        "binary",
        "binary_realpath",
        "model_path",
        "ld_library_path",
        "stack_commit",
    )
    out = {k: data.get(k) for k in keep if k in data}
    out["identity_source"] = "stack_sidecar"
    return out


def _port_of(url: str | None) -> int | None:
    if not url:
        return None
    try:
        return urlparse(url).port
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Context API
# ---------------------------------------------------------------------------


def stage_caller(**fields: Any) -> None:
    """Stage who is about to call a backend; the wait clock starts now.

    Called by the primitives layer BEFORE it takes the inference / region lock. The
    next wrapped backend call in this context consumes it.
    """
    try:
        staged = {k: v for k, v in fields.items() if v not in (None, "", {}, [])}
        staged["_ts0"] = time.time()
        _STAGED.set(staged)
    except Exception:
        pass


_QUEUE_KEYS = ("placement_wait_ms", "instance_idx", "instance_full", "enqueue_ts_epoch")


def annotate_staged(**fields: Any) -> None:
    """Add dispatch-layer facts (instance placement) to the staged caller.

    Called by ``ConcurrencyAwareBackend`` between acquiring an instance and calling
    it, i.e. after staging and before the recorded call starts. Without a staged
    caller (a direct backend user) a stage is created that carries only these facts
    and no wait clock, so no wait is claimed for it.
    """
    try:
        staged = _STAGED.get()
        if staged is None:
            staged = {}
            _STAGED.set(staged)
        for key, value in fields.items():
            if value is not None:
                staged[key] = value
    except Exception:
        pass


def peek_staged() -> dict[str, Any] | None:
    """A copy of the staged caller without consuming it (read-only; never raises)."""
    try:
        staged = _STAGED.get()
        return dict(staged) if staged else None
    except Exception:
        return None


def clear_staged() -> None:
    """Drop a staged caller that no backend call consumed (e.g. a cache hit)."""
    _STAGED.set(None)


def note(**fields: Any) -> None:
    """Attach facts to the call in flight (no-op outside a recorded call)."""
    notes = _NOTES.get()
    if notes is None:
        return
    try:
        for key, value in fields.items():
            if value is not None:
                notes[key] = value
    except Exception:
        pass


def note_timings(timings: Any, *, endpoint: str | None = None, stream: bool | None = None) -> None:
    """Record llama.cpp's ``timings`` object (verbatim subset) for the call in flight."""
    # Only a SERVER timings object counts: it always carries prompt_n / predicted_n.
    # A client-synthesized stand-in (the /completion early-stop branch builds one
    # from wall time) must never be recorded as server-measured.
    if isinstance(timings, dict) and ("prompt_n" in timings or "predicted_n" in timings):
        note(timings={k: timings[k] for k in TIMING_KEYS if k in timings})
    if endpoint is not None or stream is not None:
        note(endpoint=endpoint, stream=stream)


def note_usage(usage: Any) -> None:
    """Record an OpenAI-style ``usage`` object for the call in flight."""
    if isinstance(usage, dict) and usage:
        keep = ("prompt_tokens", "completion_tokens", "total_tokens")
        note(usage={k: usage[k] for k in keep if k in usage})


def note_server_slot(id_slot: Any) -> None:
    """Record the slot the SERVER ran the call on (llama.cpp's response ``id_slot``).

    UFH14-B4: ``request.slot_id`` is what the orchestrator *asked* for, which on the
    chat lane is never even sent. Prefix-affinity analysis needs the slot that was
    actually used, so it is recorded separately and only when the server reports it.
    """
    if isinstance(id_slot, int) and not isinstance(id_slot, bool) and id_slot >= 0:
        note(server_slot=id_slot)


# ---------------------------------------------------------------------------
# Prefix fingerprints (UFH14-B4)
# ---------------------------------------------------------------------------

#: Character depths at which the wire prompt is fingerprinted. Characters, not tokens:
#: the record must not need a tokenizer. At ~3 chars/token these are ~0.7k, ~2.7k,
#: ~11k, ~44k and ~175k tokens — spanning the system/tools head through a long
#: agentic context. A depth is fingerprinted only when the prompt reaches it.
PREFIX_FP_DEPTHS_CHARS: tuple[int, ...] = (2048, 8192, 32768, 131072, 524288)
_FP_HEX = 16


def _prompt_text_for_fingerprint(request: Any) -> str | None:
    """The text whose prefix the server will see, as closely as the client knows it.

    Client-tool mode (``chat_payload``) sends ``tools`` + ``messages`` verbatim; the
    server renders tools into the system block, so tools are fingerprinted first,
    in caller order (a reordered tool list IS a prefix change). Every other lane
    sends ``request.prompt`` (as one user message on the chat lane).
    """
    payload = getattr(request, "chat_payload", None)
    if isinstance(payload, dict):
        try:
            head = json.dumps(payload.get("tools") or [], ensure_ascii=False, default=str)
            body = json.dumps(payload.get("messages") or [], ensure_ascii=False, default=str)
            return head + "\n" + body
        except Exception:
            return None
    prompt = getattr(request, "prompt", None)
    return prompt if isinstance(prompt, str) else None


def prefix_fingerprints(request: Any) -> dict[str, Any] | None:
    """Truncated sha256 of the prompt's first N characters for each depth it reaches.

    Two calls to the same server whose fingerprints agree at depth N shared at least
    N leading characters, so a cold prefill of the second one (low ``cache_n``) was a
    MISSED reuse rather than new content. Pure function; never raises.
    """
    try:
        text = _prompt_text_for_fingerprint(request)
        if not text:
            return None
        out: dict[str, Any] = {"chars": len(text)}
        for depth in PREFIX_FP_DEPTHS_CHARS:
            if len(text) < depth:
                break
            out[f"c{depth}"] = hashlib.sha256(
                text[:depth].encode("utf-8", "surrogatepass")
            ).hexdigest()[:_FP_HEX]
        return out
    except Exception:
        return None


# KVU-15c: the admission gate credits a request with the prefix a previous call to
# the same server left cached, matched by fingerprint (src/scheduling/prefix_history.py).
# The record's five depths are too coarse for that (a 300k-char prompt extending a
# 290k-char one matches only at 131072), so the history uses a denser LADDER of the
# same fingerprint — same text, same truncated sha256 — at geometric depths (x1.25
# from 2048 chars) plus the record depths. A matched depth is then within 20% of the
# true common prefix, always BELOW it (an under-credit, never an over-credit), and
# the ladder's values at the record depths equal ``prefix_fingerprints``'.
PREFIX_LADDER_BASE_CHARS = 2048
PREFIX_LADDER_RATIO = 1.25
PREFIX_LADDER_MAX_CHARS = 1 << 23


def _ladder_depths() -> tuple[int, ...]:
    depths = set(PREFIX_FP_DEPTHS_CHARS)
    depth = float(PREFIX_LADDER_BASE_CHARS)
    while depth <= PREFIX_LADDER_MAX_CHARS:
        depths.add(int(depth))
        depth *= PREFIX_LADDER_RATIO
    return tuple(sorted(d for d in depths if d <= PREFIX_LADDER_MAX_CHARS))


PREFIX_LADDER_DEPTHS_CHARS: tuple[int, ...] = _ladder_depths()


def prefix_ladder_for_text(text: str | None) -> dict[str, Any] | None:
    """``{"chars": len(text), "fp": {depth: hex}}`` for every ladder depth ``text``
    reaches (incremental sha256: one pass over the text). Pure; never raises."""
    try:
        if not text:
            return None
        fp: dict[int, str] = {}
        digest = hashlib.sha256()
        prev = 0
        for depth in PREFIX_LADDER_DEPTHS_CHARS:
            if len(text) < depth:
                break
            digest.update(text[prev:depth].encode("utf-8", "surrogatepass"))
            prev = depth
            fp[depth] = digest.copy().hexdigest()[:_FP_HEX]
        return {"chars": len(text), "fp": fp}
    except Exception:
        return None


def prefix_ladder(request: Any) -> dict[str, Any] | None:
    """The KVU-15c history ladder of the text ``prefix_fingerprints`` fingerprints."""
    return prefix_ladder_for_text(_prompt_text_for_fingerprint(request))


# ---------------------------------------------------------------------------
# Record construction and writing
# ---------------------------------------------------------------------------


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _rotation() -> tuple[int, int]:
    try:
        max_mb = int(os.environ.get(MAX_MB_ENV, str(_DEFAULT_MAX_MB)))
    except ValueError:
        max_mb = _DEFAULT_MAX_MB
    try:
        keep = int(os.environ.get(KEEP_ENV, str(_DEFAULT_KEEP)))
    except ValueError:
        keep = _DEFAULT_KEEP
    return max_mb * 1024 * 1024, max(1, keep)


def _maybe_rotate(path: Path) -> None:
    """Size-based rotation ``f -> f.1 -> ... -> f.keep``. Caller holds the flock."""
    max_bytes, keep = _rotation()
    if max_bytes <= 0:
        return
    try:
        if path.stat().st_size < max_bytes:
            return
    except OSError:
        return
    for i in range(keep, 0, -1):
        src = path if i == 1 else path.with_name(f"{path.name}.{i - 1}")
        dst = path.with_name(f"{path.name}.{i}")
        if i == keep:
            try:
                dst.unlink()
            except OSError:
                pass
        try:
            if src.exists():
                src.rename(dst)
        except OSError:
            pass


def record_digest(record: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of ``record`` minus its own ``record_sha256``.

    Lets a reader re-derive each line's digest (a self-hashed row): a mutated line
    no longer matches, so an attestation built on it grades down rather than up.
    """
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    canonical = json.dumps(_json_safe(body), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_record(record: dict[str, Any]) -> bool:
    """Append one self-hashed record under a cross-process flock (several uvicorn workers)."""
    path = log_path()
    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = _json_safe(dict(record))
        record["record_sha256"] = record_digest(record)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        lock_path = path.with_name(path.name + ".lock")
        with _write_lock, open(lock_path, "a") as lock_fh:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                _maybe_rotate(path)
                with open(path, "a") as fh:
                    fh.write(line)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        _feed_serving_params(record)
        _bump("recorded")
        return True
    except Exception as exc:
        _bump("write_failed", exc)
        return False


def _feed_serving_params(record: dict[str, Any]) -> None:
    """UFH14-B1 F1: a written record with a long prefill is a prefill-rate sample for
    ``src/backends/serving_params.py`` in this process at once (other workers' records
    arrive through its periodic log re-read). Never raises."""
    try:
        from src.backends import serving_params

        serving_params.observe_record(record)
    except Exception:
        pass


def classify_outcome(result: Any, exc: BaseException | None, early_stop: bool) -> str:
    """One word for how the call ended.

    ``ok`` · ``early_stop`` (the caller's on_chunk stopped the stream: FINAL marker,
    repetition guard or client cancel) · ``timeout`` · ``cancelled`` ·
    ``context_overflow`` · ``failed`` · ``exception`` · ``refused`` (a gate refused
    the call before dispatch; the exception carries a structured ``refusal`` dict
    whose ``gate`` names it, e.g. ``role_parked``).
    """
    if exc is not None:
        if isinstance(getattr(exc, "refusal", None), dict):
            return "refused"
        text = f"{type(exc).__name__} {exc}".lower()
        if "timeout" in text:
            return "timeout"
        if "cancel" in text:
            return "cancelled"
        return "exception"
    reason = " ".join(
        str(getattr(result, attr, "") or "")
        for attr in ("completion_reason", "failure_reason", "error_message")
    ).lower()
    if getattr(result, "context_overflow", None):
        return "context_overflow"
    if "timeout" in reason:
        return "timeout"
    if "cancel" in reason:
        return "cancelled"
    if not getattr(result, "success", True):
        return "failed"
    if early_stop:
        return "early_stop"
    return "ok"


def _cached_prompt_tokens(result: Any, timings: Any) -> Any:
    """Cached prompt tokens for the call: the result's value, else llama's ``cache_n``.

    The ``/completion`` lane never sets ``InferenceResult.cached_prompt_tokens``, so
    without the fallback the field is null on that lane and the prefix-hit metric
    silently loses those calls (UFH14-B4). ``cache_n`` is the server's own count.
    """
    value = getattr(result, "cached_prompt_tokens", None)
    if value is None and isinstance(timings, dict):
        cache_n = timings.get("cache_n")
        if isinstance(cache_n, int) and not isinstance(cache_n, bool):
            return cache_n
    return value


#: Staged keys that become their own top-level record block, not caller fields.
#: ``kv_admission`` (KVU-15c): the shared-KV-pool admission decision for the call —
#: the long-prefill verdict, the cached-prefix credit and its source.
#: ``serving_params`` (UFH14-B1 F1): the derived prefill allowance added to the call's
#: timeout, the timeout after the deadline clamp, and ``doomed`` when the remaining
#: request budget cannot cover the prefill (the server would keep prefilling an
#: abandoned request).
_BLOCK_KEYS = ("kv_admission", "serving_params")


def _caller_block(staged: dict[str, Any] | None) -> dict[str, Any]:
    caller = {
        k: v
        for k, v in (staged or {}).items()
        if not k.startswith("_") and k not in _QUEUE_KEYS and k not in _BLOCK_KEYS
    }
    caller["source"] = "primitives" if staged and "_ts0" in staged else "unstaged"
    return caller


def build_record(
    *,
    method: str,
    role_config: Any,
    request: Any,
    base_url: str | None,
    ts_start: float,
    ts_end: float,
    result: Any = None,
    exc: BaseException | None = None,
    staged: dict[str, Any] | None = None,
    notes: dict[str, Any] | None = None,
    dispatched: bool = True,
) -> dict[str, Any]:
    """Assemble one ``serving_call.v1`` record (pure, for tests and the writer)."""
    notes = dict(notes or {})
    port = _port_of(base_url)
    early_stop = bool(notes.pop("early_stop", False))
    timings = notes.pop("timings", None)
    model = getattr(role_config, "model", None)
    queue: dict[str, Any] = {}
    if staged and "_ts0" in staged:
        queue["pre_dispatch_wait_ms"] = round(
            max(0.0, (ts_start - staged["_ts0"]) * 1000.0), 3
        )
    for key in _QUEUE_KEYS:
        if staged and key in staged:
            queue[key] = staged[key]
        if key in notes:
            queue[key] = notes.pop(key)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "record_id": uuid.uuid4().hex,
        "ts_start": _iso(ts_start),
        "ts_end": _iso(ts_end),
        "wall_ms": round((ts_end - ts_start) * 1000.0, 3),
        "dispatched": dispatched,
        "method": method,
        "role": getattr(role_config, "name", None) or getattr(request, "role", None),
        "request_role": getattr(request, "role", None),
        "server": {"base_url": base_url, "port": port, **server_identity(port)},
        "model_registry": getattr(model, "name", None) if model is not None else None,
        "caller": _caller_block(staged),
        "queue": queue,
        "timings": timings,
        "timings_source": "server" if timings else "absent",
        "outcome": classify_outcome(result, exc, early_stop),
        "provenance": process_provenance(),
    }
    for key in _BLOCK_KEYS:
        if staged and isinstance(staged.get(key), dict):
            record[key] = dict(staged[key])
    if result is not None:
        record["result"] = {
            "success": getattr(result, "success", None),
            "completion_reason": getattr(result, "completion_reason", None) or None,
            "failure_reason": getattr(result, "failure_reason", None) or None,
            "error_message": (str(getattr(result, "error_message", "") or "")[:300] or None),
            "tokens_generated": getattr(result, "tokens_generated", None),
            "prompt_tokens": getattr(result, "prompt_tokens", None),
            "cached_prompt_tokens": _cached_prompt_tokens(result, timings),
            "prompt_eval_ms": getattr(result, "prompt_eval_ms", None),
            "generation_ms": getattr(result, "generation_ms", None),
            "first_token_ms": getattr(result, "first_token_ms", None) or None,
            "stream_chunks": getattr(result, "stream_chunks", None) or None,
        }
    # UFH14-LB2/LB6: the backend's own stop signal as a first-class top-level field.
    # ``/completion`` reports ``stop_type`` (eos/limit/word), the chat lane the OpenAI
    # ``finish_reason`` (stop/length/tool_calls); both reach ``completion_reason``.
    # A ``length``/``limit`` value with ``tokens_generated`` at ``request.n_tokens`` is a
    # budget cut, which is what separates a truncated parse failure from a malformed one.
    record["finish_reason"] = (
        (record.get("result") or {}).get("completion_reason") if result is not None else None
    )
    if exc is not None:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
        refusal = getattr(exc, "refusal", None)
        if isinstance(refusal, dict):
            # A refusal is structural: ``refusal.gate`` names the gate, so readers
            # never classify the message text (UFH14-B6a). Never dispatched.
            record["refusal"] = dict(refusal)
            record["dispatched"] = False
    if request is not None:
        record["request"] = {
            "n_tokens": getattr(request, "n_tokens", None),
            "prompt_chars": len(getattr(request, "prompt", None) or ""),
            "timeout_s": getattr(request, "timeout", None),
            "slot_id": getattr(request, "slot_id", None),
            # The chat lane never puts id_slot on the wire, so a router-assigned
            # slot_id there was computed and dropped (UFH14-B4). Only /completion sends it.
            "slot_id_sent": (
                getattr(request, "slot_id", None) is not None
                and str(notes.get("endpoint") or "").startswith("/completion")
            ),
            "chat_payload": getattr(request, "chat_payload", None) is not None,
            "prefix_fp": prefix_fingerprints(request),
        }
    if request is not None:
        # RTE-LOG-1: ordered prompt blocks (kind/start/length/sha256_12 only).
        try:
            from src.backends import prompt_layout

            record.update(prompt_layout.fields_for_request(request))
        except Exception:
            pass
    if notes:
        record["notes"] = notes
    return record


def _last_call_summary(record: dict[str, Any]) -> dict[str, Any]:
    """The few fields a parse event needs about the generation it is parsing."""
    result = record.get("result") or {}
    request = record.get("request") or {}
    return {
        "record_id": record.get("record_id"),
        "role": record.get("role"),
        "method": record.get("method"),
        "finish_reason": record.get("finish_reason"),
        "n_tokens": result.get("tokens_generated"),
        "max_tokens": request.get("n_tokens"),
        "outcome": record.get("outcome"),
    }


def record_refusal(
    *,
    role: str | None,
    base_url: str | None,
    refusal: dict[str, Any],
    caller: dict[str, Any] | None = None,
    method: str = "refused",
    exc: BaseException | None = None,
) -> None:
    """Record a call a gate refused before any backend saw it. Never raises.

    ``outcome = "refused"``, ``dispatched = False`` and the structured
    ``refusal`` block (``refusal.gate`` is the gate's name). Clears any staged
    caller so a later call does not inherit it.
    """
    try:
        staged = _STAGED.get()
        _STAGED.set(None)
        now = time.time()
        record = build_record(
            method=method,
            role_config=None,
            request=None,
            base_url=base_url,
            ts_start=(staged or {}).get("_ts0", now),
            ts_end=now,
            exc=exc,
            dispatched=False,
        )
        record["role"] = role
        record["request_role"] = role
        block = _caller_block(staged)
        block.update({k: v for k, v in (caller or {}).items() if v is not None})
        record["caller"] = block
        record["outcome"] = "refused"
        record["refusal"] = dict(refusal)
        write_record(record)
    except Exception:
        pass


def remember_served(record: dict[str, Any], ladder: dict[str, Any] | None) -> None:
    """Feed a served call into the KVU-15c prefix history (the admission gate's
    cached-prefix credit). Only a completed, dispatched call counts — its prompt is
    then in a slot or in ``--cache-ram``. Never raises."""
    try:
        if not ladder:
            return
        from src.scheduling import prefix_history

        prefix_history.remember_record(record, ladder)
    except Exception:
        pass


def observe_prefix_index(record: dict[str, Any], request: Any) -> None:
    """Feed a served call into its server's RTG-58 P2 prefix index. A no-op unless
    ``ORCHESTRATOR_PREFIX_INDEX`` is on. Never raises."""
    try:
        from src.inference import prefix_index

        if not prefix_index.enabled():
            return
        prefix_index.observe_record(
            record, prefix_index.key_text_for_request(request),
            key_kind=prefix_index.key_kind_for_request(request),
        )
    except Exception:
        pass


def abandon_staged(exc: BaseException | None = None) -> None:
    """Record a staged call that never reached a backend, then clear the stage.

    A call that WAS dispatched consumed its stage and wrote its own record, so this
    is a no-op for it. What is left is a call that died queued — lock timeout,
    cancellation or deadline while waiting — which used to leave no trace at all.
    """
    staged = _STAGED.get()
    if not staged:
        return
    _STAGED.set(None)
    try:
        now = time.time()
        record = build_record(
            method="undispatched",
            role_config=None,
            request=None,
            base_url=staged.get("backend_url"),
            ts_start=staged.get("_ts0", now),
            ts_end=now,
            exc=exc,
            staged=None,
            dispatched=False,
        )
        record["role"] = staged.get("role")
        record["caller"] = _caller_block(staged)
        record["queue"] = {
            "pre_dispatch_wait_ms": round(max(0.0, (now - staged.get("_ts0", now)) * 1000.0), 3)
        }
        for key in _BLOCK_KEYS:
            if isinstance(staged.get(key), dict):
                record[key] = dict(staged[key])
        write_record(record)
    except Exception:
        pass


#: UFH14-LB6: inter-chunk gap capture. A "gap" is the wall time between two consecutive
#: ``on_chunk`` deliveries of one streamed call (first chunk to last chunk; time to first
#: token is NOT a gap). A "stall" is a gap at or above ``STALL_THRESHOLD_MS``. The raw gap
#: list is kept (capped) so any percentile can be re-derived and its sample audited: a
#: p99.9 over 80 gaps is the max of 80 gaps, and the record says so (``n_gaps``,
#: ``p999_min_samples_met``, ``stall_idx``) rather than leaving the reader to guess.
STALL_THRESHOLD_MS = 500.0
GAP_LIST_CAP = 4096
P999_MIN_SAMPLES = 1000


def _nearest_rank(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    rank = max(1, min(len(sorted_vals), math.ceil(q * len(sorted_vals) - 1e-9)))
    return sorted_vals[rank - 1]


def stream_gap_stats(gaps_ms: list[float], *, stall_ms: float = STALL_THRESHOLD_MS) -> dict[str, Any] | None:
    """Summary block for one stream's inter-chunk gaps (None when fewer than one gap).

    Percentiles are nearest-rank over the full gap list. ``stall_idx`` lists the arrival
    index (0-based position in the gap list) of each stall, so stall arrivals can be
    counted per stream and not only in aggregate. The raw list is truncated at
    ``GAP_LIST_CAP`` entries (``gaps_truncated``); the statistics always use every gap.
    """
    if not gaps_ms:
        return None
    ordered = sorted(gaps_ms)
    n = len(gaps_ms)
    stall_idx = [i for i, g in enumerate(gaps_ms) if g >= stall_ms]
    return {
        "n_gaps": n,
        "p50_ms": round(_nearest_rank(ordered, 0.50), 3),
        "p99_ms": round(_nearest_rank(ordered, 0.99), 3),
        "p999_ms": round(_nearest_rank(ordered, 0.999), 3),
        "p999_min_samples_met": n >= P999_MIN_SAMPLES,
        "max_ms": round(ordered[-1], 3),
        "stall_threshold_ms": stall_ms,
        "stall_count": len(stall_idx),
        "stall_idx": stall_idx[:256],
        "gaps_ms": [round(g, 3) for g in gaps_ms[:GAP_LIST_CAP]],
        "gaps_truncated": n > GAP_LIST_CAP,
    }


def _wrap_on_chunk(on_chunk: Callable[[str], Any] | None, notes: dict[str, Any]):
    if on_chunk is None:
        return None

    def _observed(content: str) -> Any:
        if "first_chunk_at" not in notes:
            notes["first_chunk_at"] = time.time()
        mono = time.monotonic()  # monotonic for the gap; wall time only stamps chunk 1
        prev = notes.get("_last_chunk_mono")
        if prev is not None:
            notes.setdefault("_gaps_ms", []).append((mono - prev) * 1000.0)
        notes["_last_chunk_mono"] = mono
        try:
            return on_chunk(content)
        except StopIteration:
            notes["early_stop"] = True
            raise

    return _observed


def _dispatch_direct(
    staged: dict[str, Any] | None, request: Any, role_config: Any, backend: Any,
    ts_start: float, result: Any, error: BaseException | None,
) -> None:
    """dispatch.v1 for a call that did NOT go through ``ConcurrencyAwareBackend``
    (single-server GPU / non-fleet roles). Fleet calls staged an ``instance_idx`` and
    are recorded by the fleet scope, so they are skipped here. Never raises."""
    try:
        if staged and "instance_idx" in staged:
            return
        from src.backends import dispatch_log

        st = staged or {}
        url = getattr(getattr(backend, "config", None), "base_url", None)
        prompt = getattr(request, "prompt", None)
        n_tokens = getattr(request, "n_tokens", None)
        dispatch_log.record_standalone(
            lane="direct",
            role=st.get("role") or getattr(role_config, "name", None) or getattr(request, "role", None),
            base_url=url,
            port=_port_of(url),
            ts_arrival=st.get("_ts0") or ts_start,
            ts_dispatch=ts_start,
            ts_end=time.time(),
            request_id=st.get("request_id"),
            session_id=st.get("session_id"),
            client_class=st.get("client_class"),
            client_class_provenance=st.get("client_class_provenance"),
            client=st.get("client"),
            workload_class=st.get("workload_class"),
            priority=st.get("priority"),
            trace_keys=st.get("trace_keys"),
            prompt_chars=len(prompt) if isinstance(prompt, str) else None,
            n_tokens_budget=n_tokens if isinstance(n_tokens, int) else None,
            outcome=classify_outcome(result, error, False),
            error_type=type(error).__name__ if error is not None else None,
            preceding_tool=st.get("preceding_tool"),
            turn_trigger=st.get("turn_trigger"),
            finish_reason=getattr(result, "completion_reason", None) or None,
        )
    except Exception:
        pass


def recorded_call(method: str) -> Callable:
    """Decorate a backend entry point ``(self, role_config, request, ...)``.

    Writes exactly one record per outermost call, after the call returns or raises;
    the call's result and exceptions pass through untouched.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self: Any, role_config: Any, request: Any, *args: Any, **kwargs: Any):
            if _IN_CALL.get():
                _bump("skipped_nested")
                return fn(self, role_config, request, *args, **kwargs)
            # Do not attribute a previous or unpersisted record to this call.
            _LAST_CALL.set(None)
            if log_path() is None:
                _bump("skipped_disabled")
                return fn(self, role_config, request, *args, **kwargs)
            staged = _STAGED.get()
            _STAGED.set(None)
            notes: dict[str, Any] = {}
            in_token = _IN_CALL.set(True)
            notes_token = _NOTES.set(notes)
            if "on_chunk" in kwargs:
                kwargs["on_chunk"] = _wrap_on_chunk(kwargs["on_chunk"], notes)
            elif args and callable(args[0]):
                args = (_wrap_on_chunk(args[0], notes),) + tuple(args[1:])
            ts_start = time.time()
            result: Any = None
            error: BaseException | None = None
            try:
                result = fn(self, role_config, request, *args, **kwargs)
                return result
            except BaseException as exc:  # recorded, then re-raised unchanged
                error = exc
                raise
            finally:
                _IN_CALL.reset(in_token)
                _NOTES.reset(notes_token)
                try:
                    first = notes.pop("first_chunk_at", None)
                    if first is not None:
                        notes["first_chunk_ms"] = round((first - ts_start) * 1000.0, 3)
                    notes.pop("_last_chunk_mono", None)
                    gaps = notes.pop("_gaps_ms", None)
                    if gaps:
                        notes["stream_gaps"] = stream_gap_stats(gaps)
                    record = build_record(
                        method=method,
                        role_config=role_config,
                        request=request,
                        base_url=getattr(getattr(self, "config", None), "base_url", None),
                        ts_start=ts_start,
                        ts_end=time.time(),
                        result=result,
                        exc=error,
                        staged=staged,
                        notes=notes,
                    )
                    if write_record(record):
                        _LAST_CALL.set(_last_call_summary(record))
                    try:
                        from src.backends import prompt_layout

                        prompt_layout.clear_layout()  # consumed by this record
                    except Exception:
                        pass
                    _dispatch_direct(staged, request, role_config, self, ts_start, result, error)
                    remember_served(record, prefix_ladder(request))
                    observe_prefix_index(record, request)
                except Exception as exc:
                    _bump("build_failed", exc)

        return wrapper

    return decorator


def record_http_call(
    *,
    method: str,
    role: str | None,
    base_url: str | None,
    endpoint: str | None,
    ts_start: float,
    ts_end: float,
    prompt: str | None = None,
    n_tokens: int | None = None,
    status: int | None = None,
    finish_reason: str | None = None,
    timings: Any = None,
    usage: Any = None,
    tokens_generated: int | None = None,
    prompt_tokens: int | None = None,
    exc: BaseException | None = None,
    notes: dict[str, Any] | None = None,
) -> bool:
    """UFH14-LB6.2: write one ``serving_call.v1`` record for an HTTP call that does NOT
    go through ``LlamaServerBackend`` (so ``recorded_call`` cannot wrap it).

    Same schema and writer as :func:`recorded_call` (``build_record`` + ``write_record``),
    ``method`` names the lane (``worker_pool_completion``, ``repair_chat`` ...) and
    ``notes.endpoint`` the URL path. Never raises, never changes the call; a call made
    inside a recorded backend call is not double-counted (``skipped_nested``).
    """
    if _IN_CALL.get():
        _bump("skipped_nested")
        return False
    if log_path() is None:
        _bump("skipped_disabled")
        return False
    try:
        from types import SimpleNamespace

        n: dict[str, Any] = dict(notes or {})
        if endpoint:
            n["endpoint"] = endpoint
        if status is not None:
            n["http_status"] = status
        if isinstance(timings, dict) and ("prompt_n" in timings or "predicted_n" in timings):
            n["timings"] = {k: timings[k] for k in TIMING_KEYS if k in timings}
        if isinstance(usage, dict) and usage:
            n["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage}
            if tokens_generated is None:
                tokens_generated = usage.get("completion_tokens")
            if prompt_tokens is None:
                prompt_tokens = usage.get("prompt_tokens")
        ok = exc is None and (status is None or 200 <= int(status) < 300)
        result = SimpleNamespace(
            success=ok,
            completion_reason=finish_reason,
            failure_reason=None if ok else (f"http_{status}" if status is not None else None),
            error_message=None,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            prompt_eval_ms=None,
            generation_ms=None,
            first_token_ms=None,
            stream_chunks=None,
        )
        record = build_record(
            method=method,
            role_config=SimpleNamespace(name=role, model=None),
            request=SimpleNamespace(role=role, prompt=prompt, n_tokens=n_tokens, timeout=None,
                                    slot_id=None, chat_payload=None),
            base_url=base_url,
            ts_start=ts_start,
            ts_end=ts_end,
            result=result,
            exc=exc,
            staged=None,
            notes=n,
        )
        return write_record(record)
    except Exception as e:
        _bump("build_failed", e)
        return False
