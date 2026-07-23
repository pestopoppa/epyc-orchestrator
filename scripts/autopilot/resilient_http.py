"""resilient_post — fleet-marker-aware retry wrapper around httpx.

Wraps a single httpx POST so that an exogenous service reload
(orchestrator API restart, llama-server reload by operator) doesn't
get journaled as a real autopilot failure. The wrapper captures fleet
identifiers BEFORE the POST, and on a connection-class exception it
re-queries the watcher: if any identifier changed, the failure was
exogenous and the wrapper waits for /health + retries once before
falling back.

Returns (response_dict, meta_dict). The meta dict carries enough info
for downstream consumers (eval_tower, seeder) to classify the trial:

    clean:                  no exception, no retry, no marker change
    exogenous_recovered:    retry succeeded after operator_reload
    exogenous_unrecovered:  retry exhausted / wait timed out / still failing
    external_restart:       at least one marker had source != stack_commands
    real_failure:           exception with no marker change AND service is
                            reachable (was_restarted_since returned {})
    retry_count:            how many retries actually happened (0 or 1)
    wait_s:                 seconds spent in wait_for_recovery
    marker_changes:         dict from was_restarted_since (key→classification)

If the watcher is None (autopilot off / dev mode), behavior is identical
to the existing direct httpx.post + except-all pattern: exceptions are
swallowed and the {"answer": "", "error": str(e)} dict returned, with
an empty meta. No behavior change for non-autopilot callers.

See handoffs/active/autopilot-exogenous-restart-resilience.md sections
5.3 + 5.4 (consumers of the meta dict).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.observability import classify_exception as _classify_exception
except Exception:  # pragma: no cover
    def _classify_exception(exc: Exception) -> tuple[str, str]:
        return ("unexpected_error", f"{type(exc).__name__}: {exc}")

log = logging.getLogger("autopilot.resilient")


# Reasons that warrant the exogenous-restart check. Other exception
# classes (ValueError, RuntimeError, etc.) bypass the retry path entirely.
_RETRYABLE_REASONS = {
    "connect_error",
    "connect_timeout",
    "read_timeout",
    "timeout",
    "request_error",
}


def _empty_meta() -> dict[str, Any]:
    return {
        "clean": False,
        "exogenous_recovered": False,
        "exogenous_unrecovered": False,
        "external_restart": False,
        "real_failure": False,
        "retry_count": 0,
        "wait_s": 0.0,
        "marker_changes": {},
        # Classification of the terminal exception, when the request ended in
        # failure ("" on clean success). Additive fields — downstream consumers
        # that read specific keys are unaffected. call_orchestrator_forced's
        # eval reconnect-backoff reads `reason` to decide whether a watcher-path
        # failure was connection-level (and thus worth backing off + retrying).
        "reason": "",
        "detail": "",
    }


def resilient_post(
    url: str,
    *,
    json: dict[str, Any],
    timeout: float,
    client: httpx.Client | None = None,
    watcher: Any | None = None,
    llama_port: int | None = None,
    llama_role: str | None = None,
    max_retries: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """POST to url with watcher-aware retry on exogenous reload.

    Args:
        url: full URL including endpoint path (e.g. "http://localhost:8000/chat").
        json: request body.
        timeout: per-request timeout in seconds.
        client: optional httpx.Client to reuse (connection pooling). If None,
            uses module-level httpx.post.
        watcher: OrchestratorWatcher instance. When None, behavior is
            identical to the pre-existing direct httpx call.
        llama_port: optional explicit llama port hint. When set, the watcher
            also checks that llama-server's marker. Takes precedence over
            llama_role.
        llama_role: optional role name. When set (and llama_port is None),
            the watcher resolves the port via /llama_fleet_ids.
        max_retries: budget for exogenous-failure retries only. Default 1.

    Returns:
        (response_dict, meta_dict). On exception, response_dict is
        {"answer": "", "error": str(exc)} preserving the existing
        call_orchestrator_forced shape. Meta-dict keys documented at module top.
    """
    meta = _empty_meta()

    # Resolve port from role if needed.
    if llama_port is None and llama_role and watcher is not None:
        try:
            llama_port = watcher.port_for_role(llama_role)
        except Exception:
            llama_port = None

    # Snapshot fleet identifiers before the POST.
    if watcher is not None:
        try:
            ref_ids = watcher.reference_for_role(llama_role) if llama_role else {}
            if llama_port is not None and f"llama_{llama_port}" not in ref_ids:
                # Caller supplied explicit port hint without a role; add it.
                pair = watcher.current_llama_id(llama_port)
                if pair is not None:
                    ref_ids[f"llama_{llama_port}"] = pair[0]
        except Exception as exc:
            log.debug("watcher.reference_for_role failed: %s", exc)
            ref_ids = {}
    else:
        ref_ids = {}

    def _do_post() -> dict[str, Any]:
        if client is not None:
            r = client.post(url, json=json, timeout=timeout)
        else:
            r = httpx.post(url, json=json, timeout=timeout)
        if r.status_code >= 400:
            try:
                data = r.json()
            except Exception:
                r.raise_for_status()
                raise
            if isinstance(data, dict) and (
                data.get("error_code") or data.get("error_detail")
            ):
                return data
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            # Bytes/text fallback — preserves legacy behavior where the
            # caller might not have expected JSON.
            return {"answer": r.text}

    # Attempt 1.
    try:
        result = _do_post()
        meta["clean"] = True
        return result, meta
    except Exception as exc:
        reason, detail = _classify_exception(exc)
        meta["reason"] = reason
        meta["detail"] = detail
        # Non-retryable exception classes: don't even consult the
        # watcher. Mark as real failure.
        if reason not in _RETRYABLE_REASONS or watcher is None:
            meta["real_failure"] = True
            return {"answer": "", "error": detail}, meta

        # Retryable: ask the watcher if something restarted.
        try:
            watcher.invalidate_cache()  # force fresh reads now that we know there's a problem
            changes = watcher.was_restarted_since(ref_ids)
        except Exception as wexc:
            log.debug("watcher.was_restarted_since failed: %s", wexc)
            changes = {}
        meta["marker_changes"] = changes

        if not changes:
            # Nothing restarted, and the request still failed. Real failure.
            meta["real_failure"] = True
            return {"answer": "", "error": detail}, meta

        # At least one marker changed. Wait for recovery, then retry.
        if any(c == "external_restart" for c in changes.values()):
            meta["external_restart"] = True
        wait_t0 = time.perf_counter()
        recovered = watcher.wait_for_recovery(changes)
        meta["wait_s"] = time.perf_counter() - wait_t0
        if not recovered:
            meta["exogenous_unrecovered"] = True
            return {"answer": "", "error": detail}, meta

        if max_retries < 1:
            meta["exogenous_unrecovered"] = True
            return {"answer": "", "error": detail}, meta

        # Retry up to max_retries times.
        last_detail = detail
        for attempt in range(1, max_retries + 1):
            meta["retry_count"] = attempt
            try:
                result = _do_post()
                meta["exogenous_recovered"] = True
                return result, meta
            except Exception as retry_exc:
                retry_reason, last_detail = _classify_exception(retry_exc)
                meta["reason"] = retry_reason
                meta["detail"] = last_detail
                continue

        meta["exogenous_unrecovered"] = True
        return {"answer": "", "error": last_detail}, meta
