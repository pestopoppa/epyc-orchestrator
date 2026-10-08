"""Prospective immutable HG5 intervention facts; disabled without an owned destination.

Source-authored run context never proves runtime origin. Unknown counters remain
nullable. This producer neither grades requests nor backfills legacy tap events.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = "hg5.request_intervention.v1"
SCHEMA_V2 = "hg5.request_intervention.v2"
CATEGORIES = frozenset({"OPTIMUM", "BASELINE", "CANDIDATE"})
DIRECTIONS = frozenset({"higher_better", "lower_better"})
COUNTERS = {
    "calls": "total_calls",
    "prompt_tokens": "total_prompt_tokens_reported",
    "completion_tokens": "total_tokens_generated",
    "prompt_ms": "total_prompt_eval_ms",
    "generation_ms": "total_generation_ms",
}
INTEGER_COUNTERS = frozenset({"calls", "prompt_tokens", "completion_tokens"})


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def snapshot(owner: Any) -> dict[str, int | float | None]:
    """Read actual counters; do not estimate, clamp, coerce or replace missing values."""
    result = {}
    for key, attribute in COUNTERS.items():
        try:
            value = getattr(owner, attribute, None) if owner is not None else None
        except Exception:
            value = None
        valid = type(value) is int if key in INTEGER_COUNTERS else type(value) in (int, float)
        result[key] = value if valid and value >= 0 and (type(value) is int or math.isfinite(value)) else None
    return result


def counter_delta(before: dict, after: dict) -> dict:
    result = {}
    for key in COUNTERS:
        first, last = before.get(key), after.get(key)
        result[key] = last - first if first is not None and last is not None and last >= first else None
    return result


def record_step(capture: dict, owner: Any, *, trigger: str, initial_role: str,
                target_role: str, outcome: str) -> None:
    """Keep observed outcome separate from whether a call-count delta is known."""
    after = snapshot(owner)
    before = capture.pop("step_before", {key: None for key in after})
    delta = counter_delta(before, after)
    calls = delta["calls"]
    capture["steps"].append({"trigger": trigger, "initial_role": initial_role,
                             "target_role": target_role, "outcome": outcome,
                             "call_status": "unknown" if calls is None else "not_called" if calls == 0 else "called",
                             "before": before, "after": after, "delta": delta})


def strict_json_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate source context key")
        result[key] = value
    return result


def projection_declaration(value: Any) -> bytes:
    """Validate explicit pre-capture labels; never infer direction or a protocol."""
    if not isinstance(value, dict) or set(value) != {"category", "metric_direction", "protocol_id"}:
        raise ValueError("projection declaration needs exactly category/directions/protocol_id")
    if not isinstance(value["category"], str) or value["category"] not in CATEGORIES:
        raise ValueError("unsupported explicitly declared category")
    directions = value["metric_direction"]
    if not isinstance(directions, dict) or set(directions) != set(COUNTERS) or any(
            not isinstance(direction, str) or direction not in DIRECTIONS for direction in directions.values()):
        raise ValueError("each supported cost metric needs an explicit valid direction")
    protocol = value["protocol_id"]
    # The empty string is an explicit absence declaration, not an invented citation.
    # A nonempty source citation is not human ratification or runtime authorization.
    if not isinstance(protocol, str) or (protocol != "" and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/#-]{0,127}", protocol) is None):
        raise ValueError("invalid explicitly declared protocol identity")
    return canonical(value)


def begin_capture() -> dict | None:
    """Capture prospective context only when this independent writer is explicitly enabled."""
    destination = os.environ.get("HG5_REQUEST_EVENT_DIRECTORY", "")
    if not destination:
        return None
    context_raw = os.environ.get("HG5_REQUEST_EVENT_SOURCE_CONTEXT", "")
    source = {"origin": "unknown", "commit": None, "tree": None, "producer_sha256": None}
    declaration = None
    if context_raw:
        context = json.loads(context_raw, object_pairs_hook=strict_json_pairs)
        if not isinstance(context, dict) or set(context) not in ({"commit", "tree", "producer_sha256"}, {"commit", "tree", "producer_sha256", "projection"}):
            raise ValueError("source context must contain commit/tree/producer_sha256 with optional projection only")
        for key, length in (("commit", 40), ("tree", 40), ("producer_sha256", 64)):
            value = context[key]
            if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid source context identity")
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != context["producer_sha256"]:
            raise ValueError("producer bytes differ from source-authored run context")
        source = {"origin": "source_authored_run_context", **{key: context[key] for key in ("commit", "tree", "producer_sha256")}}
        if "projection" in context:
            declaration = projection_declaration(context["projection"])
    return {"destination": destination, "event_id": uuid.uuid4().hex,
            "start_utc_ns": time.time_ns(), "start_monotonic_ns": time.monotonic_ns(),
            "source": source, "steps": [], "stage": None,
            "projection_canonical_at_start": declaration}


def finish_capture(capture: dict | None, *, request_id: str, plan: Any, counters: dict,
                   failure_type: str | None = None, failure_status: str | None = None) -> Path | None:
    """Seal native bytes once; unknown fields are facts about absence, not measured zeros."""
    if capture is None:
        return None
    if len(capture["event_id"]) != 32 or any(c not in "0123456789abcdef" for c in capture["event_id"]):
        raise ValueError("invalid native event identity")
    if type(capture.get("feature_enabled")) is not bool or len(capture["steps"]) > 1:
        raise ValueError("ambiguous feature state or request step cardinality")
    if failure_status not in (None, "none", "reported", "request_failed"):
        raise ValueError("invalid bounded failure status")
    safe_failure_type = failure_type[:80] if isinstance(failure_type, str) and failure_type.isidentifier() else None
    body = {
        "schema": SCHEMA, "event_id": capture["event_id"],
        "request_id_sha256": hashlib.sha256(request_id.encode()).hexdigest(),
        "window": {"start_utc_ns": capture["start_utc_ns"], "end_utc_ns": time.time_ns(),
                   "start_monotonic_ns": capture["start_monotonic_ns"], "end_monotonic_ns": time.monotonic_ns()},
        "source": capture["source"], "runtime_origin": "unknown",
        "requested_mode": plan.requested, "feature_enabled": capture.get("feature_enabled"),
        "eligible": plan.enabled and capture["stage"] == "direct",
        "plan_enabled": plan.enabled, "disabled_reason": plan.disabled_reason,
        "stage": capture["stage"], "initial_role": plan.from_role,
        "final_role": plan.final_answer_role or plan.from_role,
        "requested_target_role": plan.target_role, "steps": capture["steps"],
        "request_counters_at_capture": counters,
        "failure": {"status": failure_status or ("reported" if plan.error else "none"),
                    "exception_type": safe_failure_type,
                    "message": None},
        "claim_scope": "intervention_and_actual_counter_facts_only",
    }
    declaration = capture.get("projection_canonical_at_start")
    if declaration is not None:
        if not isinstance(declaration, bytes):
            raise ValueError("pre-capture declaration custody changed")
        labels = json.loads(declaration, object_pairs_hook=strict_json_pairs)
        if projection_declaration(labels) != declaration:
            raise ValueError("pre-capture declaration bytes changed")
        if body["source"]["origin"] != "source_authored_run_context":
            raise ValueError("version2 labels require captured source-authored context")
        body["schema"] = SCHEMA_V2
        body["projection"] = {"declaration": labels,
                              "declaration_sha256": hashlib.sha256(declaration).hexdigest(),
                              "binding": "source_authored_pre_capture_context",
                              "human_ratification": "not_asserted"}
    raw = canonical(body)
    sealed = canonical({"body": body, "body_sha256": hashlib.sha256(raw).hexdigest()}) + b"\n"
    directory = Path(capture["destination"])
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    file_fd = None
    try:
        state = os.fstat(fd)
        if state.st_uid != os.geteuid() or stat.S_IMODE(state.st_mode) != 0o700:
            raise ValueError("capture directory must be owned mode0700")
        name = capture["event_id"] + ".json"
        file_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        state = os.fstat(file_fd)
        if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1 or stat.S_IMODE(state.st_mode) != 0o600:
            raise ValueError("native event custody differs")
        remaining = memoryview(sealed)
        while remaining:
            written = os.write(file_fd, remaining)
            if written <= 0:
                raise OSError("short native event write")
            remaining = remaining[written:]
        os.fsync(file_fd)
        os.fsync(fd)
        return directory / name
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(fd)
