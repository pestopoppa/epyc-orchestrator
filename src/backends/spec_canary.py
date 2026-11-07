"""SPEC-CANARY.1: passive per-request acceptance-collapse canary for speculative serving roles.

Two silent speculative-decoding failures leave a request "successful" while the server burns
draft compute for nothing or loops: (1) the drafter proposes tokens and the target accepts none,
(2) the generation falls into a repeating n-gram tail. This module flags both per request, from
facts the serving-call recorder already holds, and exposes counters through
``serving_calls.health()`` (``/health/serving-calls`` -> ``spec_canary``).

Flags (thresholds are module constants, overridable by env for tests only):

* ``zero_acceptance``: ``draft_n >= 64`` and ``draft_n_accepted == 0`` on the request.
* ``repetition_tail``: the last 256 generated units are periodic with a period of at most 128
  units, i.e. one n-gram block repeats at least twice to cover the tail. Units are the server's
  token ids when the caller supplies them, otherwise a word/punctuation split of the output text
  (a proxy: a looping model repeats at either granularity).

Each flagged request is recorded with role, port, slot, n_parallel, draft settings (read from the
stack's per-port launch sidecar argv) and the request id. Observing NEVER changes serving
behaviour and never raises; it is OFF whenever the serving-call recorder is off.
"""

from __future__ import annotations

import json
import re
import threading
from collections import deque
from pathlib import Path
from typing import Any, Sequence

MIN_DRAFTED = 64
TAIL_UNITS = 256
MAX_PERIOD = TAIL_UNITS // 2
EVENT_LIMIT = 200

_UNIT = re.compile(r"\w+|[^\w\s]")
_lock = threading.Lock()
_COUNTERS: dict[str, int] = {
    "observed": 0,
    "speculative": 0,
    "zero_acceptance": 0,
    "repetition_tail": 0,
    "errors": 0,
}
_EVENTS: deque[dict[str, Any]] = deque(maxlen=EVENT_LIMIT)

_DRAFT_FLAGS = (
    "--spec-type", "--spec-draft-n-max", "--spec-draft-n-min", "-md", "--model-draft",
    "--draft-max", "--draft-min",
)


def repetition_period(units: Sequence[Any], *, tail: int = TAIL_UNITS, max_period: int = MAX_PERIOD) -> int | None:
    """Smallest period p <= ``max_period`` such that the last ``tail`` units repeat with period p.

    None when fewer than ``tail`` units exist or no such period does.
    """
    if len(units) < tail:
        return None
    t = list(units[-tail:])
    for p in range(1, max_period + 1):
        if all(t[i] == t[i + p] for i in range(tail - p)):
            return p
    return None


def text_units(text: str | None) -> list[str]:
    return _UNIT.findall(text or "")


def draft_settings(argv: Sequence[str] | None) -> dict[str, Any]:
    """n_parallel and the draft flags from a llama-server argv (absent -> empty)."""
    out: dict[str, Any] = {}
    if not argv:
        return out
    a = [str(x) for x in argv]
    for i, tok in enumerate(a):
        if tok in ("-np", "--parallel") and i + 1 < len(a):
            out["n_parallel"] = a[i + 1]
        elif tok in _DRAFT_FLAGS and i + 1 < len(a):
            out[tok] = a[i + 1]
    return out


def _sidecar_argv(port: int | None) -> list[str] | None:
    if not port:
        return None
    try:
        from src.backends import serving_calls

        data = json.loads((serving_calls.launch_sidecar_dir() / f"{int(port)}.json").read_text())
        argv = data.get("argv") if isinstance(data, dict) else None
        return [str(x) for x in argv] if isinstance(argv, list) else None
    except Exception:
        return None


def observe(
    record: dict[str, Any],
    *,
    output_text: str | None = None,
    output_units: Sequence[Any] | None = None,
) -> list[str]:
    """Check one ``serving_call.v1`` record (+ the generated output); return the flags raised.

    Never raises. A record without server timings, or whose server drafted nothing (no draft
    configured), counts as ``observed`` only.
    """
    flags: list[str] = []
    try:
        timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
        drafted = int(timings.get("draft_n") or 0)
        accepted = int(timings.get("draft_n_accepted") or 0)
        speculative = drafted > 0
        if speculative and drafted >= MIN_DRAFTED and accepted == 0:
            flags.append("zero_acceptance")
        units = list(output_units) if output_units is not None else text_units(output_text)
        period = repetition_period(units) if speculative else None
        if period is not None:
            flags.append("repetition_tail")
        with _lock:
            _COUNTERS["observed"] += 1
            _COUNTERS["speculative"] += int(speculative)
            for f in flags:
                _COUNTERS[f] += 1
        if flags:
            server = record.get("server") if isinstance(record.get("server"), dict) else {}
            port = server.get("port")
            argv = _sidecar_argv(port)
            settings = draft_settings(argv)
            event = {
                "flags": flags,
                "ts_end": record.get("ts_end"),
                "request_id": ((record.get("caller") or {}).get("request_id") if isinstance(record.get("caller"), dict) else None)
                or record.get("record_id"),
                "record_id": record.get("record_id"),
                "role": record.get("role"),
                "port": port,
                "slot": timings.get("id_slot", (record.get("request") or {}).get("slot_id") if isinstance(record.get("request"), dict) else None),
                "n_parallel": settings.get("n_parallel"),
                "draft_settings": {k: v for k, v in settings.items() if k != "n_parallel"} or None,
                "draft_n": drafted,
                "draft_n_accepted": accepted,
                "repeat_period": period,
            }
            with _lock:
                _EVENTS.append(event)
    except Exception:
        with _lock:
            _COUNTERS["errors"] += 1
    return flags


def snapshot() -> dict[str, Any]:
    """Counters plus the most recent flagged requests (newest last), for the health field."""
    with _lock:
        c = dict(_COUNTERS)
        events = list(_EVENTS)[-20:]
    c["status"] = "collapse_seen" if (c["zero_acceptance"] or c["repetition_tail"]) else "ok"
    return {**c, "recent": events, "thresholds": {
        "min_drafted": MIN_DRAFTED, "tail_units": TAIL_UNITS, "max_period": MAX_PERIOD}}


def reset() -> None:
    """Test hook."""
    with _lock:
        for k in _COUNTERS:
            _COUNTERS[k] = 0
        _EVENTS.clear()
