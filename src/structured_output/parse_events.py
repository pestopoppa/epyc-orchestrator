"""Durable parse / repair events (UFH14-LB2, intake-2388).

Before this module the structured-output repair counters
(``STRUCTURED_OUTPUT_REPAIR_COUNTS``), the tool-call JSON repair counters
(``TOOL_CALL_JSON_REPAIR_COUNTS``) and the review-decision ``ParseFailure`` were
in-process only, and no parse event recorded how the generation it was parsing
ended. A parse failure caused by ``finish_reason == "length"`` (a budget cut) is a
sizing problem; the same failure after ``finish_reason == "stop"`` is a malformed
emission. They need opposite fixes and could not be told apart.

One JSONL record per event, three kinds:

* ``structured_output`` -- every ``parse_with_repair`` / ``parse_with_repair_async``
  outcome (``parsed`` / ``repaired`` / ``declined`` / ``failed``), keyed by ``site``;
* ``tool_call_json`` -- the tool-call JSON repair (``repaired`` / ``unrecoverable``);
* ``review_decision`` -- ``parse_review_decision`` success or ``ParseFailure`` reason.

Each carries ``finish_reason``, ``n_tokens``, ``max_tokens`` and the call ``site`` of
the GENERATION being parsed. The source is, in order: an explicit
:func:`generation_context` set by a call site that knows it; else the last
``serving_calls`` record made in this same context (``finish_reason_source =
"last_call_in_context"`` -- it is the most recent call of this thread, not a proof it
produced this exact text); else null with ``finish_reason_source = "absent"``. A
reader must treat only the first two as evidence, and count the third as unknown
rather than as "malformed". A repair turn's own stop signal rides separately in
``repair_turn``.

Behaviour: additive. Every failure is swallowed (a counter and one WARNING record
it). Kill switch / path: ``ORCHESTRATOR_PARSE_EVENTS_LOG`` (``off``/``0``/``none``
disables); unset it follows ``serving_calls`` (so the suite's
``ORCHESTRATOR_SERVING_CALLS_LOG=off`` silences it too).

The 14-day classification the task asks for is :func:`classify` over the rows; run
``python -m src.structured_output.parse_events [path]`` for the per-site table.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

try:  # pragma: no cover - fcntl exists on every supported host
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

SCHEMA = "epyc.orchestrator.parse_event.v1"
LOG_ENV = "ORCHESTRATOR_PARSE_EVENTS_LOG"
_OFF = {"off", "0", "none", "false", "disabled"}

#: ``finish_reason`` / ``stop_type`` values that mean the generation hit its budget.
LENGTH_FINISH_REASONS = frozenset({"length", "limit", "max_tokens", "context_limit"})
#: Values that mean the model stopped on its own (or on a tool call).
NATURAL_FINISH_REASONS = frozenset({"stop", "eos", "word", "tool_calls", "end_turn"})

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()
_failed_once = False

_GENERATION: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "parse_events_generation", default=None
)
_REPAIR_TURN: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "parse_events_repair_turn", default=None
)


def log_path() -> Path | None:
    """Where events go, or None when disabled."""
    from src.backends import serving_calls

    override = os.environ.get(LOG_ENV, "").strip()
    if override.lower() in _OFF:
        return None
    if override:
        return Path(override)
    if serving_calls.log_path() is None:
        return None
    return serving_calls._log_dir() / "parse_events" / "parse_events.jsonl"


@contextlib.contextmanager
def generation_context(
    *,
    finish_reason: str | None = None,
    n_tokens: int | None = None,
    max_tokens: int | None = None,
    call_site: str | None = None,
) -> Iterator[None]:
    """Declare how the generation parsed inside this block ended.

    For a call site that holds the backend result and knows its own
    ``completion_reason`` / ``tokens_generated`` / budget; wins over the
    last-call-in-context fallback.
    """
    token = _GENERATION.set(
        {
            "finish_reason": finish_reason,
            "n_tokens": n_tokens,
            "max_tokens": max_tokens,
            "call_site": call_site,
        }
    )
    try:
        yield
    finally:
        _GENERATION.reset(token)


def note_repair_turn(
    *, finish_reason: str | None, max_tokens: int | None, n_tokens: int | None = None
) -> None:
    """Record how the REPAIR turn's own generation ended (set by a ``CompleteFn``)."""
    try:
        _REPAIR_TURN.set(
            {"finish_reason": finish_reason, "max_tokens": max_tokens, "n_tokens": n_tokens}
        )
    except Exception:
        pass


def note_repair_turn_from_last_call(*, max_tokens: int | None = None) -> None:
    """For a ``CompleteFn`` built on the backend layer: the call it just made is the
    last recorded call in this context, so its stop signal is the repair turn's."""
    try:
        from src.backends import serving_calls

        last = serving_calls.last_call() or {}
        note_repair_turn(
            finish_reason=last.get("finish_reason"),
            max_tokens=max_tokens if max_tokens is not None else last.get("max_tokens"),
            n_tokens=last.get("n_tokens"),
        )
    except Exception:
        pass


_SNAPSHOT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "parse_events_snapshot", default=None
)


def begin_parse() -> None:
    """Freeze the generation being parsed BEFORE any repair turn can overwrite
    ``serving_calls.last_call()``; the next :func:`record_parse_event` in this context
    consumes the snapshot. Never raises."""
    try:
        _SNAPSHOT.set(_generation_fields())
    except Exception:
        pass


def _generation_fields() -> dict[str, Any]:
    explicit = _GENERATION.get()
    if explicit and explicit.get("finish_reason") is not None:
        return {
            "finish_reason": explicit.get("finish_reason"),
            "n_tokens": explicit.get("n_tokens"),
            "max_tokens": explicit.get("max_tokens"),
            "generation_site": explicit.get("call_site"),
            "finish_reason_source": "explicit",
        }
    try:
        from src.backends import serving_calls

        last = serving_calls.last_call()
    except Exception:
        last = None
    if last and last.get("finish_reason") is not None:
        return {
            "finish_reason": last.get("finish_reason"),
            "n_tokens": last.get("n_tokens"),
            "max_tokens": last.get("max_tokens"),
            "generation_site": last.get("role"),
            "generation_record_id": last.get("record_id"),
            "finish_reason_source": "last_call_in_context",
        }
    return {
        "finish_reason": None,
        "n_tokens": (explicit or {}).get("n_tokens"),
        "max_tokens": (explicit or {}).get("max_tokens"),
        "generation_site": (explicit or {}).get("call_site"),
        "finish_reason_source": "absent",
    }


def truncated_by_length(fields: dict[str, Any]) -> bool | None:
    """True/False when the generation's stop signal is known, else None (unknown is
    NOT "malformed")."""
    fr = fields.get("finish_reason")
    if fr is None:
        return None
    fr = str(fr).lower()
    if fr in LENGTH_FINISH_REASONS:
        return True
    n, cap = fields.get("n_tokens"), fields.get("max_tokens")
    if fr in NATURAL_FINISH_REASONS:
        return False
    if isinstance(n, int) and isinstance(cap, int) and cap > 0 and n >= cap:
        return True
    return None


def record_parse_event(
    kind: str,
    site: str,
    status: str,
    *,
    reason: str = "",
    repair_calls: int = 0,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Append one ``parse_event.v1`` record. Never raises; returns whether written."""
    global _failed_once
    try:
        path = log_path()
        if path is None:
            return False
        from src.backends import serving_calls

        gen = _SNAPSHOT.get() or _generation_fields()
        _SNAPSHOT.set(None)
        repair_turn = _REPAIR_TURN.get()
        _REPAIR_TURN.set(None)
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "record_id": uuid.uuid4().hex,
            "ts": serving_calls._iso(time.time()),
            "kind": kind,
            "site": site,
            "status": status,
            "reason": (reason or "")[:300],
            "repair_calls": repair_calls,
            **gen,
            "truncated_by_length": truncated_by_length(gen),
            "repair_turn": repair_turn,
            "provenance": serving_calls.process_provenance(),
        }
        if extra:
            record["extra"] = extra
        record = serving_calls._json_safe(record)
        record["record_sha256"] = serving_calls.record_digest(record)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(path.name + ".lock")
        with _write_lock, open(lock_path, "a") as lock_fh:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                serving_calls._maybe_rotate(path)
                with open(path, "a") as fh:
                    fh.write(line)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return True
    except Exception as exc:
        if not _failed_once:
            _failed_once = True
            _log.warning("parse_events write failed (first occurrence): %s: %s",
                         type(exc).__name__, exc)
        return False


# --------------------------------------------------------------------------- analysis


def read_events(path: Path) -> Iterator[dict[str, Any]]:
    """Rows of one JSONL file whose self-hash matches; a mutated line is skipped."""
    from src.backends import serving_calls

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("record_sha256") == serving_calls.record_digest(row):
                yield row


def classify(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per (kind, site): parse failures split ``length_truncated`` / ``malformed`` /
    ``unknown_stop`` (finish_reason absent) against the ``parsed`` denominator.

    A failure is ``status in {"failed", "unrecoverable"}`` or a review ``ParseFailure``
    (any status other than ``parsed``/``repaired``/``declined`` for review_decision).
    ``repaired`` counts as a parse miss that the repair turn recovered
    (``repaired_after_length`` / ``repaired_after_stop``). A
    ``provenance_unknown`` diagnostic has its own bucket and is not a parse failure.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = f"{row.get('kind')}:{row.get('site')}"
        status = row.get("status")
        trunc = row.get("truncated_by_length")
        out[key]["events"] += 1
        if status == "parsed":
            out[key]["parsed"] += 1
        elif status == "provenance_unknown":
            out[key]["provenance_unknown"] += 1
        elif status == "declined":
            out[key]["declined"] += 1
        elif status == "repaired":
            out[key]["repaired_after_length" if trunc is True
                     else "repaired_after_stop" if trunc is False
                     else "repaired_unknown_stop"] += 1
        else:
            out[key]["length_truncated" if trunc is True
                     else "malformed" if trunc is False
                     else "unknown_stop"] += 1
    return {k: dict(v) for k, v in out.items()}


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else log_path()
    if path is None or not Path(path).exists():
        print("no parse_events file", file=sys.stderr)
        return 1
    table = classify(read_events(Path(path)))
    print(json.dumps(table, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
