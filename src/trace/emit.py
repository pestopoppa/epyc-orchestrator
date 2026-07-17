"""Live in-process push interface for the unified trace store (TM-4).

Why this exists
---------------
The file-tailing ingesters (``ingest_agent_audit``/``ingest_progress``/
``ingest_autopilot``) are *pull/offline*: they re-read append-only source files
and idempotently upsert. That is correct for logs that already have a durable
writer on disk, but the Architect->Reviewer control plane produces decisions
in-process that have **no file writer** — review verdicts, candidate packages,
verification reports, escalations, plan reminders. Those need a *push* path so
each decision becomes a durable, replayable row the moment it happens.

Write-through vs re-ingest (the TM-5 decision)
----------------------------------------------
We deliberately choose **write-through**: live events are written straight into
``events.sqlite`` via the same idempotent ``upsert_events`` used by the offline
ingesters, and they are NOT mirrored to a file that later gets re-ingested.

Rationale:
  1. Durability across restart is the whole point (H1). A file that is only
     re-ingested on the next batch run loses everything a crashed process had
     not yet flushed. Writing through commits at emit time.
  2. No second event-sourcing pipeline. The trace store stays a single queryable
     index; write-through keeps one code path (``upsert_events``) rather than a
     mirror-file + re-parse loop that could drift from the offline parsers.
  3. Avoids double-counting. If live rows were also written to a file AND that
     file were ingested, the same decision would appear twice under two
     ``source_path`` values. Write-through with a synthetic source (below)
     guarantees exactly one row per logical event.

Synthetic-source convention (no collision with file-ingested rows)
------------------------------------------------------------------
The store dedups on the ``UNIQUE(source_path, source_line)`` key. File
ingesters set ``source_path`` to a real absolute filesystem path
(``/workspace/logs/agent_audit.log``, ``/mnt/raid0/.../autopilot_journal.jsonl``,
...) and ``source_line`` to the line number. Live rows must never share a key
with those, so live rows use a synthetic URI namespace that no ingester can
ever produce:

    emit://<source>/<part>/<part>/...

Absolute POSIX paths start with ``/``; ``emit://...`` never does, so a live row
and a file row can never collide. Within the ``emit://`` namespace, idempotency
is content/identity addressed:

  * A span row keys on its ``span_id``:      emit://review_plane/span/<span_id>
  * A trace-boundary row keys on trace+kind: emit://review_plane/trace/<trace_id>/<kind>
  * A bare ``emit(Event)`` with no path keys on a stable content hash of the
    event, so re-emitting byte-identical content is a no-op.

``source_line`` is always ``0`` for synthetic rows — the full identity lives in
``source_path`` — which makes re-emitting the same logical event an
``INSERT OR IGNORE`` no-op (append-only + idempotent, matching store.py).

TracingProcessor shape (re-implementation, NOT an import)
---------------------------------------------------------
``ReviewTracingProcessor`` re-implements the 6-method OpenAI-Agents-SDK
``TracingProcessor`` interface over our store, with ZERO new dependencies
(intake-849 P6 pattern-mine, not adopt). The SDK's ``trace_id``/``group_id`` map
onto our columns as:

    trace.trace_id  -> event.session_id     (a review session / conversation)
    trace.group_id  -> event.trial_id        (best-effort int; else kept in detail)
    span.span_id    -> synthetic source_path identity

Spans inherit their parent trace's ``session_id``/``trial_id``. Spans are
persisted on ``on_span_end`` (a complete span carries its duration/outcome);
trace boundaries persist on ``on_trace_start``/``on_trace_end``.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.trace.store import (
    DEFAULT_DB_PATH,
    Event,
    EventCategory,
    EventSource,
    detail_to_json,
    ensure_schema,
    upsert_events,
)

# Namespace prefix for every live-emitted row's ``source_path``. Chosen so it
# can never equal a real filesystem path produced by a file ingester.
SYNTHETIC_SCHEME = "emit"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def synthetic_source_path(*parts: Any) -> str:
    """Build a collision-free ``source_path`` for a live-emitted row.

    ``synthetic_source_path("review_plane", "span", span_id)`` ->
    ``"emit://review_plane/span/<span_id>"``. Parts are stringified and joined
    with ``/``; empty parts are dropped.
    """
    cleaned = [str(p).strip("/") for p in parts if p is not None and str(p) != ""]
    return f"{SYNTHETIC_SCHEME}://" + "/".join(cleaned)


def is_synthetic_source_path(source_path: str | None) -> bool:
    """True if ``source_path`` is in the live-emit namespace (not a file path)."""
    return bool(source_path) and str(source_path).startswith(f"{SYNTHETIC_SCHEME}://")


def _content_key(ev: Event) -> str:
    """Stable short hash of an event's identity-bearing fields.

    Used when a caller emits an Event without supplying its own synthetic
    ``source_path``: byte-identical content collapses to one row.
    """
    material = "\x1f".join(
        str(x)
        for x in (
            ev.ts_utc,
            ev.source,
            ev.session_id,
            ev.trial_id,
            ev.role,
            ev.category,
            ev.status,
            ev.summary,
            ev.detail_json,
        )
    )
    return hashlib.sha1(material.encode("utf-8", "replace")).hexdigest()[:16]


def emit(
    event: Event,
    db_path: Path | str = DEFAULT_DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> tuple[int, int]:
    """Write-through a single live event. Returns ``(inserted, skipped_dup)``.

    Idempotency:
      * If ``event.source_path`` is already a synthetic ``emit://`` key, it is
        used verbatim (caller owns the identity).
      * Otherwise a content-addressed synthetic path is assigned so that
        re-emitting identical content is a no-op.

    ``source_line`` is forced to ``0`` for synthetic rows (identity lives in the
    path). If ``conn`` is supplied it is reused (and committed); otherwise a
    short-lived connection is opened, used, and closed.
    """
    if not event.source:
        event.source = EventSource.REVIEW_PLANE
    if not is_synthetic_source_path(event.source_path):
        event.source_path = synthetic_source_path(event.source, _content_key(event))
        event.source_line = 0
    elif event.source_line is None:
        event.source_line = 0
    if not event.ts_utc:
        event.ts_utc = _now_utc()

    own_conn = conn is None
    if own_conn:
        conn = ensure_schema(db_path)
    try:
        inserted, skipped = upsert_events(conn, [event])
    finally:
        if own_conn:
            conn.close()
    return inserted, skipped


# ---------------------------------------------------------------------------
# TracingProcessor re-implementation (6-method shape, no external dependency)
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    """Lightweight trace handle. ``trace_id`` -> session_id, ``group_id`` -> trial_id."""

    trace_id: str
    group_id: Any | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    ended_at: str | None = None

    @property
    def session_id(self) -> str:
        return self.trace_id

    @property
    def trial_id(self) -> int | None:
        return _coerce_trial(self.group_id)


@dataclass
class Span:
    """Lightweight span handle recorded on ``on_span_end``.

    ``span_data`` carries the review-plane payload: ``category`` (an
    ``EventCategory`` value), ``summary``, ``role``, ``status``, ``latency_ms``,
    plus any additional ``detail`` fields (all persisted into ``detail_json``).
    """

    span_id: str
    trace_id: str
    parent_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    span_data: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None


def _coerce_trial(group_id: Any | None) -> int | None:
    if group_id is None:
        return None
    try:
        return int(group_id)
    except (ValueError, TypeError):
        return None


class ReviewTracingProcessor:
    """Write-through processor mapping traces/spans onto trace-store events.

    Re-implements the OpenAI-Agents-SDK ``TracingProcessor`` 6-method contract
    (``on_trace_start`` / ``on_span_start`` / ``on_span_end`` / ``on_trace_end``
    / ``force_flush`` / ``shutdown``) with no external dependency. Holds one
    persistent connection; each recorded event is committed via ``emit`` so a
    crash after a decision keeps that decision.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        source: str = EventSource.REVIEW_PLANE,
    ) -> None:
        self._db_path = Path(db_path)
        self._source = source
        self._conn = ensure_schema(self._db_path)
        self._open_traces: dict[str, Trace] = {}
        self._inserted = 0
        self._skipped = 0

    # --- 6-method TracingProcessor interface --------------------------------

    def on_trace_start(self, trace: Trace) -> None:
        self._open_traces[trace.trace_id] = trace
        if not trace.started_at:
            trace.started_at = _now_utc()
        self._record(
            source_path=synthetic_source_path(self._source, "trace", trace.trace_id, "start"),
            ts_utc=trace.started_at,
            session_id=trace.session_id,
            trial_id=trace.trial_id,
            category="trace_start",
            summary=trace.name or f"trace {trace.trace_id} started",
            detail={
                "trace_id": trace.trace_id,
                "group_id": trace.group_id,
                "name": trace.name,
                "metadata": trace.metadata,
            },
        )

    def on_span_start(self, span: Span) -> None:
        # Complete spans are persisted on end; start is tracked implicitly via
        # the parent trace. Kept as an explicit no-op to honor the 6-method
        # shape (a subclass may override to record open spans).
        if not span.started_at:
            span.started_at = _now_utc()

    def on_span_end(self, span: Span) -> None:
        trace = self._open_traces.get(span.trace_id)
        data = dict(span.span_data or {})
        category = data.pop("category", None) or EventCategory.REVIEW_DECISION
        summary = data.pop("summary", None)
        role = data.pop("role", None)
        status = data.pop("status", None)
        latency_ms = data.pop("latency_ms", None)
        if latency_ms is None:
            latency_ms = _duration_ms(span.started_at, span.ended_at)
        detail = {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "parent_id": span.parent_id,
            "started_at": span.started_at,
            "ended_at": span.ended_at,
            "latency_ms": latency_ms,
            "error": span.error,
            **data,
        }
        self._record(
            source_path=synthetic_source_path(self._source, "span", span.span_id),
            ts_utc=span.ended_at or _now_utc(),
            session_id=trace.session_id if trace else span.trace_id,
            trial_id=trace.trial_id if trace else None,
            role=role,
            category=category,
            status=status or ("error" if span.error else None),
            summary=summary,
            detail=detail,
        )

    def on_trace_end(self, trace: Trace) -> None:
        if not trace.ended_at:
            trace.ended_at = _now_utc()
        opened = self._open_traces.pop(trace.trace_id, trace)
        self._record(
            source_path=synthetic_source_path(self._source, "trace", trace.trace_id, "end"),
            ts_utc=trace.ended_at,
            session_id=trace.session_id,
            trial_id=trace.trial_id,
            category="trace_end",
            summary=trace.name or f"trace {trace.trace_id} ended",
            detail={
                "trace_id": trace.trace_id,
                "group_id": trace.group_id,
                "name": trace.name,
                "started_at": opened.started_at,
                "ended_at": trace.ended_at,
                "duration_ms": _duration_ms(opened.started_at, trace.ended_at),
            },
        )

    def force_flush(self) -> None:
        """Commit any pending writes. Write-through already commits per event;
        this guarantees the connection is durably flushed on demand."""
        self._conn.commit()

    def shutdown(self) -> None:
        """Flush and close the persistent connection."""
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    # --- convenience --------------------------------------------------------

    @property
    def counts(self) -> dict[str, int]:
        return {"inserted": self._inserted, "skipped": self._skipped}

    def _record(
        self,
        *,
        source_path: str,
        ts_utc: str,
        session_id: str | None,
        trial_id: int | None,
        category: str,
        summary: str | None,
        detail: Any,
        role: str | None = None,
        status: str | None = None,
    ) -> None:
        ev = Event(
            ts_utc=ts_utc,
            source=self._source,
            source_path=source_path,
            source_line=0,
            session_id=session_id,
            trial_id=trial_id,
            role=role,
            category=category,
            status=status,
            summary=summary,
            detail_json=detail_to_json(detail),
        )
        ins, skp = emit(ev, conn=self._conn)
        self._inserted += ins
        self._skipped += skp


def _duration_ms(started_at: str | None, ended_at: str | None) -> float | None:
    if not started_at or not ended_at:
        return None
    try:
        s = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        e = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (e - s).total_seconds() * 1000.0
