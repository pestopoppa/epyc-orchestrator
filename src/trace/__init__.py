"""Unified trace / memory service.

Read-only query layer over fragmented audit/trace formats:
- /workspace/logs/agent_audit.log (JSON + legacy text dual format)
- /workspace/progress/YYYY-MM/*.md (+ .jsonl when present)
- orchestration/autopilot_journal.{tsv,jsonl} + autopilot_state.json

Source files keep their existing writers. This package ingests them
into a SQLite store with FTS5 indices for cross-source queries.

Per handoffs/active/unified-trace-memory-service.md.
"""

from src.trace.store import (
    DEFAULT_DB_PATH,
    Event,
    EventCategory,
    EventSource,
    ensure_schema,
    upsert_events,
)
from src.trace.query import query

__all__ = [
    "DEFAULT_DB_PATH",
    "Event",
    "EventCategory",
    "EventSource",
    "ensure_schema",
    "upsert_events",
    "query",
]
