"""SQLite schema + idempotent upsert for the unified trace store.

Append-only semantics: rows are keyed by (source_path, source_line) and
re-ingesting the same line is a no-op. This mirrors the append-only nature
of the source files (agent_audit.log, autopilot_journal.*) so the store
can be regenerated from sources at any time.

FTS5 virtual tables provide full-text search over `summary` and `detail_json`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Default DB path: under data/trace/ (gitignored).
# Resolves relative to the orchestrator repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = _REPO_ROOT / "data" / "trace" / "events.sqlite"


class EventSource:
    """Canonical source identifiers (string enum, no Enum class for sqlite simplicity)."""

    AGENT_AUDIT = "agent_audit"
    PROGRESS = "progress"
    PROGRESS_JSONL = "progress_jsonl"
    AUTOPILOT_JOURNAL = "autopilot_journal"
    AUTOPILOT_STATE = "autopilot_state"
    HERMES_SESSION = "hermes_session"


class EventCategory:
    """Common category strings. Parsers may emit additional categories."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TASK_START = "task_start"
    TASK_END = "task_end"
    CMD_INTENT = "cmd_intent"
    CMD_RESULT = "cmd_result"
    DECISION = "decision"
    OBSERVE = "observe"
    WARN = "warn"
    ERROR = "error"
    FILE_MODIFY = "file_modify"
    ROLLBACK = "rollback"
    DISCOVERY = "discovery"
    VERIFY = "verify"
    DOCS = "docs"
    SESSION_SUMMARY = "session_summary"
    PARETO_ACCEPT = "pareto_accept"
    MUTATION = "mutation"
    SAFETY_VERDICT = "safety_verdict"
    CONTROLLER_SNAPSHOT = "controller_snapshot"
    SOURCE_UNAVAILABLE = "source_unavailable"


@dataclass
class Event:
    """Normalized event row.

    Field semantics:
    - ts_utc: ISO8601 string, UTC. Best-effort timezone normalization.
    - source: from EventSource (free-form string in storage).
    - category: from EventCategory (free-form string in storage).
    - session_id, trial_id, role, status: optional join keys / facets.
    - summary: short human-readable description.
    - detail_json: full original record encoded as JSON.
    - source_path + source_line: dedup key.
    """

    ts_utc: str
    source: str
    source_path: str
    source_line: int | None = None
    session_id: str | None = None
    trial_id: int | None = None
    role: str | None = None
    category: str | None = None
    status: str | None = None
    summary: str | None = None
    detail_json: str | None = None
    redacted: int = 0

    def as_row(self) -> tuple:
        return (
            self.ts_utc,
            self.source,
            self.source_path,
            self.source_line,
            self.session_id,
            self.trial_id,
            self.role,
            self.category,
            self.status,
            self.summary,
            self.detail_json,
            self.redacted,
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS event (
  id INTEGER PRIMARY KEY,
  ts_utc TEXT NOT NULL,
  source TEXT NOT NULL,
  source_path TEXT NOT NULL,
  source_line INTEGER,
  session_id TEXT,
  trial_id INTEGER,
  role TEXT,
  category TEXT,
  status TEXT,
  summary TEXT,
  detail_json TEXT,
  redacted INTEGER NOT NULL DEFAULT 0,
  UNIQUE(source_path, source_line)
);

CREATE INDEX IF NOT EXISTS event_ts ON event(ts_utc);
CREATE INDEX IF NOT EXISTS event_session ON event(session_id);
CREATE INDEX IF NOT EXISTS event_trial ON event(trial_id);
CREATE INDEX IF NOT EXISTS event_source ON event(source);
CREATE INDEX IF NOT EXISTS event_category ON event(category);

CREATE VIRTUAL TABLE IF NOT EXISTS event_fts USING fts5(
  summary, detail_json,
  content='event', content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

-- Triggers to keep FTS in sync with event.
CREATE TRIGGER IF NOT EXISTS event_ai AFTER INSERT ON event BEGIN
  INSERT INTO event_fts(rowid, summary, detail_json)
  VALUES (new.id, new.summary, new.detail_json);
END;

CREATE TRIGGER IF NOT EXISTS event_ad AFTER DELETE ON event BEGIN
  INSERT INTO event_fts(event_fts, rowid, summary, detail_json)
  VALUES('delete', old.id, old.summary, old.detail_json);
END;

CREATE TRIGGER IF NOT EXISTS event_au AFTER UPDATE ON event BEGIN
  INSERT INTO event_fts(event_fts, rowid, summary, detail_json)
  VALUES('delete', old.id, old.summary, old.detail_json);
  INSERT INTO event_fts(rowid, summary, detail_json)
  VALUES (new.id, new.summary, new.detail_json);
END;
"""


def ensure_schema(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create the schema if absent, return an open connection.

    Caller is responsible for closing the connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def upsert_events(conn: sqlite3.Connection, events: Iterable[Event]) -> tuple[int, int]:
    """Idempotently insert events. Returns (inserted, skipped_duplicates).

    `INSERT OR IGNORE` honors the (source_path, source_line) UNIQUE constraint:
    re-ingesting the same source line is a no-op.
    """
    inserted = 0
    skipped = 0
    cur = conn.cursor()
    for ev in events:
        cur.execute(
            "INSERT OR IGNORE INTO event "
            "(ts_utc, source, source_path, source_line, session_id, trial_id, "
            "role, category, status, summary, detail_json, redacted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ev.as_row(),
        )
        if cur.rowcount == 1:
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    return inserted, skipped


def event_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM event").fetchone()[0]


def detail_to_json(detail: object) -> str:
    """Encode an arbitrary record to canonical JSON for the detail_json column."""
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, default=str, sort_keys=True)
