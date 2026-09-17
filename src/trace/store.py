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
from dataclasses import dataclass
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
    # Live in-process push source (TM-4). Rows carry a synthetic source_path
    # (see src/trace/emit.py) so they never collide with file-ingested rows.
    REVIEW_PLANE = "review_plane"
    # Live in-process push from AutoPilot's trial loop (2026-08-03).
    # DISTINCT from AUTOPILOT_JOURNAL, which is the same trials scraped
    # back out of the journal file afterwards. Before this existed the
    # trace store only ever learned about a trial if someone remembered to
    # run ingest — trial 1460 wrote 9 episodic rows and 0 trace events.
    AUTOPILOT_LIVE = "autopilot_live"


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

    # --- Architect->Reviewer control-plane categories (TM-2) ---
    # Emitted live via the review-plane push path (src/trace/emit.py) and
    # consumed by the decision-chain replay (src/trace/query.py::decision_chain).
    REVIEW_DECISION = "review_decision"
    CANDIDATE_PACKAGE = "candidate_package"
    VERIFICATION_REPORT = "verification_report"
    REVIEW_ESCALATION = "review_escalation"
    PLAN_REMINDER = "plan_reminder"

    # --- CP2 semantics-layer categories (spec §12.2) ---
    # Additive: only the §12.2 categories not already present above. The
    # immutable-decision invariant (§5.5) is realized as APPEND-ONLY events —
    # a material-input change appends DECISION_INVALIDATED referencing the
    # superseded decision; history is never rewritten in place (§12.3).
    DECISION_INVALIDATED = "decision_invalidated"
    ESCALATION_CREATED = "escalation_created"
    ESCALATION_RESOLVED = "escalation_resolved"
    EVIDENCE_REQUESTED = "evidence_requested"
    EVIDENCE_RESULT = "evidence_result"


# Event-row schema version. v1 = the original 13 columns (T1). v2 (UTM-P1,
# 2026-09-17) adds the cross-harness pairing keys below plus a per-row
# ``schema_version`` stamp. Bumped only on additive, backward-compatible changes.
EVENT_SCHEMA_VERSION = 2

# Added by ALTER TABLE on a v1 store. Every column is nullable and existing rows
# keep NULL; the CREATE TABLE below already carries them for a fresh store.
EVENT_PAIRING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("harness", "TEXT"),
    ("seed", "INTEGER"),
    ("turn_ordinal", "INTEGER"),
    ("task_key", "TEXT"),
    ("schema_version", "INTEGER"),
)


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
    - harness, seed, turn_ordinal, task_key: UTM-P1 pairing keys (event schema
      v2). Two runs of the same ``task_key`` under different ``harness`` values,
      with the same ``seed``, are paired turn-by-turn on ``turn_ordinal`` (the
      horizon / turn index, 0-based). All four are optional: file-ingested
      sources that carry none of them leave them NULL.
    - schema_version: the event-row schema the row was WRITTEN under.
      ``EVENT_SCHEMA_VERSION`` for new rows; rows written before v2 read back
      as NULL, which means "pairing keys were never captured" -- not "absent".
      Never back-fill it (a stamp invented after the fact would claim a capture
      that never happened).
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
    # --- UTM-P1 pairing keys (event schema v2) ---
    harness: str | None = None
    seed: int | None = None
    turn_ordinal: int | None = None
    task_key: str | None = None
    schema_version: int = 0  # 0 -> stamped with EVENT_SCHEMA_VERSION in __post_init__

    def __post_init__(self) -> None:
        if not self.schema_version:
            self.schema_version = EVENT_SCHEMA_VERSION
        self.seed = _coerce_optional_int(self.seed, "seed")
        self.turn_ordinal = _coerce_optional_int(self.turn_ordinal, "turn_ordinal")
        if self.turn_ordinal is not None and self.turn_ordinal < 0:
            raise ValueError(f"turn_ordinal must be >= 0, got {self.turn_ordinal}")
        if self.harness is not None:
            self.harness = str(self.harness).strip() or None
        if self.task_key is not None:
            self.task_key = str(self.task_key).strip() or None

    def pairing_key(self) -> tuple[str | None, int | None, int | None]:
        """``(task_key, seed, turn_ordinal)`` -- the identity two harnesses share."""
        return (self.task_key, self.seed, self.turn_ordinal)

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
            self.harness,
            self.seed,
            self.turn_ordinal,
            self.task_key,
            self.schema_version,
        )


def _coerce_optional_int(value: object, field_name: str) -> int | None:
    """Accept ints and integral strings; refuse bools, floats and junk loudly.

    A seed silently coerced from ``True`` or ``3.7`` would pair runs that were
    never on the same seed, which is exactly the defect UTM-P1 exists to remove.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise TypeError(f"{field_name} must be an int or integral string, got {value!r}")


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
  harness TEXT,
  seed INTEGER,
  turn_ordinal INTEGER,
  task_key TEXT,
  schema_version INTEGER,
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


# --- H4 RC-1: reviewer FA/FR calibration ledger --------------------------------
# Additive, append-only table living alongside `event` in the same store. One row
# per reviewer DECISION (decision ≈ question — RC-7 evidence-plane alignment). The
# column list is exactly the handoff's, plus provenance links back to the trace
# `event` rows (`event_source_path` = the emit:// synthetic path a live review
# verdict was written under; `event_id` = that row's integer id), an RA-10
# `schema_version` stamp threaded through from the review-artifact schema, and
# (RC-9, `review_ledger.v2`) the `rubric_json` / `per_item_grades_json` snapshots.
#
# Writer/reader API + the sequential demotion monitor live in
# `src/trace/review_ledger.py`; this module owns only the DDL (idempotent
# CREATE TABLE IF NOT EXISTS) so a single `ensure_schema()` yields the full store.
_REVIEW_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_ledger (
  id INTEGER PRIMARY KEY,
  decision_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  reviewer_model_quant TEXT,
  grading_model TEXT,
  rubric_version TEXT,
  corpus_id TEXT,
  candidate_id TEXT,
  domain TEXT,
  decision TEXT,
  tripwire INTEGER,
  confidence REAL,
  gold_label TEXT,
  gold_source TEXT,
  gold_instrument_version TEXT,
  rationale_cause_match INTEGER,
  latency_ms REAL,
  tokens INTEGER,
  family_match_flag INTEGER,
  era TEXT,
  event_source_path TEXT,
  event_id INTEGER,
  schema_version TEXT,
  created_ts_utc TEXT NOT NULL,
  rubric_json TEXT,
  per_item_grades_json TEXT,
  UNIQUE(decision_id)
);
CREATE INDEX IF NOT EXISTS rl_ts ON review_ledger(ts);
CREATE INDEX IF NOT EXISTS rl_reviewer ON review_ledger(reviewer_model_quant);
CREATE INDEX IF NOT EXISTS rl_corpus ON review_ledger(corpus_id);
CREATE INDEX IF NOT EXISTS rl_candidate ON review_ledger(candidate_id);
CREATE INDEX IF NOT EXISTS rl_domain ON review_ledger(domain);
CREATE INDEX IF NOT EXISTS rl_group ON review_ledger(
  reviewer_model_quant, grading_model, rubric_version, corpus_id, domain
);
"""


# RC-9 `review_ledger.v2`: nullable canonical-JSON snapshots of the full rubric
# applied and the per-item grades behind a decision. Added by ALTER TABLE on a
# pre-v2 (`review_ledger.v1`) table; existing rows keep NULL (no back-fill — a
# snapshot invented after the fact would claim a capture that never happened).
# Order matters only for ALTER; the CREATE above already carries them.
_REVIEW_LEDGER_V2_COLUMNS: tuple[tuple[str, str], ...] = (
    ("rubric_json", "TEXT"),
    ("per_item_grades_json", "TEXT"),
)


def _migrate_review_ledger_v2(conn: sqlite3.Connection) -> None:
    """Add the RC-9 v2 columns to a v1 table. Idempotent, additive, no back-fill.

    A read-only connection (a reader opening a v1 ledger it may not write) and a
    concurrent migrator that won the race are both tolerated: readers decode the
    absent columns as NULL, so neither needs the ALTER to succeed.
    """
    present = {row[1] for row in conn.execute("PRAGMA table_info(review_ledger)")}
    for name, decl in _REVIEW_LEDGER_V2_COLUMNS:
        if name in present:
            continue
        try:
            conn.execute(f"ALTER TABLE review_ledger ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "duplicate column" in msg or "readonly" in msg or "read-only" in msg:
                continue
            raise
    try:
        conn.commit()
    except sqlite3.OperationalError:
        pass


def ensure_review_ledger_schema(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create the H4 `review_ledger` table if absent, then migrate it to v2.

    Idempotent + additive: a fresh store gets the v2 DDL; a v1 store gains the
    RC-9 columns via ALTER TABLE, with its existing rows left NULL. On a
    read-only connection to an existing ledger the DDL cannot run; that is
    tolerated (readers decode absent v2 columns as NULL), anything else raises.
    """
    try:
        conn.executescript(_REVIEW_LEDGER_SCHEMA)
        conn.commit()
    except sqlite3.OperationalError as exc:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_ledger'"
        ).fetchone()
        if not (exists and "readonly" in str(exc).lower()):
            raise
        return conn
    _migrate_review_ledger_v2(conn)
    return conn


# --- CP2: append-only DecisionEnvelope ledger (spec §6.6 / §12) ----------------
# One row per policy decision emitted by the deterministic reducer (immutable-
# decision invariant §5.5). Content-addressed material inputs (subject +
# governance + inputs hashes) are stored so a decision is replayable (§12.4) and
# a material change is detectable (§12.3). `idempotency_key` (= material hash)
# collapses byte-identical re-decisions (§20.1.7); `sequence_no` gives monotonic
# causal order. Corrections/appeals INSERT a NEW row that `supersedes` a prior
# one; invalidation is recorded as a DECISION_INVALIDATED *event* (never an
# UPDATE of this row). Writer/reader/invalidation/replay API live in
# `src/trace/review_ledger.py`; this module owns only the idempotent DDL so a
# single `ensure_schema()` yields the full store.
_DECISION_ENVELOPE_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_envelope (
  id INTEGER PRIMARY KEY,
  decision_event_id TEXT NOT NULL,
  sequence_no INTEGER,
  created_at TEXT,
  idempotency_key TEXT NOT NULL,
  -- subject (content-addressed)
  task_id TEXT,
  artifact_hash TEXT,
  specification_hash TEXT,
  candidate_package_hash TEXT,
  -- governance (content-addressed)
  assurance_profile_hash TEXT,
  policy_hash TEXT,
  rubric_hash TEXT,
  verifier_registry_hash TEXT,
  -- inputs (content-addressed)
  review_decision_hash TEXT,
  verification_report_hash TEXT,
  -- calibration snapshot
  cohort_id TEXT,
  sample_count INTEGER,
  estimated_error_rate REAL,
  upper_risk_bound REAL,
  -- policy_result
  action TEXT,
  blocking_reason_codes TEXT,          -- JSON array
  -- validity lineage (set at write time only; never back-edited)
  supersedes TEXT,
  invalidated_by TEXT,
  valid_until_material_change INTEGER,
  -- combined content hash of ALL §12.3 material inputs (invalidation key)
  material_hash TEXT,
  -- full envelope payload as emitted (JSON), for replay
  envelope_json TEXT,
  schema_version TEXT,
  created_ts_utc TEXT NOT NULL,
  UNIQUE(idempotency_key)
);
CREATE INDEX IF NOT EXISTS de_decision ON decision_envelope(decision_event_id);
CREATE INDEX IF NOT EXISTS de_seq ON decision_envelope(sequence_no);
CREATE INDEX IF NOT EXISTS de_task ON decision_envelope(task_id);
CREATE INDEX IF NOT EXISTS de_artifact ON decision_envelope(artifact_hash);
CREATE INDEX IF NOT EXISTS de_material ON decision_envelope(material_hash);
CREATE INDEX IF NOT EXISTS de_supersedes ON decision_envelope(supersedes);
"""


def ensure_decision_envelope_schema(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create the CP2 `decision_envelope` table if absent. Idempotent + additive."""
    conn.executescript(_DECISION_ENVELOPE_SCHEMA)
    conn.commit()
    return conn


_EVENT_PAIRING_INDEX = (
    "CREATE INDEX IF NOT EXISTS event_pairing ON event(task_key, seed, turn_ordinal, harness)"
)


def event_columns(conn: sqlite3.Connection) -> set[str]:
    """Column names currently present on ``event`` (empty if the table is absent)."""
    return {row[1] for row in conn.execute("PRAGMA table_info(event)")}


def migrate_event_pairing_columns(conn: sqlite3.Connection) -> bool:
    """Bring a v1 ``event`` table to v2. Idempotent, additive, no back-fill.

    Returns True when the table carries every pairing column afterwards. A
    read-only connection to a v1 store, or a concurrent migrator that won the
    race, is tolerated: readers (``src.trace.query``) project absent columns as
    NULL, so neither needs the ALTER to succeed.
    """
    present = event_columns(conn)
    if not present:
        return False
    for name, decl in EVENT_PAIRING_COLUMNS:
        if name in present:
            continue
        try:
            conn.execute(f"ALTER TABLE event ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "duplicate column" in msg or "readonly" in msg or "read-only" in msg:
                continue
            raise
    present = event_columns(conn)
    complete = all(name in present for name, _ in EVENT_PAIRING_COLUMNS)
    if complete:
        try:
            conn.execute(_EVENT_PAIRING_INDEX)
        except sqlite3.OperationalError as exc:
            if "readonly" not in str(exc).lower() and "read-only" not in str(exc).lower():
                raise
    try:
        conn.commit()
    except sqlite3.OperationalError:
        pass
    return complete


def ensure_schema(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create the schema if absent, return an open connection.

    Caller is responsible for closing the connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    # UTM-P1: a store created before v2 keeps its table (CREATE IF NOT EXISTS is
    # a no-op on it), so the pairing columns arrive by additive ALTER instead.
    migrate_event_pairing_columns(conn)
    # Apply the shared harness/trace schema (intake-607 cluster: HLE/BSV/URE/EXM).
    # Imported lazily to keep the event-store core importable on its own.
    from src.trace.harness_schema import ensure_harness_schema

    ensure_harness_schema(conn)
    # H4 RC-1: additive reviewer calibration ledger (co-located in the same store).
    ensure_review_ledger_schema(conn)
    # CP2: additive append-only DecisionEnvelope ledger (co-located in the same store).
    ensure_decision_envelope_schema(conn)
    return conn


def upsert_events(conn: sqlite3.Connection, events: Iterable[Event]) -> tuple[int, int]:
    """Idempotently insert events. Returns (inserted, skipped_duplicates).

    `INSERT OR IGNORE` honors the (source_path, source_line) UNIQUE constraint:
    re-ingesting the same source line is a no-op.
    """
    inserted = 0
    skipped = 0
    # A caller may hand in a connection to a v1 store that never went through
    # ensure_schema(); migrate it rather than failing on the v2 INSERT.
    if not migrate_event_pairing_columns(conn):
        raise sqlite3.OperationalError(
            "event table is missing the v2 pairing columns and could not be migrated"
        )
    cur = conn.cursor()
    for ev in events:
        cur.execute(
            "INSERT OR IGNORE INTO event "
            "(ts_utc, source, source_path, source_line, session_id, trial_id, "
            "role, category, status, summary, detail_json, redacted, "
            "harness, seed, turn_ordinal, task_key, schema_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
