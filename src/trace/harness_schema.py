"""Shared harness / trace schema — the single event family for the intake-607 cluster.

Per `handoffs/active/unified-trace-memory-service.md` § "Shared Harness/Trace Schema"
(owner of record) and `routing-and-optimization-index.md` P24 implementation spine,
this module is the **build-first foundation** that HLE / BSV / URE / EXM all write to:

| family             | task   | producer                         |
|--------------------|--------|----------------------------------|
| harness_metrics    | HLE-1  | eval tower / trace ingest        |
| oracle_adequacy    | HLE-2  | suite registration               |
| behavior_signature | BSV-1  | autopilot archive accept path    |
| approval_record    | URE-2  | router / escalation              |
| failure_case       | EXM-1  | trace ingest                     |
| working_state      | EXM-2  | orchestrator working memory      |

Contract (from the handoff):
1. every record carries `schema_version` and is keyed by a stable integer `id`;
2. cross-references use ids (`event_id`, `harness_metrics_id`, ...), never duplicated payloads;
3. schema changes are additive + versioned — no silent field repurposing;
4. consumers tolerate missing fields (`signature_confidence='partial'` for backfilled rows).

These tables live alongside the `event` table in the same SQLite store
(`src/trace/store.py`); `store.ensure_schema()` also applies this schema so a single
`ensure_schema()` yields the complete store.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Bumped only on additive, backward-compatible schema changes.
SCHEMA_VERSION = 1


class GovernanceLevel:
    """Experience-governance tiers (EXM-3 / MemGovern). Higher = more trusted."""

    RAW = "raw"
    AUTO_VERIFIED = "auto_verified"
    HUMAN_REVIEWED = "human_reviewed"
    APPROVED_BASELINE = "approved_baseline"
    DEPRECATED = "deprecated"

    #: rank for relevance scoring; DEPRECATED is intentionally below RAW.
    ORDER = {
        DEPRECATED: -1,
        RAW: 0,
        AUTO_VERIFIED: 1,
        HUMAN_REVIEWED: 2,
        APPROVED_BASELINE: 3,
    }

    ALL = (RAW, AUTO_VERIFIED, HUMAN_REVIEWED, APPROVED_BASELINE, DEPRECATED)


class WorkingStateScope:
    REQUEST = "request"
    TRIAL = "trial"
    SESSION = "session"
    HANDOFF = "handoff"
    ALL = (REQUEST, TRIAL, SESSION, HANDOFF)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jdump(value: object) -> str | None:
    """Encode a list/dict to canonical JSON; pass through None and str."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


# ─── dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class HarnessMetrics:
    """HLE-1: per-trial scores on the named harness axes (rule-based, evidence-linked)."""

    suite: str | None = None
    trial_id: int | None = None
    event_id: int | None = None
    execution_fidelity: float | None = None
    feedback_interpretation: float | None = None
    planning_stability: float | None = None
    memory_coherence: float | None = None
    recovery_rate: float | None = None
    evidence_event_ids: list | str | None = None
    confidence: float | None = None
    schema_version: int = SCHEMA_VERSION
    created_ts_utc: str = field(default_factory=_now_iso)


@dataclass
class OracleAdequacy:
    """HLE-2: whether a suite's success oracle actually covers the failure modes."""

    suite: str = ""
    oracle_type: str | None = None
    coverage_claim: str | None = None
    known_blind_spots: list | str | None = None
    shortcut_risk: str | None = None
    requires_external_answer: bool | None = None
    deterministic: bool | None = None
    reviewed_by: str | None = None
    schema_version: int = SCHEMA_VERSION
    created_ts_utc: str = field(default_factory=_now_iso)


@dataclass
class BehaviorSignature:
    """BSV-1: behavioral fingerprint of an archive member for silent-regression diffing."""

    archive_member_id: str | None = None
    trial_id: int | None = None
    sentinel_outcomes: list | dict | str | None = None
    answer_hash: str | None = None
    route_path_hash: str | None = None
    tool_sequence_hash: str | None = None
    escalation_path_hash: str | None = None
    latency_bucket: str | None = None
    token_bucket: str | None = None
    harness_metrics_id: int | None = None
    oracle_adequacy_version: int | None = None
    signature_hash: str | None = None
    signature_confidence: str = "full"  # 'full' | 'partial'
    schema_version: int = SCHEMA_VERSION
    created_ts_utc: str = field(default_factory=_now_iso)


@dataclass
class ApprovalRecord:
    """URE-2: an escalation/approval decision recorded as first-class harness state."""

    request_id: str | None = None
    event_id: int | None = None
    task_signature: str | None = None
    selected_role: str | None = None
    selected_model: str | None = None
    alternatives: list | str | None = None
    quality_score: float | None = None
    uncertainty_score: float | None = None
    uncertainty_components: dict | str | None = None
    trigger_reason: str | None = None
    approval_boundary: str | None = None
    actor: str | None = None
    downstream_outcome: str | None = None
    behavior_signature_id: int | None = None
    schema_version: int = SCHEMA_VERSION
    created_ts_utc: str = field(default_factory=_now_iso)


@dataclass
class FailureCase:
    """EXM-1: a failed trajectory stored for pattern-matched avoidance."""

    failure_id: str | None = None
    task_signature: str | None = None
    suite: str | None = None
    role_path: str | None = None
    tool_sequence_hash: str | None = None
    files_touched: list | str | None = None
    error_class: str | None = None
    root_cause_label: str | None = None
    avoidance_advice: str | None = None
    evidence_event_ids: list | str | None = None
    resolved_by_event_id: int | None = None
    governance_level: str = GovernanceLevel.RAW
    validity_score: float | None = None
    schema_version: int = SCHEMA_VERSION
    created_ts_utc: str = field(default_factory=_now_iso)


@dataclass
class WorkingState:
    """EXM-2: externalized mid-task working memory (LLMs fail at latent-state persistence)."""

    key: str = ""
    scope: str = WorkingStateScope.REQUEST
    owner: str | None = None
    state_id: str | None = None
    value_json: dict | list | str | None = None
    created_from_event_id: int | None = None
    expires_at: str | None = None
    supersedes_state_id: str | None = None
    superseded: int = 0
    schema_version: int = SCHEMA_VERSION
    created_ts_utc: str = field(default_factory=_now_iso)


# ─── DDL ──────────────────────────────────────────────────────────────────────


_HARNESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS harness_metrics (
  id INTEGER PRIMARY KEY,
  event_id INTEGER,
  trial_id INTEGER,
  suite TEXT,
  execution_fidelity REAL,
  feedback_interpretation REAL,
  planning_stability REAL,
  memory_coherence REAL,
  recovery_rate REAL,
  evidence_event_ids TEXT,
  confidence REAL,
  schema_version INTEGER NOT NULL,
  created_ts_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS hm_trial ON harness_metrics(trial_id);
CREATE INDEX IF NOT EXISTS hm_suite ON harness_metrics(suite);
CREATE INDEX IF NOT EXISTS hm_event ON harness_metrics(event_id);

CREATE TABLE IF NOT EXISTS oracle_adequacy (
  id INTEGER PRIMARY KEY,
  suite TEXT NOT NULL,
  oracle_type TEXT,
  coverage_claim TEXT,
  known_blind_spots TEXT,
  shortcut_risk TEXT,
  requires_external_answer INTEGER,
  deterministic INTEGER,
  reviewed_by TEXT,
  schema_version INTEGER NOT NULL,
  created_ts_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS oa_suite ON oracle_adequacy(suite);

CREATE TABLE IF NOT EXISTS behavior_signature (
  id INTEGER PRIMARY KEY,
  archive_member_id TEXT,
  trial_id INTEGER,
  sentinel_outcomes TEXT,
  answer_hash TEXT,
  route_path_hash TEXT,
  tool_sequence_hash TEXT,
  escalation_path_hash TEXT,
  latency_bucket TEXT,
  token_bucket TEXT,
  harness_metrics_id INTEGER,
  oracle_adequacy_version INTEGER,
  signature_hash TEXT,
  signature_confidence TEXT NOT NULL DEFAULT 'full',
  schema_version INTEGER NOT NULL,
  created_ts_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS bs_archive ON behavior_signature(archive_member_id);
CREATE INDEX IF NOT EXISTS bs_trial ON behavior_signature(trial_id);
CREATE INDEX IF NOT EXISTS bs_hash ON behavior_signature(signature_hash);

CREATE TABLE IF NOT EXISTS approval_record (
  id INTEGER PRIMARY KEY,
  request_id TEXT,
  event_id INTEGER,
  task_signature TEXT,
  selected_role TEXT,
  selected_model TEXT,
  alternatives TEXT,
  quality_score REAL,
  uncertainty_score REAL,
  uncertainty_components TEXT,
  trigger_reason TEXT,
  approval_boundary TEXT,
  actor TEXT,
  downstream_outcome TEXT,
  behavior_signature_id INTEGER,
  schema_version INTEGER NOT NULL,
  created_ts_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ar_request ON approval_record(request_id);
CREATE INDEX IF NOT EXISTS ar_task ON approval_record(task_signature);

CREATE TABLE IF NOT EXISTS failure_case (
  id INTEGER PRIMARY KEY,
  failure_id TEXT,
  task_signature TEXT,
  suite TEXT,
  role_path TEXT,
  tool_sequence_hash TEXT,
  files_touched TEXT,
  error_class TEXT,
  root_cause_label TEXT,
  avoidance_advice TEXT,
  evidence_event_ids TEXT,
  resolved_by_event_id INTEGER,
  governance_level TEXT NOT NULL DEFAULT 'raw',
  validity_score REAL,
  schema_version INTEGER NOT NULL,
  created_ts_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fc_task ON failure_case(task_signature);
CREATE INDEX IF NOT EXISTS fc_suite ON failure_case(suite);
CREATE INDEX IF NOT EXISTS fc_error ON failure_case(error_class);
CREATE INDEX IF NOT EXISTS fc_gov ON failure_case(governance_level);
CREATE VIRTUAL TABLE IF NOT EXISTS failure_case_fts USING fts5(
  task_signature,
  suite,
  role_path,
  files_touched,
  error_class,
  root_cause_label,
  avoidance_advice,
  content='failure_case',
  content_rowid='id'
);

CREATE TABLE IF NOT EXISTS working_state (
  id INTEGER PRIMARY KEY,
  state_id TEXT,
  scope TEXT NOT NULL,
  owner TEXT,
  key TEXT NOT NULL,
  value_json TEXT,
  created_from_event_id INTEGER,
  expires_at TEXT,
  supersedes_state_id TEXT,
  superseded INTEGER NOT NULL DEFAULT 0,
  schema_version INTEGER NOT NULL,
  created_ts_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ws_lookup ON working_state(scope, owner, key, superseded);
CREATE INDEX IF NOT EXISTS ws_state_id ON working_state(state_id);
"""


def ensure_harness_schema(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create the harness/memory record tables if absent. Idempotent."""
    conn.executescript(_HARNESS_SCHEMA)
    conn.execute("INSERT INTO failure_case_fts(failure_case_fts) VALUES('rebuild')")
    conn.commit()
    return conn


# ─── insert helpers ─────────────────────────────────────────────────────────────


def _insert(conn: sqlite3.Connection, table: str, cols: list[str], values: list) -> int:
    placeholders = ", ".join("?" for _ in cols)
    cur = conn.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_harness_metrics(conn: sqlite3.Connection, m: HarnessMetrics) -> int:
    return _insert(
        conn,
        "harness_metrics",
        ["event_id", "trial_id", "suite", "execution_fidelity",
         "feedback_interpretation", "planning_stability", "memory_coherence",
         "recovery_rate", "evidence_event_ids", "confidence",
         "schema_version", "created_ts_utc"],
        [m.event_id, m.trial_id, m.suite, m.execution_fidelity,
         m.feedback_interpretation, m.planning_stability, m.memory_coherence,
         m.recovery_rate, _jdump(m.evidence_event_ids), m.confidence,
         m.schema_version, m.created_ts_utc],
    )


def insert_oracle_adequacy(conn: sqlite3.Connection, o: OracleAdequacy) -> int:
    return _insert(
        conn,
        "oracle_adequacy",
        ["suite", "oracle_type", "coverage_claim", "known_blind_spots",
         "shortcut_risk", "requires_external_answer", "deterministic",
         "reviewed_by", "schema_version", "created_ts_utc"],
        [o.suite, o.oracle_type, o.coverage_claim, _jdump(o.known_blind_spots),
         o.shortcut_risk,
         None if o.requires_external_answer is None else int(o.requires_external_answer),
         None if o.deterministic is None else int(o.deterministic),
         o.reviewed_by, o.schema_version, o.created_ts_utc],
    )


def insert_behavior_signature(conn: sqlite3.Connection, b: BehaviorSignature) -> int:
    return _insert(
        conn,
        "behavior_signature",
        ["archive_member_id", "trial_id", "sentinel_outcomes", "answer_hash",
         "route_path_hash", "tool_sequence_hash", "escalation_path_hash",
         "latency_bucket", "token_bucket", "harness_metrics_id",
         "oracle_adequacy_version", "signature_hash", "signature_confidence",
         "schema_version", "created_ts_utc"],
        [b.archive_member_id, b.trial_id, _jdump(b.sentinel_outcomes), b.answer_hash,
         b.route_path_hash, b.tool_sequence_hash, b.escalation_path_hash,
         b.latency_bucket, b.token_bucket, b.harness_metrics_id,
         b.oracle_adequacy_version, b.signature_hash, b.signature_confidence,
         b.schema_version, b.created_ts_utc],
    )


def insert_approval_record(conn: sqlite3.Connection, a: ApprovalRecord) -> int:
    return _insert(
        conn,
        "approval_record",
        ["request_id", "event_id", "task_signature", "selected_role",
         "selected_model", "alternatives", "quality_score", "uncertainty_score",
         "uncertainty_components", "trigger_reason", "approval_boundary", "actor",
         "downstream_outcome", "behavior_signature_id", "schema_version", "created_ts_utc"],
        [a.request_id, a.event_id, a.task_signature, a.selected_role,
         a.selected_model, _jdump(a.alternatives), a.quality_score, a.uncertainty_score,
         _jdump(a.uncertainty_components), a.trigger_reason, a.approval_boundary, a.actor,
         a.downstream_outcome, a.behavior_signature_id, a.schema_version, a.created_ts_utc],
    )


def insert_failure_case(conn: sqlite3.Connection, f: FailureCase) -> int:
    if f.governance_level not in GovernanceLevel.ALL:
        raise ValueError(f"invalid governance_level: {f.governance_level!r}")
    row_id = _insert(
        conn,
        "failure_case",
        ["failure_id", "task_signature", "suite", "role_path", "tool_sequence_hash",
         "files_touched", "error_class", "root_cause_label", "avoidance_advice",
         "evidence_event_ids", "resolved_by_event_id", "governance_level",
         "validity_score", "schema_version", "created_ts_utc"],
        [f.failure_id, f.task_signature, f.suite, f.role_path, f.tool_sequence_hash,
         _jdump(f.files_touched), f.error_class, f.root_cause_label, f.avoidance_advice,
         _jdump(f.evidence_event_ids), f.resolved_by_event_id, f.governance_level,
         f.validity_score, f.schema_version, f.created_ts_utc],
    )
    _index_failure_case_fts(conn, row_id, f)
    return row_id


def set_working_state(conn: sqlite3.Connection, w: WorkingState) -> int:
    """Insert a working-state record; supersede any prior live row for (scope, owner, key).

    Returns the new row id. Prior rows for the same key are marked superseded=1
    (history is preserved, not deleted).
    """
    if w.scope not in WorkingStateScope.ALL:
        raise ValueError(f"invalid scope: {w.scope!r}")
    conn.execute(
        "UPDATE working_state SET superseded = 1 "
        "WHERE scope = ? AND owner IS ? AND key = ? AND superseded = 0",
        (w.scope, w.owner, w.key),
    )
    return _insert(
        conn,
        "working_state",
        ["state_id", "scope", "owner", "key", "value_json", "created_from_event_id",
         "expires_at", "supersedes_state_id", "superseded", "schema_version", "created_ts_utc"],
        [w.state_id, w.scope, w.owner, w.key, _jdump(w.value_json), w.created_from_event_id,
         w.expires_at, w.supersedes_state_id, w.superseded, w.schema_version, w.created_ts_utc],
    )


# ─── retrieval helpers ───────────────────────────────────────────────────────────


def find_failure_cases(
    conn: sqlite3.Connection,
    task_signature: str,
    *,
    suite: str | None = None,
    exclude_deprecated: bool = True,
    limit: int = 10,
) -> list[dict]:
    """EXM-1 cheap first pass: retrieve prior failures by exact or lexical task match.

    Ordered by exact match, governance trust (approved_baseline first), then recency.
    Embedding refinement is layered on top by the caller per the handoff.
    """
    fts_query = _failure_case_fts_query(task_signature)
    clauses = ["(task_signature = ?"]
    params: list = [task_signature]
    if fts_query:
        clauses[0] += " OR id IN (SELECT rowid FROM failure_case_fts WHERE failure_case_fts MATCH ?)"
        params.append(fts_query)
    clauses[0] += ")"
    if suite is not None:
        clauses.append("suite = ?")
        params.append(suite)
    if exclude_deprecated:
        clauses.append("governance_level != ?")
        params.append(GovernanceLevel.DEPRECATED)
    where = " AND ".join(clauses)
    order = (
        "CASE WHEN task_signature = ? THEN 1 ELSE 0 END DESC, "
        "CASE governance_level "
        "WHEN 'approved_baseline' THEN 3 WHEN 'human_reviewed' THEN 2 "
        "WHEN 'auto_verified' THEN 1 WHEN 'raw' THEN 0 ELSE -1 END DESC, "
        "created_ts_utc DESC"
    )
    params.append(task_signature)
    rows = conn.execute(
        f"SELECT * FROM failure_case WHERE {where} ORDER BY {order} LIMIT ?",
        (*params, limit),
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM failure_case LIMIT 0").description]
    return [
        _annotate_failure_case_match(dict(zip(cols, row)), task_signature)
        for row in rows
    ]


def _index_failure_case_fts(conn: sqlite3.Connection, row_id: int, f: FailureCase) -> None:
    conn.execute(
        "INSERT INTO failure_case_fts("
        "rowid, task_signature, suite, role_path, files_touched, error_class, "
        "root_cause_label, avoidance_advice"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row_id,
            f.task_signature,
            f.suite,
            f.role_path,
            _jdump(f.files_touched),
            f.error_class,
            f.root_cause_label,
            f.avoidance_advice,
        ),
    )
    conn.commit()


def _failure_case_fts_query(task_signature: str) -> str:
    tokens = _failure_case_match_tokens(task_signature)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens[:8])


def _annotate_failure_case_match(row: dict, task_signature: str) -> dict:
    query_terms = set(_failure_case_match_tokens(task_signature))
    searchable_text = " ".join(
        str(row.get(field) or "")
        for field in (
            "task_signature",
            "suite",
            "role_path",
            "files_touched",
            "error_class",
            "root_cause_label",
            "avoidance_advice",
        )
    )
    row_terms = set(_failure_case_match_tokens(searchable_text))
    matched_terms = sorted(query_terms & row_terms)
    row["_match_type"] = "exact" if row.get("task_signature") == task_signature else "lexical"
    row["_matched_terms"] = matched_terms
    return row


def _failure_case_match_tokens(text: str | None) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_/-]{3,}", text or "")]


def get_working_state(
    conn: sqlite3.Connection, scope: str, key: str, *, owner: str | None = None
) -> dict | None:
    """Return the current (non-superseded) working-state row for (scope, owner, key), or None."""
    row = conn.execute(
        "SELECT * FROM working_state "
        "WHERE scope = ? AND owner IS ? AND key = ? AND superseded = 0 "
        "ORDER BY id DESC LIMIT 1",
        (scope, owner, key),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM working_state LIMIT 0").description]
    return dict(zip(cols, row))


def latest_behavior_signature(
    conn: sqlite3.Connection, archive_member_id: str
) -> dict | None:
    """Most-recent behavior signature for an archive member (for BSV-2 diffing)."""
    row = conn.execute(
        "SELECT * FROM behavior_signature WHERE archive_member_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (archive_member_id,),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM behavior_signature LIMIT 0").description]
    return dict(zip(cols, row))


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts for every harness table (diagnostics / tests)."""
    tables = [
        "harness_metrics", "oracle_adequacy", "behavior_signature",
        "approval_record", "failure_case", "working_state",
    ]
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
