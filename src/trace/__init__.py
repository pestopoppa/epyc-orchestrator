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
from src.trace.harness_schema import (
    SCHEMA_VERSION,
    GovernanceLevel,
    WorkingStateScope,
    HarnessMetrics,
    OracleAdequacy,
    BehaviorSignature,
    ApprovalRecord,
    FailureCase,
    WorkingState,
    ensure_harness_schema,
    insert_harness_metrics,
    insert_oracle_adequacy,
    insert_behavior_signature,
    insert_approval_record,
    insert_failure_case,
    set_working_state,
    find_failure_cases,
    get_working_state,
    latest_behavior_signature,
    table_counts,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "Event",
    "EventCategory",
    "EventSource",
    "ensure_schema",
    "upsert_events",
    "query",
    # shared harness/trace schema (intake-607 cluster)
    "SCHEMA_VERSION",
    "GovernanceLevel",
    "WorkingStateScope",
    "HarnessMetrics",
    "OracleAdequacy",
    "BehaviorSignature",
    "ApprovalRecord",
    "FailureCase",
    "WorkingState",
    "ensure_harness_schema",
    "insert_harness_metrics",
    "insert_oracle_adequacy",
    "insert_behavior_signature",
    "insert_approval_record",
    "insert_failure_case",
    "set_working_state",
    "find_failure_cases",
    "get_working_state",
    "latest_behavior_signature",
    "table_counts",
]
