"""Tests for the Architect->Reviewer trace control plane (H1: TM-1/2/4/5).

Covers:
  * TM-2 — review-plane EventCategory / EventSource enum values exist.
  * TM-4 — live push interface: emit() write-through, synthetic-source
           collision-avoidance + idempotency, and the ReviewTracingProcessor
           6-method shape mapping trace_id->session_id / group_id->trial_id.
  * TM-5 — decision_chain replay reconstructs an ordered task->plan->review->
           gate->outcome chain by session_id / trial_id.
  * TM-1 — rotated-shard discovery (shard_paths) picks primary + _<n> shards
           and excludes .bak-* backups.

All DB-touching tests use an isolated tmp_path SQLite file; none touch the
shared materialized data/trace/events.sqlite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.trace import ensure_schema, upsert_events
from src.trace.store import Event, EventCategory, EventSource
from src.trace import ingest_autopilot
from src.trace.emit import (
    ReviewTracingProcessor,
    Span,
    Trace,
    emit,
    is_synthetic_source_path,
    synthetic_source_path,
)
from src.trace.query import decision_chain, query, stats


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "events.sqlite"
    conn = ensure_schema(p)
    conn.close()
    return p


# --------------------------------------------------------------------------
# TM-2 — review-plane categories / source
# --------------------------------------------------------------------------


def test_tm2_review_categories_exist():
    assert EventCategory.REVIEW_DECISION == "review_decision"
    assert EventCategory.CANDIDATE_PACKAGE == "candidate_package"
    assert EventCategory.VERIFICATION_REPORT == "verification_report"
    assert EventCategory.REVIEW_ESCALATION == "review_escalation"
    assert EventCategory.PLAN_REMINDER == "plan_reminder"
    assert EventSource.REVIEW_PLANE == "review_plane"


# --------------------------------------------------------------------------
# TM-4 — synthetic source convention + emit() write-through
# --------------------------------------------------------------------------


def test_tm4_synthetic_path_never_collides_with_file_path():
    sp = synthetic_source_path("review_plane", "span", "abc123")
    assert sp == "emit://review_plane/span/abc123"
    assert is_synthetic_source_path(sp)
    # Real absolute file paths (as produced by ingesters) are NOT synthetic.
    assert not is_synthetic_source_path("/workspace/logs/agent_audit.log")
    assert not is_synthetic_source_path("/mnt/raid0/llm/x/autopilot_journal.jsonl")
    assert not is_synthetic_source_path(None)


def test_tm4_emit_writethrough_and_idempotent(db_path: Path):
    ev = Event(
        ts_utc="2026-07-17T10:00:00+00:00",
        source=EventSource.REVIEW_PLANE,
        source_path=synthetic_source_path("review_plane", "decision", "d1"),
        source_line=0,
        session_id="sess-A",
        trial_id=7,
        category=EventCategory.REVIEW_DECISION,
        status="approve",
        summary="approve candidate",
        detail_json='{"confidence": 0.9}',
    )
    ins, skp = emit(ev, db_path=db_path)
    assert (ins, skp) == (1, 0)

    # Re-emit identical logical row -> idempotent no-op (append-only dedup).
    ins2, skp2 = emit(ev, db_path=db_path)
    assert (ins2, skp2) == (0, 1)

    rows = query(db_path=db_path, session_id="sess-A")
    assert len(rows) == 1
    assert rows[0]["category"] == "review_decision"
    assert rows[0]["source"] == "review_plane"
    assert rows[0]["source_path"].startswith("emit://")


def test_tm4_emit_content_addressed_when_no_path(db_path: Path):
    # No source_path supplied -> emit assigns a content-addressed synthetic key.
    ev = Event(
        ts_utc="2026-07-17T10:05:00+00:00",
        source=EventSource.REVIEW_PLANE,
        source_path="",
        session_id="sess-B",
        category=EventCategory.PLAN_REMINDER,
        summary="reminder body",
    )
    ins, _ = emit(ev, db_path=db_path)
    assert ins == 1
    stored = query(db_path=db_path, session_id="sess-B")[0]
    assert is_synthetic_source_path(stored["source_path"])
    # Byte-identical content collapses to the same row.
    ev2 = Event(
        ts_utc="2026-07-17T10:05:00+00:00",
        source=EventSource.REVIEW_PLANE,
        source_path="",
        session_id="sess-B",
        category=EventCategory.PLAN_REMINDER,
        summary="reminder body",
    )
    ins2, skp2 = emit(ev2, db_path=db_path)
    assert (ins2, skp2) == (0, 1)


def test_tm4_emit_does_not_collide_with_file_ingested_rows(db_path: Path):
    # A file-ingested row and a live-emitted row can share nothing.
    conn = ensure_schema(db_path)
    file_row = Event(
        ts_utc="2026-07-17T09:00:00+00:00",
        source=EventSource.AGENT_AUDIT,
        source_path="/workspace/logs/agent_audit.log",
        source_line=42,
        session_id="sess-C",
        category=EventCategory.DECISION,
        summary="file row",
    )
    upsert_events(conn, [file_row])
    conn.close()

    live = Event(
        ts_utc="2026-07-17T09:00:01+00:00",
        source=EventSource.REVIEW_PLANE,
        source_path=synthetic_source_path("review_plane", "span", "s-42"),
        session_id="sess-C",
        category=EventCategory.REVIEW_DECISION,
        summary="live row",
    )
    emit(live, db_path=db_path)

    rows = query(db_path=db_path, session_id="sess-C", limit=50)
    assert {r["source"] for r in rows} == {"agent_audit", "review_plane"}
    assert len(rows) == 2


# --------------------------------------------------------------------------
# TM-4 — ReviewTracingProcessor 6-method shape
# --------------------------------------------------------------------------


def test_tm4_tracing_processor_six_methods_and_mapping(db_path: Path):
    proc = ReviewTracingProcessor(db_path=db_path)
    trace = Trace(trace_id="sess-D", group_id="55", name="review turn")

    # 1) trace start  2) span start  3) span end  4) trace end
    proc.on_trace_start(trace)
    span = Span(
        span_id="span-1",
        trace_id="sess-D",
        started_at="2026-07-17T11:00:00+00:00",
        ended_at="2026-07-17T11:00:00.250000+00:00",
        span_data={
            "category": EventCategory.REVIEW_DECISION,
            "summary": "reject: missing evidence",
            "role": "reviewer",
            "status": "reject",
        },
    )
    proc.on_span_start(span)
    proc.on_span_end(span)
    proc.on_trace_end(trace)
    # 5) force_flush  6) shutdown
    proc.force_flush()
    proc.shutdown()

    rows = query(db_path=db_path, session_id="sess-D", limit=50)
    cats = {r["category"] for r in rows}
    assert "trace_start" in cats
    assert "trace_end" in cats
    assert "review_decision" in cats

    # group_id "55" mapped to integer trial_id; span inherits it.
    decision = [r for r in rows if r["category"] == "review_decision"][0]
    assert decision["trial_id"] == 55
    assert decision["session_id"] == "sess-D"  # trace_id -> session_id
    assert decision["role"] == "reviewer"
    assert decision["status"] == "reject"
    # latency computed from span start/end (~250 ms).
    assert '"latency_ms": 250' in decision["detail_json"]


def test_tm4_processor_idempotent_on_span_replay(db_path: Path):
    # Re-processing the same span (e.g. on resume) must not duplicate rows.
    proc = ReviewTracingProcessor(db_path=db_path)
    trace = Trace(trace_id="sess-E", group_id=3)
    proc.on_trace_start(trace)
    span = Span(span_id="dup-span", trace_id="sess-E",
                span_data={"category": EventCategory.VERIFICATION_REPORT, "summary": "gate pass"})
    proc.on_span_end(span)
    proc.on_span_end(span)  # replay
    proc.shutdown()

    rows = query(db_path=db_path, session_id="sess-E", category="verification_report")
    assert len(rows) == 1


# --------------------------------------------------------------------------
# TM-5 — decision-chain replay
# --------------------------------------------------------------------------


def _seed_chain(db_path: Path, session_id: str, trial_id: int) -> None:
    events = [
        ("2026-07-17T12:00:00+00:00", EventCategory.TASK_START, "task begins"),
        ("2026-07-17T12:00:01+00:00", EventCategory.CANDIDATE_PACKAGE, "plan/candidate"),
        ("2026-07-17T12:00:02+00:00", EventCategory.PLAN_REMINDER, "remember constraints"),
        ("2026-07-17T12:00:03+00:00", EventCategory.REVIEW_DECISION, "reviewer verdict"),
        ("2026-07-17T12:00:04+00:00", EventCategory.VERIFICATION_REPORT, "gate results"),
        ("2026-07-17T12:00:05+00:00", EventCategory.REVIEW_ESCALATION, "escalate"),
        ("2026-07-17T12:00:06+00:00", EventCategory.TASK_END, "outcome"),
    ]
    for i, (ts, cat, summary) in enumerate(events):
        emit(
            Event(
                ts_utc=ts,
                source=EventSource.REVIEW_PLANE,
                source_path=synthetic_source_path("review_plane", session_id, str(i)),
                session_id=session_id,
                trial_id=trial_id,
                category=cat,
                summary=summary,
            ),
            db_path=db_path,
        )


def test_tm5_decision_chain_ordered_and_phased(db_path: Path):
    _seed_chain(db_path, "sess-F", 101)
    # A second, unrelated session must not bleed in.
    _seed_chain(db_path, "sess-G", 202)

    chain = decision_chain(db_path=db_path, session_id="sess-F")
    cats = [r["category"] for r in chain["chain"]]
    # Ordered by timestamp: task -> plan -> reminder -> review -> gate -> escalation -> outcome
    assert cats == [
        "task_start",
        "candidate_package",
        "plan_reminder",
        "review_decision",
        "verification_report",
        "review_escalation",
        "task_end",
    ]
    assert chain["counts"]["chain"] == 7
    assert chain["by_phase"]["task"][0]["summary"] == "task begins"
    assert chain["by_phase"]["review"][0]["summary"] == "reviewer verdict"
    assert chain["by_phase"]["gate"][0]["category"] == "verification_report"
    assert chain["by_phase"]["outcome"][0]["summary"] == "outcome"
    # isolation: none of sess-G's rows appear.
    assert all(r["session_id"] == "sess-F" for r in chain["chain"])


def test_tm5_decision_chain_by_trial_id(db_path: Path):
    _seed_chain(db_path, "sess-H", 303)
    chain = decision_chain(db_path=db_path, trial_id=303)
    assert chain["counts"]["chain"] == 7
    assert all(r["trial_id"] == 303 for r in chain["chain"])


def test_tm5_decision_chain_requires_a_selector(db_path: Path):
    with pytest.raises(ValueError):
        decision_chain(db_path=db_path)


# --------------------------------------------------------------------------
# TM-1 — rotated-shard discovery
# --------------------------------------------------------------------------


def test_tm1_shard_paths_reads_primary_plus_numbered(tmp_path: Path):
    primary = tmp_path / "autopilot_journal.jsonl"
    primary.write_text("{}\n")
    (tmp_path / "autopilot_journal_1.jsonl").write_text("{}\n")
    (tmp_path / "autopilot_journal_2.jsonl").write_text("{}\n")
    # backups + unrelated files must be excluded.
    (tmp_path / "autopilot_journal.jsonl.bak-20260509").write_text("x")
    (tmp_path / "autopilot_journal.jsonl.run3-poisoned").write_text("x")
    (tmp_path / "autopilot_journal_x.jsonl").write_text("{}\n")  # non-numeric

    shards = [p.name for p in ingest_autopilot.shard_paths(primary)]
    assert shards == [
        "autopilot_journal.jsonl",
        "autopilot_journal_1.jsonl",
        "autopilot_journal_2.jsonl",
    ]


def test_tm1_shard_paths_missing_primary_still_finds_shards(tmp_path: Path):
    # Primary absent but a rotated shard present -> shard still ingested.
    (tmp_path / "autopilot_journal_1.jsonl").write_text("{}\n")
    shards = [p.name for p in ingest_autopilot.shard_paths(tmp_path / "autopilot_journal.jsonl")]
    assert shards == ["autopilot_journal_1.jsonl"]


def test_tm1_materialized_db_smoke():
    # If the shared DB has been materialized, sanity-check it is non-empty and
    # exposes the expected offline sources. Skips cleanly on a fresh checkout.
    from src.trace.store import DEFAULT_DB_PATH

    if not Path(DEFAULT_DB_PATH).exists():
        pytest.skip("events.sqlite not materialized on this host")
    s = stats(DEFAULT_DB_PATH)
    assert s["total"] > 0
    assert "agent_audit" in s["by_source"]
