#!/usr/bin/env python3
"""Tests for the reviewer FA/FR calibration ledger (H4 RC-1/RC-5/RC-7 + RA-10).

Hermetic: every DB-touching test uses an isolated tmp_path SQLite file; none
touch the shared materialized data/trace/events.sqlite. NO inference.

Covers:
  * RC-1  — additive review_ledger DDL + append-only writer (dedup on decision_id)
            + provenance columns + RA-10 schema_version stamp; additive migration
            does not disturb existing `event` rows.
  * FA/FR — the shared polarity classifiers (false-accept / false-reject).
  * RC-5  — symmetric FA-tolerance AND FR-tolerance sequential demotion monitor.
  * RC-7  — to_question_ledger_row evidence-plane adapter stub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.trace.store import Event, ensure_schema, ensure_review_ledger_schema, upsert_events
from src.trace import review_ledger as rl
from src.trace.review_ledger import (
    REVIEW_DECISION_SCHEMA_VERSION,
    ReviewerDemotionMonitor,
    ReviewerToleranceConfig,
    ReviewLedgerRow,
    calibration_summary,
    decision_correct,
    insert_review_ledger_row,
    insert_review_ledger_rows,
    is_false_accept,
    is_false_reject,
    iter_review_ledger_rows,
    record_review_decision,
    review_decision_to_ledger_row,
    review_ledger_count,
    to_question_ledger_row,
)


@pytest.fixture
def conn(tmp_path: Path):
    c = ensure_schema(tmp_path / "events.sqlite")
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# RC-1 — DDL + append-only writer + RA-10
# --------------------------------------------------------------------------- #
def test_rc1_ddl_created_by_ensure_schema(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(review_ledger)").fetchall()}
    # exactly the handoff's column list + provenance links + RA-10 + bookkeeping
    for expected in (
        "decision_id", "ts", "reviewer_model_quant", "grading_model", "rubric_version",
        "corpus_id", "candidate_id", "domain", "decision", "tripwire", "confidence",
        "gold_label", "gold_source", "gold_instrument_version", "rationale_cause_match",
        "latency_ms", "tokens", "family_match_flag", "era",
        "event_source_path", "event_id", "schema_version",
    ):
        assert expected in cols, f"missing column {expected}"


def test_rc1_append_only_writer_and_dedup(conn):
    row = ReviewLedgerRow(
        decision_id="d1",
        reviewer_model_quant="GLM-5.2/UD-IQ2_M",
        grading_model="Qwen3-Coder-30B",
        rubric_version="v3",
        corpus_id="nearmiss-v1",
        candidate_id="cand-1",
        domain="code",
        decision="approve",
        tripwire=False,
        confidence=0.8,
        gold_label="fail",
        gold_source="gate_runner+evaltower",
        gold_instrument_version="v2",
        rationale_cause_match=True,
        latency_ms=1234.0,
        tokens=456,
        family_match_flag=False,
        era="reviewer_plane/iv-1",
        event_source_path="emit://review_plane/span/abc",
        event_id=99,
    )
    ins, skip = insert_review_ledger_row(conn, row)
    assert (ins, skip) == (1, 0)
    # re-insert same decision_id is an idempotent no-op
    ins2, skip2 = insert_review_ledger_row(conn, row)
    assert (ins2, skip2) == (0, 1)
    assert review_ledger_count(conn) == 1

    got = next(iter_review_ledger_rows(conn))
    assert got["decision_id"] == "d1"
    assert got["decision"] == "approve"
    assert got["tripwire"] == 0  # bool -> INTEGER
    assert got["rationale_cause_match"] == 1
    assert got["event_source_path"] == "emit://review_plane/span/abc"
    assert got["event_id"] == 99
    # RA-10: schema_version stamped through by default
    assert got["schema_version"] == REVIEW_DECISION_SCHEMA_VERSION


def test_ra10_schema_version_default_and_override(conn):
    insert_review_ledger_row(conn, ReviewLedgerRow(decision_id="a"))
    insert_review_ledger_row(conn, ReviewLedgerRow(decision_id="b", schema_version="2.1.0"))
    rows = {r["decision_id"]: r for r in iter_review_ledger_rows(conn)}
    assert rows["a"]["schema_version"] == REVIEW_DECISION_SCHEMA_VERSION
    assert rows["b"]["schema_version"] == "2.1.0"


def test_rc1_additive_migration_does_not_disturb_events(tmp_path):
    """Adding/using the ledger must not touch existing `event` rows."""
    db = tmp_path / "events.sqlite"
    conn = ensure_schema(db)
    try:
        upsert_events(
            conn,
            [Event(ts_utc="2026-07-17T00:00:00+00:00", source="s", source_path="/f", source_line=1)],
        )
        before = conn.execute("SELECT COUNT(*) FROM event").fetchone()[0]
        # re-applying the ledger schema is idempotent + additive
        ensure_review_ledger_schema(conn)
        insert_review_ledger_rows(
            conn, [ReviewLedgerRow(decision_id=f"d{i}", decision="approve") for i in range(5)]
        )
        after = conn.execute("SELECT COUNT(*) FROM event").fetchone()[0]
        assert before == after == 1
        assert review_ledger_count(conn) == 5
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# FA / FR polarity classifiers
# --------------------------------------------------------------------------- #
def test_fa_fr_classifiers():
    fa = {"decision": "approve", "gold_label": "fail"}
    fr = {"decision": "reject", "gold_label": "accept"}
    ta = {"decision": "approve", "gold_label": "pass"}
    tr = {"decision": "reject", "gold_label": "fail"}
    assert is_false_accept(fa) and not is_false_reject(fa)
    assert is_false_reject(fr) and not is_false_accept(fr)
    assert not is_false_accept(ta) and not is_false_reject(ta)
    assert decision_correct(ta) is True
    assert decision_correct(tr) is True
    assert decision_correct(fa) is False
    # non-terminal / ungolded -> None
    assert decision_correct({"decision": "escalate", "gold_label": "fail"}) is None
    assert decision_correct({"decision": "approve", "gold_label": None}) is None


# --------------------------------------------------------------------------- #
# RC-5 — symmetric FA / FR sequential demotion monitor
# --------------------------------------------------------------------------- #
def test_rc5_fa_breach_demotes():
    # A reviewer that false-accepts every actually-bad candidate blows the FA
    # e-process past confirm_e -> demote-to-shadow on the FA axis.
    rows = [
        {"decision_id": f"b{i}", "decision": "approve", "gold_label": "fail"}
        for i in range(25)
    ]
    mon = ReviewerDemotionMonitor()
    verdicts = list(mon.run(rows))
    assert mon.breached is True
    assert mon.breach_axis == "fa"
    assert any(v.breached for v in verdicts)
    summary = mon.summary()
    assert summary["fa"]["state"] == "confirmed"
    assert summary["thresholds_are_placeholders"] is True


def test_rc5_fr_breach_demotes():
    # A reviewer that false-rejects every actually-good candidate blows the FR
    # e-process (independent axis) -> demote-to-shadow on the FR axis.
    rows = [
        {"decision_id": f"g{i}", "decision": "reject", "gold_label": "accept"}
        for i in range(30)
    ]
    mon = ReviewerDemotionMonitor()
    list(mon.run(rows))
    assert mon.breached is True
    assert mon.breach_axis == "fr"
    assert mon.summary()["fr"]["state"] == "confirmed"


def test_rc5_clean_reviewer_does_not_breach():
    # Well-calibrated: rejects bad, accepts good -> neither axis breaches.
    rows = []
    for i in range(40):
        if i % 2 == 0:
            rows.append({"decision_id": f"c{i}", "decision": "reject", "gold_label": "fail"})
        else:
            rows.append({"decision_id": f"c{i}", "decision": "approve", "gold_label": "pass"})
    mon = ReviewerDemotionMonitor()
    list(mon.run(rows))
    assert mon.breached is False
    assert mon.breach_axis is None


def test_rc5_tolerance_config_validation():
    with pytest.raises(ValueError):
        ReviewerToleranceConfig(fa_tolerance=0.0)
    with pytest.raises(ValueError):
        ReviewerToleranceConfig(fr_tolerance=1.0)
    cfg = ReviewerToleranceConfig(fa_tolerance=0.02, fr_tolerance=0.30)
    # FR tolerance is set above FA tolerance (overcorrection prior).
    assert cfg.fr_tolerance > cfg.fa_tolerance


# --------------------------------------------------------------------------- #
# RC-7 — evidence-plane per-question-ledger adapter
# --------------------------------------------------------------------------- #
def test_rc7_to_question_ledger_row():
    row = {
        "decision_id": "d-42",
        "domain": "code",
        "decision": "approve",
        "gold_label": "pass",
        "confidence": 0.9,
        "latency_ms": 500.0,
        "tokens": 128,
    }
    q = to_question_ledger_row(row)
    assert q["qid"] == "d-42"          # decision ≈ question
    assert q["suite"] == "code"
    assert q["correct"] is True        # verdict matched gold
    assert q["confidence"] == 0.9
    assert q["tokens_generated"] == 128
    assert q["_adapter"].endswith("stub")
    # ungolded / non-terminal -> correct is None (evidence-plane partial-row)
    assert to_question_ledger_row({"decision_id": "x", "decision": "escalate"})["correct"] is None


# --------------------------------------------------------------------------- #
# B1 — calibration_summary (digest-facing FA/FR summary)
# --------------------------------------------------------------------------- #
def _seed_calibration_rows(conn) -> None:
    """4 actually-bad (1 FA) + 5 actually-good (2 FR) => FA 0.25, FR 0.4, ratio 0.625."""
    rows = []
    i = 0
    # actually-bad: 1 false-accept (approve), 3 true-reject
    rows.append(("approve", "fail")); 
    for _ in range(3):
        rows.append(("reject", "fail"))
    # actually-good: 2 false-reject, 3 true-accept
    for _ in range(2):
        rows.append(("reject", "pass"))
    for _ in range(3):
        rows.append(("approve", "pass"))
    for decision, gold in rows:
        insert_review_ledger_row(
            conn,
            ReviewLedgerRow(
                decision_id=f"cs{i}", candidate_id=f"c{i}",
                decision=decision, gold_label=gold, latency_ms=100.0,
            ),
        )
        i += 1


def test_b1_calibration_summary_exact_fa_fr(conn):
    _seed_calibration_rows(conn)
    s = calibration_summary(conn=conn)
    assert s["n_decisions"] == 9
    assert s["reviewer_fa_rate"] == 0.25
    assert s["reviewer_fr_rate"] == 0.4
    assert abs(s["reviewer_fa_fr_ratio"] - 0.625) < 1e-9
    assert s["review_decision_latency_ms"] == 100.0


def test_b1_calibration_summary_empty_conn_returns_empty(conn):
    # schema present but zero rows -> {} (the digest's "no data yet" signal).
    assert calibration_summary(conn=conn) == {}


def test_b1_calibration_summary_missing_db_returns_empty(tmp_path):
    assert calibration_summary(db_path=tmp_path / "does-not-exist.sqlite") == {}


def test_b1_calibration_summary_db_path_roundtrip(tmp_path):
    from src.trace.store import ensure_schema

    db = tmp_path / "review_ledger.sqlite"
    c = ensure_schema(db)
    try:
        _seed_calibration_rows(c)
    finally:
        c.close()
    s = calibration_summary(db_path=db)
    assert s["n_decisions"] == 9 and s["reviewer_fa_rate"] == 0.25


def test_b1_calibration_summary_resolves_in_digest_getattr_loop():
    # digest.py iterates ("calibration_summary","summarize","summary","recent_summary")
    # and calls the first callable module attribute -> must resolve to our function.
    from src.trace import review_ledger as ledger

    fn = None
    for attr in ("calibration_summary", "summarize", "summary", "recent_summary"):
        cand = getattr(ledger, attr, None)
        if callable(cand):
            fn = cand
            break
    assert fn is ledger.calibration_summary


# --------------------------------------------------------------------------- #
# B4 — record_review_decision convenience writer (AP-6 seam)
# --------------------------------------------------------------------------- #
def _review_decision_obj(**over):
    obj = {
        "decision_id": "dec-1",
        "decision": "approve",
        "confidence": 0.9,
        "blocking": {"tripwire": False},
        "subtask_id": "cand-1",
        "reviewed_at": "2026-07-16T10:00:00+00:00",
        "telemetry": {"tokens_out": 42, "wall_ms": 123.0},
        "provenance": {"model": "GLM-5.2", "quant": "UD-IQ2_M", "instrument_era": "era-x"},
    }
    obj.update(over)
    return obj


def test_b4_record_review_decision_roundtrip(conn):
    inserted, skipped = record_review_decision(
        _review_decision_obj(), source="codex", role="architect_general", conn=conn
    )
    assert (inserted, skipped) == (1, 0)
    got = list(iter_review_ledger_rows(conn))
    assert len(got) == 1
    row = got[0]
    assert row["decision_id"] == "dec-1"
    assert row["decision"] == "approve"
    assert row["confidence"] == 0.9
    assert row["tripwire"] == 0  # False -> 0/1/NULL tri-state
    assert row["candidate_id"] == "cand-1"
    assert row["reviewer_model_quant"] == "GLM-5.2/UD-IQ2_M"
    assert row["latency_ms"] == 123.0
    assert row["tokens"] == 42
    assert row["era"] == "era-x"
    assert row["ts"] == "2026-07-16T10:00:00+00:00"


def test_b4_record_review_decision_passive_noop_without_target():
    # The planner seam calls record_review_decision(obj, source=…, role=…) with no
    # conn/db_path: it must be a pure no-op (zero live write) that returns None.
    assert record_review_decision(_review_decision_obj(), source="codex", role="r") is None


def test_b4_record_review_decision_db_path(tmp_path):
    db = tmp_path / "sink" / "review_ledger.sqlite"  # parent auto-created
    assert record_review_decision(_review_decision_obj(decision_id="dec-2"), db_path=db) == (1, 0)
    from src.trace.store import ensure_schema

    c = ensure_schema(db)
    try:
        rows = list(iter_review_ledger_rows(c))
    finally:
        c.close()
    assert [r["decision_id"] for r in rows] == ["dec-2"]


def test_b4_record_review_decision_idempotent(conn):
    obj = _review_decision_obj(decision_id="dup")
    assert record_review_decision(obj, conn=conn) == (1, 0)
    assert record_review_decision(obj, conn=conn) == (0, 1)  # INSERT OR IGNORE dedup


def test_b4_synthetic_decision_id_when_absent(conn):
    obj = _review_decision_obj()
    obj.pop("decision_id")
    row = review_decision_to_ledger_row(obj, source="codex", role="r")
    assert row.decision_id.startswith("revdec-")
    # deterministic: same obj -> same id (idempotency)
    row2 = review_decision_to_ledger_row(obj, source="codex", role="r")
    assert row.decision_id == row2.decision_id
