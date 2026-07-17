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
    decision_correct,
    insert_review_ledger_row,
    insert_review_ledger_rows,
    is_false_accept,
    is_false_reject,
    iter_review_ledger_rows,
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
