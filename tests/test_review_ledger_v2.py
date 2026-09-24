#!/usr/bin/env python3
"""RC-9 — ``review_ledger.v2`` rubric persistence.

Hermetic (tmp_path SQLite only), NO inference. Covers:
  * fresh DDL carries ``rubric_json`` / ``per_item_grades_json``;
  * v1 -> v2 migration: additive, idempotent, v1 rows still read, no back-fill;
  * a read-only v1 ledger still reads (decoded aliases None);
  * writer canonicalisation + refusal of non-JSON snapshots;
  * reader keeps raw forensic columns and adds decoded aliases;
  * producers: GradeResult mapper, ReviewDecision rubric_ref, events materializer
    (explicit + legacy plan_rubric), corpus field-order arm, plan_rubric emission.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from src.trace import review_ledger as rl
from src.trace.review_ledger import (
    ReviewLedgerRow,
    calibration_summary,
    canonical_json,
    grade_result_to_ledger_row,
    insert_review_ledger_row,
    iter_review_ledger_rows,
    record_review_decision,
    review_decision_to_ledger_row,
    rubric_version_from_ref,
)
from src.trace.store import ensure_review_ledger_schema, ensure_schema

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

# Verbatim review_ledger.v1 DDL (orchestrator origin/main before RC-9).
_V1_DDL = """
CREATE TABLE review_ledger (
  id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, ts TEXT NOT NULL,
  reviewer_model_quant TEXT, grading_model TEXT, rubric_version TEXT,
  corpus_id TEXT, candidate_id TEXT, domain TEXT, decision TEXT, tripwire INTEGER,
  confidence REAL, gold_label TEXT, gold_source TEXT, gold_instrument_version TEXT,
  rationale_cause_match INTEGER, latency_ms REAL, tokens INTEGER,
  family_match_flag INTEGER, era TEXT, event_source_path TEXT, event_id INTEGER,
  schema_version TEXT, created_ts_utc TEXT NOT NULL, UNIQUE(decision_id)
);
INSERT INTO review_ledger (decision_id, ts, rubric_version, decision, gold_label,
  confidence, schema_version, created_ts_utc)
VALUES ('v1-a', '2026-07-01T00:00:00Z', 'r@1.0.0', 'approve', 'fail', 0.9, '1.0.0',
  '2026-07-01T00:00:00Z'),
       ('v1-b', '2026-07-01T00:00:01Z', NULL, 'reject', 'pass', 0.4, '1.0.0',
  '2026-07-01T00:00:01Z');
"""

RUBRIC = {
    "schema_version": "1.0.0",
    "rubric_id": "code-fix",
    "version": "1.2.0",
    "domain": "code",
    "items": [
        {"id": "R1", "text": "Does it compile?", "axis": "runtime", "weight": 3},
        {"id": "R2", "text": "Matches spec?", "axis": "spec-alignment", "weight": 2},
    ],
}
GRADES = [
    {"item": "R1", "axis": "runtime", "weight": 3, "raw_score": 1.0, "binary": 1,
     "graded": True, "note": ""},
    {"item": "R2", "axis": "spec-alignment", "weight": 2, "raw_score": 0.0, "binary": 0,
     "graded": False, "note": ""},
]


def _v1_db(path: Path) -> Path:
    c = sqlite3.connect(str(path))
    c.executescript(_V1_DDL)
    c.commit()
    c.close()
    return path


def _cols(conn: sqlite3.Connection) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(review_ledger)")]


def test_ddl_version_is_v2():
    assert rl.LEDGER_DDL_VERSION == "review_ledger.v2"


def test_fresh_schema_has_v2_columns(tmp_path):
    conn = ensure_schema(tmp_path / "events.sqlite")
    try:
        assert {"rubric_json", "per_item_grades_json"} <= set(_cols(conn))
    finally:
        conn.close()


def test_v1_migration_additive_idempotent_no_backfill(tmp_path):
    path = _v1_db(tmp_path / "v1.sqlite")
    conn = sqlite3.connect(str(path))
    try:
        before = _cols(conn)
        assert "rubric_json" not in before
        ensure_review_ledger_schema(conn)
        ensure_review_ledger_schema(conn)  # idempotent
        after = _cols(conn)
        assert after[: len(before)] == before
        assert after[len(before):] == ["rubric_json", "per_item_grades_json"]
        raw = conn.execute(
            "SELECT decision_id, rubric_json, per_item_grades_json FROM review_ledger ORDER BY id"
        ).fetchall()
        assert raw == [("v1-a", None, None), ("v1-b", None, None)]
        rows = list(iter_review_ledger_rows(conn))
        assert [r["decision_id"] for r in rows] == ["v1-a", "v1-b"]
        assert all(r["rubric"] is None and r["per_item_grades"] is None for r in rows)
        # v1 rows keep feeding the calibration panel unchanged.
        summary = calibration_summary(conn=conn)
        assert summary["n_decisions"] == 2
        assert summary["reviewer_fa_rate"] == 1.0 and summary["reviewer_fr_rate"] == 1.0
        # a v2 write lands next to the untouched v1 rows
        insert_review_ledger_row(
            conn, ReviewLedgerRow(decision_id="v2-a", rubric=RUBRIC, per_item_grades=GRADES)
        )
        by_id = {r["decision_id"]: r for r in iter_review_ledger_rows(conn)}
        assert by_id["v2-a"]["rubric"] == RUBRIC
        assert by_id["v1-a"]["rubric_json"] is None
    finally:
        conn.close()


def test_readonly_v1_ledger_still_reads(tmp_path):
    path = _v1_db(tmp_path / "ro.sqlite")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = list(iter_review_ledger_rows(conn))
        assert len(rows) == 2
        for r in rows:
            assert r["rubric_json"] is None and r["rubric"] is None
            assert r["per_item_grades_json"] is None and r["per_item_grades"] is None
        assert "rubric_json" not in _cols(conn)  # nothing was written
    finally:
        conn.close()


def test_writer_canonicalises_and_reader_decodes(tmp_path):
    conn = ensure_schema(tmp_path / "events.sqlite")
    try:
        insert_review_ledger_row(
            conn, ReviewLedgerRow(decision_id="d1", rubric=RUBRIC, per_item_grades=GRADES)
        )
        raw_rubric, raw_grades = conn.execute(
            "SELECT rubric_json, per_item_grades_json FROM review_ledger"
        ).fetchone()
        assert raw_rubric == json.dumps(RUBRIC, sort_keys=True, separators=(",", ":"))
        assert json.loads(raw_grades) == GRADES
        row = next(iter_review_ledger_rows(conn))
        assert row["rubric_json"] == raw_rubric  # forensic raw kept
        assert row["rubric"] == RUBRIC and row["per_item_grades"] == GRADES
        # legacy writer (no snapshots) still valid -> NULLs
        insert_review_ledger_row(conn, ReviewLedgerRow(decision_id="d2", decision="approve"))
        d2 = [r for r in iter_review_ledger_rows(conn) if r["decision_id"] == "d2"][0]
        assert d2["rubric_json"] is None and d2["per_item_grades"] is None
    finally:
        conn.close()


def test_canonical_json_contract():
    assert canonical_json(None) is None
    # string input is re-canonicalised, so key order does not matter
    assert canonical_json('{"b": 1, "a": [1, 2]}') == '{"a":[1,2],"b":1}'
    assert canonical_json({"t": "é"}) == '{"t":"é"}'
    with pytest.raises(json.JSONDecodeError):
        canonical_json("{not json")
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def test_non_json_snapshot_writes_nothing(tmp_path):
    conn = ensure_schema(tmp_path / "events.sqlite")
    try:
        with pytest.raises(TypeError):
            insert_review_ledger_row(conn, ReviewLedgerRow(decision_id="bad", rubric={"x": object()}))
        assert conn.execute("SELECT COUNT(*) FROM review_ledger").fetchone()[0] == 0
    finally:
        conn.close()


def test_undecodable_stored_blob_keeps_raw(tmp_path):
    conn = ensure_schema(tmp_path / "events.sqlite")
    try:
        conn.execute(
            "INSERT INTO review_ledger (decision_id, ts, created_ts_utc, rubric_json) "
            "VALUES ('x', 't', 't', '{broken')"
        )
        row = next(iter_review_ledger_rows(conn))
        assert row["rubric_json"] == "{broken" and row["rubric"] is None
    finally:
        conn.close()


def test_rubric_version_from_ref():
    assert rubric_version_from_ref("code-fix@1.2.0") == "1.2.0"
    assert rubric_version_from_ref("bare-ref") == "bare-ref"
    assert rubric_version_from_ref("") is None
    assert rubric_version_from_ref(None) is None


def test_grade_result_to_ledger_row_from_real_grade_candidate(tmp_path):
    """``grade_result_to_ledger_row`` is duck-typed (it never imports
    ``src.proactive_delegation`` -- see its docstring) so this exercises the
    mapping against a GradeResult-*shaped* mapping directly, matching exactly
    what ``rubric_review.grade_candidate`` used to produce for this rubric/grader
    pair (one graded item R1, one conservatively-ungraded item R2 -> graded=False,
    binary=0). ``rubric_review.py`` (RD-2's two-turn rubric engine) was archived
    2026-09-24 (TD-21 rubric-engine decision: no runtime caller ever wired it in
    the ~2 months since it landed, and no active handoff names a live-imminent
    one) -- this test's coverage of the LEDGER'S mapping is independent of that
    engine's own code and stays exactly as tested.
    """
    result = {
        "rubric_id": "code-fix",
        "rubric_version": "1.2.0",
        "rubric_ref": "code-fix@1.2.0",
        "S": 0.6,
        "decision": "x",
        "confidence": 0.6,
        "per_item": GRADES,
        "passes": [],
        "k_used": 1,
        "near_edge": False,
        "flakiness": 0.0,
        "rubric": RUBRIC,
        "bands": {"approve_at": 0.85, "reject_at": 0.5, "edge_margin": 0.05, "binarize_at": 0.5},
    }
    row = grade_result_to_ledger_row(
        result, decision_id="g1", reviewer_model_quant="m/q", gold_label="pass"
    )
    assert row.rubric_version == "1.2.0"
    assert row.domain == "code"
    assert row.decision == result["decision"]
    assert row.confidence == result["confidence"]
    assert row.gold_label == "pass"
    conn = ensure_schema(tmp_path / "events.sqlite")
    try:
        insert_review_ledger_row(conn, row)
        stored = next(iter_review_ledger_rows(conn))
    finally:
        conn.close()
    assert stored["rubric"] == RUBRIC
    assert stored["per_item_grades"] == GRADES
    assert [g["item"] for g in stored["per_item_grades"]] == ["R1", "R2"]
    assert stored["per_item_grades"][1]["graded"] is False
    # a plain dict maps identically to a to_dict()-shaped mapping (duck typing)
    assert (
        grade_result_to_ledger_row(result, decision_id="g1").per_item_grades == GRADES
    )
    with pytest.raises(TypeError):
        grade_result_to_ledger_row(42, decision_id="g2")


def test_review_decision_rubric_ref_and_passthrough(tmp_path):
    obj = {
        "decision_id": "rd-1",
        "decision": "approve",
        "confidence": 0.8,
        "blocking": {"tripwire": False},
        "rubric_ref": "code-fix@1.2.0",
    }
    row = review_decision_to_ledger_row(obj)
    assert row.rubric_version == "1.2.0"
    assert row.rubric is None and row.per_item_grades is None
    conn = ensure_schema(tmp_path / "events.sqlite")
    try:
        record_review_decision(obj, conn=conn, rubric=RUBRIC, per_item_grades=GRADES)
        stored = next(iter_review_ledger_rows(conn))
    finally:
        conn.close()
    assert stored["rubric"] == RUBRIC and stored["per_item_grades"] == GRADES


def test_events_materializer_snapshots():
    import reviewer_events_to_ledger as etl

    def ev(i, detail):
        return {"id": i, "ts_utc": "t", "source_path": f"emit://{i}", "role": "architect",
                "status": detail.get("decision"), "detail_json": json.dumps(detail)}

    explicit = etl.event_to_ledger_row(
        ev(1, {"decision": "approve", "rubric": {"rubric_id": "plan_rubric"},
               "per_item_grades": [{"item": "order", "binary": 1}]})
    )
    assert explicit.rubric == {"rubric_id": "plan_rubric"}
    assert explicit.per_item_grades == [{"item": "order", "binary": 1}]

    legacy = {"mode": "plan_rubric", "decision": "approve", "phase_coverage": True,
              "order": False, "executor_alignment": True}
    parsed = etl.event_to_ledger_row(ev(2, {**legacy, "parse_ok": True}))
    assert parsed.rubric is None  # template not recorded by legacy events
    assert parsed.per_item_grades == [
        {"item": "phase_coverage", "binary": 1},
        {"item": "order", "binary": 0},
        {"item": "executor_alignment", "binary": 1},
    ]
    # unparsed emission: axis booleans are defaults, not grades
    assert etl.event_to_ledger_row(ev(3, {**legacy, "parse_ok": False})).per_item_grades is None
    assert etl.event_to_ledger_row(ev(4, legacy)).per_item_grades is None
    # ordinary review event: nothing to persist
    plain = etl.event_to_ledger_row(ev(5, {"decision": "reject"}))
    assert plain.rubric is None and plain.per_item_grades is None


def test_corpus_field_order_arm_carries_rubric_snapshot(tmp_path):
    import reviewer_corpus_ledger_run as crl

    corpus_row = {"row_id": "r1", "task": "T", "candidate": "C", "gold_label": "pass"}
    arm = crl.map_decision_to_ledger_row(
        "rev", corpus_row, {"decision": "approve"}, field_order="reversed"
    )
    assert arm["rubric"]["field_order"] == "reversed"
    assert arm["rubric"]["fields"] == [["CANDIDATE", "candidate"], ["TASK", "task"]]
    assert "per_item_grades" not in arm
    single = crl.map_decision_to_ledger_row("rev", corpus_row, {"decision": "approve"})
    assert "rubric" not in single
    db = tmp_path / "ledger.sqlite"
    assert crl.emit_ledger_sqlite([arm, single], db) == (2, 0)
    conn = sqlite3.connect(str(db))
    try:
        stored = {r["decision_id"]: r for r in iter_review_ledger_rows(conn)}
    finally:
        conn.close()
    assert stored[arm["decision_id"]]["rubric"] == arm["rubric"]
    assert stored[single["decision_id"]]["rubric"] is None


class _Prims:
    def __init__(self, response):
        self.response = response

    def llm_call(self, prompt, role=None, n_tokens=None, **kwargs):
        return self.response


@pytest.mark.parametrize(
    "response, expect_grades",
    [
        (json.dumps({"decision": "approve", "confidence": 0.9, "phase_coverage": True,
                     "order": False, "executor_alignment": True}), [1, 0, 1]),
        ("not json at all", None),
    ],
)
def test_plan_rubric_event_carries_snapshots(response, expect_grades):
    import reviewer_events_to_ledger as etl
    from src.proactive_delegation.review_service import (
        PLAN_RUBRIC_AXES,
        ArchitectReviewService,
    )

    events: list = []
    svc = ArchitectReviewService(_Prims(response), trace_sink=events.append)
    svc.review_plan_rubric("obj", "code", [{"id": "S1", "actor": "coder", "action": "x"}])
    detail = json.loads(events[-1].detail_json)
    assert detail["rubric"]["rubric_id"] == "plan_rubric"
    assert detail["rubric"]["items"] == list(PLAN_RUBRIC_AXES) == list(etl.PLAN_RUBRIC_AXES)
    assert len(detail["rubric"]["template_sha256"]) == 64
    if expect_grades is None:
        assert detail["per_item_grades"] is None
    else:
        assert [g["binary"] for g in detail["per_item_grades"]] == expect_grades
    # the emitted event round-trips through the materializer
    row = etl.event_to_ledger_row(
        {"id": 1, "ts_utc": "t", "source_path": "emit://x", "status": detail["decision"],
         "detail_json": events[-1].detail_json}
    )
    assert row.rubric == detail["rubric"]
    assert row.per_item_grades == detail["per_item_grades"]
