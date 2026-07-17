#!/usr/bin/env python3
"""Fixture tests for scripts/analysis/reviewer_events_to_ledger.py (Mechanism A).

INFERENCE-FREE. We seed a temp ``events.sqlite`` with a handful of synthetic
REVIEW_DECISION events written through the REAL emit path (``src/trace/emit.py``
-> ``event`` table), matching the exact detail schema the reviewer shadow plane
emits (``review_service.review`` L479-499: ``mode/subtask_id/decision/score/
confidence/tripwire/latency_ms/tokens{tokens_out,chars_out}/assigned_role``).

We then exercise:
  1. the pure event->ledger-row mapper (exact mapped fields; NULL gold pre-join);
  2. the optional --corpus gold join by candidate_id;
  3. the full round-trip: materialize -> decisions.jsonl AND review_ledger.sqlite,
     feed each to the REAL ``reviewer_calibration_report.py`` and assert
     FA / FR / FA-FR-ratio compute to the hand-computed values;
  4. idempotency (re-run into the same ledger DB inserts 0);
  5. the ZERO-EVENTS path: exit 0 with the loud "0 events materialized" message.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_ANALYSIS = _REPO_ROOT / "scripts" / "analysis"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mat = _load("reviewer_events_to_ledger", _ANALYSIS / "reviewer_events_to_ledger.py")
report = _load("reviewer_calibration_report", _ANALYSIS / "reviewer_calibration_report.py")

from src.trace.store import (  # noqa: E402
    Event,
    EventCategory,
    EventSource,
    detail_to_json,
)
from src.trace.emit import emit  # noqa: E402

REVIEWER_ROLE = "architect_general"

# subtask_id -> (decision, gold_label, confidence, latency_ms, tokens_out)
#   cand-A approve + gold fail(bad)  -> FALSE ACCEPT
#   cand-B reject  + gold pass(good) -> FALSE REJECT
#   cand-C approve + gold pass(good) -> correct accept
#   cand-D reject  + gold fail(bad)  -> correct reject
# => over 2 actually-bad: FA=1 (rate 0.5); over 2 actually-good: FR=1 (rate 0.5); ratio 1.0
_CASES = {
    "cand-A": ("approve", "fail", 0.9, 120.0, 50),
    "cand-B": ("reject", "pass", 0.8, 90.0, 40),
    "cand-C": ("approve", "pass", 0.7, 110.0, 45),
    "cand-D": ("reject", "fail", 0.6, 100.0, 30),
}


def _emit_review_decision(db_path, subtask_id, decision, confidence, latency_ms, tokens_out, ts):
    """Write ONE REVIEW_DECISION event exactly as review_service._emit_review_event does."""
    detail = {
        "mode": "review",
        "subtask_id": subtask_id,
        "decision": decision,
        "score": 0.5,
        "confidence": confidence,
        "tripwire": False,
        "quick_mode": False,
        "parse_ok": True,
        "latency_ms": latency_ms,
        "tokens": {"tokens_out": tokens_out, "chars_out": tokens_out * 4},
        "assigned_role": "verifier",  # RD-7 tag added by _emit_review_event
    }
    ev = Event(
        ts_utc=ts,
        source=EventSource.REVIEW_PLANE,
        source_path="",  # emit() assigns content-addressed synthetic path
        source_line=None,
        session_id="sess-1",
        trial_id=7,
        role=REVIEWER_ROLE,
        category=EventCategory.REVIEW_DECISION,
        status=decision,
        summary=f"review {subtask_id}: {decision}",
        detail_json=detail_to_json(detail),
    )
    emit(ev, db_path=db_path)


@pytest.fixture()
def events_db(tmp_path):
    """A temp events.sqlite seeded with the 4 review decisions + one decoy non-review event."""
    db = tmp_path / "events.sqlite"
    for i, (subtask_id, (decision, _gold, conf, lat, toks)) in enumerate(_CASES.items()):
        _emit_review_decision(
            db, subtask_id, decision, conf, lat, toks,
            ts=f"2026-07-16T10:00:0{i}+00:00",
        )
    # Decoy: a non-REVIEW_DECISION event that must NOT be materialized.
    emit(
        Event(
            ts_utc="2026-07-16T10:00:09+00:00",
            source=EventSource.REVIEW_PLANE,
            source_path="",
            source_line=None,
            session_id="sess-1",
            trial_id=7,
            category=EventCategory.TASK_START,
            status="ok",
            summary="task start (decoy)",
            detail_json=detail_to_json({"mode": "task"}),
        ),
        db_path=db,
    )
    return db


@pytest.fixture()
def corpus_jsonl(tmp_path):
    """A tiny near-miss corpus keyed by row_id == candidate_id (subtask_id)."""
    path = tmp_path / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for subtask_id, (_dec, gold, *_r) in _CASES.items():
            fh.write(json.dumps({
                "row_id": subtask_id,
                "gold_label": gold,
                "gold_source": "multi_oracle",
                "gold_instrument_version": "nearmiss-v1",
                "domain": "code",
                "corpus_id": "nearmiss-v1",
            }) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 1. Reading + pure mapping
# --------------------------------------------------------------------------- #
def test_reads_only_review_decision_events(events_db):
    events = mat.read_review_events(events_db)
    assert len(events) == 4  # the decoy task_start is filtered out
    assert all(e["category"] == EventCategory.REVIEW_DECISION for e in events)
    # ascending, stable order by ts
    ids = [e["id"] for e in events]
    assert ids == sorted(ids)


def test_event_to_ledger_row_maps_fields(events_db):
    events = mat.read_review_events(events_db)
    by_cand = {}
    for ev in events:
        row = mat.event_to_ledger_row(ev)
        by_cand[row.candidate_id] = row

    a = by_cand["cand-A"]
    assert a.decision_id.startswith("revevt-")
    assert a.decision == "approve"
    assert a.latency_ms == 120.0
    assert a.confidence == 0.9
    assert a.tokens == 50 and isinstance(a.tokens, int)  # flattened from {"tokens_out":50}
    assert a.tripwire is False
    assert a.reviewer_model_quant == REVIEWER_ROLE  # event.role best-effort
    assert a.event_source_path.startswith("emit://")
    assert a.event_id is not None
    # events carry no gold pre-join
    assert a.gold_label is None and a.gold_source is None
    assert a.domain is None and a.corpus_id is None


def test_decision_id_stable_and_unique(events_db):
    events = mat.read_review_events(events_db)
    ids1 = [mat.event_to_ledger_row(e).decision_id for e in events]
    ids2 = [mat.event_to_ledger_row(e).decision_id for e in mat.read_review_events(events_db)]
    assert ids1 == ids2  # stable across re-reads
    assert len(set(ids1)) == len(ids1)  # unique per event


def test_reviewer_model_quant_stamp_override(events_db):
    ev = mat.read_review_events(events_db)[0]
    row = mat.event_to_ledger_row(ev, reviewer_model_quant="GLM-5.2/UD-IQ2_M")
    assert row.reviewer_model_quant == "GLM-5.2/UD-IQ2_M"


# --------------------------------------------------------------------------- #
# 2. Corpus gold join
# --------------------------------------------------------------------------- #
def test_corpus_join_fills_gold(events_db, corpus_jsonl):
    events = mat.read_review_events(events_db)
    corpus = mat._load_corpus(corpus_jsonl)
    rows = mat.materialize_rows(events, corpus=corpus)
    by_cand = {r.candidate_id: r for r in rows}
    assert by_cand["cand-A"].gold_label == "fail"
    assert by_cand["cand-B"].gold_label == "pass"
    assert by_cand["cand-A"].domain == "code"
    assert by_cand["cand-A"].corpus_id == "nearmiss-v1"
    assert by_cand["cand-A"].gold_source == "multi_oracle"


# --------------------------------------------------------------------------- #
# 3. Full round-trip through the REAL calibration report
# --------------------------------------------------------------------------- #
def _assert_fa_fr(rep):
    overall = rep["overall"]
    assert overall["fa_rate"]["successes"] == 1
    assert overall["fa_rate"]["n"] == 2  # actually-bad candidates
    assert overall["fa_rate"]["rate"] == 0.5
    assert overall["fr_rate"]["successes"] == 1
    assert overall["fr_rate"]["n"] == 2  # actually-good candidates
    assert overall["fr_rate"]["rate"] == 0.5
    assert overall["fa_fr_ratio"] == 1.0


def test_roundtrip_decisions_jsonl(events_db, corpus_jsonl, tmp_path):
    rc = mat.main([
        "--events", str(events_db),
        "--corpus", str(corpus_jsonl),
        "--output", str(tmp_path / "out"),
        "--emit", "decisions-jsonl",
    ])
    assert rc == 0
    jsonl = tmp_path / "out" / "reviewer_decisions.jsonl"
    assert jsonl.exists()

    rows = report.load_decisions_jsonl(jsonl)
    assert len(rows) == 4
    rep = report.build_report(rows)
    _assert_fa_fr(rep)

    # The report CLI itself runs clean end-to-end.
    out_json = tmp_path / "report.json"
    assert report.main(["--decisions", str(jsonl), "--out-json", str(out_json)]) == 0
    assert json.loads(out_json.read_text())["overall"]["fa_fr_ratio"] == 1.0


def test_roundtrip_ledger_sqlite(events_db, corpus_jsonl, tmp_path):
    rc = mat.main([
        "--events", str(events_db),
        "--corpus", str(corpus_jsonl),
        "--output", str(tmp_path / "out"),
        "--emit", "ledger-sqlite",
    ])
    assert rc == 0
    ledger = tmp_path / "out" / "review_ledger.sqlite"
    assert ledger.exists()

    rows = report.load_ledger_sqlite(ledger)
    assert len(rows) == 4
    rep = report.build_report(rows)
    _assert_fa_fr(rep)

    assert report.main(["--ledger", str(ledger)]) == 0


# --------------------------------------------------------------------------- #
# 4. Idempotency
# --------------------------------------------------------------------------- #
def test_ledger_sqlite_idempotent(events_db, corpus_jsonl, tmp_path):
    events = mat.read_review_events(events_db)
    corpus = mat._load_corpus(corpus_jsonl)
    rows = mat.materialize_rows(events, corpus=corpus)
    out = tmp_path / "out"

    path1, ins1, skip1 = mat.write_ledger_sqlite(rows, out)
    assert (ins1, skip1) == (4, 0)
    path2, ins2, skip2 = mat.write_ledger_sqlite(rows, out)
    assert path2 == path1
    assert (ins2, skip2) == (0, 4)  # re-run: all skipped as dups

    assert len(report.load_ledger_sqlite(path1)) == 4  # no duplication


# --------------------------------------------------------------------------- #
# 5. Zero-events path: loud, exit 0, non-error
# --------------------------------------------------------------------------- #
def test_zero_events_exits_zero_with_empty_message(tmp_path, capsys):
    # An events.sqlite that has NO review_decision rows (only a decoy).
    db = tmp_path / "empty_events.sqlite"
    emit(
        Event(
            ts_utc="2026-07-16T11:00:00+00:00",
            source=EventSource.AGENT_AUDIT,
            source_path="",
            source_line=None,
            category=EventCategory.TASK_START,
            summary="not a review event",
            detail_json=detail_to_json({"mode": "task"}),
        ),
        db_path=db,
    )
    rc = mat.main(["--events", str(db), "--output", str(tmp_path / "out")])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "0 events materialized" in captured
    assert "EXPECTED-EMPTY" in captured
    # Nothing was written.
    assert not (tmp_path / "out").exists()


def test_missing_db_is_expected_empty_not_error(tmp_path, capsys):
    rc = mat.main(["--events", str(tmp_path / "does_not_exist.sqlite")])
    assert rc == 0
    assert "0 events materialized" in capsys.readouterr().out


def test_dry_run_writes_nothing(events_db, tmp_path, capsys):
    rc = mat.main([
        "--events", str(events_db),
        "--output", str(tmp_path / "out"),
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "4 REVIEW_DECISION events found" in out
    assert "DRY-RUN" in out
    assert not (tmp_path / "out").exists()
