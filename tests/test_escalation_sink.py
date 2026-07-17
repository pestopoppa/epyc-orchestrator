"""Tests for the durable escalation sink (CP3).

Spec: control-plane spec §21 Phase-0 (durable escalation sink), §12.2 event
categories, §24 durability checklist ("Escalation cannot enter a dead state").

ALL tests here are LIVE — the escalation sink is landed in this wave. Every
DB-touching test uses an isolated ``tmp_path`` SQLite file; none touch the shared
``data/trace/events.sqlite``. NO inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.proactive_delegation.escalation_sink import (
    ESCALATION_CREATED,
    ESCALATION_REASONS,
    ESCALATION_RESOLVED,
    RESOLUTION_CODES,
    EscalationNotFound,
    EscalationSink,
    EscalationStateError,
)


@pytest.fixture
def sink(tmp_path: Path):
    s = EscalationSink(tmp_path / "events.sqlite")
    yield s
    s.close()


# ── raise / open ─────────────────────────────────────────────────────────
def test_escalate_creates_open_escalation(sink):
    eid = sink.escalate({"decision_id": "dev-1"}, "CONFLICTING_AUTHORITATIVE_EVIDENCE")
    assert eid.startswith("esc-")
    opens = sink.open_escalations()
    assert len(opens) == 1
    assert opens[0]["escalation_id"] == eid
    assert opens[0]["reason_code"] == "CONFLICTING_AUTHORITATIVE_EVIDENCE"
    assert opens[0]["status"] == "open"


def test_escalate_accepts_string_subject(sink):
    eid = sink.escalate("candpkg-77", "MANUAL")
    got = sink.get(eid)
    assert got["subject"] == {"ref": "candpkg-77"}


def test_reason_code_must_be_known(sink):
    with pytest.raises(ValueError):
        sink.escalate("x", "NOT_A_REAL_REASON")


def test_all_spec_reasons_accepted(sink):
    for i, reason in enumerate(sorted(ESCALATION_REASONS)):
        eid = sink.escalate({"i": i}, reason)
        assert sink.get(eid)["reason_code"] == reason


# ── resolve / terminal ───────────────────────────────────────────────────
def test_resolve_moves_to_terminal(sink):
    eid = sink.escalate("pkg", "REVIEWER_ABSTAIN")
    assert sink.resolve(eid, {"code": "RESOLVED_HUMAN", "note": "adjudicated"}) is True
    assert sink.open_escalations() == []
    resolved = sink.resolved_escalations()
    assert len(resolved) == 1
    assert resolved[0]["escalation_id"] == eid
    assert resolved[0]["resolution_code"] == "RESOLVED_HUMAN"
    assert sink.get(eid)["status"] == "resolved"


def test_resolve_accepts_bare_code_string(sink):
    eid = sink.escalate("pkg", "MANUAL")
    sink.resolve(eid, "RESOLVED_ABANDONED")
    assert sink.get(eid)["resolution_code"] == "RESOLVED_ABANDONED"


def test_resolution_code_must_be_known(sink):
    eid = sink.escalate("pkg", "MANUAL")
    with pytest.raises(ValueError):
        sink.resolve(eid, {"code": "RESOLVED_SOMEHOW"})
    # still open — the failed resolve did not mutate state.
    assert len(sink.open_escalations()) == 1


def test_resolve_unknown_id_raises(sink):
    with pytest.raises(EscalationNotFound):
        sink.resolve("esc-does-not-exist", "RESOLVED_HUMAN")


def test_reresolve_terminal_raises(sink):
    """Terminal-state-safe: a resolved escalation cannot be silently rewritten."""
    eid = sink.escalate("pkg", "MANUAL")
    sink.resolve(eid, "RESOLVED_HUMAN")
    with pytest.raises(EscalationStateError):
        sink.resolve(eid, "RESOLVED_AUTOMATED")
    # state unchanged: still exactly one resolved, terminal record intact.
    assert sink.get(eid)["resolution_code"] == "RESOLVED_HUMAN"


# ── idempotency / append-only ────────────────────────────────────────────
def test_idempotent_escalate_by_key_does_not_dup(sink):
    a = sink.escalate("pkg", "MANUAL", idempotency_key="k-1")
    b = sink.escalate("pkg", "MANUAL", idempotency_key="k-1")
    assert a == b
    assert len(sink.open_escalations()) == 1
    assert sink.stats()["total"] == 1


def test_distinct_escalations_without_key_are_unique(sink):
    a = sink.escalate("pkg", "MANUAL")
    b = sink.escalate("pkg", "MANUAL")
    assert a != b
    assert sink.stats()["total"] == 2


def test_append_only_no_update_or_delete(sink, tmp_path):
    """Resolving appends a NEW event; it never mutates or deletes the CREATED row."""
    eid = sink.escalate("pkg", "MANUAL")
    conn = sink._conn  # noqa: SLF001 — white-box check of the append-only invariant
    n_after_create = conn.execute(
        "SELECT COUNT(*) FROM event WHERE source_path LIKE 'escalation://%'"
    ).fetchone()[0]
    assert n_after_create == 1
    sink.resolve(eid, "RESOLVED_HUMAN")
    rows = conn.execute(
        "SELECT category, source_line FROM event WHERE source_path = ? ORDER BY source_line",
        (f"escalation://{eid}",),
    ).fetchall()
    # Both events persist: CREATED(seq0) + RESOLVED(seq1). Nothing overwritten.
    assert [r[0] for r in rows] == [ESCALATION_CREATED, ESCALATION_RESOLVED]
    assert [r[1] for r in rows] == [0, 1]


# ── no dead state / durability ───────────────────────────────────────────
def test_no_dead_state_open_always_queryable(sink):
    """An unresolved escalation is never lost — it stays in open_escalations()."""
    ids = [sink.escalate({"n": n}, "EVIDENCE_BUDGET_EXHAUSTED") for n in range(5)]
    sink.resolve(ids[1], "RESOLVED_REPLANNED")
    sink.resolve(ids[3], "RESOLVED_APPROVED")
    open_ids = {e["escalation_id"] for e in sink.open_escalations()}
    resolved_ids = {e["escalation_id"] for e in sink.resolved_escalations()}
    # Every escalation is in exactly one bucket — none dropped.
    assert open_ids | resolved_ids == set(ids)
    assert open_ids & resolved_ids == set()
    assert open_ids == {ids[0], ids[2], ids[4]}


def test_integrity_check_healthy_by_construction(sink):
    ids = [sink.escalate({"n": n}, "MANUAL") for n in range(3)]
    sink.resolve(ids[0], "RESOLVED_HUMAN")
    assert sink.integrity_check() == []


def test_integrity_check_flags_orphan_resolved(sink):
    """A RESOLVED with no CREATED (out-of-band writer) is a dead-state violation."""
    eid = sink.escalate("pkg", "MANUAL")
    sink.resolve(eid, "RESOLVED_HUMAN")
    conn = sink._conn  # noqa: SLF001
    # Fabricate an orphan terminal event directly in the ledger.
    from src.trace.store import Event, upsert_events

    upsert_events(
        conn,
        [
            Event(
                ts_utc="2026-07-17T00:00:00Z",
                source="review_plane",
                source_path="escalation://esc-orphan",
                source_line=1,
                category=ESCALATION_RESOLVED,
                status="RESOLVED_HUMAN",
                detail_json='{"escalation_id":"esc-orphan","event":"escalation_resolved"}',
            )
        ],
    )
    violations = sink.integrity_check()
    assert any(v["kind"] == "orphan_resolved" for v in violations)


def test_durable_across_reopen(tmp_path: Path):
    db = tmp_path / "events.sqlite"
    s1 = EscalationSink(db)
    eid_open = s1.escalate("pkg-open", "SEVERE_DEFECT_SUSPECTED")
    eid_done = s1.escalate("pkg-done", "MANUAL")
    s1.resolve(eid_done, "RESOLVED_HUMAN")
    s1.close()

    s2 = EscalationSink(db)
    try:
        open_ids = {e["escalation_id"] for e in s2.open_escalations()}
        assert open_ids == {eid_open}
        assert s2.get(eid_done)["status"] == "resolved"
    finally:
        s2.close()


def test_categories_and_reason_codes_are_reason_coded(sink):
    """Every stored event carries a reason/resolution code (never free-form-only)."""
    eid = sink.escalate("pkg", "NO_REVIEWER_AVAILABLE")
    sink.resolve(eid, "RESOLVED_SUPERSEDED")
    conn = sink._conn  # noqa: SLF001
    rows = conn.execute(
        "SELECT category, status FROM event WHERE source_path = ? ORDER BY source_line",
        (f"escalation://{eid}",),
    ).fetchall()
    assert rows[0][0] == ESCALATION_CREATED and rows[0][1] in ESCALATION_REASONS
    assert rows[1][0] == ESCALATION_RESOLVED and rows[1][1] in RESOLUTION_CODES
