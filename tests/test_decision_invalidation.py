"""Tests for the CP2 DecisionEnvelope ledger + hash-bound invalidation + replay.

Covers (spec §5.5 / §12.3 / §12.4 / §20.1 / §20.3):
  * DecisionEnvelope append-only writer + idempotency (identical material inputs
    collapse to one row).
  * MaterialInputs content hashing: deterministic, and a change to ANY §12.3
    material input flips the hash.
  * Automatic invalidation: appends a DECISION_INVALIDATED event referencing the
    superseded decision, records WHICH input changed, and NEVER rewrites the
    envelope row (immutable-decision invariant).
  * Each §12.3 material input independently triggers invalidation.
  * Replay: reconstruct the reviewed package from content-addressed refs, report
    validity, and audit which input change invalidated approval.
  * Additive migration: the new table disturbs neither `event` nor `review_ledger`.

Hermetic: every test uses an isolated tmp_path SQLite file; none touch the
shared materialized data/trace/events.sqlite. NO inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.trace.store import (
    Event,
    EventCategory,
    ensure_schema,
    ensure_decision_envelope_schema,
    upsert_events,
)
from src.trace.review_ledger import (
    DECISION_ENVELOPE_SCHEMA_VERSION,
    MATERIAL_INPUT_FIELDS,
    DecisionEnvelopeRow,
    MaterialInputs,
    compute_material_hash,
    decision_envelope_count,
    detect_material_change,
    get_decision_envelope,
    invalidate_decision,
    invalidate_on_material_change,
    invalidations_for,
    is_decision_valid,
    iter_decision_envelopes,
    next_sequence_no,
    record_decision_envelope,
    replay_review_package,
)
from src.trace.review_ledger import ReviewLedgerRow, insert_review_ledger_row, review_ledger_count


@pytest.fixture
def conn(tmp_path: Path):
    c = ensure_schema(tmp_path / "events.sqlite")
    yield c
    c.close()


def _material(**over) -> MaterialInputs:
    base = dict(
        artifact_hash="sha256:art",
        specification_hash="sha256:spec",
        plan_hash="sha256:plan",
        assurance_profile_hash="sha256:prof",
        policy_hash="sha256:pol",
        rubric_hash="sha256:rub",
        verifier_registry_hash="sha256:ver",
        environment_hash="sha256:env",
        reviewer_model_hash="sha256:glm",
        prompt_hash="sha256:pmt",
        decoding_parameters_hash="sha256:dec",
        retrieved_evidence_hash="sha256:ret",
        security_policy_hash="sha256:secpol",
        evidence_assumptions_hash="sha256:asm",
    )
    base.update(over)
    return MaterialInputs(**base)


def _envelope_row(decision_event_id="devent_1", material: MaterialInputs | None = None, **over):
    m = material or _material()
    base = dict(
        decision_event_id=decision_event_id,
        task_id="task_1",
        artifact_hash=m.artifact_hash,
        specification_hash=m.specification_hash,
        candidate_package_hash="sha256:pkg",
        assurance_profile_hash=m.assurance_profile_hash,
        policy_hash=m.policy_hash,
        rubric_hash=m.rubric_hash,
        verifier_registry_hash=m.verifier_registry_hash,
        review_decision_hash="sha256:rev",
        verification_report_hash="sha256:vrp",
        cohort_id="swe:qwen:glm:iq2",
        sample_count=438,
        estimated_error_rate=0.074,
        upper_risk_bound=0.112,
        action="replan",
        blocking_reason_codes=["CONCLUSIVE_HIGH_SEVERITY_FAILURE"],
        material=m,
    )
    base.update(over)
    return DecisionEnvelopeRow(**base)


# --------------------------------------------------------------------------- #
# DDL + append-only writer + idempotency (§20.1)
# --------------------------------------------------------------------------- #
def test_envelope_ddl_created_by_ensure_schema(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(decision_envelope)").fetchall()}
    for expected in (
        "decision_event_id", "sequence_no", "idempotency_key", "task_id",
        "artifact_hash", "specification_hash", "candidate_package_hash",
        "assurance_profile_hash", "policy_hash", "rubric_hash", "verifier_registry_hash",
        "review_decision_hash", "verification_report_hash", "cohort_id",
        "action", "blocking_reason_codes", "supersedes", "invalidated_by",
        "valid_until_material_change", "material_hash", "envelope_json", "schema_version",
    ):
        assert expected in cols, f"missing column {expected}"


def test_append_only_writer_and_idempotent(conn):
    row = _envelope_row()
    ins, skp = record_decision_envelope(conn, row)
    assert (ins, skp) == (1, 0)
    assert row.sequence_no == 0
    assert row.idempotency_key == row.material_hash
    # Re-record an IDENTICAL decision (same material inputs) -> no-op (§20.1.7).
    ins2, skp2 = record_decision_envelope(conn, _envelope_row())
    assert (ins2, skp2) == (0, 1)
    assert decision_envelope_count(conn) == 1

    got = get_decision_envelope(conn, "devent_1")
    assert got["decision_event_id"] == "devent_1"
    assert got["action"] == "replan"
    assert got["blocking_reason_codes"] == ["CONCLUSIVE_HIGH_SEVERITY_FAILURE"]
    assert got["valid_until_material_change"] is True
    assert got["schema_version"] == DECISION_ENVELOPE_SCHEMA_VERSION
    # the stored schema-valid envelope round-trips
    assert got["envelope"]["subject"]["artifact_hash"] == "sha256:art"


def test_sequence_no_monotonic(conn):
    assert next_sequence_no(conn) == 0
    record_decision_envelope(conn, _envelope_row("d1", _material(artifact_hash="sha256:1")))
    record_decision_envelope(conn, _envelope_row("d2", _material(artifact_hash="sha256:2")))
    record_decision_envelope(conn, _envelope_row("d3", _material(artifact_hash="sha256:3")))
    seqs = [e["sequence_no"] for e in iter_decision_envelopes(conn)]
    assert seqs == [0, 1, 2]
    assert next_sequence_no(conn) == 3


# --------------------------------------------------------------------------- #
# MaterialInputs hashing (§12.3)
# --------------------------------------------------------------------------- #
def test_material_hash_deterministic():
    assert compute_material_hash(_material()) == compute_material_hash(_material())
    # mapping vs dataclass agree
    assert compute_material_hash(_material()) == compute_material_hash(_material().as_dict())


def test_material_fields_cover_spec_1223():
    # The §12.3 invalidation primitives are all present.
    for f in (
        "artifact_hash", "specification_hash", "plan_hash", "assurance_profile_hash",
        "policy_hash", "rubric_hash", "verifier_registry_hash", "environment_hash",
        "reviewer_model_hash", "prompt_hash", "decoding_parameters_hash",
        "retrieved_evidence_hash", "security_policy_hash", "evidence_assumptions_hash",
    ):
        assert f in MATERIAL_INPUT_FIELDS


@pytest.mark.parametrize("field_name", MATERIAL_INPUT_FIELDS)
def test_any_material_change_flips_hash(field_name):
    base = _material()
    changed = _material(**{field_name: "sha256:CHANGED"})
    assert compute_material_hash(base) != compute_material_hash(changed)
    assert detect_material_change(base, changed) == [field_name]


def test_detect_no_change():
    assert detect_material_change(_material(), _material()) == []


# --------------------------------------------------------------------------- #
# Automatic invalidation (§12.3) — append-only, never rewrite
# --------------------------------------------------------------------------- #
def test_invalidation_appends_event_never_rewrites(conn):
    record_decision_envelope(conn, _envelope_row())
    env_before = decision_envelope_count(conn)
    ev_before = conn.execute("SELECT COUNT(*) FROM event").fetchone()[0]
    row_before = get_decision_envelope(conn, "devent_1")

    new_m = _material(reviewer_model_hash="sha256:QWEN")
    res = invalidate_on_material_change(conn, "devent_1", new_m, new_decision_event_id="devent_2")
    assert res is not None
    assert res["changed_inputs"] == ["reviewer_model_hash"]

    # envelope table untouched (no new row, no rewrite); exactly one new event.
    assert decision_envelope_count(conn) == env_before
    assert conn.execute("SELECT COUNT(*) FROM event").fetchone()[0] == ev_before + 1
    assert get_decision_envelope(conn, "devent_1") == row_before  # byte-identical

    assert is_decision_valid(conn, "devent_1") is False
    invs = invalidations_for(conn, "devent_1")
    assert len(invs) == 1
    assert invs[0]["new_decision_event_id"] == "devent_2"
    # the invalidation event is a real trace row under the DECISION_INVALIDATED category
    cat = conn.execute(
        "SELECT category FROM event WHERE category = ?", (EventCategory.DECISION_INVALIDATED,)
    ).fetchone()
    assert cat[0] == "decision_invalidated"


def test_no_op_when_material_unchanged(conn):
    record_decision_envelope(conn, _envelope_row())
    assert invalidate_on_material_change(conn, "devent_1", _material()) is None
    assert is_decision_valid(conn, "devent_1") is True
    assert invalidations_for(conn, "devent_1") == []


def test_invalidation_idempotent(conn):
    record_decision_envelope(conn, _envelope_row())
    new_m = _material(policy_hash="sha256:POL2")
    r1 = invalidate_on_material_change(conn, "devent_1", new_m)
    r2 = invalidate_on_material_change(conn, "devent_1", new_m)  # same new state
    assert r1["inserted"] == 1
    assert r2["skipped"] == 1  # same synthetic key -> append-only no-op
    assert len(invalidations_for(conn, "devent_1")) == 1


@pytest.mark.parametrize("field_name", MATERIAL_INPUT_FIELDS)
def test_each_material_input_independently_invalidates(field_name, conn):
    """§20.3: change each material input independently and assert invalidation."""
    record_decision_envelope(conn, _envelope_row())
    new_m = _material(**{field_name: "sha256:DELTA"})
    res = invalidate_on_material_change(conn, "devent_1", new_m)
    assert res is not None, f"{field_name} did not invalidate"
    assert res["changed_inputs"] == [field_name]
    assert is_decision_valid(conn, "devent_1") is False


def test_invalidate_decision_direct(conn):
    # Low-level API: append an invalidation without diffing material.
    record_decision_envelope(conn, _envelope_row())
    res = invalidate_decision(
        conn, "devent_1", changed_inputs=["rubric_hash"], reason="rubric edited"
    )
    assert res["reason"] == "rubric edited"
    assert not is_decision_valid(conn, "devent_1")


def test_invalidate_unknown_decision_raises(conn):
    with pytest.raises(KeyError):
        invalidate_on_material_change(conn, "nope", _material())


# --------------------------------------------------------------------------- #
# Supersession (§17.3 appeal) — append a new row, prior row survives
# --------------------------------------------------------------------------- #
def test_superseding_decision_is_a_distinct_row(conn):
    record_decision_envelope(conn, _envelope_row("devent_1"))
    # An appeal after a fixed artifact: new material -> new envelope that supersedes.
    new_m = _material(artifact_hash="sha256:FIXED")
    superseding = _envelope_row(
        "devent_2", material=new_m, artifact_hash="sha256:FIXED",
        action="continue", blocking_reason_codes=[], supersedes="devent_1",
    )
    record_decision_envelope(conn, superseding)
    assert decision_envelope_count(conn) == 2
    # prior decision still on record (append-only, not erased)
    assert get_decision_envelope(conn, "devent_1")["action"] == "replan"
    assert get_decision_envelope(conn, "devent_2")["envelope"]["validity"]["supersedes"] == "devent_1"


# --------------------------------------------------------------------------- #
# Replay (§12.4)
# --------------------------------------------------------------------------- #
def test_replay_reconstructs_and_reports_validity(conn):
    record_decision_envelope(conn, _envelope_row())
    blobs = {
        "sha256:pkg": {"package_id": "cp-1"},
        "sha256:rev": {"decision": "reject"},
        "sha256:vrp": {"report_id": "vr-1"},
        "sha256:art": {"bytes": "..."},
    }
    rp = replay_review_package(conn, "devent_1", blob_resolver=blobs)
    assert rp["valid"] is True
    # content-addressed refs present for package reconstruction
    assert rp["content_addressed_refs"]["candidate_package_hash"] == "sha256:pkg"
    # resolved via the blob store
    assert rp["resolved"]["candidate_package_hash"] == {"package_id": "cp-1"}
    assert rp["resolved"]["review_decision_hash"] == {"decision": "reject"}
    # inputs to rerun deterministic reduction
    assert rp["reduction_inputs"]["verification_report_hash"] == "sha256:vrp"
    assert rp["reduction_inputs"]["policy_hash"] == "sha256:pol"
    # the reconstructed envelope is the schema-valid artifact
    assert rp["envelope"]["decision_event_id"] == "devent_1"
    assert rp["material_inputs"]["reviewer_model_hash"] == "sha256:glm"


def test_replay_callable_resolver_and_invalidation_audit(conn):
    record_decision_envelope(conn, _envelope_row())
    invalidate_on_material_change(conn, "devent_1", _material(environment_hash="sha256:NEWENV"))

    resolved_calls = []

    def resolver(h):
        resolved_calls.append(h)
        return f"content-of-{h}"

    rp = replay_review_package(conn, "devent_1", blob_resolver=resolver)
    assert rp["valid"] is False  # a material input changed
    assert len(rp["invalidations"]) == 1
    # §12.4: audit WHICH input change invalidated approval
    assert rp["invalidations"][0]["changed_inputs"] == ["environment_hash"]
    assert rp["resolved"]["artifact_hash"] == "content-of-sha256:art"
    assert resolved_calls  # resolver was actually invoked


def test_replay_refs_only_without_resolver(conn):
    record_decision_envelope(conn, _envelope_row())
    rp = replay_review_package(conn, "devent_1")
    assert rp["resolved"] == {}
    assert rp["content_addressed_refs"]["artifact_hash"] == "sha256:art"


def test_replay_unknown_decision_raises(conn):
    with pytest.raises(KeyError):
        replay_review_package(conn, "ghost")


# --------------------------------------------------------------------------- #
# Additive migration safety (§20.3 / A1)
# --------------------------------------------------------------------------- #
def test_envelope_table_does_not_disturb_events_or_ledger(tmp_path):
    db = tmp_path / "events.sqlite"
    conn = ensure_schema(db)
    try:
        upsert_events(
            conn,
            [Event(ts_utc="2026-07-17T00:00:00+00:00", source="s", source_path="/f", source_line=1)],
        )
        insert_review_ledger_row(conn, ReviewLedgerRow(decision_id="d1", decision="approve"))
        ev_before = conn.execute("SELECT COUNT(*) FROM event").fetchone()[0]
        rl_before = review_ledger_count(conn)

        # exercising the envelope ledger is idempotent + additive
        ensure_decision_envelope_schema(conn)
        record_decision_envelope(conn, _envelope_row())

        # the DECISION_INVALIDATED event is the ONLY new event row from invalidation
        invalidate_on_material_change(conn, "devent_1", _material(prompt_hash="sha256:PMT2"))

        assert review_ledger_count(conn) == rl_before == 1
        # events grew by exactly one (the invalidation event), review_ledger untouched
        assert conn.execute("SELECT COUNT(*) FROM event").fetchone()[0] == ev_before + 1
        assert decision_envelope_count(conn) == 1
    finally:
        conn.close()


def test_new_event_categories_exist():
    assert EventCategory.DECISION_INVALIDATED == "decision_invalidated"
    assert EventCategory.ESCALATION_CREATED == "escalation_created"
    assert EventCategory.ESCALATION_RESOLVED == "escalation_resolved"
    assert EventCategory.EVIDENCE_REQUESTED == "evidence_requested"
    assert EventCategory.EVIDENCE_RESULT == "evidence_result"
