"""UTM-V1/V2/V3-structural: envelope, transitions, structural verifier, shadow proposer."""

from __future__ import annotations

import json

import pytest

from orchestration.repl_memory.memory_operations import (
    TOMBSTONE_HASH,
    MemoryEngine,
    MemoryOperation,
    MemoryOperationError,
    ScopedWriteProposer,
    WriteContext,
    agreement_report,
    calibration_report,
    structural_check,
    value_hash,
)

SRC = {"ev1", "ev2", "ev3"}
APP = {"scope": "user", "user_id": "u1"}


def eng(**kw):
    return MemoryEngine(source_exists=lambda s: s in SRC, **kw)


def add(op_id="o1", key="k", value="likes tea", version=1, **kw):
    base = dict(op_id=op_id, session_id="s1", actor_id="agent-a", op="ADD", subject_id="u1", key=key,
                source_event_ids=("ev1",), value=value, after_hash=value_hash(value), version=version,
                applicability=APP)
    base.update(kw)
    return MemoryOperation(**base)


def upd(op_id, prev_op, prev_value, value, version, key="k", **kw):
    base = dict(op_id=op_id, session_id="s1", actor_id="agent-a", op="UPDATE", subject_id="u1", key=key,
                source_event_ids=("ev2",), value=value, after_hash=value_hash(value),
                before_hash=value_hash(prev_value), supersedes=prev_op, version=version, applicability=APP)
    base.update(kw)
    return MemoryOperation(**base)


def delete(op_id, prev_op, prev_value, version, key="k", **kw):
    base = dict(op_id=op_id, session_id="s1", actor_id="agent-a", op="DELETE", subject_id="u1", key=key,
                source_event_ids=("ev3",), after_hash=TOMBSTONE_HASH, before_hash=value_hash(prev_value),
                supersedes=prev_op, version=version, applicability=APP)
    base.update(kw)
    return MemoryOperation(**base)


def run(e, op):
    e.submit(op)
    e.validate(op.op_id)
    if e.ops[op.op_id].state == "validated":
        e.commit(op.op_id)
    return e.ops[op.op_id]


# ---- envelope + engine
def test_add_update_versions_history_kept_and_log_append_only(tmp_path):
    p = tmp_path / "log.jsonl"
    e = eng(log_path=p)
    assert run(e, add()).state == "committed"
    assert run(e, upd("o2", "o1", "likes tea", "likes coffee", 2)).state == "committed"
    assert e.head_value("k") == "likes coffee"
    assert [v.value for v in e.restore_candidates("k")] == ["likes coffee", "likes tea"]
    seqs = [r["seq"] for r in e.log]
    assert seqs == list(range(1, len(seqs) + 1))
    assert len(p.read_text().splitlines()) == len(e.log) == 6  # 2 ops x (proposed, validated, committed)
    assert json.loads(p.read_text().splitlines()[0])["to"] == "proposed"


def test_soft_delete_tombstones_preserves_history_and_blocks_update():
    e = eng()
    run(e, add())
    d = run(e, delete("o2", "o1", "likes tea", 2))
    assert d.state == "committed" and e.head_value("k") is None
    assert [v.value for v in e.restore_candidates("k")] == [None, "likes tea"]
    bad = run(e, upd("o3", "o2", "x", "again", 3, before_hash=TOMBSTONE_HASH))
    assert bad.state == "rejected" and "tombstone:update_on_tombstone" in bad.error


def test_rollback_restores_prior_head_keeps_versions_and_version_not_reused():
    e = eng()
    run(e, add())
    run(e, upd("o2", "o1", "likes tea", "likes coffee", 2))
    e.rollback("o2")
    assert e.ops["o2"].state == "rolled_back" and e.head_value("k") == "likes tea"
    assert len(e.restore_candidates("k")) == 2  # nothing erased
    stale = run(e, upd("o3", "o1", "likes tea", "x", 2))  # version 2 was used: never reused
    assert stale.state == "rejected" and "version:expected_3_got_2" in stale.error
    ok = run(e, upd("o4", "o1", "likes tea", "likes mate", 3))
    assert ok.state == "committed" and e.head_value("k") == "likes mate"


def test_rollback_of_delete_restores_value_and_non_head_rollback_refused():
    e = eng()
    run(e, add())
    run(e, delete("o2", "o1", "likes tea", 2))
    e.rollback("o2")
    assert e.head_value("k") == "likes tea"
    run(e, upd("o3", "o1", "likes tea", "likes coffee", 2 + 1))
    with pytest.raises(MemoryOperationError, match="rollback_not_head"):
        e.rollback("o1")


def test_stale_fencing_at_commit_time():
    e = eng()
    run(e, add())
    a = upd("oa", "o1", "likes tea", "A", 2)
    b = upd("ob", "o1", "likes tea", "B", 2)
    e.submit(a); e.submit(b)
    e.validate("oa"); e.validate("ob")  # both validate against the same head
    e.commit("oa")
    r = e.commit("ob")  # head moved: recheck rejects
    assert r.state == "rejected" and "commit_recheck" in r.error and e.head_value("k") == "A"


def test_illegal_transitions_and_smuggled_state_refused():
    e = eng()
    with pytest.raises(MemoryOperationError):
        e.submit(add(state="committed"))
    e.submit(add())
    with pytest.raises(MemoryOperationError):
        e.commit("o1")  # proposed -> committed is illegal
    e.validate("o1"); e.commit("o1")
    with pytest.raises(MemoryOperationError):
        e.reject("o1", "late")
    with pytest.raises(MemoryOperationError):
        e.submit(add())  # duplicate op_id


def test_tombstone_is_not_erasure_and_erasure_needs_receipt():
    e = eng()
    run(e, add())
    run(e, delete("o2", "o1", "likes tea", 2))
    with pytest.raises(MemoryOperationError):
        e.confirm_erasure("k", "receipt-1")  # a tombstone cannot be confirmed as erased
    e2 = eng()
    run(e2, add())
    run(e2, delete("o2", "o1", "likes tea", 2, delete_mode="erase"))
    assert e2.items["k"].erasure == "requested"
    with pytest.raises(MemoryOperationError):
        e2.confirm_erasure("k", " ")
    e2.confirm_erasure("k", "erasure-receipt-7")
    assert e2.items["k"].erasure == "erased"
    after = run(e2, add("o9", value="back", version=3))
    assert after.state == "rejected" and "tombstone:item_erasure_erased" in after.error


# ---- structural checks, one per class
@pytest.mark.parametrize(
    "op,needle",
    [
        (add(op_id=""), "schema:op_id_missing"),
        (add(value=""), "schema:value_missing"),
        (add(after_hash="deadbeef"), "schema:after_hash_mismatch"),
        (add(key=""), "schema:key_missing"),
        (add(version=None), "schema:version_missing"),
        (add(source_event_ids=()), "provenance:mutation_without_sources"),
        (add(source_event_ids=("ev1", "ev1")), "schema:source_event_ids_duplicate"),
        (add(source_event_ids=("ghost",)), "provenance:missing_source:ghost"),
        (add(applicability={}), "scope:applicability_scope_missing_or_unknown"),
        (add(applicability={"scope": "user", "user_id": "someone-else"}), "scope:user_id_mismatch_for_user"),
        (add(applicability={"scope": "run", "run_id": "other"}), "scope:run_id_mismatch_for_run"),
        (add(applicability={"scope": "agent"}), "scope:agent_id_missing_for_agent"),
        (MemoryOperation("o", "s", "a", "TRUNCATE", "u"), "schema:unknown_op:'TRUNCATE'"),
        (MemoryOperation("o", "s", "a", "SUMMARIZE", "u", applicability=APP), "provenance:summary_without_sources"),
        (MemoryOperation("o", "s", "a", "RETRIEVE", "u", value="x", applicability=APP),
         "schema:nonmutating_carries_mutation_fields"),
    ],
)
def test_structural_reasons(op, needle):
    errs = structural_check(op, None, source_exists=lambda s: s in SRC)
    assert needle in errs, errs


def test_missing_source_is_rejected_not_assumed_safe():
    e = eng()
    r = run(e, add(source_event_ids=("ghost",)))
    assert r.state == "rejected" and "missing_source:ghost" in r.error and "k" not in e.items


def test_update_requires_head_supersession_and_matching_before_hash():
    e = eng()
    run(e, add())
    assert "version:supersedes_not_head" in structural_check(
        upd("o2", "other", "likes tea", "x", 2), e.items["k"], source_exists=lambda s: True)
    assert "stale:before_hash" in structural_check(
        upd("o2", "o1", "WRONG", "x", 2), e.items["k"], source_exists=lambda s: True)
    assert "version:item_exists" in structural_check(add("o2", version=2), e.items["k"],
                                                     source_exists=lambda s: True)
    assert any(x.startswith("version:no_head") for x in structural_check(
        upd("o2", "o1", "x", "y", 1, key="nokey"), None, source_exists=lambda s: True))


def test_nonmutating_ops_commit_without_touching_items():
    e = eng()
    for i, kind in enumerate(("RETRIEVE", "FILTER", "SELECT_EPISODE", "NOOP")):
        op = MemoryOperation(f"n{i}", "s1", "a", kind, "u1", applicability=APP, source_event_ids=("ev1",))
        assert run(e, op).state == "committed"
    assert e.items == {}
    summ = MemoryOperation("sm", "s1", "a", "SUMMARIZE", "u1", applicability=APP, source_event_ids=("ev1", "ev2"))
    assert run(e, summ).state == "committed" and e.items == {}  # a summary never replaces raw evidence


# ---- V2 shadow proposer
CTX = WriteContext("s1", "agent-a", "u1", "user", "u1")


def test_proposer_adds_updates_noops_and_never_commits():
    e = eng()
    pr = ScopedWriteProposer(e)
    p1 = pr.propose("lives in Oslo", ["ev1"], CTX, op_id="p1", key_hint="home")
    assert p1.op == "ADD" and p1.state == "proposed" and p1.actor_id == "agent-a" and p1.subject_id == "u1"
    assert e.items == {}  # proposing wrote nothing
    run(e, p1)
    p2 = pr.propose("lives in Oslo", ["ev2"], CTX, op_id="p2")
    assert p2.op == "NOOP"
    p3 = pr.propose("lives in Oslo Norway now", ["ev2"], CTX, op_id="p3")
    assert p3.op == "UPDATE" and p3.supersedes == "p1" and p3.version == 2 and p3.before_hash == value_hash("lives in Oslo")
    assert run(e, p3).state == "committed"


def test_proposer_conflict_retrieval_stays_in_scope_and_subject():
    e = eng()
    pr = ScopedWriteProposer(e)
    run(e, pr.propose("prefers dark mode", ["ev1"], CTX, op_id="p1", key_hint="pref"))
    other_subject = WriteContext("s1", "agent-a", "u2", "user", "u2")
    p = pr.propose("prefers dark mode theme", ["ev1"], other_subject, op_id="p2")
    assert p.op == "ADD" and p.subject_id == "u2"  # u1's memory is not a neighbour for u2
    run_ctx = WriteContext("s1", "agent-a", "u1", "run", "s1")
    assert pr.propose("prefers dark mode theme", ["ev1"], run_ctx, op_id="p3").op == "ADD"


def test_proposer_injected_decide_delete_and_undecidable_becomes_noop():
    e = eng()
    run(e, ScopedWriteProposer(e).propose("owns a cat", ["ev1"], CTX, op_id="p1", key_hint="pet"))
    pr = ScopedWriteProposer(e, decide_fn=lambda f, n: ("DELETE", n[0][0]))
    d = pr.propose("owns a cat", ["ev3"], CTX, op_id="p2")
    assert d.op == "DELETE" and d.after_hash == TOMBSTONE_HASH and run(e, d).state == "committed"
    bad = ScopedWriteProposer(e, decide_fn=lambda f, n: ("UPDATE", "no-such-key"))
    assert bad.propose("x", ["ev1"], CTX, op_id="p3").op == "NOOP"


def test_agreement_and_calibration_reports_do_not_invent():
    ops = [MemoryOperation("a", "s", "x", "ADD", "u"), MemoryOperation("b", "s", "x", "NOOP", "u"),
           MemoryOperation("c", "s", "x", "ADD", "u")]
    rep = agreement_report(ops, {"a": "ADD", "b": "UPDATE"})
    assert rep == {"agree": 1, "disagree": 1, "unlabelled": 1, "confusion": {"UPDATE->NOOP": 1}}
    cal = calibration_report([
        {"class": "stale", "label_valid": False, "verdict_valid": True},
        {"class": "stale", "label_valid": False, "verdict_valid": False},
        {"class": "duplicate", "label_valid": True, "verdict_valid": False},
    ])
    assert cal == {"stale": {"n": 2, "false_pass": 1, "false_fail": 0},
                   "duplicate": {"n": 1, "false_pass": 0, "false_fail": 1}}
    assert "private" not in cal  # absent class is absent, not 0%
