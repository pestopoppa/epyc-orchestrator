"""Evidence-semantics tests (control-plane spec §20.2).

Six invariants over criterion-scoped authority + logical/execution status:

  1. A sound counterexample blocks only its criterion.
  2. A passing bounded test does not claim universal proof.
  3. A verifier crash is an operational error, not a failure.
  4. Solver UNKNOWN remains an epistemic unknown.
  5. Conflicting authoritative evidence escalates.
  6. A reviewer cannot override conclusive scoped evidence without differing scope/assumptions.

These test the CP1 ``authority`` module (landed) and the landed RD-3 verifier
precedence in ``review_service`` (which CP1's ``policy_reducer`` will subsume).
Reducer-level escalation/precedence assertions are guarded PENDING-CP1. NO inference.
"""

from __future__ import annotations

import importlib

import pytest


def _try_import(modpath: str):
    try:
        return importlib.import_module(modpath)
    except Exception:
        return None


authority = _try_import("src.proactive_delegation.authority")  # CP1
policy_reducer = _try_import("src.proactive_delegation.policy_reducer")  # CP1

requires_authority = pytest.mark.skipif(
    authority is None, reason="pending-CP1: src.proactive_delegation.authority not landed"
)
requires_reducer = pytest.mark.skipif(
    policy_reducer is None or not hasattr(policy_reducer, "reduce_decision"),
    reason="pending-CP1: policy_reducer.reduce_decision not landed",
)


# ── §20.2.1 — sound counterexample blocks ONLY its criterion ──────────────
@requires_authority
class TestSoundCounterexampleScoped:
    def test_sound_refutation_may_block_not_approve(self):
        a = authority.Authority.from_dict(
            {"class": "sound_refutation", "valid_for": ["migration_idempotence"]}
        )
        assert a.may_block() is True
        assert a.may_approve() is False  # a passing run proves nothing (§7.1)

    def test_counterexample_scopes_to_its_own_criterion_only(self):
        a = authority.Authority.from_dict(
            {"class": "sound_refutation", "valid_for": ["migration_idempotence"]}
        )
        assert a.scopes_criterion("migration_idempotence") is True
        assert a.scopes_criterion("backward_compatibility") is False

    def test_conclusive_failure_only_on_scoped_criterion(self):
        a = authority.Authority.from_dict({"class": "sound_refutation", "valid_for": ["c1"]})
        L, E = authority.LogicalStatus, authority.ExecutionStatus
        # A clean FAIL under sound_refutation is a conclusive block for c1...
        assert authority.is_conclusive_failure(a, L.FAIL, E.OK) is True
        # ...but its authority does not extend to an unrelated criterion.
        assert a.scopes_criterion("c2") is False


# ── §20.2.2 — passing bounded test ≠ universal proof ──────────────────────
@requires_authority
class TestBoundedTestNotUniversalProof:
    def test_bounded_pass_does_not_auto_approve(self):
        a = authority.Authority.from_dict({"class": "bounded_test"})
        L, E = authority.LogicalStatus, authority.ExecutionStatus
        # Without a calibrated policy grant a bounded PASS cannot establish the property.
        assert authority.is_conclusive_pass(a, L.PASS, E.OK) is False
        # A policy grant (calibrated) may promote it — but that is an explicit decision.
        assert authority.is_conclusive_pass(a, L.PASS, E.OK, policy_grant=True) is True

    def test_proof_pass_is_universal_but_bounded_is_not(self):
        proof = authority.Authority.from_dict({"class": "proof"})
        bounded = authority.Authority.from_dict({"class": "bounded_test"})
        L, E = authority.LogicalStatus, authority.ExecutionStatus
        assert authority.is_conclusive_pass(proof, L.PASS, E.OK) is True
        assert authority.is_conclusive_pass(bounded, L.PASS, E.OK) is False


# ── §20.2.3 — verifier crash = operational error, not failure ─────────────
@requires_authority
class TestVerifierCrashIsOperational:
    def test_execution_error_is_operational(self):
        E = authority.ExecutionStatus
        assert authority.is_operational_error(E.ERROR) is True
        assert authority.is_operational_error(E.TIMEOUT) is True
        assert authority.is_operational_error(E.UNAVAILABLE) is True
        assert authority.is_operational_error(E.OK) is False

    def test_crash_is_not_a_conclusive_failure(self):
        """A proof-class verifier that CRASHED must not masquerade as a logical fail."""
        a = authority.Authority.from_dict({"class": "proof"})
        L, E = authority.LogicalStatus, authority.ExecutionStatus
        # Even a would-be FAIL verdict is void when execution did not complete cleanly.
        assert authority.is_conclusive_failure(a, L.FAIL, E.ERROR) is False
        assert authority.is_conclusive_failure(a, L.UNKNOWN, E.ERROR) is False


# ── §20.2.4 — solver UNKNOWN stays an epistemic unknown ───────────────────
@requires_authority
class TestSolverUnknownStaysUnknown:
    def test_unknown_is_neither_pass_nor_fail(self):
        a = authority.Authority.from_dict({"class": "complete_decider"})
        L, E = authority.LogicalStatus, authority.ExecutionStatus
        # logical=unknown, execution=ok: the solver ran fine but could not decide.
        assert authority.is_conclusive_failure(a, L.UNKNOWN, E.OK) is False
        assert authority.is_conclusive_pass(a, L.UNKNOWN, E.OK) is False

    def test_inconclusive_outcome_maps_to_unknown_not_fail(self):
        # v1.0 back-compat mapping: a report 'inconclusive' outcome is epistemic unknown.
        assert authority.coerce_logical("inconclusive") is authority.LogicalStatus.UNKNOWN


# ── §20.2.5 — conflicting authoritative evidence escalates ────────────────
@requires_authority
class TestConflictEscalates:
    def test_conflict_is_neither_pass_nor_fail(self):
        a = authority.Authority.from_dict({"class": "proof"})
        L, E = authority.LogicalStatus, authority.ExecutionStatus
        assert authority.is_conclusive_failure(a, L.CONFLICT, E.OK) is False
        assert authority.is_conclusive_pass(a, L.CONFLICT, E.OK) is False

    def test_conflict_routes_to_durable_escalation_live(self, tmp_path):
        """LIVE: a conflict is escalated to the durable sink (never auto-resolved by a model vote)."""
        from src.proactive_delegation.escalation_sink import EscalationSink

        sink = EscalationSink(tmp_path / "events.sqlite")
        try:
            eid = sink.escalate(
                {"criterion_id": "migration_idempotence"}, "CONFLICTING_AUTHORITATIVE_EVIDENCE"
            )
            opens = sink.open_escalations()
            assert [e["escalation_id"] for e in opens] == [eid]
            assert opens[0]["reason_code"] == "CONFLICTING_AUTHORITATIVE_EVIDENCE"
        finally:
            sink.close()

    @requires_reducer
    def test_reducer_escalates_on_conflict(self):
        """PENDING-CP1: reduce_decision must return an escalate action with the
        CONFLICTING_AUTHORITATIVE_EVIDENCE reason code (spec §8 step 2)."""
        verification = {
            "report_id": "vr-x",
            "checks": [
                {"check_id": "a", "kind": "test", "outcome": "pass"},
                {"check_id": "b", "kind": "constraint_check", "outcome": "fail",
                 "certificate": {"type": "counterexample", "payload": "x"}},
            ],
            "summary": {"conclusive_verdict": "conflict"},
        }
        profile = {"profile_id": "p", "criteria": {}, "policy": {}}
        try:
            result = policy_reducer.reduce_decision({"package_id": "cp"}, verification, None, profile, {})
        except TypeError:
            pytest.skip("pending-CP1: reduce_decision signature not yet spec-aligned (§8)")
        blob = repr(getattr(result, "__dict__", result)) + repr(result)
        assert "CONFLICTING_AUTHORITATIVE_EVIDENCE" in blob or "escalate" in blob.lower()


# ── §20.2.6 — reviewer cannot override conclusive scoped evidence ─────────
@requires_authority
class TestReviewerCannotOverrideConclusive:
    def test_heuristic_cannot_override_conclusive_block(self):
        """A reviewer's heuristic authority cannot widen to block/approve over sound evidence (§7.3 rule 3)."""
        heuristic = authority.Authority.from_dict({"class": "heuristic_static"})
        assert heuristic.may_block() is False
        assert heuristic.may_approve() is False

    def test_reviewer_approve_cannot_override_conclusive_fail_live(self):
        """LIVE (landed RD-3 precedence): reviewer APPROVE + conclusive gate FAIL is
        overridden — the reviewer cannot let a conclusive failure through."""
        from src.proactive_delegation.review_service import ArchitectReviewService
        from src.proactive_delegation.types import ArchitectReview, ReviewDecision

        class _Stub:
            def llm_call(self, *a, **k):
                return ""

        svc = ArchitectReviewService(_Stub(), trace_sink=lambda ev: None)
        review = ArchitectReview(subtask_id="S1", decision=ReviewDecision.APPROVE)
        report = {
            "report_id": "vr-1",
            "checks": [
                {
                    "check_id": "unit",
                    "kind": "test",
                    "outcome": "fail",
                    "certificate": {"type": "failing_assertion", "payload": "assert x==1"},
                }
            ],
        }
        adjusted = svc.apply_verifier_precedence(review, report)
        assert adjusted.decision != ReviewDecision.APPROVE  # override — not let through

    def test_reviewer_reject_cannot_stand_over_conclusive_pass_live(self):
        """LIVE: reviewer REJECT + conclusive gate PASS never yields a reject (§7.3 rule 2)."""
        from src.proactive_delegation.review_service import ArchitectReviewService
        from src.proactive_delegation.types import ArchitectReview, ReviewDecision

        class _Stub:
            def llm_call(self, *a, **k):
                return ""

        svc = ArchitectReviewService(_Stub(), trace_sink=lambda ev: None)
        review = ArchitectReview(subtask_id="S1", decision=ReviewDecision.REJECT)
        report = {
            "report_id": "vr-2",
            "checks": [{"check_id": "unit", "kind": "test", "outcome": "pass"}],
        }
        adjusted = svc.apply_verifier_precedence(review, report)
        assert adjusted.decision not in (ReviewDecision.REJECT, ReviewDecision.REJECT_TO_EMPTY)
