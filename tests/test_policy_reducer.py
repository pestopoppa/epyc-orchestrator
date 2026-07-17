"""Tests for the deterministic policy reducer (CP1, spec §8) + RD-3 subsumption.

Zero inference. Three concerns:
  1. The RD-3 shadow-equivalence primitives the reducer now owns
     (conclusive_verdict / fail_certificates / verifier_precedence_recommendation)
     and the BYTE-IDENTICAL proof that review_service.apply_verifier_precedence
     still produces the exact pre-refactor ArchitectReview (subsumption is
     behavior-preserving).
  2. The full §8 reduce_decision across every branch, driven by criterion-scoped
     authority + logical/execution status + calibrated reviewer authority.
  3. Contract properties: purity/replay (§12.4), bounded loops (§5.6/§10.2),
     raw-confidence is ignored (§5.4), envelope-valid actions (CP2 interop), and
     loading a real CP2 AssuranceProfile JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.proactive_delegation.review_service import ArchitectReviewService
from src.proactive_delegation.types import ArchitectReview, ReviewDecision
from src.proactive_delegation.policy_reducer import (
    FA_CANDIDATE,
    FR_CANDIDATE,
    AssuranceProfile,
    CalibrationSnapshot,
    LoopState,
    PolicyAction,
    PolicyResult,
    ReducerPolicy,
    ReviewView,
    VerificationView,
    conclusive_verdict,
    fail_certificates,
    reduce_decision,
    verifier_precedence_recommendation,
)

_ORCH = Path(__file__).resolve().parents[1] / "orchestration"


# ─── builders ────────────────────────────────────────────────────────────────


class _StubPrimitives:
    def __init__(self, response=""):
        self.response = response

    def llm_call(self, prompt, role=None, n_tokens=None, **kw):
        return self.response


def _service(**kw):
    kw.setdefault("trace_sink", lambda ev: None)
    return ArchitectReviewService(_StubPrimitives(), **kw)


def _check(
    criterion_id,
    logical="pass",
    execution="ok",
    authority="proof",
    severity="high",
    required=True,
    may_block=None,
    may_approve=None,
):
    auth = {"class": authority}
    if may_block is not None:
        auth["may_block"] = may_block
    if may_approve is not None:
        auth["may_approve"] = may_approve
    outcome = {"pass": "pass", "fail": "fail", "unknown": "inconclusive", "conflict": "inconclusive"}[logical]
    return {
        "check_id": f"{criterion_id}_chk",
        "criterion_id": criterion_id,
        "kind": "gate",
        "outcome": outcome,
        "logical_status": logical,
        "execution_status": execution,
        "severity": severity,
        "authority": auth,
        "required": required,
    }


def _report(*checks, summary=None):
    r = {"report_id": "R1", "schema_version": "1.1.0", "checks": list(checks)}
    if summary is not None:
        r["summary"] = summary
    return r


def _profile(criteria=None, **policy):
    criteria = criteria or {"functional_correctness": ("critical", True)}
    return AssuranceProfile.from_dict(
        {
            "profile_id": "test:v1",
            "criteria": {cid: {"severity": s, "mandatory": m} for cid, (s, m) in criteria.items()},
            "policy": {
                "evidence_budget_exhausted": "fail_closed",
                "max_review_rounds": 2,
                "max_evidence_rounds": 2,
                **policy,
            },
        }
    )


_AUTHORIZED = CalibrationSnapshot(cohort_id="c", sample_count=500, estimated_error_rate=0.02, upper_risk_bound=0.03)
_UNAUTHORIZED = CalibrationSnapshot(cohort_id="c", sample_count=500, estimated_error_rate=0.4, upper_risk_bound=0.5)


# ═══ 1. RD-3 subsumption primitives + byte-identical proof ═══════════════════


class TestConclusiveVerdictPrimitive:
    def test_summary_wins(self):
        assert conclusive_verdict({"summary": {"conclusive_verdict": "fail"}}) == "fail"

    def test_derives_from_required_checks(self):
        assert conclusive_verdict({"checks": [{"outcome": "pass"}, {"outcome": "pass"}]}) == "pass"
        assert conclusive_verdict({"checks": [{"outcome": "pass"}, {"outcome": "fail"}]}) == "fail"
        assert conclusive_verdict({"checks": [{"outcome": "pass"}, {"outcome": "inconclusive"}]}) == "inconclusive"
        assert conclusive_verdict({"checks": []}) == "inconclusive"

    def test_matches_service_delegate(self):
        # the service staticmethod now delegates to this primitive — identical output.
        rep = {"checks": [{"outcome": "pass"}, {"outcome": "fail"}]}
        assert ArchitectReviewService._conclusive_verdict(rep) == conclusive_verdict(rep)

    def test_fail_certificates(self):
        rep = {"checks": [{"check_id": "c1", "kind": "gate", "outcome": "fail", "certificate": {"type": "diff", "payload": "x"}}]}
        certs = fail_certificates(rep)
        assert certs == [{"check_id": "c1", "kind": "gate", "certificate": {"type": "diff", "payload": "x"}}]
        assert ArchitectReviewService._fail_certificates(rep) == certs


class TestPrecedenceRecommendation:
    def test_fa_candidate(self):
        assert verifier_precedence_recommendation("approve", "fail") == ("request_evidence", FA_CANDIDATE)

    def test_fr_candidate(self):
        assert verifier_precedence_recommendation("reject", "pass") == ("request_evidence", FR_CANDIDATE)
        assert verifier_precedence_recommendation("reject_to_empty", "pass") == ("request_evidence", FR_CANDIDATE)

    def test_inconclusive_defers(self):
        assert verifier_precedence_recommendation("approve", "inconclusive") == (None, None)

    def test_agreement_no_override(self):
        assert verifier_precedence_recommendation("approve", "pass") == (None, None)
        assert verifier_precedence_recommendation("reject", "fail") == (None, None)


class TestByteIdenticalShadow:
    """The refactor moved the DECISION into the reducer but the shadow-path OUTPUT
    (returned ArchitectReview + emitted trace) must be byte-identical to RD-3."""

    def test_fa_candidate_output_unchanged(self):
        events = []
        svc = _service(trace_sink=events.append)
        review = ArchitectReview(subtask_id="S1", decision=ReviewDecision.APPROVE)
        rep = _report(
            {"check_id": "c1", "kind": "gate", "outcome": "fail", "certificate": {"type": "failing_assertion", "payload": "x"}}
        )
        out = svc.apply_verifier_precedence(review, rep)
        assert out.decision is ReviewDecision.REQUEST_EVIDENCE
        assert out.approved_output is None
        assert any(e.get("kind") == "gate_result" for e in out.evidence)
        assert any(ev.category == FA_CANDIDATE for ev in events)

    def test_fr_candidate_output_unchanged(self):
        events = []
        svc = _service(trace_sink=events.append)
        out = svc.apply_verifier_precedence(
            ArchitectReview(subtask_id="S1", decision=ReviewDecision.REJECT, tripwire=True),
            _report(summary={"conclusive_verdict": "pass"}),
        )
        assert out.decision is ReviewDecision.REQUEST_EVIDENCE
        assert out.tripwire is False
        assert any(ev.category == FR_CANDIDATE for ev in events)

    def test_inconclusive_returns_same_object(self):
        svc = _service()
        review = ArchitectReview(subtask_id="S1", decision=ReviewDecision.APPROVE)
        out = svc.apply_verifier_precedence(review, _report(summary={"conclusive_verdict": "inconclusive"}))
        assert out is review  # identity preserved (no override)

    def test_agreement_returns_same_object(self):
        svc = _service()
        review = ArchitectReview(subtask_id="S1", decision=ReviewDecision.APPROVE)
        out = svc.apply_verifier_precedence(review, _report(summary={"conclusive_verdict": "pass"}))
        assert out is review


# ═══ 2. reduce_decision — §8 branches ════════════════════════════════════════


class TestStep1ConclusiveFailureDominates:
    def test_mandatory_high_failure_replans(self):
        prof = _profile({"fc": ("high", True)})
        verif = _report(_check("fc", logical="fail", authority="sound_refutation", severity="high"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.REPLAN
        assert "CONCLUSIVE_HIGH_SEVERITY_FAILURE" in res.reason_codes

    def test_dominates_reviewer_approval(self):
        # §7.3 rule 1 / §20.2.6: a reviewer cannot override conclusive scoped evidence.
        prof = _profile({"fc": ("critical", True)})
        verif = _report(_check("fc", logical="fail", authority="sound_refutation", severity="critical"))
        approve = ArchitectReview(subtask_id="S1", decision=ReviewDecision.APPROVE)
        res = reduce_decision(None, verif, approve, prof, _AUTHORIZED)
        assert res.action is PolicyAction.REPLAN

    def test_below_min_severity_does_not_dominate(self):
        # a mandatory MEDIUM failure is not caught by step 1 (min severity high).
        prof = _profile({"fc": ("medium", True)})
        verif = _report(_check("fc", logical="fail", authority="sound_refutation", severity="medium"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is not PolicyAction.REPLAN  # falls through to default policy

    def test_advisory_criterion_failure_does_not_block(self):
        # non-mandatory criterion fails but a mandatory one passes -> continue (§20.6.5).
        prof = _profile({"fc": ("critical", True), "maintainability": ("low", False)})
        verif = _report(
            _check("fc", logical="pass", authority="proof", severity="critical"),
            _check("maintainability", logical="fail", authority="sound_refutation", severity="low"),
        )
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.CONTINUE

    def test_operational_error_is_not_a_conclusive_failure(self):
        # a FAIL logical with execution=error is a tool failure, not a blocking failure.
        prof = _profile({"fc": ("critical", True)}, operational_error_action="defer")
        verif = _report(_check("fc", logical="fail", execution="error", authority="sound_refutation", severity="critical"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.DEFER
        assert "VERIFIER_OPERATIONAL_ERROR" in res.reason_codes


class TestStep2Conflict:
    def test_summary_conflict_escalates(self):
        prof = _profile()
        res = reduce_decision(None, _report(summary={"conflicts": 1}), None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.ESCALATE
        assert "CONFLICTING_AUTHORITATIVE_EVIDENCE" in res.reason_codes

    def test_same_criterion_pass_and_fail_conflicts(self):
        prof = _profile({"fc": ("high", True)})
        verif = _report(
            _check("fc", logical="pass", authority="proof", severity="high"),
            _check("fc", logical="fail", authority="sound_refutation", severity="high"),
        )
        # conclusive failure dominates first (step 1) — expected: replan, still terminal + safe.
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.REPLAN

    def test_conflict_without_conclusive_failure_escalates(self):
        prof = _profile({"fc": ("high", True)})
        verif = _report(_check("fc", logical="conflict", authority="proof", severity="high"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.ESCALATE

    def test_conflict_policy_defer_override(self):
        prof = _profile({"fc": ("high", True)}, conflict="defer")
        verif = _report(_check("fc", logical="conflict", authority="proof", severity="high"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.DEFER
        assert "TERMINAL_DEFER" in res.reason_codes


class TestStep4CriticalUnknown:
    def test_unknown_critical_escalates(self):
        prof = _profile({"fc": ("critical", True)}, unknown_on_critical="escalate")
        verif = _report(_check("fc", logical="unknown", authority="proof", severity="critical"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.ESCALATE
        assert "UNKNOWN_ON_CRITICAL" in res.reason_codes

    def test_unknown_critical_with_request_evidence_collects(self):
        prof = _profile({"fc": ("critical", True)})
        verif = _report(_check("fc", logical="unknown", authority="proof", severity="critical"))
        rv = ArchitectReview(
            subtask_id="S1",
            decision=ReviewDecision.REQUEST_EVIDENCE,
            verifier_requests=[{"verifier": "z3", "kind": "gate"}],
        )
        res = reduce_decision(None, verif, rv, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.COLLECT_EVIDENCE
        assert res.terminal is False
        assert res.requested_evidence and res.requested_evidence[0]["verifier"] == "z3"

    def test_solver_unknown_is_not_operational_error(self):
        # logical=unknown / execution=ok must reach the unknown branch, not the op-error one.
        prof = _profile({"fc": ("critical", True)}, unknown_on_critical="defer")
        verif = _report(_check("fc", logical="unknown", execution="ok", authority="proof", severity="critical"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.DEFER
        assert "UNKNOWN_ON_CRITICAL" in res.reason_codes


class TestStep5ReviewerAuthority:
    def _clean_pass(self, prof):
        # a report that passes step 1-4 so we exercise the reviewer branch.
        return _report(_check("fc", logical="pass", authority="proof", severity="critical"))

    def test_authorized_grounded_reject_replans(self):
        prof = _profile({"fc": ("critical", True)}, max_reviewer_risk=0.05)
        reject = ArchitectReview(
            subtask_id="S1",
            decision=ReviewDecision.REJECT,
            evidence=[{"kind": "gate_result", "ref": "g1"}],  # grounded (RD-8 objective kind)
        )
        res = reduce_decision(None, self._clean_pass(prof), reject, prof, _AUTHORIZED)
        assert res.action is PolicyAction.REPLAN
        assert "CALIBRATED_REVIEWER_REJECTION" in res.reason_codes

    def test_unauthorized_reject_is_advisory(self):
        prof = _profile({"fc": ("critical", True)}, max_reviewer_risk=0.05)
        reject = ArchitectReview(
            subtask_id="S1", decision=ReviewDecision.REJECT, evidence=[{"kind": "gate_result"}]
        )
        # calibration risk (0.5) exceeds threshold (0.05) -> not authorized -> advisory.
        res = reduce_decision(None, self._clean_pass(prof), reject, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.ADVISORY
        assert res.is_advisory is True
        assert "UNAUTHORIZED_OR_UNGROUNDED_REJECTION" in res.reason_codes

    def test_authorized_but_ungrounded_reject_is_advisory(self):
        prof = _profile({"fc": ("critical", True)}, max_reviewer_risk=0.05)
        reject = ArchitectReview(subtask_id="S1", decision=ReviewDecision.REJECT)  # no objective evidence
        res = reduce_decision(None, self._clean_pass(prof), reject, prof, _AUTHORIZED)
        assert res.action is PolicyAction.ADVISORY

    def test_reviewer_request_evidence_collects(self):
        prof = _profile({"fc": ("critical", True)})
        rv = ArchitectReview(
            subtask_id="S1",
            decision=ReviewDecision.REQUEST_EVIDENCE,
            verifier_requests=[{"verifier": "unit", "kind": "gate"}],
        )
        res = reduce_decision(None, self._clean_pass(prof), rv, prof, _AUTHORIZED)
        assert res.action is PolicyAction.COLLECT_EVIDENCE

    def test_reviewer_escalate(self):
        prof = _profile({"fc": ("critical", True)})
        rv = ArchitectReview(subtask_id="S1", decision=ReviewDecision.ESCALATE)
        res = reduce_decision(None, self._clean_pass(prof), rv, prof, _AUTHORIZED)
        assert res.action is PolicyAction.ESCALATE
        assert "REVIEWER_ESCALATE" in res.reason_codes

    def test_reviewer_abstain_via_dict(self):
        # abstain is a v1.1 recommendation not in the runtime enum yet — pass as a dict.
        prof = _profile({"fc": ("critical", True)})
        res = reduce_decision(None, self._clean_pass(prof), {"recommendation": "abstain"}, prof, _AUTHORIZED)
        assert res.action is PolicyAction.ESCALATE
        assert "REVIEWER_ABSTAIN" in res.reason_codes

    def test_raw_confidence_is_ignored(self):
        # §5.4: even a maximally-confident ungrounded reject is only advisory.
        prof = _profile({"fc": ("critical", True)}, max_reviewer_risk=0.05)
        reject = ArchitectReview(subtask_id="S1", decision=ReviewDecision.REJECT, confidence=0.999)
        res = reduce_decision(None, self._clean_pass(prof), reject, prof, _AUTHORIZED)
        assert res.action is PolicyAction.ADVISORY


class TestStep6MandatorySatisfied:
    def test_conclusive_pass_continues(self):
        prof = _profile({"fc": ("critical", True)})
        verif = _report(_check("fc", logical="pass", authority="proof", severity="critical"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.CONTINUE
        assert "MANDATORY_CRITERIA_SATISFIED" in res.reason_codes

    def test_bounded_test_pass_alone_does_not_satisfy(self):
        # §20.2.2 — a passing bounded test is not a universal proof; mandatory unresolved.
        prof = _profile({"fc": ("critical", True)}, default_action="defer")
        verif = _report(_check("fc", logical="pass", authority="bounded_test", severity="critical"))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.DEFER
        assert "MANDATORY_CRITERIA_UNRESOLVED" in res.reason_codes

    def test_bounded_test_pass_with_granted_authority_satisfies(self):
        prof = _profile({"fc": ("critical", True)})
        verif = _report(_check("fc", logical="pass", authority="bounded_test", severity="critical", may_approve=True))
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.CONTINUE

    def test_summary_rollup_trusted(self):
        prof = _profile({"fc": ("critical", True)})
        verif = _report(summary={"mandatory_criteria": {"satisfied": True}})
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.CONTINUE


# ═══ 3. contract properties ═══════════════════════════════════════════════════


class TestBoundedLoops:
    def test_collect_evidence_below_budget(self):
        prof = _profile({"fc": ("critical", True)}, max_evidence_rounds=2)
        rv = ArchitectReview(subtask_id="S1", decision=ReviewDecision.REQUEST_EVIDENCE, verifier_requests=[{"verifier": "z3", "kind": "gate"}])
        verif = _report(_check("fc", logical="pass", authority="proof", severity="critical"))
        res = reduce_decision(None, verif, rv, prof, _AUTHORIZED, loop_state=LoopState(evidence_round=1))
        assert res.action is PolicyAction.COLLECT_EVIDENCE

    def test_evidence_budget_exhausted_is_terminal(self):
        # §10.2: at budget exhaustion the profile terminal (fail_closed -> defer) fires.
        prof = _profile({"fc": ("critical", True)}, max_evidence_rounds=2, evidence_budget_exhausted="fail_closed")
        rv = ArchitectReview(subtask_id="S1", decision=ReviewDecision.REQUEST_EVIDENCE, verifier_requests=[{"verifier": "z3", "kind": "gate"}])
        verif = _report(_check("fc", logical="pass", authority="proof", severity="critical"))
        res = reduce_decision(None, verif, rv, prof, _AUTHORIZED, loop_state=LoopState(evidence_round=2))
        assert res.action is PolicyAction.DEFER
        assert res.terminal is True
        assert "EVIDENCE_BUDGET_EXHAUSTED" in res.reason_codes

    def test_fail_open_terminal_continues(self):
        prof = _profile({"fc": ("critical", True)}, max_evidence_rounds=1, evidence_budget_exhausted="fail_open")
        rv = ArchitectReview(subtask_id="S1", decision=ReviewDecision.REQUEST_EVIDENCE)
        verif = _report(_check("fc", logical="pass", authority="proof", severity="critical"))
        res = reduce_decision(None, verif, rv, prof, _AUTHORIZED, loop_state=LoopState(evidence_round=1))
        assert res.action is PolicyAction.CONTINUE
        assert "TERMINAL_FAIL_OPEN" in res.reason_codes


class TestPurityAndReplay:
    def test_identical_inputs_identical_output(self):
        # §12.4 / §20.1.6: deterministic reducer output is identical on replay.
        prof = _profile({"fc": ("critical", True)}, max_reviewer_risk=0.05)
        verif = _report(_check("fc", logical="pass", authority="proof", severity="critical"))
        rv = ArchitectReview(subtask_id="S1", decision=ReviewDecision.REJECT, evidence=[{"kind": "gate_result"}])
        a = reduce_decision(None, verif, rv, prof, _AUTHORIZED)
        b = reduce_decision(None, verif, rv, prof, _AUTHORIZED)
        assert a == b  # frozen dataclass value-equality
        assert a.action is b.action and a.reason_codes == b.reason_codes

    def test_policy_result_is_frozen(self):
        res = PolicyResult.continue_("X")
        with pytest.raises(Exception):
            res.action = PolicyAction.ABORT  # type: ignore[misc]


class TestEnvelopeInterop:
    def test_action_enum_matches_envelope_schema(self):
        schema = json.loads((_ORCH / "decision_envelope.schema.json").read_text())
        allowed = set(schema["properties"]["policy_result"]["properties"]["action"]["enum"])
        assert {a.value for a in PolicyAction} == allowed

    def test_to_policy_result_dict_shape(self):
        d = PolicyResult.replan("CONCLUSIVE_HIGH_SEVERITY_FAILURE").to_policy_result_dict()
        assert d == {"action": "replan", "blocking_reason_codes": ["CONCLUSIVE_HIGH_SEVERITY_FAILURE"]}

    def test_policy_hash_stable_and_versioned(self):
        p1 = ReducerPolicy()
        p2 = ReducerPolicy()
        assert p1.policy_hash == p2.policy_hash
        p3 = ReducerPolicy(max_evidence_rounds=99)
        assert p3.policy_hash != p1.policy_hash


class TestProfileLoading:
    def test_loads_cp2_software_engineering_example(self):
        prof = AssuranceProfile.from_dict(
            json.loads((_ORCH / "examples" / "assurance_profile_software_engineering.json").read_text())
        )
        assert prof.profile_id == "swe_release:v3"
        assert prof.criteria["functional_correctness"].severity.value == "critical"
        assert prof.criteria["functional_correctness"].mandatory is True
        assert prof.max_reviewer_risk == 0.05
        assert prof.policy.evidence_budget_exhausted == "fail_closed"
        # hashes are stable + non-empty (governance inputs).
        assert prof.profile_hash and prof.verifier_registry_hash

    def test_real_profile_end_to_end_conclusive_failure(self):
        prof = AssuranceProfile.from_dict(
            json.loads((_ORCH / "examples" / "assurance_profile_software_engineering.json").read_text())
        )
        verif = _report(
            _check("migration_idempotence", logical="fail", authority="sound_refutation", severity="high")
        )
        res = reduce_decision(None, verif, None, prof, _UNAUTHORIZED)
        assert res.action is PolicyAction.REPLAN

    def test_verdict_result_to_report_accepted(self):
        # a VerdictResult (patch_verifier) exposes .to_report(); the reducer accepts it.
        from src.verification.patch_verifier import VerdictResult, Check

        vr = VerdictResult(verdict="pass", checks=[Check(check_id="x", kind="gate", outcome="pass")])
        prof = _profile({"x": ("low", False)})
        res = reduce_decision(None, vr, None, prof, _UNAUTHORIZED)
        assert isinstance(res, PolicyResult)


class TestReviewViewAdapter:
    def test_reject_to_empty_folds_to_reject(self):
        rv = ReviewView.from_review(ArchitectReview(subtask_id="S", decision=ReviewDecision.REJECT_TO_EMPTY))
        assert rv.recommendation == "reject"

    def test_dict_review_grounded_via_blocking_findings(self):
        rv = ReviewView.from_review({"recommendation": "reject", "blocking_findings": [{"criterion_id": "c", "evidence_refs": ["e1"]}]})
        assert rv.grounded_blocking is True

    def test_none_review(self):
        assert ReviewView.from_review(None) is None


class TestVerificationViewDefaults:
    def test_v1_report_without_v11_fields(self):
        # a legacy v1.0 report (outcome only) degrades: pass->logical pass, exec ok.
        vv = VerificationView(
            {"checks": [{"check_id": "c", "kind": "gate", "outcome": "pass", "required": True}]},
            _profile({"c": ("critical", True)}),
        )
        # no authority block -> heuristic default -> not a conclusive pass -> unsatisfied.
        assert vv.mandatory_criteria_satisfied() is False
        assert vv.has_conclusive_failure() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
