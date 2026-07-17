"""Tests for the criterion-scoped evidence-authority model (CP1, spec §5.2 / §7).

Zero inference. Pure data-model unit tests over authority.py:
  * §7.1 approve/block grant table for all nine classes;
  * §7.2 logical_status vs execution_status kept SEPARATE (tool failure != unknown);
  * anti-authority-laundering: explicit may_* may narrow a YES / set a POLICY class
    but can NEVER widen a NO class (§13.2 / §7.3 rule 3);
  * conclusive-failure / conclusive-pass primitives (§7.3 building blocks);
  * criterion scoping (valid_for).
"""

from __future__ import annotations

import pytest

from src.proactive_delegation.authority import (
    Authority,
    AuthorityClass,
    ExecutionStatus,
    Grant,
    LogicalStatus,
    Severity,
    coerce_authority_class,
    coerce_execution,
    coerce_logical,
    coerce_severity,
    is_conclusive_failure,
    is_conclusive_pass,
    is_operational_error,
    severity_at_least,
)


def _auth(cls, **kw):
    return Authority(cls=cls, **kw)


# ═══ §7.1 authority-class grant table ════════════════════════════════════════


class TestAuthorityTable:
    @pytest.mark.parametrize(
        "cls, approve, block",
        [
            (AuthorityClass.PROOF, True, True),
            (AuthorityClass.COMPLETE_DECIDER, True, True),
            (AuthorityClass.SOUND_REFUTATION, False, True),  # a pass proves nothing
            (AuthorityClass.SOUND_ACCEPTANCE, True, False),  # a fail may be incomplete
            (AuthorityClass.HEURISTIC_STATIC, False, False),
        ],
    )
    def test_fixed_grants(self, cls, approve, block):
        a = _auth(cls)
        assert a.may_approve() is approve
        assert a.may_block() is block

    @pytest.mark.parametrize(
        "cls",
        [
            AuthorityClass.BOUNDED_TEST,
            AuthorityClass.STATISTICAL_EVIDENCE,
            AuthorityClass.LLM_JUDGMENT,
            AuthorityClass.HUMAN_ATTESTATION,
        ],
    )
    def test_policy_dependent_default_off_grant_on(self, cls):
        a = _auth(cls)
        # policy-dependent classes default to NEITHER until policy grants.
        assert a.may_approve() is False
        assert a.may_block() is False
        assert a.may_approve(policy_grant=True) is True
        assert a.may_block(policy_grant=True) is True

    def test_llm_judgment_needs_calibration_grant(self):
        """§7.1: llm_judgment may block/approve only after calibration + policy grant."""
        a = _auth(AuthorityClass.LLM_JUDGMENT)
        assert a.may_block() is False  # raw model verdict never blocks on its own
        assert a.may_block(policy_grant=True) is True


class TestExplicitOverrideTrustModel:
    def test_explicit_narrows_yes_class(self):
        # a proof whose producer marks may_block=False loses blocking (narrow-only).
        a = _auth(AuthorityClass.PROOF, explicit_may_block=False)
        assert a.may_block() is False
        assert a.may_approve() is True  # untouched

    def test_explicit_sets_policy_class(self):
        # a trusted producer declaring a bounded_test may block -> honored.
        a = _auth(AuthorityClass.BOUNDED_TEST, explicit_may_block=True)
        assert a.may_block() is True
        assert a.may_approve() is False  # not declared -> policy default off

    def test_explicit_cannot_widen_no_class(self):
        # anti-laundering: a heuristic claiming may_block=True still cannot block.
        a = _auth(AuthorityClass.HEURISTIC_STATIC, explicit_may_block=True)
        assert a.may_block() is False
        assert a.may_block(policy_grant=True) is False
        # sound_refutation claiming approve authority still cannot approve.
        r = _auth(AuthorityClass.SOUND_REFUTATION, explicit_may_approve=True)
        assert r.may_approve() is False


class TestCriterionScoping:
    def test_scopes_own_criterion_by_default(self):
        a = _auth(AuthorityClass.PROOF)
        assert a.scopes_criterion("anything") is True

    def test_valid_for_restricts_scope(self):
        a = _auth(AuthorityClass.PROOF, valid_for=("c1", "c2"))
        assert a.scopes_criterion("c1") is True
        assert a.scopes_criterion("c3") is False

    def test_from_dict_parses_all_fields(self):
        a = Authority.from_dict(
            {"class": "bounded_test", "valid_for": ["x"], "may_block": True, "may_approve": False}
        )
        assert a.cls is AuthorityClass.BOUNDED_TEST
        assert a.valid_for == ("x",)
        assert a.may_block() is True
        assert a.may_approve() is False

    def test_unknown_class_defaults_to_weakest(self):
        # unclassified/absent authority -> heuristic_static (neither approve nor block).
        assert coerce_authority_class("nonsense") is AuthorityClass.HEURISTIC_STATIC
        a = Authority.from_dict({})
        assert a.may_approve() is False and a.may_block() is False


# ═══ §7.2 logical vs execution status kept separate ══════════════════════════


class TestLogicalVsExecution:
    def test_solver_unknown_is_epistemic(self):
        # logical=unknown / execution=ok — an honest "don't know", NOT a tool failure.
        assert is_operational_error(ExecutionStatus.OK) is False
        assert (
            is_conclusive_failure(
                _auth(AuthorityClass.PROOF), LogicalStatus.UNKNOWN, ExecutionStatus.OK
            )
            is False
        )

    def test_crash_is_operational_not_failure(self):
        # verifier crash: logical=unknown / execution=error — never proof of failure.
        for exec_bad in (ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT, ExecutionStatus.UNAVAILABLE):
            assert is_operational_error(exec_bad) is True
            # even a FAIL logical does NOT count as a conclusive failure if execution failed.
            assert (
                is_conclusive_failure(
                    _auth(AuthorityClass.SOUND_REFUTATION), LogicalStatus.FAIL, exec_bad
                )
                is False
            )

    def test_counterexample_is_conclusive_failure(self):
        # property test finds a counterexample: logical=fail / execution=ok, sound_refutation.
        assert (
            is_conclusive_failure(
                _auth(AuthorityClass.SOUND_REFUTATION), LogicalStatus.FAIL, ExecutionStatus.OK
            )
            is True
        )

    def test_conflict_is_its_own_logical_state(self):
        assert coerce_logical("conflict") is LogicalStatus.CONFLICT
        # a conflict is neither a conclusive pass nor a conclusive failure.
        assert (
            is_conclusive_failure(
                _auth(AuthorityClass.PROOF), LogicalStatus.CONFLICT, ExecutionStatus.OK
            )
            is False
        )


class TestConclusivePass:
    def test_sound_acceptance_pass_is_conclusive(self):
        assert (
            is_conclusive_pass(
                _auth(AuthorityClass.SOUND_ACCEPTANCE), LogicalStatus.PASS, ExecutionStatus.OK
            )
            is True
        )

    def test_sound_refutation_pass_proves_nothing(self):
        # §7.1: sound_refutation approve=No — a passing run cannot approve.
        assert (
            is_conclusive_pass(
                _auth(AuthorityClass.SOUND_REFUTATION), LogicalStatus.PASS, ExecutionStatus.OK
            )
            is False
        )

    def test_bounded_test_pass_needs_policy_grant(self):
        # §20.2.2: a passing bounded test does not claim universal proof by default.
        a = _auth(AuthorityClass.BOUNDED_TEST)
        assert is_conclusive_pass(a, LogicalStatus.PASS, ExecutionStatus.OK) is False
        assert (
            is_conclusive_pass(a, LogicalStatus.PASS, ExecutionStatus.OK, policy_grant=True) is True
        )


# ═══ severity ordering + coercion ════════════════════════════════════════════


class TestSeverity:
    def test_ordering(self):
        assert severity_at_least(Severity.CRITICAL, Severity.HIGH) is True
        assert severity_at_least(Severity.HIGH, Severity.HIGH) is True
        assert severity_at_least(Severity.MEDIUM, Severity.HIGH) is False
        assert severity_at_least(Severity.LOW, Severity.MEDIUM) is False

    def test_coerce_defaults(self):
        assert coerce_severity("critical") is Severity.CRITICAL
        assert coerce_severity(None) is Severity.MEDIUM
        assert coerce_severity("bogus", Severity.LOW) is Severity.LOW

    def test_coerce_logical_execution(self):
        assert coerce_logical("fail") is LogicalStatus.FAIL
        assert coerce_logical("bogus") is LogicalStatus.UNKNOWN  # fail-safe to unknown
        assert coerce_execution("timeout") is ExecutionStatus.TIMEOUT
        assert coerce_execution("bogus") is ExecutionStatus.OK  # absent -> ok


class TestGrantEnum:
    def test_grant_values(self):
        assert {g.value for g in Grant} == {"yes", "no", "policy"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
