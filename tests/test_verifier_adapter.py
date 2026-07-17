"""Tests for verifier_adapter.py (RD-4) + config/gates.yaml parity.

Zero inference. Gate execution uses a *test* gates.yaml whose commands are the
trivial no-ops ``true`` / ``false`` (hermetic, sub-millisecond) — we never run the
real ``make lint`` here. Formalizer tests are pure. The parity test proves that
introducing ``config/gates.yaml`` does NOT change GateRunner's effective default
behaviour.
"""

from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.gate_runner import GateRunner
from src.proactive_delegation.verifier_adapter import (
    FormalizerRegistry,
    InvariantAssertionFormalizer,
    JsonSchemaFormalizer,
    NumericAnswerFormalizer,
    RegexConstraintFormalizer,
    RetrievalGroundingFormalizer,
    default_registry,
    hypothesis_formalizer,
    run_verifier_requests,
    souffle_formalizer,
    z3_formalizer,
    _load_report_schema,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATES_YAML = _REPO_ROOT / "config" / "gates.yaml"


def _valid_report(report):
    errors = list(Draft202012Validator(_load_report_schema()).iter_errors(report))
    assert errors == [], errors


# ── config/gates.yaml parity (must not change default behaviour) ───────


class TestGatesYamlParity:
    _FIELDS = ("command", "timeout", "required", "retry_count", "description", "parallelizable")

    def test_gates_yaml_exists(self):
        assert _GATES_YAML.exists(), "config/gates.yaml must exist for RD-4"

    def test_yaml_reproduces_in_code_defaults(self, tmp_path):
        # No config file → GateRunner falls back to _get_default_gates().
        default_runner = GateRunner(config_path=tmp_path / "nonexistent.yaml")
        yaml_runner = GateRunner(config_path=_GATES_YAML)

        d = {g.name: g for g in default_runner.gates}
        y = {g.name: g for g in yaml_runner.gates}
        assert set(d) == set(y), "gate NAME set differs from in-code defaults"
        assert default_runner.get_gate_names() == yaml_runner.get_gate_names(), "gate ORDER differs"
        for name in d:
            for f in self._FIELDS:
                assert getattr(d[name], f) == getattr(y[name], f), (
                    f"gate {name!r} field {f!r} differs: "
                    f"default={getattr(d[name], f)!r} yaml={getattr(y[name], f)!r}"
                )


# ── gate_runner bridge (trivial no-op commands) ───────────────────────


@pytest.fixture
def noop_gates(tmp_path):
    """A gates.yaml whose commands are hermetic no-ops (true=pass, false=fail)."""
    cfg = tmp_path / "gates.yaml"
    cfg.write_text(
        "gates:\n"
        "  - name: unit\n    command: 'true'\n    timeout: 5\n    required: true\n"
        "  - name: lint\n    command: 'false'\n    timeout: 5\n    required: true\n",
        encoding="utf-8",
    )
    return GateRunner(config_path=cfg, working_dir=tmp_path)


class TestGateBridge:
    def test_passing_gate_maps_to_pass(self, noop_gates):
        report = run_verifier_requests(
            [{"verifier": "unit", "kind": "gate"}], "cand", "code", gate_runner=noop_gates
        )
        _valid_report(report)
        check = report["checks"][0]
        assert check["outcome"] == "pass"
        assert check["kind"] == "gate"
        assert report["summary"]["conclusive_verdict"] == "pass"

    def test_failing_gate_maps_to_fail_with_certificate(self, noop_gates):
        report = run_verifier_requests(
            [{"verifier": "lint", "kind": "lint"}], "cand", "code", gate_runner=noop_gates
        )
        _valid_report(report)
        check = report["checks"][0]
        assert check["outcome"] == "fail"
        # a FAIL must carry a certificate (the request_evidence payload)
        assert check["certificate"]["type"] == "failing_assertion"
        assert report["summary"]["conclusive_verdict"] == "fail"

    def test_unknown_gate_is_inconclusive(self, noop_gates):
        report = run_verifier_requests(
            [{"verifier": "does_not_exist", "kind": "gate"}], "cand", "code", gate_runner=noop_gates
        )
        _valid_report(report)
        check = report["checks"][0]
        assert check["outcome"] == "inconclusive"
        assert "unknown gate" in check["inconclusive_reason"].lower()

    def test_fail_dominates_aggregate(self, noop_gates):
        report = run_verifier_requests(
            [{"verifier": "unit", "kind": "gate"}, {"verifier": "lint", "kind": "gate"}],
            "cand",
            "code",
            gate_runner=noop_gates,
        )
        _valid_report(report)
        assert report["summary"]["passed"] == 1
        assert report["summary"]["failed"] == 1
        assert report["summary"]["conclusive_verdict"] == "fail"


# ── Tier 1 formalizers ────────────────────────────────────────────────


class TestJsonSchemaFormalizer:
    _SCHEMA = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}

    def test_pass(self):
        f = JsonSchemaFormalizer()
        r = f.check({"schema": self._SCHEMA}, {"x": 3}, "code")
        assert r.outcome == "pass"

    def test_fail_has_constraint_certificate(self):
        f = JsonSchemaFormalizer()
        r = f.check({"schema": self._SCHEMA}, {"x": "nope"}, "code")
        assert r.outcome == "fail"
        assert r.certificate["type"] == "constraint_violation"

    def test_no_schema_is_inconclusive(self):
        r = JsonSchemaFormalizer().check({}, {"x": 1}, "code")
        assert r.outcome == "inconclusive"


class TestInvariantFormalizer:
    def test_registered_predicate_pass_fail(self):
        f = InvariantAssertionFormalizer()
        f.register("nonempty", lambda cand, req, dom: (bool(cand), "candidate empty"))
        assert f.check({"invariant": "nonempty"}, "x", "code").outcome == "pass"
        r = f.check({"invariant": "nonempty"}, "", "code")
        assert r.outcome == "fail"
        assert r.certificate["type"] == "failing_assertion"

    def test_unknown_predicate_inconclusive(self):
        r = InvariantAssertionFormalizer().check({"invariant": "missing"}, "x", "code")
        assert r.outcome == "inconclusive"

    def test_raising_predicate_is_inconclusive_not_fail(self):
        f = InvariantAssertionFormalizer()

        def _boom(cand, req, dom):
            raise RuntimeError("boom")

        f.register("boom", _boom)
        r = f.check({"invariant": "boom"}, "x", "code")
        assert r.outcome == "inconclusive"
        assert "boom" in r.inconclusive_reason.lower()


class TestRegexConstraintFormalizer:
    def test_must_contain_pass_and_fail(self):
        f = RegexConstraintFormalizer()
        assert f.check({"must_contain": "TODO"}, "has TODO here", "code").outcome == "pass"
        r = f.check({"must_contain": "TODO"}, "no marker", "code")
        assert r.outcome == "fail"
        assert r.certificate["type"] == "constraint_violation"

    def test_max_words(self):
        f = RegexConstraintFormalizer()
        assert f.check({"max_words": 3}, "one two", "general").outcome == "pass"
        assert f.check({"max_words": 3}, "one two three four", "general").outcome == "fail"

    def test_multiple_constraints_all_required(self):
        f = RegexConstraintFormalizer()
        req = {"must_contain": "A", "must_not_contain": "B"}
        assert f.check(req, "A only", "general").outcome == "pass"
        assert f.check(req, "A and B", "general").outcome == "fail"

    def test_no_constraints_inconclusive(self):
        assert RegexConstraintFormalizer().check({}, "x", "general").outcome == "inconclusive"


class TestNumericAnswerFormalizer:
    def test_correct_answer_pass(self):
        r = NumericAnswerFormalizer().check({"expected": 42}, "the answer is 42", "math")
        assert r.outcome == "pass"

    def test_wrong_answer_counterexample(self):
        r = NumericAnswerFormalizer().check({"expected": 42}, "the answer is 41", "math")
        assert r.outcome == "fail"
        assert r.certificate["type"] == "counterexample"
        assert r.certificate["payload"]["expected"] == 42.0

    def test_no_expected_inconclusive(self):
        assert NumericAnswerFormalizer().check({}, "42", "math").outcome == "inconclusive"

    def test_no_number_inconclusive(self):
        r = NumericAnswerFormalizer().check({"expected": 1}, "no digits here", "math")
        assert r.outcome == "inconclusive"


class TestRetrievalGroundingFormalizer:
    def test_grounded_spans_pass(self):
        f = RetrievalGroundingFormalizer()
        r = f.check(
            {"sources": ["the sky is blue"], "spans": ["sky is blue"]}, "cand", "qa"
        )
        assert r.outcome == "pass"

    def test_ungrounded_span_fail(self):
        f = RetrievalGroundingFormalizer()
        r = f.check({"sources": ["the sky is blue"], "spans": ["grass is purple"]}, "cand", "qa")
        assert r.outcome == "fail"
        assert "ungrounded_spans" in r.certificate["payload"]

    def test_no_sources_inconclusive(self):
        assert RetrievalGroundingFormalizer().check({"spans": ["x"]}, "c", "qa").outcome == "inconclusive"

    def test_extracts_quoted_spans_from_candidate(self):
        f = RetrievalGroundingFormalizer()
        r = f.check({"sources": ["alpha beta gamma"]}, 'I cite "beta gamma" here', "qa")
        assert r.outcome == "pass"


# ── Tier 2 stubs degrade gracefully ───────────────────────────────────


class TestTier2Stubs:
    def test_stubs_return_inconclusive(self):
        for factory in (hypothesis_formalizer, souffle_formalizer, z3_formalizer):
            f = factory()
            r = f.check({}, "candidate", "code")
            assert r.outcome == "inconclusive"
            reason = r.inconclusive_reason
            assert ("not_installed" in reason) or ("not_implemented" in reason)

    def test_souffle_binary_absent_says_not_installed(self):
        # souffle binary is not on this host → not_installed (never fabricated pass/fail)
        r = souffle_formalizer().check({}, "c", "logic")
        assert r.outcome == "inconclusive"
        assert "not_installed" in r.inconclusive_reason


# ── registry ──────────────────────────────────────────────────────────


class TestRegistry:
    def test_default_registry_contents(self):
        reg = default_registry()
        names = reg.names()
        # Tier 1 implemented
        for n in ("jsonschema", "invariant", "regex_constraint", "numeric_answer", "retrieval_grounding"):
            assert n in names
        # Tier 2 stubs
        for n in ("hypothesis", "souffle", "z3"):
            assert n in names

    def test_for_request_resolves_by_name_then_kind(self):
        reg = default_registry()
        by_name = reg.for_request({"verifier": "numeric_answer", "kind": "math_check"})
        assert by_name.name == "numeric_answer"
        by_kind = reg.for_request({"verifier": "unregistered", "kind": "math_check"})
        assert by_kind.name == "numeric_answer"  # resolved by kind fallback

    def test_invariant_predicates_injected_into_default_registry(self):
        reg = default_registry(invariant_predicates={"ok": lambda c, r, d: (True, "")})
        report = run_verifier_requests(
            [{"verifier": "invariant", "kind": "constraint_check", "invariant": "ok"}],
            "cand",
            "code",
            registry=reg,
        )
        _valid_report(report)
        assert report["checks"][0]["outcome"] == "pass"


# ── run_verifier_requests end-to-end shape ────────────────────────────


class TestRunVerifierRequests:
    def test_report_is_schema_valid_and_has_ids(self):
        report = run_verifier_requests(
            [{"verifier": "numeric_answer", "kind": "math_check", "expected": 7}],
            "answer: 7",
            "math",
        )
        _valid_report(report)
        assert report["report_id"]
        assert report["schema_version"] == "1.0.0"
        assert report["summary"]["conclusive_verdict"] == "pass"

    def test_no_requests_yields_inconclusive_noop(self):
        report = run_verifier_requests([], "cand", "code")
        _valid_report(report)
        assert report["checks"][0]["outcome"] == "inconclusive"
        assert report["summary"]["conclusive_verdict"] == "inconclusive"

    def test_unroutable_formalizer_request_inconclusive(self):
        report = run_verifier_requests(
            [{"verifier": "nope", "kind": "scorer"}], "cand", "code"
        )
        _valid_report(report)
        assert report["checks"][0]["outcome"] == "inconclusive"

    def test_candidate_ref_propagates(self):
        report = run_verifier_requests(
            [{"verifier": "numeric_answer", "kind": "math_check", "expected": 1}],
            "1",
            "math",
            candidate_ref="pkg-123",
        )
        _valid_report(report)
        assert report["candidate_ref"] == "pkg-123"

    def test_empty_registry_reports_no_formalizer(self):
        report = run_verifier_requests(
            [{"verifier": "x", "kind": "scorer"}],
            "cand",
            "code",
            registry=FormalizerRegistry(),
        )
        _valid_report(report)
        assert report["checks"][0]["outcome"] == "inconclusive"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
