"""Tests for the reviewer control-plane typed artifacts (H2 / RA-1..RA-6, RA-11).

Covers:
  * RA-1..RA-4 schemas validated through orchestration/validate_ir.py (same
    subprocess convention as tests/unit/test_validate_ir.py).
  * RA-5 validator wiring (new `review|candidate|verification|rubric` kinds).
  * RA-6 types.py evolution (new enum members + extended ArchitectReview, backward
    compat preserved).
  * RA-11 non-breaking IR hardening (architecture_ir interfaces section, task_ir
    typed call_edges) — new fields optional, existing docs still valid.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_PATH = REPO_ROOT / "orchestration" / "validate_ir.py"


def run_validator(kind: str, doc: dict) -> tuple[int, str, str]:
    """Run validate_ir.py against a doc via stdin; return (code, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), kind, "-"],
        input=json.dumps(doc),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


# ── Fixtures: minimal valid instances ─────────────────────────────────


@pytest.fixture
def valid_review_decision() -> dict:
    return {
        "decision": "approve",
        "confidence": 0.9,
        "blocking": {"tripwire": False, "blocking_issues": []},
        "advisory": {"score": 0.85, "feedback": "meets acceptance checks"},
        "evidence": [
            {"kind": "test_result", "ref": "unit::test_foo", "summary": "passed"},
            {"kind": "answer_span", "locator": {"char_start": 0, "char_end": 42}},
        ],
        "verifier_requests": [],
        "human_review_required": False,
    }


@pytest.fixture
def valid_candidate_package() -> dict:
    return {
        "schema_version": "1.0.0",
        "package_id": "cp-001",
        "task_ref": "task-001",
        "provenance": {"model": "qwen3.5-122b", "quant": "UD-IQ2_M", "role": "coder"},
        "sanitized_view": {
            "task_ref": "task-001",
            "outputs": [{"type": "answer", "ref": "ans-1"}],
            "sanitization": {"applied": True, "removed_fields": ["author_self_assessment"]},
        },
    }


@pytest.fixture
def valid_verification_report() -> dict:
    return {
        "schema_version": "1.0.0",
        "report_id": "vr-001",
        "checks": [
            {"check_id": "format", "kind": "gate", "outcome": "pass"},
            {
                "check_id": "unit",
                "kind": "test",
                "outcome": "fail",
                "certificate": {"type": "failing_assertion", "payload": "assert x == 1"},
            },
            {
                "check_id": "z3-solver",
                "kind": "constraint_check",
                "outcome": "inconclusive",
                "inconclusive_reason": "solver timeout",
            },
        ],
    }


@pytest.fixture
def valid_review_rubric() -> dict:
    return {
        "schema_version": "1.0.0",
        "rubric_id": "code-review-v1",
        "version": "1.0.0",
        "domain": "code",
        "items": [
            {"id": "R1", "text": "Does the change match the spec?", "axis": "spec-alignment", "weight": 3},
            {"id": "R2", "text": "Are there obvious runtime errors?", "axis": "runtime", "weight": 2},
        ],
    }


# ── RA-1: review_decision ─────────────────────────────────────────────


class TestReviewDecisionSchema:
    def test_valid(self, valid_review_decision):
        code, out, _ = run_validator("review", valid_review_decision)
        assert code == 0, out

    def test_all_decision_enum_values(self, valid_review_decision):
        for d in ["approve", "reject", "reject_to_empty", "request_changes", "request_evidence", "escalate"]:
            valid_review_decision["decision"] = d
            code, out, _ = run_validator("review", valid_review_decision)
            assert code == 0, f"{d}: {out}"

    def test_bad_decision_enum(self, valid_review_decision):
        valid_review_decision["decision"] = "maybe"
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    def test_confidence_out_of_range(self, valid_review_decision):
        valid_review_decision["confidence"] = 1.5
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    def test_missing_blocking_channel(self, valid_review_decision):
        del valid_review_decision["blocking"]
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    def test_blocking_requires_tripwire(self, valid_review_decision):
        valid_review_decision["blocking"] = {"blocking_issues": []}
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    def test_bad_evidence_kind(self, valid_review_decision):
        valid_review_decision["evidence"] = [{"kind": "vibes"}]
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    def test_scorer_and_retrieval_evidence_kinds(self, valid_review_decision):
        valid_review_decision["evidence"] = [
            {"kind": "scorer_result", "ref": "grounding=0.9"},
            {"kind": "retrieval_provenance", "locator": {"doc_id": "d1"}},
            {"kind": "protocol_claim", "ref": "protocol-42"},
        ]
        code, out, _ = run_validator("review", valid_review_decision)
        assert code == 0, out

    def test_verifier_request(self, valid_review_decision):
        valid_review_decision["decision"] = "request_evidence"
        valid_review_decision["verifier_requests"] = [
            {"verifier": "unit", "kind": "test", "target": "ans-1"}
        ]
        code, out, _ = run_validator("review", valid_review_decision)
        assert code == 0, out

    def test_additional_properties_rejected(self, valid_review_decision):
        valid_review_decision["extra"] = 1
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    def test_advisory_optional(self, valid_review_decision):
        del valid_review_decision["advisory"]
        code, out, _ = run_validator("review", valid_review_decision)
        assert code == 0, out


# ── RA-2: candidate_package ───────────────────────────────────────────


class TestCandidatePackageSchema:
    def test_valid(self, valid_candidate_package):
        code, out, _ = run_validator("candidate", valid_candidate_package)
        assert code == 0, out

    def test_missing_sanitized_view(self, valid_candidate_package):
        del valid_candidate_package["sanitized_view"]
        code, _, _ = run_validator("candidate", valid_candidate_package)
        assert code == 2

    def test_provenance_requires_model(self, valid_candidate_package):
        del valid_candidate_package["provenance"]["model"]
        code, _, _ = run_validator("candidate", valid_candidate_package)
        assert code == 2

    def test_full_package_may_carry_author_fields(self, valid_candidate_package):
        # The full package retains author self-assessment for audit; sanitized_view excludes it.
        valid_candidate_package["author_self_assessment"] = "I am very confident this is correct"
        valid_candidate_package["quality_labels"] = ["refined", "final"]
        code, out, _ = run_validator("candidate", valid_candidate_package)
        assert code == 0, out

    def test_sanitized_view_rejects_author_fields(self, valid_candidate_package):
        # sanitized_view has additionalProperties:false, so leaking author fields fails.
        valid_candidate_package["sanitized_view"]["author_self_assessment"] = "leak"
        code, _, _ = run_validator("candidate", valid_candidate_package)
        assert code == 2

    def test_sanitization_requires_applied(self, valid_candidate_package):
        valid_candidate_package["sanitized_view"]["sanitization"] = {}
        code, _, _ = run_validator("candidate", valid_candidate_package)
        assert code == 2


# ── RA-3: verification_report ─────────────────────────────────────────


class TestVerificationReportSchema:
    def test_valid_three_valued(self, valid_verification_report):
        code, out, _ = run_validator("verification", valid_verification_report)
        assert code == 0, out

    def test_fail_requires_certificate(self, valid_verification_report):
        del valid_verification_report["checks"][1]["certificate"]
        code, _, _ = run_validator("verification", valid_verification_report)
        assert code == 2

    def test_inconclusive_requires_reason(self, valid_verification_report):
        del valid_verification_report["checks"][2]["inconclusive_reason"]
        code, _, _ = run_validator("verification", valid_verification_report)
        assert code == 2

    def test_bad_outcome_enum(self, valid_verification_report):
        valid_verification_report["checks"][0]["outcome"] = "maybe"
        code, _, _ = run_validator("verification", valid_verification_report)
        assert code == 2

    def test_instrument_version(self, valid_verification_report):
        valid_verification_report["checks"][0]["instrument"] = {"name": "ruff", "version": "0.4.2"}
        code, out, _ = run_validator("verification", valid_verification_report)
        assert code == 0, out

    def test_pass_check_needs_no_certificate(self, valid_verification_report):
        # A pass check without a certificate is fine.
        assert "certificate" not in valid_verification_report["checks"][0]
        code, out, _ = run_validator("verification", valid_verification_report)
        assert code == 0, out


# ── RA-4: review_rubric ───────────────────────────────────────────────


class TestReviewRubricSchema:
    def test_valid(self, valid_review_rubric):
        code, out, _ = run_validator("rubric", valid_review_rubric)
        assert code == 0, out

    def test_weight_must_be_1_2_3(self, valid_review_rubric):
        for bad in [0, 4, 2.5]:
            valid_review_rubric["items"][0]["weight"] = bad
            code, _, _ = run_validator("rubric", valid_review_rubric)
            assert code == 2, f"weight={bad} should fail"

    def test_item_requires_axis(self, valid_review_rubric):
        del valid_review_rubric["items"][0]["axis"]
        code, _, _ = run_validator("rubric", valid_review_rubric)
        assert code == 2

    def test_bad_domain(self, valid_review_rubric):
        valid_review_rubric["domain"] = "astrology"
        code, _, _ = run_validator("rubric", valid_review_rubric)
        assert code == 2

    def test_item_id_pattern(self, valid_review_rubric):
        valid_review_rubric["items"][0]["id"] = "item1"
        code, _, _ = run_validator("rubric", valid_review_rubric)
        assert code == 2


# ── RA-5: validator wiring ────────────────────────────────────────────


class TestValidatorWiring:
    def test_unknown_kind_lists_new_kinds(self):
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "bogus", "-"],
            input="{}",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        for kind in ["review", "candidate", "verification", "rubric"]:
            assert kind in result.stdout


# ── RA-11: non-breaking IR hardening ──────────────────────────────────


@pytest.fixture
def base_architecture_ir() -> dict:
    return {
        "name": "test-project",
        "version": "1.0.0",
        "goals": ["Build a test system"],
        "non_goals": [],
        "global_invariants": [],
        "repo_layout": {"folders": [{"path": "src/", "owner_role": "coder", "purpose": "Source"}]},
        "modules": [
            {
                "id": "core",
                "name": "Core",
                "responsibilities": ["logic"],
                "public_api": [],
                "dependencies": {"allows": [], "forbids": []},
                "files": [{"path": "src/core.py", "purpose": "impl"}],
            }
        ],
        "contracts": [],
        "cross_cutting": {"logging": [], "errors": {"strategy": "exceptions"}, "config": [], "security": []},
        "acceptance": {"tests": [], "benchmarks": [], "definition_of_done": ["done"]},
    }


@pytest.fixture
def base_task_ir() -> dict:
    return {
        "task_id": "task-001",
        "task_type": "code",
        "priority": "interactive",
        "objective": "Test",
        "inputs": [],
        "constraints": [],
        "assumptions": [],
        "agents": [{"tier": "B", "role": "coder"}],
        "plan": {"steps": [{"id": "S1", "actor": "coder", "action": "do", "outputs": ["o.py"]}]},
        "gates": ["format"],
        "definition_of_done": ["done"],
        "escalation": {"max_level": "B1", "on_second_failure": True},
    }


class TestArchitectureIRHardening:
    def test_backward_compat_without_interfaces(self, base_architecture_ir):
        code, out, _ = run_validator("arch", base_architecture_ir)
        assert code == 0, out

    def test_interfaces_section_valid(self, base_architecture_ir):
        base_architecture_ir["interfaces"] = {
            "data_structures": [
                {
                    "name": "ReviewDecision",
                    "kind": "dataclass",
                    "fields": [{"name": "decision", "type": "str"}, {"name": "confidence", "type": "float"}],
                    "owner_module": "core",
                }
            ],
            "interface_definitions": [
                {
                    "id": "reviewer",
                    "name": "Reviewer",
                    "provider_module": "core",
                    "operations": [{"name": "review", "signature": "review(pkg) -> ReviewDecision"}],
                }
            ],
        }
        code, out, _ = run_validator("arch", base_architecture_ir)
        assert code == 0, out

    def test_interfaces_reject_additional_props(self, base_architecture_ir):
        base_architecture_ir["interfaces"] = {"bogus": []}
        code, _, _ = run_validator("arch", base_architecture_ir)
        assert code == 2


class TestTaskIRHardening:
    def test_backward_compat_without_call_edges(self, base_task_ir):
        code, out, _ = run_validator("task", base_task_ir)
        assert code == 0, out

    def test_typed_call_edges_valid(self, base_task_ir):
        base_task_ir["plan"]["steps"].append(
            {
                "id": "S2",
                "actor": "worker",
                "action": "consume",
                "outputs": ["r.txt"],
                "depends_on": ["S1"],
                "call_edges": [{"to": "S1", "type": "data", "artifact": "o.py"}],
            }
        )
        code, out, _ = run_validator("task", base_task_ir)
        assert code == 0, out

    def test_call_edge_bad_type(self, base_task_ir):
        base_task_ir["plan"]["steps"][0]["call_edges"] = [{"to": "S1", "type": "telepathy"}]
        code, _, _ = run_validator("task", base_task_ir)
        assert code == 2


# ── RA-6: types.py evolution + backward compat ────────────────────────


class TestTypesEvolution:
    def test_new_enum_members(self):
        from src.proactive_delegation.types import ReviewDecision

        assert ReviewDecision.REQUEST_EVIDENCE.value == "request_evidence"
        assert ReviewDecision.REJECT_TO_EMPTY.value == "reject_to_empty"
        # legacy members intact
        assert ReviewDecision.APPROVE.value == "approve"
        assert ReviewDecision("request_changes") is ReviewDecision.REQUEST_CHANGES

    def test_architect_review_backward_compat_construction(self):
        from src.proactive_delegation.types import ArchitectReview, ReviewDecision

        # Legacy construction (no new fields) still works with old defaults.
        r = ArchitectReview(subtask_id="s1", decision=ReviewDecision.APPROVE)
        assert r.score == 0.0
        assert r.feedback == ""
        assert r.suggested_changes == []
        assert r.approved_output is None
        # New fields default safely.
        assert r.confidence == 0.0
        assert r.tripwire is False
        assert r.evidence == []
        assert r.verifier_requests == []

    def test_architect_review_extended_fields(self):
        from src.proactive_delegation.types import ArchitectReview, ReviewDecision

        r = ArchitectReview(
            subtask_id="s1",
            decision=ReviewDecision.REQUEST_EVIDENCE,
            confidence=0.7,
            tripwire=True,
            evidence=[{"kind": "test_result", "ref": "t1"}],
            verifier_requests=[{"verifier": "unit", "kind": "test"}],
        )
        d = r.to_dict()
        # Legacy keys preserved.
        for k in ["subtask_id", "decision", "feedback", "score", "suggested_changes", "approved_output"]:
            assert k in d
        # New keys additive.
        assert d["confidence"] == 0.7
        assert d["tripwire"] is True
        assert d["decision"] == "request_evidence"
        assert d["evidence"] == [{"kind": "test_result", "ref": "t1"}]

    def test_iteration_context_records_new_decisions(self):
        from src.proactive_delegation.types import IterationContext, ReviewDecision

        ctx = IterationContext()
        ctx.record_iteration("sub1", ReviewDecision.REJECT_TO_EMPTY, "discard")
        assert ctx.iteration_history[-1]["decision"] == "reject_to_empty"
