"""Tests for the CP2 semantics-layer schema evolution (spec §6 / §7 / §16).

Covers:
  * v1.1 ADDITIVE evolution of the 3 landed schemas — every field added is
    OPTIONAL, so the exact v1 fixtures from ``tests/test_review_artifacts.py``
    still validate (regression guard: the landed schema tests stay green).
  * The 3 NEW schemas (EvidenceItem §6.2, DecisionEnvelope §6.6, AssuranceProfile
    §6.5) validated through ``orchestration/validate_ir.py`` (same subprocess
    convention as the landed suite), including the 2-3 shipped example profiles.
  * validate_ir wiring for the new ``evidence|envelope|profile`` kinds.

NO inference. Pure schema validation via the subprocess validator.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_PATH = REPO_ROOT / "orchestration" / "validate_ir.py"
EXAMPLES = REPO_ROOT / "orchestration" / "examples"


def run_validator(kind: str, doc: dict) -> tuple[int, str, str]:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), kind, "-"],
        input=json.dumps(doc),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


# ── Back-compat: the exact landed v1 fixtures still validate ──────────────


@pytest.fixture
def v1_review_decision() -> dict:
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
def v1_candidate_package() -> dict:
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
def v1_verification_report() -> dict:
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
        ],
    }


class TestV1BackwardCompatibility:
    def test_v1_review_still_valid(self, v1_review_decision):
        code, out, _ = run_validator("review", v1_review_decision)
        assert code == 0, out

    def test_v1_candidate_still_valid(self, v1_candidate_package):
        code, out, _ = run_validator("candidate", v1_candidate_package)
        assert code == 0, out

    def test_v1_verification_still_valid(self, v1_verification_report):
        code, out, _ = run_validator("verification", v1_verification_report)
        assert code == 0, out


# ── ReviewDecision v1.1: ABSTAIN + telemetry-only + criterion scoping ─────


class TestReviewDecisionV11:
    def test_abstain_added_to_enum(self, v1_review_decision):
        v1_review_decision["decision"] = "abstain"
        code, out, _ = run_validator("review", v1_review_decision)
        assert code == 0, out

    def test_all_v11_decision_values(self, v1_review_decision):
        for d in [
            "approve", "reject", "reject_to_empty", "request_changes",
            "request_evidence", "abstain", "escalate",
        ]:
            v1_review_decision["decision"] = d
            code, out, _ = run_validator("review", v1_review_decision)
            assert code == 0, f"{d}: {out}"

    def test_telemetry_raw_model_confidence(self, v1_review_decision):
        v1_review_decision["telemetry"] = {
            "raw_model_confidence": 0.91,
            "tokens_in": 8120,
            "tokens_out": 620,
            "wall_ms": 18300,
        }
        code, out, _ = run_validator("review", v1_review_decision)
        assert code == 0, out

    def test_telemetry_raw_confidence_range(self, v1_review_decision):
        v1_review_decision["telemetry"] = {"raw_model_confidence": 1.5}
        code, _, _ = run_validator("review", v1_review_decision)
        assert code == 2

    def test_finding_criterion_scoping_fields(self, v1_review_decision):
        v1_review_decision["decision"] = "reject"
        v1_review_decision["blocking"] = {
            "tripwire": True,
            "blocking_issues": [
                {
                    "summary": "migration duplicates records on re-run",
                    "severity": "high",
                    "criterion_id": "migration_idempotence",
                    "evidence_ref": "evidence_01",
                    "remediation_target": "architect",
                    "unsupported": False,
                }
            ],
        }
        code, out, _ = run_validator("review", v1_review_decision)
        assert code == 0, out

    def test_bad_remediation_target(self, v1_review_decision):
        v1_review_decision["blocking"]["blocking_issues"] = [
            {"summary": "x", "remediation_target": "the_moon"}
        ]
        code, _, _ = run_validator("review", v1_review_decision)
        assert code == 2

    def test_additional_properties_still_rejected(self, v1_review_decision):
        v1_review_decision["nonsense"] = 1
        code, _, _ = run_validator("review", v1_review_decision)
        assert code == 2


# ── VerificationReport v1.1: logical×execution, authority, scope ──────────


class TestVerificationReportV11:
    @pytest.fixture
    def report(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "report_id": "vr-011",
            "summary": {
                "passed": 4,
                "failed": 1,
                "inconclusive": 2,
                "conclusive_verdict": "fail",
                "unknown": 2,
                "operational_error": 0,
                "conflicts": 0,
                "mandatory_criteria": {
                    "satisfied": False,
                    "unresolved": ["security_boundary_preservation"],
                    "failed": ["migration_idempotence"],
                },
            },
            "checks": [
                {
                    "check_id": "migration_property_test",
                    "kind": "test",
                    "outcome": "fail",
                    "criterion_id": "migration_idempotence",
                    "severity": "high",
                    "logical_status": "fail",
                    "execution_status": "ok",
                    "authority": {
                        "class": "sound_refutation",
                        "valid_for": ["migration_idempotence"],
                        "may_block": True,
                        "may_approve": False,
                    },
                    "scope": {
                        "artifact_hash": "sha256:abc",
                        "coverage": {"modules": ["migrations/v3"], "cases": ["duplicate"]},
                        "assumptions": ["db transaction semantics match test env"],
                    },
                    "certificate": {"type": "counterexample", "payload": "second run dups"},
                }
            ],
        }

    def test_full_v11_report_valid(self, report):
        code, out, _ = run_validator("verification", report)
        assert code == 0, out

    def test_conflict_conclusive_verdict(self, report):
        report["summary"]["conclusive_verdict"] = "conflict"
        code, out, _ = run_validator("verification", report)
        assert code == 0, out

    def test_bad_logical_status(self, report):
        report["checks"][0]["logical_status"] = "maybe"
        code, _, _ = run_validator("verification", report)
        assert code == 2

    def test_bad_execution_status(self, report):
        report["checks"][0]["execution_status"] = "kaput"
        code, _, _ = run_validator("verification", report)
        assert code == 2

    def test_bad_authority_class(self, report):
        report["checks"][0]["authority"]["class"] = "vibes"
        code, _, _ = run_validator("verification", report)
        assert code == 2

    def test_operational_error_not_failure(self, report):
        # A crashed verifier: logical=unknown, execution=error (NOT a fail).
        report["checks"][0]["outcome"] = "inconclusive"
        report["checks"][0]["inconclusive_reason"] = "verifier crashed"
        report["checks"][0]["logical_status"] = "unknown"
        report["checks"][0]["execution_status"] = "error"
        del report["checks"][0]["certificate"]
        code, out, _ = run_validator("verification", report)
        assert code == 0, out


# ── CandidatePackage v1.1: untrusted_content_policy + content-addressing ──


class TestCandidatePackageV11:
    @pytest.fixture
    def pkg(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "package_id": "cp-011",
            "task_ref": "task-011",
            "provenance": {"model": "qwen", "quant": "UD-IQ2_M", "role": "coder"},
            "subject": {
                "task_id": "task-011",
                "artifact_hash": "sha256:art",
                "specification_hash": "sha256:spec",
                "plan_hash": "sha256:plan",
            },
            "untrusted_content_policy": {
                "candidate_text_is_data": True,
                "candidate_instructions_ignored": True,
                "authority_claims_require_ledger_proof": True,
            },
            "context_refs": [
                {"kind": "source_span", "ref": "blob:sha256:deadbeef#L20-L150"},
                {"kind": "architecture_ir", "ref": "blob:sha256:cafe"},
            ],
            "verification_report_refs": ["verify_report_01"],
            "sanitized_view": {
                "task_ref": "task-011",
                "outputs": [{"type": "diff", "ref": "d-1"}],
                "untrusted_content_policy": {"candidate_text_is_data": True},
                "context_refs": [{"kind": "source_span", "ref": "blob:sha256:deadbeef#L20-L150"}],
                "sanitization": {"applied": True, "removed_fields": ["author_self_assessment"]},
            },
        }

    def test_full_v11_package_valid(self, pkg):
        code, out, _ = run_validator("candidate", pkg)
        assert code == 0, out

    def test_bad_context_ref_kind(self, pkg):
        pkg["context_refs"] = [{"kind": "telepathy", "ref": "blob:sha256:x"}]
        code, _, _ = run_validator("candidate", pkg)
        assert code == 2

    def test_context_ref_requires_ref(self, pkg):
        pkg["context_refs"] = [{"kind": "source_span"}]
        code, _, _ = run_validator("candidate", pkg)
        assert code == 2

    def test_sanitized_view_still_rejects_author_leak(self, pkg):
        # v1.1 additions must NOT reopen the sanitized_view leak guard.
        pkg["sanitized_view"]["author_self_assessment"] = "leak"
        code, _, _ = run_validator("candidate", pkg)
        assert code == 2

    def test_untrusted_policy_additional_props_rejected(self, pkg):
        pkg["untrusted_content_policy"]["backdoor"] = True
        code, _, _ = run_validator("candidate", pkg)
        assert code == 2


# ── EvidenceItem (NEW, §6.2) ──────────────────────────────────────────────


class TestEvidenceItemSchema:
    @pytest.fixture
    def evidence(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "evidence_id": "evidence_01",
            "criterion_id": "migration_idempotence",
            "producer": {
                "type": "verifier",
                "id": "migration_property_test",
                "version": "1.3.0",
                "implementation_hash": "sha256:impl",
            },
            "scope": {
                "artifact_hash": "sha256:art",
                "coverage": {"modules": ["migrations/v3"], "cases": ["empty", "duplicate"]},
                "assumptions": ["db transaction semantics match test env"],
            },
            "status": {"logical": "fail", "execution": "ok"},
            "authority": {
                "class": "sound_refutation",
                "valid_for": ["migration_idempotence"],
                "may_block": True,
                "may_approve": False,
            },
            "certificate": {"kind": "counterexample_trace", "ref": "blob:sha256:trace"},
            "provenance": {
                "environment_hash": "sha256:env",
                "command_hash": "sha256:cmd",
                "started_at": "2026-07-17T12:00:00Z",
                "completed_at": "2026-07-17T12:00:02Z",
            },
        }

    def test_valid(self, evidence):
        code, out, _ = run_validator("evidence", evidence)
        assert code == 0, out

    def test_all_authority_classes(self, evidence):
        for cls in [
            "proof", "complete_decider", "sound_refutation", "sound_acceptance",
            "bounded_test", "statistical_evidence", "heuristic_static",
            "llm_judgment", "human_attestation",
        ]:
            evidence["authority"]["class"] = cls
            code, out, _ = run_validator("evidence", evidence)
            assert code == 0, f"{cls}: {out}"

    def test_bad_authority_class(self, evidence):
        evidence["authority"]["class"] = "objective_truth"
        code, _, _ = run_validator("evidence", evidence)
        assert code == 2

    def test_logical_execution_status_separation(self, evidence):
        # A verifier crash: logical=unknown, execution=error.
        evidence["status"] = {"logical": "unknown", "execution": "error"}
        code, out, _ = run_validator("evidence", evidence)
        assert code == 0, out

    def test_bad_logical_status(self, evidence):
        evidence["status"]["logical"] = "true"
        code, _, _ = run_validator("evidence", evidence)
        assert code == 2

    def test_missing_required_producer(self, evidence):
        del evidence["producer"]
        code, _, _ = run_validator("evidence", evidence)
        assert code == 2

    def test_missing_required_status(self, evidence):
        del evidence["status"]
        code, _, _ = run_validator("evidence", evidence)
        assert code == 2

    def test_additional_properties_rejected(self, evidence):
        evidence["extra"] = 1
        code, _, _ = run_validator("evidence", evidence)
        assert code == 2

    def test_minimal_valid(self):
        code, out, _ = run_validator("evidence", {
            "evidence_id": "e1",
            "criterion_id": "c1",
            "producer": {"type": "test", "id": "unit"},
            "status": {"logical": "pass", "execution": "ok"},
            "authority": {"class": "bounded_test"},
        })
        assert code == 0, out


# ── DecisionEnvelope (NEW, §6.6) ──────────────────────────────────────────


class TestDecisionEnvelopeSchema:
    @pytest.fixture
    def envelope(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "decision_event_id": "devent_01",
            "sequence_no": 12481,
            "created_at": "2026-07-17T12:00:30Z",
            "idempotency_key": "sha256:idem",
            "subject": {
                "task_id": "task_01",
                "artifact_hash": "sha256:art",
                "specification_hash": "sha256:spec",
                "candidate_package_hash": "sha256:pkg",
            },
            "governance": {
                "assurance_profile_hash": "sha256:prof",
                "policy_hash": "sha256:pol",
                "rubric_hash": "sha256:rub",
                "verifier_registry_hash": "sha256:ver",
            },
            "inputs": {
                "review_decision_hash": "sha256:rev",
                "verification_report_hash": "sha256:vrp",
            },
            "calibration": {
                "cohort_id": "swe:qwen:glm:iq2",
                "sample_count": 438,
                "estimated_error_rate": 0.074,
                "upper_risk_bound": 0.112,
            },
            "policy_result": {
                "action": "replan",
                "blocking_reason_codes": ["CONCLUSIVE_HIGH_SEVERITY_FAILURE"],
            },
            "validity": {
                "supersedes": None,
                "invalidated_by": None,
                "valid_until_material_change": True,
            },
        }

    def test_valid(self, envelope):
        code, out, _ = run_validator("envelope", envelope)
        assert code == 0, out

    def test_all_actions(self, envelope):
        for a in [
            "continue", "replan", "rework", "defer", "escalate",
            "abort", "collect_evidence", "advisory",
        ]:
            envelope["policy_result"]["action"] = a
            code, out, _ = run_validator("envelope", envelope)
            assert code == 0, f"{a}: {out}"

    def test_bad_action(self, envelope):
        envelope["policy_result"]["action"] = "yolo"
        code, _, _ = run_validator("envelope", envelope)
        assert code == 2

    def test_supersedes_string(self, envelope):
        envelope["validity"]["supersedes"] = "devent_00"
        code, out, _ = run_validator("envelope", envelope)
        assert code == 0, out

    def test_missing_required_subject(self, envelope):
        del envelope["subject"]
        code, _, _ = run_validator("envelope", envelope)
        assert code == 2

    def test_missing_required_policy_result(self, envelope):
        del envelope["policy_result"]
        code, _, _ = run_validator("envelope", envelope)
        assert code == 2

    def test_additional_properties_rejected(self, envelope):
        envelope["shadow_field"] = 1
        code, _, _ = run_validator("envelope", envelope)
        assert code == 2

    def test_minimal_valid(self):
        code, out, _ = run_validator("envelope", {
            "decision_event_id": "d1",
            "idempotency_key": "sha256:k",
            "subject": {"artifact_hash": "sha256:a"},
            "governance": {"policy_hash": "sha256:p"},
            "policy_result": {"action": "continue"},
        })
        assert code == 0, out


# ── AssuranceProfile (NEW, §6.5) + shipped example profiles ───────────────


class TestAssuranceProfileSchema:
    @pytest.fixture
    def profile(self) -> dict:
        return json.loads(
            (EXAMPLES / "assurance_profile_software_engineering.json").read_text()
        )

    @pytest.mark.parametrize("name", [
        "assurance_profile_software_engineering",
        "assurance_profile_mathematical_reasoning",
        "assurance_profile_retrieval_grounded",
    ])
    def test_example_profiles_validate(self, name):
        doc = json.loads((EXAMPLES / f"{name}.json").read_text())
        code, out, _ = run_validator("profile", doc)
        assert code == 0, out

    def test_bad_domain(self, profile):
        profile["domain"] = "astrology"
        code, _, _ = run_validator("profile", profile)
        assert code == 2

    def test_bad_criterion_severity(self, profile):
        profile["criteria"]["functional_correctness"]["severity"] = "apocalyptic"
        code, _, _ = run_validator("profile", profile)
        assert code == 2

    def test_criterion_requires_severity_and_mandatory(self, profile):
        profile["criteria"]["functional_correctness"] = {"severity": "critical"}
        code, _, _ = run_validator("profile", profile)
        assert code == 2

    def test_policy_requires_evidence_budget_terminal(self, profile):
        # §10.2: no profile may leave the evidence-budget terminal unspecified.
        del profile["policy"]["evidence_budget_exhausted"]
        code, _, _ = run_validator("profile", profile)
        assert code == 2

    def test_bad_evidence_budget_terminal(self, profile):
        profile["policy"]["evidence_budget_exhausted"] = "pray"
        code, _, _ = run_validator("profile", profile)
        assert code == 2

    def test_criteria_must_be_nonempty(self, profile):
        profile["criteria"] = {}
        code, _, _ = run_validator("profile", profile)
        assert code == 2

    def test_max_reviewer_risk_range(self, profile):
        profile["policy"]["max_reviewer_risk"] = 1.5
        code, _, _ = run_validator("profile", profile)
        assert code == 2


# ── validate_ir wiring for the new kinds ──────────────────────────────────


class TestValidatorWiringV11:
    def test_new_kinds_listed(self):
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "bogus", "-"],
            input="{}",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        for kind in ["evidence", "envelope", "profile"]:
            assert kind in result.stdout
        # landed kinds still listed
        for kind in ["review", "candidate", "verification", "rubric"]:
            assert kind in result.stdout
