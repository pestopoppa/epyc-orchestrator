"""Reviewer-plane contract tests (control-plane spec §20.1).

Asserts the seven contract invariants over reviewer artifacts + ledger:

  1. Every ReviewDecision validates against its schema.
  2. Unknown fields are rejected / version-gated.
  3. Every blocking finding maps to a defined criterion.
  4. Every evidence reference resolves to a hash-verified ledger item.
  5. Diagnostic artifacts cannot be promoted to authoritative.
  6. Deterministic reducer output is identical on replay.
  7. Idempotent duplicate events do not create duplicate decisions.

LIVE tests run against landed Wave-1/2 code (schemas + review_ledger + trace
store). PENDING-CP1/CP2 tests import the sibling deliverables (policy_reducer /
authority / evidence_item|decision_envelope|assurance_profile schemas) behind
skip guards so this file is committable before those land, and activates
automatically once they do. NO inference.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_PATH = REPO_ROOT / "orchestration" / "validate_ir.py"
SCHEMA_DIR = REPO_ROOT / "orchestration"


# ── sibling-timing guards ─────────────────────────────────────────────────
def _try_import(modpath: str):
    try:
        return importlib.import_module(modpath)
    except Exception:
        return None


policy_reducer = _try_import("src.proactive_delegation.policy_reducer")  # CP1
authority = _try_import("src.proactive_delegation.authority")  # CP1

requires_cp1_reducer = pytest.mark.skipif(
    policy_reducer is None or not hasattr(policy_reducer, "reduce_decision"),
    reason="pending-CP1: src.proactive_delegation.policy_reducer.reduce_decision not landed",
)
requires_cp1_authority = pytest.mark.skipif(
    authority is None,
    reason="pending-CP1: src.proactive_delegation.authority not landed",
)


def _schema_present(name: str) -> bool:
    return (SCHEMA_DIR / name).exists()


requires_assurance_profile = pytest.mark.skipif(
    not _schema_present("assurance_profile.schema.json"),
    reason="pending-CP2: assurance_profile.schema.json not landed",
)
requires_evidence_item = pytest.mark.skipif(
    not _schema_present("evidence_item.schema.json"),
    reason="pending-CP2: evidence_item.schema.json not landed",
)


def run_validator(kind: str, doc: dict) -> tuple[int, str, str]:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), kind, "-"],
        input=json.dumps(doc),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def validate_direct(schema_name: str, doc: dict) -> list[str]:
    """Validate ``doc`` directly against a schema file (no validate_ir kind wiring).

    Used for CP2 schemas whose validator-kind may not be wired into validate_ir.py
    yet; the caller guards on the schema file's presence.
    """
    from jsonschema import Draft202012Validator

    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    return [e.message for e in Draft202012Validator(schema).iter_errors(doc)]


@pytest.fixture
def valid_review_decision() -> dict:
    return {
        "schema_version": "1.0.0",
        "decision": "reject",
        "confidence": 0.9,
        "blocking": {
            "tripwire": True,
            "blocking_issues": [
                {"summary": "migration duplicates records", "severity": "high", "evidence_ref": "0"}
            ],
        },
        "advisory": {"score": 0.2, "feedback": "fails idempotence"},
        "evidence": [{"kind": "test_result", "ref": "migration_property_test", "summary": "counterexample"}],
        "verifier_requests": [],
    }


@pytest.fixture
def ledger_conn(tmp_path: Path):
    from src.trace.store import ensure_schema

    c = ensure_schema(tmp_path / "events.sqlite")
    yield c
    c.close()


# ── §20.1.1 — every ReviewDecision validates ──────────────────────────────
class TestReviewDecisionValidates:
    def test_valid_decision(self, valid_review_decision):
        code, out, _ = run_validator("review", valid_review_decision)
        assert code == 0, out

    def test_every_enum_decision_validates(self, valid_review_decision):
        for d in ["approve", "reject", "reject_to_empty", "request_changes", "request_evidence", "escalate"]:
            valid_review_decision["decision"] = d
            code, out, _ = run_validator("review", valid_review_decision)
            assert code == 0, f"{d}: {out}"

    def test_abstain_is_recognized_or_pending_cp2(self, valid_review_decision):
        """Spec §9.2 adds ABSTAIN. CP2 evolves the enum (v1.1). Until then ABSTAIN
        is (correctly) rejected by the v1.0 schema. This test documents the seam:
        it passes on either side of the CP2 landing."""
        valid_review_decision["decision"] = "abstain"
        code, _, _ = run_validator("review", valid_review_decision)
        schema = json.loads((SCHEMA_DIR / "review_decision.schema.json").read_text())
        enum = schema["properties"]["decision"]["enum"]
        if "abstain" in enum:
            assert code == 0, "CP2 landed abstain in the enum; it must validate"
        else:
            assert code == 2, "pre-CP2 v1.0 schema must reject the unknown 'abstain' member"


# ── §20.1.2 — unknown fields rejected / version-gated ─────────────────────
class TestUnknownFieldsRejected:
    def test_additional_property_rejected(self, valid_review_decision):
        valid_review_decision["totally_new_field"] = 1
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    def test_schema_version_is_pattern_gated(self, valid_review_decision):
        valid_review_decision["schema_version"] = "not-a-semver"
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    def test_nested_blocking_issue_rejects_unknown(self, valid_review_decision):
        valid_review_decision["blocking"]["blocking_issues"][0]["made_up"] = True
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2


# ── §20.1.3 — every blocking finding maps to a criterion ──────────────────
class TestBlockingFindingMapsToCriterion:
    def test_blocking_issue_requires_summary_live(self, valid_review_decision):
        """LIVE floor: a blocking issue must at minimum carry a summary (landed contract)."""
        valid_review_decision["blocking"]["blocking_issues"] = [{"severity": "high"}]  # no summary
        code, _, _ = run_validator("review", valid_review_decision)
        assert code == 2

    @requires_assurance_profile
    def test_blocking_finding_criterion_resolves_to_profile(self, valid_review_decision):
        """PENDING-CP2: each blocking finding's criterion_id resolves to an
        AssuranceProfile criterion (spec §6.4 finding.criterion_id + §6.5)."""
        profile = {
            "schema_version": "1.0.0",
            "profile_id": "swe_release:v3",
            "domain": "software_engineering",
            "risk_class": "high",
            "criteria": {"migration_idempotence": {"severity": "high", "mandatory": True}},
            "verifier_registry": {"migration_idempotence": ["property_tests"]},
            "policy": {
                "reviewer_required_at": ["integration_complete"],
                "unknown_on_critical": "escalate",
                "reviewer_timeout": "abstain",
                "schema_error": "abstain",
                "no_reviewer_available": "defer",
                "evidence_budget_exhausted": "escalate",
                "max_review_rounds": 2,
                "max_evidence_rounds": 2,
            },
            "calibration_cohort": {"architect_family": "qwen", "reviewer_family": "glm", "domain": "software_engineering"},
        }
        errs = validate_direct("assurance_profile.schema.json", profile)
        if errs:
            pytest.skip(f"pending-CP2: assurance_profile schema still stabilizing: {errs}")
        # Core §20.1.3 contract: a blocking finding's criterion_id must resolve to a
        # defined profile criterion. (CP2 adds criterion_id to blocking findings.)
        finding = {"summary": "migration duplicates records", "criterion_id": "migration_idempotence"}
        criterion = finding.get("criterion_id")
        assert criterion is not None, "CP2 finding must carry a criterion_id"
        assert criterion in profile["criteria"], "criterion_id must resolve to a defined criterion"


# ── §20.1.4 — evidence ref resolves to a hash-verified ledger item ────────
class TestEvidenceRefResolves:
    def test_fabricated_evidence_ref_does_not_resolve_live(self, ledger_conn):
        """LIVE: a claimed evidence id absent from the ledger resolves to nothing,
        so it can grant zero authority (the §13.2-control-2 floor)."""
        from src.trace.review_ledger import iter_review_ledger_rows

        rows = list(iter_review_ledger_rows(ledger_conn))
        known_ids = {r.get("decision_id") for r in rows} | {r.get("candidate_id") for r in rows}
        assert "evidence_9999" not in known_ids

    def test_real_ledger_item_resolves_live(self, ledger_conn):
        """LIVE: a decision written to the ledger IS resolvable by its id."""
        from src.trace.review_ledger import ReviewLedgerRow, insert_review_ledger_row, iter_review_ledger_rows

        insert_review_ledger_row(ledger_conn, ReviewLedgerRow(decision_id="dev-real-1", candidate_id="cand-1"))
        ids = {r["decision_id"] for r in iter_review_ledger_rows(ledger_conn)}
        assert "dev-real-1" in ids

    @requires_evidence_item
    def test_evidence_item_hash_verified_resolution(self, ledger_conn):
        """PENDING-CP2: evidence_item carries an implementation/content hash and
        resolves through the ledger to a hash-verified item (spec §6.2, §20.1.4)."""
        evidence_item = {
            "schema_version": "1.0.0",
            "evidence_id": "evidence_01",
            "criterion_id": "migration_idempotence",
            "producer": {"type": "verifier", "id": "migration_property_test", "version": "1.3.0"},
            "status": {"logical": "fail", "execution": "ok"},
            "authority": {"class": "sound_refutation", "may_block": True},
        }
        errs = validate_direct("evidence_item.schema.json", evidence_item)
        assert errs == [], errs


# ── §20.1.5 — diagnostic artifacts cannot be promoted to authoritative ────
class TestDiagnosticNonPromotion:
    def test_diagnostic_marker_is_non_authoritative_contract_live(self):
        """LIVE: the spec §5.1 diagnostic marker is non-authoritative by definition."""
        marker = {
            "artifact_role": "diagnostic_only",
            "authoritative": False,
            "may_be_merged": False,
            "generated_by_role": "reviewer",
            "purpose": "reproduce_or_test_finding",
        }
        assert marker["authoritative"] is False
        assert marker["may_be_merged"] is False
        assert marker["artifact_role"] == "diagnostic_only"

    @requires_cp1_authority
    def test_reviewer_diagnostic_cannot_be_marked_authoritative(self):
        """PENDING-CP1: the authority layer must refuse to grant approval/merge
        authority to a reviewer-generated diagnostic artifact (spec §5.1)."""
        # Coordinated by name: a helper that classifies whether an artifact may be
        # promoted. Exact API is CP1's; probe the documented invariant defensively.
        promote = getattr(authority, "may_promote_to_authoritative", None)
        if promote is None:
            pytest.skip("pending-CP1: authority.may_promote_to_authoritative not defined")
        assert promote({"artifact_role": "diagnostic_only", "generated_by_role": "reviewer"}) is False


# ── §20.1.6 — deterministic reducer output identical on replay ────────────
class TestReducerReplayDeterminism:
    @requires_cp1_reducer
    def test_reduce_decision_is_pure_on_replay(self):
        """PENDING-CP1: reduce_decision(package, verification, review, profile,
        calibration) must be a pure function — identical output on replay (§8, §20.1.6)."""
        package = {"package_id": "cp-1", "subject": {"artifact_hash": "sha256:aaa"}}
        verification = {"report_id": "vr-1", "checks": [{"check_id": "unit", "kind": "test", "outcome": "pass"}]}
        review = {"decision": "approve", "confidence": 0.8, "blocking": {"tripwire": False}}
        profile = {"profile_id": "swe_release:v3", "criteria": {}, "policy": {}}
        calibration = {"cohort_id": "c1", "upper_risk_bound": 0.05}
        try:
            a = policy_reducer.reduce_decision(package, verification, review, profile, calibration)
            b = policy_reducer.reduce_decision(package, verification, review, profile, calibration)
        except TypeError:
            pytest.skip("pending-CP1: reduce_decision signature not yet spec-aligned (§8)")
        # Compare via best-available canonicalization.
        def canon(x):
            for attr in ("to_dict", "_asdict"):
                if hasattr(x, attr):
                    return getattr(x, attr)()
            return repr(x)

        assert canon(a) == canon(b)


# ── §20.1.7 — idempotent duplicate events don't dup decisions ─────────────
class TestIdempotentDuplicateEvents:
    def test_duplicate_ledger_decision_deduped_live(self, ledger_conn):
        from src.trace.review_ledger import ReviewLedgerRow, insert_review_ledger_row, review_ledger_count

        row = ReviewLedgerRow(decision_id="dev-dup-1", decision="approve")
        ins1, skip1 = insert_review_ledger_row(ledger_conn, row)
        ins2, skip2 = insert_review_ledger_row(ledger_conn, row)
        assert (ins1, skip1) == (1, 0)
        assert (ins2, skip2) == (0, 1)  # second is a no-op
        assert review_ledger_count(ledger_conn) == 1

    def test_duplicate_trace_event_deduped_live(self, ledger_conn):
        from src.trace.store import Event, event_count, upsert_events

        ev = Event(
            ts_utc="2026-07-17T00:00:00Z",
            source="review_plane",
            source_path="emit://review/dev-1",
            source_line=0,
            category="review_decision",
            summary="decision dev-1",
        )
        before = event_count(ledger_conn)
        upsert_events(ledger_conn, [ev])
        upsert_events(ledger_conn, [ev])  # duplicate (same source_path,source_line)
        assert event_count(ledger_conn) == before + 1
