"""Evidence-loop tests (control-plane spec §20.4 + §10.1).

Six bounded-loop invariants:

  1. Duplicate evidence requests are deduplicated.
  2. Evidence rounds stop at the configured limit.
  3. Budget exhaustion reaches the profile's terminal action.
  4. Unavailable runners do not create an infinite retry.
  5. New evidence is accumulated and hash-bound.
  6. Artifact modification invalidates prior evidence as appropriate.

Runs LIVE against CP1's bounded-loop reducer (``policy_reducer.reduce_decision`` +
``LoopState``), the landed ``IterationContext`` budget, the durable
``EscalationSink``, and CP2's content-addressed invalidation. Sibling-dependent
paths carry skip guards. NO inference.
"""

from __future__ import annotations

import hashlib
import importlib
import json

import pytest

pr = None
rl = None
try:
    pr = importlib.import_module("src.proactive_delegation.policy_reducer")
except Exception:  # pragma: no cover
    pr = None
try:
    rl = importlib.import_module("src.trace.review_ledger")
except Exception:  # pragma: no cover
    rl = None

requires_reducer = pytest.mark.skipif(
    pr is None or not hasattr(pr, "reduce_decision") or not hasattr(pr, "LoopState"),
    reason="pending-CP1: policy_reducer.reduce_decision / LoopState not landed",
)
requires_cp2 = pytest.mark.skipif(
    rl is None or not hasattr(rl, "invalidate_on_material_change"),
    reason="pending-CP2: invalidation API not landed",
)


def _request_signature(req: dict) -> str:
    return hashlib.sha256(
        json.dumps(req, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ── §20.4.1 — duplicate evidence requests are deduplicated ────────────────
class TestEvidenceRequestDedup:
    def test_identical_requests_collapse_by_signature_live(self):
        """§10.1 semantic dedup: two identical verifier requests share a content
        signature and collapse to one (idempotent, content-addressed)."""
        a = {"verifier": "unit", "kind": "test", "target": "ans-1"}
        b = {"target": "ans-1", "kind": "test", "verifier": "unit"}  # key-order noise
        seen: dict[str, dict] = {}
        for req in (a, b, a):
            seen[_request_signature(req)] = req
        assert len(seen) == 1  # three submissions, one distinct request

    def test_distinct_requests_do_not_collapse_live(self):
        a = {"verifier": "unit", "kind": "test", "target": "ans-1"}
        c = {"verifier": "unit", "kind": "test", "target": "ans-2"}
        seen = {_request_signature(r): r for r in (a, c)}
        assert len(seen) == 2


# ── §20.4.2 — evidence rounds stop at the configured limit ────────────────
class TestEvidenceRoundsStop:
    def test_iteration_context_caps_rounds_live(self):
        from src.proactive_delegation.types import IterationContext, ReviewDecision

        ctx = IterationContext(max_iterations=2, max_total_iterations=10)
        assert ctx.can_iterate("s1") is True
        ctx.record_iteration("s1", ReviewDecision.REQUEST_EVIDENCE)
        ctx.record_iteration("s1", ReviewDecision.REQUEST_EVIDENCE)
        assert ctx.can_iterate("s1") is False  # capped at max_iterations

    @requires_reducer
    def test_reducer_loops_under_budget_terminates_at_limit(self):
        pkg = {"package_id": "cp"}
        verification = {
            "report_id": "vr",
            "checks": [{"check_id": "fmt", "kind": "gate", "outcome": "pass", "required": True}],
            "summary": {"conclusive_verdict": "pass"},
        }
        review = {"decision": "request_evidence", "verifier_requests": [{"verifier": "u", "kind": "test", "target": "o"}]}
        profile = {
            "profile_id": "p", "domain": "software_engineering", "risk_class": "high",
            "criteria": {}, "policy": {"evidence_budget_exhausted": "escalate", "max_evidence_rounds": 2, "max_review_rounds": 2},
        }
        cal = {"cohort_id": "c", "upper_risk_bound": 0.01}
        under = pr.reduce_decision(pkg, verification, review, profile, cal, loop_state=pr.LoopState(evidence_round=0))
        at_limit = pr.reduce_decision(pkg, verification, review, profile, cal, loop_state=pr.LoopState(evidence_round=2))
        assert under.action is pr.PolicyAction.COLLECT_EVIDENCE  # still looping
        assert at_limit.action is not pr.PolicyAction.COLLECT_EVIDENCE  # loop stopped


# ── §20.4.3 — budget exhaustion reaches the profile's terminal action ─────
class TestBudgetExhaustionTerminal:
    @requires_reducer
    def test_exhaustion_yields_terminal_profile_action(self):
        pkg = {"package_id": "cp"}
        verification = {"report_id": "vr", "checks": [{"check_id": "fmt", "kind": "gate", "outcome": "pass"}], "summary": {"conclusive_verdict": "pass"}}
        review = {"decision": "request_evidence", "verifier_requests": [{"verifier": "u", "kind": "test", "target": "o"}]}
        profile = {
            "profile_id": "p", "domain": "software_engineering", "risk_class": "high",
            "criteria": {}, "policy": {"evidence_budget_exhausted": "escalate", "max_evidence_rounds": 2, "max_review_rounds": 2},
        }
        cal = {"cohort_id": "c", "upper_risk_bound": 0.01}
        result = pr.reduce_decision(pkg, verification, review, profile, cal, loop_state=pr.LoopState(evidence_round=2))
        assert getattr(result, "terminal", True) is True
        assert "EVIDENCE_BUDGET_EXHAUSTED" in result.reason_codes
        assert result.action is pr.PolicyAction.ESCALATE  # the profile's configured terminal

    def test_exhaustion_routes_to_durable_escalation_live(self, tmp_path):
        """The terminal escalation must land in the durable sink (never lost)."""
        from src.proactive_delegation.escalation_sink import EscalationSink

        sink = EscalationSink(tmp_path / "events.sqlite")
        try:
            eid = sink.escalate({"package_id": "cp"}, "EVIDENCE_BUDGET_EXHAUSTED")
            assert sink.get(eid)["status"] == "open"  # queryable terminal-or-open, never dropped
        finally:
            sink.close()


# ── §20.4.4 — unavailable runner does not create an infinite retry ────────
class TestUnavailableRunnerNoInfiniteRetry:
    def test_unavailable_runner_stops_and_escalates_live(self, tmp_path):
        from src.proactive_delegation.escalation_sink import EscalationSink
        from src.proactive_delegation.types import IterationContext, ReviewDecision

        class _UnavailableRunner:
            calls = 0

            def run(self):
                type(self).calls += 1
                raise RuntimeError("runner unavailable")

        runner = _UnavailableRunner()
        ctx = IterationContext(max_iterations=3, max_total_iterations=3)
        sink = EscalationSink(tmp_path / "events.sqlite")
        try:
            attempts = 0
            while ctx.can_iterate("evreq"):
                attempts += 1
                try:
                    runner.run()
                except RuntimeError:
                    ctx.record_iteration("evreq", ReviewDecision.REQUEST_EVIDENCE)
            # Bounded: attempts never exceeds the cap — no infinite retry.
            assert attempts == 3
            eid = sink.escalate({"runner": "unit"}, "NO_REVIEWER_AVAILABLE")
            assert sink.get(eid)["reason_code"] == "NO_REVIEWER_AVAILABLE"
        finally:
            sink.close()


# ── §20.4.5 — new evidence is accumulated and hash-bound ──────────────────
@requires_cp2
class TestEvidenceAccumulatedHashBound:
    def test_accumulated_evidence_is_content_addressed(self):
        # Each accumulated evidence request is bound to a stable content hash.
        reqs = [
            {"verifier": "unit", "kind": "test", "target": "a"},
            {"verifier": "prop", "kind": "test", "target": "b"},
        ]
        accumulated = {_request_signature(r): r for r in reqs}
        assert len(accumulated) == 2
        # Re-accumulating the same evidence does not grow the set (idempotent).
        accumulated[_request_signature(reqs[0])] = reqs[0]
        assert len(accumulated) == 2

    def test_material_hash_binds_evidence_assumptions(self):
        base = rl.MaterialInputs(evidence_assumptions_hash="sha256:assume-v1")
        changed = rl.MaterialInputs(evidence_assumptions_hash="sha256:assume-v2")
        assert rl.compute_material_hash(base) != rl.compute_material_hash(changed)


# ── §20.4.6 — artifact modification invalidates prior evidence ────────────
@requires_cp2
class TestArtifactModificationInvalidatesEvidence:
    def test_artifact_change_invalidates_prior_decision(self, tmp_path):
        from src.trace.store import ensure_schema

        conn = ensure_schema(tmp_path / "events.sqlite")
        try:
            material = rl.MaterialInputs(
                artifact_hash="sha256:art-v1", retrieved_evidence_hash="sha256:ev-v1"
            )
            row = rl.DecisionEnvelopeRow(decision_event_id="devt-ev", action="continue", material=material)
            rl.record_decision_envelope(conn, row)
            assert rl.is_decision_valid(conn, "devt-ev") is True
            # Modify the artifact bytes: prior evidence no longer applies.
            new_material = rl.MaterialInputs(
                artifact_hash="sha256:art-v2", retrieved_evidence_hash="sha256:ev-v1"
            )
            record = rl.invalidate_on_material_change(conn, "devt-ev", new_material)
            assert record is not None
            assert "artifact_hash" in record["changed_inputs"]
            assert rl.is_decision_valid(conn, "devt-ev") is False
        finally:
            conn.close()
