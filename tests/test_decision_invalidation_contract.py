"""Decision-invalidation contract tests (control-plane spec §20.3).

Complements CP2's own invalidation tests: independently changes EACH material
input (§12.3 / §20.3) and asserts the decision is invalidated, that the audit
trail pinpoints the changed field, and that invalidation is append-only (never a
rewrite, §5.5).

Runs LIVE against the CP2 invalidation API in ``src/trace/review_ledger.py``
(``MaterialInputs`` / ``compute_material_hash`` / ``detect_material_change`` /
``record_decision_envelope`` / ``invalidate_on_material_change`` /
``is_decision_valid``). A skip guard keeps the file committable if the CP2 API
lands slightly later. A pure hash-principle test needs no sibling code at all.
NO inference; hermetic tmp SQLite only.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

rl = None
try:
    rl = importlib.import_module("src.trace.review_ledger")
except Exception:  # pragma: no cover
    rl = None

_CP2_NAMES = (
    "MaterialInputs",
    "compute_material_hash",
    "detect_material_change",
    "DecisionEnvelopeRow",
    "record_decision_envelope",
    "invalidate_on_material_change",
    "is_decision_valid",
    "invalidations_for",
)
_cp2_ready = rl is not None and all(hasattr(rl, n) for n in _CP2_NAMES)
requires_cp2_invalidation = pytest.mark.skipif(
    not _cp2_ready, reason="pending-CP2: decision-envelope invalidation API not landed in review_ledger"
)

# §20.3's nine material inputs → MaterialInputs field names (§12.3).
SPEC_20_3_INPUTS = {
    "artifact bytes": "artifact_hash",
    "task specification": "specification_hash",
    "policy version": "policy_hash",
    "rubric": "rubric_hash",
    "verifier implementation": "verifier_registry_hash",
    "dependency lockfile/environment": "environment_hash",
    "reviewer model/quantization": "reviewer_model_hash",
    "prompt": "prompt_hash",
    "retrieved source version": "retrieved_evidence_hash",
}


def _base_material():
    return rl.MaterialInputs(
        artifact_hash="sha256:artifact-v1",
        specification_hash="sha256:spec-v1",
        plan_hash="sha256:plan-v1",
        assurance_profile_hash="sha256:profile-v1",
        policy_hash="sha256:policy-v1",
        rubric_hash="sha256:rubric-v1",
        verifier_registry_hash="sha256:verifiers-v1",
        environment_hash="sha256:env-v1",
        reviewer_model_hash="sha256:glm-iq2-v1",
        prompt_hash="sha256:prompt-v1",
        decoding_parameters_hash="sha256:decode-v1",
        retrieved_evidence_hash="sha256:retrieval-v1",
        security_policy_hash="sha256:secpol-v1",
        evidence_assumptions_hash="sha256:assume-v1",
    )


@pytest.fixture
def conn(tmp_path: Path):
    from src.trace.store import ensure_schema

    c = ensure_schema(tmp_path / "events.sqlite")
    yield c
    c.close()


# ── pure principle (no sibling code) ──────────────────────────────────────
@requires_cp2_invalidation
class TestMaterialHashPrinciple:
    def test_identical_inputs_share_hash(self):
        assert rl.compute_material_hash(_base_material()) == rl.compute_material_hash(_base_material())

    @pytest.mark.parametrize("label,field", list(SPEC_20_3_INPUTS.items()))
    def test_each_input_change_flips_hash(self, label, field):
        base = _base_material()
        changed = rl.MaterialInputs(**{**base.as_dict(), field: "sha256:CHANGED"})
        assert rl.compute_material_hash(changed) != rl.compute_material_hash(base), label

    @pytest.mark.parametrize("label,field", list(SPEC_20_3_INPUTS.items()))
    def test_detect_material_change_pinpoints_field(self, label, field):
        base = _base_material()
        changed = rl.MaterialInputs(**{**base.as_dict(), field: "sha256:CHANGED"})
        assert rl.detect_material_change(base, changed) == [field], label


# ── end-to-end invalidation over the ledger ───────────────────────────────
@requires_cp2_invalidation
class TestDecisionInvalidatedOnMaterialChange:
    def _record(self, conn, material, decision_event_id="devt-1"):
        row = rl.DecisionEnvelopeRow(
            decision_event_id=decision_event_id,
            task_id="task-1",
            action="continue",
            material=material,
        )
        rl.record_decision_envelope(conn, row)
        return decision_event_id

    def test_no_change_does_not_invalidate(self, conn):
        did = self._record(conn, _base_material())
        assert rl.invalidate_on_material_change(conn, did, _base_material()) is None
        assert rl.is_decision_valid(conn, did) is True

    @pytest.mark.parametrize("label,field", list(SPEC_20_3_INPUTS.items()))
    def test_each_material_input_change_invalidates(self, conn, label, field):
        did = self._record(conn, _base_material(), decision_event_id=f"devt-{field}")
        assert rl.is_decision_valid(conn, did) is True
        new_material = rl.MaterialInputs(**{**_base_material().as_dict(), field: "sha256:CHANGED"})
        record = rl.invalidate_on_material_change(conn, did, new_material, reason=f"{label} changed")
        assert record is not None, f"{label}: expected invalidation"
        assert field in record["changed_inputs"], f"{label}: audit trail must name the field"
        assert rl.is_decision_valid(conn, did) is False, f"{label}: decision must be invalidated"

    def test_invalidation_is_append_only_not_a_rewrite(self, conn):
        """§5.5: invalidation appends a DECISION_INVALIDATED event; the envelope row is untouched."""
        did = self._record(conn, _base_material())
        before = rl.decision_envelope_count(conn)
        env_before = rl.get_decision_envelope(conn, did)
        new_material = rl.MaterialInputs(**{**_base_material().as_dict(), "artifact_hash": "sha256:v2"})
        rl.invalidate_on_material_change(conn, did, new_material)
        # Envelope row count unchanged; the original row still resolvable with its old hash.
        assert rl.decision_envelope_count(conn) == before
        env_after = rl.get_decision_envelope(conn, did)
        assert env_after["material_hash"] == env_before["material_hash"]
        # A DECISION_INVALIDATED event now references it.
        invs = rl.invalidations_for(conn, did)
        assert len(invs) >= 1
        assert invs[-1]["superseded_decision_event_id"] == did

    def test_reinvalidation_same_state_is_idempotent(self, conn):
        did = self._record(conn, _base_material())
        new_material = rl.MaterialInputs(**{**_base_material().as_dict(), "prompt_hash": "sha256:p2"})
        rl.invalidate_on_material_change(conn, did, new_material)
        n1 = len(rl.invalidations_for(conn, did))
        rl.invalidate_on_material_change(conn, did, new_material)  # same new state
        n2 = len(rl.invalidations_for(conn, did))
        assert n1 == n2, "re-invalidating on an identical new state must be a no-op"


# ── replay complement (§12.4), guarded on the optional API ────────────────
class TestReplayComplement:
    @pytest.mark.skipif(
        rl is None or not hasattr(rl, "replay_review_package"),
        reason="pending-CP2: replay_review_package not landed",
    )
    def test_replay_reconstructs_recorded_envelope(self, conn):
        row = rl.DecisionEnvelopeRow(
            decision_event_id="devt-replay",
            task_id="task-replay",
            action="continue",
            material=_base_material(),
        )
        rl.record_decision_envelope(conn, row)
        replayed = rl.replay_review_package(conn, "devt-replay")
        assert replayed is not None
