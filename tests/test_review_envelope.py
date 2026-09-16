"""RA-12 — immutable machine-review envelope (intake-1004).

The regression fixture in ``tests/fixtures/review_envelope/`` is FROZEN on disk:
an old review signed against ``source_abstract_v1.txt``, and the current
``source_abstract_v2.txt`` that withdraws the figure the old review approved.
Every splice of that old body into a current binding must be refused.

Hermetic, pure, NO inference.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.proactive_delegation import review_envelope as rev

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "review_envelope"
VALIDATOR_PATH = REPO_ROOT / "orchestration" / "validate_ir.py"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def old_envelope() -> dict:
    return _load("old_envelope.json")


@pytest.fixture
def old_body() -> dict:
    return _load("old_signed_review_body.json")


@pytest.fixture
def old_binding(old_envelope) -> rev.ReviewBinding:
    return rev.ReviewBinding.from_schema(old_envelope["binding"])


@pytest.fixture
def current_binding(old_binding) -> rev.ReviewBinding:
    """Current inputs: the v2 abstract, everything else unchanged."""
    return replace(
        old_binding,
        source_hash=rev.content_hash((FIXTURES / "source_abstract_v2.txt").read_bytes()),
        source_version="v2",
    )


# ── the fixture itself is what it claims to be ─────────────────────────────


def test_fixture_is_bound_to_the_old_abstract(old_binding):
    assert old_binding.source_hash == rev.content_hash(
        (FIXTURES / "source_abstract_v1.txt").read_bytes()
    )
    assert old_binding.source_hash != rev.content_hash(
        (FIXTURES / "source_abstract_v2.txt").read_bytes()
    )


def test_old_review_is_current_for_its_own_inputs(old_envelope, old_body, old_binding):
    """Control: without this, every refusal below could be a vacuous always-refuse."""
    result = rev.require_current(old_envelope, old_body, old_binding)
    assert result.current and result.reasons == ()


# ── THE regression: current abstract + old signed body is refused ──────────


def test_old_envelope_refused_against_current_abstract(old_envelope, old_body, current_binding):
    result = rev.check_binding(old_envelope, old_body, current_binding)
    assert not result.current
    assert set(result.changed_fields) == {"source_hash", "source_version"}
    with pytest.raises(rev.StaleReviewError, match="inputs changed since review"):
        rev.require_current(old_envelope, old_body, current_binding)


def test_old_body_cannot_be_rewrapped_in_a_current_envelope(old_body, current_binding):
    """The emitter refuses outright to build a current envelope around an old body."""
    with pytest.raises(rev.ReviewEnvelopeError, match="different binding"):
        rev.build_envelope(
            envelope_id="splice-1",
            binding=current_binding,
            signed_body=old_body,
            validation_status="valid",
        )


def test_forged_current_envelope_around_old_body_refused(old_envelope, old_body, current_binding):
    """Hand-forged: binding and binding_digest rewritten to the current inputs, body kept."""
    forged = copy.deepcopy(old_envelope)
    forged["binding"] = current_binding.as_schema()
    forged["binding_digest"] = current_binding.digest()
    result = rev.check_binding(forged, old_body, current_binding)
    assert not result.current
    assert result.changed_fields == ()  # the binding itself now LOOKS current…
    assert any("signed for different inputs" in r for r in result.reasons)  # …the body does not


def test_forged_body_resigned_but_output_hash_kept_refused(old_envelope, old_body, current_binding):
    forged_env = copy.deepcopy(old_envelope)
    forged_env["binding"] = current_binding.as_schema()
    forged_env["binding_digest"] = current_binding.digest()
    forged_body = dict(old_body, binding_digest=current_binding.digest())
    result = rev.check_binding(forged_env, forged_body, current_binding)
    assert not result.current
    assert any("output_hash" in r for r in result.reasons)


def test_binding_edited_without_digest_refused(old_envelope, old_body, old_binding):
    tampered = copy.deepcopy(old_envelope)
    tampered["binding"]["reviewer"]["quant"] = "Q8_0"
    result = rev.check_binding(tampered, old_body, old_binding)
    assert not result.current
    assert any("binding_digest does not match" in r for r in result.reasons)


def test_body_edit_refused(old_envelope, old_body, old_binding):
    edited = copy.deepcopy(old_body)
    edited["review"]["decision"] = "reject"
    result = rev.check_binding(old_envelope, edited, old_binding)
    assert any("output_hash" in r for r in result.reasons)


# ── every material input invalidates independently ─────────────────────────


@pytest.mark.parametrize("field", rev.MATERIAL_BINDING_FIELDS)
def test_each_material_field_invalidates(field, old_envelope, old_body, old_binding):
    value = getattr(old_binding, field)
    changed = replace(old_binding, **{field: (value or "") + "-changed"})
    result = rev.check_binding(old_envelope, old_body, changed)
    assert not result.current
    assert result.changed_fields == (field,)


def test_material_fields_cover_the_ra12_list():
    required = {
        "source_hash", "source_version", "candidate_hash", "candidate_version",
        "reviewer_model", "reviewer_quant", "prompt_bundle_hash",
        "pipeline_version", "review_schema_version",
    }
    assert required <= set(rev.MATERIAL_BINDING_FIELDS)


def test_locator_move_alone_is_not_material(old_binding):
    moved = replace(old_binding, source_ref="moved/abstract.txt")
    assert rev.changed_binding_fields(old_binding, moved) == []


def test_non_valid_emission_is_never_current(old_body, old_binding):
    env = rev.build_envelope(
        envelope_id="bad-1", binding=old_binding, signed_body=old_body,
        validation_status="schema_invalid", created_at="2026-08-05T10:00:00+00:00",
    )
    result = rev.check_binding(env, old_body, old_binding)
    assert not result.current and any("schema_invalid" in r for r in result.reasons)


# ── supersession: stale is history, never spliced into the current page ────


def _rerun(current_binding, supersedes="rev-2026-08-05-0001"):
    body = rev.sign_review(current_binding, {"decision": "reject", "confidence": 0.8,
                                             "blocking": {"tripwire": True, "blocking_issues": ["1.8x withdrawn"]}})
    env = rev.build_envelope(
        envelope_id="rev-2026-09-16-0002", binding=current_binding, signed_body=body,
        validation_status="valid", supersedes=supersedes,
        objections=[rev.new_objection("obj-2", "Summary repeats the withdrawn 1.8x figure.")],
        created_at="2026-09-16T10:00:00+00:00",
    )
    return env, body


def test_current_view_keeps_old_as_history_and_never_merges_objections(
    old_envelope, old_body, current_binding
):
    new_env, new_body = _rerun(current_binding)
    view = rev.current_view(
        [old_envelope, new_env],
        {old_envelope["envelope_id"]: old_body, new_env["envelope_id"]: new_body},
        current_binding,
    )
    assert view.current is new_env
    assert view.stale_history == (old_envelope,)
    assert [o["objection_id"] for o in view.objections] == ["obj-2"]
    assert old_envelope == _load("old_envelope.json")  # history retained unchanged


def test_current_view_with_only_stale_review_shows_nothing(old_envelope, old_body, current_binding):
    view = rev.current_view([old_envelope], {old_envelope["envelope_id"]: old_body}, current_binding)
    assert view.current is None and view.objections == ()
    assert view.refusals and not view.refusals[0].current


def test_superseded_but_still_matching_review_is_not_current(old_envelope, old_body, old_binding):
    """A re-run on identical inputs supersedes the first; the first must drop out."""
    rerun_body = rev.sign_review(old_binding, {"decision": "approve", "confidence": 0.5,
                                               "blocking": {"tripwire": False, "blocking_issues": []}})
    rerun = rev.build_envelope(
        envelope_id="rerun", binding=old_binding, signed_body=rerun_body,
        validation_status="valid", supersedes=old_envelope["envelope_id"],
        created_at="2026-08-06T10:00:00+00:00",
    )
    view = rev.current_view(
        [old_envelope, rerun],
        {old_envelope["envelope_id"]: old_body, "rerun": rerun_body},
        old_binding,
    )
    assert view.current is rerun
    assert any(r.reasons == ("superseded",) for r in view.refusals)


def test_two_unlinked_current_reviews_are_refused(old_envelope, old_body, old_binding):
    twin = copy.deepcopy(old_envelope)
    twin["envelope_id"] = "twin"
    with pytest.raises(rev.ReviewEnvelopeError, match="more than one current review"):
        rev.current_view([old_envelope, twin], {"rev-2026-08-05-0001": old_body, "twin": old_body}, old_binding)


@pytest.mark.parametrize(
    "links, fragment",
    [
        ({"a": "ghost"}, "unknown envelope"),
        ({"b": "a", "c": "a"}, "superseded twice"),
        ({"a": "b", "b": "a"}, "cycle"),
    ],
)
def test_supersession_chain_defects(links, fragment):
    envs = [{"envelope_id": i, "supersedes": links.get(i)} for i in ("a", "b", "c")]
    assert any(fragment in p for p in rev.check_supersession_chain(envs))


# ── objections: unverified leads, never corroboration ──────────────────────


def test_emitted_objections_are_unverified_leads(old_envelope):
    assert all(o["status"] == rev.UNVERIFIED_LEAD for o in old_envelope["objections"])
    assert rev.independent_corroboration_count(old_envelope["objections"]) == 0
    assert rev.confirmed_objections(old_envelope["objections"]) == []


def test_resolution_by_stale_primary_artifact_refused(old_envelope, current_binding):
    stale_hash = rev.content_hash((FIXTURES / "source_abstract_v1.txt").read_bytes())
    with pytest.raises(rev.ReviewEnvelopeError, match="not current"):
        rev.resolve_objection(
            old_envelope["objections"][0], status=rev.CONFIRMED, kind=rev.PRIMARY_ARTIFACT,
            ref="abstract.txt", resolver_hash=stale_hash, current=current_binding,
        )


def test_resolution_by_current_primary_artifact_still_not_corroboration(old_envelope, current_binding):
    resolved = rev.resolve_objection(
        old_envelope["objections"][0], status=rev.CONFIRMED, kind=rev.PRIMARY_ARTIFACT,
        ref="abstract.txt", resolver_hash=current_binding.source_hash, current=current_binding,
    )
    assert resolved["independent_corroboration"] is False
    assert rev.confirmed_objections([resolved]) == [resolved]
    assert rev.independent_corroboration_count([resolved]) == 0
    assert old_envelope["objections"][0]["status"] == rev.UNVERIFIED_LEAD  # input untouched


def test_objection_claiming_corroboration_is_a_defect():
    bad = dict(rev.new_objection("x", "y"), independent_corroboration=True)
    with pytest.raises(rev.ReviewEnvelopeError):
        rev.independent_corroboration_count([bad])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update(independent_corroboration=True),
        lambda o: o.update(status="confirmed"),  # resolved status without resolved_by
        lambda o: o.update(resolved_by={"kind": "objective_verifier", "ref": "r", "content_hash": "sha256:" + "0" * 64}),
        lambda o: o.update(status="corroborated"),
    ],
)
def test_schema_rejects_malformed_objections(mutate, old_envelope):
    env = copy.deepcopy(old_envelope)
    mutate(env["objections"][0])
    assert rev.schema_errors(env)


# ── validate_ir CLI wiring ─────────────────────────────────────────────────


def _cli(doc: dict) -> int:
    return subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "review_envelope", "-"],
        input=json.dumps(doc), capture_output=True, text=True,
    ).returncode


def test_cli_accepts_fixture_and_rejects_missing_supersedes(old_envelope):
    assert _cli(old_envelope) == 0
    broken = copy.deepcopy(old_envelope)
    del broken["supersedes"]
    assert _cli(broken) == 2
