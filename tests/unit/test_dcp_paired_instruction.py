from dataclasses import replace

import pytest

from scripts.benchmark import dcp_paired_instruction as instrument
from scripts.benchmark.dcp_paired_instruction import SEALED_CASES, assemble_pairs
from scripts.benchmark.paired_edit_instrument import FIXTURES


def test_native_assembly_has_both_conditions_and_one_shared_bundle():
    pairs = assemble_pairs()
    assert [p["problem_id"] for p in pairs] == [f.id for f in FIXTURES] == ["clamp", "unique"]
    for fixture, pair in zip(FIXTURES, pairs):
        assert fixture.initial in pair["bundle_context"]
        assert pair["bundle_manifest"]["fits"]
        assert len(pair["identity_sha256"]) == 64
        descriptive, minimal = pair["conditions"]
        assert [descriptive["condition"], minimal["condition"]] == ["descriptive", "minimal"]
        assert descriptive["instruction"] == fixture.descriptive
        assert minimal["instruction"] == fixture.minimal
        assert pair["bundle_context"] in descriptive["prompt"]
        assert pair["bundle_context"] in minimal["prompt"]
        assert descriptive["prompt"].replace(fixture.descriptive, fixture.minimal, 1) == minimal["prompt"]


def test_bad_manifests_fail_closed():
    a, b = SEALED_CASES
    bad = [(), (a,), (a, a), (replace(a, problem_id="unknown"), b),
           (replace(a, instructions=a.instructions[:1]), b),
           (replace(a, instructions=(a.instructions[0], a.instructions[0])), b),
           (replace(a, instructions=(a.instructions[0], ("other", "x"))), b),
           (replace(a, initial="changed"), b),
           (replace(a, instructions=(("descriptive", "changed"), a.instructions[1])), b)]
    for cases in bad:
        with pytest.raises(ValueError):
            assemble_pairs(cases)


def test_changed_source_seal_fails_closed(monkeypatch):
    monkeypatch.setattr(instrument, "FIXTURE_SEAL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="fixture seal"):
        assemble_pairs()


def test_native_prompt_builder_axis_drift_fails_closed(monkeypatch):
    original = instrument.build_root_lm_prompt
    calls = iter(("first", "second"))

    def drifting_builder(**kwargs):
        return original(**kwargs) + "\nmetadata: " + next(calls)

    monkeypatch.setattr(instrument, "build_root_lm_prompt", drifting_builder)
    with pytest.raises(ValueError, match="controlled axis"):
        assemble_pairs()
