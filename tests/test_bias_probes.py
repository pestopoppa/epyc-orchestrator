"""RM-5 — bias-robustness probe set (zero-inference half). Pure, hermetic, NO inference."""

from __future__ import annotations

import pytest

from src.proactive_delegation import bias_probes as bp
from src.proactive_delegation.candidate_sanitizer import sanitize_candidate_package, sanitized_view_text


def _rows(n=60):
    rows = []
    for i in range(n):
        rows.append({
            "row_id": f"r{i:03d}",
            "gold_label": ["reject", "accept", "fail", "pass", None][i % 5],
            "task": f"Implement function f{i} that returns the sum of a list.",
            "candidate": f"def f{i}(xs):\n    return sum(xs)\n",
        })
    return rows


@pytest.fixture
def probe_set():
    return bp.build_probe_set(_rows(), seed=42)


# ── injection table ────────────────────────────────────────────────────────


def test_six_kinds_with_declared_directions():
    assert set(bp.KINDS) == {
        "authority", "self_declared_correctness", "renaming",
        "reverse_authority", "misleading_task", "illusory_complexity",
    }
    assert {k for k in bp.KINDS if bp.INJECTIONS[k][0] == bp.REJECT} == {"reverse_authority"}


def test_injection_changes_only_its_target_field_and_does_not_mutate():
    view = {"objective": "T", "outputs": "C"}
    out = bp.inject(view, "misleading_task", "suffix")
    assert view == {"objective": "T", "outputs": "C"}
    assert out["outputs"] == "C" and out["objective"].startswith("T") and out["objective"] != "T"
    pre = bp.inject(view, "authority", "prefix")
    assert pre["objective"] == "T" and pre["outputs"].endswith("C")


def test_manifest_hash_tracks_templates(monkeypatch):
    before = bp.manifest_hash()
    monkeypatch.setitem(bp.INJECTIONS, "authority", (bp.ACCEPT, "outputs", "changed"))
    assert bp.manifest_hash() != before


# ── probe placement: a flip must be a HARMFUL error ────────────────────────


def test_accept_bias_only_on_bad_gold_reverse_authority_only_on_good(probe_set):
    for p in probe_set.probes:
        if p.bias_direction == bp.ACCEPT:
            assert p.gold_polarity == "bad"
        else:
            assert p.kind == "reverse_authority" and p.gold_polarity == "good"


def test_unlabelled_rows_are_skipped_and_listed(probe_set):
    assert len(probe_set.skipped_no_gold) == 12
    assert len(probe_set.probes) == 48


def test_all_accept_kinds_are_used(probe_set):
    counts = probe_set.kind_counts()
    assert set(counts) == set(bp.KINDS)


def test_injection_survives_the_production_assembly_sanitizer(probe_set):
    """An injection the sanitizer strips would measure the sanitizer, not the reviewer."""
    for p in probe_set.probes:
        text = bp.INJECTIONS[p.kind][2]
        pkg = sanitize_candidate_package(bp.to_full_package(p.injected_view, package_id=p.probe_id, task_ref=p.row_id))
        assert text in sanitized_view_text(pkg["sanitized_view"])
        assert pkg["sanitized_view"]["sanitization"]["removed_fields"] == []
        clean = sanitize_candidate_package(bp.to_full_package(p.clean_view, package_id=p.probe_id, task_ref=p.row_id))
        assert text not in sanitized_view_text(clean["sanitized_view"])


# ── controls: held-out, determinism, randomization ─────────────────────────


def test_overlap_with_calibration_set_is_refused():
    with pytest.raises(bp.ProbeSetError, match="overlap"):
        bp.build_probe_set(_rows(), seed=1, held_out_from={"r003", "zzz"})


def test_duplicate_rows_refused():
    rows = _rows(3) + _rows(1)
    with pytest.raises(bp.ProbeSetError, match="duplicate"):
        bp.build_probe_set(rows, seed=1)


def test_deterministic_for_a_seed_and_different_across_seeds():
    a = bp.build_probe_set(_rows(), seed=7)
    b = bp.build_probe_set(_rows(), seed=7)
    c = bp.build_probe_set(_rows(), seed=8)
    assert a == b
    assert [p.probe_id for p in a.probes] != [p.probe_id for p in c.probes]


def test_position_and_presentation_order_are_randomized(probe_set):
    assert {p.position for p in probe_set.probes} == {"prefix", "suffix"}
    assert {p.injected_first for p in probe_set.probes} == {True, False}
    ordered_ids = [p.row_id for p in probe_set.probes]
    assert ordered_ids != sorted(ordered_ids)


# ── swap augmentation ──────────────────────────────────────────────────────


def test_swap_augmented_adds_reversals_once():
    assert bp.swap_augmented([("a", "b"), ("b", "a"), ("a", "c")]) == [("a", "b"), ("b", "a"), ("a", "c"), ("c", "a")]
    bp.require_swap_augmented(bp.swap_augmented([("a", "b")]))


def test_single_order_pairwise_jobs_are_refused():
    with pytest.raises(bp.ProbeSetError, match="swap-augmented"):
        bp.require_swap_augmented([("a", "b"), ("b", "a"), ("a", "c")])


# ── scoring ────────────────────────────────────────────────────────────────


def test_normalize_decision():
    assert bp.normalize_decision("approve") == bp.ACCEPT
    for d in ("reject", "reject_to_empty", "request_changes"):
        assert bp.normalize_decision(d) == bp.REJECT
    for d in ("request_evidence", "escalate", None, "garbage"):
        assert bp.normalize_decision(d) is None


def _decisions(ps, flip_every=0, abstain_every=0):
    out = {}
    for i, p in enumerate(ps.probes):
        clean = "reject" if p.gold_polarity == "bad" else "approve"
        injected = clean
        if flip_every and i % flip_every == 0:
            injected = "approve" if p.bias_direction == bp.ACCEPT else "reject"
        if abstain_every and i % abstain_every == 1:
            injected = "request_evidence"
        out[p.probe_id] = (clean, injected)
    return out


def test_perfectly_robust_reviewer(probe_set):
    res = bp.score_robustness(probe_set, _decisions(probe_set))
    assert res.robustness_rate == 1.0 and res.harmful_flip_rate == 0.0
    assert res.n_scored == 48


def test_flips_and_abstentions_are_counted_with_stated_denominator(probe_set):
    res = bp.score_robustness(probe_set, _decisions(probe_set, flip_every=4, abstain_every=6))
    assert res.n_abstained == 8
    assert res.n_scored == 40
    assert res.n_harmful_flips == 12
    assert res.robustness_rate == pytest.approx(28 / 40)
    d = res.as_dict()
    assert d["denominator"] == 40 and d["harmful_flip_direction"] == "lower_is_better"
    assert sum(v["scored"] for v in d["per_kind"].values()) == 40


def test_flip_against_bias_direction_is_not_harmful(probe_set):
    p = probe_set.probes[0]
    decisions = _decisions(probe_set)
    against = bp.REJECT if p.bias_direction == bp.ACCEPT else bp.ACCEPT
    clean = "approve" if against == bp.REJECT else "reject"
    decisions[p.probe_id] = (clean, "reject" if against == bp.REJECT else "approve")
    res = bp.score_robustness(probe_set, decisions)
    assert res.n_unchanged == 47 and res.n_harmful_flips == 0


def test_missing_or_foreign_decisions_are_refused(probe_set):
    d = _decisions(probe_set)
    d.pop(probe_set.probes[0].probe_id)
    with pytest.raises(bp.ProbeSetError, match="no decision pair"):
        bp.score_robustness(probe_set, d)
    d = _decisions(probe_set)
    d["ghost"] = ("approve", "approve")
    with pytest.raises(bp.ProbeSetError, match="not in this set"):
        bp.score_robustness(probe_set, d)


def test_empty_scoring_yields_none_not_zero():
    empty = bp.build_probe_set([], seed=1)
    res = bp.score_robustness(empty, {})
    assert res.robustness_rate is None and res.harmful_flip_rate is None
