"""AEI-RERUN-ATTEST: paired attestation carries R ref + rerun-flip string; no-R refused."""

import json

import pytest

from scripts.autopilot.paired_attestation import (
    append_paired_ledger_row,
    build_paired_attestation,
    format_paired_claim,
)
from scripts.autopilot.paired_stats import (
    QuestionOutcome,
    RerunArmMissingError,
    apply_rerun_noise_rule,
    mcnemar_verdict,
    rerun_flip_rate,
)


def _vec(correct):
    return {f"q{i}": QuestionOutcome(qid=f"q{i}", suite="s", correct=bool(v), trial_id=0) for i, v in enumerate(correct)}


def _verdict(b, c, n, rerun):
    v = mcnemar_verdict(b, c)
    return apply_rerun_noise_rule(v, b=b, c=c, n_pairs=n, rerun=rerun)


def _rerun(f_ar, f_ra, n_extra=20):
    a = [1] * f_ar + [0] * f_ra + [1] * n_extra
    r = [0] * f_ar + [1] * f_ra + [1] * n_extra
    return rerun_flip_rate(_vec(a), _vec(r))


def test_attestation_with_r_carries_rerun_ref_and_flip_string():
    v = _verdict(30, 2, 100, _rerun(1, 0))
    assert v["verdict"] == "a_better"
    rec = build_paired_attestation(
        v, date="2026-11-24", attest_ref="runs/ab/summary.json", rerun_ref="runs/r/summary.json"
    )
    assert "rerun-flip 1/21" in rec["claim"]
    assert rec["claim"].endswith("attest runs/ab/summary.json, rerun runs/r/summary.json]")
    assert rec["claim"].startswith("a_better [P-PAIRED, 32/mcnemar, ")
    assert rec["rerun_ref"] == "runs/r/summary.json"
    assert rec["rerun_flip"] == "rerun-flip 1/21"


def test_within_rerun_noise_renders_no_claim_form():
    v = _verdict(8, 1, 100, _rerun(10, 10, n_extra=0))
    assert v["reason"] == "within_rerun_noise"
    s = format_paired_claim(v, date="2026-11-24", attest_ref="a.json", rerun_ref="r.json")
    assert s == "within rerun noise [P-AB-1, rerun-flip 20/20, 2026-11-24]"


def test_no_r_is_refused_for_decision_use():
    v = mcnemar_verdict(30, 2)  # no rerun block
    with pytest.raises(RerunArmMissingError):
        build_paired_attestation(v, date="2026-11-24", attest_ref="a.json", rerun_ref="r.json")


def test_missing_rerun_ref_is_refused():
    v = _verdict(30, 2, 100, _rerun(1, 0))
    for ref in (None, "", "  "):
        with pytest.raises(RerunArmMissingError):
            build_paired_attestation(v, date="2026-11-24", attest_ref="a.json", rerun_ref=ref)


def test_ledger_row_roundtrip_and_refusal(tmp_path):
    v = _verdict(30, 2, 100, _rerun(1, 0))
    rec = build_paired_attestation(
        v, date="2026-11-24", attest_ref="a.json", rerun_ref="r.json", arm_a="A", arm_b="B"
    )
    p = append_paired_ledger_row(tmp_path / "ledger.jsonl", rec)
    row = json.loads(p.read_text().splitlines()[0])
    assert row["rerun_ref"] == "r.json" and row["rerun_flip"] == "rerun-flip 1/21"
    with pytest.raises(RerunArmMissingError):
        append_paired_ledger_row(tmp_path / "x.jsonl", {"claim": "x"})
    assert not (tmp_path / "x.jsonl").exists()
