#!/usr/bin/env python3
"""Tests for the reviewer calibration report (H4 RC-4) + gold-label pipeline (RC-2).

Hermetic: synthetic ledger/decision fixtures + tiny synthetic corpus; a CLI smoke
run against a small decisions JSONL. Reads the REAL corpus manifest.json (small,
no rows) only to confirm the instrument stamp binds to nearmiss-v1. NO inference.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import reviewer_calibration_report as rcr  # noqa: E402

from src.proactive_delegation.gold_labels import (  # noqa: E402
    MULTI_ORACLE,
    OBSERVATION,
    SINGLE_ORACLE,
    OracleOutcome,
    corpus_row_gold,
    outcome_from_eval_scorer,
    outcome_from_gate_result,
    outcomes_from_verification_report,
    resolve_gold_label,
)


# --------------------------------------------------------------------------- #
# RC-4 metric primitives (EV-tier-mirrored)
# --------------------------------------------------------------------------- #
def test_wilson_interval_brackets_point_estimate():
    lo, hi = rcr.wilson_interval(2, 10)
    assert 0.0 <= lo < 0.2 < hi <= 1.0
    assert rcr.wilson_interval(0, 0) == (0.0, 1.0)


def test_brier_ece_roc_auc_known_values():
    # perfectly calibrated + perfectly separating
    conf = [0.0, 0.0, 1.0, 1.0]
    corr = [0.0, 0.0, 1.0, 1.0]
    assert rcr.brier(conf, corr) == 0.0
    assert rcr.ece(conf, corr) == 0.0
    assert rcr.roc_auc(conf, corr) == 1.0
    # AUC 0.75 worked example (0.35 pos ranks below 0.4 neg)
    assert math.isclose(rcr.roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
    # single class -> AUC undefined
    assert rcr.roc_auc([0.2, 0.3], [1, 1]) is None
    assert rcr.brier([], []) is None


# --------------------------------------------------------------------------- #
# RC-4 report over synthetic decisions
# --------------------------------------------------------------------------- #
def _fa_fr_fixture() -> list[dict]:
    base = {
        "reviewer_model_quant": "GLM-5.2/UD-IQ2_M",
        "grading_model": "Qwen3-Coder-30B",
        "rubric_version": "v3",
        "corpus_id": "nearmiss-v1",
        "domain": "code",
    }
    rows: list[dict] = []
    i = 0
    # 10 actually-bad: 2 false-accept, 8 true-reject
    for _ in range(2):
        rows.append({**base, "decision_id": f"d{i}", "candidate_id": f"c{i}",
                     "decision": "approve", "gold_label": "fail", "confidence": 0.7}); i += 1
    for _ in range(8):
        rows.append({**base, "decision_id": f"d{i}", "candidate_id": f"c{i}",
                     "decision": "reject", "gold_label": "fail", "confidence": 0.8}); i += 1
    # 10 actually-good: 3 false-reject, 7 true-accept
    for _ in range(3):
        rows.append({**base, "decision_id": f"d{i}", "candidate_id": f"c{i}",
                     "decision": "reject", "gold_label": "accept", "confidence": 0.6}); i += 1
    for _ in range(7):
        rows.append({**base, "decision_id": f"d{i}", "candidate_id": f"c{i}",
                     "decision": "approve", "gold_label": "accept", "confidence": 0.9}); i += 1
    # 2 parse failures (no decision)
    for _ in range(2):
        rows.append({**base, "decision_id": f"d{i}", "candidate_id": f"c{i}",
                     "decision": None, "gold_label": None, "confidence": None}); i += 1
    return rows


def test_build_report_fa_fr_rates():
    report = rcr.build_report(_fa_fr_fixture(), k=2)
    assert report["measurement"]["grade"] == "observation"
    assert report["n_groups"] == 1
    m = report["overall"]
    assert math.isclose(m["fa_rate"]["rate"], 0.2)
    assert math.isclose(m["fr_rate"]["rate"], 0.3)
    assert math.isclose(m["fa_fr_ratio"], 0.2 / 0.3, rel_tol=1e-9)
    assert math.isclose(m["acceptance_rate"]["rate"], 9 / 20)
    assert math.isclose(m["parse_failure_rate"]["rate"], 2 / 22)
    # calibration computed over the 20 golded terminal rows
    assert m["calibration_n"] == 20
    assert m["brier"] is not None and m["ece"] is not None and m["auc"] is not None
    # wilson interval present + brackets the point estimate
    lo, hi = m["fa_rate"]["wilson95"]
    assert lo <= 0.2 <= hi


def test_fa_fr_ratio_null_when_no_false_rejects():
    rows = [
        {"corpus_id": "c", "domain": "code", "decision_id": "a", "candidate_id": "a",
         "decision": "reject", "gold_label": "fail"},  # TR (bad)
        {"corpus_id": "c", "domain": "code", "decision_id": "b", "candidate_id": "b",
         "decision": "approve", "gold_label": "accept"},  # TA (good) -> FR rate 0
    ]
    m = rcr.build_report(rows)["overall"]
    assert m["fr_rate"]["rate"] == 0.0
    assert m["fa_fr_ratio"] is None  # divide-by-zero guarded


def test_consistency_rate_and_passk():
    base = {"corpus_id": "c", "domain": "code", "gold_label": "fail"}
    rows = [
        # candidate A: 2 runs, both reject (agree) + both correct
        {**base, "decision_id": "a1", "candidate_id": "A", "decision": "reject"},
        {**base, "decision_id": "a2", "candidate_id": "A", "decision": "reject"},
        # candidate B: 2 runs, disagree (reject vs approve)
        {**base, "decision_id": "b1", "candidate_id": "B", "decision": "reject"},
        {**base, "decision_id": "b2", "candidate_id": "B", "decision": "approve"},
    ]
    m = rcr.build_report(rows, k=2)["overall"]
    cr = m["consistency_rate"]
    assert cr["n_candidates_multi_run"] == 2
    assert cr["n_all_agree"] == 1
    assert math.isclose(cr["rate"], 0.5)
    passk = m["pass_hat_2"]
    # A: 2 correct rejects -> pass; B: reject(correct)+approve(incorrect) -> fail
    assert passk["n_candidates_ge_k"] == 2
    assert passk["n_pass"] == 1
    assert math.isclose(passk["rate"], 0.5)


# --------------------------------------------------------------------------- #
# RC-4 CLI smoke (synthetic decisions -> JSON + markdown)
# --------------------------------------------------------------------------- #
def test_cli_smoke_writes_json_and_md(tmp_path):
    decisions = tmp_path / "decisions.jsonl"
    with open(decisions, "w") as fh:
        for row in _fa_fr_fixture():
            fh.write(json.dumps(row) + "\n")
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    rc = rcr.main(["--decisions", str(decisions), "--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0
    report = json.loads(out_json.read_text())
    assert report["measurement"]["grade"] == "observation"
    assert math.isclose(report["overall"]["fa_rate"]["rate"], 0.2)
    md = out_md.read_text()
    assert "observation-grade" in md
    assert "OVERALL" in md
    assert "FA/FR" in md


def test_cli_run_manifest_can_stamp_p_rev1_decision_grade(tmp_path):
    decisions = tmp_path / "decisions.jsonl"
    with open(decisions, "w") as fh:
        for row in _fa_fr_fixture():
            fh.write(json.dumps(row) + "\n")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "glm52_reviewer_corpus_direct_run_manifest.v1",
                "measurement_protocol": "p_rev1",
                "observation_only": False,
                "protocol_attestation": "attest-test",
            }
        )
    )
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    rc = rcr.main(
        [
            "--decisions",
            str(decisions),
            "--run-manifest",
            str(manifest),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert rc == 0
    report = json.loads(out_json.read_text())
    assert report["measurement"]["grade"] == "decision"
    assert report["measurement"]["protocol"] == "P-REV-1"
    assert report["measurement"]["run_manifest"]["protocol_attestation"] == "attest-test"
    md = out_md.read_text()
    assert "decision-grade" in md
    assert "attest-test" in md


def test_cli_corpus_join(tmp_path):
    # decisions carry only candidate_id + decision; gold comes from corpus join.
    decisions = tmp_path / "d.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    with open(decisions, "w") as fh:
        fh.write(json.dumps({"decision_id": "x", "candidate_id": "row-1", "decision": "approve"}) + "\n")
    with open(corpus, "w") as fh:
        fh.write(json.dumps({"row_id": "row-1", "gold_label": "fail", "gold_source": "gate",
                             "gold_instrument_version": "v1", "domain": "code",
                             "corpus_id": "nearmiss-v1"}) + "\n")
    rows = rcr.join_corpus_gold(rcr.load_decisions_jsonl(decisions), corpus)
    assert rows[0]["gold_label"] == "fail"
    m = rcr.build_report(rows)["overall"]
    assert m["fa_rate"]["rate"] == 1.0  # approve on a fail == false-accept


def test_instrument_binds_to_real_corpus_manifest():
    manifest = Path("/mnt/raid0/llm/datasets/nearmiss-corpus-v1/manifest.json")
    if not manifest.exists():
        pytest.skip("real corpus manifest not present")
    meta = json.loads(manifest.read_text())
    assert meta["corpus_id"] == "nearmiss-v1"
    # the report's instrument stamp can carry real corpus provenance
    report = rcr.build_report(
        _fa_fr_fixture(),
        instrument={"corpus_id": meta["corpus_id"], "content_sha256": meta["content_sha256"][:8]},
    )
    assert report["instrument"]["corpus_id"] == "nearmiss-v1"


# --------------------------------------------------------------------------- #
# RC-2 — gold-label resolution pipeline
# --------------------------------------------------------------------------- #
def test_rc2_two_agreeing_oracles_gate_worthy():
    outcomes = [
        OracleOutcome(source="gate_runner:unit", verdict="pass", instrument_name="unit"),
        OracleOutcome(source="evaltower:programmatic", verdict="pass", instrument_name="prog"),
    ]
    res = resolve_gold_label(outcomes)
    assert res.gold_label == "pass"
    assert res.gold_confidence == MULTI_ORACLE
    assert res.gate_worthy is True
    assert res.needs_arbitration is False


def test_rc2_disagreeing_oracles_route_to_arbitration():
    outcomes = [
        OracleOutcome(source="gate_runner:unit", verdict="pass"),
        OracleOutcome(source="evaltower:prog", verdict="fail"),
    ]
    res = resolve_gold_label(outcomes)
    assert res.gold_label is None  # mark, don't decide
    assert res.ambiguous_tail is True
    assert res.needs_arbitration is True
    assert res.gate_worthy is False


def test_rc2_single_oracle_not_gate_worthy():
    res = resolve_gold_label([OracleOutcome(source="gate_runner:unit", verdict="fail")])
    assert res.gold_label == "fail"
    assert res.gold_confidence == SINGLE_ORACLE
    assert res.gate_worthy is False
    assert res.ambiguous_tail is True  # corpus rule: single oracle -> arbitration


def test_rc2_corpus_fallback_and_human_arbitration():
    corpus_row = {"gold_label": "reject", "gold_confidence": "multi_oracle",
                  "gold_source": "c-crab", "gold_instrument_version": "c-crab-v1"}
    res = resolve_gold_label([], corpus_row=corpus_row)
    assert res.gold_label == "reject"
    assert res.gate_worthy is True
    assert res.gold_source == "c-crab"
    # human arbitration is authoritative + gate-worthy
    res2 = resolve_gold_label([OracleOutcome("x", "pass")], human_arbitration="fail")
    assert res2.gold_label == "fail"
    assert res2.gate_worthy is True
    assert res2.gold_source == "human_arbitration"


def test_rc2_no_signal_is_observation():
    res = resolve_gold_label([OracleOutcome("x", "inconclusive")])
    assert res.gold_label is None
    assert res.gold_confidence == OBSERVATION
    assert res.gate_worthy is False


def test_rc2_normalizers():
    assert outcome_from_gate_result({"gate_name": "unit", "passed": True}).verdict == "pass"
    assert outcome_from_gate_result({"gate_name": "unit", "passed": False}).verdict == "fail"
    report = {"schema_version": "1.0.0", "checks": [
        {"check_id": "t1", "kind": "test", "outcome": "pass"},
        {"check_id": "t2", "kind": "lint", "outcome": "inconclusive", "inconclusive_reason": "x"},
    ]}
    outs = outcomes_from_verification_report(report)
    assert [o.verdict for o in outs] == ["pass", "inconclusive"]
    # only the conclusive check votes -> single_oracle
    assert resolve_gold_label(outs).gold_confidence == SINGLE_ORACLE
    # eval-tower scorer
    assert outcome_from_eval_scorer({"correct": True, "scoring_method": "exact_match"}).verdict == "pass"
    ce = outcome_from_eval_scorer(
        {"scoring_method": "code_execution", "scoring_config": {"pass_rate": 0.5}},
        pass_threshold=1.0,
    )
    assert ce.verdict == "fail"


def test_rc2_corpus_row_gold_reader():
    label, conf, ver = corpus_row_gold(
        {"gold_label": "REJECT", "gold_confidence": "single_oracle", "gold_instrument_version": "v9"}
    )
    assert label == "reject" and conf == "single_oracle" and ver == "v9"
