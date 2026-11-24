"""AEI-REV-FLIPS: per-arm fix/break counts and the Kamoi-style fairness line."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "analysis"))

import reviewer_calibration_report as rcr  # noqa: E402


def _row(arm: str, cid: str, decision: str, gold: str) -> dict:
    return {
        "reviewer_model_quant": arm,
        "candidate_id": cid,
        "decision": decision,
        "gold_label": gold,
        "confidence": 0.8,
    }


ROWS = [
    # arm A: 2 fixes, 1 worker-bad missed, 1 break, 1 worker-good kept
    _row("A", "1", "reject", "reject"),
    _row("A", "2", "reject", "reject"),
    _row("A", "3", "approve", "reject"),
    _row("A", "4", "reject", "accept"),
    _row("A", "5", "approve", "accept"),
    # arm B: no fixes, 2 breaks
    _row("B", "1", "approve", "reject"),
    _row("B", "4", "reject", "accept"),
    _row("B", "5", "reject", "accept"),
    # non-terminal and un-golded rows must not count
    _row("B", "6", "escalate", "reject"),
    _row("B", "7", "reject", ""),
]


def _by_arm(report):
    return {g["group"]["reviewer_model_quant"]: g["metrics"] for g in report["groups"]}


def test_fix_and_break_counts_per_arm():
    arms = _by_arm(rcr.build_report(ROWS))
    assert (arms["A"]["fix_count"], arms["A"]["n_worker_wrong"]) == (2, 3)
    assert (arms["A"]["break_count"], arms["A"]["n_worker_right"]) == (1, 2)
    assert (arms["B"]["fix_count"], arms["B"]["n_worker_wrong"]) == (0, 1)
    assert (arms["B"]["break_count"], arms["B"]["n_worker_right"]) == (2, 2)


def test_break_count_equals_false_reject_count():
    for m in _by_arm(rcr.build_report(ROWS)).values():
        assert m["break_count"] == m["fr_rate"]["successes"]


def test_fairness_defaults_unrecorded_and_not_fair():
    fl = rcr.build_report(ROWS)["fairness"]
    assert fl["initial_prompt_strength"] == "unrecorded"
    assert fl["fair"] is False


def test_fairness_fair_only_when_all_three_met():
    fl = rcr.fairness_line(
        {"initial_prompt_strength": "comprehensive", "oracle_stop": False, "matched_token_baseline": True}
    )
    assert fl["fair"] is True
    assert rcr.fairness_line({**fl, "oracle_stop": True})["fair"] is False
    assert rcr.fairness_line({**fl, "matched_token_baseline": False})["fair"] is False


def test_markdown_prints_new_columns_and_fairness_line():
    rep = rcr.build_report(
        ROWS,
        fairness={"initial_prompt_strength": "basic", "oracle_stop": False, "matched_token_baseline": False},
    )
    md = rcr.render_markdown(rep)
    header = next(line for line in md.splitlines() if line.startswith("| reviewer"))
    assert "| fix | break |" in header
    sep = md.splitlines()[md.splitlines().index(header) + 1]
    assert sep.count("|") - 1 == header.count("|") - 1
    assert "| 2/3 | 1/2 |" in md
    assert "Fairness (Kamoi-style): initial-prompt strength = basic; oracle stop = no; matched-token baseline = absent" in md
