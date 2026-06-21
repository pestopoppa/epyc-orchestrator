from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autopilot import bsv_paired_report


def _row(
    trial_id: int,
    fingerprint: str,
    outcomes: list[tuple[str, bool]],
    *,
    avg_prompt_tokens: int = 1000,
    routing_distribution: dict[str, float] | None = None,
) -> dict:
    return {
        "trial_id": trial_id,
        "config_snapshot": {"config_fingerprint": fingerprint},
        "avg_prompt_tokens": avg_prompt_tokens,
        "routing_distribution": routing_distribution or {"frontdoor": 1.0},
        "eval_details": {
            "question_results": [
                {"qid": qid, "suite": "suite", "correct": correct}
                for qid, correct in outcomes
            ]
        },
    }


def test_trial_pair_blocks_on_prior_pass_regression() -> None:
    rows = [
        _row(1, "base", [("q1", True), ("q2", True), ("q3", False)]),
        _row(2, "cand", [("q1", True), ("q2", False), ("q3", False)]),
    ]

    report = bsv_paired_report.build_trial_pair_report(
        rows,
        baseline_trial=1,
        candidate_trial=2,
        min_shared_qids=3,
    )

    assert report["gate_decision"] == "block"
    assert report["paired_stats"]["shared_qids"] == 3
    assert report["signature_diff"]["severity"] == "blocking"
    assert "behavior signature severity is blocking" in report["blockers"]
    assert report["baseline_signature"]["archive_member_id"] == "trial:1"
    assert report["candidate_signature"]["archive_member_id"] == "trial:2"


def test_trial_pair_passes_watch_when_no_regression_and_enough_coverage() -> None:
    rows = [
        _row(1, "base", [("q1", True), ("q2", False), ("q3", False)]),
        _row(
            2,
            "cand",
            [("q1", True), ("q2", True), ("q3", False)],
            routing_distribution={"frontdoor": 0.1, "worker": 0.9},
        ),
    ]

    report = bsv_paired_report.build_trial_pair_report(
        rows,
        baseline_trial=1,
        candidate_trial=2,
        min_shared_qids=3,
    )

    assert report["gate_decision"] == "pass"
    assert report["signature_diff"]["severity"] == "watch"
    assert report["paired_stats"]["delta_b_minus_a"] > 0
    assert report["blockers"] == []


def test_fingerprint_pair_uses_majority_vectors_and_records_trials() -> None:
    rows = [
        _row(1, "base", [("q1", True), ("q2", False), ("q3", False)]),
        _row(2, "base", [("q1", True), ("q2", False), ("q3", True)]),
        _row(3, "cand", [("q1", True), ("q2", True), ("q3", True)]),
        _row(4, "cand", [("q1", True), ("q2", True), ("q3", False)]),
    ]

    report = bsv_paired_report.build_fingerprint_pair_report(
        rows,
        baseline_fingerprint="base",
        candidate_fingerprint="cand",
        min_shared_qids=2,
    )

    assert report["comparison_type"] == "fingerprint_pair"
    assert report["gate_decision"] == "pass"
    assert report["paired_stats"]["shared_qids"] == 2
    assert report["baseline_trials"] == [1, 2]
    assert report["candidate_trials"] == [3, 4]


def test_blocks_when_shared_qid_coverage_is_too_low() -> None:
    rows = [
        _row(1, "base", [("q1", True)]),
        _row(2, "cand", [("q1", True)]),
    ]

    report = bsv_paired_report.build_trial_pair_report(
        rows,
        baseline_trial=1,
        candidate_trial=2,
        min_shared_qids=2,
    )

    assert report["gate_decision"] == "block"
    assert report["blockers"] == ["shared_qids 1 < 2"]


def test_cli_trial_pair_returns_nonzero_on_block(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    rows = [
        _row(1, "base", [("q1", True), ("q2", True)]),
        _row(2, "cand", [("q1", True), ("q2", False)]),
    ]
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    code = bsv_paired_report.main([
        "--journal",
        str(journal),
        "--min-shared-qids",
        "2",
        "trial-pair",
        "1",
        "2",
    ])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate_decision"] == "block"


def test_cli_markdown_no_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    rows = [
        _row(1, "base", [("q1", True)]),
        _row(2, "cand", [("q1", True)]),
    ]
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    code = bsv_paired_report.main([
        "--journal",
        str(journal),
        "--min-shared-qids",
        "2",
        "--markdown",
        "--no-fail",
        "trial-pair",
        "1",
        "2",
    ])

    assert code == 0
    assert "# BSV-2 Paired Behavior Report" in capsys.readouterr().out
