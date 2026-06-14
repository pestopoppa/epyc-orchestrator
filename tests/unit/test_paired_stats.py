from __future__ import annotations

import json
from pathlib import Path

from scripts.autopilot import paired_stats


def _row(
    trial_id: int,
    fingerprint: str,
    outcomes: list[tuple[str, bool]],
) -> dict:
    return {
        "trial_id": trial_id,
        "config_snapshot": {"config_fingerprint": fingerprint},
        "eval_details": {
            "question_results": [
                {"qid": qid, "suite": "suite", "correct": correct} for qid, correct in outcomes
            ]
        },
    }


def test_mcnemar_from_vectors_counts_discordant_pairs() -> None:
    rows = [
        _row(1, "a", [("q1", True), ("q2", True), ("q3", False), ("q4", False)]),
        _row(2, "b", [("q1", True), ("q2", False), ("q3", True), ("q4", False)]),
    ]
    vectors = paired_stats.trial_vectors(rows)

    result = paired_stats.mcnemar_from_vectors(vectors[1], vectors[2], "1", "2")

    assert result.shared_qids == 4
    assert result.same_correct == 1
    assert result.same_wrong == 1
    assert result.a_correct_b_wrong == 1
    assert result.a_wrong_b_correct == 1
    assert result.p_value_two_sided == 1.0
    assert result.delta_b_minus_a == 0.0


def test_compare_fingerprints_uses_majority_per_qid() -> None:
    rows = [
        _row(1, "base", [("q1", True), ("q2", False), ("q3", False)]),
        _row(2, "base", [("q1", True), ("q2", False), ("q3", True)]),
        _row(3, "cand", [("q1", True), ("q2", True), ("q3", True)]),
        _row(4, "cand", [("q1", True), ("q2", True), ("q3", False)]),
    ]

    result = paired_stats.compare_fingerprints(rows, "cand", "base")

    assert result["shared_qids"] == 2  # q3 ties in both groups and is dropped
    assert result["a_correct_b_wrong"] == 0
    assert result["a_wrong_b_correct"] == 1
    assert result["delta_b_minus_a"] == 0.5
    assert result["candidate_trials"] == [3, 4]
    assert result["baseline_trials"] == [1, 2]


def test_summary_reports_no_vectors_for_current_style_rows(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    journal.write_text(
        json.dumps(
            {
                "trial_id": 10,
                "config_snapshot": {"type": "seed_batch"},
                "eval_details": {"details": {"correct": 2, "total": 3}},
            }
        )
        + "\n"
    )

    summary = paired_stats.summarize(tmp_path)

    assert summary["rows"] == 1
    assert summary["vector_trials"] == 0
    assert summary["fingerprints_with_vectors"] == {}


def test_iter_journal_rows_folds_supersession_events(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    rows = [
        _row(1, "base", [("q1", True)]),
        _row(2, "cand", [("q1", False)]),
        {
            "type": "supersession",
            "target_trial_ids": [2],
            "fields": {"bug_corrupted_by": "resource_contention"},
        },
    ]
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    loaded = list(paired_stats.iter_journal_rows(journal))

    assert [row["trial_id"] for row in loaded] == [1, 2]
    assert loaded[1]["bug_corrupted_by"] == "resource_contention"
    assert "type" not in loaded[1]
