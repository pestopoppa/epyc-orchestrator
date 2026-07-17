"""Tests for AP-27 prompt-free RLVR environment export."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.autopilot.export_rlvr_environment import (
    ROW_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    export_environment_rows,
    main,
)


def test_export_environment_rows_strips_private_question_text() -> None:
    rows, summary = export_environment_rows(
        [
            {
                "trial_id": 42,
                "action_type": "deep_eval",
                "tier": 2,
                "quality": 2.4,
                "reliability": 0.9,
                "eval_details": {
                    "ece": 0.05,
                    "auroc": 0.8,
                    "question_results": [
                        {
                            "qid": "q1",
                            "suite": "math",
                            "correct": True,
                            "answer_hash": "sha256:abc",
                            "prompt": "private prompt",
                            "answer": "private answer",
                            "expected": "private expected",
                        }
                    ],
                },
            }
        ],
        source_label="unit",
    )

    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["rows"] == 1
    assert summary["ready_for_training"] == 1
    assert rows[0]["schema_version"] == ROW_SCHEMA_VERSION
    assert rows[0]["reward_policy"] == "ap27_rlvr_tier_reward_v1"
    assert rows[0]["tier"] == 2
    assert rows[0]["ready_for_training"] is True
    assert rows[0]["suite_counts"] == {"math": 1}
    assert rows[0]["question_results"] == [
        {
            "answer_hash": "sha256:abc",
            "correct": True,
            "qid": "q1",
            "suite": "math",
        }
    ]


def test_export_environment_rows_records_training_blockers() -> None:
    rows, summary = export_environment_rows(
        [
            {
                "eval_result": {
                    "tier": 1,
                    "quality": 2.0,
                    "reliability": 0.8,
                    "eval_details": {"ece": None, "auroc": 0.0},
                }
            }
        ]
    )

    assert rows[0]["ready_for_training"] is False
    assert rows[0]["blockers"] == ["ece_missing", "auroc_missing_or_degenerate"]
    assert summary["blocked"] == 1
    assert summary["blocker_counts"] == {
        "auroc_missing_or_degenerate": 1,
        "ece_missing": 1,
    }


def test_cli_writes_jsonl_and_summary(tmp_path: Path) -> None:
    source = tmp_path / "eval.json"
    output = tmp_path / "rlvr.jsonl"
    summary = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "tier": 0,
                "quality": 3.0,
                "reliability": 1.0,
                "question_results": [{"question_id": "q1", "suite": "general", "correct": True}],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(source),
                "--output-jsonl",
                str(output),
                "--summary-json",
                str(summary),
                "--source-label",
                "fixture",
            ]
        )
        == 0
    )

    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert exported[0]["source_label"] == "fixture"
    assert exported[0]["reward_signal"] == "binary_outcome"
    assert json.loads(summary.read_text(encoding="utf-8"))["tier_counts"] == {"0": 1}


def test_cli_fail_on_blockers_returns_one_after_writing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "eval.json"
    output = tmp_path / "rlvr.jsonl"
    source.write_text(
        json.dumps({"tier": 1, "quality": 1.0, "reliability": 0.5}),
        encoding="utf-8",
    )

    assert main([str(source), "--output-jsonl", str(output), "--fail-on-blockers"]) == 1
    assert output.exists()
