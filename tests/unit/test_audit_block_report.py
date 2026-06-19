from __future__ import annotations

import json
from pathlib import Path

from scripts.autopilot import audit_block_report


def _trial(
    trial_id: int,
    core_correct: int,
    core_total: int,
    audit_correct: int,
    audit_total: int,
    *,
    missing_partition_core: int = 0,
) -> dict:
    question_results = []
    for idx in range(core_total):
        question_results.append(
            {
                "qid": f"core-{trial_id}-{idx}",
                "correct": idx < core_correct,
                "partition": "core" if idx >= missing_partition_core else None,
            }
        )
    for idx in range(audit_total):
        question_results.append(
            {
                "qid": f"audit-{trial_id}-{idx}",
                "correct": idx < audit_correct,
                "partition": "audit",
            }
        )
    return {
        "trial_id": trial_id,
        "eval_details": {"question_results": question_results},
    }


def _write_journal(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_partition_extraction_and_missing_partition_defaults_to_core(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    _write_journal(
        journal,
        [
            _trial(1, core_correct=1, core_total=2, audit_correct=1, audit_total=2, missing_partition_core=1),
        ],
    )

    report = audit_block_report.build_report(audit_block_report.load_journal_rows([journal]))

    assert report["trial_count"] == 1
    assert report["audited_trial_count"] == 1
    trial = report["trials"][0]
    assert trial["core_correct"] == 1
    assert trial["core_total"] == 2
    assert trial["audit_correct"] == 1
    assert trial["audit_total"] == 2
    assert trial["core_quality_0_3"] == 1.5
    assert trial["audit_quality_0_3"] == 1.5
    assert trial["delta_audit_minus_core"] == 0.0


def test_skip_rows_without_audit_block(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    _write_journal(
        journal,
        [
            _trial(1, core_correct=2, core_total=2, audit_correct=1, audit_total=2),
            {
                "trial_id": 2,
                "eval_details": {
                    "question_results": [
                        {"qid": "core-2-0", "correct": True},
                        {"qid": "core-2-1", "correct": False},
                    ]
                },
            },
            {"type": "ledger", "trial_id": 3, "eval_details": {"question_results": []}},
            {"type": "supersession", "target_trial_ids": [1]},
        ],
    )

    report = audit_block_report.build_report(audit_block_report.load_journal_rows([journal]))

    assert report["trial_count"] == 2
    assert report["audited_trial_count"] == 1
    assert [trial["trial_id"] for trial in report["trials"]] == [1]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    _write_journal(
        journal,
        [
            _trial(7, core_correct=1, core_total=2, audit_correct=2, audit_total=2),
        ],
    )

    exit_code = audit_block_report.main(
        ["--journal", str(journal), "--out-json", str(out_json), "--out-md", str(out_md)]
    )

    assert exit_code == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["trials"][0]["trial_id"] == 7
    assert report["trials"][0]["audit_quality_0_3"] == 3.0
    markdown = out_md.read_text(encoding="utf-8")
    assert "# W6 Rotating Audit Block Report" in markdown
    assert "| trial_id | core | audit | core_q | audit_q | delta_audit_minus_core |" in markdown
    assert "7 | 1/2 | 2/2 | 1.500 | 3.000 | +1.500" in markdown


def test_transfer_diagnostic_counts_divergences(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    _write_journal(
        journal,
        [
            _trial(1, core_correct=1, core_total=2, audit_correct=1, audit_total=2),
            _trial(2, core_correct=2, core_total=2, audit_correct=1, audit_total=2),
            _trial(3, core_correct=2, core_total=2, audit_correct=0, audit_total=2),
        ],
    )

    report = audit_block_report.build_report(audit_block_report.load_journal_rows([journal]))

    diagnostic = report["transfer_diagnostic"]
    assert diagnostic["audited_trial_count"] == 3
    assert diagnostic["potential_overfit_divergences"] == 1
    assert diagnostic["events"] == [
        {
            "trial_id": 2,
            "previous_trial_id": 1,
            "core_delta": 1.5,
            "audit_delta": 0.0,
        }
    ]


def test_no_gaming_alarm_with_insufficient_audited_history(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    _write_journal(
        journal,
        [
            _trial(1, core_correct=1, core_total=2, audit_correct=1, audit_total=2),
            _trial(2, core_correct=2, core_total=2, audit_correct=1, audit_total=2),
        ],
    )

    report = audit_block_report.build_report(audit_block_report.load_journal_rows([journal]))

    assert report["gaming_alarm"] is False
    assert report["gaming_events"] == []
    assert report["transfer_diagnostic"]["potential_overfit_divergences"] == 0


def test_gaming_alarm_detects_core_improving_audit_flat_or_worsening(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    _write_journal(
        journal,
        [
            _trial(1, core_correct=1, core_total=2, audit_correct=1, audit_total=2),
            _trial(2, core_correct=2, core_total=2, audit_correct=1, audit_total=2),
            _trial(3, core_correct=2, core_total=2, audit_correct=2, audit_total=2),
        ],
    )

    report = audit_block_report.build_report(audit_block_report.load_journal_rows([journal]))

    assert report["gaming_alarm"] is True
    assert report["gaming_events"] == [
        {
            "trial_id": 2,
            "previous_trial_id": 1,
            "core_delta": 1.5,
            "audit_delta": 0.0,
        }
    ]


def test_markdown_includes_gaming_alarm(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    _write_journal(
        journal,
        [
            _trial(1, core_correct=1, core_total=2, audit_correct=1, audit_total=2),
            _trial(2, core_correct=2, core_total=2, audit_correct=1, audit_total=2),
            _trial(3, core_correct=2, core_total=2, audit_correct=2, audit_total=2),
        ],
    )

    markdown = audit_block_report.render_markdown(
        audit_block_report.build_report(audit_block_report.load_journal_rows([journal]))
    )

    assert "## Audit Gaming Alarm" in markdown
    assert "triggered" in markdown
    assert "trial 2 vs 1" in markdown
