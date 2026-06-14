from __future__ import annotations

from scripts.analysis import task_rate_goodput_replay as replay


def test_report_rows_use_folded_supersession_values(tmp_path) -> None:
    output = tmp_path / "report.md"
    rows = [
        {
            "trial_id": 2,
            "quality": 9.0,
            "speed": 99.0,
            "cost": 0.5,
            "reliability": 1.0,
            "tier": 1,
        },
        {
            "type": "supersession",
            "target_trial_ids": [2],
            "fields": {"quality": 1.23, "speed": 4.56},
        },
    ]
    legacy = {
        "frontier": [{"trial_id": 2, "objectives": [9.0, 99.0, -0.5, 1.0]}],
        "all_entries": [{"trial_id": 2}],
        "hypervolume_history": [[2, 1.0]],
    }
    task_rate = {
        "frontier": [],
        "all_entries": [],
        "hypervolume_history": [[2, 0.0]],
    }

    replay._write_report(
        output,
        journal=tmp_path / "autopilot_journal.jsonl",
        rows=rows,
        malformed=0,
        legacy=legacy,
        task_rate=task_rate,
        current_run_only=False,
    )

    report = output.read_text()

    assert "| 2 | 1.230 | 4.56 |" in report
    assert "| 2 | 9.000 | 99.00 |" not in report
