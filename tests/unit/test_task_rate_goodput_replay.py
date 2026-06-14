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


def test_report_includes_baseline_promotion_evidence_for_scoped_rows(tmp_path) -> None:
    output = tmp_path / "report.md"
    rows = [
        {
            "trial_id": 2,
            "quality": 1.2,
            "speed": 10.0,
            "cost": 0.5,
            "reliability": 0.9,
            "tier": 1,
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 2,
            "tier": 1,
            "previous_quality": 1.0,
            "new_quality": 1.2,
            "timestamp": "2026-06-14T00:00:00+00:00",
            "reason": "accepted | escaped",
            "proof": {
                "matrix_status": "ok",
                "speed_metric_mode": "aggregate_batch_tps",
            },
            "result_metrics": {
                "quality": 1.2,
                "speed": 10.0,
                "pareto_status": "frontier",
            },
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 99,
            "tier": 1,
            "new_quality": 9.9,
            "reason": "out of scope",
        },
    ]
    archive = {
        "frontier": [{"trial_id": 2, "objectives": [1.2, 10.0, -0.5, 0.9]}],
        "all_entries": [{"trial_id": 2}],
        "hypervolume_history": [[2, 1.0]],
    }

    replay._write_report(
        output,
        journal=tmp_path / "autopilot_journal.jsonl",
        rows=rows,
        malformed=0,
        legacy=archive,
        task_rate=archive,
        current_run_only=True,
    )

    report = output.read_text()

    assert "## Baseline Promotion Evidence" in report
    assert "| 2 | yes | 1 | 1.000 | 1.200 | 0.200 | 1.200 | 10.00 | frontier | ok | aggregate_batch_tps | accepted \\| escaped |" in report
    assert "out of scope" not in report


def test_report_baseline_promotion_table_tolerates_incomplete_events(tmp_path) -> None:
    output = tmp_path / "report.md"
    rows = [
        {
            "trial_id": 7,
            "quality": 1.0,
            "speed": 5.0,
            "cost": 0.5,
            "reliability": 0.9,
            "tier": 1,
        },
        {"type": "baseline_promotion", "source_trial_id": 7},
    ]
    archive = {
        "frontier": [{"trial_id": 7, "objectives": [1.0, 5.0, -0.5, 0.9]}],
        "all_entries": [{"trial_id": 7}],
        "hypervolume_history": [[7, 1.0]],
    }

    replay._write_report(
        output,
        journal=tmp_path / "autopilot_journal.jsonl",
        rows=rows,
        malformed=0,
        legacy=archive,
        task_rate=archive,
        current_run_only=False,
    )

    report = output.read_text()

    assert "## Baseline Promotion Evidence" in report
    assert "| 7 | yes | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in report
