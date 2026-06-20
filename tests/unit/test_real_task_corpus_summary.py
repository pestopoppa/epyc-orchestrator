from __future__ import annotations

import json
from pathlib import Path

from scripts.tasks import summarize_real_task_corpus as summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_summary_keeps_source_strata_and_privacy_counts(tmp_path: Path) -> None:
    live_manifest = tmp_path / "live" / "manifest.json"
    live_rows = tmp_path / "live" / "rows.jsonl"
    hist_manifest = tmp_path / "historical" / "manifest.json"
    hist_rows = tmp_path / "historical" / "rows.jsonl"
    _write_json(
        live_manifest,
        {
            "counts": {
                "written": 2,
                "by_class": {"code_change_implementation": 1, "benchmark_eval_measurement": 1},
                "by_outcome": {"success": 2},
            }
        },
    )
    _write_jsonl(
        live_rows,
        [
            {"task_id": "a", "class": "code_change_implementation", "outcome": "success", "wall_s": 1.0},
            {"task_id": "b", "class": "benchmark_eval_measurement", "outcome": "success", "wall_s": 2.0},
        ],
    )
    _write_json(
        hist_manifest,
        {
            "counts": {
                "written": 1,
                "by_class": {"research_intake_deep_dive": 1},
                "by_outcome": {"success": 1},
            }
        },
    )
    _write_jsonl(
        hist_rows,
        [
            {
                "task_id": "hist-a",
                "class": "research_intake_deep_dive",
                "outcome": "success",
                "wall_s": 3.0,
                "tokens": {"total": 10},
            }
        ],
    )

    built = summary.build_summary(
        [
            {
                "label": "live",
                "source_family": "live_progress",
                "evidence_role": "operator_progress",
                "weight": 1.0,
                "manifest": str(live_manifest),
                "rows": str(live_rows),
            },
            {
                "label": "historical",
                "source_family": "historical_operator_conversation",
                "evidence_role": "operator_demand_backfill",
                "weight": 1.0,
                "manifest": str(hist_manifest),
                "rows": str(hist_rows),
            },
        ],
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert built["totals"]["raw_records"] == 3
    assert built["totals"]["by_source_family_raw"] == {
        "historical_operator_conversation": 1,
        "live_progress": 2,
    }
    assert built["totals"]["by_class_raw"] == {
        "benchmark_eval_measurement": 1,
        "code_change_implementation": 1,
        "research_intake_deep_dive": 1,
    }
    assert built["totals"]["token_payloads"] == 1
    assert built["totals"]["prompt_text_rows"] == 0
    assert built["gate_readout"]["class_outcome_count_gate"] is False
    assert built["gate_readout"]["multiple_source_families"] is True
    assert built["gate_readout"]["token_payload_coverage"] is True
    assert built["gate_readout"]["privacy_prompt_text_free"] is True


def test_parse_source_requires_complete_pipe_spec() -> None:
    parsed = summary.parse_source("live|live_progress|operator_progress|1.0|manifest.json|-")

    assert parsed == {
        "label": "live",
        "source_family": "live_progress",
        "evidence_role": "operator_progress",
        "weight": 1.0,
        "manifest": "manifest.json",
        "rows": None,
    }
