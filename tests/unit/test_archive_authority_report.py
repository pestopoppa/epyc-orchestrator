"""Tests for the read-only AutoPilot archive authority report."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from archive_authority_report import (  # noqa: E402
    build_archive_authority_report,
    main,
    render_markdown,
)
from src.autopilot_core.journal_reconstruction import (  # noqa: E402
    reconstruct_archive_from_journal_rows,
)


def _row(trial_id: int, *, quality: float = 1.0) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "timestamp": f"2026-06-14T00:00:0{trial_id}Z",
        "species": "unit",
        "action_type": "seed_batch",
        "tier": 1,
        "quality": quality,
        "speed": 40.0,
        "cost": 0.2,
        "reliability": 0.9,
        "pareto_status": "frontier",
    }


def _archive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    archive = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)
    assert archive is not None
    return archive


def test_report_marks_aligned_archive_ok() -> None:
    rows = [_row(1, quality=1.2)]
    report = build_archive_authority_report(
        {"trial_counter": 2, "pareto_archive": _archive(rows)},
        rows,
    )

    assert report["ok"] is True
    assert report["diagnostic"]["status"] == "match"
    assert report["entry_id_delta"]["state_only_count"] == 0
    assert report["entry_id_delta"]["journal_only_count"] == 0
    assert report["frontier_id_delta"]["state_only_count"] == 0
    assert report["frontier_id_delta"]["journal_only_count"] == 0
    assert report["entry_mismatches"]["count"] == 0


def test_report_summarizes_id_and_value_drift() -> None:
    rows = [_row(1, quality=1.2), _row(2, quality=1.1)]
    archive = json.loads(json.dumps(_archive([rows[0], _row(3, quality=1.3)])))
    archive["all_entries"][0]["objectives"][0] = 0.5
    report = build_archive_authority_report(
        {"trial_counter": 4, "pareto_archive": archive},
        rows,
        max_examples=5,
    )

    assert report["ok"] is False
    assert report["diagnostic"]["status"] == "drift"
    assert report["entry_id_delta"]["state_only_examples"] == [3]
    assert report["entry_id_delta"]["journal_only_examples"] == [2]
    assert report["entry_mismatches"]["count"] == 1
    assert report["entry_mismatches"]["examples"][0]["trial_id"] == 1
    rendered = render_markdown(report)
    assert "State-only entries: 1 [3]" in rendered
    assert "Journal-only entries: 1 [2]" in rendered


def test_cli_json_strict_returns_nonzero_on_drift(
    tmp_path: Path,
    capsys,
) -> None:
    rows = [_row(1), _row(2)]
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    state_path.write_text(
        json.dumps({"trial_counter": 3, "pareto_archive": _archive([rows[0]])}),
        encoding="utf-8",
    )
    journal_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--state",
            str(state_path),
            "--journal",
            str(journal_path),
            "--json",
            "--strict",
        ]
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert out["ok"] is False
    assert out["entry_id_delta"]["journal_only_examples"] == [2]
