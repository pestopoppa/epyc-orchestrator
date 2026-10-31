"""EV-14g: a short decision draw survives aggregation, sealing and journal readback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

from eval_tower import (  # noqa: E402
    EvalTower,
    QuestionResult,
    _stamp_eval_instrument,
    eval_result_count_fields,
)
from experiment_journal import ExperimentJournal, JournalEntry  # noqa: E402
from src.autopilot_core.tier_specs import (  # noqa: E402
    task_rate_qph_from,
    task_rate_qph_from_row,
    task_rate_qph_from_row_if_available,
)


def test_short_draw_survives_actual_aggregate_and_journal(tmp_path: Path) -> None:
    tower = EvalTower.__new__(EvalTower)
    requested = [
        {"id": str(i), "suite": "unit", "prompt": f"q{i}", "expected": "yes"} for i in range(3)
    ]
    delivered = [
        QuestionResult(
            question_id=q["id"],
            suite="unit",
            prompt=q["prompt"],
            expected="yes",
            answer="yes",
            correct=True,
            elapsed_s=1.0,
        )
        for q in requested[:2]
    ]
    result = tower._aggregate(delivered, tier=1)
    result.eval_wall_s = 3600.0
    _stamp_eval_instrument(
        result, questions=requested, core_id="ev14g-short", test_profile={"tier": 1}
    )
    assert result.n_questions == 2  # Existing aggregate semantics: delivered rows.
    assert eval_result_count_fields(result) == {
        "n_questions_requested": 3,
        "n_questions_completed": 2,
    }
    assert result.details["n_questions_requested"] == 3
    assert result.details["n_questions_completed"] == 2
    assert task_rate_qph_from(result) == 2.0

    journal = ExperimentJournal(tmp_path, segment_snapshots=False)
    journal.record(
        JournalEntry(
            trial_id=1,
            timestamp="2026-10-31T00:00:00Z",
            species="test",
            action_type="seed_batch",
            tier=1,
            quality=result.quality,
            speed=result.speed,
            cost=result.cost,
            reliability=result.reliability,
            pareto_status="dominated",
            eval_details={
                **eval_result_count_fields(result),
                "details": result.details,
                "eval_wall_s": result.eval_wall_s,
            },
        )
    )
    paths = list(tmp_path.glob("*.jsonl"))
    assert len(paths) == 1
    row = json.loads(paths[0].read_text().strip())
    assert row["eval_details"]["n_questions_requested"] == 3
    assert row["eval_details"]["n_questions_completed"] == 2
    assert task_rate_qph_from_row(row) == 2.0
    assert task_rate_qph_from_row_if_available(row) == 2.0


def test_zero_completed_does_not_fall_back_to_requested() -> None:
    row = {
        "n_questions": 3,
        "eval_wall_s": 3600.0,
        "eval_details": {"n_questions_requested": 3, "n_questions_completed": 0},
    }
    assert task_rate_qph_from_row(row) == 0.0
    assert task_rate_qph_from_row_if_available(row) == 0.0
    del row["eval_details"]["n_questions_completed"]
    assert task_rate_qph_from_row(row) == 3.0  # Legacy fallback.
    assert task_rate_qph_from_row_if_available(row) == 3.0
