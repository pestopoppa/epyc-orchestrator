from __future__ import annotations

from dataclasses import dataclass

from src.autopilot_core.planner_evidence import format_planner_evidence_section


def _row(
    trial_id: int,
    *,
    config: dict | None = None,
    seq: dict | None = None,
    quality: float = 1.8,
    reliability: float = 0.98,
    corrupt: str = "",
    question_count: int = 50,
) -> dict:
    return {
        "trial_id": trial_id,
        "tier": 1,
        "quality": quality,
        "reliability": reliability,
        "bug_corrupted_by": corrupt,
        "config_snapshot": config or {"type": "seed_batch", "n_questions": 10},
        "eval_details": {
            "eval_wall_s": 600.0,
            "question_results": [
                {"qid": f"q{i}", "correct": i % 2 == 0}
                for i in range(question_count)
            ],
        },
        "seq": seq,
    }


def test_empty_evidence_section_is_stable() -> None:
    text = format_planner_evidence_section([])

    assert "no trusted per-question vectors yet" in text
    assert "no trusted vector-bearing candidates yet" in text


def test_vector_rows_collapse_by_behavioral_fingerprint() -> None:
    rows = [
        _row(10, config={"type": "seed_batch", "n_questions": 10, "reasoning": "a"}),
        _row(11, config={"n_questions": 10, "type": "seed_batch", "reasoning": "b"}),
        _row(12, config={"type": "numeric_trial", "surface": "memrl_retrieval"}),
    ]

    text = format_planner_evidence_section(rows)

    assert "vector_trials=3 candidates=2" in text
    assert "quality_quantum~0.060" in text
    assert "seq=not_logged_yet" in text


def test_corrupted_and_audit_only_rows_are_excluded() -> None:
    rows = [
        _row(1, corrupt="resource_contention"),
        {**_row(2), "tier": 0},
        _row(3),
    ]

    text = format_planner_evidence_section(rows)

    assert "vector_trials=1 candidates=1" in text
    assert "trials=[3]" in text
    assert "trials=[1" not in text
    assert "trials=[2" not in text


def test_seq_rows_fold_by_candidate_and_skip_malformed_z() -> None:
    candidate = "candidate-a"
    rows = [
        _row(
            20,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": candidate, "core_id": "core_v1", "z": 1.0},
        ),
        _row(
            21,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": candidate, "core_id": "core_v1", "z": 1.0},
        ),
        _row(
            22,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": candidate, "core_id": "core_v1", "z": "bad"},
        ),
    ]

    text = format_planner_evidence_section(rows)

    assert "seq_candidates=1" in text
    assert "fp=candidate-a" in text
    assert "seq=accumulating k=2 E_quality=1.650" in text
    assert "trials=[20,21,22]" in text


def test_seq_rows_ignore_non_matching_core_ids() -> None:
    rows = [
        _row(
            24,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": "candidate-a", "core_id": "old_core", "z": 1.0},
        ),
        _row(
            25,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": "candidate-a", "core_id": "core_v1", "z": 1.0},
        ),
    ]

    text = format_planner_evidence_section(rows, core_id="core_v1")

    assert "seq_candidates=1" in text
    assert "seq=accumulating k=1" in text
    assert "trials=[24,25]" in text


def test_missing_timing_fields_do_not_invent_task_rate() -> None:
    row = _row(30)
    row["eval_details"] = {
        "question_results": [{"qid": "q1", "correct": True}],
    }

    text = format_planner_evidence_section([row])

    assert "task_rate=0.0 goodput=0.0" in text


def test_dataclass_rows_are_normalized_at_boundary() -> None:
    @dataclass
    class Row:
        trial_id: int
        tier: int
        quality: float
        reliability: float
        config_snapshot: dict
        eval_details: dict
        bug_corrupted_by: str = ""

    row = Row(
        trial_id=40,
        tier=1,
        quality=2.0,
        reliability=1.0,
        config_snapshot={"type": "seed_batch", "n_questions": 10},
        eval_details={
            "eval_wall_s": 60.0,
            "question_results": [{"qid": "q1", "correct": True}],
        },
    )

    text = format_planner_evidence_section([row])

    assert "vector_trials=1 candidates=1" in text
    assert "trials=[40]" in text
