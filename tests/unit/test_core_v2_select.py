from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

core_v2_select = importlib.import_module("core_v2_select")


def _row(trial_id: int, results: list[dict]) -> dict:
    return {"trial_id": trial_id, "eval_details": {"question_results": results}}


def test_collect_item_stats_treats_missing_partition_as_core() -> None:
    rows = [
        _row(
            1,
            [
                {"qid": "q1", "suite": "math", "correct": True},
                {"qid": "q2", "suite": "math", "correct": True, "partition": "audit"},
            ],
        ),
        _row(2, [{"qid": "q1", "suite": "math", "correct": False}]),
        {"type": "supersession", "target_trial_ids": [1]},
    ]

    stats = core_v2_select.collect_item_stats(rows, include_partitions={"core"})

    assert set(stats) == {("math", "q1")}
    assert stats[("math", "q1")].attempts == 2
    assert stats[("math", "q1")].correct == 1
    assert stats[("math", "q1")].p_correct == 0.5


def test_select_core_items_prefers_medium_items_and_stratifies() -> None:
    stats = {
        ("math", "easy"): core_v2_select.ItemStats(
            qid="easy", suite="math", attempts=5, correct=5
        ),
        ("math", "medium-math"): core_v2_select.ItemStats(
            qid="medium-math", suite="math", attempts=5, correct=3
        ),
        ("coder", "medium-coder"): core_v2_select.ItemStats(
            qid="medium-coder", suite="coder", attempts=5, correct=2
        ),
        ("coder", "hard"): core_v2_select.ItemStats(
            qid="hard", suite="coder", attempts=5, correct=0
        ),
    }

    selected = core_v2_select.select_core_items(
        stats,
        target_size=2,
        min_attempts=2,
        p_min=0.2,
        p_max=0.8,
        max_per_suite=1,
    )

    assert [(item.suite, item.qid) for item in selected] == [
        ("coder", "medium-coder"),
        ("math", "medium-math"),
    ]


def test_pool_lookup_maps_stable_eval_qids_to_full_question_rows(tmp_path: Path) -> None:
    pool = tmp_path / "question_pool.jsonl"
    question = {
        "id": "math-1",
        "suite": "math",
        "prompt": "What is two plus two?",
        "expected": "4",
        "scoring_method": "exact_match",
    }
    stable_qid = core_v2_select._stable_question_qid("math", question["prompt"])
    pool.write_text(
        json.dumps({"__pool_metadata__": True}) + "\n" + json.dumps(question) + "\n",
        encoding="utf-8",
    )

    lookup = core_v2_select.load_pool_lookup(pool)

    assert lookup[("math", "math-1")]["prompt"] == question["prompt"]
    assert lookup[("math", stable_qid)]["id"] == "math-1"


def test_write_core_jsonl_uses_pool_rows_and_records_selection(tmp_path: Path) -> None:
    selected = [
        core_v2_select.ItemStats(qid="q-stable", suite="math", attempts=3, correct=2),
    ]
    report = {
        "generated_at": "2026-06-14T00:00:00Z",
        "parameters": {"target_size": 1},
        "selected_count": 1,
    }
    out = tmp_path / "core_v2.jsonl"

    unresolved = core_v2_select.write_core_jsonl(
        path=out,
        core_id="core_v2",
        selected=selected,
        pool_lookup={
            (
                "math",
                "q-stable",
            ): {
                "id": "math-1",
                "suite": "math",
                "prompt": "2+2?",
                "expected": "4",
                "scoring_method": "exact_match",
            }
        },
        report=report,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert unresolved == []
    assert rows[0]["__core_metadata__"] is True
    assert rows[0]["core_id"] == "core_v2"
    assert rows[1]["id"] == "math-1"
    assert rows[1]["core_selection"]["p_correct"] == 0.666667


def test_cli_writes_report_and_core_file(tmp_path: Path, monkeypatch) -> None:
    pool = tmp_path / "pool.jsonl"
    journal = tmp_path / "journal.jsonl"
    report = tmp_path / "report.json"
    core = tmp_path / "core.jsonl"
    question = {
        "id": "math-1",
        "suite": "math",
        "prompt": "2+2?",
        "expected": "4",
        "scoring_method": "exact_match",
    }
    stable_qid = core_v2_select._stable_question_qid("math", question["prompt"])
    pool.write_text(
        json.dumps({"__pool_metadata__": True}) + "\n" + json.dumps(question) + "\n",
        encoding="utf-8",
    )
    journal.write_text(
        "\n".join(
            [
                json.dumps(
                    _row(
                        1,
                        [{"qid": stable_qid, "suite": "math", "correct": True}],
                    )
                ),
                json.dumps(
                    _row(
                        2,
                        [{"qid": stable_qid, "suite": "math", "correct": False}],
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core_v2_select.py",
            "--journal",
            str(journal),
            "--pool",
            str(pool),
            "--out-core",
            str(core),
            "--report-json",
            str(report),
            "--target-size",
            "1",
            "--min-attempts",
            "2",
        ],
    )

    assert core_v2_select.main() == 0

    report_obj = json.loads(report.read_text(encoding="utf-8"))
    core_rows = [json.loads(line) for line in core.read_text(encoding="utf-8").splitlines()]
    assert report_obj["selected_count"] == 1
    assert report_obj["eligible_items"] == 1
    assert core_rows[1]["id"] == "math-1"
