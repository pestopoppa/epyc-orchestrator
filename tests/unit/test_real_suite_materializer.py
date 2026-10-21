from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.tasks import materialize_real_suite_v1 as materializer


def _recon_row(task_id: str, class_id: str, qid: str, *, outcome: str = "success") -> dict:
    return {
        "task_id": task_id,
        "class": class_id,
        "outcome": outcome,
        "yaml_materialization_status": "expected_backed_ready",
        "question_pool_suite": "math",
        "question_pool_id": qid,
    }


def test_select_expected_backed_rows_respects_class_quotas() -> None:
    rows = []
    for class_id in materializer.CLASS_ORDER:
        for idx in range(8):
            rows.append(
                _recon_row(
                    f"{class_id}-{idx}",
                    class_id,
                    f"{class_id}-{idx}",
                    outcome="failure" if idx == 0 else "success",
                )
            )

    selected, quotas, shortages = materializer.select_expected_backed_rows(rows, total=50)

    assert len(selected) == 50
    assert sum(quotas.values()) == 50
    assert shortages == {}
    assert selected[0]["outcome"] == "failure"


def test_select_expected_backed_rows_filters_unscoreable_pool_matches() -> None:
    rows = []
    pool = {}
    for class_id in materializer.CLASS_ORDER:
        for idx in range(9):
            qid = f"{class_id}-{idx}"
            rows.append(_recon_row(f"{class_id}-{idx}", class_id, qid))
            pool[("math", qid)] = {
                "expected": "" if idx == 0 else f"Expected {idx}",
                "scoring_method": "exact_match",
            }

    selected, _, shortages = materializer.select_expected_backed_rows(
        rows, question_pool=pool, total=50
    )

    assert len(selected) == 50
    assert shortages == {}
    assert all(not row["question_pool_id"].endswith("-0") for row in selected)


def test_select_expected_backed_rows_accepts_code_execution_test_code() -> None:
    rows = []
    pool = {}
    for class_id in materializer.CLASS_ORDER:
        for idx in range(9):
            qid = f"{class_id}-{idx}"
            rows.append(_recon_row(f"{class_id}-{idx}", class_id, qid))
            pool[("math", qid)] = {
                "expected": "",
                "scoring_method": "code_execution",
                "scoring_config": {"test_code": "assert candidate() == 42"},
            }

    selected, _, shortages = materializer.select_expected_backed_rows(
        rows, question_pool=pool, total=50
    )

    assert len(selected) == 50
    assert shortages == {}


def test_select_expected_backed_rows_redistributes_scoreable_shortage() -> None:
    rows = []
    for class_id in materializer.CLASS_ORDER:
        count = 4 if class_id == "debug_root_cause" else 10
        for idx in range(count):
            rows.append(_recon_row(f"{class_id}-{idx}", class_id, f"{class_id}-{idx}"))

    selected, _, shortages = materializer.select_expected_backed_rows(rows, total=50)

    assert len(selected) == 50
    assert shortages == {"debug_root_cause": 3}
    assert sum(1 for row in selected if row["class"] == "debug_root_cause") == 4


def test_materializer_writes_scoreable_yaml_and_prompt_free_manifest(tmp_path: Path) -> None:
    reconstruction = tmp_path / "reconstruction.jsonl"
    pool = tmp_path / "question_pool.jsonl"
    yaml_out = tmp_path / "real_suite_v1.yaml"
    report_dir = tmp_path / "report"

    recon_rows = []
    pool_rows = [{"__pool_metadata__": True}]
    for class_id in materializer.CLASS_ORDER:
        for idx in range(8):
            qid = f"{class_id}_{idx}"
            recon_rows.append(_recon_row(f"task-{qid}", class_id, qid))
            pool_rows.append(
                {
                    "id": qid,
                    "suite": "math",
                    "prompt": f"Prompt {qid}",
                    "expected": "" if class_id == "debug_root_cause" else f"Expected {qid}",
                    "scoring_method": "code_execution"
                    if class_id == "debug_root_cause"
                    else "exact_match",
                    "scoring_config": {"test_code": "assert candidate() == 42"}
                    if class_id == "debug_root_cause"
                    else {},
                    "tier": 1,
                }
            )
    reconstruction.write_text("\n".join(json.dumps(row) for row in recon_rows) + "\n")
    pool.write_text("\n".join(json.dumps(row) for row in pool_rows) + "\n")

    result = materializer.run(
        materializer.build_parser().parse_args(
            [
                "--reconstruction",
                str(reconstruction),
                "--question-pool",
                str(pool),
                "--yaml-output",
                str(yaml_out),
                "--report-dir",
                str(report_dir),
            ]
        )
    )

    suite = yaml.safe_load(yaml_out.read_text())
    manifest = [json.loads(line) for line in (report_dir / "selected_rows.jsonl").read_text().splitlines()]
    assert result["selected_rows"] == 50
    assert suite["suite"] == "real_suite_v1"
    assert len(suite["questions"]) == 50
    assert any(question["scoring_method"] == "code_execution" for question in suite["questions"])
    assert "prompt" not in manifest[0]
    assert "expected" not in manifest[0]


def _vacuous(qid: str, source_id: str) -> dict:
    return {
        "id": qid, "tier": 2, "prompt": "old", "expected": "def ", "scoring_method": "substring",
        "scoring_config": {"language": "python", "timeout": 30, "case_sensitive": True, "substring": "def "},
        "source_suite": "livecodebench", "source_question_id": source_id,
    }


def test_rederive_vacuous_oracles_repairs_or_marks_not_scorable() -> None:
    fresh = {
        ("livecodebench", "leetcode_good"): {
            "prompt": "new prompt\n", "expected": "solve", "scoring_method": "code_execution",
            "scoring_config": {"entry_point": "solve", "entry_point_cases": [{"args": [1], "expected": 1}]},
        }
    }
    keep = {"id": "k", "scoring_method": "substring", "scoring_config": {"case_sensitive": False}, "expected": "x"}
    qs = [_vacuous("a", "leetcode_good"), _vacuous("b", "leetcode_dropped"), keep]
    changes = materializer.rederive_vacuous_oracles(qs, fresh)
    assert [(c["id"], c["action"]) for c in changes] == [
        ("a", "rederived_executable_oracle"), ("b", "marked_not_scorable")]
    assert qs[0]["scoring_method"] == "code_execution" and qs[0]["expected"] == "solve"
    assert qs[1]["scoring_method"] == "code_execution" and qs[1]["expected"] == ""
    assert "not_scorable_reason" in qs[1]["scoring_config"]
    assert qs[2]["scoring_method"] == "substring"  # non-code substring rows untouched
    assert not any(materializer.is_vacuous_code_substring(q) for q in qs)


def test_vacuous_code_substring_source_is_never_selectable() -> None:
    assert not materializer._has_scoreable_expected(_vacuous("a", "x") | {"expected": "def "})
