"""Contract tests for the bounded E8 T2/r2 recovery preflight."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
spec = importlib.util.spec_from_file_location("e8_partial_r2_recovery", MODULE_PATH)
assert spec is not None and spec.loader is not None
recovery = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recovery
spec.loader.exec_module(recovery)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "aborted"
    questions = []
    public = []
    for ordinal in range(500):
        method = "llm_judge" if ordinal in (6, 24, 44) else "exact_match"
        qid = f"q-{ordinal}"
        public.append({"qid": qid})
        questions.append(
            {
                "qid": qid,
                "suite": "suite",
                "scoring_method": method,
                "expected": "expected",
                "scoring_config": {},
            }
        )
    for name, rows in (("question_vector.T2.json", public), ("scoring_vector.T2.json", questions)):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text(json.dumps({"tier": 2, "n": 500, "questions": rows}) + "\n")
    sidecar = [
        {
            "row_type": "batch_start",
            "requested_n": 500,
            "concurrency": recovery.V4.CONCURRENCY,
            "complete": False,
        }
    ]
    for ordinal, question in enumerate(questions[:79]):
        error = None
        answer = "answer"
        if ordinal in {2, 3, 5, 7, 20, 26, 29, 38, 41, 45, 48, 51, 53, 54, 65, 72, 75}:
            error = "[ERROR: placement timeout role=frontdoor reason=race_lost holders=[0, 1, 2] after 90.0s]"
            answer = ""
        elif ordinal in {6, 24, 44}:
            error = "scoring_unavailable: judge unavailable"
        result = {
            "qid": question["qid"],
            "question_id": question["qid"],
            "suite": "suite",
            "scoring_method": question["scoring_method"],
            "route": "frontdoor",
            "tokens_generated": 0 if error and not answer else 1,
            "correct": not bool(error),
        }
        if error:
            result.update({"error": True, "error_detail": error})
        else:
            result.update({"answer_hash": recovery.V5._normalized_answer_hash(answer)})
        sidecar.append(
            {"row_type": "question_result", "ordinal": ordinal, "answer": answer, "result": result}
        )
    _write(source / "eval_sidecars/question_results.e8-t2-r2.jsonl", sidecar)
    _write(source / "judge_traces.T2.r2.jsonl", [])
    return source


def test_plan_reuses_only_clean_rows_and_bounds_generation(tmp_path: Path) -> None:
    plan = recovery.build_plan(_source(tmp_path))
    assert len(plan["reuse_ordinals"]) == 59
    assert plan["scorer_replay_ordinals"] == [6, 24, 44]
    assert len(plan["generation_ordinals"]) == 438
    assert set(plan["scorer_replay_ordinals"]).isdisjoint(plan["generation_ordinals"])
    assert plan["generation_concurrency"] == recovery.V4.CONCURRENCY


def test_plan_rejects_unapproved_saved_error(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = source / "eval_sidecars/question_results.e8-t2-r2.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in rows if row.get("ordinal") == 2)["result"]["error_detail"] = (
        "request timed out"
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="unapproved terminal"):
        recovery.build_plan(source)


def test_collect_fails_closed_without_creating_an_output_bundle(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "would-be-evidence"
    with pytest.raises(ValueError, match="held GLOBAL recovery claim"):
        recovery.execute(type("Args", (), {"source_dir": source, "output_dir": output})())
    assert not output.exists()


def test_compact_exact_match_omission_is_allowed_but_llm_judge_omission_is_not(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    path = source / "eval_sidecars/question_results.e8-t2-r2.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in rows if row.get("ordinal") == 0)["result"].pop("scoring_method")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    recovery.build_plan(source)
    next(row for row in rows if row.get("ordinal") == 6)["result"].pop("scoring_method")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="scoring method"):
        recovery.build_plan(source)


def test_plan_rejects_a_symlink_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    link = tmp_path / "source-link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        recovery.build_plan(link)


def test_receipt_rejection_precedes_any_output_write(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    plan = recovery.build_plan(source)
    claim = {"claims": [{"payload": {"request_tag": "tag", "region": "q2"}}], "global_claims": [{}]}
    monkeypatch.setattr(
        recovery, "_instrument_identity", lambda: {"commit": "c", "runner_sha256": "r"}
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"operator_attestation": "yes"}))
    with pytest.raises(ValueError, match="receipt differs"):
        recovery.validate_receipt(receipt, plan, claim=claim)


def test_snapshot_rejects_source_mutation_before_copy(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plan = recovery.build_plan(source)
    (source / "question_vector.T2.json").write_text("{}\n")
    with pytest.raises(ValueError, match="changed before snapshot"):
        recovery._snapshot_source(source, tmp_path / "output", plan)


def test_capacity_rejects_q2_q3_contention_before_generation() -> None:
    regions = {
        ("frontdoor", 1): frozenset({"q0"}),
        ("frontdoor", 2): frozenset({"q1"}),
        ("frontdoor", 3): frozenset({"q2"}),
        ("frontdoor", 4): frozenset({"q3"}),
    }
    capacity, selected = recovery.compatible_frontdoor_capacity(regions, {1, 2, 3, 4}, {"q2", "q3"})
    assert capacity == 2
    assert selected == [
        {"topology_idx": 1, "regions": ["q0"]},
        {"topology_idx": 2, "regions": ["q1"]},
    ]


def test_capacity_selects_only_mutually_disjoint_free_instances() -> None:
    regions = {
        ("frontdoor", 0): frozenset({"q0", "q1", "q2", "q3"}),
        ("frontdoor", 1): frozenset({"q0"}),
        ("frontdoor", 2): frozenset({"q1"}),
        ("frontdoor", 3): frozenset({"q2"}),
    }
    capacity, selected = recovery.compatible_frontdoor_capacity(regions, {0, 1, 2, 3}, set())
    assert capacity == 3
    assert [row["topology_idx"] for row in selected] == [1, 2, 3]
