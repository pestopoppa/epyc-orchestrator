"""Regression coverage for the E8 v5 replay-first partial-resume contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts/benchmark/resume_e8_quality_baseline_v5.py"
spec = importlib.util.spec_from_file_location("e8_v5_partial_resume_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
resume = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = resume
spec.loader.exec_module(resume)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _response(qid: str, *, error: str | None = None) -> dict:
    return {
        "qid": qid,
        "suite": "suite",
        "scoring_method": "llm_judge" if qid.endswith("98") else "exact_match",
        "answer": error or "answer",
        "correct": error is None,
        "error": error,
        "partial": False,
        "degraded": False,
        "route_used": "frontdoor",
        "scoring_config_sha256": "0" * 64,
    }


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "failed-staging"
    for tier, n in ((1, 50), (2, 500)):
        questions = [
            {
                "qid": f"t{tier}-{ordinal}",
                "suite": "suite",
                "scoring_method": "exact_match",
                "expected": "a",
                "scoring_config": {},
            }
            for ordinal in range(n)
        ]
        if tier == 2:
            questions[98]["qid"] = "physreason_cal_problem_00351_sq2"
            questions[98]["scoring_method"] = "llm_judge"
            questions[99]["qid"] = "aime_2024-I-12"
        vector = {"tier": tier, "n": n, "questions": [{"qid": q["qid"]} for q in questions]}
        scoring = {"tier": tier, "n": n, "questions": questions}
        _write(root / f"question_vector.T{tier}.json", vector)
        _write(root / f"scoring_vector.T{tier}.json", scoring)
        for repetition in (1, 2, 3) if tier == 1 else (1,):
            rows = [_response(q["qid"]) for q in questions]
            sidecars = []
            for ordinal, q in enumerate(questions):
                error = None
                if tier == 2 and repetition == 1 and ordinal in (98, 99):
                    error = "[ERROR: Inference failed: chat_completions failed: timed out]"
                    rows[ordinal] = _response(q["qid"], error=error)
                sidecars.append(
                    {
                        "row_type": "question_result",
                        "ordinal": ordinal,
                        "answer": error or "answer",
                        "result": {
                            "qid": q["qid"],
                            "question_id": q["qid"],
                            "tokens_generated": 0 if error else 1,
                            "error": bool(error),
                            "error_detail": error,
                            "route": "frontdoor",
                            "correct": not bool(error),
                        },
                    }
                )
            _write(root / f"raw.T{tier}.r{repetition}.json", {"n": n})
            (root / f"responses.T{tier}.r{repetition}.jsonl").parent.mkdir(
                parents=True, exist_ok=True
            )
            (root / f"responses.T{tier}.r{repetition}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (
                root / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl"
            ).parent.mkdir(parents=True, exist_ok=True)
            (
                root / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl"
            ).write_text("".join(json.dumps(row) + "\n" for row in sidecars))
            (root / f"judge_traces.T{tier}.r{repetition}.jsonl").write_text("")
    (root / "runtime_watch.jsonl").write_text(json.dumps({"ok": True}) + "\n")
    return root


def test_plan_allows_exactly_two_generation_targets_and_no_banked_regeneration(
    tmp_path: Path,
) -> None:
    plan = resume.build_plan(_source(tmp_path))
    assert [(row["ordinal"], row["qid"]) for row in plan["generation_tail"]["targets"]] == [
        (98, "physreason_cal_problem_00351_sq2"),
        (99, "aime_2024-I-12"),
    ]
    assert plan["replay_only"] == {"tiers": [1], "banked_t2_r1_vector": True}
    assert plan["fresh_collection"] == [{"tier": 2, "repetition": 2}, {"tier": 2, "repetition": 3}]


def test_plan_rejects_any_extra_or_missing_target(tmp_path: Path) -> None:
    root = _source(tmp_path)
    rows = [json.loads(line) for line in (root / "responses.T2.r1.jsonl").read_text().splitlines()]
    rows[100] = _response(
        rows[100]["qid"], error="[ERROR: Inference failed: chat_completions failed: timed out]"
    )
    (root / "responses.T2.r1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    sidecar = root / "eval_sidecars/question_results.e8-t2-r1.jsonl"
    sidecars = [json.loads(line) for line in sidecar.read_text().splitlines()]
    sidecars[100]["answer"] = rows[100]["answer"]
    sidecars[100]["result"].update(
        {"tokens_generated": 0, "error": True, "error_detail": rows[100]["error"]}
    )
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in sidecars))
    with pytest.raises(ValueError, match="target set differs"):
        resume.build_plan(root)


def test_immutable_copy_hashes_every_source_file_and_never_edits_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    before = {
        path.relative_to(source): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    plan = resume.build_plan(source)
    destination = tmp_path / "copy"
    resume.copy_source_immutable(source, destination, plan)
    assert {
        path.relative_to(source): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    } == before
    binding = json.loads((destination / "source_binding.json").read_text())
    assert binding["source_sha256"] == plan["source_sha256"]
    with pytest.raises(FileExistsError):
        resume.copy_source_immutable(source, destination, plan)


def test_active_segment_rejects_gap_without_an_active_load_sample() -> None:
    class Watcher:
        samples: list[dict] = []

    with pytest.raises(ValueError, match="active-load segment"):
        resume._segment_check([{"ok": True, "active_load": None}], {"tier": 2, "repetition": 1})
    resume._segment_check(
        [{"ok": True, "active_load": {"tier": 2, "repetition": 1}}], {"tier": 2, "repetition": 1}
    )


def test_scorer_history_is_limited_to_one_replay(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    row = {
        "schema": "epyc.e8_quality_llm_judge_trace.v2",
        "fixed_vector_row": {"tier": 2, "repetition": 1, "ordinal": 6, "qid": "q6"},
        "attempts": [
            {"error": {"type": "ScoringUnavailableError"}},
            {"error": None},
        ],
    }
    trace.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="exactly 15"):
        resume._scorer_recovery_rows(trace, tier=2, repetition=1)
    row["attempts"].append({"error": None})
    trace.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="one deterministic retry"):
        resume._scorer_recovery_rows(trace, tier=2, repetition=1)


def test_atomic_publication_never_replaces_existing_destination(tmp_path: Path) -> None:
    staging, destination = tmp_path / "staging", tmp_path / "published"
    staging.mkdir()
    (staging / "evidence").write_text("candidate\n")
    destination.mkdir()
    (destination / "existing").write_text("immutable\n")
    with pytest.raises(FileExistsError):
        resume.V4.atomic_publish_noreplace(staging, destination)
    assert (destination / "existing").read_text() == "immutable\n"


def test_segmented_monitor_requires_gap_free_clean_claimed_segments() -> None:
    validator_path = PROJECT_ROOT / "scripts/benchmark/validate_e8_quality_baseline_v5.py"
    spec = importlib.util.spec_from_file_location("e8_v5_segmented_validator", validator_path)
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    samples = [
        {"started_at": "2026-07-28T00:00:00Z", "ok": True},
        {"started_at": "2026-07-28T00:00:05Z", "ok": True},
        {"started_at": "2026-07-28T04:00:00Z", "ok": True},
        {"started_at": "2026-07-28T04:00:05Z", "ok": True},
    ]
    validator.validate_segmented_monitor(
        samples, [{"sample_indexes": [0, 1]}, {"sample_indexes": [2, 3]}]
    )
    with pytest.raises(ValueError, match="sampling gap"):
        validator.validate_segmented_monitor(
            samples, [{"sample_indexes": [0, 2]}, {"sample_indexes": [1, 3]}]
        )
    with pytest.raises(ValueError, match="unclaimed"):
        validator.validate_segmented_monitor(samples, [{"sample_indexes": [0, 1]}])
