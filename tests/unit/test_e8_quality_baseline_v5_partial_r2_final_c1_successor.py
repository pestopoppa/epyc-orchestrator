"""Focused admission tests for the final two-row E8 C1 successor."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_final_c1_successor.py"
SPEC = importlib.util.spec_from_file_location("e8_final_c1_test", PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

SOURCE = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "e8_quality_baseline_v5_partial_r2_race_retry_20260728T202306Z"
)


def test_final_c1_schedule_is_fixed_and_sequential() -> None:
    plan = RUNNER.build_plan(SOURCE, RUNNER.REVIEWED_SOURCE_TREE_SHA256)

    assert plan["generation_ordinals"] == [97, 279]
    assert plan["final_c1_retry_ordinals"] == [97, 279]
    assert plan["final_c1_schedule"] == {
        "concurrency": 1,
        "request_timeout_s": 300,
        "max_retries_per_target": 1,
        "sequential": True,
        "targets": [97, 279],
        "targets_sha256": RUNNER.canonical_hash([97, 279]),
    }
    assert sum(
        len(plan[key])
        for key in (
            "reuse_ordinals", "inherited_scorer_replay_ordinals",
            "imported_generation_ordinals", "scorer_replay_ordinals",
            "predecessor_generation_import_ordinals",
        )
    ) == 498


def test_final_c1_rejects_unpinned_or_wrong_source_digest() -> None:
    with pytest.raises(ValueError, match="reviewed terminal source"):
        RUNNER.build_plan(SOURCE, "0" * 64)


def test_terminal_timeout_rejects_nonzero_tokens_or_wrong_latency() -> None:
    question = {"qid": "q"}
    row = {
        "answer": "",
        "result": {
            "qid": "q", "error": True, "tokens_generated": 0,
            "error_detail": "timed out", "latency_ms": 300000,
        },
    }
    assert RUNNER._terminal_300_timeout(row, question)
    row["result"]["tokens_generated"] = 1
    assert not RUNNER._terminal_300_timeout(row, question)
    row["result"]["tokens_generated"] = 0
    row["result"]["latency_ms"] = 299000
    assert not RUNNER._terminal_300_timeout(row, question)
