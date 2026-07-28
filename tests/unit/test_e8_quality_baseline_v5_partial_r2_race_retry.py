"""Adversarial eligibility tests for the E8 r2 race-only second successor."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
SPEC = importlib.util.spec_from_file_location("e8_r2_race_retry_test", PATH)
assert SPEC and SPEC.loader
RETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETRY)


QUESTION = {"qid": "q0"}
RACE = "[ERROR: placement timeout role=frontdoor reason=race_lost holders=[0, 1, 2] after 90.0s]"


def _row(*, error: str = RACE, tokens: int = 0, answer: str | None = None) -> dict:
    return {
        "row_type": "question_result", "ordinal": 0,
        "answer": error if answer is None else answer,
        "result": {
            "qid": "q0", "question_id": "q0", "error": True,
            "error_detail": error, "tokens_generated": tokens, "route": "frontdoor",
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _watcher(started_at: str, *, binding: str = "a") -> dict:
    return {
        "ok": True,
        "active_load": {"tier": 2, "repetition": 2},
        "started_at": started_at,
        "api_probe_urls": {"frontdoor": binding},
        "runtime_artifacts": {"server": {"identity": binding}},
    }


def test_exact_race_lost_requires_zero_tokens_and_error_sentinel() -> None:
    assert RETRY._race_lost(_row(), QUESTION)
    assert not RETRY._race_lost(_row(tokens=1), QUESTION)
    assert not RETRY._race_lost(_row(answer="model output"), QUESTION)


def test_non_race_error_is_not_retry_eligible() -> None:
    assert not RETRY._race_lost(_row(error="timed out", answer=""), QUESTION)


def test_duplicate_sidecar_ordinal_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "sidecar.jsonl"
    _write_jsonl(path, [_row(), _row()])
    with pytest.raises(ValueError, match="duplicate"):
        RETRY._rows(path)


def test_tree_pin_rejects_source_tamper_before_any_artifact_read(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    item = source / "immutable.txt"
    item.write_text("before")
    expected = RETRY.canonical_hash(RETRY.source_hashes(source))
    item.write_text("after")
    with pytest.raises(ValueError, match="explicit terminal tree hash"):
        RETRY.build_plan(source, expected)


def test_contaminated_predecessor_watcher_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "watcher.jsonl"
    rows = [_watcher("2026-01-01T00:00:00Z"), _watcher("2026-01-01T00:00:05Z")]
    rows[0]["ok"] = False
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="watcher is contaminated"):
        RETRY._require_clean_predecessor_watcher(path)


def test_predecessor_watcher_rejects_gap_over_ratified_limit(tmp_path: Path) -> None:
    path = tmp_path / "watcher.jsonl"
    _write_jsonl(
        path,
        [_watcher("2026-01-01T00:00:00Z"), _watcher("2026-01-01T00:00:07.001Z")],
    )
    with pytest.raises(ValueError, match="watcher is contaminated"):
        RETRY._require_clean_predecessor_watcher(path)


def test_predecessor_watcher_rejects_immutable_binding_drift(tmp_path: Path) -> None:
    path = tmp_path / "watcher.jsonl"
    _write_jsonl(
        path,
        [_watcher("2026-01-01T00:00:00Z", binding="one"), _watcher("2026-01-01T00:00:05Z", binding="two")],
    )
    with pytest.raises(ValueError, match="watcher is contaminated"):
        RETRY._require_clean_predecessor_watcher(path)


def test_saved_rows_rejects_conflicting_ordinal_collision(tmp_path: Path) -> None:
    base, predecessor = tmp_path / "base", tmp_path / "predecessor"
    (base / "eval_sidecars").mkdir(parents=True)
    (predecessor / "eval_sidecars").mkdir(parents=True)
    base_row = _row(answer="base")
    predecessor_row = _row(answer="predecessor")
    _write_jsonl(base / "eval_sidecars/question_results.e8-t2-r2.jsonl", [base_row])
    _write_jsonl(
        predecessor / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        [predecessor_row],
    )
    with pytest.raises(ValueError, match="sources conflict"):
        RETRY._saved_rows(predecessor, base)
