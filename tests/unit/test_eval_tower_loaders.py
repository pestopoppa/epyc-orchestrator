"""EvalTower loader validation and fail-closed behavior."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from eval_tower import EvalTower  # noqa: E402


def _install_fake_question_pool(monkeypatch, pool_file: Path, load_pool):  # noqa: ANN001
    module = types.ModuleType("question_pool")
    module.POOL_FILE = pool_file
    module.load_pool = load_pool
    monkeypatch.setitem(sys.modules, "question_pool", module)


def _advance_mtime(path: Path) -> None:
    current = path.stat().st_mtime_ns
    os.utime(path, ns=(current + 1_000_000_000, current + 1_000_000_000))


def test_sentinel_loader_rejects_promptless_rows(tmp_path) -> None:
    sentinel_path = tmp_path / "sentinels.yaml"
    sentinel_path.write_text(
        """
- id: bad
  suite: general
  expected: ok
- id: good
  suite: general
  prompt: Return ok.
  expected: ok
""".lstrip(),
        encoding="utf-8",
    )

    tower = EvalTower(sentinel_path=sentinel_path)

    assert [q["id"] for q in tower._load_sentinels()] == ["good"]
    assert tower._sentinel_load_details["dropped_rows"] == 1
    assert tower._sentinel_load_details["drop_reasons"] == {"missing_prompt": 1}


def test_sentinel_loader_does_not_cache_missing_file(tmp_path) -> None:
    sentinel_path = tmp_path / "sentinels.yaml"
    tower = EvalTower(sentinel_path=sentinel_path)

    assert tower._load_sentinels() == []

    sentinel_path.write_text(
        """
- id: recovered
  suite: general
  prompt: Return ok.
  expected: ok
""".lstrip(),
        encoding="utf-8",
    )

    assert [q["id"] for q in tower._load_sentinels()] == ["recovered"]


def test_sentinel_loader_refreshes_on_mtime_change(tmp_path) -> None:
    sentinel_path = tmp_path / "sentinels.yaml"
    sentinel_path.write_text(
        """
- id: first
  suite: general
  prompt: Return first.
  expected: first
""".lstrip(),
        encoding="utf-8",
    )
    tower = EvalTower(sentinel_path=sentinel_path)
    assert [q["id"] for q in tower._load_sentinels()] == ["first"]

    sentinel_path.write_text(
        """
- id: second
  suite: general
  prompt: Return second.
  expected: second
""".lstrip(),
        encoding="utf-8",
    )
    _advance_mtime(sentinel_path)

    assert [q["id"] for q in tower._load_sentinels()] == ["second"]


def test_eval_t0_empty_sentinels_returns_details_error(tmp_path) -> None:
    result = EvalTower(sentinel_path=tmp_path / "missing.yaml").eval_t0()

    assert result.quality == 0
    assert result.reliability == 0
    assert result.details["decision_grade"] is False
    assert result.details["loader_error"]["error"] == "no_valid_sentinel_questions"
    assert result.details["loader_error"]["retryable_without_restart"] is True
    assert result.details["test_profile"]["n_questions"] == 0


def test_pool_loader_rejects_invalid_rows(monkeypatch, tmp_path) -> None:
    pool_file = tmp_path / "question_pool.jsonl"
    pool_file.write_text("{}\n", encoding="utf-8")

    def _load_pool():  # noqa: ANN001
        return {
            "math": [
                {"id": "bad", "suite": "math", "expected": "4"},
                {
                    "id": "good",
                    "suite": "math",
                    "prompt": "2+2?",
                    "expected": "4",
                },
            ]
        }

    _install_fake_question_pool(monkeypatch, pool_file, _load_pool)
    tower = EvalTower()

    pool = tower._load_pool()

    assert [q["id"] for q in pool["math"]] == ["good"]
    assert tower._pool_load_details["dropped_rows"] == 1
    assert tower._pool_load_details["drop_reasons"] == {"missing_prompt": 1}


def test_pool_loader_does_not_cache_failure(monkeypatch, tmp_path) -> None:
    pool_file = tmp_path / "question_pool.jsonl"
    pool_file.write_text("{}\n", encoding="utf-8")
    state = {"fail": True}

    def _load_pool():  # noqa: ANN001
        if state["fail"]:
            raise RuntimeError("temporary pool read failure")
        return {
            "math": [
                {
                    "id": "recovered",
                    "suite": "math",
                    "prompt": "2+2?",
                    "expected": "4",
                }
            ]
        }

    _install_fake_question_pool(monkeypatch, pool_file, _load_pool)
    tower = EvalTower()

    assert tower._load_pool() == {}

    state["fail"] = False
    assert [q["id"] for q in tower._load_pool()["math"]] == ["recovered"]


def test_pool_loader_refreshes_on_mtime_change(monkeypatch, tmp_path) -> None:
    pool_file = tmp_path / "question_pool.jsonl"
    pool_file.write_text("{}\n", encoding="utf-8")
    state = {
        "pool": {
            "math": [
                {
                    "id": "first",
                    "suite": "math",
                    "prompt": "2+2?",
                    "expected": "4",
                }
            ]
        }
    }

    def _load_pool():  # noqa: ANN001
        return state["pool"]

    _install_fake_question_pool(monkeypatch, pool_file, _load_pool)
    tower = EvalTower()
    assert [q["id"] for q in tower._load_pool()["math"]] == ["first"]

    state["pool"] = {
        "math": [
            {
                "id": "second",
                "suite": "math",
                "prompt": "3+3?",
                "expected": "6",
            }
        ]
    }
    pool_file.write_text('{"changed": true}\n', encoding="utf-8")
    _advance_mtime(pool_file)

    assert [q["id"] for q in tower._load_pool()["math"]] == ["second"]


def test_eval_t1_empty_pool_returns_details_error(monkeypatch, tmp_path) -> None:
    pool_file = tmp_path / "question_pool.jsonl"
    pool_file.write_text("{}\n", encoding="utf-8")
    _install_fake_question_pool(monkeypatch, pool_file, lambda: {})
    monkeypatch.delenv("AUTOPILOT_T1_CORE_ID", raising=False)
    monkeypatch.delenv("AUTOPILOT_T1_CORE_PATH", raising=False)

    result = EvalTower().eval_t1(n=7, seed=3)

    assert result.quality == 0
    assert result.reliability == 0
    assert result.details["decision_grade"] is False
    assert result.details["loader_error"]["error"] == "no_valid_question_pool"
    assert result.details["core_selection"] == "legacy_pool_seed"
    assert result.details["test_profile"]["n_questions"] == 0


def test_eval_t2_empty_pool_returns_details_error(monkeypatch, tmp_path) -> None:
    pool_file = tmp_path / "question_pool.jsonl"
    pool_file.write_text("{}\n", encoding="utf-8")
    _install_fake_question_pool(monkeypatch, pool_file, lambda: {})

    result = EvalTower().eval_t2(n=11, seed=5)

    assert result.quality == 0
    assert result.reliability == 0
    assert result.details["decision_grade"] is False
    assert result.details["loader_error"]["error"] == "no_valid_question_pool"
    assert result.details["promotion_eval"] is False
    assert result.details["test_profile"]["n_questions"] == 0
