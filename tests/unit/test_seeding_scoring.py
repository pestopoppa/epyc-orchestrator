"""Unit tests for benchmark seeding_scoring helper module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location("seeding_scoring_test", _ROOT / "seeding_scoring.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["seeding_scoring_test"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_score_answer_deterministic_delegates_to_debug_scorer():
    calls = {}

    def _score(answer, expected, method, config):  # noqa: ANN001
        calls["answer"] = answer
        calls["expected"] = expected
        calls["method"] = method
        calls["config"] = config
        return True

    stub = ModuleType("debug_scorer")
    stub.score_answer = _score
    prev = sys.modules.get("debug_scorer")
    sys.modules["debug_scorer"] = stub
    try:
        assert _MOD.score_answer_deterministic("a", "b") is True
    finally:
        if prev is None:
            sys.modules.pop("debug_scorer", None)
        else:
            sys.modules["debug_scorer"] = prev

    assert calls == {
        "answer": "a",
        "expected": "b",
        "method": "exact_match",
        "config": {},
    }


def test_score_answer_deterministic_passes_explicit_config():
    calls = {}

    def _score(answer, expected, method, config):  # noqa: ANN001
        calls["config"] = config
        return False

    stub = ModuleType("debug_scorer")
    stub.score_answer = _score
    prev = sys.modules.get("debug_scorer")
    sys.modules["debug_scorer"] = stub
    try:
        assert (
            _MOD.score_answer_deterministic(
                "a",
                "b",
                scoring_method="f1",
                scoring_config={"extract_patterns": ["x"]},
            )
            is False
        )
    finally:
        if prev is None:
            sys.modules.pop("debug_scorer", None)
        else:
            sys.modules["debug_scorer"] = prev

    assert calls["config"] == {"extract_patterns": ["x"]}


def test_classify_error_branches():
    assert _MOD._classify_error(None) == "none"
    assert _MOD._classify_error("Connection reset by peer") == "infrastructure"
    assert _MOD._classify_error("incorrect answer") == "task_failure"


def test_is_coding_task_heuristic():
    assert _MOD._is_coding_task("Implement a Python function with unit tests") is True
    assert _MOD._is_coding_task("What is the capital of France?") is False


def test_adaptive_timeout_bounds():
    assert _MOD._adaptive_timeout_s(
        role="frontdoor",
        mode="direct",
        prompt="x",
        is_vl=False,
        hard_timeout_s=10,
    ) == 60
    assert _MOD._adaptive_timeout_s(
        role="frontdoor",
        mode="direct",
        prompt="x",
        is_vl=False,
        hard_timeout_s=120,
    ) == 120
    assert _MOD._adaptive_timeout_s(
        role="frontdoor",
        mode="direct",
        prompt="x",
        is_vl=False,
        hard_timeout_s=0,
    ) == max(60, int(_MOD.DEFAULT_TIMEOUT))


def test_bump_timeout_from_observed_branches():
    assert _MOD._bump_timeout_from_observed(
        current_s=90,
        observed_s=0,
        factor=2.0,
        slack_s=10,
        hard_timeout_s=200,
        role_cap_s=999,
    ) == 90

    # observed budget lower than current => unchanged
    assert _MOD._bump_timeout_from_observed(
        current_s=100,
        observed_s=20,
        factor=2.0,
        slack_s=5,
        hard_timeout_s=200,
        role_cap_s=999,
    ) == 100

    # observed budget larger => bumped but capped by hard timeout floor/ceiling logic
    assert _MOD._bump_timeout_from_observed(
        current_s=10,
        observed_s=500,
        factor=2.0,
        slack_s=0,
        hard_timeout_s=300,
        role_cap_s=999,
    ) == 300

    # hard_timeout_s fallback to DEFAULT_TIMEOUT path
    assert _MOD._bump_timeout_from_observed(
        current_s=10,
        observed_s=1000,
        factor=2.0,
        slack_s=0,
        hard_timeout_s=0,
        role_cap_s=999,
    ) == max(60, int(_MOD.DEFAULT_TIMEOUT))
