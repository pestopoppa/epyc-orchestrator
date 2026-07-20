"""Regression coverage for executable code-oracle scoring."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

import pytest

from debug_scorer import ScoringUnavailableError, score_answer  # noqa: E402


def test_code_execution_without_oracle_does_not_pass_on_importable_code() -> None:
    answer = "```python\nprint('hello')\n```"

    assert not score_answer(
        answer=answer,
        expected="print",
        scoring_method="code_execution",
        scoring_config={"language": "python", "timeout": 5},
    )


def test_code_execution_comment_only_assertions_are_not_an_oracle() -> None:
    answer = "```python\ndef add(a, b):\n    return a + b\n```"

    assert not score_answer(
        answer=answer,
        expected="",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 5,
            "test_code": "# assert add(1, 2) == 3\n",
        },
    )


def test_code_execution_vacuous_assert_true_is_not_an_oracle() -> None:
    answer = "```python\ndef add(a, b):\n    return a - b\n```"

    assert not score_answer(
        answer=answer,
        expected="",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 5,
            "test_code": "assert True\n",
        },
    )


def test_code_execution_entry_point_requires_real_cases() -> None:
    answer = "```python\ndef add(a, b):\n    return a + b\n```"

    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer=answer,
            expected="add",
            scoring_method="code_execution",
            scoring_config={
                "language": "python",
                "timeout": 5,
                "entry_point": "add",
            },
        )


def test_code_execution_entry_point_cases_execute() -> None:
    assert score_answer(
        answer="```python\ndef add(a, b):\n    return a + b\n```",
        expected="",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 5,
            "entry_point": "add",
            "entry_point_cases": [
                {"args": [1, 2], "expected": 3},
                {"args": [-1, 5], "expected": 4},
            ],
        },
    )
    assert not score_answer(
        answer="```python\ndef add(a, b):\n    return a - b\n```",
        expected="",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 5,
            "entry_point": "add",
            "entry_point_cases": [{"args": [1, 2], "expected": 3}],
        },
    )


def test_code_execution_rejects_unsafe_entry_point_name() -> None:
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="```python\ndef add(a, b):\n    return a + b\n```",
            expected="",
            scoring_method="code_execution",
            scoring_config={
                "language": "python",
                "timeout": 5,
                "entry_point": "add(); import os",
                "entry_point_cases": [{"args": [1, 2], "expected": 3}],
            },
        )


def test_code_execution_runs_unittest_cases_without_embedded_runner() -> None:
    test_code = """
import unittest

class TestCases(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""

    assert score_answer(
        answer="```python\ndef add(a, b):\n    return a + b\n```",
        expected="",
        scoring_method="code_execution",
        scoring_config={"language": "python", "timeout": 5, "test_code": test_code},
    )
    assert not score_answer(
        answer="```python\ndef add(a, b):\n    return a - b\n```",
        expected="",
        scoring_method="code_execution",
        scoring_config={"language": "python", "timeout": 5, "test_code": test_code},
    )
