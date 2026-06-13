"""Regression coverage for executable code-oracle scoring."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from debug_scorer import score_answer  # noqa: E402


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
