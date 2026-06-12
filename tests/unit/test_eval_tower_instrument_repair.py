"""Evidence-plane W4 instrument repair coverage."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from eval_tower import EvalTower  # noqa: E402


def test_programmatic_scorer_runs_with_empty_expected(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": "This response is intentionally non-empty.",
            "tokens_generated": 7,
            "model": "fake",
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "ifeval-empty-expected",
                "suite": "instruction_precision",
                "prompt": "Write any non-empty answer.",
                "expected": "",
                "scoring_method": "programmatic",
                "scoring_config": {"verifier": "non_empty"},
            },
            client,
        )

    assert result.correct is True


def test_empty_expected_still_blocks_plain_exact_match(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": "A non-empty answer must not auto-pass.",
            "tokens_generated": 7,
            "model": "fake",
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "exact-empty-expected",
                "suite": "general",
                "prompt": "Write any non-empty answer.",
                "expected": "",
                "scoring_method": "exact_match",
            },
            client,
        )

    assert result.correct is False
