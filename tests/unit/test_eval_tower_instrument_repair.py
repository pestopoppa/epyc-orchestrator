"""Evidence-plane W4 instrument repair coverage."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from eval_tower import EvalTower, QuestionResult  # noqa: E402


class _PrefixRng:
    def sample(self, population, k):  # noqa: ANN001
        return list(population[:k])


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


def test_empty_expected_text_scorer_is_not_scoreable() -> None:
    assert not eval_tower._is_scoreable_question(
        {
            "id": "dead-text",
            "expected": "",
            "scoring_method": "substring",
        }
    )
    assert eval_tower._is_scoreable_question(
        {
            "id": "expected-free-programmatic",
            "expected": "",
            "scoring_method": "programmatic",
        }
    )


def test_sampling_replaces_unscoreable_items_from_same_suite() -> None:
    suite_qs = [
        {
            "id": "dead-text",
            "expected": "",
            "scoring_method": "substring",
        },
        {
            "id": "valid-text",
            "expected": "answer",
            "scoring_method": "substring",
        },
        {
            "id": "valid-programmatic",
            "expected": "",
            "scoring_method": "programmatic",
        },
    ]

    sample = eval_tower._sample_scoreable_questions(
        "instruction_precision",
        suite_qs,
        per_suite=2,
        rng=_PrefixRng(),
    )

    assert [q["id"] for q in sample] == ["valid-text", "valid-programmatic"]


def test_eval_question_records_chat_response_routed_to(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": "FRIEND",
            "routed_to": "worker_vision",
            "tokens_generated": 1,
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "vl-route-telemetry",
                "suite": "vl",
                "prompt": "what is written in the image?",
                "expected": "FRIEND",
                "scoring_method": "exact_match",
                "image_path": "/tmp/example.png",
            },
            client,
        )

    assert result.route_used == "worker_vision"


def test_t0_sentinel_suites_are_namespaced_without_mutating_source(monkeypatch) -> None:
    sentinels = [
        {
            "id": "sentinel-a",
            "suite": "general",
            "prompt": "A",
            "expected": "A",
            "scoring_method": "exact_match",
        },
        {
            "id": "sentinel-b",
            "suite": "math",
            "prompt": "B",
            "expected": "B",
            "scoring_method": "exact_match",
        },
    ]
    tower = EvalTower()
    tower._sentinels = sentinels

    def _fake_eval_batch(self, questions, client, **_kwargs):  # noqa: ANN001, ARG001
        return [
            QuestionResult(
                question_id=q["id"],
                suite=q["suite"],
                prompt=q["prompt"],
                expected=q["expected"],
                correct=q["id"] == "sentinel-a",
                tokens_generated=1,
                elapsed_s=1.0,
            )
            for q in questions
        ]

    monkeypatch.setattr(EvalTower, "_eval_batch", _fake_eval_batch)

    result = tower.eval_t0()

    assert result.per_suite_quality == {
        "sentinel_general": 3.0,
        "sentinel_math": 0.0,
    }
    assert sentinels[0]["suite"] == "general"
    assert sentinels[1]["suite"] == "math"
