"""Tests for post-hoc model grading helpers."""

from __future__ import annotations

import sys
from types import ModuleType

from src.pipeline_monitor import model_grader


def _install_fake_seeding_orchestrator(monkeypatch, calls: list[dict]) -> None:
    module = ModuleType("seeding_orchestrator")

    def call_orchestrator_forced(**kwargs):
        calls.append(kwargs)
        return {"answer": "Reasoning\nA"}

    module.call_orchestrator_forced = call_orchestrator_forced
    monkeypatch.setitem(sys.modules, "seeding_orchestrator", module)


def test_grade_answer_defaults_to_live_worker_general(monkeypatch):
    calls: list[dict] = []
    _install_fake_seeding_orchestrator(monkeypatch, calls)

    result = model_grader.grade_answer(
        {
            "spec_name": "quality",
            "prompt_template": "Question: {question}",
            "choice_strings": ["A"],
            "choice_scores": {"A": 1.0},
        },
        {"question_id": "q1"},
    )

    assert result is not None
    assert result["classification"] == "A"
    assert calls[0]["force_role"] == "worker_general"


def test_grade_answer_preserves_explicit_judge_role(monkeypatch):
    calls: list[dict] = []
    _install_fake_seeding_orchestrator(monkeypatch, calls)

    model_grader.grade_answer(
        {
            "spec_name": "quality",
            "judge_role": "architect_general",
            "prompt_template": "Question: {question}",
            "choice_strings": ["A"],
            "choice_scores": {"A": 1.0},
        },
        {"question_id": "q1"},
    )

    assert calls[0]["force_role"] == "architect_general"
