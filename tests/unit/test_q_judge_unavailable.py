"""JEV-1: judge failure is UNAVAILABLE, never a fabricated score."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestration.repl_memory.q_judge import (
    ClaudeAsJudge,
    JudgeResult,
    JudgeStatus,
    tally_judge_results,
)


def _judge() -> ClaudeAsJudge:
    return ClaudeAsJudge(model_path=Path("/nonexistent.gguf"), binary_path=Path("/nonexistent"))


TASK = {"task_type": "code", "objective": "x", "priority": "p"}


@pytest.mark.parametrize(
    "exc",
    [subprocess.TimeoutExpired(cmd="j", timeout=1), OSError("boom")],
)
def test_subprocess_failure_is_unavailable(monkeypatch, exc):
    def raise_(*a, **k):
        raise exc

    monkeypatch.setattr(subprocess, "run", raise_)
    r = _judge().score_routing(TASK, ["coder"], "ok")
    assert r.status is JudgeStatus.UNAVAILABLE
    assert r.score is None and not r.available


@pytest.mark.parametrize("text", ["", "garbage", "SCORE: nine", "SCORE: 7\nREASON: x"])
def test_unparseable_is_unavailable(text):
    r = _judge()._parse_score(text)
    assert r.status is JudgeStatus.UNAVAILABLE and r.score is None


def test_valid_score_parses():
    r = _judge()._parse_score("SCORE: 0\nREASON: wrong")
    assert r == JudgeResult(JudgeStatus.SCORED, 0, "wrong")


def test_tally_counts_unavailable_separately():
    t = tally_judge_results(
        [
            JudgeResult(JudgeStatus.SCORED, 3, ""),
            JudgeResult(JudgeStatus.SCORED, 1, ""),
            JudgeResult.unavailable("timeout"),
        ]
    )
    assert t == {"scored": 2, "unavailable": 1, "mean_score": 2.0}
    assert tally_judge_results([JudgeResult.unavailable("x")])["mean_score"] is None
