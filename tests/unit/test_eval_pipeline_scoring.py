"""Decoupled generation/scoring pipeline in EvalTower._eval_batch (2026-07-22).

The workers>1 path splits ``_eval_question`` into ``_generate_question`` (runs on
the topology-capped generation pool) and ``_score_generation`` (runs on a
separate, wider SCORING pool sized by ``AUTOPILOT_EVAL_SCORING_CONCURRENCY``) so a
scoring-bound suite stops capping total throughput at the serving fan-out. These
tests pin: verdict/order parity vs the serial path, the throughput property
(wider scoring => proportionally lower wall), the no-progress watchdog tripping on
a hung SCORER, the sidecar's scored_at_s >= ended_at_s separation, and the serial
path's byte-identical behavior (no scored_at_s, no pipeline leakage).

INFERENCE-FREE: generation and scoring are mocked; the only latency is small
injected sleeps (<= ~40ms scale). Run:
  .venv/bin/python -m pytest tests/unit/test_eval_pipeline_scoring.py -q
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from eval_tower import EvalTower, QuestionResult  # noqa: E402


# ── fakes ─────────────────────────────────────────────────────────────────────


def _mk_generate(latency_s: float = 0.0):
    """Fake GENERATION phase: builds a _GenOutcome from a question dict.

    Carries the payload the real scorer needs, so a test can run the REAL
    ``_score_generation`` on top for verdict parity, or override scoring for the
    throughput/watchdog tests. ``latency_s`` injects generation cost.
    """

    def fake_generate(q: dict, client: object) -> "eval_tower._GenOutcome":
        if latency_s:
            time.sleep(latency_s)
        answer = str(q.get("_mock_answer", q.get("expected", "")))
        suite = str(q.get("suite", "unit"))
        prompt = str(q.get("prompt", ""))
        return eval_tower._GenOutcome(
            gen_ended_at_s=time.time(),
            resp={"answer": answer, "tokens_generated": 5, "model": "mock"},
            answer=answer,
            error=None,
            tokens=5,
            elapsed=float(q.get("_mock_elapsed", 0.01)),
            host_covariates={},
            question_id=str(q.get("id", "unknown")),
            suite=suite,
            prompt=prompt,
            expected=str(q.get("expected", "")),
            stable_qid=eval_tower._stable_question_qid(suite, prompt),
            scoring_method=str(q.get("scoring_method", "exact_match")),
            scoring_config=(
                q.get("scoring_config") if isinstance(q.get("scoring_config"), dict) else {}
            ),
            eval_partition="core",
        )

    return fake_generate


def _mk_score(latency_s: float):
    """Fake SCORING phase with injected latency; verdict = (answer == expected)."""

    def fake_score(q: dict, outcome: "eval_tower._GenOutcome", client: object) -> QuestionResult:
        if outcome.final_result is not None:
            return outcome.final_result
        if latency_s:
            time.sleep(latency_s)
        return QuestionResult(
            question_id=outcome.question_id,
            suite=outcome.suite,
            prompt=outcome.prompt,
            expected=outcome.expected,
            qid=outcome.stable_qid,
            answer=outcome.answer,
            correct=(outcome.answer == outcome.expected),
            error=outcome.error,
            tokens_generated=outcome.tokens,
            elapsed_s=outcome.elapsed,
            eval_partition=outcome.eval_partition,
        )

    return fake_score


def _fixture_questions(n: int = 8) -> list[dict]:
    return [
        {
            "id": f"q{i}",
            "suite": "unit",
            "prompt": f"prompt-{i}",
            "expected": str(i % 3),
            "scoring_method": "exact_match",
            # Half match expected -> correct; half don't -> wrong.
            "_mock_answer": str(i % 3) if i % 2 == 0 else "zzz",
        }
        for i in range(n)
    ]


# ── (a) verdict / order parity: serial vs pipelined ────────────────────────────


def test_pipelined_verdicts_and_order_match_serial(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    questions = _fixture_questions(8)

    def run(concurrency: int) -> list[tuple]:
        monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", str(concurrency))
        tower = EvalTower()
        # Real _score_generation runs on top of the fake generation, in BOTH
        # paths -> any verdict difference would be the scheduling, not the scorer.
        monkeypatch.setattr(tower, "_generate_question", _mk_generate())
        results = tower._eval_batch(list(questions), client=object(), label="parity")
        return [(r.question_id, r.correct, r.error, r.scoring_method) for r in results]

    serial = run(1)
    pipelined = run(4)

    assert [r[0] for r in serial] == [q["id"] for q in questions]  # order preserved
    assert serial == pipelined  # identical per-question verdicts + order
    # sanity: the fixture actually produced a mix of correct/incorrect
    assert any(r[1] for r in serial) and any(not r[1] for r in serial)


# ── (b) throughput property: wider scoring => proportionally lower wall ─────────


def test_wider_scoring_pool_lowers_wall_proportionally(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "4")  # fixed generation width
    n = 24
    score_latency = 0.03  # >> generation latency (0), so scoring gates the wall
    questions = [
        {"id": f"q{i}", "suite": "unit", "prompt": f"p{i}", "expected": "x", "_mock_answer": "x"}
        for i in range(n)
    ]

    def wall_for(scoring_width: int) -> float:
        monkeypatch.setenv("AUTOPILOT_EVAL_SCORING_CONCURRENCY", str(scoring_width))
        tower = EvalTower()
        monkeypatch.setattr(tower, "_generate_question", _mk_generate(latency_s=0.0))
        monkeypatch.setattr(tower, "_score_generation", _mk_score(latency_s=score_latency))
        t0 = time.perf_counter()
        results = tower._eval_batch(list(questions), client=object(), label="tput")
        wall = time.perf_counter() - t0
        assert len(results) == n and all(r.correct for r in results)
        return wall

    wall4 = wall_for(4)
    wall8 = wall_for(8)
    ratio = wall4 / wall8

    # Ideal ~2x (n/4 vs n/8 waves of score_latency). Assert loosely.
    assert wall8 < wall4
    assert ratio > 1.4, f"expected wider scoring ~2x faster, got ratio={ratio:.2f} (wall4={wall4:.3f} wall8={wall8:.3f})"


# ── (c) no-progress watchdog trips on a hung SCORER ────────────────────────────


def test_no_progress_trips_on_hung_scorer(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "2")
    monkeypatch.setenv("AUTOPILOT_EVAL_SCORING_CONCURRENCY", "4")
    monkeypatch.setenv("AUTOPILOT_EVAL_NO_PROGRESS_TIMEOUT_S", "0.05")
    monkeypatch.setenv("AUTOPILOT_EVAL_ORPHAN_DRAIN_TIMEOUT_S", "0.01")
    release = threading.Event()
    tower = EvalTower(timeout=1)

    # Generation completes fast; scoring hangs on every question.
    monkeypatch.setattr(tower, "_generate_question", _mk_generate(latency_s=0.0))

    def hung_score(q, outcome, client):  # noqa: ANN001
        release.wait(5.0)
        return outcome.final_result or QuestionResult(
            question_id=outcome.question_id, suite=outcome.suite,
            prompt=outcome.prompt, expected=outcome.expected, correct=True,
        )

    monkeypatch.setattr(tower, "_score_generation", hung_score)

    try:
        results = tower._eval_batch(
            [{"id": "a", "suite": "u", "prompt": "a", "expected": "a", "_mock_answer": "a"},
             {"id": "b", "suite": "u", "prompt": "b", "expected": "b", "_mock_answer": "b"},
             {"id": "c", "suite": "u", "prompt": "c", "expected": "c", "_mock_answer": "c"}],
            client=object(),
            label="hung",
        )
    finally:
        release.set()

    assert [r.question_id for r in results] == ["a", "b", "c"]  # order preserved
    for r in results:
        assert r.error and r.error.startswith("eval_no_progress_timeout:"), r.error


# ── (d) sidecar timing: scored_at_s >= ended_at_s (generation interval) ─────────


def test_sidecar_rows_carry_scored_at_after_generation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "4")
    monkeypatch.setenv("AUTOPILOT_EVAL_SCORING_CONCURRENCY", "8")
    tower = EvalTower()
    tower.set_trial_context(707)
    monkeypatch.setattr(tower, "_generate_question", _mk_generate(latency_s=0.0))
    monkeypatch.setattr(tower, "_score_generation", _mk_score(latency_s=0.02))

    questions = [
        {"id": f"q{i}", "suite": "unit", "prompt": f"p{i}", "expected": "x", "_mock_answer": "x",
         "_mock_elapsed": 0.01}
        for i in range(6)
    ]
    tower._eval_batch(questions, client=object(), label="timing")

    rows = [
        json.loads(line)
        for line in (tmp_path / "trial_707" / "question_results.jsonl").read_text().splitlines()
    ]
    qrows = [r for r in rows if r["row_type"] == "question_result"]
    assert len(qrows) == 6
    for row in qrows:
        assert "scored_at_s" in row
        assert row["scored_at_s"] >= row["ended_at_s"]  # scoring finished after generation
        assert row["started_at_s"] <= row["ended_at_s"]
        # ended_at_s is the GENERATION interval end, not the scoring end.
        assert abs((row["ended_at_s"] - row["started_at_s"]) - row["elapsed_s"]) < 0.01


# ── (e) serial path byte-identical: no pipeline fields leak in ─────────────────


def test_serial_path_omits_pipeline_fields(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    tower.set_trial_context(808)
    monkeypatch.setattr(tower, "_generate_question", _mk_generate(latency_s=0.0))

    questions = _fixture_questions(4)
    results = tower._eval_batch(list(questions), client=object(), label="serial")

    # Behavioral identity: same verdicts as calling _eval_question directly.
    tower2 = EvalTower()
    monkeypatch.setattr(tower2, "_generate_question", _mk_generate(latency_s=0.0))
    direct = [tower2._eval_question(q, object()) for q in questions]
    assert [(r.question_id, r.correct) for r in results] == [
        (r.question_id, r.correct) for r in direct
    ]

    rows = [
        json.loads(line)
        for line in (tmp_path / "trial_808" / "question_results.jsonl").read_text().splitlines()
    ]
    qrows = [r for r in rows if r["row_type"] == "question_result"]
    assert len(qrows) == 4
    for row in qrows:
        # Serial path uses the append-time ended_at_s and adds NO scored_at_s
        # (pipeline-only) — byte-identical to the pre-pipeline writer.
        assert "scored_at_s" not in row


# ── _eval_scoring_concurrency: default floor + env override clamp ──────────────


def test_scoring_concurrency_never_below_generation_width(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_EVAL_SCORING_CONCURRENCY", raising=False)
    assert eval_tower._eval_scoring_concurrency(4) >= 4
    # Generation wider than the default => scoring floored at generation width.
    assert eval_tower._eval_scoring_concurrency(100) >= 100


def test_scoring_concurrency_env_override_clamped_to_gen_width(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_SCORING_CONCURRENCY", "9")
    assert eval_tower._eval_scoring_concurrency(4) == 9
    monkeypatch.setenv("AUTOPILOT_EVAL_SCORING_CONCURRENCY", "2")  # below gen width
    assert eval_tower._eval_scoring_concurrency(4) == 4  # clamped up
    monkeypatch.setenv("AUTOPILOT_EVAL_SCORING_CONCURRENCY", "not-an-int")
    assert eval_tower._eval_scoring_concurrency(4) >= 4  # falls back to default
