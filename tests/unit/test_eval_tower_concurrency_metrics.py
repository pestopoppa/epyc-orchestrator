"""EvalTower throughput telemetry for concurrent batches."""

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from eval_tower import EvalTower, QuestionResult  # noqa: E402
from safety_gate import Baseline, SafetyGate  # noqa: E402


def test_eval_concurrency_env_override_still_wins(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "5")
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda _role: False,
    )

    assert eval_tower._eval_concurrency() == 5


def test_eval_batch_progress_callback_reports_logged_milestones(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    events: list[dict] = []
    tower.on_progress = events.append

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            correct=bool(q["correct"]),
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    results = tower._eval_batch(
        [
            {"id": "q1", "correct": True},
            {"id": "q2", "correct": False},
            {"id": "q3", "correct": True},
        ],
        client=object(),  # type: ignore[arg-type]
        log_every=2,
        label="T2",
    )

    assert len(results) == 3
    assert events == [
        {
            "label": "T2",
            "completed_questions": 2,
            "total_questions": 3,
            "correct_questions": 1,
            "correct_pct": 50.0,
            "concurrency": 1,
        }
    ]


def test_eval_batch_fails_remaining_questions_after_no_progress_timeout(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "2")
    monkeypatch.setenv("AUTOPILOT_EVAL_NO_PROGRESS_TIMEOUT_S", "0.05")
    tower = EvalTower(timeout=1)

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        if q["id"] != "fast":
            time.sleep(0.2)
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            correct=True,
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    results = tower._eval_batch(
        [{"id": "fast"}, {"id": "stuck"}, {"id": "queued"}],
        client=object(),  # type: ignore[arg-type]
        label="T1",
    )

    assert [r.question_id for r in results] == ["fast", "stuck", "queued"]
    assert results[0].error is None
    assert results[1].error
    assert results[1].error.startswith("eval_no_progress_timeout:")
    assert results[2].error
    assert results[2].error.startswith("eval_no_progress_timeout:")


def test_eval_concurrency_uses_topology_cap_when_matrix_allows(monkeypatch) -> None:
    from src.runtime import instance_topology

    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")
    monkeypatch.setattr(instance_topology, "max_safe_concurrency", lambda role: 3)
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda role: role == "frontdoor",
    )
    monkeypatch.setattr(
        eval_tower,
        "_live_safe_concurrency",
        lambda role, cap: cap if role == "frontdoor" else 1,
    )

    assert eval_tower._eval_concurrency() == 3


def test_eval_concurrency_caps_to_live_fleet_when_static_topology_allows(monkeypatch) -> None:
    from src.runtime import instance_topology

    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")
    monkeypatch.setattr(instance_topology, "max_safe_concurrency", lambda _role: 3)
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda _role: True,
    )
    monkeypatch.setattr(
        eval_tower,
        "_live_safe_concurrency",
        lambda _role, _cap: 1,
    )

    assert eval_tower._eval_concurrency() == 1


def test_live_safe_concurrency_can_be_disabled_for_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_REQUIRE_LIVE_FLEET", "0")

    assert eval_tower._live_safe_concurrency("frontdoor", 3) == 3


def test_eval_concurrency_falls_back_to_serial_when_matrix_blocks(monkeypatch) -> None:
    from src.runtime import instance_topology

    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")
    monkeypatch.setattr(instance_topology, "max_safe_concurrency", lambda _role: 3)
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda _role: False,
    )

    assert eval_tower._eval_concurrency() == 1


def test_eval_concurrency_falls_back_to_serial_when_matrix_stale(monkeypatch) -> None:
    from src.scheduling import contention

    monkeypatch.setattr(
        contention,
        "matrix_status",
        lambda current_topology_hash: contention.MatrixStatus.STALE,
    )
    monkeypatch.setattr(contention, "topology_fingerprint", lambda _config: "hash")

    assert not eval_tower._same_role_matrix_allows_eval_fanout("frontdoor")


def test_aggregate_uses_batch_throughput_for_concurrent_objective() -> None:
    tower = EvalTower()
    results = [
        QuestionResult(
            question_id="q1",
            suite="math",
            prompt="a",
            expected="a",
            answer="a",
            correct=True,
            tokens_generated=100,
            elapsed_s=10.0,
            eval_concurrency=3,
            eval_wall_s=12.0,
        ),
        QuestionResult(
            question_id="q2",
            suite="math",
            prompt="b",
            expected="b",
            answer="b",
            correct=True,
            tokens_generated=120,
            elapsed_s=12.0,
            eval_concurrency=3,
            eval_wall_s=12.0,
        ),
        QuestionResult(
            question_id="q3",
            suite="math",
            prompt="c",
            expected="c",
            answer="c",
            correct=False,
            tokens_generated=60,
            elapsed_s=6.0,
            eval_concurrency=3,
            eval_wall_s=12.0,
        ),
    ]

    out = tower._aggregate(results, tier=1)

    assert out.speed == (280 / 12.0)
    assert out.speed_metric_mode == "aggregate_batch_tps"
    assert out.median_request_speed == 10.0
    assert out.aggregate_speed == (280 / 12.0)
    assert out.eval_concurrency == 3
    assert out.eval_wall_s == 12.0
    assert out.sum_request_elapsed_s == 28.0
    assert out.details["objective_speed_tps"] == out.speed
    assert out.details["median_request_tps"] == out.median_request_speed
    assert out.details["aggregate_tps"] == out.aggregate_speed
    assert out.details["speed_metric_mode"] == "aggregate_batch_tps"
    assert out.details["task_rate_qph"] == pytest.approx(900.0)
    assert out.details["goodput_qph"] == pytest.approx(600.0)
    assert out.details["tokens_per_solved_task"] == 140.0


def test_aggregate_emits_compact_stable_question_results() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="transient-source-id",
                suite="math",
                prompt="What is two plus two?",
                expected="4",
                answer="4",
                correct=True,
                tokens_generated=5,
                elapsed_s=1.234,
                tools_used=2,
            )
        ],
        tier=1,
    )

    expected_qid = hashlib.sha1(b"math\x00What is two plus two?").hexdigest()[:16]
    assert out.question_results == [
        {
            "qid": expected_qid,
            "suite": "math",
            "partition": "core",
            "correct": True,
            "latency_ms": 1234,
            "tools_used": 2,
        }
    ]
    assert "prompt" not in out.question_results[0]
    assert "answer" not in out.question_results[0]


def test_aggregate_emits_truthy_question_provenance_flags() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                qid="stable-q1",
                suite="coder",
                prompt="Write a function",
                expected="ok",
                answer="timeout",
                correct=False,
                error="read_timeout",
                elapsed_s=2.0,
                route_used="frontdoor->worker_general",
                scoring_method="programmatic",
                partial=True,
                degraded=True,
                exogenous_recovered=True,
                external_restart=True,
                retry_count=1,
                tools_used=1,
                tools_called=["read_file"],
                eval_partition="audit",
            )
        ],
        tier=1,
    )

    assert out.question_results == [
        {
            "qid": "stable-q1",
            "suite": "coder",
            "partition": "audit",
            "correct": False,
            "latency_ms": 2000,
            "tools_used": 1,
            "scoring_method": "programmatic",
            "route": "frontdoor->worker_general",
            "tools_called": ["read_file"],
            "error": True,
            "error_detail": "read_timeout",
            "partial": True,
            "degraded": True,
            "exogenous_recovered": True,
            "external_restart": True,
            "retry_count": 1,
        }
    ]
    assert "prompt" not in out.question_results[0]
    assert "answer" not in out.question_results[0]


def test_eval_result_grep_lines_include_concurrency_metrics() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="general",
                prompt="a",
                expected="a",
                tokens_generated=50,
                elapsed_s=5.0,
                eval_concurrency=2,
                eval_wall_s=5.0,
            )
        ],
        tier=0,
    )

    lines = out.to_grep_lines(trial_id=7, species="test")

    assert "METRIC speed: 10.00" in lines
    assert "METRIC speed_metric_mode: aggregate_batch_tps" in lines
    assert "METRIC median_request_speed: 10.00" in lines
    assert "METRIC aggregate_speed: 10.00" in lines
    assert "METRIC eval_concurrency: 2" in lines
    assert "METRIC eval_wall_s: 5.00" in lines


def test_serial_eval_keeps_median_request_speed_as_objective() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="general",
                prompt="a",
                expected="a",
                tokens_generated=100,
                elapsed_s=10.0,
                eval_concurrency=1,
                eval_wall_s=10.0,
            ),
            QuestionResult(
                question_id="q2",
                suite="general",
                prompt="b",
                expected="b",
                tokens_generated=50,
                elapsed_s=10.0,
                eval_concurrency=1,
                eval_wall_s=10.0,
            ),
        ],
        tier=0,
    )

    assert out.speed == 10.0
    assert out.speed_metric_mode == "median_request_tps"
    assert out.median_request_speed == 10.0
    assert out.aggregate_speed == 15.0


def test_safety_gate_uses_effective_speed_not_raw_median_for_concurrent_eval() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="general",
                prompt="a",
                expected="a",
                correct=True,
                tokens_generated=100,
                elapsed_s=10.0,
                eval_concurrency=3,
                eval_wall_s=15.0,
            ),
            QuestionResult(
                question_id="q2",
                suite="general",
                prompt="b",
                expected="b",
                correct=True,
                tokens_generated=100,
                elapsed_s=10.0,
                eval_concurrency=3,
                eval_wall_s=15.0,
            ),
            QuestionResult(
                question_id="q3",
                suite="general",
                prompt="c",
                expected="c",
                correct=True,
                tokens_generated=100,
                elapsed_s=10.0,
                eval_concurrency=3,
                eval_wall_s=15.0,
            ),
        ],
        tier=1,
    )
    assert out.median_request_speed == 10.0
    assert out.speed == 20.0

    gate = SafetyGate()
    gate.baseline = Baseline(quality=2.0, frontdoor_speed=18.0)
    verdict = gate.check(out)

    assert verdict.passed
    assert "throughput" not in verdict.categories


def test_deep_research_expected_contains_items_are_scoreable() -> None:
    assert eval_tower._is_scoreable_question(
        {
            "id": "dr-1",
            "suite": "deep_research_browsecomp",
            "prompt": "Research alpha beta.",
            "expected_contains": ["alpha beta", "gamma delta"],
        }
    )


def test_eval_question_populates_deterministic_rubric_scores(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_RUBRIC_JUDGE_ROLES", raising=False)
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": (
                "# Summary\n"
                "- alpha beta evidence\n"
                "- gamma delta caveat\n"
                "Source: https://example.test/report\n"
                "Therefore the comparison is grounded in the evidence."
            ),
            "tokens_generated": 20,
            "model": "fake",
            "tools_called": ["web_search", "read_file"],
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "dr-1",
                "suite": "deep_research_browsecomp",
                "prompt": "Research alpha beta.",
                "expected_contains": ["alpha beta", "gamma delta"],
                "scoring_config": {"rubric_pass_threshold": 0.5},
            },
            client,
        )

    assert result.correct is True
    assert result.scoring_method == "rubric"
    assert result.confidence >= 0.5
    assert result.rubric_scores["factual_accuracy"] == 1.0
    assert result.rubric_scores["tool_calls"] > 0


def test_parse_rubric_judge_scores_accepts_fenced_json() -> None:
    parsed = eval_tower._parse_rubric_judge_scores(
        """```json
        {"scores": {"reasoning_trajectory": 0.25, "tool_calls": 1.7, "bad": "x"}}
        ```"""
    )

    assert parsed == {"reasoning_trajectory": 0.25, "tool_calls": 1.0}


def test_eval_question_uses_configured_local_rubric_judge(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_RUBRIC_JUDGE_ROLES", "architect_general")
    monkeypatch.setenv("AUTOPILOT_RUBRIC_JUDGE_TIMEOUT_S", "1")
    tower = EvalTower()
    calls: list[dict] = []

    def _fake_call(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        if kwargs.get("scoring_method") == "rubric_judge":
            return {
                "answer": (
                    '{"scores":{"reasoning_trajectory":0.2,'
                    '"tool_calls":0.4,"outline":0.6,"content_stage":0.8,'
                    '"factual_accuracy":0.9}}'
                ),
                "model": "judge",
            }
        return {
            "answer": (
                "# Summary\n"
                "- alpha beta evidence\n"
                "- gamma delta caveat\n"
                "Source: https://example.test/report\n"
            ),
            "tokens_generated": 20,
            "model": "Llama-Generator",
            "tools_called": ["web_search"],
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "dr-judge",
                "suite": "deep_research_mixed",
                "prompt": "Research alpha beta.",
                "expected_contains": ["alpha beta", "gamma delta"],
                "scoring_config": {"rubric_pass_threshold": 0.5},
            },
            client,
        )

    assert len(calls) == 2
    assert calls[1]["force_role"] == "architect_general"
    assert calls[1]["allow_delegation"] is False
    assert calls[1]["scoring_method"] == "rubric_judge"
    assert result.rubric_scores["reasoning_trajectory"] == 0.2
    assert result.rubric_scores["content_stage"] == 0.8
    assert result.rubric_scores["factual_accuracy"] == 0.9


def test_aggregate_emits_rubric_process_means() -> None:
    tower = EvalTower()

    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="deep_research_browsecomp",
                prompt="a",
                expected="",
                correct=True,
                rubric_scores={
                    "reasoning_trajectory": 0.6,
                    "tool_calls": 0.3,
                    "outline": 0.9,
                    "content_stage": 0.5,
                    "factual_accuracy": 0.8,
                },
            ),
            QuestionResult(
                question_id="q2",
                suite="deep_research_browsecomp",
                prompt="b",
                expected="",
                correct=False,
                rubric_scores={
                    "reasoning_trajectory": 1.0,
                    "tool_calls": 0.9,
                    "outline": 0.1,
                    "content_stage": 0.7,
                    "factual_accuracy": 0.2,
                },
            ),
            QuestionResult(
                question_id="q3",
                suite="math",
                prompt="c",
                expected="c",
                correct=True,
            ),
        ],
        tier=1,
    )

    assert out.rubric_reasoning_trajectory == pytest.approx(0.8)
    assert out.rubric_tool_calls == pytest.approx(0.6)
    assert out.rubric_outline == pytest.approx(0.5)
    assert out.rubric_content_stage == pytest.approx(0.6)
    assert out.details["rubric_dimension_means"]["factual_accuracy"] == pytest.approx(0.5)
    assert out.details["rubric_n_questions"] == 2
    lines = out.to_grep_lines(trial_id=9, species="rubric")
    assert "METRIC rubric_reasoning_trajectory: 0.8000" in lines
    assert "METRIC rubric_tool_calls: 0.6000" in lines
    assert not math.isnan(out.rubric_content_stage)
