"""EvalTower throughput telemetry for concurrent batches."""

from __future__ import annotations

import hashlib
import sys
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
