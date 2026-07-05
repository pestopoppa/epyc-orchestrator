"""Tests for RI-10 canary rollout decision reporting."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis import ri10_canary_decision_report as report_mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _routing_row(
    task_id: str,
    timestamp: str,
    *,
    mode: str,
    role: str = "frontdoor",
    cost: float = 0.001,
) -> dict:
    return {
        "event_type": "routing_decision",
        "task_id": task_id,
        "timestamp": timestamp,
        "data": {
            "routing": [role],
            "factual_risk_score": 0.72,
            "factual_risk_band": "high",
            "factual_risk_mode": mode,
            "risk_gate_action": "not_enforced",
            "estimated_cost": cost,
        },
    }


def _completed(task_id: str, timestamp: str, *, elapsed: float, outcome: str = "success") -> dict:
    return {
        "event_type": "task_completed",
        "task_id": task_id,
        "timestamp": timestamp,
        "outcome": outcome,
        "outcome_details": f"Direct answer mode (frontdoor), {elapsed:.3f}s",
        "data": {"final_answer_role": "frontdoor"},
    }


def test_build_report_holds_when_factuality_is_unscored(tmp_path: Path) -> None:
    rows = [
        _routing_row("e1", "2026-06-20T00:00:00Z", mode="enforce", cost=0.001),
        _completed("e1", "2026-06-20T00:00:02Z", elapsed=1.0),
        _routing_row("s1", "2026-06-20T00:01:00Z", mode="shadow", cost=0.002),
        _completed("s1", "2026-06-20T00:01:03Z", elapsed=2.0),
    ]
    _write_jsonl(tmp_path / "2026-06-20.jsonl", rows)

    report = report_mod.build_report(
        tmp_path,
        telemetry_health_start="2026-06-20",
        decision_gate=2,
        min_arm_samples=1,
    )

    assert report["sample_coverage"]["canary_decision_ready"] is True
    assert report["decision"]["status"] == "hold_quality_unscored"
    assert report["decision"]["blockers"] == ["factuality_not_scored"]
    assert report["arms"]["enforce"]["rows"] == 1
    assert report["arms"]["shadow"]["rows"] == 1
    assert report["comparison"]["estimated_cost_mean_ratio_enforce_over_shadow"] == 0.5


def test_build_report_blocks_latency_regression(tmp_path: Path) -> None:
    rows = [
        _routing_row("e1", "2026-06-20T00:00:00Z", mode="enforce"),
        _completed("e1", "2026-06-20T00:00:30Z", elapsed=30.0),
        _routing_row("s1", "2026-06-20T00:01:00Z", mode="shadow"),
        _completed("s1", "2026-06-20T00:01:03Z", elapsed=3.0),
    ]
    _write_jsonl(tmp_path / "2026-06-20.jsonl", rows)

    report = report_mod.build_report(
        tmp_path,
        telemetry_health_start="2026-06-20",
        decision_gate=2,
        min_arm_samples=1,
    )

    assert report["decision"]["status"] == "hold"
    assert "p95_latency_regression" in report["decision"]["blockers"]
    assert "factuality_not_scored" in report["decision"]["blockers"]
    assert report["comparison"]["latency_p95_ratio_enforce_over_shadow"] == 10.0


def test_build_report_counts_escalation_and_plan_review_by_arm(tmp_path: Path) -> None:
    rows = [
        _routing_row("e1", "2026-06-20T00:00:00Z", mode="enforce"),
        {
            "event_type": "escalation_triggered",
            "task_id": "e1",
            "timestamp": "2026-06-20T00:00:01Z",
            "data": {"from_tier": "frontdoor", "to_tier": "coder_escalation"},
        },
        {
            "event_type": "plan_reviewed",
            "task_id": "e1",
            "timestamp": "2026-06-20T00:00:02Z",
            "data": {"decision": "add"},
        },
        _completed("e1", "2026-06-20T00:00:03Z", elapsed=1.0),
        _routing_row("s1", "2026-06-20T00:01:00Z", mode="shadow"),
        _completed("s1", "2026-06-20T00:01:03Z", elapsed=1.0),
    ]
    _write_jsonl(tmp_path / "2026-06-20.jsonl", rows)

    report = report_mod.build_report(
        tmp_path,
        telemetry_health_start="2026-06-20",
        decision_gate=2,
        min_arm_samples=1,
    )

    enforce = report["arms"]["enforce"]
    assert enforce["escalation_task_count"] == 1
    assert enforce["plan_review_task_count"] == 1
    assert enforce["plan_review_decisions"] == {"add": 1}
    assert "escalation_rate_inflation" in report["decision"]["blockers"]
    assert "review_rate_inflation" in report["decision"]["blockers"]


def test_build_report_awaits_telemetry_when_sample_is_not_ready(tmp_path: Path) -> None:
    rows = [
        _routing_row("e1", "2026-06-20T00:00:00Z", mode="enforce"),
        _completed("e1", "2026-06-20T00:00:03Z", elapsed=1.0),
    ]
    _write_jsonl(tmp_path / "2026-06-20.jsonl", rows)

    report = report_mod.build_report(
        tmp_path,
        telemetry_health_start="2026-06-20",
        decision_gate=2,
        min_arm_samples=1,
    )

    assert report["decision"]["status"] == "awaiting_telemetry"
    assert "telemetry_not_decision_ready" in report["decision"]["blockers"]


def test_render_markdown_includes_status_and_arm_table(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "2026-06-20.jsonl",
        [
            _routing_row("e1", "2026-06-20T00:00:00Z", mode="enforce"),
            _completed("e1", "2026-06-20T00:00:02Z", elapsed=1.0),
            _routing_row("s1", "2026-06-20T00:01:00Z", mode="shadow"),
            _completed("s1", "2026-06-20T00:01:03Z", elapsed=2.0),
        ],
    )

    markdown = report_mod.render_markdown(
        report_mod.build_report(
            tmp_path,
            telemetry_health_start="2026-06-20",
            decision_gate=2,
            min_arm_samples=1,
        )
    )

    assert "# RI-10 Canary Decision Report" in markdown
    assert "hold_quality_unscored" in markdown
    assert "| enforce |" in markdown
