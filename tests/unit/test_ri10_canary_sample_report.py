"""Tests for RI-10 canary sample coverage reporting."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis import ri10_canary_sample_report as report_mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _routing_row(
    timestamp: str,
    *,
    band: str,
    action: str = "",
    mode: str = "",
    routing: list[str] | None = None,
) -> dict:
    return {
        "event_type": "routing_decision",
        "timestamp": timestamp,
        "data": {
            "routing": routing or ["frontdoor"],
            "factual_risk_score": 0.72 if band == "high" else 0.2,
            "factual_risk_band": band,
            "risk_gate_action": action,
            "factual_risk_mode": mode,
            "decision_source": "rules",
        },
    }


def test_build_report_counts_high_risk_and_canary_arms(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "2026-04-07.jsonl",
        [
            _routing_row("2026-04-07T00:00:00Z", band="high", action="enforce"),
            _routing_row("2026-04-07T00:01:00Z", band="high", action="shadow"),
            _routing_row("2026-04-07T00:02:00Z", band="low", action="shadow"),
        ],
    )
    _write_jsonl(
        tmp_path / "2026-04-05.jsonl",
        [_routing_row("2026-04-05T00:00:00Z", band="high", action="enforce")],
    )

    summary = report_mod.build_report(
        tmp_path,
        canary_start="2026-04-06",
        decision_gate=2,
        min_arm_samples=1,
    )

    assert summary["high_risk_rows_total"] == 3
    assert summary["high_risk_rows_since_canary_start"] == 2
    assert summary["frontdoor_high_risk_rows_since_canary_start"] == 2
    assert summary["sample_count_ready"] is True
    assert summary["canary_arm_sample_count_ready"] is True
    assert summary["canary_arm_balance_ready"] is True
    assert summary["canary_decision_ready"] is True
    assert summary["canary_arm_counts_since_canary_start"] == {
        "enforce_high_risk": 1,
        "shadow_high_risk": 1,
    }


def test_build_report_requires_observable_canary_arms(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "2026-04-07.jsonl",
        [
            _routing_row(
                "2026-04-07T00:00:00Z",
                band="high",
                action="not_enforced",
                routing=["frontdoor"],
            ),
            _routing_row(
                "2026-04-07T00:01:00Z",
                band="high",
                action="not_enforced",
                routing=["frontdoor"],
            ),
        ],
    )

    summary = report_mod.build_report(tmp_path, canary_start="2026-04-06", decision_gate=2)

    assert summary["sample_count_ready"] is True
    assert summary["canary_decision_ready"] is False
    assert "observable enforce/shadow canary arms" in summary["decision_reason"]


def test_build_report_requires_decision_grade_arm_counts(tmp_path: Path) -> None:
    rows = []
    rows.extend(
        _routing_row(
            f"2026-04-07T00:{idx:02d}:00Z",
            band="high",
            mode="shadow",
            routing=["frontdoor"],
        )
        for idx in range(18)
    )
    rows.append(
        _routing_row(
            "2026-04-07T01:00:00Z",
            band="high",
            mode="enforce",
            routing=["frontdoor"],
        )
    )
    rows.extend(
        _routing_row(
            f"2026-04-07T02:{idx:02d}:00Z",
            band="high",
            action="not_enforced",
            routing=["frontdoor"],
        )
        for idx in range(40)
    )
    _write_jsonl(tmp_path / "2026-04-07.jsonl", rows)

    summary = report_mod.build_report(
        tmp_path,
        canary_start="2026-04-06",
        decision_gate=50,
        min_arm_samples=10,
    )

    assert summary["high_risk_rows_since_canary_start"] == 59
    assert summary["evaluable_canary_arm_high_risk_rows"] == 19
    assert summary["non_evaluable_high_risk_rows_since_canary_start"] == 40
    assert summary["sample_count_ready"] is True
    assert summary["canary_arm_sample_count_ready"] is False
    assert summary["canary_arm_balance_ready"] is False
    assert summary["canary_decision_ready"] is False
