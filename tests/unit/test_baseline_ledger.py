"""Tests for read-only baseline promotion ledger reconciliation."""

from __future__ import annotations

from src.autopilot_core.baseline_ledger import (
    BASELINE_PROMOTION_EVENT_TYPE,
    format_baseline_ledger_summary,
    reconcile_baseline_ledger,
)


_DEFAULT_STATE = object()


def _event(
    source_trial_id: int,
    *,
    tier: int = 1,
    new_quality: float = 1.8,
    baseline_state: object = _DEFAULT_STATE,
    timestamp: str = "2026-06-14T00:00:00+00:00",
) -> dict:
    return {
        "type": BASELINE_PROMOTION_EVENT_TYPE,
        "source_trial_id": source_trial_id,
        "tier": tier,
        "previous_quality": 1.5,
        "new_quality": new_quality,
        "reason": "accepted",
        "proof": {},
        "result_metrics": {"quality": new_quality},
        "baseline_state": (
            {"baselines_by_tier": {str(tier): new_quality}}
            if baseline_state is _DEFAULT_STATE
            else baseline_state
        ),
        "timestamp": timestamp,
    }


def test_reconcile_reports_no_events() -> None:
    result = reconcile_baseline_ledger([], {"baselines_by_tier": {"1": 1.8}})

    assert result.status == "no_events"
    assert not result.cutover_ready
    assert result.cutover_blockers == [
        "no baseline promotion events; YAML remains cold-start seed"
    ]
    assert format_baseline_ledger_summary(result) == [
        "Baseline promotion events: 0",
        "Baseline ledger state: no promotion events",
        "Baseline fold cutover dry-run: not_ready",
        "Baseline fold blocker: no baseline promotion events; YAML remains cold-start seed",
    ]


def test_reconcile_uses_append_order_not_timestamp_order() -> None:
    first = _event(1, new_quality=1.8, timestamp="2026-06-14T00:00:10+00:00")
    second = _event(2, new_quality=1.9, timestamp="2026-06-14T00:00:01+00:00")

    result = reconcile_baseline_ledger(
        [first, second],
        {"baselines_by_tier": {"1": 1.9}},
    )

    assert result.status == "match"
    assert result.cutover_ready
    assert result.cutover_blockers == []
    assert result.latest_event["source_trial_id"] == 2
    assert result.folded_state == {"baselines_by_tier": {"1": 1.9}}


def test_reconcile_canonicalizes_tier_keys() -> None:
    result = reconcile_baseline_ledger(
        [_event(1, baseline_state={"baselines_by_tier": {"1": 1.8}})],
        {"baselines_by_tier": {1: 1.8}},
    )

    assert result.status == "match"


def test_reconcile_reports_drift() -> None:
    result = reconcile_baseline_ledger(
        [_event(1, new_quality=1.8)],
        {"baselines_by_tier": {"1": 1.7}},
    )

    assert result.status == "drift"
    assert not result.cutover_ready
    assert result.cutover_blockers == [
        "ledger fold does not match current state baseline (drift)"
    ]


def test_reconcile_does_not_infer_missing_snapshot() -> None:
    result = reconcile_baseline_ledger(
        [_event(1, new_quality=1.8, baseline_state=None)],
        {"baselines_by_tier": {"1": 1.8}},
    )

    assert result.status == "unreconstructable"
    assert not result.cutover_ready
    assert result.cutover_blockers == [
        "no promotion event has a usable baseline_state snapshot"
    ]
    assert "event 0 has no usable baseline_state snapshot" in result.warnings


def test_reconcile_blocks_cutover_when_any_promotion_event_lacks_snapshot() -> None:
    result = reconcile_baseline_ledger(
        [
            _event(1, new_quality=1.8, baseline_state=None),
            _event(2, new_quality=1.9),
        ],
        {"baselines_by_tier": {"1": 1.9}},
    )

    assert result.status == "match"
    assert not result.cutover_ready
    assert "1 promotion event(s) lack usable baseline_state snapshots" in (
        result.cutover_blockers
    )


def test_reconcile_quality_mismatch_is_warning_only() -> None:
    result = reconcile_baseline_ledger(
        [_event(1, new_quality=1.8, baseline_state={"baselines_by_tier": {"1": 1.7}})],
        {"baselines_by_tier": {"1": 1.7}},
    )

    assert result.status == "match"
    assert not result.cutover_ready
    assert result.cutover_blockers == [
        "baseline promotion ledger has warning diagnostics"
    ]
    assert result.warnings == [
        "event new_quality 1.800 differs from baseline_state.baselines_by_tier[1] 1.700"
    ]
