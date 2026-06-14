"""Read-only baseline promotion ledger reconciliation.

The live safety gate still reads ``autopilot_state.json:baseline_state``.
This module only folds append-only ``baseline_promotion`` events for diagnostics
so operators can see whether ledger evidence matches current state before any
future baseline-as-fold cutover.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any


BASELINE_PROMOTION_EVENT_TYPE = "baseline_promotion"


@dataclass(frozen=True)
class BaselineLedgerReconciliation:
    """Structured result for comparing promotion-ledger state to live state."""

    status: str
    event_count: int = 0
    valid_snapshot_count: int = 0
    cutover_ready: bool = False
    cutover_blockers: list[str] = field(default_factory=list)
    latest_event: dict[str, Any] | None = None
    folded_state: dict[str, Any] | None = None
    state_baseline: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


def canonical_jsonable(value: Any) -> Any:
    """Normalize JSON-like payloads for stable read-only diagnostics."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return value


def _promotion_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("type") == BASELINE_PROMOTION_EVENT_TYPE
        and "trial_id" not in event
    ]


def _event_quality_warning(event: dict[str, Any], state: dict[str, Any]) -> str | None:
    try:
        tier_key = str(int(event["tier"]))
    except (KeyError, TypeError, ValueError):
        return None
    tier_values = state.get("baselines_by_tier")
    if not isinstance(tier_values, dict) or tier_key not in tier_values:
        return None
    try:
        new_quality = float(event["new_quality"])
        folded_quality = float(tier_values[tier_key])
    except (KeyError, TypeError, ValueError):
        return None
    if abs(new_quality - folded_quality) > 1e-9:
        return (
            f"event new_quality {new_quality:.3f} differs from "
            f"baseline_state.baselines_by_tier[{tier_key}] {folded_quality:.3f}"
        )
    return None


def reconcile_baseline_ledger(
    events: list[dict[str, Any]],
    state_baseline: dict[str, Any] | None,
) -> BaselineLedgerReconciliation:
    """Fold baseline promotion events by append order and compare to state.

    Latest valid ``baseline_state`` snapshot wins. Missing or malformed event
    state is not inferred from event metrics, YAML, Pareto archive, or current
    state; it only contributes a warning.
    """
    promotion_events = _promotion_events(events)
    if not promotion_events:
        return BaselineLedgerReconciliation(
            status="no_events",
            cutover_blockers=[
                "no baseline promotion events; YAML remains cold-start seed"
            ],
        )

    folded_state: dict[str, Any] | None = None
    latest_valid_event: dict[str, Any] | None = None
    valid_snapshot_count = 0
    warnings: list[str] = []
    for index, event in enumerate(promotion_events):
        snapshot = event.get("baseline_state")
        if not isinstance(snapshot, dict):
            warnings.append(f"event {index} has no usable baseline_state snapshot")
            continue
        folded_state = canonical_jsonable(copy.deepcopy(snapshot))
        latest_valid_event = copy.deepcopy(event)
        valid_snapshot_count += 1

    if folded_state is None:
        return BaselineLedgerReconciliation(
            status="unreconstructable",
            event_count=len(promotion_events),
            valid_snapshot_count=valid_snapshot_count,
            cutover_blockers=[
                "no promotion event has a usable baseline_state snapshot"
            ],
            warnings=warnings,
        )

    warning = _event_quality_warning(latest_valid_event or {}, folded_state)
    if warning:
        warnings.append(warning)

    canonical_state = (
        canonical_jsonable(copy.deepcopy(state_baseline))
        if isinstance(state_baseline, dict) and state_baseline
        else None
    )
    if canonical_state is None:
        status = "missing_state_baseline"
    elif canonical_state == folded_state:
        status = "match"
    else:
        status = "drift"

    cutover_blockers: list[str] = []
    missing_snapshot_count = len(promotion_events) - valid_snapshot_count
    if missing_snapshot_count:
        cutover_blockers.append(
            f"{missing_snapshot_count} promotion event(s) lack usable "
            "baseline_state snapshots"
        )
    if status != "match":
        cutover_blockers.append(
            f"ledger fold does not match current state baseline ({status})"
        )
    if warnings:
        cutover_blockers.append("baseline promotion ledger has warning diagnostics")

    return BaselineLedgerReconciliation(
        status=status,
        event_count=len(promotion_events),
        valid_snapshot_count=valid_snapshot_count,
        cutover_ready=not cutover_blockers,
        cutover_blockers=cutover_blockers,
        latest_event=latest_valid_event,
        folded_state=folded_state,
        state_baseline=canonical_state,
        warnings=warnings,
    )


def _format_optional_metric(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def format_baseline_ledger_summary(
    reconciliation: BaselineLedgerReconciliation,
) -> list[str]:
    """Human-readable status/report lines for baseline ledger diagnostics."""
    lines = [f"Baseline promotion events: {reconciliation.event_count}"]
    if reconciliation.status == "no_events":
        lines.append("Baseline ledger state: no promotion events")
    elif reconciliation.status == "unreconstructable":
        lines.append("Baseline ledger state: unreconstructable")
    else:
        event = reconciliation.latest_event or {}
        lines.append(
            "Latest baseline event: "
            f"trial #{event.get('source_trial_id', 'n/a')} "
            f"T{event.get('tier', 'n/a')} "
            f"{_format_optional_metric(event.get('previous_quality'))} -> "
            f"{_format_optional_metric(event.get('new_quality'))} "
            f"at {event.get('timestamp', 'n/a')}"
        )
        lines.append(f"Baseline ledger state status: {reconciliation.status}")
    lines.append(
        "Baseline fold cutover dry-run: "
        f"{'ready' if reconciliation.cutover_ready else 'not_ready'}"
    )
    for blocker in reconciliation.cutover_blockers:
        lines.append(f"Baseline fold blocker: {blocker}")
    for warning in reconciliation.warnings:
        lines.append(f"Baseline ledger warning: {warning}")
    return lines
