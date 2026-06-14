#!/usr/bin/env python3
"""Read-only baseline promotion ledger authority report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from preflight_audit import JOURNAL_PATH, STATE_PATH, _load_jsonl  # noqa: E402
from src.autopilot_core.baseline_ledger import (  # noqa: E402
    BaselineLedgerReconciliation,
    format_baseline_ledger_summary,
    reconcile_baseline_ledger,
)


def _latest_event_view(
    reconciliation: BaselineLedgerReconciliation,
) -> dict[str, Any]:
    event = reconciliation.latest_event
    if not isinstance(event, dict):
        return {}
    return {
        "latest_source_trial_id": event.get("source_trial_id"),
        "latest_tier": event.get("tier"),
        "latest_previous_quality": event.get("previous_quality"),
        "latest_new_quality": event.get("new_quality"),
    }


def _recommendation(reconciliation: BaselineLedgerReconciliation) -> str:
    if reconciliation.cutover_ready:
        return "baseline ledger fold is ready for evidence-plane W4 acceptance"
    if reconciliation.status == "no_events":
        return "keep live baseline_state authority; no baseline promotion ledger exists"
    if reconciliation.status == "unreconstructable":
        return "keep live baseline_state authority; promotion ledger lacks usable snapshots"
    return "keep live baseline_state authority until ledger fold blockers are resolved"


def build_baseline_authority_report(
    state: dict[str, Any],
    journal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a structured no-write report for baseline ledger fold readiness."""
    state_baseline = state.get("baseline_state")
    reconciliation = reconcile_baseline_ledger(
        journal_rows,
        state_baseline if isinstance(state_baseline, dict) else None,
    )
    report: dict[str, Any] = {
        "ok": reconciliation.cutover_ready,
        "status": reconciliation.status,
        "event_count": reconciliation.event_count,
        "valid_snapshot_count": reconciliation.valid_snapshot_count,
        "cutover_ready": reconciliation.cutover_ready,
        "cutover_blockers": reconciliation.cutover_blockers,
        "warnings": reconciliation.warnings,
        "recommendation": _recommendation(reconciliation),
    }
    report.update(_latest_event_view(reconciliation))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    reconciliation = BaselineLedgerReconciliation(
        status=str(report.get("status", "unknown")),
        event_count=int(report.get("event_count") or 0),
        valid_snapshot_count=int(report.get("valid_snapshot_count") or 0),
        cutover_ready=bool(report.get("cutover_ready")),
        cutover_blockers=list(report.get("cutover_blockers") or []),
        latest_event={
            "source_trial_id": report.get("latest_source_trial_id", "n/a"),
            "tier": report.get("latest_tier", "n/a"),
            "previous_quality": report.get("latest_previous_quality"),
            "new_quality": report.get("latest_new_quality"),
        },
        warnings=list(report.get("warnings") or []),
    )
    lines = [
        "# AutoPilot Baseline Authority Report",
        "",
        *[f"- {line}" for line in format_baseline_ledger_summary(reconciliation)],
        f"- Recommendation: {report.get('recommendation')}",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report no-inference baseline_state vs append-only baseline "
            "promotion ledger fold readiness."
        )
    )
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--journal", type=Path, default=JOURNAL_PATH)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when baseline ledger fold is not cutover-ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    state_path = args.state.expanduser().resolve()
    journal_path = args.journal.expanduser().resolve()
    if not state_path.exists():
        print(f"state file does not exist: {state_path}", file=sys.stderr)
        return 2
    if not journal_path.exists():
        print(f"journal file does not exist: {journal_path}", file=sys.stderr)
        return 2

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"state file is not valid JSON: {state_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(state, dict):
        print(f"state file is not a JSON object: {state_path}", file=sys.stderr)
        return 2

    journal_rows = _load_jsonl(journal_path)
    report = build_baseline_authority_report(state, journal_rows)
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(render_markdown(report))
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
