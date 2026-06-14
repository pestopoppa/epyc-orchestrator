#!/usr/bin/env python3
"""Read-only state-vs-journal archive authority report."""

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

from preflight_audit import (  # noqa: E402
    JOURNAL_PATH,
    STATE_PATH,
    _archive_authority_view,
    _load_jsonl,
    archive_authority_diagnostic,
)
from src.autopilot_core.journal_reconstruction import (  # noqa: E402
    reconstruct_archive_from_journal_rows,
)


def _trial_id(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _trial_sort_key(value: Any) -> tuple[int, int | str]:
    parsed = _trial_id(value)
    if isinstance(parsed, int):
        return (0, parsed)
    return (1, parsed)


def _entry_map(entries: object) -> dict[int | str, dict[str, Any]]:
    if not isinstance(entries, list):
        return {}
    mapped: dict[int | str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "trial_id" not in entry:
            continue
        mapped[_trial_id(entry.get("trial_id"))] = entry
    return mapped


def _limited_ids(values: set[int | str], max_examples: int) -> list[int | str]:
    return sorted(values, key=_trial_sort_key)[:max_examples]


def _id_delta(
    state_entries: object,
    journal_entries: object,
    *,
    max_examples: int,
) -> dict[str, Any]:
    state_map = _entry_map(state_entries)
    journal_map = _entry_map(journal_entries)
    state_ids = set(state_map)
    journal_ids = set(journal_map)
    return {
        "state_only_count": len(state_ids - journal_ids),
        "state_only_examples": _limited_ids(state_ids - journal_ids, max_examples),
        "journal_only_count": len(journal_ids - state_ids),
        "journal_only_examples": _limited_ids(journal_ids - state_ids, max_examples),
    }


def _entry_mismatches(
    state_entries: object,
    journal_entries: object,
    *,
    max_examples: int,
) -> dict[str, Any]:
    state_map = _entry_map(state_entries)
    journal_map = _entry_map(journal_entries)
    mismatch_ids = [
        trial_id
        for trial_id in sorted(set(state_map) & set(journal_map), key=_trial_sort_key)
        if state_map[trial_id] != journal_map[trial_id]
    ]
    return {
        "count": len(mismatch_ids),
        "examples": [
            {
                "trial_id": trial_id,
                "state": state_map[trial_id],
                "journal": journal_map[trial_id],
            }
            for trial_id in mismatch_ids[:max_examples]
        ],
    }


def build_archive_authority_report(
    state: dict[str, Any],
    journal_rows: list[dict[str, Any]],
    *,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Build a structured no-inference report for state/journal archive drift."""
    diagnostic = archive_authority_diagnostic(state, journal_rows)
    state_archive = state.get("pareto_archive")
    if not isinstance(state_archive, dict):
        state_archive = {}
    journal_archive = reconstruct_archive_from_journal_rows(
        journal_rows,
        None,
        current_run_only=False,
    ) or {}

    state_view = _archive_authority_view(state_archive)
    journal_view = _archive_authority_view(journal_archive)
    entry_delta = _id_delta(
        state_view.get("all_entries", []),
        journal_view.get("all_entries", []),
        max_examples=max_examples,
    )
    frontier_delta = _id_delta(
        state_view.get("frontier", []),
        journal_view.get("frontier", []),
        max_examples=max_examples,
    )
    mismatches = _entry_mismatches(
        state_view.get("all_entries", []),
        journal_view.get("all_entries", []),
        max_examples=max_examples,
    )

    ok = diagnostic.get("status") == "match"
    return {
        "ok": ok,
        "diagnostic": diagnostic,
        "entry_id_delta": entry_delta,
        "frontier_id_delta": frontier_delta,
        "entry_mismatches": mismatches,
        "state_trial_counter": diagnostic.get("state_trial_counter"),
        "journal_max_trial_id": diagnostic.get("journal_max_trial_id"),
        "recommendation": (
            "archive authority is aligned"
            if ok
            else "repair or regenerate state archive from journal authority before restart"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    diagnostic = report["diagnostic"]
    entry_delta = report["entry_id_delta"]
    frontier_delta = report["frontier_id_delta"]
    mismatches = report["entry_mismatches"]
    lines = [
        "# AutoPilot Archive Authority Report",
        "",
        f"- Status: {diagnostic.get('status')}",
        f"- Recommendation: {report.get('recommendation')}",
        (
            "- State/journal trial bounds: "
            f"state_trial_counter={report.get('state_trial_counter')}, "
            f"journal_max_trial_id={report.get('journal_max_trial_id')}"
        ),
        (
            "- Entry counts: "
            f"state={diagnostic.get('state_entry_count', 'n/a')}, "
            f"journal={diagnostic.get('journal_entry_count', 'n/a')}"
        ),
        (
            "- Frontier counts: "
            f"state={diagnostic.get('state_frontier_count', 'n/a')}, "
            f"journal={diagnostic.get('journal_frontier_count', 'n/a')}"
        ),
        (
            "- Snapshot replay: "
            f"readiness={diagnostic.get('snapshot_readiness', 'n/a')}, "
            f"status={diagnostic.get('snapshot_replay_status', 'n/a')}"
        ),
        "",
        "## Trial ID Deltas",
        "",
        (
            f"- State-only entries: {entry_delta['state_only_count']} "
            f"{entry_delta['state_only_examples']}"
        ),
        (
            f"- Journal-only entries: {entry_delta['journal_only_count']} "
            f"{entry_delta['journal_only_examples']}"
        ),
        (
            f"- State-only frontier: {frontier_delta['state_only_count']} "
            f"{frontier_delta['state_only_examples']}"
        ),
        (
            f"- Journal-only frontier: {frontier_delta['journal_only_count']} "
            f"{frontier_delta['journal_only_examples']}"
        ),
        f"- Common-entry value mismatches: {mismatches['count']}",
    ]
    warnings = diagnostic.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    if mismatches["examples"]:
        lines.extend(["", "## Mismatch Examples", ""])
        for item in mismatches["examples"]:
            lines.append(f"- Trial {item['trial_id']}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report no-inference drift between autopilot_state.json archive "
            "authority and append-only journal reconstruction."
        )
    )
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--journal", type=Path, default=JOURNAL_PATH)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when archive authority is not aligned.",
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

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        print(f"state file is not a JSON object: {state_path}", file=sys.stderr)
        return 2
    journal_rows = _load_jsonl(journal_path)
    report = build_archive_authority_report(
        state,
        journal_rows,
        max_examples=max(0, args.max_examples),
    )
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(render_markdown(report))
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
