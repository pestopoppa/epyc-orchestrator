#!/usr/bin/env python3
"""Build or append archive snapshots for the AutoPilot journal."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from experiment_journal import DEFAULT_JOURNAL_DIR, ExperimentJournal  # noqa: E402
from src.autopilot_core.journal_reconstruction import (  # noqa: E402
    reconstruct_archive_from_journal_rows,
)


DEFAULT_POLICY_VERSION = "journal-archive-snapshot-v1"


@dataclass(frozen=True)
class JournalSnapshotBuildResult:
    """Prepared snapshot payload and append metadata."""

    status: str
    through_trial_id: int | None = None
    policy_version: str = DEFAULT_POLICY_VERSION
    parent_snapshot_hash: str = ""
    trial_count: int = 0
    ledger_event_count: int = 0
    snapshot: dict[str, Any] | None = None
    warning: str = ""


def _trial_rows(journal: ExperimentJournal) -> list[dict[str, Any]]:
    return [asdict(entry) for entry in journal.all_entries()]


def _journal_rows(journal: ExperimentJournal) -> list[dict[str, Any]]:
    rows = _trial_rows(journal)
    rows.extend(journal.ledger_events())
    return rows


def _max_trial_id(rows: list[dict[str, Any]]) -> int | None:
    max_id: int | None = None
    for row in rows:
        try:
            trial_id = int(row.get("trial_id"))
        except (TypeError, ValueError):
            continue
        if max_id is None or trial_id > max_id:
            max_id = trial_id
    return max_id


def _latest_snapshot_through_trial_id(journal: ExperimentJournal) -> int | None:
    event = journal.latest_journal_snapshot_event()
    if event is None:
        return None
    try:
        return int(event.get("through_trial_id"))
    except (TypeError, ValueError):
        return None


def build_archive_snapshot(
    journal: ExperimentJournal,
    *,
    policy_version: str = DEFAULT_POLICY_VERSION,
    force: bool = False,
) -> JournalSnapshotBuildResult:
    """Build the current full-journal archive snapshot without writing it."""
    trial_rows = _trial_rows(journal)
    through_trial_id = _max_trial_id(trial_rows)
    if through_trial_id is None:
        return JournalSnapshotBuildResult(
            status="no_trials",
            policy_version=policy_version,
            ledger_event_count=len(journal.ledger_events()),
            warning="journal has no trial rows to snapshot",
        )

    latest_through = _latest_snapshot_through_trial_id(journal)
    if latest_through is not None and latest_through >= through_trial_id and not force:
        latest_event = journal.latest_journal_snapshot_event() or {}
        return JournalSnapshotBuildResult(
            status="up_to_date",
            through_trial_id=through_trial_id,
            policy_version=policy_version,
            parent_snapshot_hash=str(latest_event.get("snapshot_hash") or ""),
            trial_count=len(trial_rows),
            ledger_event_count=len(journal.ledger_events()),
            warning=(
                "latest snapshot already covers the current journal; use --force "
                "to append another snapshot"
            ),
        )

    rows = _journal_rows(journal)
    archive = reconstruct_archive_from_journal_rows(
        rows,
        None,
        current_run_only=False,
    )
    if archive is None:
        return JournalSnapshotBuildResult(
            status="empty_archive",
            through_trial_id=through_trial_id,
            policy_version=policy_version,
            trial_count=len(trial_rows),
            ledger_event_count=len(journal.ledger_events()),
            warning="journal replay produced no archive payload",
        )

    latest_event = journal.latest_journal_snapshot_event() or {}
    snapshot = {
        "archive": archive,
        "source": {
            "kind": "full_journal_replay",
            "trial_count": len(trial_rows),
            "ledger_event_count": len(journal.ledger_events()),
        },
    }
    return JournalSnapshotBuildResult(
        status="ready",
        through_trial_id=through_trial_id,
        policy_version=policy_version,
        parent_snapshot_hash=str(latest_event.get("snapshot_hash") or ""),
        trial_count=len(trial_rows),
        ledger_event_count=len(journal.ledger_events()),
        snapshot=snapshot,
    )


def append_archive_snapshot(
    journal: ExperimentJournal,
    result: JournalSnapshotBuildResult,
    *,
    actor: str = "journal_snapshot_create.py",
) -> dict[str, Any]:
    """Append a prepared snapshot event. Caller must provide a ready result."""
    if result.status != "ready" or result.through_trial_id is None or result.snapshot is None:
        raise ValueError(f"snapshot result is not appendable: {result.status}")
    return journal.append_journal_snapshot_event(
        through_trial_id=result.through_trial_id,
        snapshot=result.snapshot,
        policy_version=result.policy_version,
        actor=actor,
        parent_snapshot_hash=result.parent_snapshot_hash,
    )


def _summary_lines(result: JournalSnapshotBuildResult, event: dict[str, Any] | None) -> list[str]:
    lines = [
        f"Journal snapshot build status: {result.status}",
        f"Through trial: {result.through_trial_id if result.through_trial_id is not None else 'n/a'}",
        f"Policy version: {result.policy_version}",
        f"Trial rows: {result.trial_count}",
        f"Ledger events: {result.ledger_event_count}",
        f"Parent snapshot hash: {result.parent_snapshot_hash[:12] or 'n/a'}",
    ]
    if result.warning:
        lines.append(f"Warning: {result.warning}")
    if event is not None:
        lines.append(f"Appended snapshot hash: {str(event.get('snapshot_hash') or '')[:12]}")
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the current journal archive snapshot. Dry-run by default; "
            "use --append to write a journal_snapshot event."
        )
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help=f"Journal directory (default: {DEFAULT_JOURNAL_DIR})",
    )
    parser.add_argument(
        "--policy-version",
        default=DEFAULT_POLICY_VERSION,
        help=f"Snapshot policy version (default: {DEFAULT_POLICY_VERSION})",
    )
    parser.add_argument("--append", action="store_true", help="Append the snapshot event.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow appending a duplicate/up-to-date snapshot.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    journal_dir = args.journal_dir.expanduser().resolve()
    if not journal_dir.exists():
        print(f"journal directory does not exist: {journal_dir}", file=sys.stderr)
        return 2
    if not journal_dir.is_dir():
        print(f"journal path is not a directory: {journal_dir}", file=sys.stderr)
        return 2

    journal = ExperimentJournal(journal_dir=journal_dir)
    result = build_archive_snapshot(
        journal,
        policy_version=args.policy_version,
        force=args.force,
    )
    event: dict[str, Any] | None = None
    if args.append:
        if result.status != "ready":
            if args.json:
                print(json.dumps({"result": asdict(result), "event": None}, default=str))
            else:
                print("\n".join(_summary_lines(result, None)))
            return 1
        event = append_archive_snapshot(journal, result)

    if args.json:
        print(json.dumps({"result": asdict(result), "event": event}, default=str))
    else:
        print("\n".join(_summary_lines(result, event)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
