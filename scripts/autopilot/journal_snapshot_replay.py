#!/usr/bin/env python3
"""Read-only journal snapshot replay diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from experiment_journal import DEFAULT_JOURNAL_DIR, ExperimentJournal  # noqa: E402
from src.autopilot_core.journal_snapshot_replay import (  # noqa: E402
    build_snapshot_replay_diagnostic,
    format_snapshot_replay_summary,
)


STRICT_READY_STATUSES = {"archive_prefix_match"}


def _journal_rows(journal: ExperimentJournal) -> list[dict]:
    rows = [asdict(entry) for entry in journal.all_entries()]
    rows.extend(journal.ledger_events())
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report read-only journal snapshot replay readiness."
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help=f"Journal directory (default: {DEFAULT_JOURNAL_DIR})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured diagnostic as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless the latest snapshot archive matches prefix replay.",
    )
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
    diagnostic = build_snapshot_replay_diagnostic(
        _journal_rows(journal),
        journal.ledger_events(),
    )

    if args.json:
        print(json.dumps(asdict(diagnostic), sort_keys=True, default=str))
    else:
        print("\n".join(format_snapshot_replay_summary(diagnostic)))

    if args.strict and diagnostic.status not in STRICT_READY_STATUSES:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
