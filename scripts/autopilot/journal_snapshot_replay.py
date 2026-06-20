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
    archive_payload_from_verified_snapshot,
    build_snapshot_replay_diagnostic,
    format_snapshot_replay_summary,
)


STRICT_READY_READINESS = "current"
TAIL_FOLD_READY_READINESS = "tail_fold_ready"


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
    parser.add_argument(
        "--allow-tail-fold",
        action="store_true",
        help=(
            "With --strict, also exit zero when the latest snapshot has a "
            "verified bounded tail fold suitable for ordinary restart."
        ),
    )
    return parser.parse_args()


def _strict_readiness(
    *,
    diagnostic_readiness: str,
    tail_fold_payload: dict | None,
    allow_tail_fold: bool,
) -> str:
    if diagnostic_readiness == STRICT_READY_READINESS:
        return STRICT_READY_READINESS
    if allow_tail_fold and tail_fold_payload is not None:
        return TAIL_FOLD_READY_READINESS
    return diagnostic_readiness


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
    tail_fold_payload = None
    if args.allow_tail_fold:
        tail_fold_payload = archive_payload_from_verified_snapshot(
            _journal_rows(journal),
            journal.ledger_events(),
        )
    strict_readiness = _strict_readiness(
        diagnostic_readiness=diagnostic.bounded_replay_readiness,
        tail_fold_payload=tail_fold_payload,
        allow_tail_fold=args.allow_tail_fold,
    )

    if args.json:
        payload = asdict(diagnostic)
        if args.allow_tail_fold:
            payload["strict_readiness"] = strict_readiness
            payload["strict_ready"] = strict_readiness in {
                STRICT_READY_READINESS,
                TAIL_FOLD_READY_READINESS,
            }
            payload["tail_fold_payload_available"] = tail_fold_payload is not None
            payload["tail_fold_payload_journal_max_trial_id"] = (
                tail_fold_payload.get("journal_max_trial_id")
                if isinstance(tail_fold_payload, dict)
                else None
            )
        print(json.dumps(payload, sort_keys=True, default=str))
    else:
        lines = format_snapshot_replay_summary(diagnostic)
        if args.allow_tail_fold:
            lines.extend(
                [
                    f"Journal snapshot strict readiness: {strict_readiness}",
                    "Journal snapshot tail-fold payload available: "
                    f"{str(tail_fold_payload is not None).lower()}",
                ]
            )
        print("\n".join(lines))

    if args.strict and strict_readiness not in {
        STRICT_READY_READINESS,
        TAIL_FOLD_READY_READINESS,
    }:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
