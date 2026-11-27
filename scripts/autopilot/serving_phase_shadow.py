#!/usr/bin/env python3
"""Plan a serving investigation from saved receipts and append a shadow ledger event."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiment_journal import DEFAULT_JOURNAL_DIR, ExperimentJournal  # noqa: E402
from src.autopilot_core.serving_phase_evidence import (  # noqa: E402
    phase_report,
    plan_serving_phase,
    serving_log_shards,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="saved serving-call JSONL")
    parser.add_argument("--batch-id", action="append", help="select saved receipt batches")
    parser.add_argument("--journal-dir", type=Path, required=True, help="local shadow journal")
    args = parser.parse_args(argv)
    paths = serving_log_shards(args.log)
    if not paths:
        parser.error(f"no log at {args.log}")
    journal_dir = args.journal_dir.resolve()
    if journal_dir == DEFAULT_JOURNAL_DIR.resolve():
        parser.error("choose a separate shadow journal directory")
    report = phase_report(paths, batch_ids=set(args.batch_id) if args.batch_id else None)
    event = plan_serving_phase(report)
    journal = ExperimentJournal(journal_dir, segment_snapshots=False)
    print(json.dumps(journal.append_ledger_event(event), indent=1, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
