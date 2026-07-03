#!/usr/bin/env python3
"""Read-only Pareto stepping-stone report.

Reconstructs the Pareto archive from the append-only journal and surfaces
dominated-but-near archive rows that may be useful planner hypotheses. This is
observe-only: it does not modify planner prompts, state, journal rows, safety
gates, or production-best selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from experiment_journal import DEFAULT_JOURNAL_DIR, ExperimentJournal  # noqa: E402
from pareto_archive import pareto_archive_from_journal_rows  # noqa: E402
from src.autopilot_core.tier_specs import DEFAULT_FRONTIER_TIER  # noqa: E402


def build_stepping_stones_report_from_rows(
    rows: list[dict[str, Any]],
    *,
    tier: int = DEFAULT_FRONTIER_TIER,
    limit: int = 8,
) -> dict[str, Any]:
    """Build a structured report from already-loaded journal rows."""
    archive = pareto_archive_from_journal_rows(rows, current_run_only=False)
    if archive is None:
        return {
            "ok": False,
            "tier": tier,
            "limit": limit,
            "trial_count": len(rows),
            "stepping_stones": [],
            "text": "(no archive rows available)",
            "note": "journal replay produced no archive payload",
        }
    stones = archive.stepping_stones(tier=tier, limit=limit)
    return {
        "ok": True,
        "tier": tier,
        "limit": limit,
        "trial_count": len(rows),
        "frontier_size": archive.frontier_size(tier=tier),
        "stepping_stones": stones,
        "text": archive.stepping_stones_text(tier=tier, limit=limit),
        "note": (
            "observe-only: not replay authorization and not a Pareto, baseline, "
            "or safety-gate input"
        ),
    }


def build_stepping_stones_report(
    *,
    journal_dir: Path = DEFAULT_JOURNAL_DIR,
    tier: int = DEFAULT_FRONTIER_TIER,
    limit: int = 8,
) -> dict[str, Any]:
    """Build a structured report from the AutoPilot journal directory."""
    journal = ExperimentJournal(journal_dir=journal_dir)
    rows = [asdict(entry) for entry in journal.all_entries()]
    return build_stepping_stones_report_from_rows(rows, tier=tier, limit=limit)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report dominated-but-near Pareto archive stepping stones."
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help=f"Journal directory (default: {DEFAULT_JOURNAL_DIR})",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=DEFAULT_FRONTIER_TIER,
        help=f"Eval tier to inspect (default: {DEFAULT_FRONTIER_TIER})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum stepping-stone rows to render.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured report as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_stepping_stones_report(
        journal_dir=args.journal_dir,
        tier=args.tier,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(report["text"])
        print()
        print(report["note"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
