#!/usr/bin/env python3
"""Backfill the `assigned_role` column on legacy episodic memories (TR-2.4).

Adds a Trinity tri-role label to memories that pre-date the schema migration
in TR-2.2. The column is added by `EpisodicStore._init_db` if missing, so this
script just walks rows where `assigned_role IS NULL` and fills them with a
heuristic based on `action_type` + `action` substrings.

Heuristic (intentionally conservative — favours WORKER):
- Action mentions review / verify / validate / compliance / critique → VERIFIER
- Action mentions architect / decompose / plan / design / strategy → THINKER
- Otherwise → WORKER

Usage:
    python scripts/memory/backfill_assigned_role.py --db /path/to/episodic.db [--dry-run]

Idempotent: only updates rows where `assigned_role IS NULL`. Re-runs are no-ops.

Per `feedback_minimum_imports`: this script depends only on stdlib + the local
EpisodicStore module (for the column-migration trigger), no third-party imports.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.classifiers.role_taxonomy import TrinityRole  # noqa: E402

# Substring rules. Order matters — VERIFIER wins over THINKER if both match,
# because review semantics dominate (a "review architect output" call is a V,
# not a T).
VERIFIER_TOKENS = (
    "review",
    "verify",
    "validation",
    "validate",
    "compliance",
    "critique",
    "judge",
    "qa",
)
THINKER_TOKENS = (
    "architect",
    "decompose",
    "decomposition",
    "plan",
    "planner",
    "design",
    "strategy",
    "synthes",  # synthesis / synthesize
    "ingest_long_context",  # frontdoor's long-context summarizer is plan-class
)


def classify_role(action: str | None, action_type: str | None) -> str:
    """Return one of TrinityRole.* for a legacy memory row."""
    text = " ".join(filter(None, [action, action_type])).lower()
    for tok in VERIFIER_TOKENS:
        if tok in text:
            return TrinityRole.VERIFIER.value
    for tok in THINKER_TOKENS:
        if tok in text:
            return TrinityRole.THINKER.value
    return TrinityRole.WORKER.value


def backfill(db_path: Path, dry_run: bool) -> dict[str, int]:
    """Backfill assigned_role on rows where it is NULL.

    Returns a counts dict with keys per Trinity role + total scanned + total
    updated.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Check that the column exists (migration may not have run yet).
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if "assigned_role" not in cols:
            raise RuntimeError(
                "assigned_role column missing — initialise EpisodicStore once "
                "first to trigger the schema migration in _init_db()."
            )

        rows = conn.execute(
            "SELECT id, action, action_type FROM memories "
            "WHERE assigned_role IS NULL"
        ).fetchall()

        counts = {
            "scanned": len(rows),
            "thinker": 0,
            "worker": 0,
            "verifier": 0,
            "updated": 0,
        }

        if not rows:
            return counts

        updates: list[tuple[str, str]] = []
        for memory_id, action, action_type in rows:
            role = classify_role(action, action_type)
            counts[role] += 1
            updates.append((role, memory_id))

        if dry_run:
            return counts

        conn.executemany(
            "UPDATE memories SET assigned_role = ? WHERE id = ? AND assigned_role IS NULL",
            updates,
        )
        conn.commit()
        counts["updated"] = len(updates)
        return counts
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to the episodic SQLite database.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing.",
    )
    args = parser.parse_args()

    try:
        counts = backfill(args.db, args.dry_run)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    mode = "[dry-run] would update" if args.dry_run else "updated"
    print(f"Scanned: {counts['scanned']} rows with NULL assigned_role")
    print(f"  THINKER:  {counts['thinker']}")
    print(f"  WORKER:   {counts['worker']}")
    print(f"  VERIFIER: {counts['verifier']}")
    print(f"{mode}: {counts['updated' if not args.dry_run else 'scanned']} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
