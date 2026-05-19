#!/usr/bin/env python3
"""Backfill the `sub_decision` column on legacy episodic memories (intake-548).

Adds an orchestration sub-decision label to memories that pre-date the schema
migration. The column is added by `EpisodicStore._init_db` if missing, so this
script walks rows where `sub_decision IS NULL` and applies a conservative
heuristic based on `action` + `action_type` substrings.

The polarity is the OPPOSITE of `backfill_assigned_role.py`: a NULL row is the
expected state for most events ("this event is not a sub-decision"), so the
heuristic deliberately leaves rows unlabelled rather than guessing. We only
write a label when the row's action / action_type strongly implies one of the
five sub-decisions.

Heuristic (specific evidence → label, fallthrough → NULL stays NULL):
- `delegate` / `subagent_spawn` / `child_kani_spawn` substrings → DELEGATE
  (covers the moment a parent decides who to delegate to + what)
- `spawn` / `escalation` action_type with no `delegate` substring → SPAWN
  (covers the moment a parent decides to involve a child at all)
- `aggregate` / `merge_results` / `consolidate` / `wait_complete` / `child_return`
  → AGGREGATE
- `terminate` / `final` / `stop_iteration` / `repl_done` → STOP
- `tool_response` / `inter_agent_msg` / `delegate_message` / `communicate`
  → COMMUNICATE
- Otherwise → leave NULL

Usage:
    python scripts/memory/backfill_sub_decision.py --db /path/to/episodic.db [--dry-run]

Idempotent: only writes to rows where `sub_decision IS NULL`. Re-runs are no-ops.

Per `feedback_minimum_imports`: stdlib + local module only, no third-party.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.classifiers.subdecision_taxonomy import OrchestrationSubDecision  # noqa: E402

# Substring rules. Order matters — first match wins, and DELEGATE wins over
# SPAWN because "delegate" implies both the spawn decision AND the routing
# decision (the survey treats them as separate, but in practice an EPYC
# action like "delegate:worker_general" encodes both).
DELEGATE_TOKENS = (
    "delegate",
    "subagent_spawn",
    "child_kani_spawn",
    "redel_delegate",
)
SPAWN_TOKENS = (
    "spawn",
)
AGGREGATE_TOKENS = (
    "aggregate",
    "merge_results",
    "consolidate",
    "wait_complete",
    "child_return",
    "subagent_return",
)
STOP_TOKENS = (
    "terminate",
    "stop_iteration",
    "repl_done",
    "final_answer",
    "round_complete",
)
COMMUNICATE_TOKENS = (
    "tool_response",
    "inter_agent_msg",
    "delegate_message",
    "communicate",
)

# The "escalation" action_type was added by an earlier handoff to capture
# parent-asking-architect events. It maps to SPAWN unless the action string
# also names a specific child (which would push it to DELEGATE above).
ESCALATION_ACTION_TYPE = "escalation"


def classify_sub_decision(
    action: str | None,
    action_type: str | None,
) -> str | None:
    """Return one of OrchestrationSubDecision.* or None.

    None means "this event is not a labelled sub-decision". Do NOT change this
    to a sentinel string — the column's polarity treats NULL as absence, not
    as a default.
    """
    text = " ".join(filter(None, [action, action_type])).lower()
    if not text:
        return None
    for tok in DELEGATE_TOKENS:
        if tok in text:
            return OrchestrationSubDecision.DELEGATE.value
    for tok in AGGREGATE_TOKENS:
        if tok in text:
            return OrchestrationSubDecision.AGGREGATE.value
    for tok in STOP_TOKENS:
        if tok in text:
            return OrchestrationSubDecision.STOP.value
    for tok in COMMUNICATE_TOKENS:
        if tok in text:
            return OrchestrationSubDecision.COMMUNICATE.value
    for tok in SPAWN_TOKENS:
        if tok in text:
            return OrchestrationSubDecision.SPAWN.value
    if action_type and action_type.strip().lower() == ESCALATION_ACTION_TYPE:
        return OrchestrationSubDecision.SPAWN.value
    return None


def backfill(db_path: Path, dry_run: bool) -> dict[str, int]:
    """Backfill sub_decision on rows where it is NULL.

    Returns a counts dict with one key per sub-decision plus "scanned",
    "labelled" (rows we'll write to), "skipped" (rows that stay NULL), and
    "updated" (rows actually written to disk; equals "labelled" unless
    dry_run).
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if "sub_decision" not in cols:
            raise RuntimeError(
                "sub_decision column missing — initialise EpisodicStore once "
                "first to trigger the schema migration in _init_db()."
            )

        rows = conn.execute(
            "SELECT id, action, action_type FROM memories "
            "WHERE sub_decision IS NULL"
        ).fetchall()

        counts = {
            "scanned": len(rows),
            "spawn": 0,
            "delegate": 0,
            "communicate": 0,
            "aggregate": 0,
            "stop": 0,
            "skipped": 0,
            "labelled": 0,
            "updated": 0,
        }

        if not rows:
            return counts

        updates: list[tuple[str, str]] = []
        for memory_id, action, action_type in rows:
            label = classify_sub_decision(action, action_type)
            if label is None:
                counts["skipped"] += 1
                continue
            counts[label] += 1
            counts["labelled"] += 1
            updates.append((label, memory_id))

        if dry_run or not updates:
            return counts

        conn.executemany(
            "UPDATE memories SET sub_decision = ? WHERE id = ? AND sub_decision IS NULL",
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
    print(f"Scanned: {counts['scanned']} rows with NULL sub_decision")
    print(f"  SPAWN:       {counts['spawn']}")
    print(f"  DELEGATE:    {counts['delegate']}")
    print(f"  COMMUNICATE: {counts['communicate']}")
    print(f"  AGGREGATE:   {counts['aggregate']}")
    print(f"  STOP:        {counts['stop']}")
    print(f"  (skipped, left NULL): {counts['skipped']}")
    written = counts["updated"] if not args.dry_run else counts["labelled"]
    print(f"{mode}: {written} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
