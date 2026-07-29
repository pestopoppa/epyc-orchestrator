#!/usr/bin/env python3
"""Report non-cumulative outcome rates against cumulative episodic-store size.

This is an offline, read-only instrument.  It deliberately reports each
chronological window separately: a cumulative success rate can continue to
rise while later windows deteriorate as the memory store grows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable


def summarize_rows(rows: Iterable[tuple[str, str | None]], window_size: int) -> dict:
    """Build a per-window success-rate curve from chronological outcome rows."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    materialized = list(rows)
    windows = []
    for start in range(0, len(materialized), window_size):
        chunk = materialized[start:start + window_size]
        outcomes = [outcome for _, outcome in chunk if outcome in {"success", "failure"}]
        successes = sum(outcome == "success" for outcome in outcomes)
        windows.append({
            "window_index": len(windows) + 1,
            "start_row": start + 1,
            "end_row": start + len(chunk),
            "cumulative_store_size": start + len(chunk),
            "records": len(chunk),
            "scored_records": len(outcomes),
            "unknown_outcomes": len(chunk) - len(outcomes),
            "successes": successes,
            "success_rate": (successes / len(outcomes)) if outcomes else None,
        })
    return {
        "schema_version": "memory_store_growth_curve.v1",
        "total_records": len(materialized),
        "window_size": window_size,
        "windows": windows,
    }


def load_rows(db_path: Path) -> list[tuple[str, str | None]]:
    """Load chronological rows without mutating a potentially live SQLite store."""
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return conn.execute(
            "SELECT created_at, outcome FROM memories ORDER BY created_at ASC, id ASC"
        ).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Path to episodic.db")
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = summarize_rows(load_rows(args.db), args.window_size)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
