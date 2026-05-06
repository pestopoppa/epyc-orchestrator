"""CLI for the unified trace store.

Subcommands:
  ingest    — ingest from configured sources into events.sqlite (idempotent)
  query     — filtered query
  stats     — summary counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.trace import (
    DEFAULT_DB_PATH,
    ensure_schema,
    upsert_events,
)
from src.trace import ingest_agent_audit, ingest_autopilot, ingest_progress
from src.trace.query import query, stats


def _cmd_ingest(args: argparse.Namespace) -> int:
    db_path = Path(args.db or DEFAULT_DB_PATH)
    conn = ensure_schema(db_path)

    total_inserted = 0
    total_skipped = 0
    sources = []

    if not args.no_agent_audit:
        events = ingest_agent_audit.parse_file(args.agent_audit or ingest_agent_audit.DEFAULT_LOG_PATH)
        ins, skp = upsert_events(conn, events)
        total_inserted += ins
        total_skipped += skp
        sources.append(("agent_audit", len(events), ins, skp))

    if not args.no_progress:
        events = ingest_progress.walk_progress_root(args.progress_root or ingest_progress.DEFAULT_PROGRESS_ROOT)
        ins, skp = upsert_events(conn, events)
        total_inserted += ins
        total_skipped += skp
        sources.append(("progress", len(events), ins, skp))

    if not args.no_autopilot:
        events = ingest_autopilot.parse_all(
            args.autopilot_tsv or ingest_autopilot.DEFAULT_JOURNAL_TSV,
            args.autopilot_jsonl or ingest_autopilot.DEFAULT_JOURNAL_JSONL,
            args.autopilot_state or ingest_autopilot.DEFAULT_STATE_JSON,
        )
        ins, skp = upsert_events(conn, events)
        total_inserted += ins
        total_skipped += skp
        sources.append(("autopilot", len(events), ins, skp))

    conn.close()

    print(f"db: {db_path}")
    for name, parsed, ins, skp in sources:
        print(f"  {name:<14}  parsed={parsed:5}  inserted={ins:5}  skipped(dup)={skp:5}")
    print(f"total inserted={total_inserted}  skipped={total_skipped}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    rows = query(
        db_path=args.db or DEFAULT_DB_PATH,
        from_ts=args.from_ts,
        to_ts=args.to_ts,
        session_id=args.session,
        trial_id=args.trial,
        role=args.role,
        category=args.category,
        status=args.status,
        source=args.source,
        text=args.text,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for r in rows:
            ts = r["ts_utc"]
            cat = r["category"] or "?"
            src = r["source"]
            sid = (r["session_id"] or "-")[:18]
            tid = r["trial_id"] if r["trial_id"] is not None else "-"
            summary = (r["summary"] or "")[:90]
            print(f"{ts}  {src:<18}  {cat:<18}  ses={sid:<18}  trial={tid}  {summary}")
        print(f"\n{len(rows)} rows")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    s = stats(args.db or DEFAULT_DB_PATH)
    print(json.dumps(s, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trace", description="Unified trace / memory service CLI")
    p.add_argument("--db", help=f"sqlite path (default: {DEFAULT_DB_PATH})")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="ingest from configured sources")
    pi.add_argument("--no-agent-audit", action="store_true")
    pi.add_argument("--no-progress", action="store_true")
    pi.add_argument("--no-autopilot", action="store_true")
    pi.add_argument("--agent-audit", help="override agent_audit.log path")
    pi.add_argument("--progress-root", help="override progress/ root")
    pi.add_argument("--autopilot-tsv", help="override autopilot_journal.tsv path")
    pi.add_argument("--autopilot-jsonl", help="override autopilot_journal.jsonl path")
    pi.add_argument("--autopilot-state", help="override autopilot_state.json path")
    pi.set_defaults(func=_cmd_ingest)

    pq = sub.add_parser("query", help="query the store")
    pq.add_argument("--from-ts", dest="from_ts")
    pq.add_argument("--to-ts", dest="to_ts")
    pq.add_argument("--session")
    pq.add_argument("--trial", type=int)
    pq.add_argument("--role")
    pq.add_argument("--category")
    pq.add_argument("--status")
    pq.add_argument("--source")
    pq.add_argument("--text", help="FTS5 query against summary + detail_json")
    pq.add_argument("--limit", type=int, default=50)
    pq.add_argument("--json", action="store_true")
    pq.set_defaults(func=_cmd_query)

    ps = sub.add_parser("stats", help="summary counts")
    ps.set_defaults(func=_cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
