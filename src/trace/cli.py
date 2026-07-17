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
from src.trace.navigation import (
    get_conversation,
    get_records,
    read_file,
    search_conversation,
    search_records,
)
from src.trace.query import decision_chain, query, stats, trial_context


def _cmd_ingest(args: argparse.Namespace) -> int:
    db_path = Path(args.db or DEFAULT_DB_PATH)
    conn = ensure_schema(db_path)

    total_inserted = 0
    total_skipped = 0
    sources = []

    if not args.no_agent_audit:
        events = ingest_agent_audit.parse_file(
            args.agent_audit or ingest_agent_audit.DEFAULT_LOG_PATH
        )
        ins, skp = upsert_events(conn, events)
        total_inserted += ins
        total_skipped += skp
        sources.append(("agent_audit", len(events), ins, skp))

    if not args.no_progress:
        events = ingest_progress.walk_progress_root(
            args.progress_root or ingest_progress.DEFAULT_PROGRESS_ROOT
        )
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
            print(_format_row(r))
        print(f"\n{len(rows)} rows")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    s = stats(args.db or DEFAULT_DB_PATH)
    print(json.dumps(s, indent=2, default=str))
    return 0


def _format_row(row: dict) -> str:
    ts = row["ts_utc"]
    cat = row["category"] or "?"
    src = row["source"]
    sid = (row["session_id"] or "-")[:18]
    tid = row["trial_id"] if row["trial_id"] is not None else "-"
    summary = (row["summary"] or "")[:90]
    return f"{ts}  {src:<18}  {cat:<18}  ses={sid:<18}  trial={tid}  {summary}"


def _cmd_trial_context(args: argparse.Namespace) -> int:
    ctx = trial_context(
        db_path=args.db or DEFAULT_DB_PATH,
        trial_id=args.trial,
        window_minutes=args.window_minutes,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(ctx, indent=2, default=str))
        return 0

    print(
        f"trial={ctx['trial_id']} window={ctx['window_minutes']}m "
        f"range={ctx['from_ts'] or '-'}..{ctx['to_ts'] or '-'}"
    )
    print(f"counts={json.dumps(ctx['counts'], sort_keys=True)}")
    print("\nTimeline:")
    for row in ctx["timeline"]:
        print(_format_row(row))
    print(f"\n{len(ctx['timeline'])} rows")
    return 0


def _cmd_decision_chain(args: argparse.Namespace) -> int:
    chain = decision_chain(
        db_path=args.db or DEFAULT_DB_PATH,
        session_id=args.session,
        trial_id=args.trial,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(chain, indent=2, default=str))
        return 0
    print(
        f"session={chain['session_id'] or '-'} trial={chain['trial_id'] if chain['trial_id'] is not None else '-'} "
        f"counts={json.dumps(chain['counts'], sort_keys=True)}"
    )
    print("\nDecision chain:")
    for row in chain["chain"]:
        print(_format_row(row))
    print(f"\n{len(chain['chain'])} rows")
    return 0


def _cmd_search_records(args: argparse.Namespace) -> int:
    rows = search_records(
        args.text,
        db_path=args.db or DEFAULT_DB_PATH,
        from_ts=args.from_ts,
        to_ts=args.to_ts,
        session_id=args.session,
        trial_id=args.trial,
        role=args.role,
        category=args.category,
        status=args.status,
        source=args.source,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for row in rows:
            print(_format_row(row))
        print(f"\n{len(rows)} rows")
    return 0


def _cmd_search_conversation(args: argparse.Namespace) -> int:
    rows = search_conversation(
        args.text,
        db_path=args.db or DEFAULT_DB_PATH,
        session_id=args.session,
        trial_id=args.trial,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for row in rows:
            print(_format_row(row))
        print(f"\n{len(rows)} rows")
    return 0


def _cmd_get_records(args: argparse.Namespace) -> int:
    rows = get_records(args.event_ids, db_path=args.db or DEFAULT_DB_PATH)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for row in rows:
            print(_format_row(row))
        print(f"\n{len(rows)} rows")
    return 0


def _cmd_get_conversation(args: argparse.Namespace) -> int:
    conversation = get_conversation(
        db_path=args.db or DEFAULT_DB_PATH,
        session_id=args.session,
        trial_id=args.trial,
        window_minutes=args.window_minutes,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(conversation, indent=2, default=str))
        return 0
    rows = conversation.get("timeline", [])
    print(
        f"session={conversation.get('session_id') or '-'} "
        f"trial={conversation.get('trial_id') or '-'} "
        f"counts={json.dumps(conversation.get('counts', {}), sort_keys=True)}"
    )
    for row in rows:
        print(_format_row(row))
    print(f"\n{len(rows)} rows")
    return 0


def _cmd_read_file(args: argparse.Namespace) -> int:
    payload = read_file(
        args.path,
        allowed_roots=args.allowed_root,
        max_bytes=args.max_bytes,
    )
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(payload["content"], end="" if payload["content"].endswith("\n") else "\n")
        if payload["truncated"]:
            print(f"\n[truncated at {payload['bytes_read']} bytes]", file=sys.stderr)
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

    pt = sub.add_parser("trial-context", help="timeline around one autopilot trial")
    pt.add_argument("--trial", type=int, required=True)
    pt.add_argument("--window-minutes", type=int, default=60)
    pt.add_argument("--limit", type=int, default=200)
    pt.add_argument("--json", action="store_true")
    pt.set_defaults(func=_cmd_trial_context)

    pdc = sub.add_parser(
        "decision-chain",
        help="replay a review-plane decision chain (task->plan->review->gate->outcome)",
    )
    pdc.add_argument("--session")
    pdc.add_argument("--trial", type=int)
    pdc.add_argument("--limit", type=int, default=1000)
    pdc.add_argument("--json", action="store_true")
    pdc.set_defaults(func=_cmd_decision_chain)

    psr = sub.add_parser("search-records", help="NapMem read tool: FTS search records")
    psr.add_argument("--text", required=True, help="FTS5 query against summary + detail_json")
    psr.add_argument("--from-ts", dest="from_ts")
    psr.add_argument("--to-ts", dest="to_ts")
    psr.add_argument("--session")
    psr.add_argument("--trial", type=int)
    psr.add_argument("--role")
    psr.add_argument("--category")
    psr.add_argument("--status")
    psr.add_argument("--source")
    psr.add_argument("--limit", type=int, default=20)
    psr.add_argument("--json", action="store_true")
    psr.set_defaults(func=_cmd_search_records)

    psc = sub.add_parser(
        "search-conversation",
        help="NapMem read tool: search within one session or trial",
    )
    psc.add_argument("--text", required=True, help="FTS5 query against summary + detail_json")
    psc.add_argument("--session")
    psc.add_argument("--trial", type=int)
    psc.add_argument("--limit", type=int, default=20)
    psc.add_argument("--json", action="store_true")
    psc.set_defaults(func=_cmd_search_conversation)

    pgr = sub.add_parser("get-records", help="NapMem read tool: fetch exact event ids")
    pgr.add_argument("event_ids", nargs="+", type=int)
    pgr.add_argument("--json", action="store_true")
    pgr.set_defaults(func=_cmd_get_records)

    pgc = sub.add_parser(
        "get-conversation",
        help="NapMem read tool: session timeline or trial-centered context",
    )
    pgc.add_argument("--session")
    pgc.add_argument("--trial", type=int)
    pgc.add_argument("--window-minutes", type=int, default=60)
    pgc.add_argument("--limit", type=int, default=200)
    pgc.add_argument("--json", action="store_true")
    pgc.set_defaults(func=_cmd_get_conversation)

    prf = sub.add_parser("read-file", help="NapMem read tool: allowlisted file read")
    prf.add_argument("path")
    prf.add_argument("--allowed-root", action="append")
    prf.add_argument("--max-bytes", type=int, default=64_000)
    prf.add_argument("--json", action="store_true")
    prf.set_defaults(func=_cmd_read_file)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
