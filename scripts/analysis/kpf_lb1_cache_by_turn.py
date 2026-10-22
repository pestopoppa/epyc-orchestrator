#!/usr/bin/env python3
"""KPF-LB1: tabulate cache_n / prompt_n by role x endpoint x turn index.

Two read-only sources, each labelled with exactly what it can and cannot key on:

* ``opencode.db`` (live store, ``mode=ro``, LIMITed; run under ``nice -n 19``): per assistant
  step of a local-provider session, ``tokens.input`` (= prompt_n, tokens evaluated this step) and
  ``tokens.cache.read`` (= cache_n). Turn index = the step's ordinal within its session. The
  endpoint is /v1/chat/completions (opencode's local providers speak only that).
* ``logs/serving_calls/serving_calls.jsonl``: orchestrator ``timings`` per call, with
  ``role`` and ``notes.endpoint`` (so /completion vs chat). It carries NO session or turn key
  (``caller`` has request_id only), so it cannot be bucketed by turn: it is tabulated by role x
  endpoint, with n shown, and the turn column is reported as unavailable.

The /completion x turn cell is therefore NOT measurable from retained logs.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict

DB = "/home/node/.local/share/opencode/opencode.db"
BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 10**9)]


def bucket(turn: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= turn <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
    return "?"


def cell(rows: list[tuple[int, int]]) -> dict:
    """rows: (prompt_n, cache_n). Token-weighted hit = sum(cache)/(sum(cache)+sum(prompt))."""
    p = [r[0] for r in rows]
    c = [r[1] for r in rows]
    tot = sum(p) + sum(c)
    return {
        "n": len(rows),
        "median_prompt_n": statistics.median(p),
        "median_cache_n": statistics.median(c),
        "token_weighted_hit": round(sum(c) / tot, 4) if tot else None,
        "share_cache_below_10pct": round(sum(1 for a, b in rows if a + b and b / (a + b) < 0.10) / len(rows), 4),
    }


def opencode_table(db: str, providers: list[str], sessions: int, per_session: int) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    con.execute("PRAGMA query_only=ON")
    out: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for prov in providers:
        sess = [r[0] for r in con.execute(
            "SELECT id FROM session WHERE model LIKE ? ORDER BY time_updated DESC LIMIT ?",
            (f'%"providerID":"{prov}"%', sessions))]
        for sid in sess:
            turn = 0
            for inp, cr in con.execute(
                "SELECT json_extract(data,'$.tokens.input'), json_extract(data,'$.tokens.cache.read') "
                "FROM message WHERE session_id=? AND json_extract(data,'$.role')='assistant' "
                "ORDER BY time_created LIMIT ?", (sid, per_session)):
                if not isinstance(inp, int) or not isinstance(cr, int):
                    continue
                turn += 1
                out[prov][bucket(turn)].append((inp, cr))
    return {prov: {b: cell(rows) for b, rows in sorted(by.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+")))}
            for prov, by in out.items()}


def serving_table(path: str) -> dict:
    by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        t = r.get("timings") or {}
        if "prompt_n" in t and "cache_n" in t:
            by[f"{r.get('role')} | {(r.get('notes') or {}).get('endpoint')} | port {r.get('server', {}).get('port')}"].append(
                (t["prompt_n"], t["cache_n"]))
    return {k: cell(v) for k, v in sorted(by.items())}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--serving-calls")
    ap.add_argument("--providers", default="qwen-gpu,qwen-local")
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--per-session", type=int, default=400)
    a = ap.parse_args(argv)
    res = {"opencode_chat_by_turn": opencode_table(a.db, a.providers.split(","), a.sessions, a.per_session)}
    if a.serving_calls:
        res["serving_calls_by_role_endpoint_no_turn_key"] = serving_table(a.serving_calls)
    print(json.dumps(res, indent=1, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
