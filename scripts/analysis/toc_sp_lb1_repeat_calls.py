#!/usr/bin/env python3
"""TOC-SP-LB1: rate of consecutive identical (tool name, args) calls, and what follows.

READ-ONLY over conversation stores; never writes to a store. Sources:

* ``--vault-claude DIR``  Claude Code jsonl (``tool_use`` / ``tool_result``), e.g.
  ``/mnt/raid0/llm/cloud-llm-vault/claude``;
* ``--vault-codex DIR``   Codex rollout jsonl (``function_call`` / ``function_call_output``);
* ``--opencode-db FILE``  opencode.db (``mode=ro``, ``PRAGMA query_only``), ``--providers`` filter,
  sessions and parts LIMITed; run under ``nice -n 19``.

Per session the tool-call sequence is read in store order. A call is a REPEAT when its
(name, canonical-args) equals the immediately preceding call's. For each repeat we record
whether the two results were byte-equal after volatile-line normalisation (``result_equal``),
and what the model did next: ``same`` (a further identical call), ``other_tool``, ``text``
(a text reply with no further call), or ``end`` (session ends). Only counts and tool NAMES leave
this script; no argument or output text is printed.

The decision this feeds (TOC-SP-LB1): wire a result-equality repeat notice, or model-voiced
steering with a one-reply tool ban. A notice only helps repeats whose result is equal; steering
is the fallback for repeats with a changed result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter

_VOLATILE = re.compile(r"^(Chunk ID|Wall time|Original token count):.*$", re.M)


def _h(s: object) -> str:
    if not isinstance(s, str):
        s = json.dumps(s, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(_VOLATILE.sub("", s).strip().encode("utf-8", "replace")).hexdigest()[:16]


def _args_key(a: object) -> str:
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return _h(a)
    return _h(json.dumps(a, sort_keys=True, ensure_ascii=False, default=str))


#: Polling tools whose identical consecutive calls are the protocol, not a stuck model.
POLLING_TOOLS = frozenset({"write_stdin", "wait", "sleep", "BashOutput", "TaskOutput"})


def analyse_sequence(events: list[tuple], exclude: frozenset = frozenset()) -> Counter:
    """events: ("call", name, args_key, out_hash|None) | ("text",). Returns counters.

    Calls whose tool name is in ``exclude`` are dropped before pairing (``POLLING_TOOLS``).
    """
    if exclude:
        events = [e for e in events if not (e[0] == "call" and e[1] in exclude)]
    c: Counter = Counter()
    calls = [(i, e) for i, e in enumerate(events) if e[0] == "call"]
    c["calls"] += len(calls)
    for n, (i, e) in enumerate(calls):
        if n == 0:
            continue
        pi, pe = calls[n - 1]
        if (pe[1], pe[2]) != (e[1], e[2]):
            continue
        c["repeats"] += 1
        c[f"repeat_tool:{e[1]}"] += 1
        if pe[3] is None or e[3] is None:
            c["result_unknown"] += 1
        elif pe[3] == e[3]:
            c["result_equal"] += 1
        else:
            c["result_changed"] += 1
        text_between = any(x[0] == "text" for x in events[pi + 1:i])
        c["text_between" if text_between else "back_to_back"] += 1
        nxt = calls[n + 1][1] if n + 1 < len(calls) else None
        after = events[i + 1:(calls[n + 1][0] if nxt else len(events))]
        if nxt is not None and (nxt[1], nxt[2]) == (e[1], e[2]):
            c["next_same"] += 1
        elif nxt is not None:
            c["next_other_tool"] += 1
        elif any(x[0] == "text" for x in after):
            c["next_text"] += 1
        else:
            c["next_end"] += 1
        if e[3] is not None and pe[3] == e[3]:
            c["equal_and_next_other_or_text"] += int(not (nxt is not None and (nxt[1], nxt[2]) == (e[1], e[2])))
    return c


def claude_events(path: str) -> list[tuple]:
    ev: list[tuple] = []
    idx: dict[str, int] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if '"tool_use"' not in line and '"tool_result"' not in line and '"role":"assistant"' not in line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            m = o.get("message") or {}
            content = m.get("content")
            if not isinstance(content, list):
                continue
            has_call = False
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    has_call = True
                    idx[b.get("id")] = len(ev)
                    ev.append(("call", b.get("name"), _args_key(b.get("input")), None))
                elif b.get("type") == "tool_result" and b.get("tool_use_id") in idx:
                    j = idx[b["tool_use_id"]]
                    ev[j] = ev[j][:3] + (_h(b.get("content")),)
            if m.get("role") == "assistant" and not has_call and any(
                    isinstance(b, dict) and b.get("type") == "text" and b.get("text") for b in content):
                ev.append(("text",))
    return ev


def codex_events(path: str) -> list[tuple]:
    ev: list[tuple] = []
    idx: dict[str, int] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if "function_call" not in line and '"role":"assistant"' not in line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            p = o.get("payload") or {}
            t = p.get("type")
            if t == "function_call":
                idx[p.get("call_id")] = len(ev)
                ev.append(("call", p.get("name"), _args_key(p.get("arguments")), None))
            elif t == "function_call_output" and p.get("call_id") in idx:
                j = idx[p["call_id"]]
                ev[j] = ev[j][:3] + (_h(p.get("output")),)
            elif t == "message" and p.get("role") == "assistant":
                ev.append(("text",))
    return ev


def opencode_sessions(db: str, providers: list[str], sessions: int, parts_per_session: int):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    con.execute("PRAGMA query_only=ON")
    for prov in providers:
        ids = [r[0] for r in con.execute(
            "SELECT id FROM session WHERE model LIKE ? ORDER BY time_updated DESC LIMIT ?",
            (f'%"providerID":"{prov}"%', sessions))]
        for sid in ids:
            ev: list[tuple] = []
            for (t, tool, inp, out, status) in con.execute(
                "SELECT json_extract(p.data,'$.type'), json_extract(p.data,'$.tool'), "
                "json_extract(p.data,'$.state.input'), json_extract(p.data,'$.state.output'), "
                "json_extract(p.data,'$.state.status') FROM part p JOIN message m ON m.id=p.message_id "
                "WHERE p.session_id=? AND json_extract(p.data,'$.type') IN ('tool','text') "
                "ORDER BY m.time_created, p.id LIMIT ?", (sid, parts_per_session)):
                if t == "tool":
                    ev.append(("call", tool, _args_key(inp), _h(out) if status == "completed" else None))
                elif t == "text":
                    ev.append(("text",))
            yield prov, ev


def walk(root: str, max_files: int):
    files = []
    for d, _, fs in os.walk(root):
        files += [os.path.join(d, f) for f in fs if f.endswith(".jsonl")]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[:max_files]


def report(c: Counter) -> dict:
    n, r = c["calls"], c["repeats"]
    out = {k: v for k, v in sorted(c.items()) if not k.startswith("repeat_tool:")}
    out["repeat_rate"] = round(r / n, 5) if n else None
    out["result_equal_share_of_repeats"] = round(c["result_equal"] / r, 4) if r else None
    out["next_same_share_of_repeats"] = round(c["next_same"] / r, 4) if r else None
    out["top_repeat_tools"] = dict(Counter({k[12:]: v for k, v in c.items() if k.startswith("repeat_tool:")}).most_common(8))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vault-claude")
    ap.add_argument("--vault-codex")
    ap.add_argument("--opencode-db")
    ap.add_argument("--providers", default="qwen-gpu,qwen-local,deepseek")
    ap.add_argument("--keep-polling", action="store_true", help="do not drop POLLING_TOOLS")
    ap.add_argument("--max-files", type=int, default=300)
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--parts-per-session", type=int, default=3000)
    a = ap.parse_args(argv)
    ex = frozenset() if a.keep_polling else POLLING_TOOLS
    res: dict = {}
    if a.vault_claude:
        c: Counter = Counter()
        fl = walk(a.vault_claude, a.max_files)
        for f in fl:
            c.update(analyse_sequence(claude_events(f), ex))
        res["vault_claude"] = {"files": len(fl), **report(c)}
    if a.vault_codex:
        c = Counter()
        fl = walk(a.vault_codex, a.max_files)
        for f in fl:
            c.update(analyse_sequence(codex_events(f), ex))
        res["vault_codex"] = {"files": len(fl), **report(c)}
    if a.opencode_db:
        by: dict[str, Counter] = {}
        ns: Counter = Counter()
        for prov, ev in opencode_sessions(a.opencode_db, a.providers.split(","), a.sessions, a.parts_per_session):
            by.setdefault(prov, Counter()).update(analyse_sequence(ev, ex))
            ns[prov] += 1
        res["opencode"] = {p: {"sessions": ns[p], **report(c)} for p, c in by.items()}
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
