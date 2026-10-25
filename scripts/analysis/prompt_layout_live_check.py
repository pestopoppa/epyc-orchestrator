#!/usr/bin/env python3
"""RTE-LOG-1 live-record check: do the first REAL records carry the layout fields?

Run it on ``dispatch.jsonl`` / ``serving_calls.jsonl`` (files or directories) after the stack
has served at least one request with the layout log on.  It passes only if, per record
kind, at least ``--min-records`` records exist and EVERY record that has ``blocks`` also
carries non-empty ``blocks[]`` entries (kind, start, length, sha256_12), ``layout_chars``,
``prompt_tokens`` and ``cached_prompt_tokens`` (null counts as missing), and it reports:

* ``coverage``: records with blocks / records seen, per record kind and per source label;
* ``repl_blocks``: records whose role path is the REPL turn (a ``tool_output`` or ``state`` kind
  present) with the kinds seen, and how many are ``rebased`` with ``tail_chars``;
* ``unlabelled_source``: records with ``source`` null (set ``ORCHESTRATOR_REQUEST_SOURCE`` in the
  autopilot / benchmark / experiment launchers);
* ``problems``: each failing record by timestamp, with the missing field names. No text exists
  to leak: records hold hashes and lengths only.

Exit 0 pass, 1 fail, 2 no usable records (zero records is NOT a pass).
Usage: prompt_layout_live_check.py [--min-records N] [--json] FILE_OR_DIR...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_layout_reader import iter_records  # noqa: E402

BLOCK_KEYS = ("kind", "start", "length", "sha256_12")
REQUIRED_KINDS = ("dispatch", "serving_call")


def _kind(rec: dict) -> str | None:
    k = str(rec.get("kind") or rec.get("schema") or "")
    if "dispatch" in k:
        return "dispatch"
    if "serving" in k:
        return "serving_call"
    return None


def check(paths: list[Path], min_records: int = 1) -> dict:
    seen: Counter = Counter()
    with_blocks: Counter = Counter()
    sources: Counter = Counter()
    unlabelled = 0
    repl = Counter()
    problems: list[dict] = []
    for rec in iter_records(paths):
        k = _kind(rec)
        if k is None:
            continue
        seen[k] += 1
        blocks = rec.get("blocks")
        if not blocks:
            continue
        with_blocks[k] += 1
        miss = []
        if not all(all(key in b for key in BLOCK_KEYS) for b in blocks):
            miss.append("block_fields")
        for f in ("layout_chars", "prompt_tokens", "cached_prompt_tokens"):
            if rec.get(f) is None and (rec.get("result") or {}).get(f) is None:
                miss.append(f)
        if miss:
            problems.append(
                {
                    "record": k,
                    "ts": rec.get("ts_arrival") or rec.get("ts_start") or rec.get("ts"),
                    "missing": miss,
                }
            )
        src = rec.get("source")
        sources[src or "null"] += 1
        unlabelled += src is None
        kinds = {b.get("kind") for b in blocks}
        if kinds & {"tool_output", "state"}:
            repl["records"] += 1
            repl["rebased"] += rec.get("rebased") is True
            repl["with_tool_output"] += "tool_output" in kinds
    total_blocks = sum(with_blocks.values())
    shortfalls = {
        k: min_records - with_blocks[k] for k in REQUIRED_KINDS if with_blocks[k] < min_records
    }
    ok = total_blocks > 0 and not shortfalls and not problems
    return {
        "verdict": "pass" if ok else ("no_records" if total_blocks == 0 else "fail"),
        "records_seen": dict(seen),
        "records_with_blocks": dict(with_blocks),
        "record_kind_shortfalls": shortfalls,
        "sources": dict(sources),
        "unlabelled_source": unlabelled,
        "repl_blocks": dict(repl),
        "problems": problems[:50],
        "n_problems": len(problems),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--min-records", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = check([Path(p) for p in a.paths], a.min_records)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return {"pass": 0, "fail": 1}.get(rep["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
