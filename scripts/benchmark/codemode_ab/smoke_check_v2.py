#!/usr/bin/env python3
"""Stage-0 smoke predicate for probe v2 (PREDECLARED_v2.md). Exit 0 iff every check passes; the problems are printed one per line.

    smoke_check_v2.py OUT_DIR [--mode smoke] [--min-episodes 8]

Checks (ABORT the run on any failure):
  1. every episode has a transcript file whose line count equals ``transcript_turns`` (and is >= 1);
  2. no B1/B2/C3 episode ends ``final`` without an explicit ``FINAL(`` in the model's own code (``final_kind == "explicit"``): proves the
     explicit-FINAL fix is live;
  3. the persisted FIRST-turn prompt of every B1/B2/C3 episode contains none of: "STOP generating", "shortestPalindrome",
     "run_python_code", "ALWAYS test" (proves the v2 prompt, not the v1 text, was served);
  4. the v2 REPL arms ran with the raised turn cap (``max_turns`` equals arm A's 40);
  5. no ``llm_error`` / harness-crash episode, and no episode of a v2 arm lacks the per-turn failure-reason enum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

V2_REPL = ("B1", "B2", "C3")
BANNED_IN_V2_PROMPT = ("STOP generating", "shortestPalindrome", "run_python_code", "ALWAYS test")
EXPECTED_V2_CAP = 40


def episodes(out_dir: Path, mode: str) -> list[dict[str, Any]]:
    res = out_dir / "results.jsonl"
    rows = []
    for ln in (res.read_text().splitlines() if res.exists() else []):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("kind") == "episode" and r.get("mode", "full") == mode:
            rows.append(r)
    return rows


def check(out_dir: str | Path, mode: str = "smoke", min_episodes: int = 1) -> list[str]:
    out = Path(out_dir)
    eps = episodes(out, mode)
    problems: list[str] = []
    if len(eps) < min_episodes:
        problems.append(f"only {len(eps)} {mode} episodes (need >= {min_episodes})")
    for r in eps:
        uid = r.get("unit_id", "?")
        end = str(r.get("end_reason", ""))
        if end.startswith("llm_error"):
            problems.append(f"{uid}: {end[:80]}")
        tp = r.get("transcript_path")
        lines: list[str] = []
        if not tp or not (out / tp).is_file():
            problems.append(f"{uid}: no transcript file")
        else:
            lines = [x for x in (out / tp).read_text().splitlines() if x.strip()]
            if not lines:
                problems.append(f"{uid}: empty transcript")
            elif r.get("transcript_turns") is not None and len(lines) != r["transcript_turns"]:
                problems.append(f"{uid}: transcript has {len(lines)} turns, record says {r['transcript_turns']}")
        if r.get("arm") in V2_REPL:
            if end == "final" and r.get("final_kind") != "explicit":
                problems.append(f"{uid}: ended final without an explicit FINAL (final_kind={r.get('final_kind')})")
            if r.get("max_turns") != EXPECTED_V2_CAP:
                problems.append(f"{uid}: turn cap {r.get('max_turns')} != {EXPECTED_V2_CAP}")
            if not r.get("turn_failure_reasons"):
                problems.append(f"{uid}: no per-turn failure-reason enum")
            if lines:
                first = json.loads(lines[0]).get("prompt", "")
                if not first:
                    problems.append(f"{uid}: first-turn prompt not persisted")
                for bad in BANNED_IN_V2_PROMPT:
                    if bad in first:
                        problems.append(f"{uid}: v2 prompt still contains {bad!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--min-episodes", type=int, default=1)
    a = ap.parse_args(argv)
    problems = check(a.out_dir, a.mode, a.min_episodes)
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
