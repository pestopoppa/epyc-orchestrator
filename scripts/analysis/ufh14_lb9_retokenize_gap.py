#!/usr/bin/env python3
"""UFH14-LB9: size non-canonical re-tokenization from logged reply text.

For each assistant message of a local-provider opencode session whose reply is TEXT ONLY
(no reasoning part, no tool part, ``finish == "stop"``), compare the provider-reported
``tokens.output`` with the length of the canonical re-tokenization of the reply text under
the model's tokenizer. A reply the model generated as a NON-canonical token sequence (e.g.
a word split into pieces the BPE merge order would have merged) re-tokenizes SHORTER than it
was generated; the follow-on turn then re-sends the canonical form and diverges inside the
previous output (one of the two divergence classes behind 17.8% of follow-on turns).

READ-ONLY and bounded: the operator's read-only snapshots (``/mnt/raid0/llm/opencode-db/{epyc,host-daniele}.db``,
live stores refused) are opened ``mode=ro``; sessions and rows are LIMITed; run it
under ``nice -n 19``. The tokenizer is the HF ``tokenizer.json`` shipped next to the exl3 build
of the same model (the GGUF carries the same vocabulary; an HF-vs-GGUF pre-tokenizer difference
would show as a constant offset on every row, not as a spread).

Not measured here: tool-call turns (the chat template renders arguments in its own XML shape),
and reasoning turns (reasoning text is not in the reply). Both are counted and reported as
excluded, never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opencode_ro  # noqa: E402

DEFAULT_DBS = opencode_ro.OPENCODE_RO_DBS     # default: both read-only snapshots
DEFAULT_TOK = "/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw/tokenizer.json"


def summarise(rows: list[tuple[int, int]]) -> dict:
    """rows: (reported_output_tokens, retokenized_len). diff = reported - retok."""
    diffs = [r - t for r, t in rows]
    n = len(diffs)
    if not n:
        return {"n": 0}
    c = Counter(diffs)
    mode, mode_n = c.most_common(1)[0]
    # excess over the modal offset: the modal offset absorbs EOS / template-prefix tokens,
    # so a non-canonical re-tokenization shows as a diff ABOVE the mode.
    above = sum(v for d, v in c.items() if d > mode)
    below = sum(v for d, v in c.items() if d < mode)
    return {
        "n": n,
        "modal_offset": mode,
        "modal_share": round(mode_n / n, 4),
        "rows_above_mode": above,
        "rows_below_mode": below,
        "share_above_mode": round(above / n, 4),
        "mean_excess_above_mode_tokens": round(
            sum((d - mode) * v for d, v in c.items() if d > mode) / above, 3) if above else 0.0,
        "median_diff": statistics.median(diffs),
        "diff_histogram": dict(sorted(c.items())[:40]),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", action="append",
                    help="opencode READ-ONLY snapshot (repeatable). Default: both snapshots "
                         f"{', '.join(DEFAULT_DBS.values())}. Live stores are refused.")
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--model-id", default="qwen3.8-27b")
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--per-session", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=0)
    args = ap.parse_args(argv)
    dbs = opencode_ro.resolve_dbs(args.db)      # refuses live stores before any heavy work

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    rows: list[tuple[int, int]] = []
    excluded = Counter()
    n_sessions: Counter = Counter()
    for src_tag, path in dbs:
        con = opencode_ro.open_ro(path)
        sess = [r[0] for r in con.execute(
            "SELECT id FROM session WHERE model LIKE ? ORDER BY time_updated DESC LIMIT ?",
            (f'%"id":"{args.model_id}"%', args.sessions))]
        for sid in sess:
            msgs = con.execute(
                "SELECT id, json_extract(data,'$.tokens.output'), json_extract(data,'$.finish') "
                "FROM message WHERE session_id=? AND json_extract(data,'$.role')='assistant' "
                "ORDER BY time_created LIMIT ?", (sid, args.per_session)).fetchall()
            for mid, out, finish in msgs:
                parts = [json.loads(p[0]) for p in con.execute(
                    "SELECT data FROM part WHERE message_id=? ORDER BY id", (mid,))]
                kinds = {p.get("type") for p in parts}
                if "tool" in kinds:
                    excluded["tool_call_turn"] += 1
                elif "reasoning" in kinds:
                    excluded["reasoning_turn"] += 1
                elif finish != "stop":
                    excluded[f"finish_{finish}"] += 1
                elif not isinstance(out, int) or out <= 0:
                    excluded["no_output_tokens"] += 1
                else:
                    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                    if len(text) < args.min_chars or not text:
                        excluded["empty_or_short_text"] += 1
                    else:
                        rows.append((out, len(tok.encode(text, add_special_tokens=False).ids)))
        n_sessions[src_tag] += len(sess)
        con.close()
    print(json.dumps({"sessions": sum(n_sessions.values()), "sessions_by_source": dict(n_sessions), "model_id": args.model_id, "text_only_stop_turns": summarise(rows),
                      "excluded": dict(excluded)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
