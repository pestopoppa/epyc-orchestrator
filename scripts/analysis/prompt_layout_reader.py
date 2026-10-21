#!/usr/bin/env python3
"""RTE-LOG-1 offline reader: how much of the prompt traffic is repeated, and where.

Reads ``dispatch.v1`` and/or ``serving_call.v1`` JSONL records that carry the
``blocks[]`` field written by ``src/backends/prompt_layout.py`` (hashes and lengths
only; this reader never sees prompt text) and reports, per (kind, role, source):

* ``first``      -- the block's bytes were not seen before for that role;
* ``prefix_repeat`` -- seen before with the SAME preceding blocks (identical
  prefix): the server's prefix cache already serves this ("not ours");
* ``moved_repeat``  -- seen before at a DIFFERENT offset and after a different
  predecessor chain: what a chunk-reuse path could add;
* ``pred_repeat``   -- seen before at the same offset but after a different
  predecessor chain (a prefix-cache miss caused by something earlier changing).

Shares are by occurrences and by characters.  Per request it also reports the
distinct-chunk count and chunk length (``reference_code`` / ``doc_chunk`` chunks) and
two block-granular prefix shares against the best earlier prompt of the same role
(window ``--window``): ``plain_prefix_share`` (blocks in common from the start, what
a prefix cache gives) and ``reconstructible_share`` (the longest prefix of that
earlier prompt whose blocks all occur, unused, in the incoming prompt: the KPF-61
reorder-planner idea at block granularity).  Both are fractions of the request's
typed-block characters, so untyped gap text (blank separators) is outside the
denominator.  Records are processed in timestamp order; ``--role`` / ``--source``
filter.  Records without ``blocks`` are counted, not guessed at.

Usage: prompt_layout_reader.py [--format json|md] [--window N] FILE_OR_DIR...
Report format: ``docs/reference/prompt-layout-report-format.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

REPORT_SCHEMA = "epyc.orchestrator.prompt_layout_report.v1"
CHUNK_KINDS = ("reference_code", "doc_chunk")
CLASSES = ("first", "prefix_repeat", "moved_repeat", "pred_repeat")


def iter_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for p in paths:
        files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        yield rec


def _ts(rec: dict[str, Any]) -> str:
    return str(rec.get("ts_arrival") or rec.get("ts_start") or rec.get("ts") or "")


def _units(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered occurrence units: chunk entries replace their enclosing block."""
    units: list[dict[str, Any]] = []
    for b in sorted(rec.get("blocks") or [], key=lambda b: b.get("start", 0)):
        chunks = b.get("chunks") if b.get("kind") in CHUNK_KINDS else None
        if chunks:
            for c in chunks:
                start = c.get("start")
                units.append({
                    "kind": b["kind"], "hash": c["sha256_12"], "start": b["start"] if start is None else start,
                    "length": c.get("length") or 0, "rank": c.get("rank"), "is_chunk": True,
                })
        else:
            units.append({
                "kind": b["kind"], "hash": b["sha256_12"], "start": b.get("start", 0),
                "length": b.get("length") or 0, "rank": None, "is_chunk": False,
            })
    return units


def _prefix_id(keys: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:12]


def _prefix_shares(cur: list[dict[str, Any]], prior: list[list[dict[str, Any]]]) -> tuple[float, float]:
    total = sum(u["length"] for u in cur)
    if not total or not prior:
        return 0.0, 0.0
    best_plain = best_rec = 0
    cur_pool: dict[tuple[str, str], int] = defaultdict(int)
    for u in cur:
        cur_pool[(u["kind"], u["hash"])] += 1
    for old in prior:
        plain = 0
        for a, b in zip(cur, old):
            if (a["kind"], a["hash"]) != (b["kind"], b["hash"]):
                break
            plain += a["length"]
        pool = dict(cur_pool)
        rec = 0
        for b in old:
            k = (b["kind"], b["hash"])
            if pool.get(k, 0) <= 0:
                break
            pool[k] -= 1
            rec += b["length"]
        best_plain, best_rec = max(best_plain, plain), max(best_rec, rec)
    return best_plain / total, best_rec / total


def analyse(
    records: Iterable[dict[str, Any]], *, window: int = 50,
    role: str | None = None, source: str | None = None,
) -> dict[str, Any]:
    recs = [r for r in records if r.get("kind", "dispatch") != "config"]
    recs.sort(key=_ts)
    n_in = n_blocks = n_filtered = 0
    seen: dict[str, dict[tuple[str, str], list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    for rec in recs:
        n_in += 1
        if not rec.get("blocks"):
            continue
        n_blocks += 1
        r, s = str(rec.get("role") or "unknown"), str(rec.get("source") or "unlabelled")
        if (role and r != role) or (source and s != source):
            n_filtered += 1
            continue
        units = _units(rec)
        preceding: list[tuple[str, str]] = []
        chunk_hashes: set[str] = set()
        chunk_lengths: list[int] = []
        for u in units:
            key = (u["kind"], u["hash"])
            pid = _prefix_id(preceding)
            prior = seen[r][key]
            if not prior:
                cls = "first"
            elif any(p == pid for _, p in prior):
                cls = "prefix_repeat"
            elif any(st == u["start"] for st, _ in prior):
                cls = "pred_repeat"
            else:
                cls = "moved_repeat"
            cell = cells.setdefault((u["kind"], r, s), {
                "n": {c: 0 for c in CLASSES}, "chars": {c: 0 for c in CLASSES}})
            cell["n"][cls] += 1
            cell["chars"][cls] += u["length"]
            prior.append((u["start"], pid))
            preceding.append(key)
            if u["is_chunk"]:
                chunk_hashes.add(u["hash"])
                chunk_lengths.append(u["length"])
        plain, recon = _prefix_shares(units, list(history[r]))
        history[r].append(units)
        requests.append({
            "role": r, "source": s, "n_blocks": len(units),
            "typed_chars": sum(u["length"] for u in units),
            "distinct_chunks": len(chunk_hashes), "chunk_lengths": chunk_lengths,
            "plain_prefix_share": plain, "reconstructible_share": recon,
            "prompt_tokens": rec.get("prompt_tokens") or (rec.get("result") or {}).get("prompt_tokens"),
            "cached_prompt_tokens": rec.get("cached_prompt_tokens")
            or (rec.get("result") or {}).get("cached_prompt_tokens"),
        })

    def share(cell: dict[str, Any], field: str) -> dict[str, Any]:
        tot = sum(cell[field].values())
        out: dict[str, Any] = {"total": tot}
        for c in CLASSES:
            out[c] = cell[field][c] / tot if tot else None
        return out

    per_cell = []
    for (kind, r, s), cell in sorted(cells.items()):
        per_cell.append({
            "kind": kind, "role": r, "source": s,
            "occurrences": share(cell, "n"), "chars": share(cell, "chars"),
            "n": dict(cell["n"]), "chars_by_class": dict(cell["chars"]),
        })
    by_rs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for q in requests:
        by_rs[(q["role"], q["source"])].append(q)
    per_request = []
    for (r, s), qs in sorted(by_rs.items()):
        lens = [x for q in qs for x in q["chunk_lengths"]]
        per_request.append({
            "role": r, "source": s, "requests": len(qs),
            "distinct_chunks_mean": statistics.fmean(q["distinct_chunks"] for q in qs),
            "chunk_length_median": statistics.median(lens) if lens else None,
            "chunk_length_mean": statistics.fmean(lens) if lens else None,
            "plain_prefix_share_mean": statistics.fmean(q["plain_prefix_share"] for q in qs),
            "reconstructible_share_mean": statistics.fmean(q["reconstructible_share"] for q in qs),
            "reconstructible_minus_plain_mean": statistics.fmean(
                q["reconstructible_share"] - q["plain_prefix_share"] for q in qs),
        })
    return {
        "schema": REPORT_SCHEMA,
        "inputs": {"records": n_in, "with_blocks": n_blocks, "filtered_out": n_filtered,
                   "analysed_requests": len(requests), "window": window},
        "units": "characters of the rendered prompt; block-granular, typed blocks only",
        "classes": list(CLASSES),
        "per_kind_role_source": per_cell,
        "per_request": per_request,
    }


def _pct(x: Any) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render_markdown(rep: dict[str, Any]) -> str:
    i = rep["inputs"]
    out = [
        "# Prompt layout reuse report", "",
        f"Records {i['records']}, with blocks {i['with_blocks']}, analysed requests "
        f"{i['analysed_requests']} (window {i['window']}). Units: {rep['units']}.", "",
        "## Reuse by kind, role, source (share of characters)", "",
        "| kind | role | source | chars | first | prefix_repeat (not ours) | moved_repeat | pred_repeat |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in rep["per_kind_role_source"]:
        ch = c["chars"]
        out.append(f"| {c['kind']} | {c['role']} | {c['source']} | {ch['total']} | {_pct(ch['first'])} | "
                   f"{_pct(ch['prefix_repeat'])} | {_pct(ch['moved_repeat'])} | {_pct(ch['pred_repeat'])} |")
    out += ["", "## Per request (role, source)", "",
            "| role | source | requests | distinct chunks/req | chunk len median | plain prefix | reconstructible | gain |",
            "|---|---|---|---|---|---|---|---|"]
    for q in rep["per_request"]:
        med = q["chunk_length_median"]
        out.append(f"| {q['role']} | {q['source']} | {q['requests']} | {q['distinct_chunks_mean']:.2f} | "
                   f"{'-' if med is None else f'{med:g}'} | {_pct(q['plain_prefix_share_mean'])} | "
                   f"{_pct(q['reconstructible_share_mean'])} | {_pct(q['reconstructible_minus_plain_mean'])} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--format", choices=("json", "md"), default="md")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--role")
    ap.add_argument("--source")
    a = ap.parse_args(argv)
    rep = analyse(iter_records(a.paths), window=a.window, role=a.role, source=a.source)
    sys.stdout.write(json.dumps(rep, indent=2, sort_keys=True) + "\n" if a.format == "json" else render_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
