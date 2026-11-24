#!/usr/bin/env python3
"""AEI-TOKEN-SPAN-RATIO: all-span versus root-span token ratio over a stored run.

For one stored request, ``ratio = sum(per-call tokens) / root-request tokens``. The per-call side is
the usage ledger (``epyc.request_call_usage.v1`` ``settled`` rows, each ``call_id`` counted once;
censored or unmetered rows carry no count and are not summed). The root side is the
``ChatResponse.output_tokens_reported`` (or ``prompt_tokens``) of that request. A ratio other than
1.0 means a nested span was counted twice or a call was missed.

Offline and read-only. It never invents a ratio: a request with no root total, no complete call
counts or a zero root is reported with a named ``status`` and ``ratio = null``.

Usage:
    token_span_ratio.py --ledger usage_provenance.jsonl --root roots.json [--field completion_tokens]

``--root`` is either ``roots.json`` or the ``usage_roots.jsonl`` written by the ``usage_root_capture``
flag (AEI-TOKEN-ROOT-CAPTURE). ``roots.json`` maps request_id -> {"output_tokens_reported": int|null, "prompt_tokens": int|null}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

FIELDS = {"completion_tokens": "output_tokens_reported", "prompt_tokens": "prompt_tokens"}


def settled_calls(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    """request_id -> call_id -> last settled call row (a call is charged at most once)."""
    out: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("event") != "settled":
            continue
        call = row.get("call") or {}
        if call.get("call_id"):
            out[str(row.get("request_id"))][str(call["call_id"])] = call
    return out


def span_ratio(
    calls: Mapping[str, Mapping[str, Any]], root_tokens: Any, *, field: str = "completion_tokens"
) -> dict[str, Any]:
    """Ratio for one request. ``calls`` is call_id -> settled call; ``root_tokens`` the root total."""
    counted = [
        c[field]
        for c in calls.values()
        if not c.get("pending")
        and not c.get("censored")
        and isinstance(c.get(field), int)
        and not isinstance(c.get(field), bool)
    ]
    result: dict[str, Any] = {
        "n_calls": len(calls),
        "n_calls_counted": len(counted),
        "all_span_tokens": sum(counted) if counted else None,
        "root_tokens": root_tokens if isinstance(root_tokens, int) and not isinstance(root_tokens, bool) else None,
        "ratio": None,
    }
    if result["root_tokens"] is None:
        result["status"] = "no_root_total"
    elif not counted:
        result["status"] = "no_call_counts"
    elif result["root_tokens"] == 0:
        result["status"] = "zero_root"
    else:
        result["ratio"] = result["all_span_tokens"] / result["root_tokens"]
        result["status"] = "ok" if result["ratio"] == 1.0 else "DEFECT_ratio_not_1"
    return result


def check(
    ledger_rows: Iterable[Mapping[str, Any]],
    roots: Mapping[str, Mapping[str, Any]],
    *,
    field: str = "completion_tokens",
) -> dict[str, Any]:
    """Per-request ratios for every request that has ledger rows; ``summary`` counts by status."""
    per_request: dict[str, dict[str, Any]] = {}
    for rid, calls in sorted(settled_calls(ledger_rows).items()):
        root = (roots.get(rid) or {}).get(FIELDS[field])
        per_request[rid] = span_ratio(calls, root, field=field)
    summary: dict[str, int] = defaultdict(int)
    for r in per_request.values():
        summary[r["status"]] += 1
    return {"field": field, "n_requests": len(per_request), "summary": dict(summary), "requests": per_request}


ROOT_SCHEMA = "epyc.request_root_usage.v1"


def load_roots(path: str) -> dict[str, dict[str, Any]]:
    """Read ``--root``: a roots.json mapping, or the ``usage_roots.jsonl`` the usage_root_capture
    flag writes beside the ledger (one ``epyc.request_root_usage.v1`` row per request; last wins)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict) and loaded.get("schema") != ROOT_SCHEMA:
            return loaded
        rows = [loaded]
    except ValueError:
        rows = []
        for line in text.splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return {
        str(r["request_id"]): {k: r.get(k) for k in ("prompt_tokens", "output_tokens_reported", "path")}
        for r in rows
        if isinstance(r, dict) and r.get("schema") == ROOT_SCHEMA and r.get("request_id")
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--field", choices=sorted(FIELDS), default="completion_tokens")
    args = ap.parse_args(argv)
    with open(args.ledger, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    roots = load_roots(args.root)
    report = check(rows, roots, field=args.field)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 1 if report["summary"].get("DEFECT_ratio_not_1") else 0


if __name__ == "__main__":
    raise SystemExit(main())
