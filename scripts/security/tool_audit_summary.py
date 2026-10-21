#!/usr/bin/env python3
"""SEC-EXEC-7: zero-token daily summary of flagged model tool calls.

Reads the SEC-EXEC-4 audit log (``tool_call_audit.jsonl`` and its rotated siblings) and lists
the calls a human should look at for the last N hours:

* ``first_seen_host``      web_fetch / fetch_docs to a destination never reached before
* ``write_outside_root``   run_shell / run_python_code that wrote outside the allowed roots
* ``spawned_process``      run_python_code that started a subprocess (the write hook is blind there)
* refused calls, grouped by refusal reason (containment, web guard, exec refusals)

No model is involved. Exit status is always 0 unless the log cannot be read at all (2), so a cron
wrapper can publish the report without gating on its content. Records carry credential-redacted
text only; this script never prints more than the record holds, and caps each flagged line.

Usage:
    tool_audit_summary.py [--log PATH] [--since-hours 24] [--out-dir DIR] [--json]

``--out-dir`` writes ``YYYY-MM-DD.md`` (and ``.json`` with ``--json``) there; without it the
markdown goes to stdout. Cron (to enable; not installed by this script):
    17 6 * * *  cd <orchestrator> && python3 scripts/security/tool_audit_summary.py --out-dir logs/tool_audit_daily
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

LINE_CAP = 200


def default_log() -> str:
    override = os.environ.get("ORCHESTRATOR_TOOL_AUDIT_LOG")
    if override:
        return override
    log_dir = os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", "/mnt/raid0/llm/epyc-orchestrator/logs")
    return os.path.join(log_dir, "tool_call_audit.jsonl")


def log_files(path: str) -> list[Path]:
    """The live file plus RotatingFileHandler backups (``path.1`` ... ``path.N``), oldest first."""
    base = Path(path)
    found = [p for p in base.parent.glob(base.name + ".*") if p.suffix[1:].isdigit()]
    found.sort(key=lambda p: int(p.suffix[1:]), reverse=True)
    return found + ([base] if base.exists() else [])


def read_records(files: Iterable[Path], since: float) -> tuple[list[dict[str, Any]], int]:
    """(records at or after ``since``, count of unparseable lines)."""
    records: list[dict[str, Any]] = []
    bad = 0
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    if isinstance(rec, dict) and float(rec.get("ts", 0) or 0) >= since:
                        records.append(rec)
        except OSError:
            continue
    return records, bad


def summarize(records: list[dict[str, Any]], bad_lines: int = 0) -> dict[str, Any]:
    by_tool = collections.Counter(r.get("tool", "?") for r in records)
    by_outcome = collections.Counter(r.get("outcome", "?") for r in records)
    refusals = collections.Counter(
        (r.get("tool", "?"), str(r.get("refusal_reason") or "")[:120]) for r in records if r.get("outcome") == "refused"
    )

    def pick(key: str) -> list[dict[str, Any]]:
        return [r for r in records if r.get(key) is True]

    return {
        "total": len(records),
        "unparseable_lines": bad_lines,
        "by_tool": dict(by_tool),
        "by_outcome": dict(by_outcome),
        "refusals": [{"tool": t, "reason": why, "count": n} for (t, why), n in refusals.most_common()],
        "first_seen_host": pick("first_seen_host"),
        "write_outside_root": pick("write_outside_root"),
        "spawned_process": pick("spawned_process"),
        "not_analysed_writes": sum(
            1 for r in records if r.get("tool") in ("run_shell", "run_python_code") and r.get("write_outside_root") is None
        ),
    }


def _line(r: dict[str, Any]) -> str:
    when = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(float(r.get("ts", 0) or 0)))
    text = str(r.get("text") or "").replace("\n", " ")[:LINE_CAP]
    return (
        f"- {when} `{r.get('tool')}` outcome={r.get('outcome')} role={r.get('role')} "
        f"req={r.get('request_id')} host={r.get('host')} sha256={str(r.get('sha256') or '')[:12]} :: {text}"
    )


def render(summary: dict[str, Any], hours: float, log: str) -> str:
    out = [
        f"# Tool-call audit summary (last {hours:g} h)",
        "",
        f"Source: `{log}` (+ rotated files). Calls: {summary['total']}; "
        f"unparseable lines: {summary['unparseable_lines']}.",
        "",
        "## Counts",
        "",
        "| tool | calls |",
        "|---|---|",
        *[f"| {t} | {n} |" for t, n in sorted(summary["by_tool"].items())],
        "",
        "| outcome | calls |",
        "|---|---|",
        *[f"| {o} | {n} |" for o, n in sorted(summary["by_outcome"].items())],
        "",
        "## Refusals by reason",
        "",
    ]
    out += [f"- {r['count']} x `{r['tool']}`: {r['reason']}" for r in summary["refusals"]] or ["- none"]
    for key, title in (
        ("first_seen_host", "First-seen destination hosts"),
        ("write_outside_root", "Writes outside the allowed roots"),
        ("spawned_process", "Python calls that spawned a subprocess (write hook blind)"),
    ):
        out += ["", f"## {title} ({len(summary[key])})", ""]
        out += [_line(r) for r in summary[key]] or ["- none"]
    out += [
        "",
        f"Shell/python calls whose write flag was not analysed (flag off or refused before launch): "
        f"{summary['not_analysed_writes']}.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", default=default_log())
    ap.add_argument("--since-hours", type=float, default=24.0)
    ap.add_argument("--out-dir")
    ap.add_argument("--json", action="store_true", help="also write the structured summary (needs --out-dir)")
    args = ap.parse_args(argv)

    files = log_files(args.log)
    if not files:
        print(f"no audit log found at {args.log}", file=sys.stderr)
        return 2
    records, bad = read_records(files, time.time() - args.since_hours * 3600)
    summary = summarize(records, bad)
    text = render(summary, args.since_hours, args.log)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d", time.gmtime())
        (out_dir / f"{stamp}.md").write_text(text, encoding="utf-8")
        if args.json:
            (out_dir / f"{stamp}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {out_dir / (stamp + '.md')} ({summary['total']} calls, "
              f"{len(summary['first_seen_host'])} first-seen, {len(summary['write_outside_root'])} out-of-root)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
