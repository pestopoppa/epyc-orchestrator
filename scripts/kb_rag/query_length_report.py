#!/usr/bin/env python3
"""Report KB-RAG query-length telemetry (internal-kb-rag.md H2).

Reads the append-only log written by `kb_rag.query`, which contains one untruncated
token count per query. Prints p50 / p95 / max / over-cap rate per (encoder, cap)
group. Zero-inference: this reads a file and never loads a model.

    python scripts/kb_rag/query_length_report.py                 # human summary
    python scripts/kb_rag/query_length_report.py --json          # full report
    python scripts/kb_rag/query_length_report.py --since 2026-09-16T00:00:00Z
    python scripts/kb_rag/query_length_report.py --out report.json   # persist the report,
        # including the `belief_measurements` rows that the belief-kernel adapter projects

Exit status: 0 on a report (even an empty one), 2 when the log does not exist.
An empty log prints "no observations". It never prints a 0 % rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.retrieval import kb_rag_query_telemetry as telemetry  # noqa: E402


def _human(report: dict) -> str:
    log = report["log"]
    lines = [
        f"log: {log['path']} ({log['bytes']} bytes, sha256 {log['sha256'][:16]}…, "
        f"{log['malformed_lines']} malformed, {log['foreign_lines']} foreign)",
    ]
    if report["since"]:
        lines.append(f"since: {report['since']}")
    if not report["groups"]:
        lines.append("no observations — the over-cap rate is UNKNOWN, not 0 %")
        return "\n".join(lines)
    for g in report["groups"]:
        lines.append(
            f"{g['encoder_model_dir']} [{g['prefix_convention']}] cap={g['cap']}: "
            f"n={g['n']} p50={g['p50']} p95={g['p95']} max={g['max']} "
            f"over-cap={g['over_cap_count']}/{g['n']} ({g['over_cap_rate']:.2%}) "
            f"window {g['first_ts']}..{g['last_ts']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", type=Path, default=None,
                        help=f"log path (default: ${telemetry.LOG_ENV} or {telemetry.DEFAULT_LOG_PATH})")
    parser.add_argument("--since", default=None, help="only records with ts >= this ISO-8601 UTC stamp")
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)

    path = args.log or telemetry.log_path() or telemetry.DEFAULT_LOG_PATH
    if not path.exists():
        print(f"no query-length log at {path} — the instrument has not observed any query yet",
              file=sys.stderr)
        return 2
    report = telemetry.build_report(path, since=args.since)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_human(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
