#!/usr/bin/env python3
"""Non-production records in the production serving-call log (READ-ONLY).

Until 2026-11-02 every process that imported ``src/backends/serving_calls.py`` appended to
``logs/serving_calls/serving_calls.jsonl`` whatever port it called, so private and
experimental servers (8102, 8199, 18283, 18973 ...) sit in the evidence base of the gates
that read that log (e.g. UFH14-B4a's 200-call window). The writer now routes those calls
to ``serving_calls.private.jsonl``; the records already written are NOT rewritten. This
script reports them so a gate reader can exclude them.

A record is production when its ``server.port`` is a production port (the writer's own
rule: ``serving_calls.production_ports()``, i.e. every serving port of a ``live_stack``
role in ``orchestration/derived/stack_priors.yaml``, or ``--production-ports``). A record
without a port is an orchestrator-side event (gate refusal, call abandoned before a
backend was chosen) and counts as production, as the writer treats it.

Nothing is written: the log and its rotated shards are only read.

Usage:
  python3 scripts/analysis/serving_calls_nonprod_report.py [--log PATH] [--production-ports 8070,8083]
      [--json] [--digests]

``--digests`` adds the ``record_sha256`` of every non-production record (the exclusion
list); ``--json`` prints one JSON object instead of the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_LOG = REPO / "logs" / "serving_calls" / "serving_calls.jsonl"


def shards(log: Path) -> list[Path]:
    """The log and its rotated shards ``.1`` ``.2`` …, oldest first."""
    rotated = []
    i = 1
    while log.with_name(f"{log.name}.{i}").exists():
        rotated.append(log.with_name(f"{log.name}.{i}"))
        i += 1
    return list(reversed(rotated)) + ([log] if log.exists() else [])


def resolve_production_ports(spec: str | None) -> frozenset[int] | None:
    if spec:
        return frozenset(int(p) for p in spec.replace(" ", "").split(",") if p)
    from src.backends import serving_calls

    return serving_calls.production_ports()


def classify(paths: list[Path], production: frozenset[int]) -> dict:
    prod: Counter = Counter()
    nonprod: Counter = Counter()
    digests: dict[int, list[str]] = {}
    unparsed = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    unparsed += 1
                    continue
                server = rec.get("server") if isinstance(rec.get("server"), dict) else {}
                port = server.get("port")
                if port is None or (isinstance(port, int) and port in production):
                    prod["none" if port is None else port] += 1
                    continue
                nonprod[port] += 1
                digests.setdefault(port, []).append(rec.get("record_sha256"))
    return {
        "files": [str(p) for p in paths],
        "production_ports": sorted(production),
        "total": sum(prod.values()) + sum(nonprod.values()),
        "production": sum(prod.values()),
        "non_production": sum(nonprod.values()),
        "unparsed_lines": unparsed,
        "production_by_port": {str(k): v for k, v in sorted(prod.items(), key=lambda kv: str(kv[0]))},
        "non_production_by_port": {str(k): v for k, v in sorted(nonprod.items(), key=lambda kv: -kv[1])},
        "_digests": {str(k): v for k, v in digests.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--production-ports", help="comma-separated; default: the writer's production_ports()")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--digests", action="store_true", help="include record_sha256 of each non-production record")
    a = ap.parse_args(argv)
    production = resolve_production_ports(a.production_ports)
    if production is None:
        print("REFUSED: production ports unresolvable (stack priors missing?); pass --production-ports", file=sys.stderr)
        return 2
    paths = shards(a.log)
    if not paths:
        print(f"REFUSED: no log at {a.log}", file=sys.stderr)
        return 2
    out = classify(paths, production)
    digests = out.pop("_digests")
    if a.digests:
        out["non_production_record_sha256"] = digests
    if a.json:
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    print(f"files: {', '.join(out['files'])}")
    print(f"production ports: {','.join(map(str, out['production_ports']))}")
    print(f"records: {out['total']}  production: {out['production']}  non-production: {out['non_production']}"
          f"  unparsed: {out['unparsed_lines']}")
    for port, n in out["non_production_by_port"].items():
        print(f"  non-production port {port}: {n}")
    for port, n in out["production_by_port"].items():
        print(f"  production port {port}: {n}")
    if a.digests:
        for port, ds in digests.items():
            for d in ds:
                print(f"exclude {port} {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
