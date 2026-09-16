#!/usr/bin/env python3
"""SC83: score one reviewer run against an RA-9 decoy corpus and persist the FA rate.

NO inference. Offline: this script scores verdicts that already exist.

    python3 scripts/review/score_false_accept.py \
        --corpus gold_annotations.jsonl       # RA-9 annotations, incl. status=invalid decoys
        --verdicts verdicts.jsonl             # {annotation_id, envelope, signed_body} per line
        --current-bindings bindings.json      # {annotation_id: RA-12 binding of CURRENT inputs}
        --run-id <token> [--out PATH] [--dry-run]

It appends one ``epyc.reviewer.false_accept_run.v1`` line to ``--out``. The default
is ``data/reviewer_eval/false_accept_runs.jsonl``. That line is what
``cli.py ingest reviewer-fa`` reads in epyc-root. A stale verdict is dropped before
scoring, and the line lists it. Exit codes: 0 = written (or dry run), 2 = refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.proactive_delegation import false_accept_record as far  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SC83 negative-control false-accept rate writer")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--current-bindings", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", type=Path, default=far.DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        line = far.build_run_line(
            corpus_path=args.corpus, verdicts_path=args.verdicts,
            bindings_path=args.current_bindings, run_id=args.run_id, out=args.out)
        if not args.dry_run:
            far.append_line(args.out, line)
    except (far.FalseAcceptRecordError, ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    summary = {
        "out": str(args.out), "dry_run": args.dry_run, "run_id": line["run_id"],
        "result": line["result"], "stale": line["stale"],
        "belief_rows": len(line["belief_measurements"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
