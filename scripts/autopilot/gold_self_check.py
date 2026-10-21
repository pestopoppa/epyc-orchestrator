#!/usr/bin/env python3
"""EV-19 driver: run each eval suite's scorer on its own gold answers (zero inference, CPU).

    python3 scripts/autopilot/gold_self_check.py [--pool PATH] [--max-per-suite N] [--out JSON]

Deterministic stride sample of at most N questions per suite (0 = all). Writes the JSON report
consumed by src.autopilot_core.gold_self_check.header_text and prints a per-suite table.
Refuses to score code_execution / llm_judge rows (not checkable without running code or a model).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))

from src.autopilot_core.gold_self_check import DEFAULT_REPORT, classify, summarize  # noqa: E402

DEFAULT_POOL = Path(os.environ.get("EPYC_RESEARCH_ROOT", "/mnt/raid0/llm/epyc-inference-research")) \
    / "benchmarks" / "prompts" / "question_pool.jsonl"


def _stride(rows: list, n: int) -> list:
    if n <= 0 or len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--max-per-suite", type=int, default=300)
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    a = ap.parse_args(argv)

    from seeding_scoring import score_answer_or_error

    by_suite: dict[str, list] = {}
    with open(a.pool, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("__pool_metadata__"):
                continue
            by_suite.setdefault(str(q.get("suite", "unknown")), []).append(q)
    results = []
    for suite, rows in sorted(by_suite.items()):
        for q in _stride(rows, a.max_per_suite):
            outcome, detail = classify(q, score_answer_or_error)
            results.append((suite, outcome, detail))
    report = summarize(results)
    report.update(pool=str(a.pool), max_per_suite=a.max_per_suite,
                  pool_size_bytes=a.pool.stat().st_size, method="stride sample, bare gold then 8 wrappers")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{'suite':28} {'n':>5} {'chk':>5} {'bare_ok':>7} {'wrap_ok':>7} {'wrong':>5} {'err':>4} {'rate':>7}")
    for s, r in report["suites"].items():
        rate = "-" if r["rate"] is None else f"{100 * r['rate']:.1f}%"
        print(f"{s:28} {r['n']:5d} {r['checkable']:5d} {r['ok_bare']:7d} {r['ok_wrapped']:7d} "
              f"{r['wrong']:5d} {r['scorer_error']:4d} {rate:>7}")
    print("flagged:", ", ".join(report["flagged"]) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
