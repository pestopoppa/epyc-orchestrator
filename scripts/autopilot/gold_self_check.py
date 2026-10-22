#!/usr/bin/env python3
"""EV-19 driver: run each eval suite's scorer on its own gold answers (zero inference, CPU).

    python3 scripts/autopilot/gold_self_check.py [--pool PATH] [--max-per-suite N] [--out JSON]
        [--belief-out DIR --eval-era-row ROW [--pool-id ID] [--run-id ID] [--epyc-root PATH]]

Deterministic stride sample of at most N questions per suite (0 = all). Writes the JSON report
consumed by src.autopilot_core.gold_self_check.header_text and prints a per-suite table.
Refuses to score code_execution / llm_judge rows (not checkable without running code or a model).

The report also carries ``pool_sha256``, ``pool_rows`` (non-metadata rows) and ``generated_at`` (additive keys).
``--belief-out DIR`` (VB-EV19-GOLD-SELFCHECK-WIRE) then writes ``ev19_gold_selfcheck.beliefs.jsonl`` into DIR through
the root belief-kernel writer, AFTER the report is on disk; a refusal is loud (exit 3). The claim is about the
SCORER and POOL only, never a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
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


def pool_identity(path: Path) -> tuple[str, int]:
    """(sha256 of the pool file, number of non-metadata rows): the identity the belief row is bound to."""
    h, rows = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for line in fh:
            h.update(line)
            if line.strip() and b'"__pool_metadata__"' not in line[:64]:
                rows += 1
    return h.hexdigest(), rows


def write_belief_sidecar(a: argparse.Namespace, report: dict) -> int:
    """Emit the EV-19 sidecar from the report already written at ``a.out``; exit code 3 on refusal."""
    from scripts.lib.epyc_root import load_adapter

    try:
        Path(a.belief_out).mkdir(parents=True, exist_ok=True)
        side = load_adapter("ev19_gold_selfcheck", a.epyc_root).write_belief_measurements(
            a.belief_out, run_id=a.run_id, producer="epyc-orchestrator/scripts/autopilot/gold_self_check.py",
            report_path=a.out, pool_path=a.pool, pool_id=a.pool_id or a.pool.stem,
            eval_era_row=a.eval_era_row, expected_pool_sha256=report["pool_sha256"],
            emitted_at=report["generated_at"])
    except Exception as exc:  # noqa: BLE001 - the report exists; surface the refusal, never hide it
        print(f"!!! BELIEF CAPTURE FAILED [ev19]: REFUSED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"belief sidecar: {side}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--max-per-suite", type=int, default=300)
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--belief-out", type=Path, help="also write ev19_gold_selfcheck.beliefs.jsonl into this dir (root writer)")
    ap.add_argument("--eval-era-row", help="era row the pool belongs to (required with --belief-out; never guessed)")
    ap.add_argument("--pool-id", help="pool id (default: the pool file's stem)")
    ap.add_argument("--run-id", default="ev19-gold-self-check")
    ap.add_argument("--epyc-root", help="root checkout (default: $EPYC_ROOT, epyc-root-main view, /workspace)")
    a = ap.parse_args(argv)
    if a.belief_out and not a.eval_era_row:
        ap.error("--belief-out needs --eval-era-row")

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
    report["pool_sha256"], report["pool_rows"] = pool_identity(a.pool)
    report["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{'suite':28} {'n':>5} {'chk':>5} {'bare_ok':>7} {'wrap_ok':>7} {'wrong':>5} {'err':>4} {'rate':>7}")
    for s, r in report["suites"].items():
        rate = "-" if r["rate"] is None else f"{100 * r['rate']:.1f}%"
        print(f"{s:28} {r['n']:5d} {r['checkable']:5d} {r['ok_bare']:7d} {r['ok_wrapped']:7d} "
              f"{r['wrong']:5d} {r['scorer_error']:4d} {rate:>7}")
    print("flagged:", ", ".join(report["flagged"]) or "none")
    if a.belief_out:
        return write_belief_sidecar(a, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
