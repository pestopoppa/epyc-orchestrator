#!/usr/bin/env python3
"""Analyze the frontdoor verifier's shadow-mode predictions vs actual outcomes.

Reads the progress JSONL log(s), joins `routing_decision` events (which
carry the verifier metadata via `last_decision_meta.routing_meta`) with
`task_completed` / `task_failed` / `escalation_triggered` events for the
same `task_id`, then reports:

    - Volume of routing decisions per source (classifier / learned / rules)
    - Verifier coverage (% of frontdoor routes where the verifier ran)
    - Per-verdict outcome rates (accept / reject vs. success / failure)
    - Brier score, ROC-AUC, ECE for `verifier_p_success` against task outcome

Run after at least a few hundred routing decisions have accumulated. For
autopilot at ~1k requests/day, 24-48h is enough.

Usage:
    python3 scripts/maintenance/analyze_verifier_shadow.py
    python3 scripts/maintenance/analyze_verifier_shadow.py --days 3
    python3 scripts/maintenance/analyze_verifier_shadow.py --log /path/to/X.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

DEFAULT_LOG_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/logs/progress")


def collect_log_paths(days: int, log_dir: Path) -> list[Path]:
    today = date.today()
    paths = []
    for offset in range(days):
        d = today - timedelta(days=offset)
        p = log_dir / f"{d.isoformat()}.jsonl"
        if p.exists():
            paths.append(p)
    return paths


def load_events(paths: list[Path]) -> tuple[dict, dict]:
    """Return ({task_id: routing_meta}, {task_id: outcome})."""
    routing_meta_by_task: dict[str, dict] = {}
    outcome_by_task: dict[str, str] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = e.get("event_type")
                tid = e.get("task_id")
                if not tid:
                    continue
                if ev == "routing_decision":
                    d = e.get("data", {})
                    # routing_meta is spread directly into the data dict via
                    # `**(routing_meta or {})` in progress_logger.log_task_started,
                    # NOT nested under data.routing_meta. So all the fields we
                    # care about (decision_source, chosen_action, verifier_*)
                    # live at data.X directly.
                    if d.get("decision_source"):
                        routing_meta_by_task[tid] = d
                elif ev == "task_completed":
                    outcome_by_task[tid] = "success"
                elif ev == "task_failed":
                    outcome_by_task.setdefault(tid, "failure")
                elif ev == "escalation_triggered":
                    outcome_by_task[tid] = "escalation"
    return routing_meta_by_task, outcome_by_task


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(scores))
    rp = ranks[labels == 1].sum()
    return float((rp - pos * (pos - 1) / 2) / (pos * neg))


def _ece(p: np.ndarray, y: np.ndarray, nb: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, nb + 1)
    e, N = 0.0, len(p)
    for i in range(nb):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi if i < nb - 1 else p <= hi)
        if not m.any():
            continue
        e += (m.sum() / N) * abs(p[m].mean() - y[m].mean())
    return float(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifier shadow-mode analyzer")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of recent days to scan (default 7)")
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--log", type=str, default=None,
                        help="Single log file path (overrides --days/--log-dir)")
    args = parser.parse_args()

    if args.log:
        paths = [Path(args.log)]
    else:
        paths = collect_log_paths(args.days, Path(args.log_dir))
    if not paths:
        print(f"No log files found in {args.log_dir} for last {args.days} days")
        return 1
    print(f"Scanning {len(paths)} log file(s): {[p.name for p in paths]}\n")

    routing_meta, outcomes = load_events(paths)
    print(f"Total routing_decision events with meta: {len(routing_meta):,}")
    print(f"Total tasks with outcome: {len(outcomes):,}")
    joined = sum(1 for tid in routing_meta if tid in outcomes)
    print(f"Joined (routing + outcome): {joined:,}\n")

    # Decision-source distribution
    sources = Counter(rm.get("decision_source", "?") for rm in routing_meta.values())
    print("Decision sources:")
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n:,}")
    print()

    # Verifier coverage
    classifier_routes = [
        (tid, rm) for tid, rm in routing_meta.items()
        if rm.get("decision_source") in ("classifier", "classifier_verifier_reject")
    ]
    verifier_ran = [
        (tid, rm) for tid, rm in classifier_routes
        if "verifier_p_success" in rm
    ]
    frontdoor_routes = [
        (tid, rm) for tid, rm in classifier_routes
        if rm.get("chosen_action") == "frontdoor"
    ]
    print(f"Classifier fast-path decisions: {len(classifier_routes):,}")
    print(f"  ↳ where verifier ran:          {len(verifier_ran):,}")
    print(f"  ↳ frontdoor-class decisions:   {len(frontdoor_routes):,}")
    print(f"  ↳ verifier coverage (of frontdoor): "
          f"{100 * len(verifier_ran) / max(len(frontdoor_routes), 1):.1f}%\n")

    # Verdict × outcome cross-tab
    if verifier_ran:
        verdicts = Counter(rm.get("verifier_verdict", "?") for _, rm in verifier_ran)
        print("Verifier verdict distribution:")
        for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
            print(f"  {v}: {n:,}")
        print()

        # Cross-tab with outcomes
        crosstab: dict[tuple[str, str], int] = Counter()
        for tid, rm in verifier_ran:
            outcome = outcomes.get(tid)
            if outcome is None:
                continue
            verdict = rm.get("verifier_verdict", "?")
            crosstab[(verdict, outcome)] += 1
        print(f"Verifier verdict × actual outcome (n={sum(crosstab.values()):,}):")
        print(f"  {'verdict':<10} {'success':>10} {'failure':>10} {'escalation':>12}  "
              f"{'success-rate':>14}")
        for verdict in ("accept", "reject"):
            s = crosstab.get((verdict, "success"), 0)
            f = crosstab.get((verdict, "failure"), 0)
            e = crosstab.get((verdict, "escalation"), 0)
            total = s + f + e
            rate = f"{100 * s / total:.1f}%" if total else "—"
            print(f"  {verdict:<10} {s:>10,} {f:>10,} {e:>12,}  {rate:>14}")
        print()

        # Brier / ROC-AUC / ECE for verifier_p_success vs binary correctness
        ps, ys = [], []
        for tid, rm in verifier_ran:
            outcome = outcomes.get(tid)
            if outcome is None:
                continue
            p = rm.get("verifier_p_success")
            if p is None:
                continue
            ps.append(float(p))
            ys.append(1.0 if outcome == "success" else 0.0)
        if ps:
            ps = np.array(ps, dtype=np.float32)
            ys = np.array(ys, dtype=np.float32)
            brier = _brier(ps, ys)
            auc = _roc_auc(ps, ys.astype(int))
            ece = _ece(ps, ys)
            print(f"Calibration vs binary success (n={len(ps):,}, base_rate={ys.mean():.3f}):")
            print(f"  Brier:    {brier:.4f}")
            print(f"  ROC-AUC:  {auc:.4f}")
            print(f"  ECE:      {ece:.4f}")
            print()
            # Compare against offline gates (P6.2.5 / A2 thresholds)
            print("Gate comparison (from offline A2 retrain):")
            print(f"  Brier ≤ 0.05 (offline gate-margin)?  {'YES' if brier <= 0.05 else 'NO'} ({brier:.4f})")
            print(f"  ROC-AUC ≥ 0.75 (P6.2.5)?              {'YES' if auc >= 0.75 else 'NO'} ({auc:.4f})")
            print(f"  ECE ≤ 0.05 (P6.2.5)?                  {'YES' if ece <= 0.05 else 'NO'} ({ece:.4f})")
            print()
            verdict = (brier <= 0.05) and (auc >= 0.75) and (ece <= 0.05)
            print(f"Shadow-validation overall: {'PASS — safe to flip to enforcing' if verdict else 'INCOMPLETE — collect more data or review caveats'}")
    else:
        print("No verifier-instrumented routes found yet — gather more shadow data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
