#!/usr/bin/env python3
"""effort_tb7_budget_mining.py — TB-7: mine AutoPilot's recorded token stats for budget defaults.

Task: handoffs/active/reasoning-effort-levels.md TB-7 (L150). NO new inference — this only mines
the AutoPilot journal shards, which already record per-question `tokens_generated` alongside the
`suite` (task-class), `route` (role/model), and `correct` outcome. From the realized-demand
distribution of *successfully completed* tasks we derive per-(task-class, model) completion-budget
defaults: the deployable budget is ~the high percentile of realized demand (per TB-7), later capped
by the VRAM/concurrency budget (TB-6).

Data source (verified 2026-07-22):
  eval_details.question_results[] entries with a `tokens_generated` field
  (5,135 of 38,100 question records carry it), each with {suite, route, correct, tokens_generated}.
  eval_details.details.{tokens_generated, tokens_per_solved_task} give trial-level aggregates
  (cross-check only — mixed-suite, so not task-class attributable).

CAVEATS (stated in the report too):
  * Realized-demand percentiles are computed over SUCCESSFUL completions. Tasks that failed due to
    truncation needed *more* budget and are invisible here, so p95(success-tokens) is a LOWER bound
    on the true knee — cross-check against a TB-1 sweep before it gates any production change.
  * The journal is a mixture over historical configs (enable_thinking on/off, prompt-effort, model
    era). route->model binding is the *current* production mapping; per the reasoning-effort
    INVARIANT the budget is a (model, quant) property, so a model/quant swap re-opens this.
  * No `truncated`/`finish_reason` field exists at the per-question level in the journal, so
    truncation-rate vs budget (the TB-1 knee) cannot be computed here — only realized demand.

All numbers are OBSERVATIONS per MEASUREMENT.md (no protocol-id/attestation): usable to *propose*
budget defaults, not to gate a production deploy without the TB-1 cross-check.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

DEFAULT_SHARDS = [
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/autopilot_journal_1.jsonl",
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/autopilot_journal.jsonl",
]

# Current production route -> (model, quant) binding (for labelling only; NOT inherited across
# swaps). Source: CLAUDE.md repo map + model_registry. The INVARIANT means a swap re-opens the curve.
ROUTE_MODEL = {
    "frontdoor": "Qwen3.6-35B-A3B (frontdoor)",
    "worker_general": "gemma-4-26B-A4B Q4KM (worker)",
    "worker_vision": "gemma-4-26B-A4B (worker_vision)",
    "coder_escalation": "coder-escalation model",
    "architect_general": "Qwen3.5-122B-A10B-IQ2 (architect)",
    "ingest_long_context": "Qwen3-Next-80B (ingest)",
}
CANON_ROUTES = set(ROUTE_MODEL)

PCTS = [50, 75, 90, 95, 99]


def load_records(shards: List[str]):
    q_records = []  # per-question token-bearing records
    trial_tps = []  # trial-level tokens_per_solved_task
    trial_tg = []   # trial-level tokens_generated
    ts_min = ts_max = None
    for sh in shards:
        p = Path(sh)
        if not p.exists():
            continue
        for ln in p.open():
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("event") or d.get("type"):
                continue
            ed = d.get("eval_details") or {}
            det = ed.get("details") or {}
            if det.get("tokens_per_solved_task"):
                trial_tps.append(det["tokens_per_solved_task"])
            if det.get("tokens_generated"):
                trial_tg.append(det["tokens_generated"])
            ts = d.get("timestamp")
            for q in ed.get("question_results") or []:
                if "tokens_generated" not in q:
                    continue
                tok = q.get("tokens_generated")
                if tok is None:
                    continue
                q_records.append({
                    "suite": q.get("suite"),
                    "route": q.get("route"),
                    "correct": bool(q.get("correct")),
                    "tokens": int(tok),
                    "ts": ts,
                })
                if ts:
                    ts_min = ts if ts_min is None else min(ts_min, ts)
                    ts_max = ts if ts_max is None else max(ts_max, ts)
    return q_records, trial_tps, trial_tg, (ts_min, ts_max)


def pct_table(tokens: List[int]) -> Dict:
    if not tokens:
        return {"n": 0}
    a = np.array(tokens)
    out = {"n": int(a.size), "min": int(a.min()), "mean": round(float(a.mean()), 1),
           "max": int(a.max())}
    for p in PCTS:
        out[f"p{p}"] = int(np.percentile(a, p))
    return out


def recommend_budget(p95: int, headroom: float = 1.15, quantum: int = 512) -> int:
    """Deployable default = p95 realized demand + headroom, rounded up to a token quantum."""
    raw = p95 * headroom
    return int(math.ceil(raw / quantum) * quantum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="*", default=DEFAULT_SHARDS)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--min-n", type=int, default=30,
                    help="min successful samples for a (suite,route) budget recommendation")
    args = ap.parse_args()

    q, trial_tps, trial_tg, (ts_min, ts_max) = load_records(args.shards)
    print(f"[load] per-question token records: {len(q)}  (time {ts_min} .. {ts_max})")

    # group by (suite, route) — success only for the deployable budget; keep all for context
    by_sr_succ = defaultdict(list)
    by_sr_all = defaultdict(list)
    by_suite_succ = defaultdict(list)
    by_route_succ = defaultdict(list)
    noncanon = 0
    for r in q:
        s, rt = r["suite"], r["route"]
        if rt not in CANON_ROUTES:
            noncanon += 1
            continue
        by_sr_all[(s, rt)].append(r["tokens"])
        if r["correct"]:
            by_sr_succ[(s, rt)].append(r["tokens"])
            by_suite_succ[s].append(r["tokens"])
            by_route_succ[rt].append(r["tokens"])

    # per-(suite,route) recommendations
    recs = []
    for (s, rt), toks in sorted(by_sr_succ.items(), key=lambda kv: (-len(kv[1]))):
        if len(toks) < args.min_n:
            continue
        t = pct_table(toks)
        recs.append({
            "task_class": s, "route": rt, "model": ROUTE_MODEL[rt],
            "n_success": t["n"], "median_tokens": t["p50"], "p90": t["p90"],
            "p95": t["p95"], "p99": t["p99"], "max": t["max"],
            "recommended_budget_tokens": recommend_budget(t["p95"]),
        })

    per_suite = {s: pct_table(v) for s, v in sorted(by_suite_succ.items())
                 if len(v) >= args.min_n}
    per_route = {rt: pct_table(v) for rt, v in sorted(by_route_succ.items())}

    # per-suite recommendation (model-agnostic fallback when a (suite,route) cell is thin)
    per_suite_rec = {s: recommend_budget(t["p95"]) for s, t in per_suite.items()}

    out = {
        "measurement_note": "OBSERVATION per MEASUREMENT.md — realized-demand mining, no protocol-id; "
                            "proposes budget defaults, does not gate production without a TB-1 knee sweep.",
        "source_shards": args.shards,
        "time_range": {"min": ts_min, "max": ts_max},
        "n_question_token_records": len(q),
        "n_noncanonical_route_dropped": noncanon,
        "budget_rule": "recommended_budget = ceil(p95(success-tokens) * 1.15 / 512) * 512 ; a LOWER "
                       "bound on the knee (truncated failures excluded); cap by TB-6 VRAM budget.",
        "trial_level_crosscheck": {
            "tokens_per_solved_task": pct_table([int(x) for x in trial_tps]) if trial_tps else {},
            "tokens_generated_per_trial": pct_table([int(x) for x in trial_tg]) if trial_tg else {},
        },
        "per_task_class_model_recommendations": recs,
        "per_task_class_pooled": per_suite,
        "per_task_class_pooled_recommended_budget": per_suite_rec,
        "per_route_pooled": per_route,
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    # console summary
    print(f"[recs] {len(recs)} (task-class,model) cells with n>={args.min_n} successful samples")
    print(f"{'task_class':22s} {'route':20s} {'n':>5s} {'med':>6s} {'p90':>6s} {'p95':>6s} {'p99':>6s} {'budget':>7s}")
    for r in recs:
        print(f"{r['task_class']:22s} {r['route']:20s} {r['n_success']:5d} {r['median_tokens']:6d} "
              f"{r['p90']:6d} {r['p95']:6d} {r['p99']:6d} {r['recommended_budget_tokens']:7d}")
    print(f"\n[done] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
