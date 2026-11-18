#!/usr/bin/env python3
"""PREDECLARED scoring script for codemode-ab-20261117. The ONLY source of the verdict; zero compute.

    score.py RESULTS.jsonl [--stage-upto 1|2|3] [--out verdict.json]

Reads ``kind == "episode"`` records (one per model x arm x task x seed) and compares arms PAIRED on the same
(model, task_id, seed). Constants below are the predeclaration (sha256 of this file is pinned in
PREDECLARED.md and checked by precond.sh); change nothing after the first live record.

Contrasts: P1 = B vs A (does code mode beat direct calls?), P2 = C1 vs B and C2 vs B (does a per-request
selected tool subset beat today's static list + list_tools()), P2n = the same restricted to tier "novel"
(tools NOT in the static list, where arm C should matter), P3 = C1 vs C2 selector comparison.

Per cell (contrast, model, stratum) the statistics are paired differences over (task_id, seed) pairs:
  d_success   mean(success_X - success_Y), 95% paired bootstrap CI (10000 resamples, seed 20261117) and the exact
              two-sided sign test on discordant pairs;
  r_tokens    median of ln((in+out+selector tokens)_X / (...)_Y); in and out also reported SEPARATELY;
  r_wall, r_turns likewise (ln ratios); plus mean failed_snippet_rate, param_accuracy, duplicate_writes,
  and (C arms) selector recall.
Direction: success higher = better; tokens/wall/turns lower = better. A ratio ln<0 means X is cheaper than Y.

Rules (constants):
  DELTA_SUCC = 0.10  absolute success gap that counts as an effect
  RATIO_EPS  = 0.10  |ln ratio| < ln(1.10) counts as "no cost gap"
  FUTILITY (interim, N=20 look, formal-gate-safe = futility ONLY): for the P1 contrast pooled over the N<=20 stratum,
       stop if |d_success| < FUT_SUCC (0.05) AND |median ln r_tokens| < ln(1.10) AND |median ln r_wall| < ln(1.10)
       for EVERY model scored so far. Otherwise continue.
  N=50 LOOK: futility rule above on the N<=50 stratum, plus DESCRIPTIVE stop-on-answer ("sign is clear"):
       d_success CI excludes 0 OR the token ln-ratio bootstrap CI excludes 0, AND the sign agrees in all seeds.
       A stop-on-answer is recorded as grade "descriptive_interim" and NEVER as a formal-gate claim.
  FINAL verdict per contrast/model/stratum: "X_better"  if d_success CI excludes 0 with sign +, or (d_success CI
       contains 0 and token ln-ratio CI excludes 0 with sign -  [X cheaper at equal success]);
       "Y_better" mirror image; "no_difference" if both CIs contain 0 and |d_success|<DELTA_SUCC and
       |median ln r_tokens|<ln(1+RATIO_EPS); else "inconclusive".
  Under-powered cells (fewer than MIN_PAIRS=6 pairs) are reported "underpowered", never a verdict.
Censoring: an episode that ended llm_error is NOT scored as failure; it is dropped with its pair and counted.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DELTA_SUCC = 0.10
RATIO_EPS = 0.10
FUT_SUCC = 0.05
MIN_PAIRS = 6
BOOT = 10000
BOOT_SEED = 20261117
CONTRASTS = {"P1": ("B", "A"), "P2_C1": ("C1", "B"), "P2_C2": ("C2", "B"), "P3": ("C1", "C2")}
STRATA = {"n_le_20": lambda r: r["n"] <= 20, "n_eq_50": lambda r: r["n"] == 50, "n_ge_100": lambda r: r["n"] >= 100,
          "all": lambda r: True, "novel_all": lambda r: r["tier"] == "novel", "static_all": lambda r: r["tier"] == "static"}


def total_tokens(r: dict[str, Any]) -> int:
    return r["in_tokens"] + r["out_tokens"] + r.get("selector_in_tokens", 0) + r.get("selector_out_tokens", 0)


def load(path: str | Path, mode: str = "full") -> list[dict[str, Any]]:
    out = []
    for ln in Path(path).read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("kind") == "episode" and r.get("mode", "full") == mode:
            out.append(r)
    return out


def _boot_ci(vals: list[float], stat=statistics.fmean) -> tuple[float, float]:
    rng = random.Random(BOOT_SEED)
    n = len(vals)
    est = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(BOOT))
    return est[int(0.025 * BOOT)], est[int(0.975 * BOOT) - 1]


def _sign_p(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _lnr(x: float, y: float) -> float:
    return math.log(max(x, 1e-9) / max(y, 1e-9))


def cell(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """pairs of (X record, Y record)."""
    n = len(pairs)
    if n < MIN_PAIRS:
        return {"n_pairs": n, "verdict": "underpowered"}
    ds = [float(x["success"]) - float(y["success"]) for x, y in pairs]
    d_ci = _boot_ci(ds)
    tok = [_lnr(total_tokens(x), total_tokens(y)) for x, y in pairs]
    tok_ci = _boot_ci(tok, statistics.median)
    pos, neg = sum(d > 0 for d in ds), sum(d < 0 for d in ds)
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (x, _), d in zip(pairs, ds):
        by_seed[x["seed"]].append(d)
    seed_signs = {s: (1 if statistics.fmean(v) > 0 else -1 if statistics.fmean(v) < 0 else 0) for s, v in by_seed.items()}
    d_mean = statistics.fmean(ds)
    tok_med = statistics.median(tok)
    # Success-direction verdict needs evidence of a success difference: the paired-bootstrap CI excludes 0 (the sign
    # test is reported alongside and must not contradict it). A token-only verdict (the predeclared "X cheaper at equal
    # success") additionally needs a cost gap of at least RATIO_EPS: below it the predeclaration says "no cost gap".
    # Tied success with a sign-test p of 1.0 therefore never yields a success-direction verdict.
    eps = math.log(1 + RATIO_EPS)
    ci_straddles = d_ci[0] <= 0 <= d_ci[1]
    basis = "none"
    if d_ci[0] > 0:
        verdict, basis = "X_better", "success"
    elif d_ci[1] < 0:
        verdict, basis = "Y_better", "success"
    elif ci_straddles and tok_ci[1] < 0 and tok_med <= -eps:
        verdict, basis = "X_better", "tokens_only"
    elif ci_straddles and tok_ci[0] > 0 and tok_med >= eps:
        verdict, basis = "Y_better", "tokens_only"
    elif abs(d_mean) < DELTA_SUCC and abs(tok_med) < eps:
        verdict = "no_difference"
    else:
        verdict = "inconclusive"
    mean = lambda f, rs: (statistics.fmean([v for v in (f(r) for r in rs) if v is not None]) if any(f(r) is not None for r in rs) else None)  # noqa: E731
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "n_pairs": n, "success_x": mean(lambda r: float(r["success"]), xs), "success_y": mean(lambda r: float(r["success"]), ys),
        "d_success": d_mean, "d_success_ci95": list(d_ci), "sign_test_p": _sign_p(pos, neg), "discordant": [pos, neg],
        "ln_ratio_total_tokens_median": tok_med, "ln_ratio_total_tokens_ci95": list(tok_ci),
        "ln_ratio_in_tokens_median": statistics.median(_lnr(x["in_tokens"], y["in_tokens"]) for x, y in pairs),
        "ln_ratio_out_tokens_median": statistics.median(_lnr(x["out_tokens"], y["out_tokens"]) for x, y in pairs),
        "ln_ratio_wall_median": statistics.median(_lnr(x["wall_s"], y["wall_s"]) for x, y in pairs),
        "ln_ratio_turns_median": statistics.median(_lnr(max(x["turns"], 1), max(y["turns"], 1)) for x, y in pairs),
        "failed_snippet_rate_x": mean(lambda r: r["failed_snippet_rate"], xs), "failed_snippet_rate_y": mean(lambda r: r["failed_snippet_rate"], ys),
        "param_accuracy_x": mean(lambda r: r["param_accuracy"], xs), "param_accuracy_y": mean(lambda r: r["param_accuracy"], ys),
        "duplicate_writes_x": mean(lambda r: r["duplicate_writes"], xs), "duplicate_writes_y": mean(lambda r: r["duplicate_writes"], ys),
        "selector_recall_x": mean(lambda r: (r.get("selector") or {}).get("recall"), xs),
        "selector_all_in_topk_x": mean(lambda r: float((r.get("selector") or {}).get("needed_all_in_topk")) if r.get("selector") else None, xs),
        "seed_signs": {str(k): v for k, v in sorted(seed_signs.items())}, "verdict": verdict, "verdict_basis": basis,
    }


def evaluate(records: list[dict[str, Any]], stage_upto: int = 3) -> dict[str, Any]:
    recs = [r for r in records if r["stage"] <= stage_upto]
    censored = sum(1 for r in recs if str(r.get("end_reason", "")).startswith("llm_error"))
    recs = [r for r in recs if not str(r.get("end_reason", "")).startswith("llm_error")]
    idx: dict[tuple[str, str, int, str], dict[str, Any]] = {(r["model"], r["task_id"], r["seed"], r["arm"]): r for r in recs}
    models = sorted({r["model"] for r in recs})
    out: dict[str, Any] = {"schema": "epyc.codemode_ab.verdict.v1", "stage_upto": stage_upto, "n_episodes": len(recs),
                           "censored_llm_error": censored, "models": models, "cells": {}}
    for m in models:
        for cname, (xa, ya) in CONTRASTS.items():
            for sname, pred in STRATA.items():
                pairs = []
                for (mm, tid, sd, arm), x in idx.items():
                    if mm == m and arm == xa and pred(x) and (m, tid, sd, ya) in idx:
                        pairs.append((x, idx[(m, tid, sd, ya)]))
                if pairs:
                    out["cells"][f"{m}|{cname}|{sname}"] = cell(pairs)
    out["interim"] = {"n20": interim(out, "n_le_20"), "n50": interim(out, "n_eq_50", n50=True)}
    return out


def interim(res: dict[str, Any], stratum: str, n50: bool = False) -> dict[str, Any]:
    """Futility-only at interim looks (formal-gate rule); the N=50 descriptive stop-on-answer is graded as such."""
    futile, answer, seen = [], [], 0
    for m in res["models"]:
        c = res["cells"].get(f"{m}|P1|{stratum}")
        if not c or c.get("verdict") == "underpowered":
            continue
        seen += 1
        futile.append(abs(c["d_success"]) < FUT_SUCC and abs(c["ln_ratio_total_tokens_median"]) < math.log(1 + RATIO_EPS)
                      and abs(c["ln_ratio_wall_median"]) < math.log(1 + RATIO_EPS))
        signs = set(c["seed_signs"].values())
        excl = c["d_success_ci95"][0] > 0 or c["d_success_ci95"][1] < 0 or c["ln_ratio_total_tokens_ci95"][0] > 0 or c["ln_ratio_total_tokens_ci95"][1] < 0
        answer.append(excl and len(signs - {0}) == 1 and 0 not in signs)
    d = {"models_scored": seen, "stop_on_futility": bool(seen) and all(futile)}
    if n50:
        d["stop_on_answer_descriptive"] = bool(seen) and all(answer)
        d["grade"] = "descriptive_interim"
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--stage-upto", type=int, default=3)
    ap.add_argument("--mode", default="full", help="record mode to score (full; smoke only for harness tests)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = evaluate(load(a.results, a.mode), a.stage_upto)
    text = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
