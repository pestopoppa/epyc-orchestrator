#!/usr/bin/env python3
"""PREDECLARED_v2 scoring script for probe v2. The ONLY source of the v2 verdict; zero compute. DRAFT: pin its sha256 in
PREDECLARED_v2.md (and in precond.sh) before the first live v2 record; change nothing after.

    score_v2.py RESULTS.jsonl [--stage-upto 1|2|3] [--out verdict_v2.json]

Reuses the statistics of ``score.py`` (paired bootstrap, exact sign test, ln ratios, verdict rule) with the v2 contrasts, and adds
(a) the token ratio on PAIRED SUCCESSES only (the token claim), (b) the harm checks for the hardened picker on the direct-call path.
Contrast keys (X vs Y; success higher = better, tokens lower = better):

  Q1a  B1 vs B       harness fixes (explicit FINAL + prompt v2 + wide preview/compact list_tools)
  Q2   B2 vs B1      turn memory
  Q2b  C3 vs B2      typed stubs + hardened picker + name repair
  Q3   C3 vs A_pick  PRIMARY: full code mode vs direct calls that ALSO get the picker
  Q3a  C3 vs A       secondary, comparable with v1
  QP   A_pick vs A   the picker on the direct-call path: OFF vs hardened (per model: 35B, 27B, Flash-Next)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from scripts.benchmark.codemode_ab import score

CONTRASTS_V2 = {"Q1a": ("B1", "B"), "Q2": ("B2", "B1"), "Q2b": ("C3", "B2"), "Q3": ("C3", "A_pick"), "Q3a": ("C3", "A"),
                "QP": ("A_pick", "A")}
FUT_SUCC = 0.05
RATIO_EPS = 0.10
# harm stops for the hardened picker on the direct-call path (QP), per model
HARM_SUCC = 0.10
HARM_MIN_DISCORDANT = 6
HARM_RECALL = 0.90
HARM_MIN_EPISODES = 50
HARM_RECOVERY = 0.10
HARM_P95_MS = 500.0
HARM_BREAKER_OPENS = 2


def load(path: str | Path, mode: str = "full") -> list[dict[str, Any]]:
    return score.load(path, mode)


def _idx(recs: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    return {(r["model"], r["task_id"], r["seed"], r["arm"]): r for r in recs}


def paired_success_token_ratio(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Median ln(total tokens X / Y) over pairs where BOTH succeeded: the token claim (all-episode ratios are descriptive only,
    because a failing episode that stops early is cheap)."""
    both = [(x, y) for x, y in pairs if x["success"] and y["success"]]
    if not both:
        return {"n_pairs_both_success": 0, "ln_ratio_median": None}
    return {"n_pairs_both_success": len(both),
            "ln_ratio_median": statistics.median(math.log(score.total_tokens(x) / score.total_tokens(y)) for x, y in both)}


def evaluate(records: list[dict[str, Any]], stage_upto: int = 3) -> dict[str, Any]:
    prev = score.CONTRASTS
    score.CONTRASTS = CONTRASTS_V2
    try:
        out = score.evaluate(records, stage_upto)
    finally:
        score.CONTRASTS = prev
    out["schema"] = "epyc.codemode_ab.verdict.v2"
    recs = [r for r in records if r["stage"] <= stage_upto and not str(r.get("end_reason", "")).startswith("llm_error")]
    idx = _idx(recs)
    out["paired_success_tokens"] = {}
    for m in out["models"]:
        for cname, (xa, ya) in CONTRASTS_V2.items():
            for sname, pred in score.STRATA.items():
                pairs = [(x, idx[(m, tid, sd, ya)]) for (mm, tid, sd, arm), x in idx.items()
                         if mm == m and arm == xa and pred(x) and (m, tid, sd, ya) in idx]
                if pairs:
                    out["paired_success_tokens"][f"{m}|{cname}|{sname}"] = paired_success_token_ratio(pairs)
    out["interim"] = {"n20": interim(out, "n_le_20"), "n50": interim(out, "n_eq_50", n50=True)}
    out["harm"] = {m: harm_check(recs, m) for m in out["models"]}
    out["end_reasons"] = reasons_table(recs)
    return out


def reasons_table(recs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per arm: counts of the per-turn failure-reason enum (transcript-derived, written by arms.run_episode)."""
    t: dict[str, dict[str, int]] = {}
    for r in recs:
        for k, v in (r.get("turn_failure_reasons") or {}).items():
            t.setdefault(r["arm"], {})[k] = t.setdefault(r["arm"], {}).get(k, 0) + int(v)
    return t


def interim(res: dict[str, Any], stratum: str, n50: bool = False) -> dict[str, Any]:
    """Futility-only at interim looks (formal-gate rule). ``stop_on_futility`` stops the WHOLE v2: B1 vs B shows |d_success| < 0.05 for every
    scored model (the diagnosis was wrong: re-run the taxonomy on the transcripts first). ``drop_b2``: B2 vs B1 futile on every scored model.
    N=50: ``c3_vs_apick_futile``: |d_success| < 0.05 AND |median ln token ratio on paired successes| < ln 1.10. Stop-on-answer is
    descriptive only (grade descriptive_interim), never a gate claim."""
    def d(model: str, c: str) -> dict[str, Any] | None:
        x = res["cells"].get(f"{model}|{c}|{stratum}")
        return None if not x or x.get("verdict") == "underpowered" else x

    ms = res["models"]
    q1 = [d(m, "Q1a") for m in ms]
    q2 = [d(m, "Q2") for m in ms]
    out: dict[str, Any] = {
        "models_scored_q1a": sum(1 for x in q1 if x),
        "stop_on_futility": any(q1) and all(abs(x["d_success"]) < FUT_SUCC for x in q1 if x),
        "drop_b2": any(q2) and all(abs(x["d_success"]) < FUT_SUCC for x in q2 if x),
    }
    qp: dict[str, bool] = {}
    for m in ms:  # per-model QP futility: HARD adds nothing measurable (tokens on paired successes, wall, success)
        x = d(m, "QP")
        ps = res["paired_success_tokens"].get(f"{m}|QP|{stratum}", {})
        if x and ps.get("ln_ratio_median") is not None:
            lim = math.log(1 + RATIO_EPS)
            qp[m] = abs(x["d_success"]) < FUT_SUCC and abs(ps["ln_ratio_median"]) < lim and abs(x["ln_ratio_wall_median"]) < lim
    out["qp_futile_by_model"] = qp
    if n50:
        fut = []
        for m in ms:
            x = d(m, "Q3")
            ps = res["paired_success_tokens"].get(f"{m}|Q3|{stratum}", {})
            if x and ps.get("ln_ratio_median") is not None:
                fut.append(abs(x["d_success"]) < FUT_SUCC and abs(ps["ln_ratio_median"]) < math.log(1 + RATIO_EPS))
        out["c3_vs_apick_futile"] = bool(fut) and all(fut)
        ans = []
        for m in ms:
            x = d(m, "Q3")
            if x:
                signs = set(x["seed_signs"].values())
                ans.append((x["d_success_ci95"][0] > 0 or x["d_success_ci95"][1] < 0) and len(signs - {0}) == 1 and 0 not in signs)
        out["stop_on_answer_descriptive"] = bool(ans) and all(ans)
        out["grade"] = "descriptive_interim"
    return out


def harm_check(recs: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Harm stops for the hardened picker on the direct-call path (A_pick vs A). ``harm`` True => revert that model/role to picker OFF."""
    ap = [r for r in recs if r["model"] == model and r["arm"] == "A_pick"]
    idx = _idx([r for r in recs if r["model"] == model])
    pos = neg = 0
    for r in ap:
        a = idx.get((model, r["task_id"], r["seed"], "A"))
        if a is not None and bool(r["success"]) != bool(a["success"]):
            pos += bool(r["success"])
            neg += bool(a["success"])
    d_succ = (sum(bool(r["success"]) for r in ap) / len(ap) - statistics.fmean(
        [bool(idx[(model, r["task_id"], r["seed"], "A")]["success"]) for r in ap if (model, r["task_id"], r["seed"], "A") in idx] or [0])) if ap else 0.0
    sels = [r.get("selector") or {} for r in ap]
    pr = [s["proxy_recall"] for s in sels if s.get("proxy_recall") is not None]
    lat = sorted(float(s.get("elapsed_ms", 0.0)) for s in sels if s)
    p95 = lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)] if lat else 0.0
    rec_rate = statistics.fmean([1.0 if s.get("recovery_fired") else 0.0 for s in sels]) if sels else 0.0
    opens = max([int(s.get("breaker_opens", 0)) for s in sels] or [0])
    checks = {
        "success_drop": bool(ap) and d_succ <= -HARM_SUCC and neg >= HARM_MIN_DISCORDANT and pos == 0,
        "proxy_recall_low": len(pr) >= HARM_MIN_EPISODES and statistics.fmean(pr) < HARM_RECALL,
        "recovery_high": len(sels) >= HARM_MIN_EPISODES and rec_rate > HARM_RECOVERY,
        "p95_latency": bool(lat) and p95 > HARM_P95_MS,
        "breaker_opens": opens > HARM_BREAKER_OPENS,
    }
    return {"n_a_pick": len(ap), "d_success": d_succ, "discordant_neg": neg, "discordant_pos": pos,
            "proxy_recall_mean": statistics.fmean(pr) if pr else None, "recovery_rate": rec_rate, "p95_picker_ms": p95,
            "breaker_opens": opens, "checks": checks, "harm": any(checks.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--stage-upto", type=int, default=3)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    text = json.dumps(evaluate(load(a.results, a.mode), a.stage_upto), indent=1, sort_keys=True, default=str)
    if a.out:
        Path(a.out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
