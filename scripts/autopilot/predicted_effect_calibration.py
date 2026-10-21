#!/usr/bin/env python3
"""MHS-17 — predicted-effect calibration readout (read-only, no LLM, no inference).

Scores the MHS-14 annotation (`predicted_fixed`, `predicted_at_risk`) against what a
mutation actually did (observed fail-to-pass and pass-to-fail task sets) with plain set
intersection, then aggregates per round beside a random-prediction baseline.

Why the baseline: intake-1132#record found at-risk prediction near random, so a
precision number means nothing without the number a random predictor of the same size
would have scored on the same panel. For a panel of n tasks, a predictor that names k
tasks at random, and a true set of size F (the observed flip count):

    expected precision = F / n          (each random pick is a true flipper w.p. F/n)
    expected recall    = k / n          (a random k-set covers k/n of any fixed set)

Both are computed from panel size and the observed flip rate, nothing else.

Row shape (one per mutation; extra keys ignored). Fields may sit at the top level, under
`predicted_effect`, or under `eval_details` (journal row) / `eval_details.predicted_effect`:

    mutation_id, round (or epoch/trial), panel_size (or panel: [task ids]),
    predicted_fixed, predicted_at_risk, observed_fail_to_pass, observed_pass_to_fail

Rows without any predicted field are counted as `unannotated`, never dropped. A pure
module: it reads nothing from disk unless `main()` is given a JSONL path.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Iterable, Mapping

__all__ = ["score_mutation", "calibration_report", "extract_row", "main"]

_PRED = ("predicted_fixed", "predicted_at_risk")
_OBS = ("observed_fail_to_pass", "observed_pass_to_fail")


def _layers(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = [row]
    for key in ("predicted_effect",):
        if isinstance(row.get(key), Mapping):
            out.append(row[key])
    ed = row.get("eval_details")
    if isinstance(ed, Mapping):
        out.append(ed)
        if isinstance(ed.get("predicted_effect"), Mapping):
            out.append(ed["predicted_effect"])
    return out


def _get(row: Mapping[str, Any], *names: str) -> Any:
    for layer in _layers(row):
        for n in names:
            if n in layer and layer[n] is not None:
                return layer[n]
    return None


def extract_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise one journal / AP-53 row. `annotated` is False when no prediction exists."""
    pf, pr = _get(row, "predicted_fixed"), _get(row, "predicted_at_risk")
    f2p, p2f = _get(row, "observed_fail_to_pass"), _get(row, "observed_pass_to_fail")
    panel = _get(row, "panel")
    n = _get(row, "panel_size")
    if n is None and panel is not None:
        n = len(set(panel))
    rnd = _get(row, "round", "epoch", "trial")
    return {
        "mutation_id": _get(row, "mutation_id", "id", "trial_id"),
        "round": rnd,
        "panel_size": int(n) if n is not None else None,
        "predicted_fixed": set(pf or ()),
        "predicted_at_risk": set(pr or ()),
        "observed_fail_to_pass": set(f2p or ()),
        "observed_pass_to_fail": set(p2f or ()),
        "annotated": pf is not None or pr is not None,
        "observed_present": f2p is not None and p2f is not None,
    }


def _ratio(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def score_mutation(x: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic per-mutation verdict by set intersection (no LLM)."""
    pf, pr = x["predicted_fixed"], x["predicted_at_risk"]
    f2p, p2f = x["observed_fail_to_pass"], x["observed_pass_to_fail"]
    fix_hit = pf & f2p
    reg_hit = pr & p2f
    unattributed = sorted(p2f - pr)
    if p2f and not unattributed:
        reg_verdict = "attributed"
    elif p2f:
        reg_verdict = "unattributed" if not reg_hit else "partly_attributed"
    else:
        reg_verdict = "no_regression"
    if not f2p and not pf:
        fix_verdict = "no_fix_predicted_or_observed"
    elif fix_hit and fix_hit == f2p == pf:
        fix_verdict = "exact"
    elif fix_hit:
        fix_verdict = "partial"
    elif pf and not f2p:
        fix_verdict = "predicted_not_observed"
    elif f2p and not pf:
        fix_verdict = "observed_not_predicted"
    else:
        fix_verdict = "miss"
    return {
        "mutation_id": x["mutation_id"],
        "round": x["round"],
        "fix_verdict": fix_verdict,
        "regression_verdict": reg_verdict,
        "fix_hits": sorted(fix_hit),
        "regression_hits": sorted(reg_hit),
        "unattributed_regressions": unattributed,
    }


def _agg(rows: list[dict[str, Any]], pred_key: str, obs_key: str) -> dict[str, Any]:
    tp = sum(len(r[pred_key] & r[obs_key]) for r in rows)
    npred = sum(len(r[pred_key]) for r in rows)
    nobs = sum(len(r[obs_key]) for r in rows)
    # Random baseline per mutation, summed (expected hits), then divided by the same totals.
    exp_tp = 0.0
    sized = True
    for r in rows:
        n = r["panel_size"]
        if not n:
            sized = False
            continue
        exp_tp += len(r[pred_key]) * len(r[obs_key]) / n  # k * F / n
    return {
        "predicted": npred,
        "observed": nobs,
        "hits": tp,
        "precision": _ratio(tp, npred),
        "recall": _ratio(tp, nobs),
        "random_expected_hits": exp_tp if sized else None,
        "random_precision": _ratio(exp_tp, npred) if sized else None,
        "random_recall": _ratio(exp_tp, nobs) if sized else None,
        "baseline_complete": sized,
    }


def calibration_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-mutation verdicts, per-round precision/recall beside random, unattributed list."""
    total = 0
    unannotated = 0
    no_observed = 0
    scored: list[dict[str, Any]] = []
    for raw in rows:
        total += 1
        x = extract_row(raw)
        if not x["annotated"]:
            unannotated += 1
            continue
        if not x["observed_present"]:
            no_observed += 1
            continue
        scored.append(x)
    by_round: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for x in scored:
        by_round[x["round"]].append(x)
    rounds = {}
    for rnd in sorted(by_round, key=lambda v: (v is None, str(v))):
        rs = by_round[rnd]
        rounds[str(rnd)] = {
            "mutations": len(rs),
            "fixes": _agg(rs, "predicted_fixed", "observed_fail_to_pass"),
            "regressions": _agg(rs, "predicted_at_risk", "observed_pass_to_fail"),
        }
    verdicts = [score_mutation(x) for x in scored]
    unattributed = [
        {"mutation_id": v["mutation_id"], "round": v["round"], "tasks": v["unattributed_regressions"]}
        for v in verdicts
        if v["unattributed_regressions"]
    ]
    return {
        "rows_total": total,
        "rows_unannotated": unannotated,
        "rows_annotated_without_observed": no_observed,
        "rows_scored": len(scored),
        "verdicts": verdicts,
        "rounds": rounds,
        "overall": {
            "fixes": _agg(scored, "predicted_fixed", "observed_fail_to_pass"),
            "regressions": _agg(scored, "predicted_at_risk", "observed_pass_to_fail"),
        },
        "unattributed regressions": unattributed,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: predicted_effect_calibration.py ROWS.jsonl", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    json.dump(calibration_report(rows), sys.stdout, indent=2, sort_keys=True, default=list)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
