"""Replay-flip report with the three mandatory control arms (HIL-16).

A bare outcome flip is weak evidence of decisiveness (intake-2335#record: placebo and wrong-step
interventions corrected a large share of runs). The report therefore carries targeted, no_hint,
placebo and wrong_step side by side and refuses to call a step decisive unless the targeted
flip-rate beats ALL controls. Denominators are every sample STARTED; errors count as non-flips and
are shown beside the rates.
"""
from __future__ import annotations

import json
from math import comb
from typing import Any

from src.llm_primitives.stat_tests import wilson_interval

SCHEMA = "replay_flip_report/v1"
ARMS = ("targeted", "no_hint", "placebo", "wrong_step")
CONTROLS = ("no_hint", "placebo", "wrong_step")


def fisher_greater(a: int, n1: int, b: int, n2: int) -> float:
    """One-sided Fisher exact p for rate(a/n1) > rate(b/n2)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    k, n = a + b, n1 + n2
    tot = comb(n, k)
    return min(1.0, sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot)


def _arm_row(arm: dict, samples: list[dict]) -> dict:
    n = len(samples)
    passes = sum(s["result"] == "pass" for s in samples)
    lo, hi = wilson_interval(passes, n)
    return {"restart_turn": arm["restart_turn"], "hint_chars": arm.get("hint_chars"), "n_started": n,
            "n_pass": passes, "n_fail": sum(s["result"] == "fail" for s in samples),
            "n_error": sum(s["result"] == "error" for s in samples),
            "flip_rate": passes / n if n else None, "wilson95": [lo, hi]}


def build_report(*, run_id: str, task_key: str, original_outcome: str, restart_turn: int, hint: str,
                 arms: dict[str, dict], samples: dict[str, list[dict]], alpha: float = 0.05,
                 label: str = "measurement") -> dict[str, Any]:
    """``arms``: name -> {"restart_turn", "hint_chars"}; ``samples``: name -> per-sample dicts."""
    rows = {a: _arm_row(arms[a], samples[a]) for a in ARMS if a in samples}
    missing = [a for a in ARMS if a not in rows]
    contrasts: dict[str, dict] = {}
    if "targeted" in rows:
        t = rows["targeted"]
        for c in CONTROLS:
            if c in rows:
                r = rows[c]
                contrasts[c] = {"diff": (t["flip_rate"] or 0) - (r["flip_rate"] or 0),
                                "fisher_one_sided_p": fisher_greater(t["n_pass"], t["n_started"], r["n_pass"], r["n_started"])}
    allowed = not missing and original_outcome == "fail"
    decisive = allowed and all(contrasts[c]["diff"] > 0 and contrasts[c]["fisher_one_sided_p"] < alpha for c in CONTROLS)
    if missing:
        verdict = "no_claim: control arms missing: " + ",".join(missing)
    elif original_outcome != "fail":
        verdict = "no_claim: original run did not fail"
    elif decisive:
        verdict = "targeted step flips outcome beyond no_hint, placebo and wrong_step"
    else:
        verdict = "not distinguished from controls: a flip here is not evidence the step was decisive"
    return {"schema": SCHEMA, "label": label, "run_id": run_id, "task_key": task_key,
            "original_outcome": original_outcome, "restart_turn": restart_turn, "hint_sha_chars": len(hint),
            "alpha": alpha, "arms": rows, "contrasts_vs_targeted": contrasts, "claim_allowed": allowed,
            "decisive": decisive, "verdict": verdict,
            "denominator_policy": "all samples started; errors counted as non-flips and reported"}


def dumps(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"
