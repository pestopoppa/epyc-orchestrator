"""The L2 scan report. Reads only status, exception, category and count fields (plus declared step fields)."""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from statistics import mean

from src.trace.l2.detectors import (
    CATEGORIES, JUDGE_ONLY, DetectorConfig, Detection, counter_source, detect,
    first_visible_deviation, loop_ratio,
)
from src.trace.l2.model import L2Trace

HEADER = (
    "This report reads only status, exception, category and count fields (and the declared per-step "
    "structured fields). There is no decision-to-provenance edge in the recorded traces "
    "(intake-2315#record): each 'first visible deviation' is a first-event label, not a "
    "counterfactual root cause."
)

# HIL-12 item (3) crosswalk. EPYC additions are marked; loop, truncation and premature wrap-up
# have no AgentRx leaf.
CROSSWALK = {
    "tool_error": {"trail": ["Rate Limiting", "Authentication Errors", "Service Errors", "Resource Not Found", "Timeout Issues"], "agentrx": ["System Failure"], "epyc_addition": False},
    "truncation": {"trail": ["Resource Exhaustion"], "agentrx": [], "epyc_addition": True},
    "loop": {"trail": ["Resource Abuse"], "agentrx": [], "epyc_addition": True},
    "premature_wrap_up": {"trail": ["Goal Deviation"], "agentrx": [], "epyc_addition": True},
    "zero_step_violation": {"trail": ["Instruction Non-compliance"], "agentrx": ["Instruction/Plan Adherence Failure"], "epyc_addition": False},
    "invalid_invocation": {"trail": ["Formatting Errors"], "agentrx": ["Invalid Invocation"], "epyc_addition": False},
    "fabricated_fact": {"trail": ["Language-only", "Tool-related"], "agentrx": [], "epyc_addition": False, "judge_only": True},
    "instruction_not_followed": {"trail": ["Instruction Non-compliance"], "agentrx": ["Instruction/Plan Adherence Failure"], "epyc_addition": False, "judge_only": True},
}

_INFRA = re.compile(r"connection (refused|reset)|immediate_failure|oom|out of memory|no space|unreachable|502|503|server (down|error)", re.I)
_MODEL_API = re.compile(r"request_timeout|rate.?limit|429|context_length|invalid_request|model.*(unavailable|overloaded)|api", re.I)
_AGENT = re.compile(r"plan_rejected|step cap|max_(turns|steps)|protocol|loop halted|tool|incomplete|truncat", re.I)
_OK_STATUS = {None, "", "ok", "success", "completed", "complete"}


def domain_of(t: L2Trace) -> str | None:
    """infra | model_api | agent | none_wrong_outcome | unattributed; None for non-failed traces."""
    if t.outcome != "fail":
        return None
    text = f"{t.status or ''} {t.exception or ''}"
    if not t.exception and (t.status or None) in _OK_STATUS:
        return "none_wrong_outcome"
    for dom, rx in (("infra", _INFRA), ("model_api", _MODEL_API), ("agent", _AGENT)):
        if rx.search(text):
            return dom
    return "unattributed"


def _ci(values: list[tuple[str, float]], rng: random.Random, B: int) -> list[float | None]:
    """Percentile interval resampling TASKS (not runs): values are (task_key, 0/1)."""
    by: dict[str, list[float]] = defaultdict(list)
    for k, v in values:
        by[k].append(v)
    tasks = list(by)
    if len(tasks) < 2:
        return [None, None]
    stats = []
    for _ in range(B):
        pick = [rng.choice(tasks) for _ in tasks]
        flat = [v for k in pick for v in by[k]]
        stats.append(sum(flat) / len(flat))
    stats.sort()
    return [round(stats[int(0.025 * B)], 4), round(stats[min(B - 1, int(0.975 * B))], 4)]


def _quant(xs: list[int]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]
    return {"n": len(s), "mean": round(mean(s), 3), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9)}


def scan(traces: list[L2Trace], cfg: DetectorConfig | None = None, seed: int = 0, B: int = 1000) -> dict:
    cfg = cfg or DetectorConfig()
    rng = random.Random(seed)
    # Traces with no observed steps (task-level logs) never support zero-step claims:
    # steps are unobserved, not absent.
    dets: dict[str, list[Detection]] = {
        t.trace_id: [d for d in detect(t, cfg) if t.steps or d.category != "zero_step_violation"]
        for t in traces
    }
    failed = [t for t in traces if t.outcome == "fail"]
    passed = [t for t in traces if t.outcome == "pass"]
    observed = lambda t: bool(t.steps)

    rep: dict = {
        "header": HEADER,
        "reads": ["outcome", "status", "exception", "role", "model_pin", "arm", "task_key", "step_cap", "ctx_limit", "step fields in l2.model.STEP_FIELDS"],
        "loop_counter_source": counter_source(),
        "n": {"traces": len(traces), "failed": len(failed), "passed": len(passed), "unknown_outcome": len(traces) - len(failed) - len(passed)},
        "categories_implemented": list(CATEGORIES),
        "categories_judge_only_unimplemented": list(JUDGE_ONLY),
        "crosswalk": CROSSWALK,
    }
    # (1) domain split
    dom = Counter(domain_of(t) for t in failed)
    rep["domain_split"] = {"denominator_failed": len(failed), "counts": dict(sorted(dom.items()))}

    def fires(t, cat):
        if cat == "zero_step_violation" and not observed(t):
            return None   # unobserved, not absent
        return any(d.category == cat for d in dets[t.trace_id])

    # (4) category rates on failed AND passed, task-resampled intervals
    rates = {}
    for cat in CATEGORIES:
        row = {}
        for name, grp in (("failed", failed), ("passed", passed)):
            vals = [(t.task_key, 1.0 if fires(t, cat) else 0.0) for t in grp if fires(t, cat) is not None]
            n, k = len(vals), int(sum(v for _, v in vals))
            row[name] = {"n": n, "fires": k, "rate": round(k / n, 4) if n else None, "ci95_task_resampled": _ci(vals, rng, B) if n else [None, None]}
        rates[cat] = row
    rep["category_rates"] = rates

    # (3) mechanical-signature share and unclassified share (no forced label)
    def has_sig(t):
        return any(d.category != "zero_step_violation" or observed(t) for d in dets[t.trace_id])
    sig = [t for t in failed if has_sig(t)]
    rep["coverage"] = {
        "denominator_failed": len(failed),
        "mechanical_signature": len(sig),
        "share_mechanical_signature": round(len(sig) / len(failed), 4) if failed else None,
        "unclassified": len(failed) - len(sig),
        "share_unclassified": round(1 - len(sig) / len(failed), 4) if failed else None,
    }
    # first visible deviation
    fvd = Counter()
    first_eq_last = 0
    with_dets = 0
    for t in failed:
        ds = dets[t.trace_id]
        d = first_visible_deviation(ds)
        if d is None:
            continue
        with_dets += 1
        fvd[d.category] += 1
        if len({x.step_idx for x in ds}) == 1:
            first_eq_last += 1
    rep["first_visible_deviation"] = {"n_failed_with_event": with_dets, "by_category": dict(sorted(fvd.items())), "first_equals_last_event": first_eq_last}
    # length distributions + step ratio per role
    steps_of = lambda t: len(t.tool_steps())
    rep["length_distribution"] = {
        "failed": _quant([steps_of(t) for t in failed if observed(t)]),
        "passed": _quant([steps_of(t) for t in passed if observed(t)]),
    }
    roles = sorted({t.role for t in traces if t.role is not None} | ({None} if any(t.role is None for t in traces) else set()), key=str)
    ratio = {}
    caps = sorted({t.step_cap for t in traces if t.step_cap})
    for r in roles:
        f = [steps_of(t) for t in failed if t.role == r and steps_of(t) > 0]
        p = [steps_of(t) for t in passed if t.role == r and steps_of(t) > 0]
        ratio[str(r)] = {"failed_n": len(f), "passed_n": len(p), "ratio_failed_over_passed": round(mean(f) / mean(p), 4) if f and p else None}
    rep["step_ratio_per_role"] = {"rows": ratio, "step_cap": caps or None, "zero_step_traces_excluded": True}
    # loop-ratio per (role, model_pin, arm); never pooled across arms; within-task pairs listed
    lr: dict[str, list[float]] = defaultdict(list)
    per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in traces:
        v = loop_ratio(t)
        if v is None:
            continue
        lr[f"{t.role}|{t.model_pin}|{t.arm}"].append(v)
        per_task[t.task_key][str(t.arm)].append(v)
    rep["loop_ratio"] = {
        "by_role_model_arm": {k: {"n": len(v), "mean": round(mean(v), 4)} for k, v in sorted(lr.items())},
        "within_task_paired_arms": {k: {a: round(mean(v), 4) for a, v in sorted(arms.items())} for k, arms in sorted(per_task.items()) if len(arms) > 1},
        "note": "read within task across paired arms; never pooled with-loop versus without-loop (intake-2338#record)",
    }
    return rep


def dumps(rep: dict) -> str:
    """Canonical serialisation: byte-identical for identical declared inputs."""
    return json.dumps(rep, sort_keys=True, indent=1, default=str)
