"""HIL-14: calibrate the L2 rule detectors against external gold and fixtures. JSON/JSONL only, no model."""
from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.trace.l2.detectors import (
    DetectorConfig, baseline_first_error_step, baseline_last_step, detect, first_visible_deviation,
)
from src.trace.l2.loaders import load_claude_session, load_whowhen_pro
from src.trace.l2.model import L2Step, L2Trace

MIN_N = 10   # categories under this N are reported unscored


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": None if p is None else round(p, 4),
            "recall": None if r is None else round(r, 4)}


def score_binary(pred: list[bool], gold: list[bool], label: str) -> dict:
    n_pos = sum(gold)
    out = {"label": label, "n_total": len(gold), "n_positive": n_pos}
    if n_pos < MIN_N:
        out["scored"] = False
        out["reason"] = f"N<{MIN_N}"
        return out
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    out.update(scored=True, **prf(tp, fp, fn))
    return out


def best_length_baseline(lengths: list[int], gold: list[bool]) -> dict:
    """Construction-only baseline: the best single threshold on step count (either direction)."""
    best = None
    for thr in sorted(set(lengths)):
        for sign in ("<=", ">="):
            pred = [(l <= thr) if sign == "<=" else (l >= thr) for l in lengths]
            tp = sum(p and g for p, g in zip(pred, gold)); fp = sum(p and not g for p, g in zip(pred, gold))
            fn = sum((not p) and g for p, g in zip(pred, gold))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if best is None or f1 > best[0]:
                best = (f1, thr, sign, tp, fp, fn)
    if best is None:
        return {}
    f1, thr, sign, tp, fp, fn = best
    return {"rule": f"n_steps {sign} {thr}", "f1": round(f1, 4), **prf(tp, fp, fn)}


# ---------------------------------------------------------------- Who&When Pro
PRO_TARGETS = {"loop": ("A.4", "loop"), "premature_wrap_up": ("A.3", "premature_wrap_up"),
               "tool_error": ("A.1", "tool_error"), "output_format": ("A.2", "invalid_invocation")}


def whowhen_pro_table(path, cfg: DetectorConfig | None = None) -> dict:
    """Detector vs injected-mode label. Tests a detector on INJECTED behaviour, not natural frequency.

    Every trace is a failure by construction; for target mode M the positives are traces labelled M
    and all other traces are negatives (they may fire for real unrelated reasons, which counts as
    a false positive here and is stated).
    """
    cfg = cfg or DetectorConfig()
    rows = [(t, gt, detect(t, cfg)) for t, gt in load_whowhen_pro(path)]
    mode_n = Counter(gt.get("mode") for _, gt, _ in rows)
    lengths = [len(t.tool_steps()) for t, _, _ in rows]
    res = {"n_traces": len(rows), "per_mode_N": dict(sorted(mode_n.items())), "targets": {},
           "scope": "injected behaviour, not natural frequency"}
    for name, (mode, cat) in PRO_TARGETS.items():
        gold = [gt.get("mode") == mode for _, gt, _ in rows]
        pred = [any(d.category == cat for d in ds) for _, _, ds in rows]
        res["targets"][name] = {
            "mode": mode, "detector": cat, "detector_score": score_binary(pred, gold, name),
            "construction_only_baseline": best_length_baseline(lengths, gold) if sum(gold) >= MIN_N else None,
        }
    return res


# ----------------------------------------------------------------- TraceElephant
def _lit(s):
    if isinstance(s, (list, dict)) or s is None:
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return None


_TOOL_ERR = re.compile(r"^(error|traceback)|exit code [1-9]|command not found|no such file|\bTimeoutError\b|\[error\]", re.I)


def traceelephant_trace(run_dir: Path) -> tuple[L2Trace, dict]:
    meta = json.loads((run_dir / "trace_metadata.json").read_text())
    recs = json.loads((run_dir / "step_records.json").read_text())
    steps: list[L2Step] = []
    inputs = []
    for r in recs:
        out, inp = _lit(r.get("output")) or {}, _lit(r.get("input")) or {}
        inputs.append(inp if isinstance(inp, dict) else {})
        ch = (out.get("choices") or [{}])[0] if isinstance(out, dict) else {}
        msg = ch.get("message") or {}
        tcs = msg.get("tool_calls") or []
        fr = ch.get("finish_reason")
        s = L2Step(idx=int(r["step_id"]), kind="tool_call" if tcs else "message", finish_reason=fr)
        if tcs:
            fn = (tcs[0].get("function") or {})
            s.tool = fn.get("name")
            raw = fn.get("arguments")
            try:
                a = json.loads(raw) if isinstance(raw, str) else raw
                s.args = a if isinstance(a, dict) else None
                if s.args is None:
                    s.args_raw = str(raw)
            except Exception:
                s.args_raw = str(raw)
        steps.append(s)
    # a step's observation is the tool result the NEXT record received (derived flag, named here)
    for i, s in enumerate(steps[:-1]):
        msgs = inputs[i + 1].get("messages") or []
        tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
        if s.kind == "tool_call" and tool_msgs:
            c = tool_msgs[-1].get("content")
            s.observation = c if isinstance(c, str) else json.dumps(c)
            s.error = bool(_TOOL_ERR.search(s.observation.strip()))
    t = L2Trace(trace_id=meta.get("run_id", run_dir.name), task_key=str(meta.get("task_id")), outcome="fail",
                source="traceelephant", role=meta.get("system_name"), steps=steps)
    return t, meta


def _agree(pred: int | None, gold: int) -> dict:
    if pred is None:
        return {"exact": False, "within1": False, "within5": False}
    d = abs(pred - gold)
    return {"exact": d == 0, "within1": d <= 1, "within5": d <= 5}


def traceelephant_table(data_dir, cfg: DetectorConfig | None = None) -> dict:
    cfg = cfg or DetectorConfig()
    agg: dict[str, Counter] = defaultdict(Counter)
    fired_cats = Counter()
    n = 0
    for run in sorted(Path(data_dir).glob("*/*/trace_metadata.json")):
        t, meta = traceelephant_trace(run.parent)
        try:
            gold = int(float(meta.get("mistake_step")))
        except (TypeError, ValueError):
            continue
        n += 1
        ds = [d for d in detect(t, cfg) if d.step_idx is not None]
        fired_cats.update({d.category for d in ds})
        fvd = first_visible_deviation(ds)
        sys = t.role or "?"
        for key, pred in (("detector_fvd", fvd.step_idx if fvd else None),
                          ("baseline_first_error", baseline_first_error_step(t)),
                          ("baseline_last_step", baseline_last_step(t))):
            for grp in (sys, "ALL"):
                a = _agree(pred, gold)
                c = agg[f"{grp}|{key}"]
                c["n"] += 1
                c["has_prediction"] += pred is not None
                for k, v in a.items():
                    c[k] += v
        if fvd is not None:
            idxs = {d.step_idx for d in ds}
            for grp in (sys, "ALL"):
                agg[f"{grp}|detector_fvd"]["first_equals_last"] += len(idxs) == 1
    return {"n_traces": n, "gold": "human decisive step (mistake_step)", "detector_categories_fired": dict(fired_cats),
            "detectors_firing_on_none": [c for c in ("tool_error", "truncation", "loop", "premature_wrap_up", "invalid_invocation") if c not in fired_cats],
            "agreement": {k: dict(v) for k, v in sorted(agg.items())},
            "note": "zero_step_violation is trace-level (no step index) and excluded from step agreement"}


# --------------------------------------------------------------------- TRAIL
EXEC_LEAVES = {
    "api_issues": ["Rate Limiting", "Authentication Errors", "Service Errors", "Resource Not Found"],
    "timeout": ["Timeout Issues"], "resource_exhaustion": ["Resource Exhaustion"],
    "formatting": ["Formatting Errors"],
}


def trail_tabulate(ann_dirs: dict[str, Path]) -> dict:
    """Counts per leaf per split from processed_annotations_* JSON ({"errors": [{"category", ...}]})."""
    out = {}
    for split, d in ann_dirs.items():
        c, n = Counter(), 0
        for f in sorted(Path(d).glob("*.json")):
            n += 1
            for e in json.loads(f.read_text()).get("errors", []):
                c[e.get("category")] += 1
        out[split] = {"n_traces": n, "per_leaf": dict(sorted(c.items(), key=lambda kv: -kv[1]))}
    return out


# ------------------------------------------------------------ KernelBench-Hard
def kbh_firing(files: list[Path], cfg: DetectorConfig | None = None) -> dict:
    """Per-detector firing counts on natural traces; no outcome label is read."""
    cfg = cfg or DetectorConfig()
    fires, per = Counter(), []
    for f in files:
        t = load_claude_session(f)
        cats = {d.category for d in detect(t, cfg)}
        fires.update(cats)
        per.append({"file": Path(f).name, "n_tool_steps": len(t.tool_steps()), "categories": sorted(cats)})
    allc = ("tool_error", "truncation", "loop", "premature_wrap_up", "zero_step_violation", "invalid_invocation")
    return {"n_files": len(files), "firing_counts": {c: fires.get(c, 0) for c in allc},
            "detectors_firing_on_none": [c for c in allc if fires.get(c, 0) == 0], "per_file": per}
