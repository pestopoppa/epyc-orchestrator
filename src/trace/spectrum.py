"""HIL-18: spectrum-style across-run attribution over replays of ONE task.

Element = step-type key ``(role, tool, exit_status)``. Run spectrum = the set of keys the run
executed (presence, not count). With N replays of one task at temperature > 0 labelled pass/fail by
the existing verifier, each key is ranked by Kulczynski2:

    K2 = 0.5 * ( ef/(ef+nf) + ef/(ef+ep) )

ef/nf = failed runs that did/did not execute the key; ep = passed runs that did. A key every failing
run executes and no passing run executes scores 1.0. This is a RANKING of co-occurrence with failure
over replays; it is not a causal claim (that is HIL-16's job, with control arms).

The key is deterministic. A step whose exit status cannot be determined (no exit code, no error
flag) gets status ``unknown`` and is AMBIGUOUS; ``resolve_ambiguous`` is the only place a local
model may be consulted, and without one the ambiguous keys stay labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from src.trace.l2.detectors import DetectorConfig, detect
from src.trace.l2.model import L2Step, L2Trace

Key = tuple[str, str, str]
UNKNOWN = "unknown"


def exit_status(step: L2Step) -> str:
    if step.exit_status is not None:
        return f"exit:{step.exit_status}"
    if step.error is True:
        return f"error:{step.error_class or 'error'}"
    if step.error is False:
        return "ok"
    return UNKNOWN


def step_key(step: L2Step, role: str | None) -> Key:
    return (role or "?", step.tool or f"<{step.kind}>", exit_status(step))


def trace_keys(trace: L2Trace) -> set[Key]:
    return {step_key(s, trace.role) for s in trace.steps if s.kind in ("tool_call", "final", "message")}


def kulczynski2(ef: int, nf: int, ep: int) -> float:
    if ef == 0:
        return 0.0
    return 0.5 * (ef / (ef + nf) + ef / (ef + ep))


@dataclass(frozen=True)
class Ranked:
    key: Key
    score: float
    ef: int
    nf: int
    ep: int
    ambiguous: bool


@dataclass
class TaskRanking:
    task_key: str
    n_runs: int
    n_fail: int
    n_pass: int
    ranking: list[Ranked] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return 0 < self.n_pass and 0 < self.n_fail


def rank_task(traces: Sequence[L2Trace], task_key: str | None = None) -> TaskRanking:
    """Rank keys for replays of one task. Traces with unknown outcome are excluded and counted nowhere."""
    labelled = [t for t in traces if t.outcome in ("pass", "fail")]
    fails = [trace_keys(t) for t in labelled if t.outcome == "fail"]
    passes = [trace_keys(t) for t in labelled if t.outcome == "pass"]
    out = TaskRanking(task_key or (traces[0].task_key if traces else ""), len(labelled), len(fails), len(passes))
    keys = set().union(*fails, *passes) if labelled else set()
    rows = []
    for k in keys:
        ef = sum(k in s for s in fails)
        ep = sum(k in s for s in passes)
        rows.append(Ranked(k, kulczynski2(ef, len(fails) - ef, ep), ef, len(fails) - ef, ep, k[2] == UNKNOWN))
    rows.sort(key=lambda r: (-r.score, -r.ef, r.ep, r.key))   # deterministic tie-break
    out.ranking = rows
    return out


def eligible_for_spectrum(n_pass: int, n_fail: int, min_replays: int = 20) -> bool:
    """Pass-rate strictly between 0 and 1 over >= min_replays replays."""
    return n_pass + n_fail >= min_replays and n_pass > 0 and n_fail > 0


def ambiguous_keys(ranking: TaskRanking) -> list[Key]:
    return [r.key for r in ranking.ranking if r.ambiguous]


def relabel_ambiguous(traces: Sequence[L2Trace], judge: Callable[[L2Trace, L2Step], bool]) -> list[L2Trace]:
    """Return copies where each status-unknown step gets ``error`` from ``judge(trace, step)`` (True = error).

    The judge (a local model, wired by the HIL-18 run) is called ONLY for steps the deterministic key
    cannot classify; every other step is untouched. Re-run ``rank_task`` on the result.
    """
    import dataclasses

    out = []
    for t in traces:
        steps = [dataclasses.replace(s, error=bool(judge(t, s))) if s.exit_status is None and s.error is None
                 and s.kind == "tool_call" else s for s in t.steps]
        out.append(dataclasses.replace(t, steps=steps))
    return out


# --------------------------------------------------------------------------- agreement with L2
def l2_agreement(traces: Sequence[L2Trace], ranking: TaskRanking, top_m: int = 3,
                 cfg: DetectorConfig | None = None) -> dict[str, Any]:
    """Does the top-m spectrum key coincide with a step the L2 detectors flag in failed runs?

    Per L2 category: ``failed_runs_flagged`` (detector fired at a step), ``top_key_at_flagged_step``
    (the flagged step's key is in the top-m). Failed runs with no L2 detection are the
    ``unclassified`` row, the denominator HIL-17 consumes.
    """
    top = {r.key for r in ranking.ranking[:top_m]}
    cats: dict[str, dict[str, int]] = {}
    unclassified = 0
    fails = [t for t in traces if t.outcome == "fail"]
    for t in fails:
        dets = detect(t, cfg)
        if not dets:
            unclassified += 1
            continue
        for cat in {d.category for d in dets}:
            row = cats.setdefault(cat, {"failed_runs_flagged": 0, "top_key_at_flagged_step": 0})
            row["failed_runs_flagged"] += 1
            steps = {s.idx: s for s in t.steps}
            if any(d.category == cat and d.step_idx in steps and step_key(steps[d.step_idx], t.role) in top for d in dets):
                row["top_key_at_flagged_step"] += 1
    return {"top_m": top_m, "n_failed_runs": len(fails), "unclassified": unclassified, "by_category": cats}


def overlap_with_human(ranking: TaskRanking, human_keys: Iterable[Key], top_m: int = 3) -> dict[str, Any]:
    """Overlap of the top-m spectrum keys with a human-marked key set (the HIL-12 item-6 overlap sample)."""
    top = [r.key for r in ranking.ranking[:top_m]]
    h = set(human_keys)
    hit = [k for k in top if k in h]
    return {"top_m": top_m, "human_n": len(h), "hits": len(hit), "hit_keys": [list(k) for k in hit]}


def ranking_to_dict(r: TaskRanking, limit: int = 10) -> dict[str, Any]:
    return {"task_key": r.task_key, "n_runs": r.n_runs, "n_fail": r.n_fail, "n_pass": r.n_pass, "usable": r.usable,
            "top": [{"key": list(x.key), "k2": round(x.score, 6), "ef": x.ef, "nf": x.nf, "ep": x.ep,
                     "ambiguous": x.ambiguous} for x in r.ranking[:limit]]}
