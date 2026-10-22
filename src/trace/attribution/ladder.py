"""The attribution ladder of intake-2320#record (AgentDebugX D2), on local rungs.

    rung 0  L2 rules          free: the existing mechanical detectors (src/trace/l2); candidate =
                              the first visible deviation's step (a first-event label, not a root cause)
    rung 1  whole-trace pass  one judge call over the whole trace
    rung 2  bisection         ceil(log2 n) "upper or lower half" probes down to one step
    rung 3  adjudicator       pairwise cross-examination, ONLY when the rungs disagree

Policy ``calibrate`` runs rungs 0-2 on every case so each rung's standalone accuracy is measurable;
``escalate`` is the deployed cost-aware order (rung 2 runs only if rungs 0 and 1 do not agree).
Either way rung 3 runs only when the non-abstaining candidates have no majority (two or more
distinct steps and no step named by at least two rungs). Every judge call is charged to its rung's
``TokenMeter``. A rung that cannot answer (parse failure, out-of-range step, empty trace) abstains
with a reason; it never guesses.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.trace.attribution.adapter import to_l2_trace
from src.trace.attribution.judge import Judge, TokenMeter, as_int, extract_json, metered
from src.trace.attribution.prompts import (
    PromptConfig,
    adjudicate_request,
    bisect_request,
    localize_request,
)
from src.trace.attribution.schema import AgentTrajectory
from src.trace.l2.detectors import DetectorConfig, detect, first_visible_deviation

RUNGS = ("rung0_l2", "rung1_whole", "rung2_bisect", "rung3_adjudicate")


@dataclass
class RungResult:
    rung: str
    step: int | None = None
    ran: bool = True
    reason: str | None = None  # why it abstained / did not run
    tokens: TokenMeter = field(default_factory=TokenMeter)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "rung": self.rung,
            "step": self.step,
            "ran": self.ran,
            "reason": self.reason,
            "tokens": self.tokens.as_dict(),
            "detail": self.detail,
        }


@dataclass
class LadderResult:
    case_id: str
    rungs: dict[str, RungResult]
    final_step: int | None
    decided_by: str
    n_steps: int

    def tokens_total(self) -> TokenMeter:
        m = TokenMeter()
        for r in self.rungs.values():
            m.add(r.tokens)
        return m

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "final_step": self.final_step,
            "decided_by": self.decided_by,
            "n_steps": self.n_steps,
            "tokens_total": self.tokens_total().as_dict(),
            "rungs": {k: v.as_dict() for k, v in self.rungs.items()},
        }


@dataclass
class LadderConfig:
    policy: str = "calibrate"  # calibrate | escalate
    prompt: PromptConfig = field(default_factory=PromptConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)


# ---------------------------------------------------------------------------------- rungs
def rung0_l2(traj: AgentTrajectory, cfg: DetectorConfig) -> RungResult:
    dets = detect(to_l2_trace(traj), cfg)
    fvd = first_visible_deviation([d for d in dets if d.step_idx is not None])
    r = RungResult(
        "rung0_l2", detail={"detections": [[d.category, d.step_idx, d.signature] for d in dets]}
    )
    if fvd is None:
        r.reason = "no step-level detection"
    else:
        r.step = fvd.step_idx
        r.detail["category"] = fvd.category
    return r


def rung1_whole(
    traj: AgentTrajectory, judge: Judge, cfg: PromptConfig, reference: str | None
) -> RungResult:
    r = RungResult("rung1_whole")
    steps = set(traj.step_indices())
    if not steps:
        r.reason = "no attributable steps"
        return r
    resp = metered(judge, r.tokens)(localize_request(traj, cfg, reference))
    obj = extract_json(resp.text)
    step = as_int(obj.get("step_index")) if isinstance(obj, dict) else None
    if obj is None or not isinstance(obj, dict):
        r.reason = "parse_failure"
    elif step is None:
        r.reason = "judge_returned_null"
    elif step not in steps:
        r.reason = f"out_of_range:{step}"
    else:
        r.step = step
        r.detail["agent_name"] = obj.get("agent_name")
    return r


def rung2_bisect(
    traj: AgentTrajectory, judge: Judge, cfg: PromptConfig, reference: str | None
) -> RungResult:
    r = RungResult("rung2_bisect")
    ev = traj.steps()
    if not ev:
        r.reason = "no attributable steps"
        return r
    call = metered(judge, r.tokens)
    lo, hi, probes = 0, len(ev) - 1, []
    while lo < hi:
        mid = (lo + hi) // 2
        obj = extract_json(
            call(bisect_request(traj, ev[lo : mid + 1], ev[mid + 1 : hi + 1], cfg, reference)).text
        )
        half = str(obj.get("half", "")).strip().lower() if isinstance(obj, dict) else ""
        probes.append([ev[lo].step_index, ev[hi].step_index, half or None])
        if half == "upper":
            hi = mid
        elif half == "lower":
            lo = mid + 1
        else:
            r.reason = "parse_failure"
            r.detail["probes"] = probes
            return r
    r.step = ev[lo].step_index
    r.detail["probes"] = probes
    r.detail["max_probes"] = math.ceil(math.log2(len(ev))) if len(ev) > 1 else 0
    return r


def majority(cands: list[int | None]) -> tuple[int | None, bool]:
    """(agreed step, needs_adjudication). Agreed = named by >= 2 rungs, or the only distinct answer."""
    named = [c for c in cands if c is not None]
    if not named:
        return None, False
    cnt = Counter(named)
    top, n = cnt.most_common(1)[0]
    if len(cnt) == 1:
        return top, False
    if n >= 2 and list(cnt.values()).count(n) == 1:
        return top, False
    return None, True


def rung3_adjudicate(
    traj: AgentTrajectory, cands: list[int], judge: Judge, cfg: PromptConfig, reference: str | None
) -> RungResult:
    """Pairwise tournament over the distinct candidates in rung order: champion vs each challenger."""
    r = RungResult("rung3_adjudicate", detail={"candidates": cands, "bouts": []})
    call = metered(judge, r.tokens)
    champ = cands[0]
    for ch in cands[1:]:
        obj = extract_json(call(adjudicate_request(traj, champ, ch, cfg, reference)).text)
        win = as_int(obj.get("step")) if isinstance(obj, dict) else None
        if win not in (champ, ch):
            r.detail["bouts"].append([champ, ch, None])
            r.reason = "parse_failure_kept_champion"
            continue
        r.detail["bouts"].append([champ, ch, win])
        champ = win
    r.step = champ
    return r


# ---------------------------------------------------------------------------------- ladder
def run_ladder(
    case_id: str,
    traj: AgentTrajectory,
    judge: Judge,
    cfg: LadderConfig | None = None,
    reference: str | None = None,
) -> LadderResult:
    cfg = cfg or LadderConfig()
    if cfg.policy not in ("calibrate", "escalate"):
        raise ValueError(f"unknown policy {cfg.policy!r}")
    rs: dict[str, RungResult] = {
        "rung0_l2": rung0_l2(traj, cfg.detector),
        "rung1_whole": rung1_whole(traj, judge, cfg.prompt, reference),
    }
    r0, r1 = rs["rung0_l2"].step, rs["rung1_whole"].step
    n = len(traj.step_indices())
    if cfg.policy == "escalate" and r0 is not None and r0 == r1:
        rs["rung2_bisect"] = RungResult(
            "rung2_bisect", ran=False, reason="skipped: rungs 0 and 1 agree"
        )
        rs["rung3_adjudicate"] = RungResult(
            "rung3_adjudicate", ran=False, reason="skipped: no disagreement"
        )
        return LadderResult(case_id, rs, r1, "rung0+rung1", n)
    rs["rung2_bisect"] = rung2_bisect(traj, judge, cfg.prompt, reference)
    cands = [r0, r1, rs["rung2_bisect"].step]
    agreed, need = majority(cands)
    if not need:
        rs["rung3_adjudicate"] = RungResult(
            "rung3_adjudicate", ran=False, reason="skipped: no disagreement"
        )
        who = "+".join(
            k.split("_")[0] for k, c in zip(RUNGS[:3], cands) if c is not None and c == agreed
        )
        return LadderResult(case_id, rs, agreed, who or "none", n)
    distinct = list(dict.fromkeys(c for c in cands if c is not None))
    rs["rung3_adjudicate"] = rung3_adjudicate(traj, distinct, judge, cfg.prompt, reference)
    return LadderResult(case_id, rs, rs["rung3_adjudicate"].step, "rung3", n)
