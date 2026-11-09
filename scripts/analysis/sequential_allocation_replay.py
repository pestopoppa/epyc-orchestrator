#!/usr/bin/env python3
"""SEQ-4: zero-inference allocation replay over the stored sequential pool.

Question: would a bandit allocator have reached ``confirm_e`` on the stored pool,
instead of the fixed ``budget=8`` heuristic?

ZERO INFERENCE, OFFLINE. Reads ``orchestration/autopilot_journal*.jsonl`` and re-folds
the recorded per-trial ``z`` through ``EProcessState`` (same e-process, same
``confirm_e=20``; nothing is relaxed in the statistics). It does not touch SEQ-3 /
SEQ-B1 and re-arms no gate.

ARMS. One arm per (candidate, core_id) era group; an arm's pulls are the recorded z
values in recorded ``k`` order. A pull consumes the NEXT recorded z; an exhausted arm
cannot be pulled. THE COUNTERFACTUAL IS THEREFORE BOUNDED BY THE STORED POOL: an arm
with one recorded trial can never be "extended". A "not confirmed" result is a ceiling
on what these stored trials show, not a claim about trials that were never run. A
"confirmed" result would be a lower bound.

ALLOCATORS (all with the same total pull budget = number of recorded trials):
  chronological  the as-run order (journal timestamp) under the DEFAULT policy
                 (budget=8 / budget_min_e=2 stop rule), the fixed-budget baseline.
  ucb            BaSE-style UCB1 on per-arm mean log-growth (intake-1456), relaxed stop.
  thompson       Gaussian Thompson on per-arm mean log-growth, seeded, relaxed stop.
  leader_auc     fluid-style portfolio objective (intake-1457): maximise the area under
                 the best-so-far log-wealth curve by pulling the arm that currently
                 holds the highest wealth (new arms get one pull first). OPERATIONALISED
                 FROM THE HANDOFF LINE ONLY: the intake text was not read here, so treat
                 this as one reading of "AUC objective", not the paper's algorithm.

REPORT. Per allocator: confirmed arms, pulls-to-first-confirm, endpoint max E, and
AUC-frontier = mean over pulls of log(best-so-far E). AUC-frontier and endpoint are
reported separately because they can rank allocators differently.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.autopilot_core.sequential_verdict import (  # noqa: E402
    DEFAULT_POLICY,
    EProcessState,
    SequentialPolicy,
    _z_lower_bound,
)

ALLOCATORS = ("chronological", "ucb", "thompson", "leader_auc")


@dataclass
class Arm:
    key: tuple[str, str]
    zs: list[float]
    stamps: list[str]
    pos: int = 0
    state: EProcessState = EProcessState()
    log_growth_sum: float = 0.0
    stopped: bool = False

    @property
    def available(self) -> bool:
        return self.pos < len(self.zs) and not self.stopped


def load_arms(pattern: str, *, era_fence: bool = True) -> list[Arm]:
    rows: dict[tuple[str, str], list[tuple[int, str, float]]] = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                seq = row.get("seq")
                if not (isinstance(seq, dict) and seq.get("candidate")) or seq.get("z") is None:
                    continue
                core = "ALL" if not era_fence else str(seq.get("core_id"))
                rows[(seq["candidate"], core)].append(
                    (int(seq.get("k") or 0), str(row.get("timestamp") or ""), float(seq["z"]))
                )
    arms = []
    for key, items in sorted(rows.items()):
        items.sort(key=lambda r: r[0])
        arms.append(Arm(key=key, zs=[r[2] for r in items], stamps=[r[1] for r in items]))
    return arms


def _pull(arm: Arm, policy: SequentialPolicy, z_floor: float) -> None:
    z = arm.zs[arm.pos]
    arm.pos += 1
    if not math.isfinite(z) or z < z_floor - 1e-9:
        return  # skipped exactly as rebuild_candidate_view does: factor 1.0
    before = arm.state.wealth
    arm.state, _ = arm.state.update(z, policy=policy)
    if before > 0 and arm.state.wealth > 0:
        arm.log_growth_sum += math.log(arm.state.wealth / before)
    if arm.state.state_name(policy) == "refuted":
        arm.stopped = True


def _choose(name: str, arms: Sequence[Arm], t: int, rng: random.Random) -> Arm | None:
    avail = [a for a in arms if a.available]
    if not avail:
        return None
    fresh = [a for a in avail if a.state.k == 0]
    if fresh and name != "thompson":
        return fresh[0]
    if name == "ucb":
        return max(avail, key=lambda a: (a.log_growth_sum / max(a.state.k, 1))
                   + math.sqrt(2.0 * math.log(max(t, 2)) / max(a.state.k, 1)))
    if name == "thompson":
        return max(avail, key=lambda a: rng.gauss(a.log_growth_sum / (a.state.k + 1), 1.0 / math.sqrt(a.state.k + 1)))
    if name == "leader_auc":
        return max(avail, key=lambda a: a.state.wealth)
    raise ValueError(name)


def replay(arms_in: Sequence[Arm], allocator: str, *, total_pulls: int | None = None,
           policy: SequentialPolicy = DEFAULT_POLICY, seed: int = 0) -> dict:
    if allocator not in ALLOCATORS:
        raise ValueError(f"unknown allocator {allocator!r}")
    arms = [Arm(a.key, list(a.zs), list(a.stamps)) for a in arms_in]
    budget = total_pulls if total_pulls is not None else sum(len(a.zs) for a in arms)
    # Bandits get the relaxed STOP rule (statistics untouched): the point of an allocator
    # is to decide where trials go, so it must not be pre-cut by the budget heuristic.
    run_policy = policy if allocator == "chronological" else replace(policy, budget=10**9, budget_min_e=0.0)
    z_floor = _z_lower_bound(run_policy)
    rng = random.Random(seed)
    order = None
    if allocator == "chronological":
        order = sorted(((a.stamps[i], ai, i) for ai, a in enumerate(arms) for i in range(len(a.zs))))
    best_curve: list[float] = []
    first_confirm = None
    pulls = 0
    for step in range(budget):
        if order is not None:
            if step >= len(order):
                break
            _, ai, _i = order[step]
            arm = arms[ai]
            if not arm.available:
                best_curve.append(best_curve[-1] if best_curve else 1.0)
                continue
        else:
            arm = _choose(allocator, arms, step + 1, rng)
            if arm is None:
                break
        _pull(arm, run_policy, z_floor)
        pulls += 1
        best = max(a.state.wealth for a in arms)
        best_curve.append(max(best, best_curve[-1] if best_curve else 1.0))
        if first_confirm is None and arm.state.wealth >= policy.confirm_e:
            first_confirm = pulls
    logs = [math.log(max(b, 1e-300)) for b in best_curve]
    top = sorted(arms, key=lambda a: -a.state.wealth)[:3]
    return {
        "allocator": allocator, "pulls": pulls, "seed": seed if allocator == "thompson" else None,
        "confirmed": sum(1 for a in arms if a.state.wealth >= policy.confirm_e),
        "pulls_to_first_confirm": first_confirm,
        "endpoint_max_E": best_curve[-1] if best_curve else 1.0,
        "auc_frontier_mean_log_best_E": sum(logs) / len(logs) if logs else 0.0,
        "arms_touched": sum(1 for a in arms if a.state.k > 0),
        "top3": [{"candidate": a.key[0], "core_id": a.key[1], "k": a.state.k,
                  "E": round(a.state.wealth, 4)} for a in top],
    }


def run_all(arms: Sequence[Arm], *, thompson_seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict:
    out = {"n_arms": len(arms), "n_recorded_trials": sum(len(a.zs) for a in arms),
           "arms_with_one_trial": sum(1 for a in arms if len(a.zs) == 1),
           "bound": "pulls limited to recorded trials; not-confirmed is a ceiling on the stored pool",
           "results": []}
    for name in ALLOCATORS:
        if name == "thompson":
            out["results"].extend(replay(arms, name, seed=s) for s in thompson_seeds)
        else:
            out["results"].append(replay(arms, name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal-glob", default=str(REPO / "orchestration/autopilot_journal*.jsonl"))
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    arms = load_arms(args.journal_glob)
    if not arms:
        print("no sequential trials found")
        return 1
    report = run_all(arms)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
