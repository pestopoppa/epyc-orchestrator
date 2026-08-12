#!/usr/bin/env python3
"""readjudicate_sequential_candidates.py — did the Autopilot program ever have a real effect?

ZERO INFERENCE. Reads only `orchestration/autopilot_journal*.jsonl` and re-folds the
already-recorded per-trial z statistics through the project's own e-process
(`sequential_verdict.rebuild_candidate_view`). No new trials, no new evaluation, no
episodic-memory dependency — the journal is a separate store from
`orchestration/repl_memory/sessions/` and was untouched by the 2026-07-27 reseed.

WHY
---
Measured over 393 sequential trials / 141 candidates on 2026-07-27:

* median **1.0** trial per candidate (mean 2.79, max 41)
* **0 of 121** refuted trials were killed by futility (E <= futility_e)
* the strongest candidate, ``70902e4b665474e7``, trips ``k>=8 and E<2.0`` at k=8 with
  E=1.68, then climbs to **E_quality 11.55 by k=40 while still labelled ``refuted``** —
  5.8x the kill bar and 58% of the confirm bar
* **RETRACTED 2026-08-12** — this bullet used to read: *"56 of 393 trials carry
  ``state="refuted"`` while ``E>=budget_min_e`` and ``k>=budget`` … the label is STICKY."*
  **There is no stale label, and there never was.** That population was manufactured by this
  script's own SEQ-A block, which compared the JOINT ``state`` against the SINGLE
  ``E_quality`` axis (see the comment above ``refuted_by`` below for the mechanism, and
  ``f2ad030e`` for the fix). ``safety_gate.py`` recomputes ``state`` on EVERY trial from
  ``q_name == REFUTED or rate_name == REFUTED``, so a healthy quality axis beside a refuted
  RATE axis read as a label that had failed to update. Since ``E_rate_noninf`` never exceeds
  2.0 anywhere in the corpus (max 1.1100) against ``budget_min_e = 2.0``, essentially every
  candidate's rate axis refutes once ``k >= budget``. Re-measured after the fix: 6 refuted on
  the quality axis, 3 on the rate axis only, and **0 unexplained** — the empty third bucket is
  the finding, because it is the only bucket a genuinely stale label could occupy. The live
  question is joint-gate-vs-quality-primary (SEQ-B1), not staleness. This bullet is retracted
  in place rather than deleted: the ``70902e4b665474e7`` bullet above is still true and reads
  as evidence for staleness unless the retraction sits next to it.

``confirm_e=20.0`` is the Ville bound for alpha=0.05 and is NOT touched here. ``budget``
and ``budget_min_e`` are a compute-allocation heuristic with no bearing on
anytime-validity, so relaxing them is statistically free — that is the whole point of
this script.

WHAT IT ANSWERS
---------------
Under the recorded evidence alone, with the budget rule relaxed:
  * does any candidate reach ``confirm_e`` (a genuine confirm)?
  * how many were still *rising* when they were cut?
  * what is the per-trial wealth growth of the survivors, i.e. how many more trials
    would each have needed?

Era-fenced by ``core_id`` by default: folding z's across instrument eras would mix
non-comparable evidence, which is the defect the E7/E8 era work exists to prevent.

Usage:
    python scripts/analysis/readjudicate_sequential_candidates.py
    python scripts/analysis/readjudicate_sequential_candidates.py --out-json report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def axis_refuted(wealth: float | None, k: int, pol) -> bool:
    """Apply the policy's refutation rule to ONE evidence axis.

    Mirrors `EProcessState.state_name`: an axis refutes on futility, or on the
    budget rule once `k >= budget`. `None` (axis not recorded) never refutes —
    an absent measurement is not evidence against.

    Exists at module level so the axis attribution in the report is testable
    without running the whole re-adjudication over the journal.
    """
    if wealth is None:
        return False
    return wealth <= pol.futility_e or (k >= pol.budget and wealth < pol.budget_min_e)


def _axis_refuted_factory(pol):
    """Bind `axis_refuted` to a policy for use inside the report loop."""
    def _bound(wealth: float | None, k: int) -> bool:
        return axis_refuted(wealth, k, pol)
    return _bound


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_trials(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            seq = row.get("seq")
            if isinstance(seq, dict) and seq.get("candidate"):
                seq = dict(seq)
                seq["_trial_id"] = row.get("trial_id")
                seq["_timestamp"] = row.get("timestamp")
                out.append(seq)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--journal-glob", default=str(REPO / "orchestration/autopilot_journal*.jsonl"))
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--no-era-fence",
        action="store_true",
        help="fold across core_ids (NOT recommended — mixes instrument eras)",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from src.autopilot_core.sequential_verdict import (  # noqa: E402
        DEFAULT_POLICY,
        rebuild_candidate_view,
    )

    trials = load_trials(args.journal_glob)
    if not trials:
        log("no sequential trials found")
        return 1

    pol = DEFAULT_POLICY
    log("=== POLICY (unchanged vs relaxed) ===")
    log(f"  confirm_e     = {pol.confirm_e}   <- Ville bound for alpha=0.05, NOT touched")
    log(f"  futility_e    = {pol.futility_e}")
    log(f"  budget        = {pol.budget}      <- compute heuristic")
    log(f"  budget_min_e  = {pol.budget_min_e}   <- compute heuristic")
    log()

    # Relaxed policy: keep the statistics, drop the allocation cut-off only.
    relaxed = replace(pol, budget=10**9, budget_min_e=0.0)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in trials:
        key = (t["candidate"], "ALL" if args.no_era_fence else str(t.get("core_id")))
        groups[key].append(t)

    log(f"=== {len(trials)} trials | {len({t['candidate'] for t in trials})} candidates "
        f"| {len(groups)} candidate x era groups ===")
    log()

    results = []
    for (cand, core), rows in groups.items():
        rows.sort(key=lambda r: (r.get("k") or 0))
        obs = [(r.get("_trial_id"), float(r["z"])) for r in rows if r.get("z") is not None]
        if not obs:
            continue
        as_is = rebuild_candidate_view(
            candidate=cand, core_id=core, observations=obs, policy=pol
        )
        relax = rebuild_candidate_view(
            candidate=cand, core_id=core, observations=obs, policy=relaxed
        )
        recorded = rows[-1].get("state")
        e = float(relax.quality_state.wealth)
        k = int(relax.quality_state.k)
        # per-trial growth factor over the observed run, for a trials-to-confirm estimate
        growth = e ** (1.0 / k) if k > 0 and e > 0 else 1.0
        need = (
            math.ceil(math.log(pol.confirm_e / e) / math.log(growth))
            if e > 0 and growth > 1.0 + 1e-9 and e < pol.confirm_e
            else None
        )
        results.append(
            {
                "candidate": cand,
                "core_id": core,
                "k": k,
                "recorded_state": recorded,
                "state_as_is": as_is.state,
                "state_relaxed": relax.state,
                "E_as_is": round(float(as_is.quality_state.wealth), 4),
                "E_relaxed": round(e, 4),
                "growth_per_trial": round(growth, 5),
                "trials_to_confirm": need,
            }
        )

    confirmed = [r for r in results if r["state_relaxed"] == "confirmed"]
    rising = [r for r in results if r["growth_per_trial"] > 1.0 and r["state_relaxed"] != "confirmed"]
    flipped = [r for r in results if r["recorded_state"] == "refuted" and r["state_relaxed"] != "refuted"]

    log("=== HEADLINE ===")
    log(f"  candidates CONFIRMED under the relaxed budget: {len(confirmed)}")
    log(f"  recorded 'refuted' that are NOT refuted once the budget rule is dropped: {len(flipped)}")
    log(f"  candidates still RISING when the run ended (growth > 1.0/trial): {len(rising)}")
    log()

    if confirmed:
        log("  CONFIRMED:")
        for r in sorted(confirmed, key=lambda r: -r["E_relaxed"]):
            log(f"    {r['candidate']}  core={r['core_id']}  k={r['k']}  E={r['E_relaxed']}")
        log()

    top = sorted(results, key=lambda r: -r["E_relaxed"])[:10]
    log("=== TOP 10 BY RELAXED E ===")
    log(f"  {'candidate':<20} {'core':<10} {'k':>4} {'E':>10} {'grow':>8} {'need':>6}  recorded")
    for r in top:
        need = r["trials_to_confirm"]
        log(
            f"  {r['candidate']:<20} {str(r['core_id'])[:10]:<10} {r['k']:>4} "
            f"{r['E_relaxed']:>10.4f} {r['growth_per_trial']:>8.4f} "
            f"{(str(need) if need is not None else '-'):>6}  {r['recorded_state']}"
        )
    log()

    ks = [r["k"] for r in results]
    log("=== ALLOCATION ===")
    log(f"  trials per candidate x era: median={st.median(ks)}  mean={st.mean(ks):.2f}  max={max(ks)}")
    log(f"  groups with only 1 trial: {sum(1 for k in ks if k == 1)}/{len(ks)}")
    log()

    # SEQ-A: WHY is a candidate's final label `refuted`? Attribute it to an axis.
    #
    # This block used to read:
    #     if last["state"] == "refuted" and float(last["E_quality"]) >= pol.budget_min_e
    # and report the result as "STICKY REFUTED LABELS". That comparison is invalid:
    # `state` is the JOINT verdict — `safety_gate.py` stamps it `refuted` when
    # EITHER axis refutes (`q_name == REFUTED or rate_name == REFUTED`) and
    # recomputes it on every trial — while `E_quality` is a SINGLE axis. Comparing
    # the two and calling the mismatch staleness mistakes a healthy quality axis
    # sitting next to a refuted RATE axis for a label that failed to update.
    #
    # It is not a small mis-read: `E_rate_noninf` never exceeds 2.0 anywhere in the
    # corpus (max 1.1100) against `budget_min_e = 2.0`, so essentially EVERY
    # candidate's rate axis refutes once k >= budget — which manufactures the
    # entire "stuck" population.
    #
    # So attribute instead of guess. `refuted_by` applies the policy's own rule to
    # one axis at a time; the residual bucket (`joint refuted, NEITHER axis
    # refutes`) is the only thing that would constitute a genuinely stale label.
    refuted_by = _axis_refuted_factory(pol)

    by_quality: list[tuple[str, int, float, float | None]] = []
    by_rate_only: list[tuple[str, int, float, float | None]] = []
    unexplained: list[tuple[str, int, float, float | None]] = []
    for (cand, core), rows_ in groups.items():
        rows_.sort(key=lambda r: (r.get("k") or 0))
        last = rows_[-1]
        if last.get("state") != "refuted":
            continue
        k = int(last.get("k") or 0)
        e_q = float(last["E_quality"])
        raw_rate = last.get("E_rate_noninf")
        e_r = float(raw_rate) if raw_rate is not None else None
        row = (cand, k, e_q, e_r)
        if refuted_by(e_q, k):
            by_quality.append(row)
        elif refuted_by(e_r, k):
            by_rate_only.append(row)
        else:
            unexplained.append(row)

    def _fmt(rows: list[tuple[str, int, float, float | None]]) -> None:
        for c, k, e_q, e_r in sorted(rows, key=lambda t: -t[2]):
            rate = f"{e_r:8.4f}" if e_r is not None else "    n/a"
            log(f"    {c}  k={k:>3}  E_quality={e_q:8.4f}  E_rate={rate}")

    log("=== SEQ-A: WHY EACH `refuted` LABEL IS refuted (axis attribution) ===")
    log(f"  policy: futility_e={pol.futility_e}  budget={pol.budget}  "
        f"budget_min_e={pol.budget_min_e}")
    log(f"  refuted on the QUALITY axis: {len(by_quality)}")
    _fmt(by_quality)
    log(f"  refuted on the RATE axis ONLY (quality axis is healthy): {len(by_rate_only)}")
    _fmt(by_rate_only)
    log("    ^ These are CORRECTLY labelled by the joint rule, not stale. They are")
    log("      excluded from promotion and positive strategy distillation")
    log("      (learning_exclusions.py:111-119) because the JOINT gate refuses a")
    log("      candidate that buys quality with throughput. Whether that is the")
    log("      wanted policy is SEQ-B1 — an operator question, not a defect.")
    log(f"  UNEXPLAINED — joint says refuted, NEITHER axis does: {len(unexplained)}")
    _fmt(unexplained)
    log("    ^ This bucket, and only this bucket, would be a genuinely stale label.")
    log()

    log("=== VERDICT ===")
    if confirmed:
        log("  A candidate CROSSES confirm_e on evidence already purchased. The")
        log("  benchmark-Autopilot hypothesis is ALIVE — the allocation rule, not the")
        log("  effect, was the binding constraint.")
    elif rising:
        best = max(rising, key=lambda r: r["E_relaxed"])
        log("  No candidate crosses confirm_e on recorded evidence, but candidates were")
        log(f"  still RISING when cut. Best: {best['candidate']} at E={best['E_relaxed']} "
            f"after {best['k']} trials,")
        log(f"  growing {best['growth_per_trial']:.4f}x/trial -> ~{best['trials_to_confirm']} more "
            "trials to confirm.")
        log("  The hypothesis is NOT falsified; it was never funded to a verdict.")
    else:
        log("  Nothing rises even with the budget rule removed. The hypothesis is dead on")
        log("  evidence already purchased — retire the line rather than spending more.")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "policy": {
                        "confirm_e": pol.confirm_e,
                        "futility_e": pol.futility_e,
                        "budget": pol.budget,
                        "budget_min_e": pol.budget_min_e,
                    },
                    "n_trials": len(trials),
                    "n_groups": len(groups),
                    "confirmed": confirmed,
                    "flipped_off_refuted": flipped,
                    "results": sorted(results, key=lambda r: -r["E_relaxed"]),
                },
                indent=2,
            )
        )
        log(f"\n[out] {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
