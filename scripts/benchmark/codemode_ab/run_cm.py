#!/usr/bin/env python3
"""Runner for the CM-2 / CM-4 / CM-5 probe (cm_probe.py arms on top of C3). Resumable: one ``episode`` record per unit in ``OUT/results.jsonl``.

    run_cm.py --out OUT --model-key 35b|27b --url U --embed-url E --feature cm4 --upto 10      # first 10 pairs of cm4 (C3 and C3_cm4 each)
    run_cm.py --mock --out OUT --feature cm5 --upto 2                                           # scripted solver, no model/server (plumbing only)
    run_cm.py --dry-run --feature cm4 --upto 10                                                 # prints the units, writes nothing

The units of a chunk are the not-yet-recorded (task, arm) of the first ``--upto`` pairs of the feature (cm_plan.pairs), C3 first. A unit already recorded in
OUT (by any earlier chunk or feature) is skipped, so C3 runs once per task. The sequential stop rules live in the analyzer, not here. rc: 0 chunk done |
75 stop-file/deadline (resumable) | 3 environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ORCH_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ORCH_ROOT))

from scripts.benchmark.codemode_ab import cm_plan  # noqa: E402
from scripts.benchmark.codemode_ab.cm_probe import ARMS_CM, run_episode_cm  # noqa: E402
from scripts.benchmark.codemode_ab.run_probe import SCRATCH, Out, build_selectors, register_native_picker  # noqa: E402
from scripts.benchmark.codemode_ab.tasks import build_cm_tasks, build_tasks  # noqa: E402
from scripts.benchmark.codemode_ab.world import World  # noqa: E402


def task_index() -> dict:
    idx = {t.task_id: t for t in build_tasks([0, 1, 2]) if t.stage == 1}
    idx.update({t.task_id: t for t in build_cm_tasks(cm_plan.CM_SEEDS)})
    return idx


def units_for(feature: str, upto: int) -> list[tuple[str, str]]:
    arm = cm_plan.ARM_OF[feature]
    return [(t, a) for t, _k in cm_plan.pairs(feature)[:upto] for a in ("C3", arm)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--model-id", default="mock")
    ap.add_argument("--model-key", help="35b | 27b: the models_v2.py row; sets --model-id")
    ap.add_argument("--url")
    ap.add_argument("--embed-url")
    ap.add_argument("--embed-id", default="bge-large-en-v1.5")
    ap.add_argument("--feature", required=True, choices=cm_plan.FEATURES)
    ap.add_argument("--upto", type=int, required=True, help="run the first N pairs of the feature")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="record mode=smoke (never counts toward the live row)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--work", default="/mnt/raid0/llm/tmp/codemode-ab-20261117/work")
    a = ap.parse_args(argv)
    if a.model_key:
        from scripts.benchmark.codemode_ab.models_v2 import resolve

        try:
            a.model_id = resolve(a.model_key)["model_id"]
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 3
    n_pairs = len(cm_plan.pairs(a.feature))
    if not 1 <= a.upto <= n_pairs:
        print(f"--upto must be 1..{n_pairs} for {a.feature}", file=sys.stderr)
        return 3
    units = units_for(a.feature, a.upto)
    idx = task_index()
    missing = [t for t, _ in units if t not in idx]
    if missing:
        print(f"plan names tasks the harness does not build: {sorted(set(missing))[:5]}", file=sys.stderr)
        return 3
    if a.dry_run:
        print(json.dumps({"dry_run": True, "feature": a.feature, "units": len(units), "pairs": a.upto, "first": units[:4], "last": units[-2:],
                          "kinds": sorted({k for _t, k in cm_plan.pairs(a.feature)[:a.upto]})}, indent=1))
        return 0
    if not a.out:
        print("--out required", file=sys.stderr)
        return 3
    out = Out(Path(a.out), "smoke" if (a.smoke or a.mock) else "full")
    if not a.mock and not (a.url and a.embed_url):
        print("need --url and --embed-url (or --mock)", file=sys.stderr)
        return 3
    work = Path(a.work)
    if not str(work.resolve()).startswith(SCRATCH):
        print(f"--work must be under {SCRATCH}", file=sys.stderr)
        return 3
    stopf = os.environ.get("FILLER_STOPFILE", "")
    deadline = int(os.environ.get("FILLER_DEADLINE", "0"))
    template = World(work / "template", ORCH_ROOT)
    os.environ.setdefault("ORCHESTRATOR_TOOL_SELECTOR_LOG", str(out.dir / "picker_events.jsonl"))
    selectors = build_selectors(argparse.Namespace(mock=a.mock, url=a.url, embed_url=a.embed_url, model_id=a.model_id, embed_id=a.embed_id), template, out)
    register_native_picker(selectors, template)
    done = out.done()
    n_run = 0
    for tid, arm in units:
        uid = f"{a.model_id}|{arm}|{tid}"
        if uid in done:
            continue
        if (stopf and os.path.exists(stopf)) or (deadline and time.time() > deadline - 30):
            return 75
        task = idx[tid]
        world = World(work / "units" / f"{'m' if a.mock else 'r'}_{a.model_id}_{arm}_{tid}", ORCH_ROOT)
        if a.mock:
            from scripts.benchmark.codemode_ab.mock import MockLLM

            llm = MockLLM(task, world, task.seed)
        else:
            llm = selectors["llm"]
        try:
            rec = run_episode_cm(task, arm, world, llm, task.seed * 1000, selectors=selectors)
        except Exception as exc:  # noqa: BLE001 - a harness crash is a record, never a silent skip
            rec = {"arm": arm, "task_id": tid, "family": task.family, "n": task.n, "seed": task.seed, "tier": task.tier, "stage": task.stage,
                   "success": False, "in_tokens": 0, "out_tokens": 0, "end_reason": f"llm_error:harness:{type(exc).__name__}:{exc}"[:200],
                   "wall_s": 0.0, "turns": 0, "failed_snippet_rate": 0.0, "param_accuracy": None, "duplicate_writes": 0}
        tr = rec.pop("_transcript", None)
        if tr is not None:
            rec["transcript_path"] = out.transcript(uid, tr)
        out.append({"kind": "episode", "unit_id": uid, "model": a.model_id, **rec})
        n_run += 1
    return 0


if __name__ == "__main__":
    assert set(ARMS_CM) >= set(cm_plan.ARM_OF.values())
    sys.exit(main())
