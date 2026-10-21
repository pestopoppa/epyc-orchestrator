#!/usr/bin/env python3
"""HIL-16 replay tier CLI.

  verify    CAS blobs re-hash, prompt/output chain recomputes, tool outputs present (offline, no inference)
  prefix    restart-at-k prefix check with a harness: bytes of every replayed prompt must match (no inference)
  flip      the four-arm replay (targeted, no_hint, placebo, wrong_step): INFERENCE, refuses unless WINDOW_CONFIRMED=1

A harness/llm/scorer is given as ``module:factory`` (factory() returns the object; for --llm the
factory takes the sample seed). The recording comes from a run made with ORCHESTRATOR_REPLAY_RECORDING=1.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trace.replay.cas import CAS  # noqa: E402
from src.trace.replay.driver import ReplayDriver, build_arms, run_arm  # noqa: E402
from src.trace.replay.model import load_run, verify_integrity  # noqa: E402
from src.trace.replay.report import build_report, dumps  # noqa: E402


def _factory(spec: str):
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["verify", "prefix", "flip"])
    ap.add_argument("--db", required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--cas-dir"); ap.add_argument("--k", type=int)
    ap.add_argument("--harness"); ap.add_argument("--llm"); ap.add_argument("--scorer")
    ap.add_argument("--hint"); ap.add_argument("--wrong-turn", type=int)
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--original-outcome", default="fail"); ap.add_argument("--task-key")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cas_dir:
        os.environ["ORCHESTRATOR_REPLAY_CAS_DIR"] = a.cas_dir   # harness factories build their own CAS() from it
    cas, run = CAS(a.cas_dir), load_run(a.db, a.run_id)
    if a.cmd == "verify":
        problems = verify_integrity(run, cas)
        print(json.dumps({"run_id": a.run_id, "turns": len(run.turns), "problems": problems}, indent=2))
        return 1 if problems else 0
    if a.k is None or not a.harness:
        sys.exit("--k and --harness are required")
    hf = _factory(a.harness)
    if a.cmd == "prefix":
        res = ReplayDriver(run, cas, hf()).check_prefix(a.k)
        print(json.dumps(res.__dict__, indent=2, default=str))
        return 0 if res.ok else 1
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        sys.exit("refusing: flip runs inference. Set WINDOW_CONFIRMED=1 inside an operator-confirmed window.")
    if not (a.llm and a.scorer and a.hint):
        sys.exit("--llm, --scorer and --hint are required for flip")
    lf, scorer = _factory(a.llm), _factory(a.scorer)()
    arms = build_arms(run, a.k, a.hint, wrong_turn=a.wrong_turn)
    samples = {arm.name: run_arm(arm, a.n, run=run, cas=cas, harness_factory=hf, llm_factory=lf, scorer=scorer,
                                 temperature=a.temperature) for arm in arms}
    rep = build_report(run_id=a.run_id, task_key=a.task_key or a.run_id, original_outcome=a.original_outcome,
                       restart_turn=a.k, hint=a.hint,
                       arms={x.name: {"restart_turn": x.restart_turn, "hint_chars": len(x.patch.text) if x.patch else 0}
                             for x in arms}, samples=samples)
    (Path(a.out).write_text(dumps(rep)) if a.out else print(dumps(rep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
