#!/usr/bin/env python3
"""HIL-18 spectrum attribution wrapper.

  run     replay each task N times through the live API (INFERENCE: refuses unless WINDOW_CONFIRMED=1)
  report  rank step-type keys from a manifest + step-event DB (offline, zero inference)

tasks.jsonl rows: {"task_key", "prompt", "expected", "q": {scoring config for score_response}}
The API server must run with ORCHESTRATOR_STEP_EVENT_LOGGING=1 (and the dependency flag per HIL-15a).
Replay variance comes from the role's sampler: ChatRequest carries no temperature, so the report's
pass-rate-strictly-between-0-and-1 filter, not an assumption, decides which tasks are usable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trace.spectrum_replay import (RunOutcome, build_report, dumps, load_manifest, replay_task,  # noqa: E402
                                       traces_for_manifest)


from scripts.lib.epyc_root import default_epyc_root  # noqa: E402


def write_sidecar(out: str, kind: str, run_id: str, epyc_root: str) -> Path:
    """VB-HIL-REPLAY-WRITE: emit <report>.beliefs.jsonl at write time (opt-in; never backfills a finished report)."""
    import subprocess

    sys.path.insert(0, str(Path(epyc_root) / "scripts" / "vidya"))
    from adapters.hil_replay_capture import write_belief_measurements  # type: ignore

    ref = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    return write_belief_measurements(out, kind=kind, run_id=run_id,
                                     producer="epyc-orchestrator/scripts/trace/" + Path(__file__).name + " " + kind, code_ref=ref)


def _require_window() -> None:
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        sys.exit("refusing: this runs inference. Set WINDOW_CONFIRMED=1 inside an operator-confirmed window.")


def _api_runner(api: str, role: str, max_turns: int, timeout: float):
    import httpx

    def run(task: dict, run_id: str, i: int) -> RunOutcome:
        payload = {"prompt": task["prompt"], "force_role": role, "force_mode": "repl", "max_turns": max_turns,
                   "cache_prompt": False, "session_id": run_id, "mock_mode": False, "real_mode": True}
        try:
            with httpx.Client(timeout=timeout) as c:
                r = c.post(f"{api}/chat", json=payload)
            r.raise_for_status()
            b = r.json()
            return RunOutcome(final_answer=b.get("answer") or b.get("response") or "")
        except Exception as e:
            return RunOutcome(error=f"{type(e).__name__}: {e}")

    return run


def _verifier():
    sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
    from answer_scoring_vendored import score_response   # the existing benchmark scorer

    return lambda task, ans: bool(score_response(ans or "", task["expected"], task.get("q") or {}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tasks", required=True); r.add_argument("--manifest", required=True)
    r.add_argument("--n", type=int, default=20); r.add_argument("--api", default="http://localhost:8000")
    r.add_argument("--role", default="frontdoor"); r.add_argument("--max-turns", type=int, default=12)
    r.add_argument("--timeout", type=float, default=300.0); r.add_argument("--db", required=True)
    p = sub.add_parser("report")
    p.add_argument("--manifest", required=True); p.add_argument("--db", required=True); p.add_argument("--out", required=True)
    p.add_argument("--human", help="json {task_key: [[role, tool, status], ...]}"); p.add_argument("--n-tasks", type=int, default=10)
    p.add_argument("--min-replays", type=int, default=20); p.add_argument("--top-m", type=int, default=3)
    p.add_argument("--belief-sidecar", action="store_true", help="also write <out>.beliefs.jsonl via the root belief-kernel writer")
    p.add_argument("--run-id", default="spectrum-run"); p.add_argument("--epyc-root", default=default_epyc_root("scripts/vidya/adapters/hil_replay_capture.py"))
    a = ap.parse_args()
    if a.cmd == "run":
        _require_window()
        runner, ver = _api_runner(a.api, a.role, a.max_turns, a.timeout), _verifier()
        tag = uuid.uuid4().hex[:6]
        for line in Path(a.tasks).read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            ents = replay_task(task, a.n, runner, ver, a.manifest, id_prefix=f"spec-{tag}-{task['task_key']}")
            _, acct = traces_for_manifest(a.db, ents)
            print(task["task_key"], "pass", sum(e["outcome"] == "pass" for e in ents), "fail",
                  sum(e["outcome"] == "fail" for e in ents), "with_steps", acct["with_steps"], "/", acct["replays_started"], flush=True)
            if acct["with_steps"] == 0:   # join broken or flag off: stop burning the window
                sys.exit("no step events joined to any replay: is ORCHESTRATOR_STEP_EVENT_LOGGING=1 on the API server and "
                         "session id = run id? Stopping.")
        return 0
    human = json.loads(Path(a.human).read_text()) if a.human else None
    per_task, acct = traces_for_manifest(a.db, load_manifest(a.manifest))
    Path(a.out).write_text(dumps(build_report(per_task, acct, n_tasks=a.n_tasks, min_replays=a.min_replays,
                                              top_m=a.top_m, human=human)))
    if a.belief_sidecar:
        write_sidecar(a.out, "spectrum", a.run_id, a.epyc_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
