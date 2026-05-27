#!/usr/bin/env python3
"""BEP-2 falsification A/B driver (Phase 3).

Runs the scratch coding workload (data/bep_sandbox/tasks.jsonl) under two arms — batch_edit_mode
OFF (interleaved REPL baseline) vs ON (batch-edit divergence) — measuring round-trips, latency,
and quality (independent verifier), to decide the BEP-2 gate:
  PROCEED→BEP-3: batch latency >=15% down AND quality within -1pp AND parse<=5% AND apply<=2%
  REWORK / STOP otherwise.

Safety (handoffs/active/bep-dcp-falsification-harness.md hard gates):
  * All model edits land in ONE fixed scratch repo (ORCHESTRATOR_EDIT_ROOT) — never the
    orchestrator checkout. Reset to pristine per task×arm×rep.
  * Arms are an arm-BLOCK ABBA sequence (off,on,on,off …) — never one monolithic off then on —
    so time drift doesn't confound. One API restart per block (the flag is process-level).
  * REAL inference is REFUSED unless --host-quiet-confirmed is passed AND J6/autopilot is down
    (feedback_no_concurrent_inference). Default mode is --stub (no inference, no API restart):
    it applies data/bep_sandbox/solutions.jsonl (dry-run reference, NOT model-facing) to exercise
    the full plumbing — scratch reset, verifier, incremental JSONL, ABBA sequencing.
  * Verifiers run with PYTHONDONTWRITEBYTECODE=1 (avoid .pyc shadowing across reps).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ORCH = Path("/mnt/raid0/llm/epyc-orchestrator")
SANDBOX = ORCH / "data" / "bep_sandbox"
API = "http://127.0.0.1:8000"


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _reset_scratch(root: Path, files: dict[str, str]) -> None:
    """Wipe + recreate the scratch repo with the task's pristine files + a git baseline."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=root, check=False)
    subprocess.run(["git", "add", "-A"], cwd=root, check=False)
    subprocess.run(["git", "-c", "user.email=bep@x", "-c", "user.name=bep", "commit", "-qm", "pristine"],
                   cwd=root, check=False)


def _run_verifier(root: Path, cmd: str) -> bool:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=60, env=env)
        return r.returncode == 0 and "PASS" in (r.stdout + r.stderr)
    except Exception:
        return False


def _apply_solution(root: Path, sol: dict) -> None:
    """Stub-only: apply the dry-run reference edit (NOT model-facing)."""
    for rel in sol.get("delete", []):
        (root / rel).unlink(missing_ok=True)
    for rel, content in sol.get("write", {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)


def _restart_api(batch_edit_mode: bool, edit_root: Path) -> bool:
    """REAL mode only: reload the orchestrator API with the arm's flag + the scratch task-root.
    Durable placement flags are preserved by start_orchestrator's setdefaults (23b1a67)."""
    env = {
        **os.environ,
        "ORCHESTRATOR_BATCH_EDIT_MODE": "1" if batch_edit_mode else "0",
        # OFF arm gets the symmetric interleaved-edit rider so both arms actually edit files —
        # the A/B then measures interleaved-vs-batched EDIT latency, not edits-vs-prose.
        "ORCHESTRATOR_INTERLEAVED_EDIT_RIDER": "0" if batch_edit_mode else "1",
        "ORCHESTRATOR_EDIT_ROOT": str(edit_root),
    }
    r = subprocess.run(
        [sys.executable, "scripts/server/orchestrator_stack.py", "reload", "orchestrator"],
        cwd=ORCH, env=env, capture_output=True, text=True, timeout=200,
    )
    return "Orchestrator ready" in (r.stdout + r.stderr)


def _chat(prompt: str, *, max_turns: int, session_id: str, timeout: float = 180.0) -> dict:
    import httpx

    # mock_mode/real_mode MUST be explicit: ChatRequest.mock_mode defaults to True
    # (safety default), so omitting these yields "[MOCK] Processed prompt" responses
    # instead of real inference (caught 2026-05-26 — stub mode never exercises _chat).
    # force_mode="repl": without it the coder routes to "direct" (turns=1 text answer,
    # no file edits) for these tasks, so the batch-edit divergence (helpers.py:852, REPL
    # turn loop) is never reached and quality is 0 in BOTH arms — an uninterpretable A/B.
    payload = {"prompt": prompt, "force_role": "coder_escalation", "force_mode": "repl",
               "max_turns": max_turns, "cache_prompt": False, "session_id": session_id,
               "mock_mode": False, "real_mode": True}
    t0 = time.time()
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{API}/chat", json=payload)
    body = r.json()
    meta = body.get("_meta", {}) or {}
    return {
        "answer": (body.get("answer") or body.get("response") or "")[:200],
        "latency_s": round(time.time() - t0, 3),
        "turns": meta.get("turns", body.get("turns")),
        "prompt_tokens": meta.get("prompt_tokens"),
    }


def _arm_block_sequence(reps: int) -> list[bool]:
    """ABBA-style: balance arm order against time drift. reps=2 → [off,on,on,off]."""
    seq: list[bool] = []
    for i in range(reps):
        block = [False, True] if i % 2 == 0 else [True, False]  # A,B then B,A
        seq.extend(block)
    return seq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2, help="ABBA rep count (2 → off,on,on,off)")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--scratch-root", default="/tmp/bep_scratch/work")
    ap.add_argument("--output", default=str(SANDBOX / f"results-{int(time.time())}"))
    ap.add_argument("--stub", action="store_true",
                    help="No inference: apply solutions.jsonl, exercise the harness only.")
    ap.add_argument("--host-quiet-confirmed", action="store_true",
                    help="REQUIRED for real inference: operator confirms host is inference-quiet.")
    args = ap.parse_args()

    tasks = _load_jsonl(SANDBOX / "tasks.jsonl")
    solutions = {s["id"]: s for s in _load_jsonl(SANDBOX / "solutions.jsonl")}
    scratch = Path(args.scratch_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    real = not args.stub
    if real and not args.host_quiet_confirmed:
        print("REFUSING real inference run: pass --host-quiet-confirmed (and ensure J6/autopilot "
              "is stopped) per the BEP harness hard gate. Use --stub for the no-inference dry-run.",
              file=sys.stderr)
        return 2
    if real:
        ap_running = subprocess.run(["pgrep", "-f", "autopilot.py start"], capture_output=True, text=True)
        if ap_running.stdout.strip():
            print(f"REFUSING: autopilot still running (pids {ap_running.stdout.split()}). Stop J6 first.",
                  file=sys.stderr)
            return 2

    print(f"[bep_ab] mode={'REAL' if real else 'STUB'} reps={args.reps} tasks={len(tasks)} "
          f"scratch={scratch} out={results_path}")
    arm_seq = _arm_block_sequence(args.reps)
    n_written = 0
    with open(results_path, "w") as rf:
        for block_idx, batch_on in enumerate(arm_seq):
            arm = "on" if batch_on else "off"
            if real:
                if not _restart_api(batch_on, scratch):
                    print(f"  [block {block_idx}] API restart FAILED for arm={arm}", file=sys.stderr)
                    return 3
            for task in tasks:
                _reset_scratch(scratch, task["files"])
                t0 = time.time()
                if real:
                    info = _chat(task["prompt"], max_turns=args.max_turns,
                                 session_id=f"bep-{task['id']}-{arm}-{block_idx}")
                else:
                    _apply_solution(scratch, solutions[task["id"]])  # dry-run reference edit
                    info = {"answer": "[STUB]", "latency_s": round(time.time() - t0, 3),
                            "turns": None, "prompt_tokens": None}
                quality = _run_verifier(scratch, task["verifier_cmd"])
                row = {
                    "block": block_idx, "arm": arm, "task": task["id"], "kind": task["kind"],
                    "quality_pass": quality, "latency_s": info["latency_s"],
                    "turns": info.get("turns"), "prompt_tokens": info.get("prompt_tokens"),
                    "answer_preview": info["answer"], "mode": "real" if real else "stub",
                }
                rf.write(json.dumps(row) + "\n")
                rf.flush()  # incremental persistence
                n_written += 1
                print(f"  block{block_idx} arm={arm} {task['id']:24} quality={'PASS' if quality else 'fail'} "
                      f"lat={info['latency_s']}s")

    # quick aggregate
    rows = _load_jsonl(results_path)
    for arm in ("off", "on"):
        a = [r for r in rows if r["arm"] == arm]
        if a:
            q = sum(r["quality_pass"] for r in a) / len(a)
            lat = sorted(r["latency_s"] for r in a)[len(a) // 2]
            print(f"[agg] arm={arm}: n={len(a)} quality={q:.0%} median_lat={lat}s")
    print(f"[bep_ab] wrote {n_written} rows → {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
