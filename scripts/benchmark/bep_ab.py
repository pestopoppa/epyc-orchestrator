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
    it applies data/bep_sandbox/solutions.jsonl (dry-run reference, NOT model-facing) and exercises
    ONLY the harness scaffold — scratch reset, verifier, incremental JSONL, ABBA sequencing,
    artifact/schema. It does NOT touch /chat, routing, the REPL, mode selection, or the model write
    tools, so it canNOT validate the real path (mock_mode/force_mode/write semantics escaped it on
    2026-05-26). Run tests/unit/test_bep_canary.py + a live single-task smoke before any real ABBA.
  * Verifiers run with PYTHONDONTWRITEBYTECODE=1 (avoid .pyc shadowing across reps).
  * Real runs set ORCHESTRATOR_BEP_TURN_TRACE=1 so each task's per-turn model output is captured
    and saved as an artifact (decision-grade transcript; operator gate (c)).
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
ORCH_LOG = ORCH / "logs" / "orchestrator.log"
TURN_TRACE = Path("/mnt/raid0/llm/tmp/bep_turn_trace.jsonl")  # _bep_turn_trace output
MATRIX = ORCH / "orchestration" / "contention_matrix.yaml"


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
        "ORCHESTRATOR_BEP_TURN_TRACE": "1",  # (c) capture per-turn model output as artifacts
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


def _topology_hash() -> str:
    try:
        import yaml
        return str((yaml.safe_load(MATRIX.read_text()) or {}).get("topology_hash", ""))
    except Exception:
        return ""


def _orch_head() -> str:
    """Short SHA of the orchestrator checkout — recorded to prove the A/B never mutated it."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ORCH,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def _git_touched(root: Path) -> list[str]:
    """Files the model changed in the scratch repo vs the pristine baseline commit."""
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True, timeout=10)
        return [ln[3:] for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _mark(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def _trace_slice(mark: int, dest: Path) -> dict:
    """Save the bep_turn_trace.jsonl slice since `mark` to `dest`; return a per-turn summary (c)."""
    if not TURN_TRACE.exists():
        return {"turns": 0, "path": None}
    try:
        with open(TURN_TRACE, errors="replace") as f:
            f.seek(mark)
            rows = [json.loads(ln) for ln in f.read().splitlines() if ln.strip()]
    except Exception:
        return {"turns": 0, "path": None}
    if rows:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return {
        "turns": len(rows),
        "path": str(dest.relative_to(dest.parent.parent)) if rows else None,
        "any_file_write_safe": any(r.get("calls_file_write_safe") for r in rows),
        "any_open": any(r.get("calls_open") for r in rows),
    }


def _batch_edit_states(mark: int) -> dict:
    """Parse `batch_edit_state=<name>` lines emitted to orchestrator.log since `mark` (telemetry
    for the gate's parse/apply/promote rates; helpers.py:_record_batch_edit_state)."""
    if not ORCH_LOG.exists():
        return {}
    try:
        import re
        with open(ORCH_LOG, errors="replace") as f:
            f.seek(mark)
            raw = f.read()
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for m in re.finditer(r"batch_edit_state=(\w+)", raw):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


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

    topo = _topology_hash()
    print(f"[bep_ab] mode={'REAL' if real else 'STUB'} reps={args.reps} tasks={len(tasks)} "
          f"scratch={scratch} topology={topo or '?'} out={results_path}")
    arm_seq = _arm_block_sequence(args.reps)
    traces_dir = out_dir / "traces"
    # (d) run-level meta — config + orchestrator-checkout-unchanged proof (hard invariant).
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": "real" if real else "stub",
        "reps": args.reps, "max_turns": args.max_turns,
        "arm_sequence": ["on" if b else "off" for b in arm_seq],
        "tasks": [t["id"] for t in tasks], "topology_hash": topo,
        "orch_head_before": _orch_head(), "scratch_root": str(scratch),
    }, indent=2))
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
                trace_mark, log_mark = _mark(TURN_TRACE), _mark(ORCH_LOG)
                t0 = time.time()
                if real:
                    info = _chat(task["prompt"], max_turns=args.max_turns,
                                 session_id=f"bep-{task['id']}-{arm}-{block_idx}")
                else:
                    _apply_solution(scratch, solutions[task["id"]])  # dry-run reference edit
                    info = {"answer": "[STUB]", "latency_s": round(time.time() - t0, 3),
                            "turns": None, "prompt_tokens": None}
                quality = _run_verifier(scratch, task["verifier_cmd"])
                trace = _trace_slice(trace_mark, traces_dir / f"{task['id']}-{arm}-blk{block_idx}.jsonl")
                row = {
                    "block": block_idx, "arm": arm, "task": task["id"], "kind": task["kind"],
                    "batch_edit_mode": batch_on, "interleaved_edit_rider": (not batch_on),
                    "topology_hash": topo,
                    "quality_pass": quality, "latency_s": info["latency_s"],
                    "turns": info.get("turns"), "prompt_tokens": info.get("prompt_tokens"),
                    "touched_files": _git_touched(scratch),
                    "batch_edit_states": (_batch_edit_states(log_mark) if real else {}),
                    "trace": trace,
                    "answer_preview": info["answer"], "mode": "real" if real else "stub",
                }
                rf.write(json.dumps(row) + "\n")
                rf.flush()  # incremental persistence
                n_written += 1
                print(f"  block{block_idx} arm={arm} {task['id']:22} q={'PASS' if quality else 'fail'} "
                      f"lat={info['latency_s']}s t={info.get('turns')} touched={len(row['touched_files'])} "
                      f"be={row['batch_edit_states'] or '-'}")

    # confirm the orchestrator checkout was untouched by the run (hard invariant)
    meta = json.loads(meta_path.read_text())
    meta["orch_head_after"] = _orch_head()
    meta["orch_checkout_unchanged"] = bool(meta["orch_head_after"]) and (
        meta["orch_head_after"] == meta["orch_head_before"])
    meta_path.write_text(json.dumps(meta, indent=2))

    # aggregate + informational BEP-2 gate (final decision is the operator's)
    rows = _load_jsonl(results_path)
    agg: dict[str, dict] = {}
    for arm in ("off", "on"):
        a = [r for r in rows if r["arm"] == arm]
        if not a:
            continue
        q = sum(r["quality_pass"] for r in a) / len(a)
        lats = sorted(r["latency_s"] for r in a)
        agg[arm] = {"n": len(a), "quality": q, "median_lat": lats[len(a) // 2]}
        print(f"[agg] arm={arm}: n={len(a)} quality={q:.0%} median_lat={agg[arm]['median_lat']}s")
    if "off" in agg and "on" in agg and agg["off"]["median_lat"]:
        lat_delta = (agg["off"]["median_lat"] - agg["on"]["median_lat"]) / agg["off"]["median_lat"]
        q_delta_pp = (agg["on"]["quality"] - agg["off"]["quality"]) * 100
        on_states: dict[str, int] = {}
        for r in rows:
            if r["arm"] == "on":
                for k, v in (r.get("batch_edit_states") or {}).items():
                    on_states[k] = on_states.get(k, 0) + v
        print(f"[gate] batch median-latency {lat_delta:+.0%} faster (need >=+15%), "
              f"quality {q_delta_pp:+.1f}pp (need >=-1), on-arm batch_edit_states={on_states or '-'}")
    print(f"[bep_ab] wrote {n_written} rows → {results_path} (meta.json + traces/ alongside; "
          f"orch_checkout_unchanged={meta.get('orch_checkout_unchanged')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
