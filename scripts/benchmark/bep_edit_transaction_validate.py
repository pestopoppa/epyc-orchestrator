#!/usr/bin/env python3
"""Validate the edit-transaction MODULE end-to-end with the real coder on the BEP tasks.

Exercises src.edit_transaction.run_edit_transaction (assemble -> one-shot full-file prompt -> parse ->
TRANSACTIONAL apply w/ py_compile self-check + rollback) using the real coder_escalation via /chat
direct mode as the llm_call, then runs each task's deterministic verifier. Expect 5/5 — matching the
raw one-shot ablation, but now through the production module (proves the affordance, not just the shape).
Pause J6 first (feedback_no_concurrent_inference)."""
import json, subprocess, time, shutil, sys
from pathlib import Path
import httpx

ORCH = "/mnt/raid0/llm/epyc-orchestrator"
sys.path.insert(0, ORCH)
from src.edit_transaction import run_edit_transaction  # noqa: E402

TASKS = [json.loads(l) for l in open(f"{ORCH}/data/bep_sandbox/tasks.jsonl") if l.strip()]
SCRATCH = Path("/mnt/raid0/llm/tmp/bep_edittxn/work")


def reset(files):
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    for rel, content in (files or {}).items():
        (SCRATCH / rel).write_text(content)


def llm(prompt):
    payload = {"prompt": prompt, "force_role": "coder_escalation", "force_mode": "direct",
               "max_turns": 1, "cache_prompt": False, "mock_mode": False, "real_mode": True,
               "session_id": f"edittxn-{time.time()}"}
    d = httpx.post("http://localhost:8000/chat", json=payload, timeout=220).json()
    return d.get("answer") or d.get("response") or d.get("content") or ""


print("task | txn_ok | self_check | verifier | written | deleted | err")
npass = 0
for t in TASKS:
    reset(t.get("files"))
    targets = list((t.get("files") or {}).keys()) or None
    try:
        res, raw = run_edit_transaction(llm, t["prompt"], SCRATCH, target_files=targets)
    except Exception as e:
        print(f"{t['id']}: RUN-ERROR {type(e).__name__}: {e}"); continue
    try:
        v = subprocess.run(t["verifier_cmd"], shell=True, cwd=SCRATCH,
                           capture_output=True, text=True, timeout=30)
        vok = ("PASS" in (v.stdout + v.stderr)) or (v.returncode == 0 and v.stdout.strip())
    except Exception:
        vok = False
    npass += bool(res.ok and vok)
    print(f"{t['id']}: txn_ok={res.ok} verifier={'PASS' if vok else 'FAIL'} "
          f"written={res.written} del={res.deleted} rej={res.rejected} err={res.error[:50]}")
print(f"\nEDIT-TRANSACTION VALIDATION: {npass}/{len(TASKS)} tasks pass (txn applied + verifier PASS)")
