#!/usr/bin/env python3
"""LIVE wiring test for force_mode='edit' (the chat.py edit-transaction branch, 8b2).

Proves the SERVER path end-to-end — allowlist -> edit branch -> _execute_direct llm_call closure
-> transactional apply -> response shaping — which the module-level validation
(bep_edit_transaction_validate.py) does NOT exercise (it calls run_edit_transaction directly).

Assumes the orchestrator API was restarted with ORCHESTRATOR_EDIT_TRANSACTION=1 and
ORCHESTRATOR_EDIT_ROOT pointing at THIS scratch (one fixed root, reset per task). Pause J6 first
(feedback_no_concurrent_inference). A pass = response mode=='edit' (the branch fired) AND the
task's deterministic verifier PASSes against the scratch the server edited."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

ORCH = "/mnt/raid0/llm/epyc-orchestrator"
TASKS = {t["id"]: t for t in
         (json.loads(l) for l in open(f"{ORCH}/data/bep_sandbox/tasks.jsonl") if l.strip())}
ROOT = Path(os.environ["ORCHESTRATOR_EDIT_ROOT"])  # MUST match the API's env
# create (no read), read-first multi-file, and rename+delete -> covers write/edit/delete via server.
PROBE = ["t1_create_util", "t2_add_and_use", "t4_rename_module"]


def reset(files):
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    for rel, c in (files or {}).items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(c)


def chat_edit(prompt):
    payload = {"prompt": prompt, "force_role": "coder_escalation", "force_mode": "edit",
               "max_turns": 1, "cache_prompt": False, "mock_mode": False, "real_mode": True,
               "session_id": f"editwire-{time.time()}"}
    return httpx.post("http://localhost:8000/chat", json=payload, timeout=240).json()


print("task | answer_mode | verifier | answer")
npass = 0
for tid in PROBE:
    t = TASKS[tid]
    reset(t.get("files"))
    try:
        d = chat_edit(t["prompt"])
    except Exception as e:
        print(f"{tid}: CHAT-ERROR {type(e).__name__}: {e}")
        continue
    mode = d.get("mode")
    ans = (d.get("answer") or d.get("response") or "")[:90].replace("\n", " ")
    try:
        v = subprocess.run(t["verifier_cmd"], shell=True, cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        vok = ("PASS" in (v.stdout + v.stderr)) or (v.returncode == 0 and v.stdout.strip())
    except Exception:
        vok = False
    npass += bool(vok and mode == "edit")
    print(f"{tid}: mode={mode!r} verifier={'PASS' if vok else 'FAIL'} :: {ans}")
print(f"\nEDIT-MODE WIRING: {npass}/{len(PROBE)} (response mode=='edit' + verifier PASS)")
