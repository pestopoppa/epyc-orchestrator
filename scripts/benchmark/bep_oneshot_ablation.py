#!/usr/bin/env python3
"""Protocol ablation (operator 2026-05-27): one-shot full-file edit, NO REPL/BEP choreography.
Give coder_escalation (Qwen3.6) the file contents in a single direct-mode prompt; parse the full
new files it outputs; verify with the SAME task checkers. Isolates 'can the model do the edit'
(one-shot) from 'can it navigate the REPL read->edit->FINAL protocol'.
  PASS one-shot but fails REPL  -> protocol/tooling problem.
  FAIL one-shot too             -> model capability / distribution shift more plausible.
"""
import json
import subprocess
import time
import re
import shutil
from pathlib import Path
import httpx

ORCH = str(Path(__file__).resolve().parents[2])
TASKS = [json.loads(line) for line in open(f"{ORCH}/data/bep_sandbox/tasks.jsonl") if line.strip()]
SCRATCH = Path("/mnt/raid0/llm/tmp/bep_oneshot/work")
OUTDIR = Path("/mnt/raid0/llm/tmp/bep_oneshot")
OUTDIR.mkdir(parents=True, exist_ok=True)

def _safe(rel: str):
    """Resolve a model-supplied path UNDER scratch, PRESERVING nested dirs; reject escapes
    (absolute paths or `..` traversal that leave the scratch root). Returns None if unsafe.
    (Replaces the earlier Path(rel).name flatten, which only worked for top-level files.)"""
    p = (SCRATCH / rel).resolve()
    try:
        p.relative_to(SCRATCH.resolve())
    except ValueError:
        return None
    return p

def reset_scratch(files):
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    for rel, content in (files or {}).items():
        p = _safe(rel)
        if p is None:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

def build_prompt(task):
    files = task.get('files') or {}
    p = [task['prompt'].strip(), ""]
    if files:
        p.append("Current file contents:")
        for rel, content in files.items():
            p.append(f"\n--- {rel} ---\n{content}")
    p.append("\nReturn the COMPLETE final content of EVERY file that should exist after your change, "
             "each in exactly this format (and nothing else):\n"
             "<<<FILE: filename>>>\n<full file content>\n<<<END>>>\n"
             "To delete a file, output: <<<DELETE: filename>>>")
    return "\n".join(p)

def parse_files(text):
    files = {m.group(1).strip(): m.group(2) for m in
             re.finditer(r'<<<FILE:\s*(.+?)>>>\n(.*?)\n<<<END>>>', text, re.DOTALL)}
    deletes = [d.strip() for d in re.findall(r'<<<DELETE:\s*(.+?)>>>', text)]
    return files, deletes

def chat_direct(prompt, timeout=200):
    payload = {"prompt": prompt, "force_role": "coder_escalation", "force_mode": "direct",
               "max_turns": 1, "cache_prompt": False, "mock_mode": False, "real_mode": True,
               "session_id": f"oneshot-{time.time()}"}
    d = httpx.post("http://localhost:8000/chat", json=payload, timeout=timeout).json()
    return d.get('answer') or d.get('response') or d.get('content') or ("[NOANSWER] "+json.dumps(d)[:300]), d

print("task | parsed_files | verifier | think? | raw_len | note")
for task in TASKS:
    tid = task['id']
    reset_scratch(task.get('files'))
    try:
        out, meta = chat_direct(build_prompt(task))
    except Exception as e:
        print(f"{tid}: CHAT-ERROR {type(e).__name__}: {e}")
        continue
    (OUTDIR/f"out_{tid}.txt").write_text(out)
    files, deletes = parse_files(out)
    rejected = []
    for rel, content in files.items():
        p = _safe(rel)
        if p is None:
            rejected.append(rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    for d in deletes:
        p = _safe(d)
        if p is not None:
            p.unlink(missing_ok=True)
        else:
            rejected.append(d)
    try:
        res = subprocess.run(task['verifier_cmd'], shell=True, cwd=SCRATCH,
                             capture_output=True, text=True, timeout=30)
        ok = ('PASS' in (res.stdout+res.stderr)) or (res.returncode == 0 and res.stdout.strip())
        vnote = (res.stdout.strip() or res.stderr.strip())[:50]
    except Exception as e:
        ok = False
        vnote = f"verifier-error {e}"
    print(f"{tid}: files={list(files.keys())} del={deletes}"
          f"{' REJECTED='+str(rejected) if rejected else ''} -> {'PASS' if ok else 'FAIL'} "
          f"think={'<think' in out} raw_len={len(out)} :: {vnote}")
