#!/usr/bin/env python3
"""KPF-26.1: task-outcome check for byte-stable trunks (KPF-21/25).  Queue row ``s5-kpf261-outcome``.

What it measures (docs/design/kpf25-byte-stable-trunks-audit.md, last paragraph: "Under FORK the persona moves
from the system prompt that ``llm_call`` prepends into the user prompt, after the trunk: a semantic change.
KPF-26 must check task outcome on persona-bearing waves"; handoff KPF-26: "task outcome unchanged on the
replay set vs the pre-P2 path"):

    arm A   legacy prompt   StepExecutor._build_step_prompt with the KPF-25 flags OFF (``## Task`` first, input
                            keys in plan order) and the persona text PREPENDED, exactly what
                            ``LLMPrimitives._apply_persona_prefix`` does.
    arm B   byte-stable     the same step with ORCHESTRATOR_PREFIX_INDEX=1 + ORCHESTRATOR_PREFIX_INDEX_FORK=1:
                            shared-first order, input keys sorted, persona placed after the trunk.
    arm A2  legacy again    the CONTROL: A vs A2 is the run-to-run noise (batching, scheduling), same prompts.

The prompts come from the REAL builder (the flags are toggled in-process), so a change to the builder changes
this measurement.  Only the transport is stubbed: a PRIVATE llama-server that the launcher started and whose
port this process proves it owns.  Never starts a server, never touches the production stack, never :8000.
Not measured here (needs the P1 server and the stack, i.e. the rest of KPF-26): trunk-prefilled-once timings,
child TTFT, peak unique cells, admission waits.

PREDECLARED DECISION RULE (frozen before any measurement; ``summary.json`` repeats it verbatim).  Source: handoff
kv-prefix-fork-and-paged-attention.md KPF-26.1: "predeclared rule = outcome rate within the flag-off rerun-to-rerun
spread, any regression keeps the reorder off".  Reading, fixed here: a REGRESSION is a pass-rate drop of B below A that
exceeds the A-vs-A2 spread; any regression keeps the reorder OFF.
  unit      a case = (wave, step, rep); graded pass/fail by a deterministic regex checker.  EVERY case is counted:
            a transport error or timeout is a fail, never dropped.
  frozen    ``manifest.json`` (task ids + SHA-256 of the exact prompt of each arm) is written before the first request;
            a resume whose prompts differ aborts (rc 8).  Waves come from a fixed seed, none was used to pick the order.
  gates     INVALID (no verdict) unless ALL hold:
              * treatment applied: in every wave with >= 2 siblings the byte-stable arm's sibling prompts share a
                strictly longer common prefix than the legacy arm's, AND a persona-bearing step has its persona text
                after the trunk in B and before it in A;
              * >= MIN_CASES cases (40) and >= half the steps carry a persona;
              * baseline (A) pass rate >= MIN_BASELINE_PASS (0.60), so there is something to regress;
              * transport errors in the BASELINE arms A and A2 <= 5% (errors in B are regressions, not a void).
  metric    rate(X) = pass rate of arm X.  spread = |rate(A) - rate(A2)| (the flag-off rerun-to-rerun spread).
            drop = rate(A) - rate(B).  Diagnostics: regress/improve = pass->fail / fail->pass flips A->B, ctrl = A->A2
            pass->fail flips, one-sided sign-test p.
  min_effect  EXACT identity is the headline (UNCHANGED = zero flips A->B); the tolerance is the measured spread and
            nothing else.  No percentage tolerance exists.
  verdict   UNCHANGED      regress == 0
            WITHIN_SPREAD  regress > 0 and drop <= spread
            CHANGED        drop > spread   -> reorder_decision KEEP_OFF
            INVALID        a gate failed   -> no decision
            reorder_decision: KEEP_ON_ELIGIBLE for UNCHANGED / WITHIN_SPREAD in EVERY window (the handoff needs two).
  class     correctness: outcomes are compared, latency is recorded but never gates.

Modes: ``--mock-llm`` scripted answers (unit tests, plumbing, ``--dry-run``); live needs WINDOW_CONFIRMED=1,
``--server-url`` on a free private loopback port and ``--server-pid`` owning it.  Resumable: ``cases.jsonl`` is
append-only and a (rep, arm, wave, step) already written is skipped.  ``DONE`` is written last.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MIN_CASES = 40
MIN_BASELINE_PASS = 0.60
MAX_TRANSPORT_ERR = 0.05
ARMS = ("A", "B", "A2")                      # A legacy, B byte-stable, A2 legacy control (ABA order)
FLAG_ENVS = ("ORCHESTRATOR_PREFIX_INDEX", "ORCHESTRATOR_PREFIX_INDEX_FORK")
PRODUCTION_PORTS = set(range(8000, 8200))
RULE_TEXT = (
    "spread=|rate(A)-rate(A2)|; drop=rate(A)-rate(B). min_effect EXACT identity headline, tolerance = measured spread only. "
    "UNCHANGED regress==0 (zero pass->fail flips A->B); WITHIN_SPREAD regress>0 and drop<=spread; CHANGED drop>spread (reorder KEEP_OFF); "
    f"INVALID if a gate fails: treatment applied, cases>={MIN_CASES}, persona share>=0.5, baseline pass>={MIN_BASELINE_PASS}, "
    f"baseline-arm transport errors<={MAX_TRANSPORT_ERR:.0%}. Every case counted, errors are fails. Two windows needed for KEEP_ON_ELIGIBLE.")

PERSONAS = ("security_auditor", "code_reviewer", "test_designer", "data_analyst", "technical_writer",
            "performance_optimizer", None)    # None: persona-less steps are in the set too


# ---------------------------------------------------------------------------------------------- replay set
# Window-1 diagnosis (DIAGNOSIS-win1.md): with thinking OFF and "reply with the number only" the model must do 14-row
# sums / group sums / counts / rankings in ONE token; arm-A failed 49% of cases (aggregates 0-62% pass, lookups 75-88%),
# every failure a plausible wrong value, none truncated.  The answer is derivable; the format denied any work budget.
# Fix: allow brief visible work, grade ONLY the final ``ANSWER:`` line.  Identical for all three arms (calibration uses arm A only).
FMT_NUM = " Show brief work, then end with a final line 'ANSWER: <number>'."
FMT_NAME = " Show brief work, then end with a final line 'ANSWER: <name>'."


def build_waves(n_waves: int = 8, n_steps: int = 6, seed: int = 26_1) -> list[dict]:
    """Deterministic persona-bearing delegate waves.  Each wave: one objective, two upstream step outputs (a table
    and distractor notes; the children list them in DIFFERENT orders, so key sorting is exercised) and
    ``n_steps`` sibling steps with a checkable answer.  The shared trunk is several hundred tokens."""
    rng = random.Random(seed)
    names = ["alder", "birch", "cedar", "dogwood", "elm", "fir", "gum", "hazel", "ironwood", "juniper", "kauri",
             "larch", "maple", "nettle", "oak", "pine", "quince", "rowan", "spruce", "teak"]
    waves = []
    for w in range(n_waves):
        rows = [(n, rng.randint(3, 99), rng.choice("XYZ")) for n in rng.sample(names, 14)]
        table = "name,qty,zone\n" + "\n".join(f"{n},{q},{z}" for n, q, z in rows)
        notes = " ".join(f"Note {i}: shipment {rng.randint(100, 999)} was inspected on day {rng.randint(1, 28)}."
                         for i in range(1, 9))
        total = sum(q for _, q, _ in rows)
        biggest = max(rows, key=lambda r: r[1])[0]
        smallest = min(rows, key=lambda r: r[1])[0]
        zone = rng.choice("XYZ")
        zsum = sum(q for _, q, z in rows if z == zone)
        over = rng.choice([30, 40, 50, 60])
        nover = sum(1 for _, q, _ in rows if q > over)
        k = rng.randint(2, 5)
        kth = sorted(rows, key=lambda r: (-r[1], r[0]))[k - 1][0]
        qs = [
            (f"Add up every qty value in the table.{FMT_NUM}", rf"\b{total}\b"),
            (f"Which name has the largest qty?{FMT_NAME}", rf"(?i)\b{biggest}\b"),
            (f"Which name has the smallest qty?{FMT_NAME}", rf"(?i)\b{smallest}\b"),
            (f"What is the total qty of rows in zone {zone}?{FMT_NUM}", rf"\b{zsum}\b"),
            (f"How many rows have qty greater than {over}?{FMT_NUM}", rf"\b{nover}\b"),
            (f"Ranking names by qty from largest (ties by name), which is number {k}?{FMT_NAME}",
             rf"(?i)\b{kth}\b"),
        ]
        steps = []
        for i in range(n_steps):
            q, expect = qs[i % len(qs)]
            order = ["s0_table", "s0_notes"] if (w + i) % 2 == 0 else ["s0_notes", "s0_table"]
            steps.append({"id": f"w{w}s{i}", "action": f"{q} (Shipment batch {w}.)", "inputs": order, "outputs": ["answer"],
                          "persona": PERSONAS[(w + i) % len(PERSONAS)], "expect": expect})
        waves.append({"wave_id": f"w{w}", "objective": "Answer questions about the shipment table in the prior context.",
                      "step_outputs": {"s0_table": table, "s0_notes": notes}, "steps": steps})
    return waves


# ------------------------------------------------------------------------------------------ prompt rendering
def _persona_text(persona: str | None) -> str | None:
    if not persona:
        return None
    from src.parallel_step_executor import _persona_system_text
    return _persona_system_text(persona)


def render(wave: dict, step: dict, arm: str) -> tuple[str, dict]:
    """The prompt ``StepExecutor._execute_step`` would hand ``llm_call`` for this step in the arm, with the persona
    handling of ``_execute_step``/``_apply_persona_prefix`` reproduced (personas feature on).  Returns
    (prompt as the server sees it, info).  Flags are toggled in-process around the real builder only."""
    from src.parallel_step_executor import StepExecutor

    saved = {k: os.environ.get(k) for k in FLAG_ENVS}
    try:
        for k in FLAG_ENVS:
            if arm == "B":
                os.environ[k] = "1"
            else:
                os.environ.pop(k, None)
        ex = StepExecutor.__new__(StepExecutor)       # _build_step_prompt reads only step_outputs; skips live-stack reads
        ex.step_outputs = dict(wave["step_outputs"])
        task_ir = {"objective": wave["objective"]}
        ptext = _persona_text(step.get("persona"))
        if arm == "B":
            prompt = ex._build_step_prompt(task_ir, step, persona_text=ptext)     # persona (if any) after the trunk
        else:
            prompt = ex._build_step_prompt(task_ir, step)
            if ptext:                                                              # llm_call prepends it
                prompt = f"{ptext.strip()}\n\n{prompt}"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return prompt, {"persona": step.get("persona"), "persona_resolved": ptext is not None}


def common_prefix_len(texts: list[str]) -> int:
    if len(texts) < 2:
        return len(texts[0]) if texts else 0
    n = min(len(t) for t in texts)
    for i in range(n):
        if len({t[i] for t in texts}) > 1:
            return i
    return n


def treatment_check(waves: list[dict]) -> dict:
    """The treatment must actually differ: B's sibling prompts share a longer trunk than A's in every >=2-step wave,
    and a resolved persona sits before the trunk in A and after it in B."""
    per_wave, persona_ok, shared_ok = {}, False, True
    for w in waves:
        a = [render(w, s, "A")[0] for s in w["steps"]]
        b = [render(w, s, "B")[0] for s in w["steps"]]
        pa, pb = common_prefix_len(a), common_prefix_len(b)
        per_wave[w["wave_id"]] = {"shared_chars_A": pa, "shared_chars_B": pb}
        if len(w["steps"]) >= 2 and not pb > pa:
            shared_ok = False
        for s, ta, tb in zip(w["steps"], a, b):
            pt = _persona_text(s.get("persona"))
            if pt and ta.startswith(pt.strip()) and not tb.startswith(pt.strip()) and pt.strip() in tb:
                persona_ok = True
    return {"per_wave": per_wave, "shared_prefix_longer_in_B_every_wave": shared_ok,
            "persona_moved_after_trunk": persona_ok, "applied": bool(shared_ok and persona_ok)}


# ----------------------------------------------------------------------------------------------- transport
def listening_pid(port: int) -> int | None:
    out = subprocess.run(["ss", "-H", "-ltnp", f"sport = :{port}"], capture_output=True, text=True).stdout
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def prove_port_ownership(url: str, server_pid: int) -> int:
    m = re.match(r"^https?://(?:127\.0\.0\.1|localhost):(\d+)$", url.rstrip("/"))
    if not m:
        sys.exit(f"--server-url must be a loopback URL with a port, got {url!r}")
    port = int(m.group(1))
    if port in PRODUCTION_PORTS:
        sys.exit(f"refusing: port {port} is a production stack port")
    owner = listening_pid(port)
    if owner != server_pid:
        sys.exit(f"refusing: port {port} is owned by pid {owner}, not --server-pid {server_pid}")
    return port


class HttpClient:
    """POST one prompt as a single user message to /v1/chat/completions (the frontdoor chat path), greedy, thinking off."""

    def __init__(self, url: str, max_tokens: int = 1024, timeout: float = 600.0, seed: int = 261):
        self.url, self.max_tokens, self.timeout, self.seed = url.rstrip("/"), max_tokens, timeout, seed

    def __call__(self, prompt: str) -> dict:
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "seed": self.seed,
                "max_tokens": self.max_tokens, "stream": False, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        ch = d["choices"][0]
        t = d.get("timings") or {}
        return {"answer": ch["message"].get("content") or "", "finish": ch.get("finish_reason"),
                "elapsed_s": round(time.monotonic() - t0, 3), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n")}


def mock_client_for(waves: list[dict]) -> Callable[[str], dict]:
    """Oracle: finds the step whose action text is in the prompt and answers its expected token.  Plumbing only (the
    prompt arms cannot change an oracle's answer, so a mock run is UNCHANGED by construction, never a finding)."""
    table = [(s["action"], re.sub(r"\\b|\(\?i\)", "", s["expect"])) for w in waves for s in w["steps"]]

    def client(prompt: str) -> dict:
        ans = next((e for act, e in table if act in prompt), "")
        return {"answer": ans, "finish": "stop", "elapsed_s": 0.0, "prompt_n": len(prompt) // 4, "cache_n": 0}
    return client


# --------------------------------------------------------------------------------------------------- grading
def final_answer(answer: str) -> str:
    """The text after the LAST ``ANSWER:`` marker.  No marker: a short bare reply (<=40 chars, the oracle/legacy form) is
    taken whole; a long reply without the marker has no answer (intermediate numbers in the work must never pass)."""
    a = answer or ""
    hits = re.findall(r"(?is)\banswer\s*:\s*(.*)", a)
    if hits:
        return hits[-1].strip().strip("*`'\" ")
    return a.strip() if len(a.strip()) <= 40 else ""


def grade(step: dict, answer: str) -> bool:
    return bool(re.search(step["expect"], final_answer(answer)))


def decide(cases: list[dict], treatment: dict, n_steps_persona: int, n_steps: int) -> dict:
    """Pure function of the case table.  See the module docstring for the frozen rule."""
    by: dict = {}
    for c in cases:
        by.setdefault((c["rep"], c["wave"], c["step"]), {})[c["arm"]] = c
    keys = [k for k, v in by.items() if all(a in v for a in ARMS)]
    n = len(keys)

    def p(k, a):
        return bool(by[k][a]["pass"])

    base_cases = [c for c in cases if c["arm"] in ("A", "A2")]
    base_errs = sum(1 for c in base_cases if c.get("error"))
    b_errs = sum(1 for c in cases if c["arm"] == "B" and c.get("error"))
    regress = sum(1 for k in keys if p(k, "A") and not p(k, "B"))
    improve = sum(1 for k in keys if not p(k, "A") and p(k, "B"))
    ctrl = sum(1 for k in keys if p(k, "A") and not p(k, "A2"))
    ctrl_improve = sum(1 for k in keys if not p(k, "A") and p(k, "A2"))
    rate = {a: (sum(1 for k in keys if p(k, a)) / n) if n else None for a in ARMS}
    spread = abs(rate["A"] - rate["A2"]) if n else None
    drop = (rate["A"] - rate["B"]) if n else None
    base = rate["A"] or 0.0
    same_text = (sum(1 for k in keys if by[k]["A"]["answer"].strip() == by[k]["B"]["answer"].strip()) / n) if n else 0.0
    ctrl_same_text = (sum(1 for k in keys if by[k]["A"]["answer"].strip() == by[k]["A2"]["answer"].strip()) / n) if n else 0.0
    gates = {"treatment_applied": bool(treatment.get("applied")), "cases_ge_min": n >= MIN_CASES,
             "persona_share_ge_half": n_steps > 0 and n_steps_persona * 2 >= n_steps,
             "baseline_pass_ge_min": base >= MIN_BASELINE_PASS,
             "baseline_transport_errors_le_max": (base_errs / max(1, len(base_cases))) <= MAX_TRANSPORT_ERR}
    eps = 1e-12
    if not all(gates.values()):
        verdict = "INVALID"
    elif regress == 0:
        verdict = "UNCHANGED"
    elif drop <= spread + eps:
        verdict = "WITHIN_SPREAD"
    else:
        verdict = "CHANGED"
    decision = {"UNCHANGED": "KEEP_ON_ELIGIBLE", "WITHIN_SPREAD": "KEEP_ON_ELIGIBLE", "CHANGED": "KEEP_OFF"}.get(verdict)
    from math import comb
    d = regress + improve
    sign_p = sum(comb(d, i) for i in range(regress, d + 1)) / (2 ** d) if d else 1.0
    r4 = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {"verdict": verdict, "reorder_decision": decision, "gates": gates, "n_cases": n, "regress": regress,
            "improve": improve, "ctrl_regress": ctrl, "ctrl_improve": ctrl_improve, "baseline_pass_rate": r4(base),
            "pass_rate": {a: r4(v) for a, v in rate.items()}, "spread": r4(spread), "drop": r4(drop),
            "text_identical_A_B": round(same_text, 4), "text_identical_A_A2_control": round(ctrl_same_text, 4),
            "baseline_transport_errors": base_errs, "B_transport_errors": b_errs,
            "incomplete_cases": len(cases) - 3 * n if cases else 0,
            "sign_test_p_regress": round(sign_p, 6), "rule": RULE_TEXT}


def freeze_manifest(waves: list[dict], out: Path, seed: int | None) -> dict:
    """Task ids + prompt digests, written once before the first request; a resume must reproduce it exactly."""
    import hashlib
    tasks = []
    for w in waves:
        for s in w["steps"]:
            pa, pb = render(w, s, "A")[0], render(w, s, "B")[0]
            tasks.append({"wave": w["wave_id"], "step": s["id"], "persona": s.get("persona"), "expect": s["expect"],
                          "sha256_A": hashlib.sha256(pa.encode()).hexdigest(), "sha256_B": hashlib.sha256(pb.encode()).hexdigest()})
    digest = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    m = {"seed": seed, "n_tasks": len(tasks), "digest": digest, "tasks": tasks}
    path = out / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("digest") != digest:
            raise ManifestMismatch(f"{path}: frozen digest {old.get('digest')} != current {digest} (prompts changed since the first request)")
        return old
    path.write_text(json.dumps(m, indent=1))
    return m


class ManifestMismatch(Exception):
    pass


# ------------------------------------------------------------------------------------------------- the run
def run(waves: list[dict], client: Callable[[str], dict], out: Path, reps: int, concurrency: int,
        deadline: float | None = None, stopfile: str | None = None, seed: int | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = freeze_manifest(waves, out, seed)
    path = out / "cases.jsonl"
    done = set()
    cases: list[dict] = []
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                c = json.loads(ln)
                cases.append(c)
                done.add((c["rep"], c["arm"], c["wave"], c["step"]))
    treatment = treatment_check(waves)
    (out / "treatment.json").write_text(json.dumps(treatment, indent=1))
    stopped = None

    def should_stop() -> str | None:
        if stopfile and os.path.exists(stopfile):
            return "stopfile"
        if deadline is not None and time.time() >= deadline:
            return "deadline"
        return None

    def one(rep: int, arm: str, w: dict, s: dict) -> dict:
        prompt, info = render(w, s, arm)
        try:
            r = client(prompt)
            err = None
        except Exception as e:  # noqa: BLE001
            r, err = {"answer": "", "finish": None, "elapsed_s": None, "prompt_n": None, "cache_n": None}, repr(e)[:200]
        return {"rep": rep, "arm": arm, "wave": w["wave_id"], "step": s["id"], "pass": grade(s, r["answer"]) and not err,
                "answer": r["answer"], "finish": r["finish"], "elapsed_s": r["elapsed_s"], "prompt_n": r["prompt_n"],
                "cache_n": r["cache_n"], "persona": info["persona"], "prompt_chars": len(prompt), "error": err}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for rep in range(reps):
            for arm in ARMS:                                   # ABA inside every rep: drift hits both arms alike
                for w in waves:                                # one wave = one fan-out = concurrent siblings
                    todo = [s for s in w["steps"] if (rep, arm, w["wave_id"], s["id"]) not in done]
                    if not todo:
                        continue
                    stopped = should_stop()
                    if stopped:
                        break
                    res = list(ex.map(lambda s: one(rep, arm, w, s), todo))
                    with path.open("a") as f:
                        for c in res:
                            f.write(json.dumps(c) + "\n")
                            cases.append(c)
                    done.update((rep, arm, w["wave_id"], c["step"]) for c in res)
                if stopped:
                    break
            if stopped:
                break
    n_steps = sum(len(w["steps"]) for w in waves)
    n_pers = sum(1 for w in waves for s in w["steps"] if s.get("persona"))
    complete = all((r, a, w["wave_id"], s["id"]) in done for r in range(reps) for a in ARMS for w in waves for s in w["steps"])
    summary = decide(cases, treatment, n_pers, n_steps)
    summary.update({"manifest_digest": manifest["digest"], "seed": seed, "complete": complete, "stopped": stopped, "reps": reps, "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if complete:
        (out / "DONE").write_text(summary["finished_utc"] + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--waves", help="JSON file of waves (default: the built-in deterministic set)")
    ap.add_argument("--n-waves", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=261, help="wave-generator seed (window 1 = 261, window 2 = 262)")
    ap.add_argument("--concurrency", type=int, default=4, help="sibling requests in flight (match the server -np)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--server-url")
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="render + treatment check only; no requests")
    ap.add_argument("--deadline-epoch", type=float, help="stop starting new waves at this epoch")
    ap.add_argument("--stopfile")
    a = ap.parse_args(argv)
    waves = json.loads(Path(a.waves).read_text()) if a.waves else build_waves(a.n_waves, a.n_steps, a.seed)
    out = Path(a.out_dir)
    if a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        t = treatment_check(waves)
        (out / "treatment.json").write_text(json.dumps(t, indent=1))
        print(json.dumps({"dry_run": True, "waves": len(waves), "cases_per_arm_per_rep": sum(len(w["steps"]) for w in waves),
                          "treatment_applied": t["applied"]}))
        return 0 if t["applied"] else 3
    if a.mock_llm:
        client = mock_client_for(waves)
    else:
        if os.environ.get("WINDOW_CONFIRMED") != "1":
            sys.exit("refusing: live mode needs WINDOW_CONFIRMED=1 (a granted window)")
        if not a.server_url or a.server_pid is None:
            sys.exit("refusing: live mode needs --server-url and --server-pid (a private server this launcher owns)")
        prove_port_ownership(a.server_url, a.server_pid)
        client = HttpClient(a.server_url, a.max_tokens)
    try:
        s = run(waves, client, out, a.reps, a.concurrency, a.deadline_epoch, a.stopfile, None if a.waves else a.seed)
    except ManifestMismatch as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 8
    print(json.dumps({k: s[k] for k in ("verdict", "reorder_decision", "n_cases", "regress", "improve", "ctrl_regress", "complete", "stopped")}))
    return 0 if s["complete"] and s["verdict"] != "INVALID" else (4 if s["complete"] else 0)


if __name__ == "__main__":
    sys.exit(main())
