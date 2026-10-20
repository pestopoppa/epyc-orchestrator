#!/usr/bin/env python3
"""Q38-T10.1 CONTROL ARM: is per-request ``speculative.n_max: 0`` token-identical to a spec-off server?

Operator ruling 2026-10-20 (Q38-T10, per-role parity) needs this proved BEFORE the
``spec_off_bitwise_roles`` flag may be turned on (src/spec_policy.py). If it fails, the
fallback is a dedicated spec-off judge server (stack-change package).

NOT RUN by the author. Needs the MI210 exclusively (see "WINDOW" below). This script
starts its OWN llama-server processes on NON-PRODUCTION ports (default 18631/18632...),
captures their PIDs, and kills only those. It refuses production ports.

Arms (fresh server process each, same -np 2 shape so batching is comparable):
  off_A  spec-type none        : pass ALONE, pass BESIDE a concurrent non-speculating filler
  off_B  spec-type none        : same two passes (A/A: must be 100% identical to off_A or the run is VOID)
  ctl    DFlash2 n-max 7 server: (a) per-request n_max 0 ALONE, (b) per-request n_max 0 BESIDE a
         concurrent SPECULATING filler (slot 1, no override), (c) informational: no override
         (expected to diverge on some prompts; DF2-6/DF2-8) -- not part of the verdict.
Predeclared verdict (all must hold, else the control FAILS and falsifies "the per-request flag
delivers the bitwise contract without a stack change"):
  1. A/A: off_A == off_B token-for-token on every prompt, alone and beside filler (else VOID).
  2. ctl(a) tokens == off_A tokens on every prompt.   3. ctl(b) tokens == off_A tokens on every prompt.
  4. draft_n == 0 on every ctl(a)/ctl(b) response (the field was HONOURED, not identical by accident).
  5. ctl(c) shows draft_n > 0 on >=1 prompt and the filler responses show draft_n > 0 (server speculates).
  6. No response truncated differently / all stop_types equal to off_A's.
  7. Residency proven per arm: non-zero VRAM + busy sampled DURING the run, child pid in KFD.
Report is per prompt (never aggregate): id, equal?, first-divergence index, stop_type, draft_n.

WINDOW: the whole MI210 exclusively (gpu-quiet EXCLUSIVE, NO CPU region claim needed -- host
threads are pinned to the SMT siblings 184-191 as in DF2-17) for ~45-60 min (4 loads of the
27B Q8_0 target + DFlash2 draft, ~20 prompts x 256 tokens x ~7 passes). On 2026-10-20 the MI210
is held by another session (kpf18); schedule after it. Correctness-only: no timing claim.

EXACT COMMAND LINES (print without launching: add --print-commands):
  /workspace/repos/epyc-orchestrator/scripts/region-lock run --gpu-quiet exclusive --role bench \\
      --tag q38-t10-1-control -- python3 -I \\
      /workspace/repos/epyc-orchestrator/scripts/benchmark/q38_t10_1_control_arm.py \\
      --out /mnt/raid0/llm/tmp/q38-t10-1-control-<UTCSTAMP>
  # server argv (the script builds these; shown for the record), PORT=18631, build dir B below:
  taskset -c 184-191 $B/llama-server -m $TARGET -np 2 -c 32768 --no-kv-unified -t 8 -tb 8 -b 2048 -ub 2048 \\
      -ctk q8_0 -ctv q8_0 --device ROCm0 -ngl 99 -fa on --host 127.0.0.1 --port 18631 --metrics --slots \\
      --spec-type none                                                       # off_A / off_B
  ... same + -md $DRAFT -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7   # ctl
  request body: /completion {prompt, n_predict 256, temperature 0, top_k 1, top_p 1, min_p 0,
      typical_p 1, repeat_penalty 1, seed 42, cache_prompt false, return_tokens true,
      samplers [top_k, temperature], id_slot 0 [, "speculative.n_max": 0 on ctl (a)/(b)]}
  LD_LIBRARY_PATH=$B:/opt/rocm/lib (set per process; the three ggml generations differ).
Prompts: scripts/benchmark/q38_t10_1_control_prompts.json (20 prompts, staged by the GPU-staging
agent; sha256 recorded in the result). Extend the corpus to >=59 graded prompts for the serving
gate half of Q38-T10.1; this control only needs token identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROD_PORTS = {8000, 8070, 8074, 8080, 8083, 8090, 8091, 8092, 8093, 8094, 8095, 9000, 9002, 8180, 8100}
HOST = "127.0.0.1"
SPEC_KEY = "speculative.n_max"  # flat dotted llama-server schema key (tools/server/server-schema.cpp)
B = Path(os.environ.get("Q38T101_BUILD_BIN", "/mnt/raid0/llm/kernels/builds/gpu-20261006-4348de400/bin"))
TARGET = os.environ.get("Q38T101_TARGET", "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
DRAFT = os.environ.get("Q38T101_DRAFT", "/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf")
LINKAGE = "/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
VRAM = "/sys/class/drm/card2/device/mem_info_vram_used"
BUSY = "/sys/class/drm/card2/device/gpu_busy_percent"
KFD = "/sys/class/kfd/kfd/proc"
FILLER = ("Write a long, detailed essay on the history of numerical linear algebra, covering "
          "Gaussian elimination, LU, QR, SVD and iterative methods, with worked examples.")


def argv_for(port: int, kind: str) -> list[str]:
    a = ["taskset", "-c", "184-191", str(B / "llama-server"), "-m", TARGET, "-np", "2", "-c", "32768",
         "--no-kv-unified", "-t", "8", "-tb", "8", "-b", "2048", "-ub", "2048", "-ctk", "q8_0", "-ctv", "q8_0",
         "--device", "ROCm0", "-ngl", "99", "-fa", "on", "--host", HOST, "--port", str(port), "--metrics", "--slots"]
    if kind == "none":
        a += ["--spec-type", "none"]
    else:
        a += ["-md", DRAFT, "-ngld", "99", "--spec-type", "draft-dflash", "--spec-draft-n-max", "7"]
    return a


def body(prompt: str, n: int, slot: int, n_max: int | None) -> dict:
    b = {"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
         "typical_p": 1.0, "repeat_penalty": 1.0, "seed": 42, "stream": False, "cache_prompt": False,
         "return_tokens": True, "samplers": ["top_k", "temperature"], "id_slot": slot}
    if n_max is not None:
        b[SPEC_KEY] = n_max
    return b


def http(port: int, path: str, payload: dict | None = None, timeout: int = 1800) -> dict:
    req = urllib.request.Request(f"http://{HOST}:{port}{path}", data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _rd(p: str) -> int:
    try:
        return int(Path(p).read_text().strip())
    except Exception:
        return -1


class Sampler:
    """Residency sampled DURING the run (never after)."""

    def __init__(self, pid_fn):
        self.pid_fn, self.s, self._stop, self._t = pid_fn, [], threading.Event(), None

    def __enter__(self):
        def loop():
            while not self._stop.wait(0.5):
                try:
                    kf = sorted(int(x) for x in os.listdir(KFD) if x.isdigit())
                except Exception:
                    kf = []
                pid = self.pid_fn()
                self.s.append((_rd(VRAM), _rd(BUSY), pid in kf if pid else False))
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)

    @property
    def proof(self) -> dict:
        if not self.s:
            return {"n": 0}
        return {"n": len(self.s), "vram_peak": max(x[0] for x in self.s), "busy_max": max(x[1] for x in self.s),
                "busy_nonzero_samples": sum(1 for x in self.s if x[1] > 0), "child_in_kfd_samples": sum(1 for x in self.s if x[2])}


def rec_of(pid: str, r: dict) -> dict:
    t = r.get("timings") or {}
    return {"id": pid, "tokens": r.get("tokens") or [], "stop_type": r.get("stop_type"), "truncated": r.get("truncated"),
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"), "content": r.get("content")}


def run_pass(port: int, prompts: list[dict], cap: int, n_max: int | None, filler_n_max: int | None, with_filler: bool) -> list[dict]:
    """One pass over the prompts on slot 0; optionally a concurrent filler loop on slot 1."""
    stop, filler_recs = threading.Event(), []

    def filler():
        while not stop.is_set():
            try:
                r = http(port, "/completion", body(FILLER, 192, 1, filler_n_max))
                filler_recs.append(rec_of("filler", r))
            except Exception as e:  # noqa: BLE001
                filler_recs.append({"id": "filler", "error": repr(e)})
                time.sleep(1)

    th = threading.Thread(target=filler, daemon=True)
    if with_filler:
        th.start()
    out = []
    try:
        for q in prompts:
            out.append(rec_of(q["id"], http(port, "/completion", body(q["prompt"], cap, 0, n_max))))
    finally:
        stop.set()
        if with_filler:
            th.join(timeout=600)
    for r in out:
        r["filler_draft_n"] = sum((f.get("draft_n") or 0) for f in filler_recs)
        r["filler_requests"] = len(filler_recs)
    return out


def run_server(label: str, kind: str, port: int, outdir: Path, passes) -> dict:
    d = outdir / label
    d.mkdir(parents=True, exist_ok=True)
    argv = argv_for(port, kind)
    (d / "server_command.txt").write_text(" ".join(argv) + "\n")
    env = {k: v for k, v in os.environ.items() if k not in ("HSA_OVERRIDE_GFX_VERSION", "GGML_NOHUGEPAGE_PROCESS", "LLAMA_SPEC_EXACT")}
    env["LD_LIBRARY_PATH"] = f"{B}:/opt/rocm/lib"
    holder: dict = {}
    result = {"label": label, "kind": kind, "argv": argv, "passes": {}}
    with Sampler(lambda: holder.get("pid")) as smp, (d / "server.stderr").open("wb") as errf:
        proc = subprocess.Popen(argv, stdout=errf, stderr=subprocess.STDOUT, env=env)
        holder["pid"] = pid = proc.pid
        try:
            t0 = time.time()
            while time.time() - t0 < 900:
                if proc.poll() is not None:
                    raise RuntimeError(f"{label}: server exited rc={proc.returncode}")
                try:
                    if http(port, "/health", timeout=5).get("status") == "ok":
                        break
                except Exception:
                    time.sleep(3)
            else:
                raise RuntimeError(f"{label}: not ready in 900 s")
            for name, fn in passes:
                result["passes"][name] = fn(port)
                (d / f"pass_{name}.json").write_text(json.dumps(result["passes"][name]))
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=60)
    try:
        os.kill(pid, 0)
        result["server_dead_after"] = False
    except ProcessLookupError:
        result["server_dead_after"] = True
    result["server_pid"], result["residency"] = pid, smp.proof
    return result


def first_div(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(ref: list[dict], arm: list[dict]) -> list[dict]:
    rows = []
    for r, a in zip(ref, arm):
        fd = first_div(r["tokens"], a["tokens"])
        rows.append({"id": r["id"], "equal": fd is None and r["stop_type"] == a["stop_type"], "first_divergence": fd,
                     "stop_ref": r["stop_type"], "stop_arm": a["stop_type"], "draft_n": a["draft_n"], "n_ref": len(r["tokens"]), "n_arm": len(a["tokens"])})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(HERE / "q38_t10_1_control_prompts.json"))
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--port", type=int, default=18631)
    ap.add_argument("--print-commands", action="store_true", help="print the server argv + exact invocation and exit; starts nothing")
    a = ap.parse_args(argv)
    ports = [a.port + i for i in range(3)]
    if any(p in PROD_PORTS for p in ports):
        print("refusing: production port in range", ports, file=sys.stderr)
        return 2
    if a.print_commands:
        for kind in ("none", "dflash"):
            print(" ".join(argv_for(a.port, kind)))
        print(json.dumps(body("<prompt>", a.cap, 0, 0)))
        return 0
    prompts = json.loads(Path(a.prompts).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest()
    (out / "prompt_digest.txt").write_text(digest + "\n")
    cap = a.cap

    off_passes = [("alone", lambda p: run_pass(p, prompts, cap, None, None, False)),
                  ("beside_filler", lambda p: run_pass(p, prompts, cap, None, None, True))]
    ctl_passes = [("a_alone_n0", lambda p: run_pass(p, prompts, cap, 0, None, False)),
                  ("b_beside_spec_filler_n0", lambda p: run_pass(p, prompts, cap, 0, None, True)),
                  ("c_no_override_info", lambda p: run_pass(p, prompts, cap, None, None, False))]
    arms = {}
    for label, kind, port, passes in (("off_A", "none", ports[0], off_passes), ("off_B", "none", ports[1], off_passes),
                                       ("ctl", "dflash", ports[2], ctl_passes)):
        print(f"[{time.strftime('%H:%M:%S')}] arm {label}", flush=True)
        arms[label] = run_server(label, kind, port, out, passes)

    verdict = {"prompt_digest": digest, "n_prompts": len(prompts), "cap": cap, "checks": {}}
    aa = {k: compare(arms["off_A"]["passes"][k], arms["off_B"]["passes"][k]) for k in ("alone", "beside_filler")}
    verdict["aa"] = aa
    aa_ok = all(r["equal"] for rows in aa.values() for r in rows)
    ref = arms["off_A"]["passes"]["alone"]
    ca = compare(ref, arms["ctl"]["passes"]["a_alone_n0"])
    cb = compare(ref, arms["ctl"]["passes"]["b_beside_spec_filler_n0"])
    cc = compare(ref, arms["ctl"]["passes"]["c_no_override_info"])
    verdict.update(ctl_a=ca, ctl_b=cb, ctl_c_info=cc)
    c = verdict["checks"]
    c["aa_identical"] = aa_ok
    c["ctl_a_identical"] = all(r["equal"] for r in ca)
    c["ctl_b_identical"] = all(r["equal"] for r in cb)
    c["ctl_ab_draft_n_zero"] = all((r["draft_n"] or 0) == 0 for r in ca + cb)
    c["server_speculates_without_override"] = any((r["draft_n"] or 0) > 0 for r in cc)
    c["filler_speculated"] = all(r["filler_draft_n"] > 0 for r in arms["ctl"]["passes"]["b_beside_spec_filler_n0"])
    c["residency_all_arms"] = all(arms[k]["residency"].get("busy_nonzero_samples", 0) > 0 and arms[k]["residency"].get("child_in_kfd_samples", 0) > 0 for k in arms)
    c["servers_dead"] = all(arms[k]["server_dead_after"] for k in arms)
    if not aa_ok:
        verdict["verdict"] = "VOID (A/A not 100% identical)"
    elif all(c[k] for k in ("ctl_a_identical", "ctl_b_identical", "ctl_ab_draft_n_zero", "server_speculates_without_override", "filler_speculated", "residency_all_arms")):
        verdict["verdict"] = "PASS: per-request speculative.n_max=0 is token-identical to spec-off, alone and beside a speculating slot"
    else:
        verdict["verdict"] = "FAIL: falsifies the per-request flag; fallback = dedicated spec-off judge server (stack-change package)"
    verdict["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "passes"} for k, v in arms.items()}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    (out / "arms_raw.json").write_text(json.dumps(arms))
    print(json.dumps({"verdict": verdict["verdict"], "checks": c}))
    return 0 if verdict["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
