#!/usr/bin/env python3
"""RTE-PFX-2 codified role recipe: the server the stagger probe runs against.

NOT written from memory. The command, env and NUMA prefix are DERIVED at call time from the
live registry (`orchestration/model_registry.yaml` via `RegistryLoader`, `stack_topology.yaml`
via `stack_numa`, env via `stack_env.build_launch_env`) by the same builder the stack uses
(`orchestrator_stack.build_server_command`), for role `frontdoor`, instance 0 (full shape,
cpu 0-95, `numactl --interleave=all`). Whatever the registry says today (binary, -np, -c, KV
quant, --cache-ram, MTP spec, template) is what is probed; a registry change is picked up, and
the drift guards below fail loudly instead of probing a different server silently.

DECLARED DELTAS from the production argv (nothing else is changed; every one is asserted):
  --port            -> the probe port (default 18971; refused if 8000-8999 or already LISTENing)
  --slot-save-path  -> a private directory (production's kv_slots dir is never touched)
  --host            unchanged (127.0.0.1)
DECLARED PROBE REQUIREMENTS (the registry must satisfy them, else REFUSED):
  -np >= N (default 4), --cache-ram present (cache_prompt reuse is the thing measured),
  binary == kernels/production/cpu (the served kernel), model file present.
DECLARED PREFIX FILE: scripts/benchmark/rte_pfx2_prefix.txt (+ .json sidecar with its sha256 and
provenance); sha256 is verified before every run.

Modes:
  dry-run   print the resolved launch plan + every check; starts nothing (no inference).
  run       needs WINDOW_CONFIRMED=1: NUMA pre-evict, launch OWN server (own PID captured, appended to
            FILLER_PIDFILE when set), prove OUR pid owns the port, verify_ggml_linkage.sh, run the
            probe, stop OUR pid (SIGTERM -> SIGKILL, verified dead).
Metric class: uncached prompt tokens per request are COUNTS from server timings (host noise does
not move them), so the run is filler-eligible under a build-role claim; TTFT is recorded but is
observation-grade there (not a timing claim).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREFIX = HERE / "rte_pfx2_prefix.txt"
SIDECAR = HERE / "rte_pfx2_prefix.json"
PROBE = HERE / "rte_pfx2_stagger_probe.py"
ROLE = "frontdoor"
INSTANCE = 0
DEFAULT_PORT = 18971
PRIVATE_DIR = Path("/mnt/raid0/llm/tmp/rte-pfx2")
EXPECTED_CPUS = "0-95"
LINKAGE = Path("/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh")
KERNEL_CPU = Path("/mnt/raid0/llm/kernels/production/cpu")
# Roots that may hold scripts/vidya/adapters/rte_pfx2_capture.py (the belief write side). First hit wins.
BELIEF_ROOTS = ("/workspace", "/mnt/raid0/llm/worktrees/root-pbe-3c4ca599")
FILLER_ROLE = "build"   # filler_run.sh claims `region-lock run --role build --tag filler-<id>`


def _flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def retarget(argv: list[str], port: int, slot_dir: str) -> list[str]:
    """Apply the declared deltas (port, slot-save-path) to the registry-derived argv."""
    out = list(argv)
    out[out.index("--port") + 1] = str(port)
    if "--slot-save-path" in out:
        out[out.index("--slot-save-path") + 1] = slot_dir
    return out


def check_requirements(argv: list[str], n: int, port: int) -> list[str]:
    """Return a list of violated requirements (empty = ok). Pure, unit-tested."""
    bad = []
    np_ = _flag(argv, "-np")
    if np_ is None or int(np_) < n:
        bad.append(f"-np {np_} < N={n}")
    if "--cache-ram" not in argv:
        bad.append("--cache-ram absent (prompt-cache setting must be declared in the run header)")
    if 8000 <= port <= 8999:
        bad.append(f"port {port} is in the production range 8000-8999")
    return bad


def prefix_sha_ok() -> tuple[bool, str]:
    side = json.loads(SIDECAR.read_text())
    got = hashlib.sha256(PREFIX.read_bytes()).hexdigest()
    return got == side["sha256"], got


def resolve(port: int, n: int) -> dict:
    """Derive the launch plan from the live registry/topology. Imports the stack lazily."""
    sys.path.insert(0, str(REPO))
    saved = sys.argv
    sys.argv = ["orchestrator_stack"]
    try:
        import scripts.server.orchestrator_stack as stack
        from scripts.server import stack_numa
        from scripts.server.stack_env import build_launch_env
        from src.registry_loader import RegistryLoader
    finally:
        sys.argv = saved
    rc = RegistryLoader().get_role(ROLE)
    argv = stack.build_server_command(rc, port, numa_instance=INSTANCE, prepare_runtime_dirs=False)
    prefix = stack_numa._numa_prefix(ROLE, INSTANCE)
    binary, ld_paths = stack._stack_prior_runtime_overrides(ROLE)
    env = build_launch_env(ROLE, {})
    stack._apply_runtime_requirements_env(env, binary_override=binary, ld_paths=ld_paths,
                                          preserve=stack._role_env_overrides(ROLE))
    evict = stack.pre_evict_gib_for_role(stack.NUMA_CONFIG.get(ROLE))
    slot_dir = str(PRIVATE_DIR / "kv_slots")
    return {"role": ROLE, "instance": INSTANCE, "registry_argv": argv,
            "argv": retarget(argv, port, slot_dir), "numa_prefix": prefix, "env": env,
            "pre_evict_gib": evict, "port": port}


def plan_checks(plan: dict, n: int) -> list[str]:
    bad = check_requirements(plan["argv"], n, plan["port"])
    pre = plan["numa_prefix"]
    if "-c" not in pre or pre[pre.index("-c") + 1] != EXPECTED_CPUS:
        bad.append(f"numa prefix cpus {pre} != {EXPECTED_CPUS} (full shape expected)")
    binary = plan["argv"][0]
    if not os.path.exists(binary):
        bad.append(f"binary missing: {binary}")
    elif os.path.realpath(Path(binary).parent) != os.path.realpath(KERNEL_CPU):
        bad.append(f"binary {binary} is not under kernels/production/cpu")
    model = _flag(plan["argv"], "-m")
    if not model or not os.path.exists(model):
        bad.append(f"model missing: {model}")
    ok, got = prefix_sha_ok()
    if not ok:
        bad.append(f"prefix file sha256 {got[:12]} != sidecar")
    return bad


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def window_claim(env: dict, start_utc: str, role: str = ROLE) -> dict:
    """Window the probe ran under, from what the filler launcher exports (nothing invented).

    Under filler_run.sh: window id = <FILLER_ID>@<FILLER_END_ISO>, claim = the region-lock invocation the
    launcher made (cpus from FILLER_CPUS, role build, tag filler-<id>). Standalone (no FILLER_ID) the
    claim is recorded as the explicit absence ``none:WINDOW_CONFIRMED=1`` so a reader can see it."""
    fid = env.get("FILLER_ID")
    if fid:
        wid = f"{fid}@{env.get('FILLER_END_ISO', 'unknown-end')}"
        claim = f"region-lock run --cpu-list {env.get('FILLER_CPUS', 'unknown')} --role {FILLER_ROLE} --tag filler-{fid}"
    else:
        wid = f"operator-confirmed@{start_utc}"
        claim = "none:WINDOW_CONFIRMED=1 (no region-lock claim recorded)"
    return {"window_id": wid, "region_lock_claim": claim, "start_utc": start_utc, "end_utc": None, "role": role}


def filter_unreported_timings(probe: dict) -> tuple[dict, list[dict]]:
    """Drop every TRIAL holding a request whose server timings are the probe's -1 placeholder.

    The writer refuses the whole run on one such row; a row cannot be removed alone (the trial's
    total_uncached_prompt_tokens sums all its requests), so the trial is skipped with its reason."""
    kept, skipped = [], []
    for t in probe.get("results", []):
        bad = [r.get("idx") for r in t.get("rows", [])
               if not (isinstance(r.get("prompt_n"), int) and r["prompt_n"] >= 0
                       and isinstance(r.get("cache_n"), int) and r["cache_n"] >= 0)]
        if bad:
            skipped.append({"arm": t.get("arm"), "trial": t.get("trial"), "request_idx": bad,
                            "reason": "server timings prompt_n/cache_n < 0 (not reported); not invented"})
        else:
            kept.append(t)
    return {**probe, "results": kept, "belief_skipped_trials": skipped}, skipped


def belief_capture(out: Path, window: dict, role: str, run_id: str) -> dict:
    """Write chunk_kv.beliefs.jsonl after a rc=0 probe. NEVER raises or changes the probe rc: any
    refusal is recorded in <out>/belief_status.json."""
    status: dict = {"written": None, "skipped_trials": [], "error": None}
    try:
        root = next((r for r in ([os.environ.get("EPYC_ROOT")] + list(BELIEF_ROOTS)) if r and
                     (Path(r) / "scripts/vidya/adapters/rte_pfx2_capture.py").is_file()), None)
        if root is None:
            raise RuntimeError("rte_pfx2_capture adapter not found in EPYC_ROOT, /workspace or the pinned root worktree")
        sys.path.insert(0, str(Path(root) / "scripts" / "vidya"))
        from adapters import rte_pfx2_capture as cap  # type: ignore
        probe = json.loads((out / "probe.json").read_text())
        filtered, skipped = filter_unreported_timings(probe)
        status["skipped_trials"] = skipped
        src = out / "probe.json"
        if skipped:   # the attested file is the one the rows were derived from
            src = out / "probe.belief.json"
            src.write_text(json.dumps(filtered, indent=1))
        path = cap.write_belief_measurements(out, run_id=run_id, producer="rte_pfx2_role_recipe.py",
                                             probe_path=src, header_path=out / "RUN_HEADER.json",
                                             role=role, window={k: window[k] for k in
                                             ("window_id", "region_lock_claim", "start_utc", "end_utc")})
        status["written"] = str(path)
    except Exception as e:  # recorded, never fatal
        status["error"] = f"{type(e).__name__}: {e}"
    (out / "belief_status.json").write_text(json.dumps(status, indent=1))
    return status


def listening(port: int) -> bool:
    r = subprocess.run(["ss", "-ltn", f"sport = :{port}"], capture_output=True, text=True)
    return "LISTEN" in r.stdout


def _dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True


def stop_own(pid: int) -> bool:
    """SIGTERM -> SIGKILL on the PID we captured, verified dead."""
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return True
        for _ in range(wait):
            time.sleep(1)
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            if _dead(pid):
                return True
    return _dead(pid)


def do_run(a, plan: dict) -> int:
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        print("REFUSED: needs a granted window (WINDOW_CONFIRMED=1)", file=sys.stderr)
        return 3
    bad = plan_checks(plan, a.n)
    if bad or listening(a.port):
        print("REFUSED:", bad or f"port {a.port} already listening", file=sys.stderr)
        return 4
    fcpus = os.environ.get("FILLER_CPUS")
    if fcpus is not None and fcpus != EXPECTED_CPUS and not a.allow_cpus_mismatch:
        print(f"REFUSED: grant cpus {fcpus} != {EXPECTED_CPUS} (full-shape recipe)", file=sys.stderr)
        return 4
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (PRIVATE_DIR / "kv_slots").mkdir(parents=True, exist_ok=True)
    if plan["pre_evict_gib"] > 0:
        sys.path.insert(0, str(REPO))
        from scripts.server.stack_numa_evict import pre_evict_nodes
        ok, msg = pre_evict_nodes(plan["pre_evict_gib"])
        print("numa-pre-evict:", ok, msg)
        if not ok:
            return 5
    env = {**os.environ, **plan["env"]}
    log = open(out / "server.log", "ab")
    proc = subprocess.Popen(plan["numa_prefix"] + plan["argv"], stdout=log, stderr=subprocess.STDOUT, env=env)
    pid = proc.pid
    if os.environ.get("FILLER_PIDFILE"):
        with open(os.environ["FILLER_PIDFILE"], "a") as f:
            f.write(f"{pid}\n")
    rc = 1
    try:
        lib = os.environ.get("FILLER_LIB")
        wait = ["bash", "-c", f'source "{lib}"; fl_wait_own_port {a.port} {pid} {a.health_timeout}'] if lib else None
        if wait:
            if subprocess.run(wait).returncode != 0:
                return 6
        else:   # standalone: same proof (health answers AND our pid owns the LISTEN socket)
            t0 = time.time()
            while True:
                if _dead(pid):
                    return 6
                h = subprocess.run(["curl", "-sf", f"http://127.0.0.1:{a.port}/health"], capture_output=True)
                o = subprocess.run(["ss", "-ltnp", f"sport = :{a.port}"], capture_output=True, text=True).stdout
                if h.returncode == 0 and f"pid={pid}," in o:
                    break
                if time.time() - t0 > a.health_timeout:
                    return 6
                time.sleep(5)
        lk = subprocess.run(["bash", str(LINKAGE), plan["argv"][0], str(KERNEL_CPU.parent)],
                            capture_output=True, text=True, env=env)
        (out / "linkage.txt").write_text(lk.stdout + lk.stderr)
        if lk.returncode != 0:
            print("REFUSED: ggml linkage FAIL", file=sys.stderr)
            return 7
        win = window_claim(os.environ, utc_now())
        (out / "RUN_HEADER.json").write_text(json.dumps({
            "schema": "epyc.rte_pfx2.header.v1", "argv": plan["argv"], "numa_prefix": plan["numa_prefix"],
            "registry_argv": plan["registry_argv"], "pid": pid,
            "cache_ram_mib": _flag(plan["argv"], "--cache-ram"), "slot_prompt_similarity": "server default (flag absent)",
            "prefix_sha256": prefix_sha_ok()[1], "n": a.n, "trials": a.trials, "seed": a.seed,
            "binary_mtime": time.ctime(os.path.getmtime(plan["argv"][0])),
            "role": ROLE, "window": win}, indent=1))
        rc = subprocess.run([sys.executable, str(PROBE), "--server", f"http://127.0.0.1:{a.port}", "--n", str(a.n),
                             "--trials", str(a.trials), "--prefix-file", str(PREFIX), "--seed", str(a.seed),
                             "--output", str(out / "probe.json")], env={**os.environ, "WINDOW_CONFIRMED": "1"}).returncode
        win["end_utc"] = utc_now()
        try:   # record the window in both result files (RUN_HEADER is rewritten whole; probe.json gains keys)
            hp = out / "RUN_HEADER.json"
            h = json.loads(hp.read_text()); h["window"] = win; hp.write_text(json.dumps(h, indent=1))
            if (out / "probe.json").is_file():
                pj = json.loads((out / "probe.json").read_text())
                pj.update(role=ROLE, window=win); (out / "probe.json").write_text(json.dumps(pj, indent=1))
        except Exception as e:
            print(f"window record failed: {e}", file=sys.stderr)
        if rc == 0:
            st = belief_capture(out, win, ROLE, f"rtepfx2-{out.name}")
            print("belief capture:", st["written"] or st["error"], f"(skipped trials: {len(st['skipped_trials'])})")
        return rc
    finally:
        dead = stop_own(pid)
        print(f"server pid {pid} dead (verified): {dead}")
        if not dead:
            rc = rc or 8


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dry-run", "run"])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--output-dir", default=str(PRIVATE_DIR / "run"))
    ap.add_argument("--health-timeout", type=int, default=900)
    ap.add_argument("--allow-cpus-mismatch", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "run":
        if os.environ.get("WINDOW_CONFIRMED") != "1":
            print("REFUSED: needs a granted window (WINDOW_CONFIRMED=1)", file=sys.stderr)
            return 3
    plan = resolve(a.port, a.n)
    if a.mode == "dry-run":
        bad = plan_checks(plan, a.n)
        if listening(a.port):
            bad.append(f"port {a.port} already listening")
        print(json.dumps({"plan": {k: plan[k] for k in ("role", "instance", "port", "pre_evict_gib")},
                          "command": plan["numa_prefix"] + plan["argv"],
                          "env": plan["env"], "prefix": {"file": str(PREFIX), "sha256_ok": prefix_sha_ok()[0]},
                          "probe": [sys.executable, str(PROBE), "--n", str(a.n), "--trials", str(a.trials)],
                          "violations": bad}, indent=1))
        return 1 if bad else 0
    return do_run(a, plan)


if __name__ == "__main__":
    sys.exit(main())
