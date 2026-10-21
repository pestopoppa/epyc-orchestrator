#!/usr/bin/env python3
"""HIL-15a / HIL-16a / HIL-18a shared flag-on trace driver (queue row ``s5-hil15a-flagon``).

One in-process driver (modelled on ``run_task_lg_parity.py``: real graph nodes, real REPL, private
backend) runs N labelled tasks with ALL THREE telemetry flags on and then does every offline step the
three HIL items need, so one inference window feeds all of them:

    ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING=1   (HIL-15a  -> auc_table)
    ORCHESTRATOR_STEP_EVENT_LOGGING=1            (HIL-13a  -> l2_scan --step-db)
    ORCHESTRATOR_REPLAY_RECORDING=1              (HIL-16a  -> replay_flip_run.py verify)

No new runtime flag is introduced: the three above are already declared in
``orchestration/runtime_flags.spec.yaml``; everything else here is a script argument or a
launch-environment variable of THIS script (WINDOW_CONFIRMED, PYTHONHASHSEED, ORCHESTRATOR_PATHS_TMP_DIR,
ORCHESTRATOR_REPLAY_CAS_DIR, ORCHESTRATOR_SERVER_URLS_<ROLE>, all pre-existing).

Never starts a server and never touches the production stack. Live mode needs a PRIVATE llama-server that
the operator/launcher already started on its own port; the driver proves port ownership (``ss`` pid ==
``--server-pid``), refuses any port the production config maps, and points EVERY role URL at it.

Modes
  --mock-llm   scripted LLM, no server, no window needed (unit tests, plumbing proof). Output is marked
               ``"mock": true`` and its AUC table is a plumbing artefact, never a finding.
  (default)    live: refuses unless WINDOW_CONFIRMED=1.

Steps (all recorded in ``<out-dir>/manifest.json``)
  1 guards            window, PYTHONHASHSEED pinned (re-exec), port ownership, reasoning-retry off
  2 run               N labelled runs; run id == task id == ``<task_key>-r<i>`` (the session-id join key)
  3 labels            deterministic checker per task -> ``labels.json`` + ``progress/outcomes.jsonl``
  4 join check        step-bearing runs vs runs with a joined outcome; any shortfall aborts the filing
  5 L2 report         ``l2_scan.py --step-db ... --belief-sidecar``
  6 AUC table         ``dependency_features.auc_table`` over the stored ``tool_dependency`` events
  7 replay verify     ``replay_flip_run.py verify`` on every recording
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FLAGS = ("ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING", "ORCHESTRATOR_STEP_EVENT_LOGGING", "ORCHESTRATOR_REPLAY_RECORDING")
# the second main LLM call of the reasoning-retry path is not recorded and diverges the replay (HIL-16a note)
RETRY_OFF = ("ORCHESTRATOR_REASONING_LENGTH_ALARM",)

#: >= 5 task keys, deterministic pass/fail checkers (regex over the final answer). Plumbing-grade default;
#: pass --tasks with a harder set so labels are not single-class on a real model.
DEFAULT_TASKS = [
    {"task_key": "arith-1", "prompt": "Compute 17 * 23 and give only the number.", "expect": r"\b391\b"},
    {"task_key": "arith-2", "prompt": "Compute (144 / 12) + 7 and give only the number.", "expect": r"\b19\b"},
    {"task_key": "fact-1", "prompt": "What is the capital of Australia? One word.", "expect": r"(?i)\bcanberra\b"},
    {"task_key": "str-1", "prompt": "Reverse the string 'orchestra' and give only the result.", "expect": r"\bartsehcro\b"},
    {"task_key": "list-1", "prompt": "Sort the numbers 9 3 7 1 5 ascending, comma separated.", "expect": r"1\s*,\s*3\s*,\s*5\s*,\s*7\s*,\s*9"},
    {"task_key": "count-1", "prompt": "How many letters are in the word 'refrigerator'? Give only the number.", "expect": r"\b12\b"},
]


def load_tasks(path: str | None) -> list[dict]:
    tasks = json.loads(Path(path).read_text()) if path else DEFAULT_TASKS
    keys = [t["task_key"] for t in tasks]
    if len(set(keys)) != len(keys):
        sys.exit("duplicate task_key in task set")
    for t in tasks:
        if not {"task_key", "prompt", "expect"} <= set(t):
            sys.exit(f"task {t.get('task_key')!r} needs task_key, prompt, expect")
        re.compile(t["expect"])
    return tasks


def grade(task: dict, answer) -> bool:
    return bool(re.search(task["expect"], "" if answer is None else str(answer)))


def listening_pid(port: int) -> int | None:
    out = subprocess.run(["ss", "-H", "-ltnp", f"sport = :{port}"], capture_output=True, text=True).stdout
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def prove_port_ownership(url: str, server_pid: int, production_ports: set[int]) -> int:
    m = re.match(r"^(?:full:)?https?://(?:127\.0\.0\.1|localhost):(\d+)", url)
    if not m:
        sys.exit(f"--server-url must be a loopback URL with a port, got {url!r}")
    port = int(m.group(1))
    if port in production_ports:
        sys.exit(f"refusing: port {port} is a production stack port")
    owner = listening_pid(port)
    if owner != server_pid:
        sys.exit(f"refusing: port {port} is owned by pid {owner}, not --server-pid {server_pid} (a foreign server may answer)")
    return port


def pin_hashseed(seed: str, no_reexec: bool) -> str:
    if os.environ.get("PYTHONHASHSEED") == seed:
        return "pinned"
    if no_reexec:
        return "UNPINNED (no-reexec)"
    os.environ["PYTHONHASHSEED"] = seed
    os.execv(sys.executable, [sys.executable, *sys.argv])
    return "unreachable"


def setup_env(out: Path, server_url: str | None) -> dict:
    """Everything that must be fixed BEFORE src.config is first imported/cached."""
    tmp, spill, cas = out / "tmp", out / "spill", out / "cas"
    for d in (tmp, spill, cas):
        d.mkdir(parents=True, exist_ok=True)
    for f in FLAGS:
        os.environ[f] = "1"
    for f in RETRY_OFF:
        os.environ[f] = "0"
    os.environ["ORCHESTRATOR_PATHS_TMP_DIR"] = str(tmp)
    os.environ["ORCHESTRATOR_REPLAY_CAS_DIR"] = str(cas)
    os.environ["ORCHESTRATOR_SESSION_LOG"] = "1"      # the production default; the replay seed records it
    os.environ["ORCHESTRATOR_TOOLS"] = "1"
    if server_url:
        import dataclasses
        from src.config.models import ServerURLsConfig
        for f in dataclasses.fields(ServerURLsConfig):
            os.environ[f"ORCHESTRATOR_SERVER_URLS_{f.name.upper()}"] = server_url
    from src.features import reset_features
    from src.trace import store
    db = out / "events.sqlite"
    store.ensure_schema(db).close()
    store.DEFAULT_DB_PATH = db                        # producers import it at call time
    reset_features()
    return {"tmp_dir": str(tmp), "spill_dir": str(spill), "cas_dir": str(cas), "db": str(db)}


def mock_deps(spill: str, script_for):
    from types import SimpleNamespace
    from src.graph.state import GraphConfig, TaskDeps
    from src.repl_environment import REPLConfig, REPLEnvironment

    class Reg:
        def invoke(self, tool_name, role=None, caller_type="direct", **kw): return f"ok:{tool_name}"
        def list_tools(self, role=None, context=None): return []
        def get_read_only_tools(self): return set()
        def get_chainable_tools(self): return set()

    def deps_for(task, run_id):
        it = iter(script_for(task, run_id))

        def llm(prompt, role="worker", **kw):
            head = prompt.lower()[:400]
            if "summar" in head and "turn" in head and "[Session" not in prompt[:20] and task["prompt"] not in prompt:
                return "SESSION-SUMMARY"          # auxiliary call (session summary), not a scripted turn
            return next(it, "FINAL('unscripted')")
        prims = SimpleNamespace(llm_call=llm, _early_stop_check=None, _last_inference_meta={"tokens": 7, "prompt_tokens": 90},
                                _backends=True, total_tokens_generated=0, total_prompt_eval_ms=0.0, total_generation_ms=0.0,
                                _last_predicted_tps=0.0, total_http_overhead_ms=0.0, get_cache_stats=lambda: {})
        repl = REPLEnvironment(context=task["prompt"], config=REPLConfig(timeout_seconds=10, spill_dir=spill),
                               role="frontdoor", task_id=run_id, tool_registry=Reg())
        return TaskDeps(primitives=prims, repl=repl, config=GraphConfig(max_turns=8))
    return deps_for


def live_deps(spill: str):
    """Same construction as run_task_lg_parity._build_live_task_deps, plus an explicit REPL spill dir."""
    from src.config import get_config
    from src.graph.state import GraphConfig, TaskDeps
    from src.llm_primitives import LLMPrimitives
    from src.repl_environment import REPLConfig, REPLEnvironment
    sys.path.insert(0, str(Path(__file__).parent))
    from run_task_lg_parity import _load_registry  # type: ignore

    cfg = get_config()
    prims = LLMPrimitives(mock_mode=False, server_urls=cfg.server_urls.as_dict(), registry=_load_registry(), num_slots=cfg.server.num_slots)
    if not getattr(prims, "_backends", None):
        sys.exit("no LLM backends: the private server URL did not resolve")

    def deps_for(task, run_id):
        repl = REPLEnvironment(context=task["prompt"], llm_primitives=prims, role="frontdoor", task_id=run_id,
                               config=REPLConfig(spill_dir=spill))
        gc = GraphConfig.from_config()
        gc.max_turns = gc.max_turns if gc.max_turns > 0 else 15
        return TaskDeps(primitives=prims, repl=repl, config=gc)
    return deps_for


_MOCK_ANS = {"arith-1": "391", "arith-2": "19", "fact-1": "Canberra", "str-1": "artsehcro", "list-1": "1, 3, 5, 7, 9", "count-1": "12"}


def default_mock_script(task, run_id):
    """Mock: a correct answer everywhere except count/list/str tasks on odd repeats (so the labels carry both classes)."""
    wrong = task["task_key"].split("-")[0] in ("count", "list", "str") and int(run_id.rsplit("-r", 1)[1]) % 2 == 1
    ans = "7" if wrong else _MOCK_ANS.get(task["task_key"], "x")
    return ['a = TOOL("write_file", path="/work/n.txt", text="x")\nprint(a)\n', 'print(peek(12))\n',
            'b = TOOL("read_file", path="/work/n.txt")\nprint(b)\n', 'c = TOOL("read_file", path="/work/n.txt")\nprint(c)\n', f'FINAL("{ans}")\n']


async def run_all(tasks, repeats, deps_for, state_cls, run_task):
    labels, results = {}, []
    for t in tasks:
        for i in range(repeats):
            rid = f"{t['task_key']}-r{i}"
            state = state_cls(task_id=rid, prompt=t["prompt"], context=t["prompt"])
            t0, err = time.time(), None
            try:
                res = await run_task(state, deps_for(t, rid), start_role=t.get("start_role", "frontdoor"))
                ans = getattr(res, "answer", None)
            except Exception as e:  # a crashed run is a labelled failure, not a hole in the table
                ans, err = None, f"{type(e).__name__}: {e}"[:300]
            ok = err is None and grade(t, ans)
            labels[rid] = [t["task_key"], ok]
            results.append({"run_id": rid, "task_key": t["task_key"], "pass": ok, "answer": None if ans is None else str(ans)[:300],
                            "error": err, "wall_s": round(time.time() - t0, 3)})
    return labels, results


def write_progress_log(out: Path, results) -> Path:
    p = out / "progress" / "outcomes.jsonl"
    p.parent.mkdir(exist_ok=True)
    with p.open("w") as f:
        for r in results:
            f.write(json.dumps({"event_type": "task_started", "task_id": r["run_id"]}) + "\n")
            f.write(json.dumps({"event_type": "task_completed" if r["pass"] else "task_failed", "task_id": r["run_id"],
                                "outcome_details": f"{r['wall_s']}s", "data": {"producer_role": "frontdoor"}}) + "\n")
    return p


def join_accounting(db: str, progress: Path) -> dict:
    from src.trace.l2.loaders import load_progress_logs, load_step_events
    outcomes = {t.trace_id: t for t in load_progress_logs([progress])}
    steps = list(load_step_events(db, outcomes))
    joined = sum(1 for t in steps if t.outcome != "unknown")
    return {"step_bearing_runs": len(steps), "joined_outcome": joined, "task_runs": len(outcomes)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--server-url")
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--hashseed", default="0")
    ap.add_argument("--no-reexec", action="store_true")
    ap.add_argument("--min-task-keys", type=int, default=5)
    ap.add_argument("--epyc-root", default=os.environ.get("EPYC_ROOT", "/workspace"), help="root checkout that holds scripts/vidya/adapters (belief-sidecar writers)")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)

    if not a.mock_llm:
        if os.environ.get("WINDOW_CONFIRMED") != "1":
            sys.exit("refusing: live mode runs inference. Set WINDOW_CONFIRMED=1 inside an operator-confirmed window.")
        if not (a.server_url and a.server_pid):
            sys.exit("live mode needs --server-url (private loopback server) and --server-pid (port-ownership proof)")
    tasks = load_tasks(a.tasks)
    if len(tasks) < a.min_task_keys:
        sys.exit(f"need >= {a.min_task_keys} task keys with labels, got {len(tasks)}")
    hs = pin_hashseed(a.hashseed, a.no_reexec)
    if hs.startswith("UNPINNED") and not a.mock_llm:
        sys.exit("live mode requires a pinned PYTHONHASHSEED")
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"schema": "hil15a_flagon_run/v1", "mock": a.mock_llm, "hashseed": hs, "flags": list(FLAGS),
                "reasoning_retry": "disabled via " + ",".join(RETRY_OFF), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not a.mock_llm:
        import dataclasses
        from src.config.models import ServerURLsConfig
        prod = {int(m.group(1)) for v in dataclasses.asdict(ServerURLsConfig()).values() if isinstance(v, str)
                for m in [re.search(r":(\d+)", v)] if m}
        manifest["server_port"] = prove_port_ownership(a.server_url, a.server_pid, prod)
        manifest["server_pid"] = a.server_pid
    paths = setup_env(out, None if a.mock_llm else a.server_url)
    manifest["paths"] = paths

    import asyncio
    from src.graph.graph import run_task
    from src.graph.state import TaskState
    deps_for = mock_deps(paths["spill_dir"], default_mock_script) if a.mock_llm else live_deps(paths["spill_dir"])
    labels, results = asyncio.run(run_all(tasks, a.repeats, deps_for, TaskState, run_task))
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    manifest["runs"] = results
    progress = write_progress_log(out, results)
    manifest["label_classes"] = sorted({str(bool(v[1])) for v in labels.values()})
    manifest["warnings"] = []
    if len(manifest["label_classes"]) < 2:
        manifest["warnings"].append("single-class labels: auc_table will be 'insufficient'; use a harder --tasks set so some runs fail")
    if not a.mock_llm and len({k for k, _ in (v for v in labels.values())}) < a.min_task_keys:
        manifest["warnings"].append("fewer task keys than --min-task-keys")

    jn = join_accounting(paths["db"], progress)
    manifest["join"] = jn
    if jn["joined_outcome"] == 0 or jn["joined_outcome"] < jn["step_bearing_runs"] or jn["step_bearing_runs"] < jn["task_runs"]:
        manifest["verdict"] = "JOIN_FAILED: session-id != task-id for some runs; reports NOT filed (add a fix task)"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(manifest["verdict"], jn, file=sys.stderr)
        return 3

    rid = a.run_id or ("hil15a-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    l2 = subprocess.run([sys.executable, str(ROOT / "scripts/trace/l2_scan.py"), "--step-db", paths["db"], "--progress-logs", str(progress),
                         "--out", str(out / "l2_report.json"), "--belief-sidecar", "--run-id", rid, "--epyc-root", a.epyc_root],
                        capture_output=True, text=True)
    manifest["l2_scan"] = {"rc": l2.returncode, "stderr": l2.stderr[-400:], "sidecar": str(out / "l2_report.json.beliefs.jsonl")}

    from src.trace.dependency_features import auc_table, load_runs
    manifest["auc_table"] = auc_table(load_runs(paths["db"]), {k: (v[0], bool(v[1])) for k, v in labels.items()})
    (out / "auc_table.json").write_text(json.dumps(manifest["auc_table"], indent=1))

    ver = {}
    for r in results:
        p = subprocess.run([sys.executable, str(ROOT / "scripts/trace/replay_flip_run.py"), "verify", "--db", paths["db"],
                            "--run-id", r["run_id"], "--cas-dir", paths["cas_dir"]], capture_output=True, text=True)
        ver[r["run_id"]] = {"rc": p.returncode, "tail": p.stdout[-200:] if p.returncode else "clean"}
    manifest["replay_verify"] = ver
    bad = [k for k, v in ver.items() if v["rc"]]
    manifest["verdict"] = "OK" if not bad and l2.returncode == 0 else f"PROBLEMS: verify={bad} l2_rc={l2.returncode}"
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("verdict", "join", "label_classes")}, indent=1))
    return 0 if manifest["verdict"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
