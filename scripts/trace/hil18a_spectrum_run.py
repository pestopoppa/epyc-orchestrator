#!/usr/bin/env python3
"""HIL-18a in-process spectrum replay (queue row ``s5-hil18a-spectrum``): no API server, no production stack.

``scripts/trace/spectrum_replay.py run`` replays through a live API (``POST /chat``), which would need the
production stack. This driver replays the same way the HIL-15a driver does: real graph nodes + real REPL in
THIS process, ALL THREE telemetry flags on, every role URL pointed at ONE private loopback llama-server
whose pid owns the port (production ports refused). It reuses ``hil15a_flagon_run`` helpers verbatim
(pinned PYTHONHASHSEED, tmp/spill/CAS dirs, reasoning retry off, port-ownership proof) and the library in
``src.trace.spectrum_replay`` (``traces_for_manifest`` / ``build_report``), so the report is the same
``spectrum_report/v1``.

  select    tasks with outcome variance come from a prior HIL-15a run (``--select-from <manifest.json>``):
            keys whose runs are not all-pass, mixed first, at most ``--max-tasks`` (default 10). Without it
            every task in ``--tasks`` is replayed and the report says which were ineligible.
  replay    task-major, N replays per task (default 20) so a wall cap or STOP truncates whole tasks, never
            every task evenly. RESUMABLE: re-running with the same --out-dir continues from the manifest.
  stop      ``--stop-file`` / ``--deadline-epoch`` / ``--max-wall-s`` are checked between replays.
  report    join check first (replays with step events vs replays started; a total miss aborts), then
            ``build_report`` -> ``spectrum_report.json`` (+ ``--human`` overlap, belief sidecar if the
            adapter exists). Live mode refuses unless WINDOW_CONFIRMED=1; ``--mock-llm`` needs no server.
Exit: 0 report written, 3 join failed, 4 stopped before any task reached --n replays (partial; rerun resumes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import hil15a_flagon_run as h15  # noqa: E402


def select_tasks(tasks: list[dict], manifest_path: str | None, max_tasks: int) -> list[dict]:
    """Tasks with outcome variance first (some pass AND some fail), then all-fail, never all-pass."""
    if not manifest_path:
        return tasks[:max_tasks] if max_tasks else tasks
    runs = json.loads(Path(manifest_path).read_text()).get("runs", [])
    rate: dict[str, list[bool]] = {}
    for r in runs:
        rate.setdefault(r["task_key"], []).append(bool(r["pass"]))
    mixed = [k for k, v in rate.items() if any(v) and not all(v)]
    allfail = [k for k, v in rate.items() if v and not any(v)]
    order = {k: i for i, k in enumerate(mixed + allfail)}
    picked = sorted((t for t in tasks if t["task_key"] in order), key=lambda t: order[t["task_key"]])
    return picked[:max_tasks]


def done_counts(manifest: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                out[e["task_key"]] = out.get(e["task_key"], 0) + 1
    return out


def replay_all(tasks, n, deps_for, state_cls, run_task, manifest: Path, tag: str, should_stop):
    """Task-major, resumable. Appends one jsonl row per finished replay. Returns (entries_this_call, stopped)."""
    import asyncio
    loop = asyncio.new_event_loop()
    have = done_counts(manifest)
    entries, stopped = [], False
    with manifest.open("a") as fh:
        for t in tasks:
            for i in range(have.get(t["task_key"], 0), n):
                if should_stop():
                    stopped = True
                    break
                rid = f"spec-{tag}-{t['task_key']}-{i:02d}"
                t0, err, ans = time.time(), None, None
                try:
                    state = state_cls(task_id=rid, prompt=t["prompt"], context=t["prompt"])
                    res = loop.run_until_complete(run_task(state, deps_for(t, rid), start_role=t.get("start_role", "frontdoor")))
                    ans = getattr(res, "answer", None)
                except Exception as e:   # a crashed replay is counted as unknown, not silently dropped
                    err = f"{type(e).__name__}: {e}"[:300]
                outcome = "unknown" if err else ("pass" if h15.grade(t, ans) else "fail")
                e = {"task_key": t["task_key"], "run_id": rid, "replay": i, "outcome": outcome, "error": err,
                     "answer": None if ans is None else str(ans)[:200], "wall_s": round(time.time() - t0, 3)}
                fh.write(json.dumps(e, sort_keys=True) + "\n")
                fh.flush()
                entries.append(e)
            if stopped:
                break
    loop.close()
    return entries, stopped


def mock_script(task, run_id):
    """Mock replays: wrong on every third replay index so each task has pass AND fail (plumbing only)."""
    wrong = int(run_id.rsplit("-", 1)[1]) % 3 == 0
    ans = "7" if wrong else task.get("mock_answer", "x")
    return ['a = TOOL("write_file", path="/work/n.txt", text="x")\nprint(a)\n', 'print(peek(12))\n',
            'b = TOOL("read_file", path="/work/n.txt")\nprint(b)\n', 'c = TOOL("read_file", path="/work/n.txt")\nprint(c)\n', f'FINAL("{ans}")\n']


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", help="json list like hil15a --tasks (task_key, prompt, expect[, mock_answer])")
    ap.add_argument("--select-from", help="hil15a manifest.json: replay only keys with outcome variance")
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--n", type=int, default=20, help="replays per task (spectrum eligibility needs >= --min-replays)")
    ap.add_argument("--min-replays", type=int, default=20)
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--server-url"); ap.add_argument("--server-pid", type=int)
    ap.add_argument("--hashseed", default="0"); ap.add_argument("--no-reexec", action="store_true")
    ap.add_argument("--tag", default=None, help="run-id tag; fixed per out-dir so reruns resume")
    ap.add_argument("--max-wall-s", type=float, default=None)
    ap.add_argument("--stop-file", default=os.environ.get("FILLER_STOPFILE"))
    ap.add_argument("--deadline-epoch", type=float, default=float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None)
    ap.add_argument("--human"); ap.add_argument("--top-m", type=int, default=3)
    ap.add_argument("--epyc-root", default=os.environ.get("EPYC_ROOT", "/workspace"))
    a = ap.parse_args(argv)

    if not a.mock_llm:
        if os.environ.get("WINDOW_CONFIRMED") != "1":
            sys.exit("refusing: live mode runs inference. Set WINDOW_CONFIRMED=1 inside an operator-confirmed window.")
        if not (a.server_url and a.server_pid):
            sys.exit("live mode needs --server-url (private loopback server) and --server-pid (port-ownership proof)")
    tasks = select_tasks(h15.load_tasks(a.tasks), a.select_from, a.max_tasks)
    if not tasks:
        sys.exit("no tasks selected (no key with outcome variance in --select-from)")
    hs = h15.pin_hashseed(a.hashseed, a.no_reexec)
    if hs.startswith("UNPINNED") and not a.mock_llm:
        sys.exit("live mode requires a pinned PYTHONHASHSEED")
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tagf = out / "TAG"
    tag = a.tag or (tagf.read_text().strip() if tagf.exists() else time.strftime("%m%d%H%M%S", time.gmtime()))
    tagf.write_text(tag)

    manifest = {"schema": "hil18a_spectrum_run/v1", "mock": a.mock_llm, "hashseed": hs, "flags": list(h15.FLAGS),
                "tag": tag, "n": a.n, "tasks": [t["task_key"] for t in tasks]}
    if not a.mock_llm:
        import dataclasses
        import re
        from src.config.models import ServerURLsConfig
        prod = {int(m.group(1)) for v in dataclasses.asdict(ServerURLsConfig()).values() if isinstance(v, str)
                for m in [re.search(r":(\d+)", v)] if m}
        manifest["server_port"] = h15.prove_port_ownership(a.server_url, a.server_pid, prod)
        manifest["server_pid"] = a.server_pid
    paths = h15.setup_env(out, None if a.mock_llm else a.server_url)
    from src.graph.graph import run_task
    from src.graph.state import TaskState
    deps_for = h15.mock_deps(paths["spill_dir"], mock_script) if a.mock_llm else h15.live_deps(paths["spill_dir"])

    t_start = time.time()

    def should_stop() -> bool:
        return bool((a.stop_file and os.path.exists(a.stop_file)) or (a.deadline_epoch and time.time() >= a.deadline_epoch)
                    or (a.max_wall_s and time.time() - t_start > a.max_wall_s))

    mf = out / "replays.jsonl"
    new, stopped = replay_all(tasks, a.n, deps_for, TaskState, run_task, mf, tag, should_stop)
    from src.trace.spectrum_replay import build_report, dumps, load_manifest, traces_for_manifest
    entries = load_manifest(mf)
    per_task, acct = traces_for_manifest(paths["db"], entries)
    manifest.update(join=acct, replays_this_call=len(new), stopped=stopped, started_epoch=t_start)
    complete = [k for k, c in done_counts(mf).items() if c >= a.n]
    manifest["tasks_complete"] = complete
    if entries and acct["with_steps"] == 0:
        manifest["verdict"] = "JOIN_FAILED: no replay has step events (flag off or session id != run id); report NOT filed"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(manifest["verdict"], file=sys.stderr)
        return 3
    human = json.loads(Path(a.human).read_text()) if a.human else None
    report = build_report(per_task, acct, n_tasks=a.max_tasks, min_replays=a.min_replays, top_m=a.top_m, human=human,
                          label="mock" if a.mock_llm else "measurement")
    rp = out / "spectrum_report.json"
    rp.write_text(dumps(report))
    manifest["report"] = {"path": str(rp), "tasks_reported": report["tasks_reported"], "tasks_skipped": len(report["tasks_skipped"])}
    cap = Path(a.epyc_root) / "scripts" / "vidya" / "adapters" / "hil_replay_capture.py"
    if cap.is_file():
        sys.path.insert(0, str(Path(a.epyc_root) / "scripts" / "vidya"))
        from adapters.hil_replay_capture import write_belief_measurements  # type: ignore
        manifest["sidecar"] = str(write_belief_measurements(str(rp), kind="spectrum", run_id=f"hil18a-{tag}",
                                                           producer="epyc-orchestrator/scripts/trace/hil18a_spectrum_run.py", code_ref="unknown"))
    else:
        manifest["sidecar_skipped"] = f"adapter not found: {cap}"
    manifest["verdict"] = "OK" if complete else "PARTIAL: no task reached --n replays (rerun resumes)"
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("verdict", "join", "tasks_complete")}, indent=1))
    return 0 if complete else 4


if __name__ == "__main__":
    sys.exit(main())
