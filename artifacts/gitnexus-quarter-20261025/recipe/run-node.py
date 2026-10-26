#!/usr/bin/python3
"""GRANT ONLY test-side invocation of exact production lifetime protection."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def main():
    scripts = Path(sys.argv[1])
    kind = sys.argv[2]
    args = sys.argv[3:]
    spec = importlib.util.spec_from_file_location("fixture_writer", scripts / "gitnexus-writer.py")
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    guard = writer.cpu_guard()
    current = guard.claim()
    if current["region"] is None or frozenset(os.sched_getaffinity(0)) != current["cpus"]:
        raise RuntimeError("fixture needs one exact canonical quarter")
    if kind == "native-db" and len(args) == 1:
        command = [guard.NODE, "--v8-pool-size=1", str(HERE / "native-db.cjs"), str(scripts), args[0]]
    elif kind == "query" and args and args[0] in {"status", "impact", "context"}:
        command = [guard.NODE, "--v8-pool-size=1", str(HERE / "query.cjs"), str(scripts), *args]
    else:
        raise RuntimeError("unknown granted fixture")
    original_popen = writer.subprocess.Popen
    trace = None
    if kind == "native-db":
        directory = Path(args[0])
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / "supervisor-placement.jsonl"

    def launch(_production_command, **kwargs):
        # Import-only test substitution: all production signal/PDEATHSIG logic
        # and canonical claim checks still run. No production env/CLI override.
        return original_popen(command, **kwargs)

    def placement(_current, pid):
        if not os.path.samefile(f"/proc/{pid}/exe", guard.NODE):
            raise RuntimeError("fixture child is not actual Node")
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
        masks = {tid.name: sorted(os.sched_getaffinity(int(tid.name))) for tid in tasks}
        if trace is not None:
            with trace.open("a") as stream:
                stream.write(json.dumps({"node_pid": pid, "task_count": len(tasks),
                    "task_masks": masks, "region": current["region"], "grant_sha256": current["grant"]}) + "\n")
        if len(tasks) > len(current["cpus"]) or any(
                frozenset(mask) != current["cpus"] for mask in masks.values()):
            raise RuntimeError("actual fixture Node mask/count escaped quarter")

    admission = SimpleNamespace(NODE=guard.NODE, GNX=guard.GNX, installed_source=guard.installed_source,
                                native_artifact=guard.native_artifact,
                                DB_BUFFER_BYTES=guard.DB_BUFFER_BYTES, DB_MAX_BYTES=guard.DB_MAX_BYTES,
                                claim=guard.claim, node_admission=placement)
    # The real production writer lock is used only by the real bootstrap proof;
    # test queries/native fixtures own a separate scratch lock with identical
    # FD inheritance/lifetime semantics.
    with (HERE.parent / "runtime/fixture.lock").open("a+b") as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.object(writer.subprocess, "Popen", side_effect=launch):
            return writer.quarter_run("app", [], lock.fileno(), admission, current)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"fixture Node: {exc}", file=sys.stderr)
        raise SystemExit(1)
