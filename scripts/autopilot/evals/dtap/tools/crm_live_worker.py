#!/usr/bin/env python3
"""Caller-owned CRM pair worker; requires a real provider claim."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.crm_pair import SeededEndpoint, pair_manifest  # noqa: E402
from harness.crm_pair_capture import capture_pair  # noqa: E402
from harness.crm_provision import PrivateServices  # noqa: E402
from harness.crm_receipt import Receipt  # noqa: E402
from harness.crm_services import NativeServices, load_pinned_module  # noqa: E402
from harness.crm_source_manifest import verify_manifest  # noqa: E402


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _birth(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])


def _cpus(value):
    result = set()
    for part in value.split(","):
        lo, _, hi = part.partition("-")
        result.update(range(int(lo), int(hi or lo) + 1))
    return result


def _unchanged_prepared_file(fact):
    st = Path(fact["path"]).stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) == (
        fact["device"], fact["inode"], fact["size"], fact["mtime_ns"], fact["ctime_ns"])


def _wait_server(proc, port, deadline, should_stop):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    limit = min(time.monotonic() + 120, time.monotonic() + max(0, deadline - time.time()))
    while True:
        if proc.poll() is not None or should_stop() or time.monotonic() >= limit:
            raise RuntimeError("owned server did not become ready")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.25)
    sockets = subprocess.run(["ss", "-H", "-ltnp", f"( sport = :{port} )"],
                             check=True, capture_output=True, text=True, timeout=5).stdout
    if f"pid={proc.pid}," not in sockets:
        raise RuntimeError("endpoint socket does not belong to owned server")


class DeadlineEndpoint(SeededEndpoint):
    def __init__(self, *args, deadline, should_stop, **kwargs):
        self.deadline = deadline
        self.should_stop = should_stop
        super().__init__(*args, **kwargs)

    def complete(self, *args, **kwargs):
        remaining = self.deadline - time.time() - 5
        if self.should_stop() or remaining <= 0:
            raise InterruptedError("STOP or deadline before request")
        self.timeout = min(150, remaining)
        return super().complete(*args, **kwargs)


def main():
    if os.environ.get("FILLER_DRYRUN") == "1":
        raise RuntimeError("live worker refuses dryrun")
    cfg = json.loads(Path(os.environ["CRM_JOB_CONFIG"]).read_bytes())
    prep = json.loads(Path(cfg["input_manifest"]).read_bytes())
    if prep.get("schema") != "dtap.crm_input_preparation.v1":
        raise ValueError("invalid preflight input manifest")
    source = verify_manifest(cfg["source_manifest"], prep["source_commit"], prep["source_tree_sha256"])
    manifest = pair_manifest()
    cpu_text = os.environ["FILLER_CPUS"]
    cpus = _cpus(cpu_text)
    if not cpus:
        raise ValueError("empty CPU claim")
    os.sched_setaffinity(0, cpus)
    if os.sched_getaffinity(0) != cpus:
        raise RuntimeError("worker affinity differs from claim")
    deadline = int(os.environ["FILLER_DEADLINE"])
    stop = Path(os.environ["FILLER_STOPFILE"])
    def should_stop():
        return stop.exists() or time.time() >= deadline
    if should_stop():
        raise InterruptedError("STOP/deadline before launch")
    model, server, template = (prep[key] for key in ("model", "server", "template"))
    if any(not _unchanged_prepared_file(fact) for fact in (model, server, template)):
        raise ValueError("preflight input identity changed")
    if _hash(server["path"]) != server["sha256"] or _hash(template["path"]) != template["sha256"]:
        raise ValueError("server/template bytes changed")
    prefix = Path(cfg["claim_prefix"])
    grant = Path(str(prefix) + ".grant")
    opened = json.loads(Path(str(prefix) + ".open.json").read_bytes())
    if opened.get("cpu_list") != cpu_text or opened.get("grant_id") != os.environ["FILLER_ID"]:
        raise ValueError("provider claim scope differs from job")
    if _hash(grant) != opened["grant_sha256"]:
        raise ValueError("provider grant copy changed")
    env = dict(os.environ, LD_LIBRARY_PATH=str(Path(server["path"]).parent), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([cfg["verify_linkage"], server["path"], str(Path(server["path"]).parent)], env=env,
                   check=True, timeout=20, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lifecycle = Receipt(Path(cfg["lifecycle_journal"]), {"scope": "live-worker-lifecycle",
                        "source_commit": source["source_commit"], "grant_id": opened["grant_id"]})
    private = None
    proc = None
    log = None
    previous = {}
    def interrupt(_signum, _frame):
        raise InterruptedError("termination requested")
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, interrupt)
    try:
        private = PrivateServices(cpus=cpu_text, seed_path=Path(cfg["seed_file"]), receipt=lifecycle)
        private.start()
        os.environ.update(SALESFORCE_BASE_URL=private.crm_url + "/", SALESFORCE_GRANT_TYPE="client_credentials",
            SALESFORCE_CLIENT_ID="17817554-086b-83d1-bf9d-69026221f529", SALESFORCE_CLIENT_SECRET="mcp-secret-123",
            SLACK_API_URL=private.slack_url, USER_ACCESS_TOKEN="slack_token_sales_ops")
        crm = load_pinned_module(Path(cfg["crm_source"]), "crm")
        slack = load_pinned_module(Path(cfg["slack_source"]), "slack")
        log = open(cfg["server_log"], "x", encoding="utf-8")
        port = int(cfg["port"])
        proc = subprocess.Popen([server["path"], "-m", model["path"], "--host", "127.0.0.1", "--port", str(port),
            "-t", str(len(cpus)), "-ngl", "0", "-c", "16384", "--jinja", "--chat-template-file", template["path"],
            "--reasoning", "off"], env=env, stdout=log, stderr=subprocess.STDOUT)
        birth = _birth(proc.pid)
        _wait_server(proc, port, deadline, should_stop)
        if _birth(proc.pid) != birth or os.sched_getaffinity(proc.pid) != cpus:
            raise RuntimeError("server birth or affinity changed")
        if _hash(Path(f"/proc/{proc.pid}/exe")) != server["sha256"]:
            raise RuntimeError("running server differs from prepared binary")
        mappings = Path(f"/proc/{proc.pid}/maps").read_text().splitlines()
        mapped = None
        for line in mappings:
            fields = line.split(maxsplit=5)
            if len(fields) < 5:
                continue
            major, minor = (int(x, 16) for x in fields[3].split(":"))
            if (major, minor, int(fields[4])) == (os.major(model["device"]), os.minor(model["device"]), model["inode"]):
                mapped = line
                break
        if not mapped:
            raise RuntimeError("server does not map prepared model inode")
        observation = {"server_pid": proc.pid, "server_starttime": birth, "affinity": sorted(cpus),
                       "model_device": model["device"], "model_inode": model["inode"],
                       "model_map_row": mapped, "server_exe_sha256": server["sha256"],
                       "limitation": "preflight hash plus mapped inode does not exclude in-place mutation"}
        observation_path = Path(cfg["runtime_observation"])
        observation_data = (json.dumps(observation, sort_keys=True) + "\n").encode()
        fd = os.open(observation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(observation_data)
            stream.flush()
            os.fsync(stream.fileno())
        lifecycle.append("runtime_observation", observation)
        provenance = {"source_commit": source["source_commit"], "source_tree_sha256": source["source_tree_sha256"],
            "model_sha256": model["sha256"], "server_sha256": server["sha256"], "template_sha256": template["sha256"],
            "tool_schema_sha256": manifest["schema_sha256"], "case_sha256": manifest["registry_sha256"],
            "grant_id": opened["grant_id"], "model_pid": proc.pid, "model_pid_starttime": birth,
            "backend": "cpu", "model_file": Path(model["path"]).name,
            "lifecycle_journal_path": str(Path(cfg["lifecycle_journal"]).resolve()),
            "runtime_observation_path": str(observation_path.resolve()),
            "runtime_observation_sha256": hashlib.sha256(observation_data).hexdigest()}
        services = NativeServices(crm, slack, reset=private.reset, teardown=private.close)
        client = DeadlineEndpoint(f"http://127.0.0.1:{port}/v1", seed=42, model=model["path"],
                                  temperature=0.0, max_tokens=1024, retries=0, timeout=150,
                                  deadline=deadline, should_stop=should_stop)
        result, _ = capture_pair(client, services, seed=42, journal_path=cfg["journal"], seal_path=cfg["seal"],
            capture_kind="live", provenance=provenance, grant_receipt_path=grant,
            source_manifest_path=cfg["source_manifest"], should_stop=should_stop)
        if len(result["arms"]) != 2 or any(a["status"] != "complete" for a in result["arms"].values()):
            raise RuntimeError("pair incomplete; native partial journal retained")
        return 0
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        cleanup_error = None
        try:
            if private is not None:
                private.close()
        except Exception as exc:
            cleanup_error = f"service cleanup: {exc}"
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=15)
                except Exception as exc:
                    cleanup_error = f"server cleanup: {exc}"
            if proc is not None and proc.poll() is None:
                cleanup_error = "captured server did not drain"
            if log is not None:
                log.close()
            try:
                lifecycle.append("worker_cleanup", {"ok": cleanup_error is None, "error": cleanup_error,
                                                     "server_pid": proc.pid if proc else None})
            finally:
                lifecycle.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if cleanup_error:
            raise RuntimeError(cleanup_error)


if __name__ == "__main__":
    sys.exit(main())
