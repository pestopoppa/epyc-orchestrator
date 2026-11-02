#!/usr/bin/env python3
"""Model-free full-entry fixture, run inside a real CPU provider claim."""
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
from harness.crm_services import NativeServices, load_pinned_module  # noqa: E402
from harness.crm_source_manifest import verify_manifest  # noqa: E402


def main():
    if os.environ.get("FILLER_DRYRUN") == "1":
        raise RuntimeError("real-service fixture worker refuses dry-run")
    cfg = json.loads(Path(os.environ["CRM_QUAL_CONFIG"]).read_bytes())
    source = verify_manifest(cfg["source_manifest"], cfg["source_commit"], cfg["source_tree_sha256"])
    cpus = os.environ["FILLER_CPUS"]
    cpu_set = set()
    for part in cpus.split(","):
        lo, _, hi = part.partition("-")
        cpu_set.update(range(int(lo), int(hi or lo) + 1))
    os.sched_setaffinity(0, cpu_set)
    if os.sched_getaffinity(0) != cpu_set:
        raise ValueError("worker affinity differs from claim")
    deadline, stop = int(os.environ["FILLER_DEADLINE"]), Path(os.environ["FILLER_STOPFILE"])
    def should_stop():
        return stop.exists() or time.time() >= deadline
    if should_stop():
        raise InterruptedError("STOP or deadline")
    prefix = Path(cfg["claim_prefix"])
    opened = json.loads(Path(str(prefix) + ".open.json").read_bytes())
    grant = Path(str(prefix) + ".grant")
    if (opened.get("cpu_list") != cpus or opened.get("grant_id") != os.environ["FILLER_ID"]
            or hashlib.sha256(grant.read_bytes()).hexdigest() != opened.get("grant_sha256")):
        raise ValueError("provider claim/grant does not match fixture")
    fake = Path(__file__).with_name("crm_qual_server.py")
    private = PrivateServices(cpus=cpus, seed_path=Path(cfg["seed_file"]))
    proc = None
    previous = {}
    def interrupt(_sig, _frame):
        raise InterruptedError("termination requested")
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, interrupt)
    try:
        private.start()
        os.environ.update(SALESFORCE_BASE_URL=private.crm_url + "/", SALESFORCE_GRANT_TYPE="client_credentials",
                          SALESFORCE_CLIENT_ID="17817554-086b-83d1-bf9d-69026221f529",
                          SALESFORCE_CLIENT_SECRET="mcp-secret-123", SLACK_API_URL=private.slack_url,
                          USER_ACCESS_TOKEN="slack_token_sales_ops")
        crm = load_pinned_module(Path(cfg["crm_source"]), "crm")
        slack = load_pinned_module(Path(cfg["slack_source"]), "slack")
        port = int(cfg["port"])
        proc = subprocess.Popen([sys.executable, str(fake), str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        birth = int(Path(f"/proc/{proc.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        limit = time.monotonic() + 30
        while True:
            if proc.poll() is not None or should_stop() or time.monotonic() >= limit:
                raise RuntimeError("fixture server failed to start")
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        listening = subprocess.run(["ss", "-H", "-ltnp", f"( sport = :{port} )"],
                                   check=True, capture_output=True, text=True, timeout=5).stdout
        if f"pid={proc.pid}," not in listening or os.sched_getaffinity(proc.pid) != cpu_set:
            raise RuntimeError("fixture endpoint ownership/affinity mismatch")
        m = pair_manifest()
        prov = {"source_commit": source["source_commit"], "source_tree_sha256": source["source_tree_sha256"],
                "model_sha256": "0" * 64, "server_sha256": hashlib.sha256(fake.read_bytes()).hexdigest(),
                "template_sha256": "0" * 64, "tool_schema_sha256": m["schema_sha256"],
                "case_sha256": m["registry_sha256"], "grant_id": opened["grant_id"],
                "model_pid": proc.pid, "model_pid_starttime": birth, "backend": "unknown"}
        services = NativeServices(crm, slack, reset=private.reset, teardown=private.close)
        endpoint = SeededEndpoint(f"http://127.0.0.1:{port}/v1", seed=42, model="fixture", retries=0, timeout=15)
        result, seal = capture_pair(endpoint, services, seed=42, journal_path=cfg["journal"],
            seal_path=cfg["fixture_seal"], capture_kind="fixture", provenance=prov, grant_receipt_path=grant,
            source_manifest_path=cfg["source_manifest"], should_stop=should_stop)
        if (seal["capture_kind"] != "fixture" or result["arms"]["control"]["status"] != "complete"
                or result["arms"]["attack"]["status"] != "complete"
                or result["arms"]["control"]["delivery"] != "not_observed"
                or result["arms"]["attack"]["delivery"] != "delivered"):
            raise RuntimeError("full-entry fixture did not complete")
        return 0
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            private.close()
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=15)
            if proc is not None and proc.poll() is None:
                raise RuntimeError("fixture server did not drain")
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
