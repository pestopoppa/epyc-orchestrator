#!/usr/bin/env python3
"""Synthetic rowflow qualification; output is confined to the dry-run root."""
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.crm_pair import pair_manifest, run_pair  # noqa: E402
from harness.endpoint import ChatResult  # noqa: E402


class Services:
    def reset(self):
        self.state = {"salesforce": {"Notes": [], "Tasks": []}, "slack": {"channels": {}}}
        return copy.deepcopy(self.state)
    def invoke(self, name, arguments):
        if name != "create_record":
            raise ValueError("unexpected fixture tool")
        self.state["salesforce"]["Notes"].append({"id": "fixture", "attributes": arguments["attributes"]})
        return {"created": True}
    def snapshot(self):
        return copy.deepcopy(self.state)
    def close(self):
        pass


class Endpoint:
    seed = 42
    def complete(self, messages, seed, **_kwargs):
        if seed != 42:
            raise ValueError("wrong seed")
        if messages[-1]["role"] == "tool":
            return ChatResult(text="done", raw={"fixture": True})
        call = {"id": "fixture", "name": "create_record", "arguments": {
            "module_name": "Notes", "attributes": {"name": "fixture"}}}
        return ChatResult(text="", tool_calls=[call], raw={"fixture": True, "call": call})


def main():
    if os.environ.get("FILLER_DRYRUN") != "1":
        raise RuntimeError("fixture refuses non-dryrun execution")
    root = Path(os.environ["FILLER_DRYRUN_DIR"]).resolve(strict=True)
    output = Path(os.environ["FILLER_DRYRUN_RESULT_PATH"])
    if not output.is_absolute() or not output.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("fixture result escapes dryrun namespace")
    if output.exists() or output.is_symlink():
        raise FileExistsError("fixture output exists")
    if os.environ.get("CRM_FIXTURE_HOLD") == "1":
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(300)"])
        (output.parent / "fixture-child.pid").write_text(str(child.pid))
        def stop_signal(_signum, _frame):
            raise SystemExit(75)
        previous = {s: signal.signal(s, stop_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while not Path(os.environ["FILLER_STOPFILE"]).exists():
                time.sleep(0.05)
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            if child.poll() is None:
                raise RuntimeError("fixture child did not drain")
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        return 75
    manifest = pair_manifest()
    result = run_pair(Endpoint(), Services(), seed=42)
    if (result["arms"]["control"]["delivery"] != "not_observed"
            or result["arms"]["attack"]["delivery"] != "delivered"
            or result["cleanup"] != "completed_by_backend"):
        raise RuntimeError("synthetic pair failed")
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"scope": "fixture-only", "capture_kind": "fixture", "pass": True,
                   "case_sha256": manifest["registry_sha256"]}, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
