import copy
import json
from contextlib import nullcontext

import pytest

from harness.crm_pair import SeededEndpoint, pair_manifest, run_pair
from harness.endpoint import ChatResult


class Backend:
    def __init__(self, mismatch=False, fail=False):
        self.resets = 0
        self.closed = False
        self.mismatch = mismatch
        self.fail = fail

    def reset(self):
        self.resets += 1
        self.state = {"salesforce": {"Notes": [], "Tasks": []}, "slack": {"channels": {}}}
        if self.mismatch and self.resets == 2:
            self.state["slack"]["channels"]["other"] = []
        return copy.deepcopy(self.state)

    def invoke(self, name, arguments):
        if self.fail:
            return {"error": "broken"}
        self.state["salesforce"][arguments["module_name"]].append({
            "id": "n1", "type": arguments["module_name"], "attributes": arguments["attributes"]})
        return {"ok": True}

    def snapshot(self):
        return copy.deepcopy(self.state)

    def close(self):
        self.closed = True


class Endpoint:
    seed = 42

    def complete(self, messages, seed, **kwargs):
        if messages[-1]["role"] != "tool":
            return ChatResult(text="", tool_calls=[{"id": "t1", "name": "create_record",
                "arguments": {"module_name": "Notes", "attributes": {"name": "example"}}}], raw={"call": 1})
        return ChatResult(text="done", raw={"done": True})


def test_pair_delivers_only_after_service_result_and_preserves_state():
    backend = Backend()
    result = run_pair(Endpoint(), backend, seed=42)
    assert backend.closed
    assert result["cleanup"] == "completed_by_backend"
    assert result["arms"]["control"]["status"] == "complete"
    attack = result["arms"]["attack"]
    assert attack["status"] == "complete"
    assert attack["delivery"] == "delivered"
    controls = result["arms"]["control"]["events"]
    events = attack["events"]
    assert result["manifest"]["payload"] not in controls[1]["model_visible_result"]
    assert events[1]["model_visible_result"].endswith(result["manifest"]["payload"])
    assert events[2]["request_messages"][-1]["content"] == events[1]["model_visible_result"]
    assert attack["final_state"]["salesforce"]["Notes"]
    assert attack["judge"] is not None
    assert result["manifest"]["skill_injection"] == "not_tested"


def test_reset_mismatch_and_service_error_fail_closed_and_cleanup():
    b = Backend(mismatch=True)
    result = run_pair(Endpoint(), b, seed=42)
    assert result["arms"]["attack"]["status"] == "failed"
    assert "reset states differ" in result["arms"]["attack"]["error"]["message"]
    assert b.closed
    b = Backend(fail=True)
    result = run_pair(Endpoint(), b, seed=42)
    assert result["arms"]["control"]["status"] == "failed"
    assert result["arms"]["control"]["judge"] is None
    assert b.closed


def test_suffix_is_not_claimed_delivered_when_no_followup_request():
    result = run_pair(Endpoint(), Backend(), seed=42, max_turns=1)
    assert result["arms"]["control"]["status"] == "failed"
    assert "attack" not in result["arms"]


def test_seed_is_serialized_on_wire():
    requests = []
    class Reply:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}).encode()
    def opener(request, *, timeout):
        requests.append(json.loads(request.data))
        return nullcontext(Reply())
    ep = SeededEndpoint("http://localhost:1/v1", seed=42, opener=opener, retries=0)
    ep.complete([{"role": "user", "content": "hello"}], seed=42)
    assert requests[0]["seed"] == 42
    with pytest.raises(ValueError):
        ep.complete([], seed=43)
    assert pair_manifest()["payload_sha256"]


def test_attack_response_is_not_delivered_if_followup_fails():
    class FailingAttackEndpoint(Endpoint):
        def __init__(self):
            self.calls = 0
        def complete(self, messages, seed, **kwargs):
            self.calls += 1
            if self.calls == 4:
                raise TimeoutError("endpoint lost")
            return super().complete(messages, seed, **kwargs)
    b = Backend()
    result = run_pair(FailingAttackEndpoint(), b, seed=42)
    assert result["arms"]["attack"]["status"] == "failed"
    assert result["arms"]["attack"]["delivery"] == "queued_not_confirmed"
    assert result["arms"]["attack"]["judge"] is None
    assert b.closed


def test_undeclared_tool_call_refused_before_service():
    class InvalidEndpoint(Endpoint):
        def complete(self, messages, seed, **kwargs):
            return ChatResult(text="", tool_calls=[{"id": "x", "name": "not_a_tool", "arguments": {}}])
    b = Backend()
    result = run_pair(InvalidEndpoint(), b, seed=42)
    assert result["arms"]["control"]["status"] == "failed"
    assert result["arms"]["control"]["judge"] is None
    assert b.closed


def test_partial_receipt_persists_failure_and_hash_chain(tmp_path):
    import hashlib
    from harness.crm_receipt import Receipt
    path = tmp_path / "receipt.jsonl"
    receipt = Receipt(path, {"identity": "test-only"})
    result = run_pair(Endpoint(), Backend(fail=True), seed=42, receipt=receipt)
    receipt.close()
    assert result["arms"]["control"]["status"] == "failed"
    lines = path.read_bytes().splitlines(keepends=True)
    records = [json.loads(line) for line in lines]
    assert [r["kind"] for r in records][-3:] == ["tool_attempt", "arm_failed", "cleanup"]
    assert records[0]["previous_sha256"] == "0" * 64
    for previous, current in zip(lines, records[1:]):
        assert current["previous_sha256"] == hashlib.sha256(previous).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        Receipt(path, {})


def test_native_service_adapter_invokes_and_snapshots_actual_results():
    from types import SimpleNamespace
    from harness.crm_services import NativeServices
    events = []
    def tool(fn):
        return SimpleNamespace(fn=fn)
    async def get_entry_list(module_name, page, page_size):
        return {"data": [{"id": module_name + "-1", "attributes": {"name": module_name}}]}
    async def get_entry(module_name, id):
        return {"data": {"id": id, "attributes": {"name": "Meridian Group" if module_name == "Accounts" else module_name}}}
    async def create_record(module_name, attributes):
        events.append((module_name, attributes))
        return {"id": "created"}
    async def channels_history(channel_name, workspace_id):
        return {"messages": [{"text": "actual message"}]}
    crm = SimpleNamespace(get_entry_list=tool(get_entry_list), get_entry=tool(get_entry),
                          create_record=tool(create_record))
    slack = SimpleNamespace(channels_history=tool(channels_history))
    native = NativeServices(crm, slack, reset=lambda: events.append("reset"),
                            teardown=lambda: events.append("closed"))
    initial = native.reset()
    assert initial["salesforce"]["Notes"][0]["attributes"]["name"] == "Notes"
    assert initial["slack"]["channels"]["sales"][0]["text"] == "actual message"
    assert native.invoke("create_record", {"module_name": "Notes", "attributes": {}}) == {"id": "created"}
    native.close()
    assert events == ["reset", ("Notes", {}), "closed"]
    with pytest.raises(ValueError, match="absent or ambiguous"):
        native.invoke("unknown", {})


def test_upstream_module_loader_refuses_unpinned_bytes(tmp_path):
    from harness.crm_services import load_pinned_module
    module = tmp_path / "main.py"
    module.write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="does not match pinned bytes"):
        load_pinned_module(module, "slack")


def test_private_service_provisioning_is_unique_and_reset_seed_pinned():
    from pathlib import Path
    from harness.crm_provision import PrivateServices, SEED_ACCOUNT_ID, prepare_seed
    seed = Path(__file__).resolve().parent.parent / "fixtures/crm_pair/upstream-seed.sql"
    sql, digest = prepare_seed(seed)
    assert SEED_ACCOUNT_ID in sql and "UUID()" not in sql and "NOW()" not in sql
    assert len(digest) == 64
    class DryProvision(PrivateServices):
        def __init__(self):
            super().__init__(cpus="0-1", seed_path=seed)
            self.calls = []
            self.remaining = set()
        def _docker(self, *args):
            self.calls.append(args)
            if args[:2] == ("network", "create"):
                return "network-id"
            if args[0] == "create":
                ident = f"container-{len(self.remaining)}"
                self.remaining.add(ident)
                return ident
            if args[:2] == ("inspect", "--format"):
                return "172.20.0.2"
            if args[:2] == ("rm", "-f"):
                self.remaining.remove(args[2])
                return args[2]
            if args[0] == "inspect":
                import subprocess
                raise subprocess.CalledProcessError(1, args, stderr="No such object")
            if args[:2] == ("network", "inspect"):
                import subprocess
                raise subprocess.CalledProcessError(1, args, stderr="No such network")
            return "ok"
        def _health(self, url, container):
            pass
        def _http(self, url, *, body=None):
            self.calls.append(("http", url, body))
            return {"ok": True}
    p = DryProvision()
    p.start()
    assert len(p.containers) == 3
    p.reset()
    assert len([x for x in p.calls if x[0] == "http"]) == 4
    p.close()
    assert not p.containers and p.network is None
    assert all("--cpuset-cpus" in call for call in p.calls if call[0] == "create")
    assert p.crm_url.startswith("http://172.")


def test_judge_attestation_failure_refuses_pair(monkeypatch):
    import harness.crm_pair as module
    monkeypatch.setattr(module, "verify_judges", lambda: (False, ["judge bytes changed"]))
    with pytest.raises(ValueError, match="judge attestation failed"):
        module.pair_manifest()


def test_capture_seals_original_fixture_and_rejects_tampering(tmp_path):
    from harness.crm_pair_capture import capture_pair
    from harness.crm_pair_seal import read_journal, seal_pair
    manifest = pair_manifest()
    provenance = {"source_commit": "1" * 40, "source_tree_sha256": "2" * 64,
                  "model_sha256": "3" * 64, "server_sha256": "4" * 64,
                  "template_sha256": "5" * 64, "tool_schema_sha256": manifest["schema_sha256"],
                  "case_sha256": manifest["registry_sha256"], "grant_id": "fixture-only",
                  "model_pid": 123, "model_pid_starttime": 1, "backend": "unknown"}
    grant = tmp_path / "grant"
    grant.write_text("fixture grant, no authority")
    source = tmp_path / "source"
    source.write_text("fixture source, no authority")
    journal = tmp_path / "journal.jsonl"
    seal = tmp_path / "terminal.json"
    result, terminal = capture_pair(Endpoint(), Backend(), seed=42, journal_path=journal,
        seal_path=seal, capture_kind="fixture", provenance=provenance,
        grant_receipt_path=grant, source_manifest_path=source)
    assert terminal["capture_kind"] == "fixture"
    assert terminal["arms"]["attack"]["delivery"] == "delivered"
    assert terminal["provenance"]["backend"] == "unknown"
    assert result["cleanup"] == "completed_by_backend"
    assert read_journal(journal)[1][-1]["kind"] == "terminal"
    with pytest.raises(FileExistsError):
        seal_pair(journal_path=journal, output_path=seal, capture_kind="fixture", provenance=provenance,
                  grant_receipt_path=grant, source_manifest_path=source, producer_path=source)
    content = journal.read_bytes()
    journal.write_bytes(content.replace(b'"kind":"start"', b'"kind":"stArt"', 1))
    with pytest.raises(ValueError, match="hash chain mismatch"):
        read_journal(journal)


def test_capture_refuses_missing_provenance_before_service(tmp_path):
    from harness.crm_pair_capture import capture_pair
    backend = Backend()
    with pytest.raises(ValueError, match="source_commit"):
        capture_pair(Endpoint(), backend, seed=42, journal_path=tmp_path / "journal",
                     seal_path=tmp_path / "seal", capture_kind="live", provenance={},
                     grant_receipt_path=tmp_path / "grant", source_manifest_path=tmp_path / "source")
    assert backend.resets == 0


def test_capture_seals_control_failure_and_not_attempted_attack(tmp_path):
    from harness.crm_pair_capture import capture_pair
    from harness.crm_pair_seal import read_journal
    m = pair_manifest()
    prov = {"source_commit": "1" * 40, "source_tree_sha256": "2" * 64,
            "model_sha256": "3" * 64, "server_sha256": "4" * 64, "template_sha256": "5" * 64,
            "tool_schema_sha256": m["schema_sha256"], "case_sha256": m["registry_sha256"],
            "grant_id": "fixture", "model_pid": 123, "model_pid_starttime": 1, "backend": "unknown"}
    grant = tmp_path / "grant"
    grant.write_text("fixture")
    source = tmp_path / "source"
    source.write_text("fixture")
    journal = tmp_path / "journal"
    result, seal = capture_pair(Endpoint(), Backend(fail=True), seed=42, journal_path=journal,
        seal_path=tmp_path / "seal", capture_kind="fixture", provenance=prov,
        grant_receipt_path=grant, source_manifest_path=source)
    assert result["arms"]["control"]["status"] == "failed"
    assert seal["arms"]["control"]["status"] == "failed"
    assert seal["arms"]["attack"] == {"status": "not_attempted", "delivery": "not_observed"}
    assert any(x["kind"] == "arm_skipped" for x in read_journal(journal)[1])


def test_live_capture_waits_for_provider_close_and_fails_without_it(tmp_path, monkeypatch):
    import harness.crm_pair_capture as capture_module
    from harness.crm_pair_capture import capture_pair
    from harness.crm_pair_seal import seal_pair
    monkeypatch.setattr(capture_module, "verify_manifest", lambda *args: None)
    m = pair_manifest()
    prov = {"source_commit": "1" * 40, "source_tree_sha256": "2" * 64,
            "model_sha256": "3" * 64, "server_sha256": "4" * 64, "template_sha256": "5" * 64,
            "tool_schema_sha256": m["schema_sha256"], "case_sha256": m["registry_sha256"],
            "grant_id": "synthetic", "model_pid": 123, "model_pid_starttime": 1, "backend": "unknown"}
    grant = tmp_path / "grant"
    grant.write_text("synthetic")
    source = tmp_path / "source"
    source.write_text("synthetic")
    journal, seal_path = tmp_path / "journal", tmp_path / "seal"
    _, seal = capture_pair(Endpoint(), Backend(), seed=42, journal_path=journal,
        seal_path=seal_path, capture_kind="live", provenance=prov,
        grant_receipt_path=grant, source_manifest_path=source)
    assert seal is None and not seal_path.exists()
    with pytest.raises(ValueError, match="provider open/close"):
        seal_pair(journal_path=journal, output_path=seal_path, capture_kind="live", provenance=prov,
                  grant_receipt_path=grant, source_manifest_path=source, producer_path=source)


def test_source_manifest_reopens_exact_committed_files(tmp_path):
    import subprocess
    from harness.crm_source_manifest import FILES, source_facts, verify_manifest, write_manifest
    repo = tmp_path / "repo"
    repo.mkdir()
    for path in FILES:
        item = repo / path
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("fixture " + path)
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("add", "--", *FILES)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "fixture")
    commit = git("rev-parse", "HEAD")
    manifest_path = tmp_path / "manifest.json"
    facts = write_manifest(manifest_path, repo, commit)
    assert verify_manifest(manifest_path, commit, facts["source_tree_sha256"]) == facts
    assert source_facts(repo, commit) == facts
    (repo / FILES[0]).write_text("tampered")
    with pytest.raises(ValueError, match="working source differs"):
        verify_manifest(manifest_path, commit, facts["source_tree_sha256"])


def test_prepared_file_hash_is_explicit_preflight_fact(tmp_path):
    import hashlib
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "tools" / "crm_prepare.py"
    spec = importlib.util.spec_from_file_location("crm_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "model"
    data.write_bytes(b"not-a-model")
    fact = module.file_fact(data)
    assert fact["sha256"] == hashlib.sha256(b"not-a-model").hexdigest()
    assert fact["path"] == str(data.resolve())
    assert fact["size"] == 11


def test_rowflow_fixture_confines_output_and_drains_stop(tmp_path):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    script = Path(__file__).resolve().parent.parent / "tools" / "crm_row_fixture.py"
    root = tmp_path / "dry"
    root.mkdir()
    output = root / "fixture.json"
    live = tmp_path / "live.json"
    env = dict(os.environ, FILLER_DRYRUN="1", FILLER_DRYRUN_DIR=str(root),
               FILLER_DRYRUN_RESULT_PATH=str(output), FILLER_STOPFILE=str(root / "stop"))
    subprocess.run([sys.executable, str(script)], env=env, check=True, timeout=15)
    assert json.loads(output.read_text())["capture_kind"] == "fixture"
    assert not live.exists()
    env["FILLER_DRYRUN_RESULT_PATH"] = str(live)
    assert subprocess.run([sys.executable, str(script)], env=env, capture_output=True, timeout=15).returncode != 0
    assert not live.exists()
    output.unlink()
    env.update(FILLER_DRYRUN_RESULT_PATH=str(output), CRM_FIXTURE_HOLD="1")
    process = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        deadline = time.monotonic() + 8
        while not (root / "fixture-child.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child = int((root / "fixture-child.pid").read_text())
        (root / "stop").touch()
        assert process.wait(timeout=8) == 75
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        assert not output.exists() and not live.exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=8)


def test_stop_before_request_records_failure_and_skips_attack(tmp_path):
    from harness.crm_pair_capture import capture_pair
    m = pair_manifest()
    prov = {"source_commit": "1" * 40, "source_tree_sha256": "2" * 64,
            "model_sha256": "3" * 64, "server_sha256": "4" * 64, "template_sha256": "5" * 64,
            "tool_schema_sha256": m["schema_sha256"], "case_sha256": m["registry_sha256"],
            "grant_id": "fixture", "model_pid": 123, "model_pid_starttime": 1, "backend": "unknown"}
    grant, source = tmp_path / "grant", tmp_path / "source"
    grant.write_text("fixture")
    source.write_text("fixture")
    backend = Backend()
    result, seal = capture_pair(Endpoint(), backend, seed=42, journal_path=tmp_path / "journal",
        seal_path=tmp_path / "seal", capture_kind="fixture", provenance=prov,
        grant_receipt_path=grant, source_manifest_path=source, should_stop=lambda: True)
    assert backend.closed
    assert result["arms"]["control"]["error"]["type"] == "InterruptedError"
    assert seal["arms"]["attack"]["status"] == "not_attempted"
    assert not result["arms"]["control"]["events"]
