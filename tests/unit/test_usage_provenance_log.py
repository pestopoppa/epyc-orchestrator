"""Provenance survives outside the response model and in-memory call log."""

import json
import hashlib
import subprocess
import pytest

from src.backends import serving_calls
from src.llm_cache import ContentAddressableCache
from src.llm_primitives.usage import UsageLedger
from src.llm_primitives import usage_log
from tests.unit.test_usage_provenance import metered_primitives, native_result


TEST_ONLY_DECLARATION = {
    "schema": "epyc.request_call_usage.measurement_declaration.v1",
    "metrics": [{
        "source_field": "completion_tokens", "metric": "test_completion_tokens",
        "unit": "tokens", "category": "CANDIDATE", "metric_direction": "higher_better",
        "protocol_id": "", "claim": "Test fixture only: exact completion count for one call.",
        "reps_basis": "one captured call",
    }],
}


def enable_usage_log(monkeypatch, tmp_path):
    serving_path = tmp_path / "serving_calls.jsonl"
    monkeypatch.setenv(serving_calls.LOG_ENV, str(serving_path))
    return tmp_path / "usage_provenance.jsonl"


def test_pending_error_and_cache_replay_have_durable_self_hashed_identity(monkeypatch, tmp_path):
    path = enable_usage_log(monkeypatch, tmp_path)
    ledger = UsageLedger(request_id="request-one")
    pending_id = ledger.begin("worker")
    failed_id = ledger.begin("worker")
    ledger.finish(failed_id, error="connection lost", censored=True)
    replay_id = ledger.begin("worker")
    ledger.finish(
        replay_id, prompt_tokens=9, completion_tokens=0,
        provenance="cached_replay_of_server_reported", source="content_cache",
        replay_of_call_id="original-server-observation", finish_reason="stop",
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 5
    assert all(record["schema"] == "epyc.request_call_usage.v1" for record in records)
    assert all(record["request_id"] == "request-one" for record in records)
    assert all(record["record_sha256"] == serving_calls.record_digest(record) for record in records)
    assert all(record["process"]["orch_commit"] for record in records)
    assert all(record["call"]["measurement_declaration"] is None for record in records)
    assert records[0]["call"]["call_id"] == pending_id
    assert records[0]["call"]["pending"] and records[0]["call"]["censored"]
    assert "error" not in records[2]["call"]
    assert records[2]["call"]["error_category"] == "generation_error"
    assert records[2]["call"]["error_sha256"] == hashlib.sha256(b"connection lost").hexdigest()
    assert records[2]["call"]["completion_tokens"] is None
    assert records[-1]["call"]["provenance"] == "cached_replay_of_server_reported"
    assert records[-1]["call"]["replay_of_call_id"] == "original-server-observation"
    assert records[-1]["call"]["completion_tokens"] == 0
    assert not (tmp_path / "serving_calls.jsonl").exists()


def test_optional_measurement_declaration_is_detached_and_self_hashed(monkeypatch, tmp_path):
    path = enable_usage_log(monkeypatch, tmp_path)
    ledger = UsageLedger(request_id="declared-test-only")
    declaration = json.loads(json.dumps(TEST_ONLY_DECLARATION))
    call_id = ledger.begin("worker", measurement_declaration=declaration)
    declaration["metrics"][0]["claim"] = "mutated after begin"
    ledger.finish(call_id, completion_tokens=0, provenance="server_reported",
                  source="llama_server:/completion.tokens_predicted")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["begin", "settled"]
    for record in records:
        assert record["record_sha256"] == serving_calls.record_digest(record)
        assert record["call"]["measurement_declaration"] == TEST_ONLY_DECLARATION
    assert records[-1]["call"]["completion_tokens"] == 0
    assert records[-1]["call"]["measurement_declaration"] == records[0]["call"]["measurement_declaration"]


def test_usage_declaration_is_absent_by_default_and_rejects_invalid_claim_tuple_context():
    ledger = UsageLedger("unannotated-default")
    ledger.begin("worker")
    assert ledger.records()[0]["measurement_declaration"] is None

    invalid = json.loads(json.dumps(TEST_ONLY_DECLARATION))
    invalid["metrics"][0]["metric_direction"] = "neutral"
    with pytest.raises(ValueError, match="ClaimTuple direction"):
        ledger.begin("worker", measurement_declaration=invalid)


def test_real_primitive_and_cache_replay_emit_without_new_physical_decode(monkeypatch, tmp_path):
    path = enable_usage_log(monkeypatch, tmp_path)
    primitives, server = metered_primitives(monkeypatch, [native_result()])
    primitives._content_cache = ContentAddressableCache(tmp_path / "cache")
    primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    primitives.llm_call("same", n_tokens=8, skip_suffix=True)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    settled = [record["call"] for record in records if record["event"] == "settled"]
    assert len(settled) == 2
    assert len(server.calls) == 1
    assert primitives.total_tokens_generated == 4
    assert primitives.get_request_usage()["completion_tokens"] == 8
    assert settled[1]["replay_of_call_id"] == settled[0]["call_id"]
    assert settled[1]["provenance"] == "cached_replay_of_server_reported"


def test_usage_rotation_is_bounded_and_disabled_writer_stays_disabled(monkeypatch, tmp_path):
    path = enable_usage_log(monkeypatch, tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_USAGE_LOG_MAX_BYTES", "1024")
    ledger = UsageLedger("rotation")
    for _ in range(12):
        call_id = ledger.begin("worker")
        ledger.finish(call_id, prompt_tokens=1, completion_tokens=1, provenance="server_reported")
    files = list(tmp_path.glob("usage_provenance.jsonl*"))
    data_files = [file for file in files if not file.name.endswith(".lock")]
    assert 1 < len(data_files) <= 4
    assert all(file.stat().st_size <= 1024 + 32768 for file in data_files)
    previous = path.read_bytes()
    monkeypatch.setenv(serving_calls.LOG_ENV, "off")
    ledger.finish(ledger.begin("mock"), prompt_tokens=1, completion_tokens=1, provenance="estimated")
    assert path.read_bytes() == previous


def test_write_failure_is_visible_without_changing_usage_counts(monkeypatch, tmp_path):
    before = usage_log.health()["write_failed"]
    monkeypatch.setattr(usage_log, "log_path", lambda: tmp_path)
    ledger = UsageLedger("write-failure")
    ledger.finish(ledger.begin("worker"), prompt_tokens=1, completion_tokens=2, provenance="server_reported", source="llama_server:/completion.tokens_predicted")
    assert usage_log.health()["write_failed"] > before
    assert ledger.summary()["completion_tokens"] == 2


def test_error_payload_is_not_written_and_arbitrary_finish_reason_is_not_a_channel(monkeypatch, tmp_path):
    path = enable_usage_log(monkeypatch, tmp_path)
    secret = "PRIVATE PROMPT / MODEL OUTPUT / SERVER PAYLOAD"
    ledger = UsageLedger("privacy")
    ledger.finish(ledger.begin("worker"), error=secret, finish_reason=secret, censored=True)
    serialized = path.read_text()
    assert secret not in serialized
    call = json.loads(serialized.splitlines()[-1])["call"]
    assert call["finish_reason"] == "unknown"
    assert call["error_category"] == "generation_error"
    assert call["error_sha256"] == hashlib.sha256(secret.encode()).hexdigest()


def test_source_context_records_dirty_disk_snapshot_and_caches_honestly(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    relative = "src/llm_primitives/usage.py"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text("# proposed source snapshot\n")
    monkeypatch.setattr(usage_log, "_REPO_ROOT", tmp_path)
    first = usage_log._capture_source_context(str(tmp_path), 123456)
    assert first["worktree_state_at_capture"] == "dirty"
    assert first["snapshot_kind"] == "on_disk_async_snapshot"
    assert first["loaded_bytes_attestation"] == "not_available"
    assert first["head_commit_at_capture"] is None
    assert first["relevant_source_complete"] is False
    assert first["relevant_source_sha256"][relative] == hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("# changed after first event\n")
    monkeypatch.setitem(usage_log._context_cache, (str(tmp_path), usage_log.os.getpid()), first)
    second = usage_log.source_context()
    assert second == first
    assert second["relevant_source_sha256"][relative] != hashlib.sha256(source.read_bytes()).hexdigest()


def test_unavailable_git_state_does_not_claim_a_clean_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(usage_log, "_git_text", lambda *args: None)
    context = usage_log._capture_source_context(str(tmp_path), 123456)
    assert context["worktree_state_at_capture"] == "unknown"
    assert context["head_commit_at_capture"] is None
    assert context["loaded_bytes_attestation"] == "not_available"


def test_first_write_does_not_wait_for_git_or_source_hashing(monkeypatch, tmp_path):
    import threading
    path = enable_usage_log(monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(usage_log, "_REPO_ROOT", tmp_path / "unique-root")
    def blocked_capture(*args):
        entered.set()
        assert release.wait(5)
        return {"snapshot_kind": "on_disk_async_snapshot", "loaded_bytes_attestation": "not_available"}
    monkeypatch.setattr(usage_log, "_capture_source_context", blocked_capture)
    writer = threading.Thread(target=lambda: UsageLedger("nonblocking").begin("worker"))
    try:
        writer.start()
        assert entered.wait(1)
        writer.join(timeout=1)
        assert not writer.is_alive(), "request writer waited for source/Git capture"
        record = json.loads(path.read_text().splitlines()[0])
        assert record["source_context"]["reason"] == "async_capture_pending"
        assert record["source_context"]["worktree_state_at_capture"] == "unknown"
    finally:
        release.set()
        writer.join(timeout=5)


def test_log_lock_and_rotation_are_private_even_with_permissive_umask(monkeypatch, tmp_path):
    import os
    import stat
    path = enable_usage_log(monkeypatch, tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_USAGE_LOG_MAX_BYTES", "1024")
    prior = os.umask(0)
    try:
        ledger = UsageLedger("private")
        for _ in range(8):
            ledger.begin("worker")
    finally:
        os.umask(prior)
    files = list(tmp_path.glob("usage_provenance.jsonl*"))
    assert len(files) > 2
    assert all(stat.S_IMODE(file.stat().st_mode) == 0o600 for file in files)
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["persistence"] == "flushed_append_no_fsync"


def test_existing_log_permissions_are_hardened_and_symlink_is_refused(monkeypatch, tmp_path):
    import stat
    path = enable_usage_log(monkeypatch, tmp_path)
    path.write_text("")
    path.chmod(0o666)
    UsageLedger("harden").begin("worker")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    before = usage_log.health()["write_failed"]
    path.unlink()
    target = tmp_path / "unrelated"
    target.write_text("preserve me")
    path.symlink_to(target)
    UsageLedger("no-symlink").begin("worker")
    assert usage_log.health()["write_failed"] > before
    assert target.read_text() == "preserve me"


def test_multiprocess_rotation_keeps_complete_private_self_hashed_records(monkeypatch, tmp_path):
    import os
    import stat
    import sys
    enable_usage_log(monkeypatch, tmp_path)
    env = dict(os.environ, ORCHESTRATOR_USAGE_LOG_MAX_BYTES="2048", PYTHONDONTWRITEBYTECODE="1")
    code = '''
import sys
from src.llm_primitives import usage_log
usage_log.source_context = lambda: {"snapshot_kind": "synthetic_rotation_fixture", "loaded_bytes_attestation": "not_available"}
for i in range(40):
    assert usage_log.write_usage_event(sys.argv[1], "begin", {"call_id": str(i), "pending": True})
print(usage_log.health()["recorded"])
'''
    children = [subprocess.Popen([sys.executable, "-B", "-c", code, f"writer-{i}"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(3)]
    try:
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            assert stdout.strip() == "40"
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            assert child.poll() is not None
    files = [p for p in tmp_path.glob("usage_provenance.jsonl*") if p.suffix != ".lock"]
    assert 1 < len(files) <= 4
    records = []
    for file in files:
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        raw = file.read_bytes()
        assert raw.endswith(b"\n")
        records.extend(json.loads(line) for line in raw.splitlines())
    assert 0 < len(records) <= 120  # bounded rotation intentionally retires old records
    assert all(r["record_sha256"] == serving_calls.record_digest(r) for r in records)
    assert len({(r["request_id"], r["call"]["call_id"]) for r in records}) == len(records)


def _finish_owned_child(process):
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    assert not process.is_alive()


def test_fork_resets_held_recorder_context_and_ledger_mutexes(monkeypatch, tmp_path):
    import multiprocessing
    enable_usage_log(monkeypatch, tmp_path)
    monkeypatch.setattr(usage_log, "_capture_source_context", lambda *a: {"snapshot_kind": "synthetic_fork_fixture", "loaded_bytes_attestation": "not_available"})
    context = multiprocessing.get_context("fork")
    for iteration in range(5):
        ledger = UsageLedger(f"parent-{iteration}")
        receiver, sender = context.Pipe(duplex=False)
        def child():
            assert ledger.summary()["usage_status"] == "unknown"
            ledger.begin("child")
            sender.send((ledger.summary()["usage_status"], usage_log.health()["recorded"]))
        locks = (usage_log._lock, usage_log._context_lock, ledger._lock)
        for lock in locks:
            lock.acquire()
        process = context.Process(target=child)
        try:
            process.start()
            assert receiver.poll(5), "child waited on an inherited parent mutex"
            status, recorded = receiver.recv()
            assert status == "unknown" and recorded == 2
            process.join(timeout=5)
            assert process.exitcode == 0
        finally:
            for lock in reversed(locks):
                lock.release()
            _finish_owned_child(process)
            receiver.close()
            sender.close()


def test_child_closes_inherited_recorder_flock_descriptor(monkeypatch, tmp_path):
    import fcntl
    import multiprocessing
    path = enable_usage_log(monkeypatch, tmp_path)
    monkeypatch.setattr(usage_log, "source_context", lambda: {"snapshot_kind": "synthetic_fork_fixture"})
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    def child():
        sender.send("started")
        UsageLedger("child-fd").begin("worker")
        sender.send("written")
    process = context.Process(target=child)
    try:
        with usage_log._private_append(path.with_name(path.name + ".lock")) as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            process.start()
            assert receiver.poll(5) and receiver.recv() == "started"
            # Close our copy without LOCK_UN: the child must have closed its copy.
        assert receiver.poll(5) and receiver.recv() == "written"
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        _finish_owned_child(process)
        receiver.close()
        sender.close()
