"""Non-inference tests for the real-stage KPF-26 collector."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/trace/kpf26_scout_executor.py"
spec = importlib.util.spec_from_file_location("kpf26_scout_executor", PATH)
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)


def test_pool_conflict_and_missing_are_unknown():
    assert executor._pool([{"kv_pool": {"used": 8}}, {"kv_pool": {"used": 8}}]) == 8
    assert executor._pool([{"kv_pool": {"used": 8}}, {"kv_pool": {"used": 9}}]) is None
    assert executor._pool([{"id": 1}]) is None
    assert executor._pool([{"kv_pool": {"used": -1}}]) is None


def test_witness_requires_task_slot_and_temporal_overlap():
    timing = {"id_slot": 1, "id_task": 42}
    samples = [{"t": 3, "slots": [{"id": 1, "id_task": 42, "checkpoints": [{"n_tokens": 20}]}]}]
    assert executor._slot_witness(samples, timing, 0, 2) is None
    assert executor._slot_witness(samples, {"id_slot": 1}, 2, 4) is None
    assert executor._slot_witness(samples, timing, 2, 4)["task"] == 42
    assert executor._checkpoint_witness(samples, timing, 2, 4)["checkpoints"] == [{"n_tokens": 20}]
    samples[0]["slots"][0]["checkpoints"] = []
    assert executor._checkpoint_witness(samples, timing, 2, 4) is None


def test_collect_real_scout_stage_retains_native_data_and_unknown(monkeypatch, tmp_path):
    from src.api.routes.chat_pipeline import scout_stage as stage
    from src.runtime import direct_region_claim

    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "input.txt").write_text("real scout content\n")
    scope_hash = hashlib.sha256((scope / "input.txt").read_bytes()).hexdigest()
    request = {"scope_root": str(scope), "scope_files": {"input.txt": scope_hash},
               "spec": {"max": 1, "max_turns": 1, "targets": [{"file": "input.txt"}]}}
    task = {"id": "../unsafe", "siblings": 1, "request": {"sha256": "a" * 64},
            "expected": {"sha256": "b" * 64}}

    class FakeTransport:
        name = "test"
        def __init__(self, *_args, **_kwargs):
            pass
        def complete(self, messages, **kwargs):
            stage._record_scout_call("scout_stage_chat_stream", "http://test", messages, 8,
                                    time.time(), 200, "stop", {"prompt_n": 5}, {}, None,
                                    {"first_output_ms": 2.0, "scout_admission": None})
            return stage.CompletionResult("SUMMARY\nanswer", 5, 2, "stop")

    class FakeSampler:
        def __init__(self, *_args):
            self.samples = []
        def __enter__(self):
            self.samples.append({"t": time.time() - 1, "cells": 5, "slots": []})
            return self
        def __exit__(self, *_args):
            self.samples.append({"t": time.time() + 1, "cells": 6, "slots": []})

    monkeypatch.setenv("FILLER_DEADLINE", str(int(time.time()) + 100))
    monkeypatch.setattr(executor, "Sampler", FakeSampler)
    monkeypatch.setattr(stage, "ChatCompletionsTransport", FakeTransport)
    monkeypatch.setattr(stage, "resolve_cap", lambda *a, **kw: {"cap": 1, "source": "mock"})
    monkeypatch.setattr(stage, "resolve_pool_gate", lambda *a, **kw: (None, None))
    monkeypatch.setattr(stage, "_record_scout_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(direct_region_claim, "resolve_claim_target", lambda *args: None)
    out = executor.collect_task(task, request, {"summaries": ["answer"]},
                                {"url": "http://test", "role": "worker", "sample_interval_s": 1}, tmp_path, tmp_path)
    assert out["children"][0]["outcome"] is True
    assert out["children"][0]["admission_wait_ms"] is None
    assert out["children"][0]["fork_task_slot_witness"] is None
    native = json.loads((tmp_path / out["native_path"]).read_text())
    assert native["summaries"] == ["answer"]
    assert native["calls"][0]["scout"] == 0
    assert (tmp_path / out["native_path"]).parent == tmp_path
    (scope / "input.txt").write_text("changed")
    with pytest.raises(ValueError, match="scope file set or digest mismatch"):
        executor.collect_task(task, request, {"summaries": ["answer"]},
                              {"url": "http://test", "role": "worker", "sample_interval_s": 1}, tmp_path, tmp_path)



def test_live_grant_requires_launcher_window_and_unchanged_file(monkeypatch, tmp_path):
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    start = now - dt.timedelta(minutes=1)
    end = now + dt.timedelta(minutes=20)
    (tmp_path / "row.grant").write_text(f"regions=0-1 start={start:%Y-%m-%dT%H:%MZ} end={end:%Y-%m-%dT%H:%MZ} holds=\n")
    monkeypatch.setattr(executor, "GRANTS", tmp_path)
    monkeypatch.setenv("FILLER_ID", "row")
    monkeypatch.setenv("FILLER_END", str(int(end.timestamp())))
    monkeypatch.setenv("FILLER_DEADLINE", str(int(end.timestamp()) - 90))
    monkeypatch.setenv("FILLER_STOPFILE", str(tmp_path / "stop"))
    assert executor.active_grant("row")["regions"] == "0-1"
    monkeypatch.setenv("FILLER_ID", "other")
    with pytest.raises(ValueError, match="matching filler"):
        executor.active_grant("row")
    monkeypatch.setenv("FILLER_ID", "row")
    (tmp_path / "stop").touch()
    with pytest.raises(ValueError, match="stop file"):
        executor.active_grant("row")


def test_pid_must_own_the_actual_listening_socket():
    import os
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        assert executor.endpoint_identity(os.getpid(), url)["pid"] == os.getpid()
        with pytest.raises(ValueError, match="local explicit"):
            executor.endpoint_identity(os.getpid(), "http://example.org:80")


def test_complete_observation_emits_native_record(monkeypatch, tmp_path):
    def pin(name, content):
        data = json.dumps(content).encode()
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    req, expected = pin("request.json", {}), pin("expected.json", {})
    source = "1" * 40
    model = tmp_path / "model.gguf"
    model.write_bytes(b"mock model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    process = {"exe": "/server", "binary_sha256": "a" * 64, "argv": ["server", str(model)], "ggml": {"/lib": "b" * 64}}
    grant_path = tmp_path / "row.grant"
    grant_path.write_text("grant")
    grant = {"grant_id": "row", "path": str(grant_path), "sha256": "a" * 64, "start_s": 0, "end_s": 9999999999, "regions": "0-1"}
    endpoint = {"url": "http://127.0.0.1:1", "pid": 1, "listener_inodes": ["1"]}
    m = {"schema": "epyc.kpf26.paired.v1", "arm_order": ["off", "on"], "arm_timeout_s": 30,
         "max_occupancy_sample_gap_s": 2,
         "pins": {"model": str(model), "model_sha256": model_sha, "server_binary_sha256": "a" * 64, "source_commit": source,
                  "argv": "server " + str(model), "executor_sha256": "b" * 64},
         "runtime": {"off": {"url": "http://127.0.0.1:1", "role": "worker", "server_pid": 1,
                             "window_id": "w", "sample_interval_s": 1, "grant_id": "row"}},
         "tasks": [{"id": "t", "siblings": 1, "request": req, "expected": expected}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    monkeypatch.setattr(executor, "process_snapshot", lambda pid: process)
    monkeypatch.setattr(executor, "active_grant", lambda row: grant)
    monkeypatch.setattr(executor, "endpoint_identity", lambda pid, url: endpoint)
    monkeypatch.setattr(executor.subprocess, "check_output",
                        lambda command, **kwargs: source if command[1] == "rev-parse" else "")
    monkeypatch.setattr(executor, "collect_task", lambda task, *a: {
        "id": task["id"], "request_sha256": req["sha256"], "expected_sha256": expected["sha256"],
        "start_s": 1, "end_s": 2, "unique_cells_samples": [{"t": 1, "cells": 1}, {"t": 2, "cells": 1}],
        "children": [{"id": "0", "status": "ok", "ttft_ms": 1, "admission_wait_ms": 0,
                      "prefill_tokens": 1, "output_sha256": "c" * 64, "outcome": True}]})
    evidence = tmp_path / "out.json"
    assert executor.run(path, "off", evidence) == 0
    result = json.loads(evidence.read_text())
    assert result["grant_authority_verified"] is True
    row = json.loads((tmp_path / "out-native" / "belief_measurements.jsonl").read_text())
    assert row["kpf26"] == {"tasks": 1, "children": 1, "completed": 1}
    assert row["rows_sha256"] == hashlib.sha256((tmp_path / "out-native" / "belief_rows.jsonl").read_bytes()).hexdigest()


def test_required_terms_scorer_preserves_exact_and_unknown():
    exact = executor.validate_scorer({"summaries": ["answer"]}, 1)[0]
    assert executor.score_summary("answer", exact) is True
    assert executor.score_summary("Answer", exact) is False
    assert executor.score_summary(None, exact) is None
    rule = executor.validate_scorer({"schema": "epyc.kpf26.scorer.required-terms.v1",
                                     "required_terms": [["RELEASE_CODENAME", "amber-heron-47"]]}, 1)[0]
    assert executor.score_summary("release_codename is amber-heron-47", rule) is True
    assert executor.score_summary("RELEASE_CODENAME", rule) is False
    with pytest.raises(ValueError, match="required_terms"):
        executor.validate_scorer({"schema": "epyc.kpf26.scorer.required-terms.v1", "required_terms": [[]]}, 1)
