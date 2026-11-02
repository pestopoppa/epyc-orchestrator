"""Opt-in private API-stage evidence; no model or network calls."""
import asyncio
import hashlib
import json
import os
import stat
from types import SimpleNamespace

import pytest

from src.api.routes.chat_pipeline import scout_evidence as E
from scripts.trace import kpf26_api_smoke as collector


def test_default_off_and_exact_request(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_KPF26_TRACE_DIR", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_KPF26_TRACE_REQUEST_ID", raising=False)
    assert E.Capture.begin("target") is None
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_REQUEST_ID", "target")
    assert E.Capture.begin("other") is None
    assert list(tmp_path.iterdir()) == []


def test_trace_private_bound_and_threads(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_REQUEST_ID", "target")
    monkeypatch.setenv("ORCHESTRATOR_PREFIX_INDEX", "1")
    capture = E.Capture.begin("target")
    assert capture is not None
    def one(i):
        token = E.scout_start(i)
        try:
            E.record_call("scout_stage_chat_stream", "http://x", 1, 200, "stop",
                          {"id_slot": i, "id_task": i + 10}, {}, None, False)
        finally:
            E.scout_end(token)
    async def exercise():
        await asyncio.gather(*(asyncio.to_thread(one, i) for i in range(2)))
    asyncio.run(exercise())
    summary = "the complete private summary " * 20
    stage = SimpleNamespace(report={"schema": "x"}, results=[SimpleNamespace(index=0, status="ok", summary=summary)])
    capture.finish(stage)
    file = tmp_path / (hashlib.sha256(b"target").hexdigest() + ".json")
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    raw = json.loads(file.read_text())
    assert raw["request_id"] == "target" and raw["complete"] and raw["summaries"][0]["summary"] == summary
    assert raw["flags_raw"]["ORCHESTRATOR_PREFIX_INDEX"] == "1"
    assert {row["scout"] for row in raw["calls"]} == {0, 1}
    assert E._CURRENT.get() is None
    with pytest.raises(FileExistsError):
        E.Capture.begin("target")


def test_shared_directory_and_symlink_are_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_REQUEST_ID", "x")
    tmp_path.chmod(0o755)
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_DIR", str(tmp_path))
    with pytest.raises(PermissionError):
        E.Capture.begin("x")
    tmp_path.chmod(0o700)
    outside = tmp_path / "outside"
    outside.write_text("safe")
    name = hashlib.sha256(b"x").hexdigest() + ".json"
    (tmp_path / name).symlink_to(outside)
    with pytest.raises(FileExistsError):
        E.Capture.begin("x")
    assert outside.read_text() == "safe"


def test_api_trace_scores_full_summary_and_does_not_invent_witness(tmp_path):
    import time
    request_id = "kpf26-smoke"
    summary = "release amber-heron-47"
    report = {"scouts": [{"status": "ok", "summary_sha256": hashlib.sha256(summary.encode()).hexdigest()}]}
    now = time.time()
    trace = {"schema": "epyc.kpf26.api-scout-trace.v1", "complete": True, "request_id": request_id,
             "pid": os.getpid(), "start_s": now, "end_s": now + 1, "flags_raw": {"fork": None},
             "report": report, "summaries": [{"index": 0, "status": "ok", "summary": summary}],
             "calls": [{"scout": 0, "method": "scout_stage_chat_stream", "ts_start": now,
                        "ts_end": now + 1, "timings": {"id_slot": 3, "id_task": 4, "n_fork_tokens": 10}}]}
    path = tmp_path / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")
    path.write_text(json.dumps(trace))
    path.chmod(0o600)
    raw, result = collector.read_trace(tmp_path, request_id, os.getpid(), {"scouts": report}, now - 1, now + 2,
                                       [], [{"mode": "required_terms", "value": ["amber-heron-47"]}])
    assert hashlib.sha256(raw).hexdigest() == result["trace_sha256"]
    assert result["children"][0]["outcome"] is True
    assert result["children"][0]["slot_witness"] is None and result["checkpoint_witness"] is None
    trace["summaries"][0]["summary"] = "other"
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="summary"):
        collector.read_trace(tmp_path, request_id, os.getpid(), {"scouts": report}, now - 1, now + 2,
                             [], [{"mode": "required_terms", "value": ["amber-heron-47"]}])


def test_chat_stage_emits_only_selected_request_without_changing_response(monkeypatch, tmp_path):
    from src.api.routes import chat
    from src.api.routes.chat_pipeline import scout_stage
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_REQUEST_ID", "target")
    monkeypatch.setattr(chat.task_root_mod, "request_scope", lambda: None)
    summary = "a non-truncated private summary " * 12
    report = {"schema": scout_stage.SCHEMA, "scouts": [{"status": "ok", "summary_sha256": hashlib.sha256(summary.encode()).hexdigest()}]}
    stage = SimpleNamespace(report=report, block="findings", results=[SimpleNamespace(index=0, status="ok", summary=summary)])
    async def fake_run(*_args, **_kwargs):
        def call():
            token = E.scout_start(0)
            try:
                scout_stage._record_scout_call("scout_stage_chat_stream", "http://x", [], 1, 1,
                                               200, "stop", {"id_slot": 3, "id_task": 4}, {}, None)
            finally:
                E.scout_end(token)
        await asyncio.to_thread(call)
        return stage
    monkeypatch.setattr(scout_stage, "run_scouts", fake_run)
    class Request:
        scouts = SimpleNamespace(role=None)
        force_role = "r"
        request_id = "target"
        prompt = "original"
        def model_copy(self, update):
            return update
    holder = {}
    out = asyncio.run(chat._run_scout_stage(Request(), None, "r", None, 100, holder))
    assert out["prompt"].endswith("original")
    assert holder["report"] == report
    assert "summary" not in holder["report"]["scouts"][0]
    path = tmp_path / (hashlib.sha256(b"target").hexdigest() + ".json")
    trace = json.loads(path.read_text())
    assert trace["summaries"][0]["summary"] == summary
    assert trace["calls"][0]["scout"] == 0 and trace["calls"][0]["timings"]["id_task"] == 4


def test_trace_requires_same_task_slot_and_checkpoint_sample(tmp_path):
    import time
    now = time.time()
    request_id = "x"
    text = "correct"
    report = {"scouts": [{"status": "ok", "summary_sha256": hashlib.sha256(text.encode()).hexdigest()}]}
    trace = {"schema": "epyc.kpf26.api-scout-trace.v1", "complete": True, "request_id": request_id,
             "pid": os.getpid(), "start_s": now, "end_s": now + 1, "flags_raw": {}, "report": report,
             "summaries": [{"index": 0, "status": "ok", "summary": text}],
             "calls": [
                 {"scout": 0, "method": "scout_stage_chat_stream", "ts_start": now, "ts_end": now + 1,
                  "timings": {"id_slot": 3, "id_task": 4, "n_fork_tokens": 10}},
                 {"scout": 0, "method": "scout_stage_prefill_trunk", "ts_start": now, "ts_end": now + 1,
                  "timings": {"id_slot": 5, "id_task": 6}},
             ]}
    path = tmp_path / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")
    path.write_text(json.dumps(trace))
    path.chmod(0o600)
    samples = [{"t": now + 0.5, "slots": [{"id": 3, "id_task": 999},
                {"id": 5, "id_task": 6, "checkpoints": [123]}]}]
    scorer = [{"mode": "exact", "value": "correct"}]
    _, result = collector.read_trace(tmp_path, request_id, os.getpid(), {"scouts": report},
                                     now - 1, now + 2, samples, scorer)
    assert result["children"][0]["slot_witness"] is None
    assert result["checkpoint_witness"]["task"] == 6
    samples[0]["slots"][0]["id_task"] = 4
    _, result = collector.read_trace(tmp_path, request_id, os.getpid(), {"scouts": report},
                                     now - 1, now + 2, samples, scorer)
    assert result["children"][0]["slot_witness"]["task"] == 4
    samples[0]["t"] = now + 3
    _, result = collector.read_trace(tmp_path, request_id, os.getpid(), {"scouts": report},
                                     now - 1, now + 2, samples, scorer)
    assert result["children"][0]["slot_witness"] is None and result["checkpoint_witness"] is None


def test_trace_failures_do_not_change_chat_stage(monkeypatch, tmp_path):
    from src.api.routes import chat
    from src.api.routes.chat_pipeline import scout_stage
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_KPF26_TRACE_REQUEST_ID", "target")
    tmp_path.chmod(0o755)
    monkeypatch.setattr(chat.task_root_mod, "request_scope", lambda: None)
    async def run(*_args, **_kwargs):
        return SimpleNamespace(report={"ok": True}, block="", results=[])
    monkeypatch.setattr(scout_stage, "run_scouts", run)
    req = SimpleNamespace(scouts=SimpleNamespace(role=None), force_role="r", request_id="target")
    holder = {}
    assert asyncio.run(chat._run_scout_stage(req, None, "r", None, 100, holder)) is req
    assert holder["report"] == {"ok": True}
    assert list(tmp_path.iterdir()) == []
