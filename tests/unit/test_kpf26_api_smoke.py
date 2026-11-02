"""Model-free coverage of the actual-API KPF-26 route collector."""
import hashlib
import json
import time

import httpx

from scripts.trace import kpf26_api_smoke as api
from scripts.trace import kpf26_prepare_smoke as prep


def test_route_scoring_never_infers_outcome_from_preview():
    report = {"scouts": {"schema": "epyc.orchestrator.scouts.v1", "role": "architect_critic",
                         "url": "http://127.0.0.1:80", "requested": 2, "launched": 2, "completed": 2,
                         "scouts": [{"status": "ok", "summary_preview": "partial"}, {"status": "ok"}]}}
    result = api.summarize(report, 2, "architect_critic", "http://127.0.0.1:80")
    assert result["status"] == "API_SCOUTS_OBSERVED" and result["outcome"] is None and result["gate_pass"] is None
    assert api.summarize(report, 2, "architect_critic", "http://127.0.0.1:81")["status"] == "INCOMPLETE"
    report["scouts"]["url"] = None
    assert api.summarize(report, 2, "architect_critic", "http://127.0.0.1:80")["status"] == "INCOMPLETE"
    report["scouts"]["url"] = "http://127.0.0.1:80"
    report["scouts"]["scouts"][1]["status"] = "timeout"
    assert api.summarize(report, 2, "architect_critic", "http://127.0.0.1:80")["status"] == "INCOMPLETE"


def test_mocked_http_collects_actual_route_and_native_data(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"no inference")
    server = {"binary_sha256": "a" * 64, "argv": ["server", str(model)], "ggml": {"lib": "b" * 64}}
    manifest = prep.prepare(tmp_path / "packet", pid=1, url="http://127.0.0.1:1111", role="architect_critic",
                            model=model, grant="test", source="1" * 40, snapshot=server, verify_endpoint=False)
    grant = {"grant_id": "test", "end_s": time.time() + 1200}
    monkeypatch.setenv("FILLER_DEADLINE", str(int(grant["end_s"])))
    monkeypatch.setattr(api, "active_grant", lambda _: grant)
    monkeypatch.setattr(api, "endpoint_identity", lambda pid, url: {"pid": pid, "url": url})
    monkeypatch.setattr(api, "api_process", lambda pid: {"pid": pid, "start_ticks": "1"})
    monkeypatch.setattr(api, "process_snapshot", lambda _: server)
    monkeypatch.setattr(api.subprocess, "check_output", lambda command, **kw: "1" * 40 if command[1] == "rev-parse" else "")

    class FakeSampler:
        def __init__(self, *_):
            self.samples = [{"t": 1, "cells": None}]
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return None

    captured = []
    def handler(req):
        captured.append(req)
        assert req.url.path == "/chat"
        body = json.loads(req.content)
        assert body["scouts"]["reserve_slots"] == 1 and body["real_mode"] is True
        assert body["client_deadline_unix_s"] <= grant["end_s"] - 15
        return httpx.Response(200, json={"scouts": {"schema": "epyc.orchestrator.scouts.v1",
            "role": "architect_critic", "url": "http://127.0.0.1:1111", "requested": 2,
            "launched": 2, "completed": 2, "scouts": [{"status": "ok"}, {"status": "ok"}]}})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(api, "Sampler", FakeSampler)
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    out = tmp_path / "out"
    assert api.run(manifest.parent, "http://127.0.0.1:8000", 2, out) == 0
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "API_SCOUTS_OBSERVED" and result["gate_pass"] is None
    assert result["flags"] == "unverified" and result["outcome"] is None
    assert hashlib.sha256((out / "response.bin").read_bytes()).hexdigest() == result["response_sha256"]
    assert len(captured) == 1


def test_stop_cancels_live_request(monkeypatch):
    import asyncio
    import threading
    import pytest

    async def handler(_request):
        await asyncio.sleep(30)
        return httpx.Response(200)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="stop/deadline"):
        asyncio.run(api._post_bounded("http://127.0.0.1:8000/chat", {}, 60, cancel))
