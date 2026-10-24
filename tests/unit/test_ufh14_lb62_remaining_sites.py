"""UFH14-LB6.2 (remaining unrecorded HTTP call sites): scout_stage, vl_describe,
research worker synthesis, LlamaServerBackend.infer_stream."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.backends import serving_calls as sc


@pytest.fixture
def log_file(monkeypatch, tmp_path) -> Path:
    path = tmp_path / "serving_calls" / "serving_calls.jsonl"
    monkeypatch.setenv(sc.LOG_ENV, str(path))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    sc._SIDECAR_CACHE.clear()
    sc.clear_staged()
    yield path
    sc.clear_staged()


def _recs(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _boom(*a, **k):
    raise RuntimeError("recorder down")


# ---- scout_stage ---------------------------------------------------------------

class _StreamResp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self): pass
    def iter_lines(self): return iter(self._lines)
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _sse(obj) -> str:
    return "data: " + json.dumps(obj)


def _scout_lines():
    return [
        _sse({"choices": [{"delta": {"content": "he"}}]}),
        _sse({"choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 7, "completion_tokens": 2},
              "timings": {"prompt_n": 7, "predicted_n": 2}}),
        "data: [DONE]",
    ]


def test_scout_complete_records_and_result_unchanged(log_file):
    from src.api.routes.chat_pipeline.scout_stage import ChatCompletionsTransport

    client = MagicMock()
    client.stream.return_value = _StreamResp(_scout_lines())
    t = ChatCompletionsTransport("http://127.0.0.1:8091", client=client)
    res = t.complete([{"role": "user", "content": "hi"}], max_tokens=16,
                     should_stop=lambda: False, timeout_s=5)
    assert res.text == "hello" and res.completion_tokens == 2
    (r,) = _recs(log_file)
    assert r["schema"] == sc.SCHEMA and r["method"] == "scout_stage_chat_stream"
    assert r["finish_reason"] == "stop" and r["server"]["port"] == 8091
    assert r["notes"]["endpoint"] == "/v1/chat/completions"
    assert r["notes"]["stream_gaps"]["n_gaps"] == 2
    assert r["timings"]["prompt_n"] == 7


def test_scout_complete_cancel_and_recorder_failure(log_file, monkeypatch):
    from src.api.routes.chat_pipeline.scout_stage import ChatCompletionsTransport

    client = MagicMock()
    client.stream.return_value = _StreamResp(_scout_lines())
    t = ChatCompletionsTransport("http://127.0.0.1:8091", client=client)
    res = t.complete([{"role": "user", "content": "hi"}], max_tokens=16,
                     should_stop=lambda: True, timeout_s=5)
    assert res.cancelled
    (r,) = _recs(log_file)
    assert r["finish_reason"] == "cancelled" and r["outcome"] == "cancelled"

    monkeypatch.setattr(sc, "record_http_call", _boom)
    client.stream.return_value = _StreamResp(_scout_lines())
    assert t.complete([{"role": "user", "content": "hi"}], max_tokens=16,
                      should_stop=lambda: False, timeout_s=5).text == "hello"


def test_scout_prefill_trunk_records_and_failure_safe(log_file, monkeypatch):
    from src.api.routes.chat_pipeline.scout_stage import ChatCompletionsTransport

    resp = MagicMock(status_code=200)
    resp.json.return_value = {"timings": {"prompt_n": 5, "cache_n": 1}}
    client = MagicMock()
    client.post.return_value = resp
    t = ChatCompletionsTransport("http://127.0.0.1:8091", client=client)
    out = t.prefill_trunk([{"role": "user", "content": "hi"}], timeout_s=5)
    assert out == {"timings": {"prompt_n": 5, "cache_n": 1}}
    (r,) = _recs(log_file)
    assert r["method"] == "scout_stage_prefill_trunk" and r["timings"]["cache_n"] == 1

    monkeypatch.setattr(sc, "record_http_call", _boom)
    assert t.prefill_trunk([{"role": "user", "content": "hi"}], timeout_s=5)["timings"]


# ---- vl_describe ---------------------------------------------------------------

class _R:
    def __init__(self, payload, status=200):
        self._p, self.status = payload, status

    def read(self): return json.dumps(self._p).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _vl(tmp_path):
    from src.vision.analyzers.vl_describe import VLDescribeAnalyzer

    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG")
    return VLDescribeAnalyzer(server_port=8190), img


def test_vl_server_call_records(log_file, tmp_path):
    a, img = _vl(tmp_path)
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "a cat"}}],
               "usage": {"prompt_tokens": 30, "completion_tokens": 3}}
    with patch("urllib.request.urlopen", return_value=_R(payload)):
        res = a._analyze_with_server(img, 0.0)
    assert res.success
    (r,) = _recs(log_file)
    assert r["method"] == "vl_describe_chat" and r["finish_reason"] == "stop"
    assert r["server"]["port"] == 8190 and r["result"]["tokens_generated"] == 3


def test_vl_server_call_http_error_and_recorder_failure(log_file, tmp_path, monkeypatch):
    a, img = _vl(tmp_path)
    with patch("urllib.request.urlopen", return_value=_R({"x": 1}, status=500)):
        res = a._analyze_with_server(img, 0.0)
    assert not res.success
    (r,) = _recs(log_file)
    assert r["outcome"] == "failed" and r["notes"]["http_status"] == 500

    monkeypatch.setattr(sc, "record_http_call", _boom)
    payload = {"choices": [{"message": {"content": "ok"}}]}
    with patch("urllib.request.urlopen", return_value=_R(payload)):
        assert a._analyze_with_server(img, 0.0).success


# ---- research worker synthesis --------------------------------------------------

def test_research_synthesis_records_and_recorder_failure(log_file, monkeypatch):
    from src.tools.web import research

    body = {"content": " synth ", "stop_type": "eos", "tokens_predicted": 3, "tokens_evaluated": 9,
            "timings": {"prompt_n": 9, "predicted_n": 3}}
    monkeypatch.setattr(research, "_worker_synthesis_target",
                        lambda: ("http://127.0.0.1:8083/completion", "gemma-4-26B-A4B-it-Q4_K_M"))
    with patch("urllib.request.urlopen", return_value=_R(body)):
        out = research._synthesize_page("http://x.test/p", "T", "some page text", "q")
    assert out["success"] and out["synthesis"] == "synth"
    (r,) = _recs(log_file)
    assert r["method"] == "research_synthesis_completion" and r["role"] == "worker_general"
    assert r["finish_reason"] == "eos" and r["server"]["port"] == 8083

    monkeypatch.setattr(sc, "record_http_call", _boom)
    with patch("urllib.request.urlopen", return_value=_R(body)):
        assert research._synthesize_page("http://x.test/p", "T", "some page text", "q")["success"]


# ---- LlamaServerBackend.infer_stream --------------------------------------------

def _backend_and_stream(lines):
    from src.backends.llama_server import LlamaServerBackend

    b = LlamaServerBackend(base_url="http://127.0.0.1:8084")
    resp = _StreamResp(lines)
    b.client = MagicMock()
    b.client.stream.return_value = resp
    return b


def _req_role():
    from src.model_server import InferenceRequest

    return SimpleNamespace(name="worker_x", model=None), InferenceRequest(role="worker_x", prompt="p", n_tokens=8)


def test_infer_stream_records_finish_and_gaps(log_file):
    lines = [
        "data: " + json.dumps({"tokens": [1, 2]}),
        "data: " + json.dumps({"tokens": [3]}),
        "data: " + json.dumps({"stop": True, "stop_type": "eos",
                               "timings": {"prompt_n": 4, "predicted_n": 3}}),
    ]
    b = _backend_and_stream(lines)
    role, req = _req_role()
    with patch.object(type(b), "_build_payload", return_value={"prompt": "p"}):
        toks = [t for t, _ in b.infer_stream(role, req)]
    assert toks == [1, 2, 3]
    (r,) = _recs(log_file)
    assert r["method"] == "llama_server_infer_stream" and r["finish_reason"] == "eos"
    assert r["server"]["port"] == 8084 and r["result"]["tokens_generated"] == 3
    assert r["notes"]["stream_gaps"]["n_gaps"] == 2


def test_infer_stream_early_close_and_recorder_failure(log_file, monkeypatch):
    lines = ["data: " + json.dumps({"tokens": [1]}), "data: " + json.dumps({"tokens": [2]})]
    b = _backend_and_stream(lines)
    role, req = _req_role()
    with patch.object(type(b), "_build_payload", return_value={"prompt": "p"}):
        g = b.infer_stream(role, req)
        assert next(g)[0] == 1
        g.close()
    (r,) = _recs(log_file)
    assert r["outcome"] == "early_stop"

    monkeypatch.setattr(sc, "record_http_call", _boom)
    b.client.stream.return_value = _StreamResp(lines)
    with patch.object(type(b), "_build_payload", return_value={"prompt": "p"}):
        assert [t for t, _ in b.infer_stream(role, req)] == [1, 2]
