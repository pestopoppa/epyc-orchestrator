"""UFH14-LB3.1 (turn-1 output corpus) and UFH14-LB6.2 (unrecorded call sites)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backends import serving_calls as sc
from src.graph.helpers import TURN1_PROBE_HEAD_CHARS, turn1_output_probe
from src.graph.state import TaskState
from src.pipeline_monitor.diagnostic import build_diagnostic


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


# ---- LB3.1 ---------------------------------------------------------------

def test_probe_hash_head_and_truncation():
    text = 'x = CALL("t", a=1)\n' * 100
    p = turn1_output_probe(text)
    assert p["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert p["chars"] == len(text) and p["truncated"] is True
    assert p["head"] == text[:TURN1_PROBE_HEAD_CHARS] == text[:400]
    short = turn1_output_probe("FINAL('a')")
    assert short["truncated"] is False and short["head"] == "FINAL('a')"
    assert turn1_output_probe(None)["chars"] == 0


def test_taskstate_defaults_and_diagnostic_row():
    st = TaskState()
    assert st.turn1_grammar_shape == "" and st.turn1_output_probe is None
    base = dict(question_id="q", suite="s", config="c", role="frontdoor", mode="repl",
                passed=True, answer="a", expected="a", scoring_method="exact",
                error=None, error_type=None, tokens_generated=1, elapsed_s=1.0,
                role_history=[], delegation_events=[], tools_used=0, tools_called=[])
    row = build_diagnostic(**base, grammar_enforced=True, turn1_grammar_shape="legacy",
                           turn1_output_probe=turn1_output_probe("FINAL('a')"))
    assert row["grammar_enforced"] is True and row["turn1_grammar_shape"] == "legacy"
    assert row["turn1_output"]["chars"] == 10
    old = build_diagnostic(**base)  # unchanged callers
    assert old["turn1_grammar_shape"] == "" and old["turn1_output"] is None


# ---- LB6.2 ---------------------------------------------------------------

def test_record_http_call_schema_and_fields(log_file):
    ok = sc.record_http_call(
        method="worker_pool_completion", role="worker_explore", base_url="http://127.0.0.1:8082",
        endpoint="/completion", ts_start=100.0, ts_end=101.5, prompt="hello", n_tokens=64,
        status=200, finish_reason="limit",
        timings={"prompt_n": 5, "predicted_n": 64, "cache_n": 3},
        tokens_generated=64, prompt_tokens=5,
    )
    assert ok
    (r,) = _recs(log_file)
    assert r["schema"] == sc.SCHEMA and r["method"] == "worker_pool_completion"
    assert r["finish_reason"] == "limit" and r["outcome"] == "ok"
    assert r["server"]["port"] == 8082 and r["notes"]["endpoint"] == "/completion"
    assert r["timings"]["cache_n"] == 3 and r["result"]["tokens_generated"] == 64
    assert r["request"]["n_tokens"] == 64 and r["record_sha256"]


def test_record_http_call_failure_and_disabled(log_file, monkeypatch):
    sc.record_http_call(method="m", role="r", base_url=None, endpoint="/e", ts_start=1, ts_end=2,
                        status=500)
    sc.record_http_call(method="m", role="r", base_url=None, endpoint="/e", ts_start=1, ts_end=2,
                        exc=TimeoutError("timeout"))
    a, b = _recs(log_file)
    assert a["outcome"] == "failed" and b["outcome"] == "timeout"
    monkeypatch.setenv(sc.LOG_ENV, "off")
    assert sc.record_http_call(method="m", role="r", base_url=None, endpoint="/e",
                               ts_start=1, ts_end=2) is False


def test_record_http_call_not_double_counted_inside_recorded_call(log_file):
    tok = sc._IN_CALL.set(True)
    try:
        assert sc.record_http_call(method="m", role="r", base_url=None, endpoint="/e",
                                   ts_start=1, ts_end=2) is False
    finally:
        sc._IN_CALL.reset(tok)
    assert _recs(log_file) == []


async def test_worker_pool_http_call_records_and_result_unchanged(log_file):
    from src.services.worker_pool import WorkerPoolManager

    pool = WorkerPoolManager.__new__(WorkerPoolManager)
    resp = AsyncMock()
    resp.status = 200
    resp.json = AsyncMock(return_value={
        "content": "hi", "stop_type": "eos", "tokens_predicted": 2, "tokens_evaluated": 7,
        "timings": {"prompt_n": 7, "predicted_n": 2, "cache_n": 0}})
    post = MagicMock()
    post.return_value.__aenter__ = AsyncMock(return_value=resp)
    post.return_value.__aexit__ = AsyncMock(return_value=None)
    pool._http_session = SimpleNamespace(post=post)
    worker = SimpleNamespace(url="http://127.0.0.1:8092", error_count=0,
                             config=SimpleNamespace(name="worker_explore", model_path="/m/x-Q4_K_M.gguf"))
    out = await pool._http_call(worker, "prompt", temperature=0.1, max_tokens=32)
    assert out == "hi"
    (r,) = _recs(log_file)
    assert r["method"] == "worker_pool_completion" and r["role"] == "worker_explore"
    assert r["finish_reason"] == "eos" and r["server"]["port"] == 8092
    assert r["result"]["tokens_generated"] == 2 and r["timings"]["prompt_n"] == 7

    resp.status = 500
    resp.text = AsyncMock(return_value="boom")
    with pytest.raises(RuntimeError, match="Worker returned 500"):
        await pool._http_call(worker, "prompt", temperature=0.1, max_tokens=32)
    r2 = _recs(log_file)[-1]
    assert r2["outcome"] == "exception" and r2["notes"]["http_status"] == 500


def test_repair_completer_records(log_file):
    from src.structured_output.repair import http_chat_completer

    payload = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}],
               "usage": {"prompt_tokens": 9, "completion_tokens": 4}}

    class _R:
        status = 200
        def read(self): return json.dumps(payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch("urllib.request.urlopen", return_value=_R()):
        out = http_chat_completer("http://127.0.0.1:8080/v1", max_tokens=16)(
            [{"role": "user", "content": "fix"}], {"type": "object"})
    assert out == "{}"
    (r,) = [x for x in _recs(log_file) if x["method"] == "structured_repair_chat"]
    assert r["finish_reason"] == "length" and r["result"]["tokens_generated"] == 4
    assert r["server"]["port"] == 8080
