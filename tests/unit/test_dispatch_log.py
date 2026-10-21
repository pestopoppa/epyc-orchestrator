"""WP-15.1 / HSF-3.1a: the fleet dispatch log is additive, versioned and self-checking."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.backends import dispatch_log, serving_calls


def _stub(url, result=None, raises=None):
    def infer(role_config, request):
        if raises:
            raise raises
        return result

    def infer_streaming(role_config, request):
        return "handle"

    return SimpleNamespace(
        config=SimpleNamespace(base_url=url),
        infer=infer,
        infer_streaming=infer_streaming,
        infer_stream_text=lambda rc, rq, on_chunk=None: result,
    )


def _make(monkeypatch, tmp_path, result=None, raises=None, log="on"):
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "0")
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", "off")
    path = tmp_path / "dispatch.jsonl"
    monkeypatch.setenv(dispatch_log.LOG_ENV, str(path) if log == "on" else "off")
    import src.backends.concurrency_aware as CA

    cab = CA.ConcurrencyAwareBackend(
        _stub("http://localhost:8070", result, raises),
        [_stub(f"http://localhost:{p}", result, raises) for p in (8080, 8180)],
        role="frontdoor",
        full_port=8070,
    )
    return cab, path


def _recs(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _req(**kw):
    base = dict(prompt="hello world", n_tokens=64, role="frontdoor", session_id="", task_id="t1")
    base.update(kw)
    return SimpleNamespace(**base)


def test_record_fields_config_echo_and_hash(monkeypatch, tmp_path):
    res = SimpleNamespace(success=True)
    cab, path = _make(monkeypatch, tmp_path, result=res)
    serving_calls.stage_caller(
        request_id="rid-1", session_id="sess-A", client_class="opencode",
        client_class_provenance="caller_supplied:x_client_class", workload_class="interactive",
    )
    assert cab.infer(SimpleNamespace(name="frontdoor"), _req()) is res
    cfg, rec = _recs(path)
    assert cfg["kind"] == "config" and cfg["schema"] == dispatch_log.SCHEMA
    assert cfg["config"]["full_port"] == 8070 and cfg["config"]["quarter_urls"]
    assert rec["kind"] == "dispatch" and rec["config_sha256"] == cfg["config_sha256"]
    assert rec["request_id"] == "rid-1"
    assert rec["session"]["key"] == dispatch_log.session_key("sess-A")
    assert "sess-A" not in json.dumps(rec)
    assert rec["client"]["class"] == "opencode"
    assert rec["instance"]["port"] == 8070 and rec["instance"]["is_full"] is True
    assert rec["request"] == {"prompt_chars": 11, "n_tokens_budget": 64}
    assert rec["fleet_at_arrival"]["waiting"] == 1
    assert rec["fleet_at_dispatch"]["instances"]["8070"]["requests"] == 1
    assert rec["queue_wait_ms"] >= 0 and rec["outcome"] == "ok"
    assert rec["record_sha256"] == serving_calls.record_digest(rec)
    serving_calls.clear_staged()


def test_session_gap_is_derivable_across_records(monkeypatch, tmp_path):
    cab, path = _make(monkeypatch, tmp_path, result=SimpleNamespace(success=True))
    for _ in range(2):
        serving_calls.stage_caller(session_id="sess-A", client_class="x")
        cab.infer(SimpleNamespace(name="frontdoor"), _req())
    r1, r2 = [r for r in _recs(path) if r["kind"] == "dispatch"]
    assert r1["session"]["key"] == r2["session"]["key"]
    assert r2["ts_arrival"] >= r1["ts_end"]  # gap = ts_arrival(next) - ts_end(prev) >= 0


def test_exception_passes_through_unchanged_and_is_recorded(monkeypatch, tmp_path):
    cab, path = _make(monkeypatch, tmp_path, raises=ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        cab.infer(SimpleNamespace(name="frontdoor"), _req())
    rec = [r for r in _recs(path) if r["kind"] == "dispatch"][0]
    assert rec["outcome"] == "exception" and rec["error_type"] == "ValueError"
    assert rec["fleet_at_arrival"]["waiting"] == 1
    assert cab._dispatch_log._snapshot() == {"waiting": 0, "instances": {}}


def test_streaming_handle_is_marked_unobserved(monkeypatch, tmp_path):
    cab, path = _make(monkeypatch, tmp_path)
    assert cab.infer_streaming(SimpleNamespace(name="frontdoor"), _req()) == "handle"
    rec = [r for r in _recs(path) if r["kind"] == "dispatch"][0]
    assert rec["completion_observed"] is False and rec["outcome"] == "handed_off"


def test_disabled_writes_nothing_and_behaviour_unchanged(monkeypatch, tmp_path):
    res = SimpleNamespace(success=True)
    cab, path = _make(monkeypatch, tmp_path, result=res, log="off")
    assert cab.infer(SimpleNamespace(name="frontdoor"), _req()) is res
    assert not path.exists()


def test_follows_serving_calls_kill_switch_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(dispatch_log.LOG_ENV, raising=False)
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", "off")
    assert dispatch_log.log_path() is None


def test_write_failure_is_swallowed(monkeypatch, tmp_path):
    res = SimpleNamespace(success=True)
    cab, path = _make(monkeypatch, tmp_path, result=res)
    monkeypatch.setattr(dispatch_log, "write_record", lambda r: (_ for _ in ()).throw(OSError("disk")))
    assert cab.infer(SimpleNamespace(name="frontdoor"), _req()) is res
