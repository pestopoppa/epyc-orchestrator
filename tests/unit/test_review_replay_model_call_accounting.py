"""Regression: mirrored decision events are not additional model calls."""
import importlib.util
import json
import urllib.error
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "scripts/review/review_replay_50.py"
spec = importlib.util.spec_from_file_location("rd12_replay_accounting", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_model_calls_once_and_server_usage_is_not_estimate():
    questions = [{"latency_ms": 8.0, "model_calls_observed": 1, "model_call_attempts": 1},
                 {"latency_ms": 10.0, "model_calls_observed": 1, "model_call_attempts": 1}]
    usage = [{"prompt_tokens": 13, "completion_tokens": 4},
             {"prompt_tokens": 17, "completion_tokens": 5}]
    result = mod.summarize_model_calls(questions, usage)
    assert result["n_calls"] == 2
    assert result["latency_ms_total"] == 18
    assert result["endpoint_prompt_tokens"] == 30
    assert result["endpoint_completion_tokens"] == 9


def test_ambiguous_or_missing_usage_fails_closed():
    q = [{"latency_ms": 8.0, "model_calls_observed": 1, "model_call_attempts": 2}]
    assert mod.summarize_model_calls(q, [{"prompt_tokens": 5}])["status"] == "unavailable"
    q[0]["model_call_attempts"] = 1
    assert mod.summarize_model_calls(q, [{"prompt_tokens": None}])["endpoint_prompt_tokens"] is None


def test_legacy_delta_does_not_repeat_event_totals(tmp_path):
    on = tmp_path / "on.json"; off = tmp_path / "off.json"; out = tmp_path / "out.json"
    on.write_text(json.dumps({"decision_metrics": {"latency_ms": {"mean": 8}, "tokens_in": {"sum": 200}}}))
    off.write_text(json.dumps({"loop_ms_mean": 1}))
    result = mod.run_delta(on, off, out)
    assert result["reviewer_on_mean_latency_ms"] is None
    assert result["tokens_in_total"] is None


def test_attempts_include_failed_transport(monkeypatch):
    def fail(*_args, **_kwargs):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(mod.urllib.request, "urlopen", fail)
    primitives = mod.LiveServerPrimitives("http://127.0.0.1:1/v1", "fixture")
    try:
        primitives.llm_call("hello")
    except RuntimeError:
        pass
    else:
        raise AssertionError("transport failure was swallowed")
    assert primitives.call_attempts == 1
    assert primitives.usage_log == []
