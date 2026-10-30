"""Actual recording decorator writes both log files, with no raw prompt."""

import json
from types import SimpleNamespace
import pytest
from src.backends import prompt_layout, serving_calls, dispatch_log
from src.model_server import InferenceRequest, InferenceResult


class NativeBoundary:
    config = SimpleNamespace(base_url="http://127.0.0.1:25281")

    @serving_calls.recorded_call("infer")
    def infer(self, role, request):
        serving_calls.note_timings(
            {"cache_n": 7, "prompt_n": 12}, endpoint="/completion"
        )
        return InferenceResult(
            role="frontdoor",
            output="4",
            tokens_generated=1,
            generation_speed=1,
            elapsed_time=1,
            success=True,
            prompt_tokens=19,
        )


def run_call(monkeypatch, tmp_path, *, layout="on", logs="on", fleet=False):
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv(
        serving_calls.LOG_ENV,
        str(tmp_path / "serving.jsonl") if logs == "on" else "off",
    )
    monkeypatch.setenv(
        dispatch_log.LOG_ENV,
        str(tmp_path / "dispatch.jsonl") if logs == "on" else "off",
    )
    monkeypatch.setenv(prompt_layout.LOG_ENV, layout)
    serving_calls.clear_staged()
    prompt_layout.clear_layout()
    private = "SECRET-CONTEXT-MUST-NOT-LEAK"
    prompt = private + "\nWhat is two plus two?"
    with prompt_layout.source_scope("benchmark"):
        prompt_layout.record_prompt(
            prompt,
            named=[("context", private), ("task", "What is two plus two?")],
            turn_index=0,
        )
        serving_calls.stage_caller(
            role="frontdoor",
            request_id="rte-real-decorator",
            **({"instance_idx": 0} if fleet else {}),
        )
        result = NativeBoundary().infer(
            SimpleNamespace(name="frontdoor"),
            InferenceRequest(role="frontdoor", prompt=prompt, n_tokens=8),
        )
    assert result.output == "4"
    return private


def test_decorator_direct_writes_same_real_layout_and_native_counts(
    monkeypatch, tmp_path
):
    private = run_call(monkeypatch, tmp_path)
    serving = json.loads((tmp_path / "serving.jsonl").read_text().splitlines()[-1])
    records = [
        json.loads(s) for s in (tmp_path / "dispatch.jsonl").read_text().splitlines()
    ]
    dispatch = next(r for r in records if r["kind"] == "dispatch")
    assert dispatch["blocks"] == serving["blocks"] and dispatch["blocks"]
    assert dispatch["source"] == serving["source"] == "benchmark"
    assert dispatch["prompt_tokens"] == serving["result"]["prompt_tokens"] == 19
    assert (
        dispatch["cached_prompt_tokens"]
        == serving["result"]["cached_prompt_tokens"]
        == 7
    )
    assert dispatch["turn_index"] == 0 and dispatch["layout_chars"] > 0
    for path in tmp_path.glob("*.jsonl"):
        assert (
            private not in path.read_text()
            and "What is two plus two?" not in path.read_text()
        )
    assert (
        prompt_layout.fields_for_request(
            InferenceRequest(role="frontdoor", prompt=private, n_tokens=8)
        )
        == {}
    )


@pytest.mark.parametrize("layout,logs", [("off", "on"), ("on", "off")])
def test_kill_switches(monkeypatch, tmp_path, layout, logs):
    run_call(monkeypatch, tmp_path, layout=layout, logs=logs)
    if logs == "off":
        assert not list(tmp_path.glob("*.jsonl"))
    else:
        for path in tmp_path.glob("*.jsonl"):
            assert all(
                "blocks" not in json.loads(s) for s in path.read_text().splitlines()
            )


def test_no_fleet_duplicate(monkeypatch, tmp_path):
    run_call(monkeypatch, tmp_path, fleet=True)
    assert (tmp_path / "serving.jsonl").exists()
    assert not (tmp_path / "dispatch.jsonl").exists()


def test_standalone_whitelist_and_absent_counts(monkeypatch, tmp_path):
    monkeypatch.setenv(dispatch_log.LOG_ENV, str(tmp_path / "dispatch.jsonl"))
    monkeypatch.setenv(prompt_layout.LOG_ENV, "on")
    dispatch_log.record_standalone(
        lane="direct",
        role="frontdoor",
        base_url=None,
        port=None,
        ts_arrival=1,
        ts_dispatch=2,
        ts_end=3,
        layout={"source": "benchmark", "prompt": "SECRET", "provenance": "SECRET"},
    )
    record = json.loads((tmp_path / "dispatch.jsonl").read_text().splitlines()[-1])
    assert (
        record["source"] == "benchmark"
        and record["prompt_tokens"] is None
        and record["cached_prompt_tokens"] is None
    )
    assert "SECRET" not in (tmp_path / "dispatch.jsonl").read_text()
