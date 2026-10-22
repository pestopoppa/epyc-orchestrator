"""UFH14-LB2 durable parse events + UFH14-LB6 recorder health / stream-gap capture."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from src.backends import serving_calls as sc
from src.model_server import InferenceRequest, InferenceResult
from src.structured_output import parse_events as pe
from src.structured_output import repair as rp

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


@pytest.fixture
def logs(monkeypatch, tmp_path):
    calls = tmp_path / "serving_calls" / "serving_calls.jsonl"
    events = tmp_path / "parse_events" / "parse_events.jsonl"
    monkeypatch.setenv(sc.LOG_ENV, str(calls))
    monkeypatch.setenv(pe.LOG_ENV, str(events))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    sc.clear_staged()
    sc._LAST_CALL.set(None)
    rp.reset_counts_for_tests()
    return types.SimpleNamespace(calls=calls, events=events)


def _rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


class _Backend:
    def __init__(self, reason: str, tokens: int = 7):
        self.config = types.SimpleNamespace(base_url="http://localhost:8070")
        self.reason, self.tokens = reason, tokens

    @sc.recorded_call("infer")
    def infer(self, role_config, request):
        return InferenceResult(role="frontdoor", output="{", tokens_generated=self.tokens,
                               generation_speed=1.0, elapsed_time=1.0, success=True,
                               completion_reason=self.reason)


def _generate(reason, tokens=7, budget=7):
    _Backend(reason, tokens).infer(
        types.SimpleNamespace(name="frontdoor", model=None),
        InferenceRequest(role="frontdoor", prompt="x", n_tokens=budget),
    )


def test_serving_record_has_top_level_finish_reason(logs):
    _generate("length", 7, 7)
    [rec] = _rows(logs.calls)
    assert rec["finish_reason"] == "length"
    assert sc.last_call()["finish_reason"] == "length" and sc.last_call()["max_tokens"] == 7


def test_failed_parse_after_length_cut_is_classified_length_truncated(logs):
    _generate("length", 7, 7)
    res = rp.parse_with_repair('{"a": 1', schema=SCHEMA, site="t.site",
                               complete=lambda m, s: "not json")
    assert res.status == "failed"
    [ev] = _rows(logs.events)
    assert ev["kind"] == "structured_output" and ev["site"] == "t.site" and ev["status"] == "failed"
    assert ev["finish_reason"] == "length" and ev["n_tokens"] == 7 and ev["max_tokens"] == 7
    assert ev["finish_reason_source"] == "last_call_in_context"
    assert ev["truncated_by_length"] is True
    assert ev["record_sha256"] == sc.record_digest(ev)


def test_repair_turn_does_not_overwrite_the_parsed_generation(logs):
    """The generation's finish_reason is frozen at parse start, not read after the repair turn."""
    _generate("length", 7, 7)

    def complete(messages, schema):
        _generate("stop", 3, 2048)  # the repair turn is itself a recorded call
        pe.note_repair_turn_from_last_call(max_tokens=2048)
        return '{"a": 5}'

    res = rp.parse_with_repair('{"a": ', schema=SCHEMA, site="t.site", complete=complete)
    assert res.status == "repaired"
    ev = _rows(logs.events)[-1]
    assert ev["finish_reason"] == "length" and ev["truncated_by_length"] is True
    assert ev["repair_turn"]["finish_reason"] == "stop" and ev["repair_turn"]["max_tokens"] == 2048


def test_malformed_after_natural_stop_and_unknown_stop(logs):
    _generate("stop", 5, 100)
    rp.parse_with_repair("garbage", schema=SCHEMA, site="a", complete=lambda m, s: "x")
    sc._LAST_CALL.set(None)
    rp.parse_with_repair("garbage", schema=SCHEMA, site="b", complete=lambda m, s: "x")
    a, b = _rows(logs.events)
    assert a["truncated_by_length"] is False
    assert b["finish_reason"] is None and b["finish_reason_source"] == "absent"
    assert b["truncated_by_length"] is None  # unknown is not "malformed"
    table = pe.classify(pe.read_events(logs.events))
    assert table["structured_output:a"]["malformed"] == 1
    assert table["structured_output:b"]["unknown_stop"] == 1


def test_explicit_generation_context_wins(logs):
    _generate("stop", 5, 100)
    with pe.generation_context(finish_reason="length", n_tokens=100, max_tokens=100):
        rp.parse_with_repair("{", schema=SCHEMA, site="x", complete=lambda m, s: "x")
    [ev] = _rows(logs.events)
    assert ev["finish_reason_source"] == "explicit" and ev["truncated_by_length"] is True


def test_parsed_event_is_the_denominator(logs):
    rp.parse_with_repair('{"a": 1}', schema=SCHEMA, site="ok", complete=lambda m, s: "")
    [ev] = _rows(logs.events)
    assert ev["status"] == "parsed"
    assert pe.classify(pe.read_events(logs.events))["structured_output:ok"]["parsed"] == 1


def test_tool_call_repair_and_review_decision_emit_events(logs):
    from src.prompt_builders import code_utils
    from src.proactive_delegation.review_grammar import parse_review_decision

    _generate("length", 9, 9)
    code_utils._record_repair_outcome("unrecoverable", '{"name": "x"')
    obj, failure = parse_review_decision("no json here")
    assert obj is None and failure is not None
    kinds = {(e["kind"], e["status"]): e for e in _rows(logs.events)}
    tool = kinds[("tool_call_json", "unrecoverable")]
    assert tool["truncated_by_length"] is True and "raw" not in json.dumps(tool).replace("raw_bytes", "")
    assert kinds[("review_decision", "failed")]["reason"] == "no_json"


def test_disabled_log_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv(pe.LOG_ENV, "off")
    assert pe.record_parse_event("structured_output", "s", "failed") is False


def test_unwritable_log_never_raises(monkeypatch, tmp_path):
    bad = tmp_path / "f"
    bad.write_text("x")
    monkeypatch.setenv(pe.LOG_ENV, str(bad / "sub" / "e.jsonl"))
    assert pe.record_parse_event("structured_output", "s", "failed") is False


# ----------------------------------------------------------------- LB6


def test_stream_gap_stats_carry_their_sample_and_stalls():
    gaps = [10.0] * 98 + [900.0, 40.0]
    st = sc.stream_gap_stats(gaps)
    assert st["n_gaps"] == 100 and st["max_ms"] == 900.0 and st["stall_count"] == 1
    assert st["stall_idx"] == [98] and st["gaps_ms"] == gaps
    assert st["p999_ms"] == 900.0 and st["p999_min_samples_met"] is False
    assert st["p99_ms"] == 40.0  # nearest-rank: the 99th of 100 sorted gaps
    assert sc.stream_gap_stats([float(i) for i in range(1, 1001)])["p999_ms"] == 999.0
    assert sc.stream_gap_stats([]) is None


def test_stream_gap_list_is_capped_but_stats_use_every_gap():
    st = sc.stream_gap_stats([1.0] * (sc.GAP_LIST_CAP + 10))
    assert st["n_gaps"] == sc.GAP_LIST_CAP + 10 and len(st["gaps_ms"]) == sc.GAP_LIST_CAP
    assert st["gaps_truncated"] is True


def test_recorded_stream_call_carries_gaps_and_finish_reason(logs, monkeypatch):
    ticks = iter([0.0, 0.010, 0.020, 0.920, 0.930])
    monkeypatch.setattr(sc.time, "monotonic", lambda: next(ticks))

    class B:
        config = types.SimpleNamespace(base_url="http://localhost:8070")

        @sc.recorded_call("infer_stream_text")
        def infer_stream_text(self, role_config, request, on_chunk=None):
            for _ in range(5):
                on_chunk("t")
            return InferenceResult(role="frontdoor", output="ttttt", tokens_generated=5,
                                   generation_speed=1.0, elapsed_time=1.0, success=True,
                                   completion_reason="stop")

    B().infer_stream_text(types.SimpleNamespace(name="frontdoor", model=None),
                          InferenceRequest(role="frontdoor", prompt="x", n_tokens=16),
                          on_chunk=lambda c: None)
    [rec] = _rows(logs.calls)
    g = rec["notes"]["stream_gaps"]
    assert rec["finish_reason"] == "stop"
    assert g["n_gaps"] == 4 and g["stall_count"] == 1 and g["stall_idx"] == [2]
    assert g["max_ms"] == pytest.approx(900.0, abs=1.0)


def test_health_counts_skips_and_failures(monkeypatch, tmp_path):
    monkeypatch.setenv(sc.LOG_ENV, "off")
    before = sc.health()["skipped_disabled"]
    _generate("stop")
    h = sc.health()
    assert h["enabled"] is False and h["skipped_disabled"] == before + 1
    monkeypatch.setenv(sc.LOG_ENV, str(tmp_path / "f" / "x.jsonl"))
    (tmp_path / "f").write_text("not a dir")
    w = sc.health()["write_failed"]
    _generate("stop")
    assert sc.health()["write_failed"] == w + 1
