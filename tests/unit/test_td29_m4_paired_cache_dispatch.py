"""Model-free replay of the M4 original-kwargs dispatch boundary."""

import json

import pytest

from scripts.bench import td29_m4_paired_cache as bench
from src.typed_decisions import tool_args_integration
from src.typed_decisions.tool_args_pilot import build_cases, _parse_free_form_arguments


class _FreeFormOnly:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def llm_call(self, prompt, **kwargs):
        self.calls += 1
        return self.reply


def _run_original_dispatch(monkeypatch, tmp_path, raw):
    case = next(case for case in build_cases() if case.case_id == "ticket-01")
    # A single canned case reaches the normal run/dispatch path. No model calls.
    monkeypatch.setattr(bench, "build_cases", lambda: [case] * 18)
    primitives = _FreeFormOnly(raw)
    monkeypatch.setattr(bench, "_live_primitives", lambda urls: primitives)
    monkeypatch.setattr(bench, "_prefix_record", lambda url, calls: None)
    invoked = []
    shadow = []
    original_invoke = bench._Registry.invoke

    def observed_invoke(self, tool_name, role, **kwargs):
        arguments = {
            key: value
            for key, value in kwargs.items()
            if key not in {"caller_type", "chain_id", "chain_index", "context"}
        }
        invoked.append(arguments)
        return original_invoke(self, tool_name, role, **kwargs)

    monkeypatch.setattr(bench._Registry, "invoke", observed_invoke)
    # Keep the actual ContextMixin foreground and shadow-submit boundary, but
    # capture submission rather than starting a selector/model worker.
    monkeypatch.setattr(
        tool_args_integration,
        "submit_tool_argument_shadow",
        lambda **kwargs: shadow.append(kwargs) or True,
    )
    output = tmp_path / "pairs.jsonl"
    result = bench.run(
        url="http://unused.invalid", output=output, role="frontdoor", mode="json", limit=1
    )
    row = json.loads(output.read_text())
    assert primitives.calls == 1
    assert result["completed"] == 1
    assert row["pair_valid"] is False
    return case, row, invoked, shadow


def test_schema_invalid_original_reaches_actual_dispatch_unchanged(monkeypatch, tmp_path):
    original = {
        "severity": "p2",
        "escalate": False,
        "effort_points": 5,
        "categories": ["data"],
        "assignee": None,
    }
    case, row, invoked, shadow = _run_original_dispatch(monkeypatch, tmp_path, json.dumps(original))
    # General pilot policy stays strict and still returns no validated args.
    assert _parse_free_form_arguments(json.dumps(original), case.parameters)[0] is None
    assert invoked == [original]
    assert shadow[0]["original_kwargs"] == original
    assert "schema violation at $.assignee" in row["tool_error"]
    assert "assignee must be string" in row["dispatch_error"]
    assert "Missing required parameter" not in row["dispatch_error"]
    assert shadow[0]["outcome"]["status"] == "error"


@pytest.mark.parametrize("raw", ["no JSON", '{"severity":', "null", '["p1"]'])
def test_unparseable_or_nonobject_reply_does_not_fabricate_dispatch(monkeypatch, tmp_path, raw):
    _, row, invoked, shadow = _run_original_dispatch(monkeypatch, tmp_path, raw)
    assert invoked == []
    assert shadow == []
    assert row["tool_error"]
    assert row["dispatch_error"] == "not_dispatched: free-form arguments unavailable"


def test_original_empty_object_is_dispatched_and_remains_invalid(monkeypatch, tmp_path):
    _, row, invoked, shadow = _run_original_dispatch(monkeypatch, tmp_path, "{}")
    assert invoked == [{}]
    assert shadow[0]["original_kwargs"] == {}
    assert row["tool_error"]
    assert "Missing required parameter" in row["dispatch_error"]


def test_valid_original_is_dispatched_without_schema_failure(monkeypatch, tmp_path):
    original = {
        "severity": "p2",
        "escalate": False,
        "effort_points": 5,
        "categories": ["data"],
        "assignee": "alex",
    }
    _, row, invoked, shadow = _run_original_dispatch(monkeypatch, tmp_path, json.dumps(original))
    assert invoked == [original]
    assert shadow[0]["original_kwargs"] == original
    assert row["tool_error"] is None
    assert row["dispatch_error"] is None
