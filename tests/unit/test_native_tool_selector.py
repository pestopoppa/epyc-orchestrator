"""codemode-ab CM-1: per-request tool selection on the DIRECT native tool-call path; default OFF."""

from __future__ import annotations

import pytest

from src.prompt_builders import tool_selector as ts


def _oa(name, desc):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}


TOOLS = [_oa("syn_sku_price", "Look up unit price for a product SKU"), _oa("syn_calendar_create", "Create a calendar event"),
         _oa("read_file", "Read a file from disk"), _oa("syn_weather", "Weather forecast for a city"),
         _oa("syn_fx", "Currency exchange rate"), _oa("syn_mail", "Send an email message")]
MSG = [{"role": "user", "content": "what is the unit price for this SKU"}]
ON = {ts.ENV_FLAG: "embed", ts.ENV_K: "2", "ORCHESTRATOR_TOOL_SELECTOR_LOG": ""}
SERVER = {"list_tools", "search_tools", "describe_tool"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_SELECTOR_LOG", "")
    ts.set_native_selector_provider(None)
    ts.LAST_NATIVE_SELECTION.clear()
    from src.prompt_builders import tool_picker as tp

    tp.BREAKER.success()
    yield
    ts.set_native_selector_provider(None)


def names(ts_list):
    return [t["function"]["name"] for t in ts_list]


def test_default_off_returns_same_object():
    assert ts.select_native_tools(TOOLS, MSG, None, {}) is TOOLS
    assert ts.select_native_tools(TOOLS, MSG, None, {ts.ENV_FLAG: "bogus"}) is TOOLS


def test_on_without_provider_fails_open_to_the_full_list_and_records_lexical():
    """Flag on, nothing registered: the lexical ranker is low-confidence, so a list of <= 40 tools is sent whole (never worse than off)."""
    out = ts.select_native_tools(TOOLS, MSG, "auto", ON)
    assert out is TOOLS
    assert ts.LAST_NATIVE_SELECTION["fallback"] == "full" and ts.LAST_NATIVE_SELECTION["mode"] == "full"


def test_registered_provider_is_used_frozen_per_conversation_and_adds_the_server_tools():
    calls = []
    ts.set_native_selector_provider(lambda q, d, k: calls.append(q) or ["syn_sku_price"])
    out = names(ts.select_native_tools(TOOLS, MSG, None, ON))
    assert out[:3] == ["list_tools", "search_tools", "describe_tool"]
    assert out[3:] == ["syn_sku_price", "read_file"]  # client order; core read_file pinned outside k
    names(ts.select_native_tools(TOOLS, MSG, None, ON))
    assert len(calls) == 1  # frozen: stable across turns, no second pick


def test_forced_choice_and_history_tools_are_added_not_reshuffled():
    ts.set_native_selector_provider(lambda q, d, k: ["syn_sku_price"])
    first = names(ts.select_native_tools(TOOLS, MSG, None, ON))
    msgs = MSG + [{"role": "assistant", "tool_calls": [{"function": {"name": "syn_fx", "arguments": "{}"}}]}]
    out = names(ts.select_native_tools(TOOLS, msgs, {"type": "function", "function": {"name": "syn_mail"}}, ON))
    assert out[: len(first)] == first  # the frozen prefix is untouched
    assert set(out[len(first):]) == {"syn_fx", "syn_mail"}


@pytest.mark.parametrize("tools,msgs,choice", [(TOOLS[:2], MSG, None), (TOOLS, [], None), (TOOLS, MSG, "none"), (None, MSG, None)])
def test_passthrough_cases(tools, msgs, choice):
    assert ts.select_native_tools(tools, msgs, choice, ON) is tools


def test_provider_failure_fails_open():
    def boom(q, d, k):
        raise RuntimeError("x")

    ts.set_native_selector_provider(boom)
    assert ts.select_native_tools(TOOLS, MSG, None, ON) is TOOLS


def test_oneshot_provider_groups_per_tool():
    class P:
        usage_total = {}

        def llm_call(self, *a, **k):
            import json

            names = [t["function"]["name"] for t in TOOLS]
            return json.dumps({"answers": {f"needs_{n}": {"noul": n == "syn_weather"} for n in names}})

    prov = ts.make_native_provider("oneshot", primitives=P())
    # the decision model flags only syn_weather; lexical ranking alone would never pick it for this query
    assert prov("unit price for sku", [ts.openai_tool_to_dict(t) for t in TOOLS], 2) == ["syn_weather"]


def test_chat_path_wiring(monkeypatch):
    from types import SimpleNamespace

    from src.api.routes import openai_compat as oc

    seen = {}

    class Prim:
        def chat_completion_call(self, messages, **kw):
            seen.update(kw)
            return {"content": "ok", "usage": {}}

    monkeypatch.setenv(ts.ENV_FLAG, "embed")
    monkeypatch.setenv(ts.ENV_K, "2")
    ts.set_native_selector_provider(lambda q, d, k: ["syn_sku_price"])
    req = SimpleNamespace(tools=TOOLS, tool_choice="auto", max_tokens=8)
    oc._run_client_tool_completion(Prim(), req, MSG, role="worker", sampling_kwargs={})
    assert names(seen["tools"]) == ["list_tools", "search_tools", "describe_tool", "syn_sku_price", "read_file"]
    monkeypatch.delenv(ts.ENV_FLAG)
    oc._run_client_tool_completion(Prim(), req, MSG, role="worker", sampling_kwargs={})
    assert seen["tools"] is TOOLS
