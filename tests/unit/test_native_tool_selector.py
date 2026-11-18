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
ON = {ts.ENV_FLAG: "embed", ts.ENV_K: "2"}


@pytest.fixture(autouse=True)
def _clean():
    ts.set_native_selector_provider(None)
    ts.LAST_NATIVE_SELECTION.clear()
    yield
    ts.set_native_selector_provider(None)


def names(ts_list):
    return [t["function"]["name"] for t in ts_list]


def test_default_off_returns_same_object():
    assert ts.select_native_tools(TOOLS, MSG, None, {}) is TOOLS
    assert ts.select_native_tools(TOOLS, MSG, None, {ts.ENV_FLAG: "bogus"}) is TOOLS


def test_on_narrows_to_topk_keeping_order_and_records_fallback():
    out = ts.select_native_tools(TOOLS, MSG, "auto", ON)
    assert len(out) == 2 and "syn_sku_price" in names(out)
    assert names(out) == [n for n in names(TOOLS) if n in names(out)]
    assert ts.LAST_NATIVE_SELECTION["fallback"] == "lexical:no_provider"


def test_registered_provider_is_used_and_cached():
    calls = []
    ts.set_native_selector_provider(lambda q, d, k: calls.append(q) or ["read_file"])
    assert names(ts.select_native_tools(TOOLS, MSG, None, ON)) == ["read_file"]
    ts.select_native_tools(TOOLS, MSG, None, ON)
    assert len(calls) == 1  # stable across turns: no second decision call


def test_forced_choice_and_history_tools_stay():
    ts.set_native_selector_provider(lambda q, d, k: ["read_file"])
    msgs = MSG + [{"role": "assistant", "tool_calls": [{"function": {"name": "syn_fx", "arguments": "{}"}}]}]
    out = ts.select_native_tools(TOOLS, msgs, {"type": "function", "function": {"name": "syn_mail"}}, ON)
    assert set(names(out)) == {"read_file", "syn_fx", "syn_mail"}


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
    req = SimpleNamespace(tools=TOOLS, tool_choice="auto", max_tokens=8)
    oc._run_client_tool_completion(Prim(), req, MSG, role="worker", sampling_kwargs={})
    assert len(seen["tools"]) == 2
    monkeypatch.delenv(ts.ENV_FLAG)
    oc._run_client_tool_completion(Prim(), req, MSG, role="worker", sampling_kwargs={})
    assert seen["tools"] is TOOLS
