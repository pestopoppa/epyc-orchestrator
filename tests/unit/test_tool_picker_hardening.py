"""SELECTOR_HARDENING: core pin, adaptive k + family cap, ladder, budget/breaker, memoised selector, frozen picks, recovery, log,
C2 shadow, wiring. Everything is behind ORCHESTRATOR_TOOL_SELECTOR (default off)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from src.prompt_builders import tool_picker as tp
from src.prompt_builders import tool_selector as ts

CFG = tp.PickerConfig(k_min=2, k_max=4, delta=0.05, family_cap=2, full_list_max=5, budget_ms=300)


def _oa(name, desc="d"):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}


def many(n=20, extra=()):
    base = ["read_file", "write_file", "list_dir", "syn_sku_price"] + [f"syn_t{i}_x{i}" for i in range(n)] + list(extra)
    return [_oa(x, f"tool {x}") for x in base]


MSG = [{"role": "user", "content": "price of this sku"}]


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv(tp.ENV_LOG, str(tmp_path / "events.jsonl"))
    ON[tp.ENV_LOG] = str(tmp_path / "events.jsonl")  # selectors given an explicit environ log there too
    ts.set_native_selector_provider(None)
    tp.set_shadow_primitives(None)
    tp.BREAKER.success()
    tp.BREAKER._open_until = 0.0
    yield
    ts.set_native_selector_provider(None)
    tp.set_shadow_primitives(None)


def rows(tmp_path):
    p = tmp_path / "events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def provider(scores):
    p = lambda q, d, k: None  # noqa: E731
    p.scorer = lambda q, d: ({n: scores.get(n, 0.0) for n in (x["name"] for x in d)}, None, "fake-emb")
    return p


def names(tools):
    return [t["function"]["name"] for t in tools]


ON = {ts.ENV_FLAG: "embed", ts.ENV_K: "8"}


# (c) core pinned outside k ----------------------------------------------------------------------------------------
def test_core_names_and_vocabulary():
    assert tp.core_names(["syn_a", "read_file", "list_dir", "write_file"]) == ["read_file", "write_file", "list_dir"]
    assert tp.core_names(["read", "write", "bash", "grep"]) == ["read", "write"]
    assert tp.core_names(["a", "b"], {tp.ENV_CORE: "b"}) == ["b"]


def test_core_never_counts_against_k_and_is_always_sent():
    scores = {"syn_sku_price": 0.9}  # everything else 0: core scores are the worst, yet core is still sent
    ts.set_native_selector_provider(provider(scores))
    out = names(ts.select_native_tools(many(), MSG, None, {**ON, tp.ENV_K_MIN: "1", tp.ENV_K_MAX: "1"}))
    assert {"read_file", "write_file", "list_dir", "syn_sku_price"} <= set(out)
    assert out[:3] == ["list_tools", "search_tools", "describe_tool"]


# (d) adaptive k + family cap ----------------------------------------------------------------------------------------
def test_sharp_top_uses_k_min_and_broad_plateau_widens():
    order = [f"t{i}_a_{i}" for i in range(10)]
    sharp = {n: 1.0 if i == 0 else 0.1 for i, n in enumerate(order)}
    picks, k, plateau = tp.adaptive_pick(sharp, order, [], CFG)
    assert (k, plateau, len(picks)) == (2, 1, 2)
    broad = {n: 1.0 - 0.01 * i for i, n in enumerate(order)}
    picks, k, plateau = tp.adaptive_pick(broad, order, [], CFG)
    assert (k, plateau) == (4, 6)


def test_family_cap_stops_crowding():
    order = ["syn_sku_a", "syn_sku_b", "syn_sku_c", "syn_sku_d", "calc_one"]
    scores = {"syn_sku_a": 1.0, "syn_sku_b": 0.99, "syn_sku_c": 0.98, "syn_sku_d": 0.97, "calc_one": 0.5}
    picks, _k, _p = tp.adaptive_pick(scores, order, [], tp.PickerConfig(k_min=4, k_max=4, delta=0.0, family_cap=2))
    assert picks == ["syn_sku_a", "syn_sku_b", "calc_one"]
    assert tp.family("syn_sku_price") == "syn_sku" and tp.family("read_file") == "read"


# (e) ladder ---------------------------------------------------------------------------------------------------------------
def test_ladder_flat_profile_full_list_when_small_else_core_search():
    flat = {f"t{i}_a_{i}": 1.0 for i in range(8)}
    order = list(flat)
    assert tp.decide(flat, order, [], CFG, n_tools=8).mode == "core_search"  # 8 tools > full_list_max=5, plateau 8 > k_max=4
    assert tp.decide(flat, order, [], CFG, n_tools=5).mode == "full"
    full = tp.decide(flat, order, [], CFG, n_tools=3)
    assert (full.mode, full.rung, full.detail["low_confidence"]) == ("full", "full", "flat")
    sharp = {**{n: 0.1 for n in order}, order[0]: 1.0}
    assert (tp.decide(sharp, order, [], CFG).mode, tp.decide(sharp, order, [], CFG).rung) == ("narrow", "none")
    wide = {n: 1.0 - 0.02 * i for i, n in enumerate(order)}
    assert tp.decide(wide, order, [], CFG).rung == "widen"
    assert tp.decide(sharp, order, [], CFG, lexical=True, n_tools=3).mode == "full"  # embedder down = low confidence


def test_flat_conversation_through_select_returns_unpicked_list():
    ts.set_native_selector_provider(provider({}))  # all zeros: a flat profile on 24 tools (<= 40) -> full list
    tools = many()
    assert ts.select_native_tools(tools, MSG, None, ON) is tools


# (a) latency cap + breaker --------------------------------------------------------------------------------------------------
def test_slow_scorer_times_out_and_fails_open_then_breaker_opens():
    def slow(q, d):
        time.sleep(0.3)
        return {}, None, ""

    t = [0.0]
    brk = tp.Breaker(threshold=3, cooldown_s=60, clock=lambda: t[0])
    dicts = [ts.openai_tool_to_dict(x) for x in many()]
    cfg = tp.PickerConfig(budget_ms=20)
    for _ in range(3):
        r = tp.pick("q", dicts, slow, cfg, breaker=brk)
        assert (r.mode, r.rung) == ("full", "timeout")
    assert brk.is_open() and brk.opened == 1
    assert tp.pick("q", dicts, slow, cfg, breaker=brk).rung == "breaker_open"
    t[0] = 61.0
    assert not brk.is_open()


def test_scorer_error_fails_open_and_success_resets_the_count():
    brk = tp.Breaker(threshold=2)
    dicts = [ts.openai_tool_to_dict(x) for x in many()]

    def boom(q, d):
        raise RuntimeError("x")

    assert tp.pick("q", dicts, boom, CFG, breaker=brk).rung == "error"
    ok = tp.pick("q", dicts, lambda q, d: ({"syn_sku_price": 1.0}, None, "e"), CFG, breaker=brk)
    assert ok.mode == "narrow" and not brk.is_open()
    assert tp.pick("q", dicts, boom, CFG, breaker=brk).rung == "error" and not brk.is_open()  # count restarted


# (b) memoised selector -------------------------------------------------------------------------------------------------------
def test_selector_memoised_per_registry_version():
    np = pytest.importorskip("numpy")
    from src.repl_environment.context_search import DenseResult

    class Fake:
        model_id = "fake"
        calls = 0

        def embed(self, texts, *, timeout_s=0.0, admission_wait_s=0.0):
            Fake.calls += 1
            return DenseResult(np.array([[1.0 if "price" in t else 0.0, 0.5, 0.1] for t in texts], dtype="float32"), "fake")

    tp.clear_selectors()
    tp.SELECTOR_BUILDS["n"] = 0
    dicts = [ts.openai_tool_to_dict(x) for x in many()]
    prov = ts.make_native_provider("embed", embedder=Fake())
    for _ in range(3):
        prov("price of sku", dicts, 4)
    assert tp.SELECTOR_BUILDS["n"] == 1 and Fake.calls == 1 + 3  # documents embedded once; one query embed per call
    dicts[3] = {**dicts[3], "description": "changed"}
    prov("price of sku", dicts, 4)
    assert tp.SELECTOR_BUILDS["n"] == 2  # a new registry version builds a new selector


# freeze ---------------------------------------------------------------------------------------------------------------------
def test_pick_is_frozen_across_turns_and_only_grows():
    calls = []
    p = provider({"syn_sku_price": 1.0})
    scorer = p.scorer
    p.scorer = lambda q, d: calls.append(q) or scorer(q, d)
    ts.set_native_selector_provider(p)
    tools = many()
    t1 = names(ts.select_native_tools(tools, MSG, None, ON))
    follow = MSG + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "now do something entirely different"}]
    t2 = names(ts.select_native_tools(tools, follow, None, ON))
    assert t1 == t2 and len(calls) == 1  # selection ran on the first user turn only
    hist = follow + [{"role": "assistant", "tool_calls": [{"function": {"name": "syn_t5_x5", "arguments": "{}"}}]}]
    t3 = names(ts.select_native_tools(tools, hist, None, ON))
    assert t3[: len(t1)] == t1 and t3[len(t1):] == ["syn_t5_x5"]  # appended, never reshuffled


# log ------------------------------------------------------------------------------------------------------------------------------
def test_selection_row_logs_scores_k_rung_latency_and_outcome_called_not_selected(tmp_path):
    ts.set_native_selector_provider(provider({"syn_sku_price": 0.9, "syn_t1_x1": 0.2}))
    tools = many()
    narrowed, ctx = ts.select_native_tools_ex(tools, MSG, None, ON, scope="frontdoor")
    assert ctx is not None
    sel = [r for r in rows(tmp_path) if r["type"] == "selection"][0]
    assert sel["rung"] == "none" and sel["k_effective"] == 4 and sel["role"] == "frontdoor" and sel["embedder"] == "fake-emb"
    assert sel["scores_top16"][0][0] == "syn_sku_price" and sel["elapsed_ms"] >= 0 and "read_file" in sel["core"]
    tp.record_outcome(ctx, ["syn_sku_price", "syn_t9_x9", "list_tools"])
    out = [r for r in rows(tmp_path) if r["type"] == "outcome"][0]
    assert out["called_not_selected"] == ["syn_t9_x9"] and out["called"] == ["syn_sku_price", "syn_t9_x9"]
    assert out["proxy_recall"] == 0.5
    assert "syn_t9_x9" in tp.frozen_get(ctx.key).names  # added for the next turn


# (a) C2 shadow --------------------------------------------------------------------------------------------------------------------------
def test_shadow_is_off_by_default_and_never_blocks_or_applies(tmp_path):
    dicts = [ts.openai_tool_to_dict(x) for x in many()]
    assert tp.maybe_shadow_rerank("r", "q", dicts, [], 4, {}) is False
    gate = {"called": 0}

    class Slow:
        usage_total = {}

        def llm_call(self, *a, **k):
            gate["called"] += 1
            time.sleep(0.5)
            return json.dumps({"answers": {}})

    tp.set_shadow_primitives(Slow())
    env = {**ON, tp.ENV_SHADOW: "oneshot"}
    ts.set_native_selector_provider(provider({"syn_sku_price": 1.0}))
    t0 = time.perf_counter()
    out = names(ts.select_native_tools(many(), MSG, None, env))
    assert time.perf_counter() - t0 < 0.3  # the slow decision model is off the critical path
    assert "syn_sku_price" in out


def test_oneshot_native_provider_is_shadow_only_and_never_picks():
    class P:
        usage_total = {}

        def llm_call(self, *a, **k):
            raise AssertionError("C2 must not run on the critical path")

    ts.set_native_selector_provider(ts.make_native_provider("oneshot", primitives=P()))
    tools = many()
    assert ts.select_native_tools(tools, MSG, None, ON) is tools  # no picker scorer -> lexical -> low confidence -> full


# server tools + recovery (f)(g) ------------------------------------------------------------------------------------------------------
def _run(prim, tools, msgs=MSG, env=None):
    from src.api.routes import openai_compat as oc

    req = SimpleNamespace(tools=tools, tool_choice="auto", max_tokens=8)
    return oc._run_client_tool_completion(prim, req, list(msgs), role="frontdoor", sampling_kwargs={})


def _call(name, args=None, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args or {})}}


class Prim:
    def __init__(self, script):
        self.script, self.seen = list(script), []

    def chat_completion_call(self, messages, **kw):
        self.seen.append((list(messages), kw["tools"]))
        step = self.script.pop(0)
        return {"content": "", "tool_calls": step, "usage": {"prompt_tokens": 10, "completion_tokens": 1}}


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(ts.ENV_FLAG, "embed")
    monkeypatch.setenv(ts.ENV_K, "4")
    ts.set_native_selector_provider(provider({"syn_sku_price": 1.0}))


def test_list_tools_is_executed_server_side_and_always_offered(on):
    prim = Prim([[_call("list_tools")], [_call("syn_sku_price", {"x": "1"})]])
    content, calls, finish, usage = _run(prim, many())
    assert [c["function"]["name"] for c in calls] == ["syn_sku_price"] and finish == "tool_calls"
    assert len(prim.seen) == 2 and usage.prompt_tokens == 20  # token counts are summed over the internal re-call
    tool_msg = [m for m in prim.seen[1][0] if m["role"] == "tool"][0]
    assert "syn_t3_x3" in tool_msg["content"]  # the FULL list, including tools that were not picked
    assert names(prim.seen[0][1])[:3] == ["list_tools", "search_tools", "describe_tool"]


def test_search_tools_pulls_a_tool_into_the_frozen_set(on):
    prim = Prim([[_call("search_tools", {"query": "tool syn_t7_x7"})], [_call("syn_t7_x7", {"x": "1"})]])
    _content, calls, _f, _u = _run(prim, many())
    assert names(prim.seen[1][1]).count("syn_t7_x7") == 1  # callable on the next step
    assert [c["function"]["name"] for c in calls] == ["syn_t7_x7"]


def test_describe_tool_and_mixed_calls(on):
    prim = Prim([[_call("describe_tool", {"name": "syn_t9_x9"}), _call("read_file", {"x": "a"}, "c2")]])
    _c, calls, _f, _u = _run(prim, many())
    assert [c["function"]["name"] for c in calls] == ["read_file"]  # the client's call wins; the server call is dropped


def test_hallucinated_tool_name_repicks_once_and_retries(on):
    prim = Prim([[_call("syn_sku_pricee", {"x": "1"})], [_call("syn_sku_price", {"x": "1"})]])
    _c, calls, _f, _u = _run(prim, many())
    assert [c["function"]["name"] for c in calls] == ["syn_sku_price"] and len(prim.seen) == 2
    prim2 = Prim([[_call("nope1")], [_call("nope2")]])
    _c, calls, _f, _u = _run(prim2, many(), msgs=MSG + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "again"}])
    assert len(prim2.seen) == 2  # re-picked ONCE only; a second unknown name is returned as is


def test_flag_off_is_the_legacy_single_call(monkeypatch):
    monkeypatch.delenv(ts.ENV_FLAG, raising=False)
    tools = many()
    prim = Prim([[_call("list_tools")]])
    _c, calls, _f, _u = _run(prim, tools)
    assert prim.seen[0][1] is tools and [c["function"]["name"] for c in calls] == ["list_tools"]  # untouched, no server tools


# (g) code mode: the pick ADDS, search_tools always present ----------------------------------------------------------------------
def test_repl_block_adds_search_tools_and_keeps_list_tools():
    d = [ts.openai_tool_to_dict(x) for x in many(3)]
    blk = ts.render_tools_block(d, ["read_file"])
    assert "`list_tools()`" in blk and "search_tools(query)" in blk


def test_search_tools_installed_whenever_the_picker_is_on():
    from src.repl_environment import codemode_tools as cm

    env = SimpleNamespace(tool_registry=object(), _list_tools=lambda: [], llm_primitives=None)
    g: dict = {}
    assert cm.install(env, g, {}) == []
    assert cm.install(env, g, {ts.ENV_FLAG: "embed"}) == ["search_tools", "describe_tool"]


def test_repl_provider_freezes_core_pins_and_fails_open():
    np = pytest.importorskip("numpy")
    from src.repl_environment.context_search import DenseResult

    class Fake:
        model_id = "fake"

        def embed(self, texts, *, timeout_s=0.0, admission_wait_s=0.0):
            return DenseResult(np.array([[1.0 if "price" in t else 0.0, 0.0, 0.1] for t in texts], dtype="float32"), "fake")

    dicts = [ts.openai_tool_to_dict(x) for x in many(14)]
    for d in dicts:
        d["parameters"] = {"x": {"type": "string", "required": True}}
    prov = ts.make_repl_provider(Fake(), lambda: dicts)
    blk = prov("price of the sku")
    assert blk and "read_file" in blk and "syn_sku_price" in blk and "list_tools()" in blk
    assert prov("price of the sku") == blk  # frozen per task
    assert ts.make_repl_provider(Fake(), lambda: dicts[:3])("price") is None  # <= k: nothing to pick
