"""Prefix-cache layout flags: prompt_stable_prefix, history_append_only, tool_output_bound (all default off).

Flag-off output is compared to a golden rebuilt from the legacy rendering rules (a RootLMPrompt rendered the old way),
not from the code under test.
"""

from __future__ import annotations

import pytest

from src.features import _FEATURE_REGISTRY, Features, reset_features, set_features
from src.prompt_builders import tool_selector
from src.prompt_builders.builder import PromptBuilder
from src.prompt_builders.turn_memory import (
    APPEND_ONLY_CHAR_BUDGET, ENV_TURN_MEMORY, TurnMemory, bound_text, render_turn_memory,
)
from src.prompt_builders.types import PromptConfig, PromptStyle, RootLMPrompt

FLAGS = ("prompt_stable_prefix", "history_append_only", "tool_output_bound")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENV_TURN_MEMORY, raising=False)
    monkeypatch.delenv("ORCHESTRATOR_TOOL_SELECTOR", raising=False)
    reset_features()
    tool_selector.set_selector_provider(None)
    yield
    tool_selector.set_selector_provider(None)
    reset_features()


def _build(task="do the thing", state="artifacts: {}", turn=0, **kw):
    b = PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL))
    return b.build_root_lm_prompt(state=state, original_prompt=task, turn=turn, **kw)


def test_flags_registered_default_off():
    specs = {s.name: s for s in _FEATURE_REGISTRY}
    for name in FLAGS:
        assert specs[name].default_prod is False and specs[name].default_test is False
        assert getattr(Features(), name) is False


def test_flag_off_prompt_is_byte_identical_to_legacy_render():
    set_features(Features())
    out = _build(last_output="out", routing_context="rc", turn=3)
    p = _build(last_output="out", routing_context="rc", turn=3, as_structured=True)
    # legacy: head-of-state counter, legacy order, nothing from the new fields
    assert p.selected_tools == "" and p.turn_tail == "" and p.state.startswith("Turn 4\n")
    legacy = RootLMPrompt(system=p.system, tools=p.tools, rules=p.rules, state=p.state, context=p.context,
                          task=p.task, instruction=p.instruction)
    assert out == legacy.to_string(False)
    assert out == p.to_string(False) == p.to_string(False, False)


def test_flag_off_turn_memory_is_sliding_window():
    m = TurnMemory(2)
    for i in range(5):
        m.add(i, f"c{i}", f"o{i}")
    assert [e["turn"] for e in m.entries()] == [3, 4]
    assert TurnMemory(0).entries() == []


def test_flag_off_output_not_bound():
    m = TurnMemory(2)
    m.add(0, "c", "x" * 5000)
    assert len(m.entries()[0]["output"]) == 5000


def _provider(order):
    def provider(task):
        from src.prompt_builders.tool_selector import render_tools_block

        tools = [{"name": n, "description": "d", "parameters": {}} for n in ("zeta", "alpha", "mid")]
        return render_tools_block(tools, order)
    return provider


def test_stable_prefix_identical_bytes_when_only_volatile_values_differ(monkeypatch):
    set_features(Features(prompt_stable_prefix=True))
    monkeypatch.setenv(ENV_TURN_MEMORY, "4")
    h1 = [{"turn": 0, "code": "a=1", "output": "1", "error": ""}]
    tool_selector.set_selector_provider(_provider(["zeta", "alpha"]))
    monkeypatch.setattr(tool_selector, "ensure_providers", lambda *a, **k: None)
    monkeypatch.setattr(tool_selector, "selector_variant", lambda *a, **k: "embed")
    a = _build(state="artifacts: {}", turn=1, last_output="A", routing_context="r1", turn_history=h1)
    tool_selector.set_selector_provider(_provider(["mid", "zeta", "alpha"]))
    b = _build(state="artifacts: {'k': 'v'}\n  x = 5", turn=7, last_output="BBBB", last_error="",
               routing_context="r2", corpus_context="ref", turn_history=h1 + [{"turn": 1, "code": "b", "output": "2", "error": ""}])
    boundary = a.index("## Your Code")
    assert a[:boundary] == b[:boundary]
    # the stable prefix includes system, static tools, rules and task; volatile values come after it
    assert "## Task" in a[:boundary] and "## Available Tools" in a[:boundary] and "## Rules" in a[:boundary]
    for vol in ("Turn 2", "Turn 8", "x = 5", "r1", "r2"):
        assert vol not in a[:boundary] and vol not in b[:boundary]
    # selected tools are name-sorted whatever order the picker returned
    assert b.index("`alpha(") < b.index("`mid(") < b.index("`zeta(")
    # the history block precedes the state, and the older turn's bytes are a prefix of the newer render
    assert a.index("## Previous Turns") < a.index("## Current State")


def test_stable_prefix_history_is_append_only_prefix(monkeypatch):
    set_features(Features(prompt_stable_prefix=True))
    monkeypatch.setenv(ENV_TURN_MEMORY, "4")
    h = [{"turn": i, "code": f"c{i}", "output": f"o{i}", "error": ""} for i in range(3)]
    a = _build(turn=3, turn_history=h[:2])
    b = _build(turn=4, turn_history=h)
    hist_a = a[a.index("## Previous Turns"):a.index("## Current State")]
    assert b.startswith(a[:a.index("## Previous Turns")])
    assert hist_a.rstrip() in b


def test_stable_prefix_cross_task_shares_system_tools_rules(monkeypatch):
    set_features(Features(prompt_stable_prefix=True))
    monkeypatch.setattr(tool_selector, "ensure_providers", lambda *a, **k: None)
    monkeypatch.setattr(tool_selector, "selector_variant", lambda *a, **k: "embed")
    tool_selector.set_selector_provider(_provider(["alpha"]))
    a = _build(task="first task")
    tool_selector.set_selector_provider(_provider(["zeta", "mid"]))
    b = _build(task="a different one")
    cut = a.index("## Task")
    assert a[:cut] == b[:cut]
    set_features(Features())
    tool_selector.set_selector_provider(_provider(["alpha"]))
    legacy_a = _build(task="first task")
    tool_selector.set_selector_provider(_provider(["zeta", "mid"]))
    legacy_b = _build(task="a different one")
    assert legacy_a[:legacy_a.index("## Task")] != legacy_b[:legacy_b.index("## Task")]


def test_append_only_keeps_all_turns_then_truncates_once():
    m = TurnMemory(2, append_only=True)
    for i in range(10):
        m.add(i, f"c{i}", "o")
    assert [e["turn"] for e in m.entries()] == list(range(10))  # no per-turn sliding
    big = "x" * 1000
    m2 = TurnMemory(2, append_only=True)
    sizes = []
    for i in range(40):
        m2.add(i, big, big, "")
        sizes.append(len(m2.entries()))
    drops = [i for i in range(1, 40) if sizes[i] < sizes[i - 1] + 1]
    assert 1 <= len(drops) <= 2  # one checkpoint truncation, not a slide every turn
    assert m2.entries()[-1]["turn"] == 39
    assert sum(len(e["code"]) + len(e["output"]) for e in m2.entries()) <= APPEND_ONLY_CHAR_BUDGET


def test_append_only_render_shows_every_entry(monkeypatch):
    set_features(Features(history_append_only=True))
    monkeypatch.setenv(ENV_TURN_MEMORY, "2")
    h = [{"turn": i, "code": f"code{i}", "output": "o", "error": ""} for i in range(5)]
    out = _build(turn_history=h)
    assert all(f"code{i}" in out for i in range(5))
    set_features(Features())
    out = _build(turn_history=h)
    assert "code0" not in out and "code3" in out and "code4" in out


def test_tool_output_bound_is_deterministic_with_marker():
    t = "".join(chr(65 + i % 26) for i in range(9000))
    b = bound_text(t)
    assert b == bound_text(t) and len(b) < 2100
    assert b.startswith(t[:1500]) and b.endswith(t[-500:]) and "chars omitted by tool_output_bound" in b
    assert bound_text("short") == "short"
    m = TurnMemory(2, bound_output=True)
    m.add(0, "c", t, t)
    e = m.entries()[0]
    assert len(e["output"]) < 2100 and len(e["error"]) < 2100
    assert "o" in render_turn_memory([e], 1)
