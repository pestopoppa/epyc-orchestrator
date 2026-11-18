"""Codemode reliability fix 2: ORCHESTRATOR_REPL_TURN_MEMORY=N (default 0)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.prompt_builders.builder import PromptBuilder
from src.prompt_builders.turn_memory import (
    ENV_TURN_MEMORY, TurnMemory, render_turn_memory, turn_memory_kwargs, turn_memory_n,
)
from src.prompt_builders.types import PromptConfig, PromptStyle

H = [{"turn": 0, "code": "x = open('a')", "output": "", "error": "REPLSecurityError: open"},
     {"turn": 1, "code": "print(list_dir('d'))", "output": "['a.json']", "error": ""}]


@pytest.fixture(autouse=True)
def _off(monkeypatch):
    monkeypatch.delenv(ENV_TURN_MEMORY, raising=False)


def build(**kw):
    b = PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL))
    return b.build_root_lm_prompt(state="s", original_prompt="task", turn=2, **kw)


def test_parse():
    assert turn_memory_n({}) == 0 and turn_memory_n({ENV_TURN_MEMORY: "3"}) == 3
    assert turn_memory_n({ENV_TURN_MEMORY: "abc"}) == 0 and turn_memory_n({ENV_TURN_MEMORY: "-2"}) == 0
    assert turn_memory_n({ENV_TURN_MEMORY: "999"}) == 16


def test_flag_off_ignores_history_byte_identical():
    base = build(last_output="out", last_error="")
    assert build(last_output="out", last_error="", turn_history=H) == base
    assert "Previous Turns" not in base and "## Last Output" in base


def test_flag_on_renders_appended_block(monkeypatch):
    base = build(last_output="out")
    monkeypatch.setenv(ENV_TURN_MEMORY, "2")
    out = build(last_output="out", turn_history=H)
    assert "## Previous Turns" in out and "x = open('a')" in out and "REPL" in out and "['a.json']" in out
    assert out.rstrip().endswith("```")  # history is the last section
    # last output is carried inside the history, not duplicated as a separate block
    assert "## Last Output" not in out
    # nothing in front of the history block changed except the removed Last Output block
    assert out.startswith(base.split("## Last Output")[0].rstrip())


def test_flag_on_empty_history_is_noop(monkeypatch):
    monkeypatch.setenv(ENV_TURN_MEMORY, "2")
    assert build(last_output="out") == build(last_output="out", turn_history=[])
    assert "## Last Output" in build(last_output="out")


def test_prefix_stable_across_turns(monkeypatch):
    monkeypatch.setenv(ENV_TURN_MEMORY, "4")
    one = build(turn_history=H[:1], last_error="e")
    two = build(turn_history=H, last_error="e")
    head = one.split("## Previous Turns")[0]
    assert two.startswith(head)  # only the tail grew
    assert two.split("## Previous Turns")[1].startswith(one.split("## Previous Turns")[1])  # history only appended


def test_window_and_preview(monkeypatch):
    big = [{"turn": i, "code": "c%d" % i, "output": "z" * 9000, "error": ""} for i in range(5)]
    text = render_turn_memory(big, 2)
    assert "c3" in text and "c4" in text and "c2" not in text
    assert text.count("z") == 2 * 4000 and "5000 more chars" in text
    assert render_turn_memory(H, 0) == "" and render_turn_memory([], 3) == ""


def test_error_shown_instead_of_output():
    t = render_turn_memory([{"turn": 0, "code": "c", "output": "o", "error": "boom"}], 1)
    assert "boom" in t and "Output:" not in t


def test_memory_deque_and_kwargs(monkeypatch):
    m = TurnMemory(2)
    for i in range(3):
        m.add(i, f"c{i}", "o")
    assert [e["code"] for e in m.entries()] == ["c1", "c2"]
    assert TurnMemory(0).entries() == []
    st = SimpleNamespace(turn_memory=m)
    assert turn_memory_kwargs(st) == {}
    monkeypatch.setenv(ENV_TURN_MEMORY, "2")
    assert len(turn_memory_kwargs(st)["turn_history"]) == 2
    assert turn_memory_kwargs(SimpleNamespace()) == {}


def test_production_hook_records_turns(monkeypatch):
    from src.graph.session_summary import _record_session_turn

    st = SimpleNamespace(turns=1, session_log_path="", task_id="t")
    _record_session_turn(st, role="frontdoor", code="a=1", output="o")
    assert not hasattr(st, "turn_memory")  # flag off: nothing recorded
    monkeypatch.setenv(ENV_TURN_MEMORY, "2")
    _record_session_turn(st, role="frontdoor", code="a=1", output="o")
    st.turns = 2
    _record_session_turn(st, role="frontdoor", code="b=2", error="E")
    assert [(e["turn"], e["code"], e["error"]) for e in st.turn_memory.entries()] == [(0, "a=1", ""), (1, "b=2", "E")]
