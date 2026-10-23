"""UFH14-LB3.3 (turn-1 grammar failure is loud) and UFH14-LB11.1 (unclosed <think> fence cut)."""
from __future__ import annotations

import logging
from types import SimpleNamespace

from src.graph import helpers as h
from src.graph.state import TaskState


def _state():
    return SimpleNamespace(diag_counters={})


def test_grammar_error_warns_once_per_type_and_counts_every_time(caplog):
    h._WARNED_EXC_TYPES.clear()
    st = _state()
    with caplog.at_level(logging.WARNING, logger=h.log.name):
        h._note_turn1_grammar_error(st, UnboundLocalError("x"))
        h._note_turn1_grammar_error(st, UnboundLocalError("y"))
        h._note_turn1_grammar_error(st, KeyError("k"))
    msgs = [r.getMessage() for r in caplog.records if "turn-1 grammar" in r.getMessage()]
    assert len(msgs) == 2
    assert "UnboundLocalError: x" in msgs[0] and "KeyError" in msgs[1]
    assert st.diag_counters["turn1_grammar_error"] == 3


def test_taskstate_declares_diag_counters():
    assert "diag_counters" in TaskState.__dataclass_fields__


def test_note_never_raises_on_bad_state():
    h._note_turn1_grammar_error(object(), RuntimeError("boom"))


def test_unclosed_think_detection():
    assert h._ends_inside_unclosed_think("<think>reasoning ```python\nx")
    assert not h._ends_inside_unclosed_think("<think>r</think>\n```python\nx")
    assert h._ends_inside_unclosed_think("<think>a</think>ans<think>b")
    assert not h._ends_inside_unclosed_think("plain code")
    assert not h._ends_inside_unclosed_think("")


def test_trim_after_think_cuts_at_closing_fence():
    t = "<think>a\n```\nb</think>\n```python\nFINAL(1)\n```\nrepeat FINAL(1)\n```\n"
    assert h._trim_after_think_at_fence(t) == "<think>a\n```\nb</think>\n```python\nFINAL(1)"
    # no fence opener after </think>: untouched
    assert h._trim_after_think_at_fence("<think>a</think>FINAL(1)") == "<think>a</think>FINAL(1)"


def test_feature_flag_default_off():
    from src.features import get_features

    assert get_features().repl_think_fence_continue is False
    assert get_features(override={"repl_think_fence_continue": True}).repl_think_fence_continue is True


def test_turn1_grammar_block_except_is_not_silent():
    """The grammar block's handler must call the reporter (no bare ``pass`` swallow)."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(h.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_execute_turn")
    tries = [
        t for t in ast.walk(fn)
        if isinstance(t, ast.Try)
        and any("ORCHESTRATOR_TURN1_GRAMMAR" in ast.dump(b) for b in t.body)
    ]
    assert len(tries) == 1
    (handler,) = tries[0].handlers
    assert handler.name
    assert not all(isinstance(s, ast.Pass) for s in handler.body)
    assert "_note_turn1_grammar_error" in ast.dump(handler)


def test_forced_exception_in_block_is_reported(monkeypatch, caplog):
    """Drive the reporter exactly as the handler does, with a real exception from a registry."""
    h._WARNED_EXC_TYPES.clear()
    st = _state()

    class Boom:
        def generate_gbnf_grammar(self, *a, **k):
            raise ValueError("bad grammar")

    try:
        Boom().generate_gbnf_grammar("r", shape="legacy")
    except Exception as e:
        with caplog.at_level(logging.WARNING, logger=h.log.name):
            h._note_turn1_grammar_error(st, e)
    assert st.diag_counters == {"turn1_grammar_error": 1}
    assert any("ValueError: bad grammar" in r.getMessage() for r in caplog.records)
