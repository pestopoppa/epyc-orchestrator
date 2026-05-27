"""Unit tests for the REPL loop-guard (Fix A fence-repair + Fix B identical-turn breaker).

Root cause (2026-05-27 BEP-2 tap diagnosis, not a model/viability issue): the FINAL(/CALL(
early-stop in `_execute_turn` aborts streaming the instant it fires — which can land *inside*
an open ``` fence, before the closing ``` → the extractor returns nothing → the model re-emits
the identical turn until timeout. Separately, the REPL has no identical-non-advancing-turn
breaker, so a model that re-emits the same `peek()` read / unknown-tool CALL spins until the
turn budget burns out. Both fixes are pure helpers, flag-gated default-off behind
ORCHESTRATOR_REPL_LOOP_GUARD so the gitnexus-CRITICAL `_execute_turn` path is a prod no-op
until the BEP A/B validates them.
"""
from __future__ import annotations

from src.graph.helpers import (
    _repair_unclosed_code_fence,
    _loop_guard_repeat,
    _repl_loop_guard_enabled,
)


# ── Fix A: fence repair ───────────────────────────────────────────────
def test_repair_closes_fence_truncated_by_final_early_stop():
    # Exact shape from the BEP trace (TASK #2): write + FINAL in one block, early-stop ate the close.
    truncated = '```python\ncontent = "x"\nfile_write_safe("m.py", content)\nFINAL("done")'
    fixed = _repair_unclosed_code_fence(truncated)
    assert fixed.count("```") == 2
    assert fixed.rstrip().endswith("```")
    # the original content survives
    assert 'file_write_safe("m.py", content)' in fixed


def test_repair_leaves_balanced_fence_untouched():
    balanced = '```python\nfile_write_safe("m.py", "x")\n```'
    assert _repair_unclosed_code_fence(balanced) == balanced


def test_repair_noop_on_fenceless_text():
    assert _repair_unclosed_code_fence("just prose, no code") == "just prose, no code"
    assert _repair_unclosed_code_fence("") == ""


# ── Fix B: identical-turn repeat counter ──────────────────────────────
def test_repeat_increments_on_identical_no_final():
    # The read-loop shape (TASK #5/#6): same peek() re-emitted, never writes, no FINAL.
    code = "```python\npeek(1000, file_path='calc.py')\n```"
    assert _loop_guard_repeat(code, code, has_final=False, prev_count=0) == 1
    assert _loop_guard_repeat(code, code, has_final=False, prev_count=1) == 2


def test_repeat_resets_when_turn_changes():
    assert _loop_guard_repeat("read A", "write B", has_final=False, prev_count=3) == 0


def test_repeat_resets_on_final_even_if_identical():
    same = "FINAL('answer')"
    assert _loop_guard_repeat(same, same, has_final=True, prev_count=2) == 0


def test_repeat_first_turn_is_zero():
    assert _loop_guard_repeat(None, "first output", has_final=False, prev_count=0) == 0
    assert _loop_guard_repeat("", "first output", has_final=False, prev_count=0) == 0


# ── flag gating (CRITICAL path stays a prod no-op by default) ─────────
def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_REPL_LOOP_GUARD", raising=False)
    assert _repl_loop_guard_enabled() is False


def test_flag_on_when_set(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_LOOP_GUARD", "1")
    assert _repl_loop_guard_enabled() is True
