"""Tests for session_log.py — TurnRecord, ConsolidatedSegment, two-level condensation."""

import time

import pytest

from src.graph.session_log import (
    TurnRecord,
    ConsolidatedSegment,
    build_turn_record,
    build_granular_summary,
    should_consolidate,
    build_consolidation_prompt,
    build_session_summary_deterministic,
    MAX_GRANULAR_BLOCKS,
)


# ---------------------------------------------------------------------------
# TurnRecord basics
# ---------------------------------------------------------------------------


def test_build_turn_record_ok():
    rec = build_turn_record(turn=1, role="frontdoor", code="x = 1", output="1")
    assert rec.outcome == "ok"
    assert rec.code_hash
    assert rec.code_line_count == 1


def test_build_turn_record_error():
    rec = build_turn_record(turn=2, role="coder", error="NameError: x")
    assert rec.outcome == "error"
    assert "NameError" in rec.error_message


def test_build_turn_record_final():
    rec = build_turn_record(turn=3, role="frontdoor", is_final=True, output="42")
    assert rec.outcome == "final"


def test_build_turn_record_escalation():
    rec = build_turn_record(turn=4, role="frontdoor", escalation_target="coder")
    assert rec.outcome == "escalation"


def test_to_log_line():
    rec = build_turn_record(turn=1, role="frontdoor", code="x=1", output="1")
    line = rec.to_log_line()
    assert "T1(frontdoor)" in line
    assert "ok" in line


def test_to_markdown_block():
    rec = build_turn_record(turn=1, role="coder", code="print(1)", output="1")
    md = rec.to_markdown_block()
    assert "### Turn 1" in md
    assert "coder" in md


# ---------------------------------------------------------------------------
# ConsolidatedSegment
# ---------------------------------------------------------------------------


def test_consolidated_segment_serialization():
    seg = ConsolidatedSegment(
        turn_range=(1, 5),
        granular_blocks=["T1 ok", "T2 ok", "T3 error"],
        consolidated="Turns 1-5: tried X, failed at Y, succeeded with Z.",
        trigger="escalation",
        timestamp=time.time(),
    )
    d = seg.to_dict()
    restored = ConsolidatedSegment.from_dict(d)
    assert restored.turn_range == (1, 5)
    assert restored.consolidated == seg.consolidated
    assert restored.trigger == "escalation"
    assert len(restored.granular_blocks) == 3


def test_consolidated_segment_prompt_block():
    seg = ConsolidatedSegment(
        turn_range=(3, 8),
        granular_blocks=[],
        consolidated="Dense summary here.",
        trigger="block_limit",
    )
    block = seg.to_prompt_block()
    assert "[Turns 3-8" in block
    assert "Dense summary here." in block


# ---------------------------------------------------------------------------
# Granular summary (Tier 1)
# ---------------------------------------------------------------------------


def test_build_granular_summary():
    rec = build_turn_record(turn=1, role="frontdoor", code="x=1", output="1")
    summary = build_granular_summary(rec)
    assert "T1(frontdoor)" in summary
    assert "ok" in summary


# ---------------------------------------------------------------------------
# Consolidation triggers
# ---------------------------------------------------------------------------


def test_should_consolidate_escalation():
    rec = build_turn_record(turn=5, role="coder", escalation_target="architect")
    trigger = should_consolidate(["b1", "b2"], rec)
    assert trigger == "escalation"


def test_should_consolidate_role_change():
    rec = build_turn_record(turn=5, role="coder", output="ok")
    trigger = should_consolidate(["b1", "b2"], rec, prev_role="frontdoor")
    assert trigger == "escalation"


def test_should_consolidate_final():
    rec = build_turn_record(turn=5, role="frontdoor", is_final=True, output="42")
    trigger = should_consolidate(["b1"], rec)
    assert trigger == "final"


def test_should_consolidate_block_limit():
    blocks = [f"block_{i}" for i in range(MAX_GRANULAR_BLOCKS)]
    rec = build_turn_record(turn=20, role="frontdoor", output="ok")
    trigger = should_consolidate(blocks, rec)
    assert trigger == "block_limit"


def test_should_consolidate_none():
    rec = build_turn_record(turn=3, role="frontdoor", output="ok")
    trigger = should_consolidate(["b1", "b2"], rec, prev_role="frontdoor")
    assert trigger is None


# ---------------------------------------------------------------------------
# Consolidation prompt
# ---------------------------------------------------------------------------


def test_build_consolidation_prompt():
    blocks = ["T1 ok code:abc", "T2 error err:NameError", "T3 ok"]
    prompt = build_consolidation_prompt(blocks)
    assert "3 entries" in prompt
    assert "T2 error" in prompt


# ---------------------------------------------------------------------------
# Deterministic summary (existing)
# ---------------------------------------------------------------------------


def test_build_session_summary_deterministic_empty():
    assert build_session_summary_deterministic([]) == ""


def test_build_session_summary_deterministic_few():
    records = [
        build_turn_record(turn=i, role="frontdoor", output=f"out{i}")
        for i in range(3)
    ]
    summary = build_session_summary_deterministic(records)
    assert "Session History" in summary
    assert "Turns: 3" in summary


def test_build_session_summary_deterministic_many():
    records = [
        build_turn_record(turn=i, role="frontdoor", output=f"out{i}")
        for i in range(10)
    ]
    summary = build_session_summary_deterministic(records)
    assert "omitted" in summary
