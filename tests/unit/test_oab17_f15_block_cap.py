"""OAB-17 F15: scout block is bounded; evidence scan is bounded."""
from __future__ import annotations

import time
from types import SimpleNamespace

from src.api.routes.chat_pipeline import scout_stage as ss


def _result(i, summary):
    target = SimpleNamespace(describe=lambda: f"t{i}")
    return SimpleNamespace(index=i, status="ok", summary=summary, target=target, located=None,
                           turns=1, error=None)


def test_block_is_bounded_and_marked():
    results = [_result(i, "x" * 30_000) for i in range(8)]
    block = ss.render_block(results, role="r", cap=4)
    assert len(block) < ss.BLOCK_MAX_CHARS + 8 * 400 + 2_000
    assert "block cap" in block


def test_small_block_unchanged():
    block = ss.render_block([_result(0, "found it at a.c:12")], role="r", cap=1)
    assert "block cap" not in block and "found it at a.c:12" in block


def test_evidence_scan_is_bounded_not_quadratic():
    hostile = "a" * 200_000 + ".c:"                # worst case for the unbounded pattern
    t0 = time.monotonic()
    ss._EVIDENCE_RE.findall(hostile[:ss.EVIDENCE_SCAN_CHARS])
    assert time.monotonic() - t0 < 5
    assert ss.EVIDENCE_SCAN_CHARS <= 20_000
