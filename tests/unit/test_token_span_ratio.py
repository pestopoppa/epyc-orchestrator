"""AEI-TOKEN-SPAN-RATIO: the span ratio never invents a number and flags double counting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "token_span_ratio", Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "token_span_ratio.py"
)
tsr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tsr)


def _row(rid, cid, comp, *, event="settled", censored=False, pending=False, parent=None):
    return {
        "event": event,
        "request_id": rid,
        "call": {
            "call_id": cid,
            "completion_tokens": comp,
            "prompt_tokens": None,
            "censored": censored,
            "pending": pending,
            "parent_call_id": parent,
        },
    }


def test_single_counted_nested_calls_give_ratio_one():
    rows = [_row("r", "a", 10), _row("r", "b", 5, parent="span1"), _row("r", "b", 5, parent="span1")]
    out = tsr.check(rows, {"r": {"output_tokens_reported": 15}})
    assert out["requests"]["r"]["ratio"] == 1.0 and out["summary"] == {"ok": 1}


def test_double_counted_span_is_a_defect():
    rows = [_row("r", "a", 10), _row("r", "span", 10)]  # parent re-reports the child's tokens
    out = tsr.check(rows, {"r": {"output_tokens_reported": 10}})
    assert out["requests"]["r"]["ratio"] == 2.0
    assert out["summary"] == {"DEFECT_ratio_not_1": 1}


def test_missing_root_or_counts_yield_no_ratio():
    assert tsr.check([_row("r", "a", 3)], {})["requests"]["r"]["status"] == "no_root_total"
    out = tsr.check([_row("r", "a", None, censored=True)], {"r": {"output_tokens_reported": 3}})
    assert out["requests"]["r"]["status"] == "no_call_counts" and out["requests"]["r"]["ratio"] is None
    assert tsr.check([_row("r", "a", 3)], {"r": {"output_tokens_reported": 0}})["requests"]["r"]["status"] == "zero_root"


def test_begin_events_are_ignored():
    rows = [_row("r", "a", 7, event="begin"), _row("r", "a", 7)]
    assert tsr.check(rows, {"r": {"output_tokens_reported": 7}})["requests"]["r"]["ratio"] == 1.0
