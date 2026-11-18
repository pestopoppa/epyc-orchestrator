"""OAB-17 F11: the pulled-span list is bounded and the union is unchanged."""
from __future__ import annotations

from src.repl_environment import context_bundle as cb


def test_merge_preserves_union_bytes():
    text = "αβγ" * 4000
    spans = [(i * 3, i * 3 + 5) for i in range(3000)]          # heavily overlapping
    before = cb._intervals_union_bytes(text, spans)
    merged = cb._merge_intervals(spans)
    assert len(merged) < len(spans)
    assert cb._intervals_union_bytes(text, merged) == before


def test_spans_stay_bounded_under_many_pulls():
    bundle = cb.ContextBundle.from_payload(
        {"sections": [{"name": "s", "text": "x" * 20000}]})
    for i in range(2500):
        bundle.span_pull(i, i + 10, op="peek")
    per = bundle._per_section["s"]
    assert len(per["spans"]) <= cb.SPAN_MERGE_THRESHOLD + 1
    assert cb._intervals_union_bytes(bundle._by_name["s"].text, per["spans"]) == 2509
