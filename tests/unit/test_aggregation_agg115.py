"""AGG-1.15 (2026-11-16): two opt-in fixes for the 6 FY2022 long_term_debt cells ``agg_pipeline.v2`` lost vs AGG-1.4 r3.

* ``facts_typed.keep_slot_harvest`` (``keep_slot.harvest.v1``): a typed reply's rows for a slot the document did not PLAN are
  kept from any called window (v2 plans the 8 schema fields, so long_term_debt rows were always dropped). Zero extra calls.
* ``canonical.v3``: v2 plus ``guard_long_term_debt_label`` (a candidate quoting the "long-term debt" line item beats prose totals
  of all debt). v1 and v2 are unchanged. No model, no network.
"""

from __future__ import annotations

import json

from src.aggregation import facts_typed as ft
from src.aggregation.store import (CANONICAL_RULE_V2, CANONICAL_RULE_V3, CANONICAL_RULES, guard_long_term_debt_label,
                                   select_canonical)


def _f(fid, value, quote, *, field="long_term_debt", page=1):
    return {"corpus_id": "c", "fact_id": fid, "doc_hash": "h", "doc_id": "AMD", "entity": "AMD", "period": "FY2022",
            "field": field, "value": value, "unit": "USD_millions", "quote": quote, "page": page, "confidence": "high",
            "extractor": "l4_llm.v1", "audit_status": "disputed"}


AMD = [_f("a", 1474.0, "Long-term debt 1,474", page=71),
       _f("b", 2500.0, "total debt principal amount outstanding as of December 31, 2022 was $2.5 billion", page=37),
       _f("c", 2467.0, "Long-term debt, net of current portion 2,467", page=78)]


def test_keep_slot_harvest_keeps_unplanned_slots_only():
    planned = ft.SCHEMA_FIELDS
    assert ft.keep_slot_harvest({"field": "long_term_debt"}, {"revenue"}, planned)       # unplanned slot: kept anywhere
    assert not ft.keep_slot_harvest({"field": "capex"}, {"revenue"}, planned)            # planned, not chosen: dropped as before
    assert ft.keep_slot_harvest({"field": "revenue"}, {"revenue"}, planned)
    assert not ft.keep_slot({"field": "long_term_debt"}, {"revenue"})                    # keep_slot itself unchanged
    assert ft.KEEP_SLOT_HARVEST == "keep_slot.harvest.v1"


def test_canonical_v3_prefers_the_line_item_and_v2_is_unchanged():
    assert CANONICAL_RULE_V3 in CANONICAL_RULES and CANONICAL_RULE_V2 in CANONICAL_RULES
    v2 = select_canonical(AMD, rule=CANONICAL_RULE_V2)[0][0]
    v3 = select_canonical(AMD, rule=CANONICAL_RULE_V3)[0][0]
    assert v2["value"] == 2500.0 and v2["rule"] == CANONICAL_RULE_V2      # |value| desc picks the prose total (the AGG-1.15 miss)
    assert v3["value"] == 2467.0 and v3["rule"] == CANONICAL_RULE_V3


def test_long_term_debt_guard_is_narrow_and_never_empties():
    assert guard_long_term_debt_label("total_assets", AMD) == (AMD, None)
    prose = [AMD[1]]
    assert guard_long_term_debt_label("long_term_debt", prose) == (prose, None)
    named, note = guard_long_term_debt_label("long_term_debt", AMD)
    assert note == "long_term_debt_label" and [r["fact_id"] for r in named] == ["a", "c"]
    other = [dict(r, field="revenue") for r in AMD]
    assert select_canonical(other, rule=CANONICAL_RULE_V3)[0] == [dict(r, rule=CANONICAL_RULE_V3) for r in
                                                                   select_canonical(other, rule=CANONICAL_RULE_V2)[0]]


def test_fill_doc_harvest_is_opt_in(monkeypatch):
    from src.aggregation import FLAG_ENV as LAYER_FLAG
    monkeypatch.setenv(LAYER_FLAG, "1")
    from tests.unit.test_aggregation_pipeline import CID, DESCS, _two_chunk_store

    class Echo:
        def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
            ex = prompt.split("Excerpt:\n", 1)[1]
            body = {s: [] for s in ft.SLOTS}
            if "Total assets 27,165" in ex:
                body["long_term_debt"] = [{"period": "2022", "value": 27165.0, "unit": "USD_millions",
                                           "quote": "Total assets 27,165", "confidence": "high"}]
            return {"text": json.dumps(body), "prompt_tokens": 1, "cache_n": 0, "decode_tokens": 1, "finish_reason": "stop"}

    for harvest, want in ((False, 0), (True, 1)):
        st = _two_chunk_store()
        ft.TypedBackfill(st, Echo()).fill_doc(CID, ft.split_windows(st, CID), ft.SCHEMA_FIELDS, DESCS, harvest=harvest)
        got = [f for f in st.facts(CID, extractor="l4_llm.v1") if f["field"] == "long_term_debt"]
        assert len(got) == want
