"""AGG-1.16 (2026-11-16): ``agg_pipeline.v3`` = ``agg_pipeline.v2`` + ``keep_slot.harvest.v1`` + ``canonical.v3`` (the two AGG-1.15 fixes,
bundled under the same ``ORCHESTRATOR_AGG_PIPELINE`` flag). v1 stays the default; v1 and v2 specs, kwargs, records and outputs are unchanged.
No model, no network: an echo backend and in-memory stores.
"""

from __future__ import annotations

import json

import pytest

from src.aggregation import FLAG_ENV as LAYER_FLAG
from src.aggregation import facts_det as fd
from src.aggregation import facts_typed as ft
from src.aggregation import pipeline as pl
from src.aggregation.store import CANONICAL_RULE, CANONICAL_RULE_V2, CANONICAL_RULE_V3, CorpusStore, select_canonical


@pytest.fixture(autouse=True)
def _layer_on(monkeypatch):
    monkeypatch.setenv(LAYER_FLAG, "1")
    monkeypatch.delenv(pl.FLAG_ENV, raising=False)


def test_v3_is_selectable_by_the_same_flag_and_v1_stays_default(monkeypatch):
    assert pl.PIPELINES == (pl.PIPELINE_V1, pl.PIPELINE_V2, pl.PIPELINE_V3) and pl.DEFAULT == pl.PIPELINE_V1
    assert pl.pipeline_spec().version == pl.PIPELINE_V1
    monkeypatch.setenv(pl.FLAG_ENV, "agg_pipeline.v3")
    assert pl.pipeline_spec().version == pl.PIPELINE_V3
    assert pl.pipeline_spec(pl.PIPELINE_V2).version == pl.PIPELINE_V2   # explicit wins


def test_v3_composes_v2_plus_harvest_plus_canonical_v3():
    v2, v3 = pl.pipeline_spec(pl.PIPELINE_V2), pl.pipeline_spec(pl.PIPELINE_V3)
    # everything but the two AGG-1.15 pieces is v2's
    assert (v3.l3_period_version, v3.l4_stage, v3.canonical_guards) == (v2.l3_period_version, v2.l4_stage, v2.canonical_guards)
    assert v3.l3_kwargs({"D": "2022"}) == v2.l3_kwargs({"D": "2022"})
    assert v3.l3_kwargs({"D": "2022"}, fy_end=True) == v2.l3_kwargs({"D": "2022"}, fy_end=True)
    assert pl.l4_stage_for(v3) == "src.aggregation.facts_typed"
    assert v3.l4_slot_gate == ft.KEEP_SLOT_HARVEST and v3.l4_kwargs() == {"harvest": True}
    assert v3.canonical_kwargs() == {"rule": CANONICAL_RULE_V3, "guards": ("attributable_net_income",)}
    assert v3.as_record() == {"pipeline": pl.PIPELINE_V3, "l3_period": fd.L3_PERIOD_V2, "l4_stage": pl.L4_TYPED_V1,
                              "canonical_rule": CANONICAL_RULE_V3, "canonical_guards": ["attributable_net_income"],
                              "l4_slot_gate": ft.KEEP_SLOT_HARVEST}


def test_v1_and_v2_specs_are_unchanged():
    v1, v2 = pl.pipeline_spec(pl.PIPELINE_V1), pl.pipeline_spec(pl.PIPELINE_V2)
    assert v1.l4_kwargs() == {} and v2.l4_kwargs() == {}
    assert v1.as_record() == {"pipeline": pl.PIPELINE_V1, "l3_period": "legacy", "l4_stage": pl.L4_ROWS_V1,
                              "canonical_rule": CANONICAL_RULE, "canonical_guards": []}
    assert v2.as_record() == {"pipeline": pl.PIPELINE_V2, "l3_period": fd.L3_PERIOD_V2, "l4_stage": pl.L4_TYPED_V1,
                              "canonical_rule": CANONICAL_RULE_V2, "canonical_guards": ["attributable_net_income"]}
    assert v2.canonical_kwargs() == {"rule": CANONICAL_RULE_V2, "guards": ("attributable_net_income",)}
    assert v1.canonical_kwargs() == {}


class _Echo:
    """Every window's reply carries a long_term_debt row (an UNPLANNED slot under v2 / v3) and a revenue row."""

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        ex = prompt.split("Excerpt:\n", 1)[1]
        body = {s: [] for s in ft.SLOTS}
        if "Total assets 27,165" in ex:
            body["long_term_debt"] = [{"period": "2022", "value": 27165.0, "unit": "USD_millions",
                                       "quote": "Total assets 27,165", "confidence": "high"}]
        return {"text": json.dumps(body), "prompt_tokens": 1, "cache_n": 0, "decode_tokens": 1, "finish_reason": "stop"}


def _run(version):
    from tests.unit.test_aggregation_pipeline import CID, DESCS, _two_chunk_store
    spec = pl.pipeline_spec(version)
    st = _two_chunk_store()
    out = ft.TypedBackfill(st, _Echo()).fill_doc(CID, ft.split_windows(st, CID), ft.SCHEMA_FIELDS, DESCS, **spec.l4_kwargs())
    canon = spec.refresh_canonical(st, CID)
    facts = sorted((f["field"], f["period"], f["value"], f["fact_id"]) for f in st.facts(CID))
    return out, canon, facts, st


def test_v3_fill_doc_harvests_the_unplanned_slot_and_v2_does_not():
    out2, c2, f2, _ = _run(pl.PIPELINE_V2)
    out3, c3, f3, _ = _run(pl.PIPELINE_V3)
    assert out2["n_calls"] == out3["n_calls"]                        # zero extra calls
    assert not [f for f in f2 if f[0] == "long_term_debt"]
    assert [f for f in f3 if f[0] == "long_term_debt"]
    assert set(f2) <= set(f3)                                        # the harvest only adds rows
    assert c2["rule"] == CANONICAL_RULE_V2 and c3["rule"] == CANONICAL_RULE_V3


def test_v2_output_is_identical_to_the_pre_v3_explicit_call():
    """v2 through the spec == the explicit pre-AGG-1.16 call (no harvest keyword, canonical.v2 + attributable guard)."""
    from tests.unit.test_aggregation_pipeline import CID, DESCS, _two_chunk_store
    _out, _c, via_spec, st_a = _run(pl.PIPELINE_V2)
    st_b = _two_chunk_store()
    ft.TypedBackfill(st_b, _Echo()).fill_doc(CID, ft.split_windows(st_b, CID), ft.SCHEMA_FIELDS, DESCS)
    st_b.refresh_canonical(CID, rule=CANONICAL_RULE_V2, guards=("attributable_net_income",))
    explicit = sorted((f["field"], f["period"], f["value"], f["fact_id"]) for f in st_b.facts(CID))
    assert via_spec == explicit
    assert st_a.canonical_facts(CID) == st_b.canonical_facts(CID)


def test_v3_canonical_applies_the_long_term_debt_label_guard_through_the_spec():
    st = CorpusStore()
    rows = [{"corpus_id": "c", "fact_id": fid, "doc_hash": "h", "doc_id": "AMD", "entity": "AMD", "period": "FY2022",
             "field": "long_term_debt", "value": v, "unit": "USD_millions", "quote": q, "page": 1, "confidence": "high",
             "extractor": "l4_llm.v1", "audit_status": "unaudited"}
            for fid, v, q in (("b", 2500.0, "total debt principal ... was $2.5 billion"), ("c", 2467.0, "Long-term debt, net 2,467"))]
    kw2, kw3 = pl.pipeline_spec(pl.PIPELINE_V2).canonical_kwargs(), pl.pipeline_spec(pl.PIPELINE_V3).canonical_kwargs()
    assert select_canonical(rows, **kw2)[0][0]["value"] == 2500.0
    assert select_canonical(rows, **kw3)[0][0]["value"] == 2467.0
