"""AGG-1.18 (2026-11-16): the opt-in quote rule ``value_in_quote.rounded.v1`` and ``agg_pipeline.v4`` = ``agg_pipeline.v3`` + the four
accepted AGG-1.17 text-row components + that rule. OPT-IN: the default stays v3; v1 / v2 / v3 specs, kwargs and records are unchanged.
No model, no network.
"""

from __future__ import annotations

import pytest

from src.aggregation import FLAG_ENV as LAYER_FLAG
from src.aggregation import facts_det as fd
from src.aggregation import facts_typed as ft
from src.aggregation import pipeline as pl
from src.aggregation.store import CANONICAL_RULE_V3


@pytest.fixture(autouse=True)
def _layer_on(monkeypatch):
    monkeypatch.setenv(LAYER_FLAG, "1")
    monkeypatch.delenv(pl.FLAG_ENV, raising=False)


def _f(value, quote, decl):
    return {"field": "long_term_debt", "period": "FY2022", "value": value, "unit": "USD_millions", "quote": quote,
            "declared_scale": decl}


BLOCK = _f(4109.83, "4,109,829", "USD_thousands")
JNJ = _f(13895.3, "Cash dividends paid were $4.45 per share in fiscal year 2022", "USD_millions")


# ---------------------------------------------------------------- Task 1: the rule
def test_block_rounded_restatement_is_dropped_by_v1_and_kept_by_the_rounded_rule():
    assert ft.value_in_quote(BLOCK) is None                                         # the AGG-1.14 rule, unchanged
    assert ft.value_in_quote(BLOCK, ft.VALUE_IN_QUOTE_V1) is None
    assert ft.value_in_quote(BLOCK, ft.VALUE_IN_QUOTE_ROUNDED) == "restated_rounded"
    kept, dropped = ft.drop_unquoted([BLOCK], rule=ft.VALUE_IN_QUOTE_ROUNDED)
    assert kept == [BLOCK] and dropped == []
    assert ft.drop_unquoted([BLOCK])[0] == []


@pytest.mark.parametrize("decl", [None, "USD_millions", "USD_thousands", "USD_billions", "USD"])
def test_jnj_per_share_row_stays_dropped_under_the_rounded_rule(decl):
    f = dict(JNJ, declared_scale=decl)
    assert ft.value_in_quote(f, ft.VALUE_IN_QUOTE_ROUNDED) is None
    assert ft.drop_unquoted([f], rule=ft.VALUE_IN_QUOTE_ROUNDED)[0] == []


def test_rounded_rule_bounds():
    # 1 and 2 dropped decimals pass, 3 do not; half-up and half-even both accepted
    assert ft.value_in_quote(_f(4109.8, "4,109,829", "USD_thousands"), ft.VALUE_IN_QUOTE_ROUNDED) == "restated_rounded"
    assert ft.value_in_quote(_f(4110.0, "4,109,829", "USD_thousands"), ft.VALUE_IN_QUOTE_ROUNDED) is None
    assert ft.value_in_quote(_f(4109.84, "4,109,829", "USD_thousands"), ft.VALUE_IN_QUOTE_ROUNDED) is None   # not the rounding
    assert ft.value_in_quote(_f(-4109.83, "(4,109,829)", "USD_thousands"), ft.VALUE_IN_QUOTE_ROUNDED) == "restated_rounded"
    # relative-error bound: 1.235 thousands -> 0.001235 millions rounded to 0.0012 is 2.8% off
    assert ft.value_in_quote(_f(0.0012, "1,235", "USD_thousands"), ft.VALUE_IN_QUOTE_ROUNDED) is None
    # no declared scale / millions declared: nothing restated, nothing new accepted
    assert ft.value_in_quote(_f(4109.83, "4,109,829", None), ft.VALUE_IN_QUOTE_ROUNDED) is None
    assert ft.value_in_quote(_f(4109.83, "4,109.829", "USD_millions"), ft.VALUE_IN_QUOTE_ROUNDED) is None
    # what v1 accepts, the rounded rule accepts with the same basis
    assert ft.value_in_quote(_f(4109.829, "4,109,829", "USD_thousands"), ft.VALUE_IN_QUOTE_ROUNDED) == "restated"
    assert ft.value_in_quote(_f(4109829.0, "4,109,829", "USD_thousands"), ft.VALUE_IN_QUOTE_ROUNDED) == "printed"
    with pytest.raises(ValueError):
        ft.value_in_quote(BLOCK, "value_in_quote.v9")


# ---------------------------------------------------------------- Task 2: agg_pipeline.v4
def test_v4_is_selectable_and_not_the_default(monkeypatch):
    assert pl.PIPELINES[:4] == (pl.PIPELINE_V1, pl.PIPELINE_V2, pl.PIPELINE_V3, pl.PIPELINE_V4) and pl.DEFAULT == pl.PIPELINE_V3
    assert pl.pipeline_spec().version == pl.PIPELINE_V3
    monkeypatch.setenv(pl.FLAG_ENV, "agg_pipeline.v4")
    assert pl.pipeline_spec().version == pl.PIPELINE_V4
    assert pl.pipeline_spec(pl.PIPELINE_V3).version == pl.PIPELINE_V3


def test_v4_composes_v3_plus_textrow_plus_rounded_quote_rule():
    v3, v4 = pl.pipeline_spec(pl.PIPELINE_V3), pl.pipeline_spec(pl.PIPELINE_V4)
    assert (v4.l3_period_version, v4.l4_stage, v4.canonical_rule, v4.canonical_guards, v4.l4_slot_gate) == (
        v3.l3_period_version, v3.l4_stage, v3.canonical_rule, v3.canonical_guards, v3.l4_slot_gate)
    assert v4.canonical_kwargs() == v3.canonical_kwargs() == {"rule": CANONICAL_RULE_V3, "guards": ("attributable_net_income",)}
    assert pl.V4_L3_TEXTROW == ("ltd_qualifier", "note_ref", "tail_label", "pipe_cells_ltd")
    assert set(pl.V4_L3_TEXTROW) <= set(fd.L3_TEXTROW_OPTIONAL)
    comps = list(fd.L3_PERIOD_COMPONENTS) + list(pl.V4_L3_TEXTROW)
    assert v4.l3_kwargs({"D": "2022"}) == dict(v3.l3_kwargs({"D": "2022"}), components=comps)
    assert v4.l3_kwargs({"D": "2022"}, fy_end=True) == dict(v3.l3_kwargs({"D": "2022"}, fy_end=True), components=comps)
    assert v4.l4_kwargs() == {"harvest": True, "quote_rule": ft.VALUE_IN_QUOTE_ROUNDED}
    assert v4.as_record() == dict(v3.as_record(), pipeline=pl.PIPELINE_V4, l3_textrow=list(pl.V4_L3_TEXTROW),
                                  l4_quote_rule=ft.VALUE_IN_QUOTE_ROUNDED)
    fd._components(fd.L3_PERIOD_V2, comps)        # the components are legal under l3_period.v2


def test_v1_v2_v3_kwargs_and_records_are_unchanged():
    v1, v2, v3 = (pl.pipeline_spec(v) for v in (pl.PIPELINE_V1, pl.PIPELINE_V2, pl.PIPELINE_V3))
    assert v1.l4_kwargs() == {} and v2.l4_kwargs() == {} and v3.l4_kwargs() == {"harvest": True}
    for s in (v1, v2, v3):
        assert "components" not in s.l3_kwargs({"D": "2022"}) and "components" not in s.l3_kwargs({"D": "2022"}, fy_end=True)
        assert "l3_textrow" not in s.as_record() and "l4_quote_rule" not in s.as_record()
        assert s.l4_quote_rule == ft.VALUE_IN_QUOTE_V1 and s.l3_textrow == ()


def test_v4_fill_doc_runs_the_rounded_rule_and_v3_does_not():
    import json  # noqa: PLC0415

    from tests.unit.test_aggregation_pipeline import CID, DESCS, _two_chunk_store  # noqa: PLC0415

    class _Echo:
        def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
            body = {s: [] for s in ft.SLOTS}
            if "Total assets 27,165" in prompt:
                body["total_assets"] = [{"period": "2022", "value": 27.17, "unit": "USD_millions",
                                         "quote": "Total assets 27,165", "confidence": "high"}]
            return {"text": json.dumps(body), "prompt_tokens": 1, "cache_n": 0, "decode_tokens": 1, "finish_reason": "stop"}

    outs = {}
    for v in (pl.PIPELINE_V3, pl.PIPELINE_V4):
        st = _two_chunk_store()
        tb = ft.TypedBackfill(st, _Echo())
        orig = tb._declared_from_parent_chunk

        def _decl(facts, _o=orig):
            _o(facts)
            for f in facts:
                f["declared_scale"] = "USD_thousands"
        tb._declared_from_parent_chunk = _decl
        outs[v] = tb.fill_doc(CID, ft.split_windows(st, CID), ft.SCHEMA_FIELDS, DESCS, **pl.pipeline_spec(v).l4_kwargs())
    assert [d["value"] for d in outs[pl.PIPELINE_V3]["quote_dropped"]] == [27.17]
    assert outs[pl.PIPELINE_V4]["quote_dropped"] == []
