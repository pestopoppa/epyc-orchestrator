"""AGG-1.12 pipeline version flag (``agg_pipeline.v1`` default, ``agg_pipeline.v2`` opt-in) and the typed L4 stage.

No model, no server, no PDF: a fake backend and an in-memory store. The v1 tests are the byte-reproducibility contract for
Phase 0 and AGG-1.1: with the flag unset every helper makes EXACTLY the legacy call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.aggregation import FLAG_ENV as LAYER_FLAG
from src.aggregation import facts_det as fd
from src.aggregation import facts_llm, facts_typed as ft, pipeline as pl, registry
from src.aggregation.store import CANONICAL_RULE, CANONICAL_RULE_V2, CorpusStore, select_canonical
from src.models.document import DocumentPreprocessResult, Section, TableRef

CID = "c"
DOC = "BOEING_2022_10K"
# the amended L4 stage (AGG-1.3/1.4 amendment 2): the file v1 runs. The typed stage and the pipeline switch must not edit it.
FACTS_LLM_SHA = "c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451"


@pytest.fixture(autouse=True)
def _layer_on(monkeypatch):
    monkeypatch.setenv(LAYER_FLAG, "1")
    monkeypatch.delenv(pl.FLAG_ENV, raising=False)


# ---------------------------------------------------------------- the flag
def test_default_is_v1_and_env_selects_v2(monkeypatch):
    assert pl.pipeline_spec().version == pl.PIPELINE_V1
    monkeypatch.setenv(pl.FLAG_ENV, pl.PIPELINE_V2)
    assert pl.pipeline_spec().version == pl.PIPELINE_V2
    assert pl.pipeline_spec(pl.PIPELINE_V1).version == pl.PIPELINE_V1       # an explicit version wins over the env
    monkeypatch.setenv(pl.FLAG_ENV, "")
    assert pl.pipeline_spec().version == pl.PIPELINE_V1


def test_unknown_version_is_refused_never_defaulted(monkeypatch):
    with pytest.raises(ValueError):
        pl.pipeline_spec("agg_pipeline.v9")
    monkeypatch.setenv(pl.FLAG_ENV, "agg_pipeline.v3")
    with pytest.raises(ValueError):
        pl.pipeline_spec()


def test_v1_spec_is_the_legacy_call_with_no_keyword():
    s = pl.pipeline_spec(pl.PIPELINE_V1)
    assert s.l3_kwargs({"D": "2022"}) == {} and s.canonical_kwargs() == {}
    assert (s.l3_period_version, s.canonical_rule, s.canonical_guards, s.l4_stage) == (None, CANONICAL_RULE, None, pl.L4_ROWS_V1)
    assert pl.l4_stage_for(s) == "src.aggregation.facts_llm"


def test_v2_spec_is_l3_period_v2_typed_l4_and_canonical_v2_attributable_only():
    s = pl.pipeline_spec(pl.PIPELINE_V2)
    assert s.l3_kwargs({"D": "2022"}) == {"period_version": fd.L3_PERIOD_V2, "fiscal_years": {"D": "2022"}}
    assert s.canonical_kwargs() == {"rule": CANONICAL_RULE_V2, "guards": ("attributable_net_income",)}
    assert s.l4_stage == pl.L4_TYPED_V1 and pl.l4_stage_for(s) == "src.aggregation.facts_typed"
    assert "thousands_unit" not in s.canonical_guards      # measured 0 fixes / 0 breaks: not adopted
    assert s.as_record()["pipeline"] == pl.PIPELINE_V2


def test_the_v1_l4_stage_file_is_untouched():
    p = Path(facts_llm.__file__)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == FACTS_LLM_SHA


# ---------------------------------------------------------------- L3 and canonical: v1 byte-reproducible, v2 opt-in
def _doc():
    s = Section(id="s1", title="Consolidated Balance Sheets", level=1, content="Total assets $137,100 $138,552",
                page_start=57, page_end=57)
    t = TableRef(id="t1", page=57, caption="(Dollars in millions)", section_id="s1",
                 rows=[["Cash $14,614", "$8,052"], ["Total assets $137,100", "$138,552"]])
    return DocumentPreprocessResult(original_path="/x/b.pdf", sections=[s], figures=[], total_pages=100, tables=[t])


def _store() -> CorpusStore:
    st = CorpusStore()
    st.ingest(_doc(), corpus_id=CID, doc_id=DOC, entity="BOEING")
    return st


def test_v1_l3_is_byte_identical_to_the_legacy_call():
    st = _store()
    legacy = fd.extract_corpus_facts(st, CID)
    v1 = pl.pipeline_spec(pl.PIPELINE_V1).extract_l3(st, CID, {DOC: "2022"})      # a fiscal year alone changes nothing
    assert json.dumps(v1, sort_keys=True, default=str) == json.dumps(legacy, sort_keys=True, default=str)
    assert [f["value"] for f in legacy if f["field"] == "total_assets"] == [138552.0]       # the legacy 'other column' defect


def test_v2_l3_attributes_the_first_column():
    v2 = pl.pipeline_spec(pl.PIPELINE_V2).extract_l3(_store(), CID, {DOC: "2022"})
    ta = [f for f in v2 if f["field"] == "total_assets"]
    assert [(f["value"], f["period"]) for f in ta] == [(137100.0, "2022")]


def _fact(fid, field, value, period="2022", quote="q", extractor="l4_llm.v1", doc="D", tier_conf="high", unit="USD_millions"):
    return {"corpus_id": CID, "fact_id": fid, "doc_hash": "h", "doc_id": doc, "entity": "Boeing", "period": period, "field": field,
            "value": value, "unit": unit, "quote": quote, "page": 1, "confidence": tier_conf, "extractor": extractor,
            "audit_status": "unaudited"}


def test_v1_refresh_canonical_is_the_default_call_and_v2_applies_the_attributable_guard():
    st = CorpusStore()
    st.ingest(_doc(), corpus_id=CID, doc_id="D", entity="E")
    h = st.conn.execute("SELECT doc_hash FROM documents").fetchone()[0]
    rows = [dict(_fact("a", "net_income", 5200.0, quote="Net income 5,200", extractor="l3_textrow.v1"), doc_hash=h),
            dict(_fact("b", "net_income", 5000.0, quote="Net income attributable to Boeing 5,000"), doc_hash=h)]
    for r in rows:
        r.pop("corpus_id")
    st.write_facts(rows)
    legacy = select_canonical([dict(r, corpus_id=CID) for r in rows])[0]
    out1 = pl.pipeline_spec(pl.PIPELINE_V1).refresh_canonical(st, CID)
    assert out1["rule"] == CANONICAL_RULE and out1["guards"] is None
    assert [r["value"] for r in st.canonical_facts(CID)] == [r["value"] for r in legacy] == [5200.0]   # v1: the l3 tier wins
    out2 = pl.pipeline_spec(pl.PIPELINE_V2).refresh_canonical(st, CID)
    assert out2["rule"] == CANONICAL_RULE_V2 and out2["guards"] == ["attributable_net_income"]
    assert [r["value"] for r in st.canonical_facts(CID)] == [5000.0]


def test_registry_default_and_v1_pipeline_build_the_same_l3_and_v2_differs():
    a = registry.IndexRegistry(_store(), persist=False)
    b = registry.IndexRegistry(_store(), persist=False, pipeline=pl.pipeline_spec(pl.PIPELINE_V1), fiscal_years={DOC: "2022"})
    c = registry.IndexRegistry(_store(), persist=False, pipeline=pl.pipeline_spec(pl.PIPELINE_V2), fiscal_years={DOC: "2022"})
    fa, fb, fc = (r.get_or_build(CID, "l3_facts_det") for r in (a, b, c))
    assert json.dumps(fa, sort_keys=True, default=str) == json.dumps(fb, sort_keys=True, default=str)
    assert json.dumps(fa, sort_keys=True, default=str) != json.dumps(fc, sort_keys=True, default=str)


# ---------------------------------------------------------------- the typed stage
class TypedBackend:
    """Replies with a typed object whose ``total_assets`` row quotes the first number found in the excerpt."""

    def __init__(self, text=None, finish="stop"):
        self.calls, self.prompts, self.grammars, self.text, self.finish = 0, [], [], text, finish

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        self.calls += 1
        self.prompts.append(prompt)
        self.grammars.append((grammar, n_predict, cache_prompt, id_slot))
        body = self.text if self.text is not None else json.dumps({s: [] for s in ft.SLOTS})
        return {"text": body, "prompt_tokens": 1000, "cache_n": 10, "decode_tokens": 50, "finish_reason": self.finish}


DESCS = {s: f"desc of {s} FY2022" for s in ft.SLOTS}
WINDOW = {"chunk_id": "h:0001#w0", "doc_hash": "h", "doc_id": "D", "entity": "E", "page_start": 4,
          "text": "CONSOLIDATED BALANCE SHEETS\n(in millions)\nTotal assets 27,165 27,241\nTotal revenues 100 90\n"}


def _typed(rows_by_slot):
    return json.dumps({s: rows_by_slot.get(s, []) for s in ft.SLOTS})


def test_grammar_and_prompt_shape():
    g = ft.typed_grammar()
    for s in ft.SLOTS:
        assert f'"\\"{s}\\":" ws rows' in g
    assert "pchar{1,20}" in g and "qchar{1,100}" in g and 'rows ::= "[" ws ( row ( "," ws row )? )? ws "]"' in g
    p = ft.typed_prompt("EXCERPT", DESCS)
    assert p.endswith("Excerpt:\nEXCERPT") and all(f"- {s}: desc of {s} FY2022" in p for s in ft.SLOTS)
    assert "at most 2 rows" in p and "at most 100 characters" in p
    # the stable prefix (instructions + schema) comes first, so cache_prompt reuses it across windows
    assert ft.typed_prompt("A", DESCS).split("Excerpt:\n")[0] == ft.typed_prompt("B", DESCS).split("Excerpt:\n")[0]


def test_parse_typed_requires_a_list_for_every_slot():
    assert ft.parse_typed(_typed({})) == {s: [] for s in ft.SLOTS}
    assert ft.parse_typed("not json") is None and ft.parse_typed("[]") is None
    bad = json.loads(_typed({}))
    bad["revenue"] = "x"
    assert ft.parse_typed(json.dumps(bad)) is None
    del bad["revenue"]
    assert ft.parse_typed(json.dumps(bad)) is None


def test_call_window_builds_facts_through_the_unchanged_l4b_construction():
    st = _store()
    row = {"period": "2022", "value": 27165, "unit": "USD_millions", "quote": "Total assets 27,165", "confidence": "high"}
    bad_quote = {"period": "2022", "value": 1, "unit": "USD_millions", "quote": "not in the window", "confidence": "high"}
    be = TypedBackend(_typed({"total_assets": [row], "revenue": [bad_quote]}))
    r = ft.TypedBackfill(st, be).call_window(WINDOW, DESCS)
    assert be.calls == 1 and be.grammars[0][1:] == (ft.N_PREDICT, True, None)
    assert r.parsed and not r.cut and not r.bad
    assert [(f["field"], f["value"], f["extractor"]) for f in r.facts] == [("total_assets", 27165.0, "l4_llm.v1")]
    assert r.facts[0]["declared_scale"] == "USD_millions" and r.facts[0]["chunk_id"] == WINDOW["chunk_id"]


def test_a_cut_or_unparseable_reply_is_bad_and_yields_no_facts():
    st = _store()
    r1 = ft.TypedBackfill(st, TypedBackend("{", finish="length")).call_window(WINDOW, DESCS)
    assert r1.bad and r1.cut and not r1.parsed and r1.facts == []
    r2 = ft.TypedBackfill(st, TypedBackend("{}")).call_window(WINDOW, DESCS)
    assert r2.bad and not r2.cut and not r2.parsed


def test_keep_slot_gates_a_windows_rows_to_the_fields_that_chose_it():
    f = {"field": "revenue"}
    assert ft.keep_slot(f, {"revenue", "capex"}) and not ft.keep_slot(f, {"capex"})


def _two_chunk_store():
    long_a = "\n".join(f"filler line {i} about nothing in particular" for i in range(40))
    secs = [Section(id="s1", title="Income", level=1, content="Total revenues 100 90\nNet income 10 9\n" + long_a,
                    page_start=1, page_end=1),
            Section(id="s2", title="Balance", level=1, content="Total assets 27,165 27,241\n" + long_a, page_start=2, page_end=2)]
    st = CorpusStore()
    st.ingest(DocumentPreprocessResult(original_path="/x/d.pdf", sections=secs, figures=[], total_pages=2, tables=[]),
              corpus_id=CID, doc_id="D", entity="E")
    return st


def test_windows_union_candidates_and_the_doc_call_plan_dedupe_windows():
    st = _two_chunk_store()
    wins = ft.split_windows(st, CID, 10_000)
    assert wins and all(len(w["text"]) <= 10_000 for w in wins) and "".join(w["text"] for w in wins)
    cands = ft.union_candidates(wins, "total_assets", DESCS["total_assets"])
    assert len({w["chunk_id"] for w in cands}) == len(cands) >= 1
    plan, chosen = ft.doc_call_plan(wins, ft.SCHEMA_FIELDS, DESCS)
    assert len(plan) == len({w["chunk_id"] for w in plan}) <= len(wins)          # one call per distinct window
    assert set(chosen) == {w["chunk_id"] for w in plan} and all(chosen.values())


def test_fill_doc_end_to_end_gates_slots_reconciles_writes_and_records_layers():
    st = _two_chunk_store()
    wins = ft.split_windows(st, CID)
    quote_for = {"total_assets": "Total assets 27,165", "revenue": "Total revenues 100"}

    class Echo:
        calls = 0

        def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
            Echo.calls += 1
            ex = prompt.split("Excerpt:\n", 1)[1]
            body = {s: [] for s in ft.SLOTS}
            for slot, q in quote_for.items():
                if q in ex:
                    v = float(q.rsplit(" ", 1)[1].replace(",", ""))
                    body[slot] = [{"period": "2022", "value": v, "unit": "USD_millions", "quote": q, "confidence": "high"}]
            return {"text": json.dumps(body), "prompt_tokens": 500, "cache_n": 0, "decode_tokens": 20, "finish_reason": "stop"}

    seen = []
    out = ft.TypedBackfill(st, Echo()).fill_doc(CID, wins, ft.SCHEMA_FIELDS, DESCS, on_window=seen.append)
    plan, _ = ft.doc_call_plan(wins, ft.SCHEMA_FIELDS, DESCS)
    assert out["n_calls"] == Echo.calls == len(plan) == len(seen)
    got = {(f["field"], f["value"]) for f in st.facts(CID, extractor="l4_llm.v1")}
    assert ("total_assets", 27165.0) in got and ("revenue", 100.0) in got
    layers = {m["builder"] for m in st.manifest(CID) if m["layer"] == "L4"}
    assert {f"l4_facts_llm:{f}" for f in ft.SCHEMA_FIELDS} <= layers
    assert "L4" in st.built_layers(CID)


def test_typed_stage_leaves_the_v1_stage_alone():
    # nothing the v1 stage defines was replaced or wrapped
    assert facts_llm.EXTRACTOR_L4 == ft.EXTRACTOR == "l4_llm.v1"
    assert not hasattr(facts_llm, "TypedBackfill") and not hasattr(facts_llm, "typed_grammar")


# ---------------------------------------------------------------- AGG-1.14: the value-in-quote rule of the typed stage
JNJ_WINDOW = {"chunk_id": "h:0061#w0", "doc_hash": "h", "doc_id": "JNJ", "entity": "Johnson & Johnson", "page_start": 49,
              "text": ("CONSOLIDATED STATEMENTS OF CASH FLOWS\n(Dollars in millions)\n"
                        "Dividends to shareholders (11,682)\n"
                        "Cash dividends paid were $4.45 per share in fiscal year 2022\n"
                        "dividends to shareholders\n")}


def _row(value, quote, unit="USD_millions", period="FY2022"):
    return {"period": period, "value": value, "unit": unit, "quote": quote, "confidence": "high"}


def test_value_in_quote_drops_the_per_share_row_keeps_the_printed_one_and_drops_the_junk_row():
    be = TypedBackend(_typed({"dividends_paid": [_row(13895.3, "Cash dividends paid were $4.45 per share in fiscal year 2022"),
                                                 _row(11682, "Dividends to shareholders (11,682)")],
                              "revenue": [_row(11.7, "dividends to shareholders", unit="USD")]}))
    r = ft.TypedBackfill(_store(), be).call_window(JNJ_WINDOW, DESCS)
    assert len(r.facts) == 3                                       # window_facts keeps every row the model returned
    by = {(f["field"], f["value"]): ft.value_in_quote(f) for f in r.facts}
    assert by == {("dividends_paid", 13895.3): None, ("dividends_paid", 11682.0): "printed", ("revenue", 11.7): None}
    kept, dropped = ft.drop_unquoted(r.facts)
    assert [f["value"] for f in kept] == [11682.0]
    assert sorted(d["value"] for d in dropped) == [11.7, 13895.3]
    assert all(d["reason"] == "value_not_in_quote" and d["quote"] for d in dropped)


def test_value_in_quote_accepts_the_declared_scale_restatement_only():
    # printed 6.5 in a quote that names its own scale: the model restated to millions (6500) -> kept, basis "restated"
    f = {"value": 6500.0, "unit": "USD_millions", "quote": "capital expenditures of $6.5 billion", "declared_scale": "USD_billions"}
    assert ft.value_in_quote(f) == "restated"
    assert ft.value_in_quote({**f, "value": 6.5, "unit": "USD_billions"}) == "printed"
    # a thousands table: printed 11,682 (thousands) restated to 11.682 millions
    th = {"value": 11.682, "unit": "USD_millions", "quote": "Net income 11,682", "declared_scale": "USD_thousands"}
    assert ft.value_in_quote(th) == "restated"
    # the same number with no declared scale cannot be attested; a value that is neither printed nor its restatement is not backed
    assert ft.value_in_quote({**th, "declared_scale": None}) is None
    assert ft.value_in_quote({**f, "value": 6501.0}) is None
    assert ft.value_in_quote({**f, "value": None}) is None
    # a negative value is checked on its magnitude (the quote prints the parenthesised amount)
    assert ft.value_in_quote({"value": -11682.0, "quote": "Dividends to shareholders (11,682)", "declared_scale": "USD_millions"}) == "printed"


def test_value_in_quote_is_the_facts_llm_check_not_a_second_ladder():
    # the rule is built from facts_llm's own printed-number reader and matcher and agrees with unit_override's two bases
    f = {"value": 6500.0, "unit": "USD_millions", "quote": "capital expenditures of $6.5 billion", "declared_scale": "USD_billions"}
    assert facts_llm._matches_printed(f["value"], [n * facts_llm.UNIT_SCALE["USD_billions"] for n in facts_llm._printed_numbers(f["quote"])])
    src = Path(ft.__file__).read_text()
    assert "facts_llm._printed_numbers" in src and "facts_llm._matches_printed" in src and "re.compile" not in src


def test_finish_doc_drops_unquoted_rows_before_reconcile_and_never_stores_them():
    st = _store()
    tb = ft.TypedBackfill(st, TypedBackend())
    rows = {"dividends_paid": [_row(13895.3, "Cash dividends paid were $4.45 per share in fiscal year 2022"),
                               _row(11682, "Dividends to shareholders (11,682)")]}
    win = dict(JNJ_WINDOW, doc_hash=st.corpus_docs(CID)[0]["doc_hash"])           # the stored document's hash
    facts = ft.TypedBackfill(st, TypedBackend(_typed(rows))).call_window(win, DESCS).facts
    out = tb.finish_doc(CID, facts, ft.SCHEMA_FIELDS)
    assert [f["value"] for f in out["facts"]] == [11682.0] and out["conflicts"] == []        # no disputed pair left
    assert [d["value"] for d in out["quote_dropped"]] == [13895.3]
    assert [f["value"] for f in st.facts(CID, extractor="l4_llm.v1")] == [11682.0]


def test_fill_doc_applies_the_rule_end_to_end():
    st = _two_chunk_store()
    wins = ft.split_windows(st, CID)

    class Liar:
        def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
            ex = prompt.split("Excerpt:\n", 1)[1]
            body = {s: [] for s in ft.SLOTS}
            if "Total assets 27,165" in ex:
                body["total_assets"] = [_row(27165, "Total assets 27,165", period="2022"), _row(99999, "Total assets 27,165", period="2021")]
            return {"text": json.dumps(body), "prompt_tokens": 1, "cache_n": 0, "decode_tokens": 1, "finish_reason": "stop"}

    out = ft.TypedBackfill(st, Liar()).fill_doc(CID, wins, ft.SCHEMA_FIELDS, DESCS)
    assert {(f["period"], f["value"]) for f in st.facts(CID, extractor="l4_llm.v1")} == {("2022", 27165.0)}
    assert [d["value"] for d in out["quote_dropped"]] == [99999]
