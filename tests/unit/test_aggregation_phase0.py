"""AGG-0.2/0.3 unit tests: store, registry, L3 facts, SQL grammar, planner, lazy L4 (fake backend)."""

from __future__ import annotations

import json

import pytest

from src.aggregation import FLAG_ENV, AggLayerDisabled, agg_layer_enabled
from src.aggregation import facts_det, facts_llm, planner, sql_grammar
from src.aggregation.registry import IndexNotAvailable, IndexRegistry
from src.aggregation.store import CorpusStore
from src.models.document import DocumentPreprocessResult, Section, TableRef


def _doc(revenue: str = "26,000", extra: str = "") -> DocumentPreprocessResult:
    s1 = Section(id="s1", title="Results of Operations", level=1,
                 content="Net sales were $14,189 million in 2022 compared to 2021." + extra,
                 page_start=40, page_end=41)
    s2 = Section(id="s2", title="Consolidated Statements of Income", level=1,
                 content="Consolidated Statements of Income\n(in millions)\nTotal revenues 26,000",
                 page_start=60, page_end=60)
    table = TableRef(id="p60_t1", page=60, caption="(in millions)", section_id="s2",
                     rows=[["", "2022", "2021"], ["Total revenues", f"${revenue}", "$24,000"],
                           ["Net income", "(1,200)", "900"], ["Notes", "—", "—"]])
    return DocumentPreprocessResult(original_path="/x/a.pdf", sections=[s1, s2], figures=[],
                                    total_pages=100, tables=[table])


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(FLAG_ENV, raising=False)
    monkeypatch.delenv("ORCHESTRATOR_AGG_LAYER", raising=False)


def test_flag_default_off_blocks_behaviour(off):
    assert agg_layer_enabled() is False
    with pytest.raises(AggLayerDisabled):
        planner.plan("q", planner.PlannerFacts(corpus_tokens=10, n_docs=1))
    with pytest.raises(AggLayerDisabled):
        IndexRegistry(CorpusStore()).get_or_build("c", "l1_bm25")


def test_ingest_is_content_hashed_and_incremental(on, tmp_path):
    st = CorpusStore(tmp_path / "agg.sqlite")
    r1 = st.ingest(_doc(), corpus_id="c", doc_id="ACME_2022_10K", entity="ACME")
    assert r1.status == "ingested" and r1.n_chunks == 2 and r1.n_tables == 1
    assert st.ingest(_doc(), corpus_id="c", doc_id="ACME_2022_10K").status == "unchanged"
    h1 = st.corpus_content_hash("c")
    r3 = st.ingest(_doc(revenue="27,000"), corpus_id="c", doc_id="ACME_2022_10K", entity="ACME")
    assert r3.status == "replaced" and st.corpus_content_hash("c") != h1
    assert len(st.chunks("c")) == 2 and len(st.tables("c")) == 1


def test_registry_l1_l3_share_store_and_manifest(on):
    st = CorpusStore()
    st.ingest(_doc(), corpus_id="c", doc_id="ACME_2022_10K", entity="ACME")
    reg = IndexRegistry(st)
    bm = reg.get_or_build("c", "l1_bm25")
    assert bm.search("net sales")[0][0] == st.chunks("c")[0]["chunk_id"]
    assert reg.get_or_build("c", "l1_bm25") is bm  # not rebuilt
    reg.get_or_build("c", "l3_facts_det")
    assert st.built_layers("c") == {"L1", "L3"}
    with pytest.raises(IndexNotAvailable):
        reg.get_or_build("c", "l2_dense")


def _long_doc():
    """Three stored chunks, the middle one long, with a balance-sheet row deep inside it (AGG-1.5 shape)."""
    head = "Funding. We make contributions to pension trusts.\n" * 40
    row = "Total Current Assets 21,539 20,000\n"
    body = head + "Cash and cash equivalents 4,954 5,596\n" + row + "Total Assets 92,187 92,187\n" + head
    secs = [Section(id="s1", title="Overview", level=1, content="Intro text about the business.\n",
                    page_start=1, page_end=1),
            Section(id="s2", title="Funding", level=1, content=body, page_start=2, page_end=2),
            Section(id="s3", title="Tail", level=1, content="Signatures and exhibits.\n", page_start=3, page_end=3)]
    return DocumentPreprocessResult(original_path="/x/b.pdf", sections=secs, figures=[], total_pages=3, tables=[])


def test_l1_passage_index_matches_context_search_and_is_built_once(on):
    from src.aggregation.registry import CORPUS_TEXT_SEP, PassageHandle  # noqa: PLC0415
    from src.repl_environment.context_bundle import ContextBundle, Section  # noqa: PLC0415

    st = CorpusStore()
    st.ingest(_long_doc(), corpus_id="c", doc_id="LONG_2022_10K", entity="LONG")
    reg = IndexRegistry(st)
    h = reg.get_or_build("c", "l1_passage_bm25")
    assert isinstance(h, PassageHandle) and reg.get_or_build("c", "l1_passage_bm25") is h  # built once
    assert st.built_layers("c") == {"L1"}  # an L1 builder: no L2/L3/L4 row
    chunks = st.chunks("c")
    text = CORPUS_TEXT_SEP.join(c["text"] or "" for c in chunks)
    assert h.text == text
    bundle = ContextBundle([Section("c", text, "text", False, "x")], search_enabled=True)
    bundle.set_embedder(None)
    for q in ("Total Current Assets", "pension trusts contributions", "signatures exhibits", "no_such_token_zzz"):
        b = bundle.search(q, 8)
        p = h.search(q, 8)
        assert [(x["start"], x["end"], x["line"]) for x in p] == [(x["start"], x["end"], x["line"]) for x in b], q
        assert [round(x["score"], 12) for x in p] == [round(x["score"], 12) for x in b]
        for x in p:  # provenance: the stored chunk the passage starts in
            c = next(c for c in chunks if c["chunk_id"] == x["chunk_id"])
            assert h.span(x["start"], x["end"]).split("\n")[0] in text
            assert x["end"] - x["start"] <= 800
            assert text.find(c["text"] or "") <= x["start"]


def test_l1_passage_hit_shows_the_matching_text_not_the_chunk_head(on):
    st = CorpusStore()
    st.ingest(_long_doc(), corpus_id="c", doc_id="LONG_2022_10K", entity="LONG")
    h = IndexRegistry(st).get_or_build("c", "l1_passage_bm25")
    top = h.search("Total Current Assets", 3)[0]
    assert "Total Current Assets 21,539" in h.span(top["start"], top["end"])
    stored = next(c for c in st.chunks("c") if c["chunk_id"] == top["chunk_id"])
    assert "Total Current Assets" not in (stored["text"] or "")[:200]  # the whole-chunk snippet would miss it
    with pytest.raises(ValueError):
        h.search("   ")


def test_l3_table_and_regex_facts_with_provenance(on):
    st = CorpusStore()
    st.ingest(_doc(), corpus_id="c", doc_id="ACME_2022_10K", entity="ACME")
    facts = facts_det.extract_corpus_facts(st, "c")
    tab = {(f["field"], f["period"]): f for f in facts if f["extractor"] == facts_det.EXTRACTOR_TABLE}
    assert tab[("revenue", "2022")]["value"] == 26000.0
    assert tab[("revenue", "2021")]["value"] == 24000.0
    assert tab[("net_income", "2022")]["value"] == -1200.0
    rev = tab[("revenue", "2022")]
    assert rev["unit"] == "USD_millions" and rev["page"] == 60 and rev["doc_hash"]
    assert rev["quote_start"] is not None and rev["line_start"] == 4  # title line + 3
    rx = [f for f in facts if f["extractor"] == facts_det.EXTRACTOR_REGEX]
    assert rx and rx[0]["field"] == "revenue" and rx[0]["value"] == 14189.0
    assert rx[0]["period"] == "2022" and rx[0]["unit"] == "USD_millions"
    chunk = st.chunks("c")[0]["text"]
    assert chunk[rx[0]["quote_start"]:rx[0]["quote_end"]] == rx[0]["quote"]
    gold = [{"doc_id": "ACME_2022_10K", "field": "revenue", "period": "2022", "value": 26000},
            {"doc_id": "ACME_2022_10K", "field": "capex", "period": "2022", "value": 5}]
    assert facts_det.recall(facts, gold) == 0.5


def test_parse_number():
    assert facts_det.parse_number("(1,234)") == -1234.0
    assert facts_det.parse_number("$5,363") == 5363.0
    assert facts_det.parse_number("12.5%") == 12.5
    assert facts_det.parse_number("—") is None
    assert facts_det.parse_number("Total") is None


def test_sql_grammar_rejects_out_of_catalog():
    cat = sql_grammar.Catalog(columns=("entity", "period", "field", "value"), fields=("revenue",))
    g = sql_grammar.build_sql_gbnf(cat)
    assert '"revenue"' in g and "capex" not in g and "DROP" not in g
    ok = "SELECT entity, value FROM facts WHERE field = 'revenue' AND period = '2022' ORDER BY value DESC LIMIT 3"
    assert sql_grammar.validate_sql(ok, cat) == (True, "ok")
    assert not sql_grammar.validate_sql("SELECT entity FROM facts WHERE field = 'capex'", cat)[0]
    assert not sql_grammar.validate_sql("SELECT salary FROM facts", cat)[0]
    assert not sql_grammar.validate_sql("SELECT entity FROM documents", cat)[0]
    assert not sql_grammar.validate_sql("SELECT entity FROM facts; DROP TABLE facts", cat)[0]


def test_planner_size_rules_and_stub_band(on):
    small = planner.plan("What was revenue?", planner.PlannerFacts(corpus_tokens=40_000, n_docs=1))
    assert (small.path, small.llm_called, small.decided_by) == ("whole_doc", False, "size_rule")
    assert small.layers == ("L0",)
    stub = planner.plan("How many companies...?", planner.PlannerFacts(corpus_tokens=1_000_000 - 1, n_docs=10))
    assert stub.path == "retrieval" and stub.decided_by == "deterministic_default_stub"
    assert stub.receipt["schema"] == "decision_receipt.v1" and stub.receipt["stub"] is True
    assert planner.plan("q", planner.PlannerFacts(150_000, 1, questions_so_far=2)).path == "repl_nav"


def test_planner_typed_thresholds_with_injected_decider(on):
    seen = []

    def decide(state, questions):
        seen.append((json.loads(state), [q.id for q in questions]))
        return {"path": "retrieval", "needs_aggregation": 3}

    d = planner.plan("Sum capex across all filings", planner.PlannerFacts(1_200_000 - 1, 10), decide_fn=decide)
    assert d.path == "sql" and d.llm_called and "L4" in d.layers
    assert seen[0][1] == ["path", "needs_aggregation"]
    mid = planner.plan("x", planner.PlannerFacts(300_000, 3),
                       decide_fn=lambda s, q: {"path": "repl_nav", "needs_aggregation": 2})
    assert (mid.path, mid.decided_by) == ("repl_nav", "typed_choice")


class FakeBackend:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        self.calls.append({"grammar": grammar, "n_predict": n_predict,
                           "cache_prompt": cache_prompt, "id_slot": id_slot, "prompt": prompt})
        if grammar == facts_llm.GATE_GBNF:
            return {"text": "yes", "prompt_tokens": 50, "cache_n": 0}
        rows = [{"period": "2022", "value": 3.1, "unit": "USD_billions",
                 "quote": "Capital spending was $3.1 billion", "confidence": "medium"},
                {"period": "2022", "value": 9, "unit": "USD_millions",
                 "quote": "not in the chunk", "confidence": "high"}]
        return {"text": json.dumps(rows), "prompt_tokens": 400, "cache_n": 300}


def test_lazy_l4_end_to_end_fake_backend(on):
    st = CorpusStore()
    st.ingest(_doc(extra=" Capital spending was $3.1 billion in 2022."), corpus_id="c",
              doc_id="ACME_2022_10K", entity="ACME")
    chunk = st.chunks("c")[0]
    cands = [dict(chunk, sim=0.4), dict(st.chunks("c")[1], sim=0.1)]
    be = FakeBackend()
    bf = facts_llm.LazyFieldBackfill(st, be, n_predict=128, id_slot=1)
    out = bf.fill("c", "capex", "capital expenditure", cands)
    assert len(be.calls) == 2  # one gate (band), one extraction; low-sim chunk rejected
    ext = be.calls[1]
    assert ext["grammar"] == facts_llm.ROW_GBNF and ext["cache_prompt"] and ext["id_slot"] == 1
    assert ext["n_predict"] == 128 and ext["prompt"].startswith(facts_llm.extraction_prefix("capex", "capital expenditure"))
    assert len(out["facts"]) == 1 and out["facts"][0]["value"] == pytest.approx(3100.0)
    assert out["receipts"][1]["rejected"] == 1 and out["receipts"][1]["cache_n"] == 300
    assert out["audit_queue"]  # medium confidence -> audited
    assert "L4" in st.built_layers("c")
    bf.fill("c", "capex", "capital expenditure", cands)
    assert len(be.calls) == 2  # once per (field, chunk hash)


def test_reconcile_flags_conflicts():
    a = {"fact_id": "a", "doc_id": "d", "field": "f", "period": "2022", "value": 1.0,
         "unit": "USD_billions", "confidence": "high"}
    b = dict(a, fact_id="b", value=1000.0, unit="USD_millions", confidence="low")
    c = dict(a, fact_id="c", value=2.0)
    kept, conflicts = facts_llm.reconcile([a, b])
    assert len(kept) == 1 and not conflicts and kept[0]["confidence"] == "high"
    kept, conflicts = facts_llm.reconcile([a, c])
    assert len(conflicts) == 1 and all(r["audit_status"] == "disputed" for r in conflicts[0])


def test_small_corpus_path_builds_no_l2_l4_index(on):
    st = CorpusStore()
    st.ingest(_doc(), corpus_id="small", doc_id="ACME_2022_10K", entity="ACME")
    d = planner.plan("q", planner.PlannerFacts(st.corpus_tokens("small"), 1))
    assert d.path == "whole_doc" and not d.llm_called
    assert st.built_layers("small") == set()


# ----------------------------------------------------------------------------------------------
# AGG Phase 0 rerun fixes (2026-11-10): canonical facts, question scope in the planner.
def _fact(fid, doc, field, period, value, extractor, conf="high", unit="USD_millions", page=1,
          audit="unaudited"):
    return {"fact_id": fid, "doc_hash": f"h:{doc}", "doc_id": doc, "entity": doc.title(), "period": period,
            "field": field, "value": value, "unit": unit, "quote": f"{field} {value}",
            "quote_start": 0, "quote_end": 1, "chunk_id": None, "table_id": None, "page": page,
            "line_start": 1, "line_end": 1, "confidence": conf, "extractor": extractor,
            "audit_status": audit}


def _canon_store(facts):
    st = CorpusStore(":memory:")
    for d in sorted({f["doc_id"] for f in facts}):
        st.conn.execute("INSERT INTO documents (doc_hash, corpus_id, doc_id, entity, n_tokens_est) "
                        "VALUES (?,?,?,?,?)", [f"h:{d}", "c", d, d.title(), 1000])
    st.write_facts(facts)
    return st


def test_canonical_one_fact_per_entity_field_period(on):
    # General Mills-style: one doc, one field, FY2022 five L4 rows (3 agree, 2 conflict) + L3 '2022'
    facts = [
        _fact("a1", "gm", "revenue", "FY2022", 18992.8, "l4_llm.v1", page=83),
        _fact("a2", "gm", "revenue", "FY2022", 18992.8, "l4_llm.v1", page=20),
        _fact("a3", "gm", "revenue", "FY2022", 11572.0, "l4_llm.v1", page=21),
        _fact("a4", "gm", "revenue", "FY2022", 2134.3, "l4_llm.v1", page=55),
        _fact("a5", "gm", "revenue", "FY2022", 18992.8, "l4_llm.v1", page=42),
        _fact("b1", "gm", "revenue", "2022", 18992.8, "l3_textrow.v1", page=45),
        _fact("b2", "gm", "revenue", "2021", 18127.0, "l3_textrow.v1", page=45),
        _fact("c1", "vz", "revenue", "FY2022", 136835.0, "l4_llm.v1", page=102),
        _fact("c2", "vz", "revenue", "FY2022", 134578.0, "l4_llm.v1", page=101),
        _fact("c3", "vz", "revenue", "FY2022", 136835.0, "l4_llm.v1", page=54),
    ]
    st = _canon_store(facts)
    stats = st.refresh_canonical("c")
    rows = st.canonical_facts("c")
    keys = [(r["doc_id"], r["field"], r["period"]) for r in rows]
    assert len(keys) == len(set(keys)) == 3  # gm 2021, gm 2022, vz 2022: one each
    gm22 = next(r for r in rows if r["doc_id"] == "gm" and r["period"] == "2022")
    assert gm22["fact_id"] == "b1" and gm22["extractor"] == "l3_textrow.v1"  # tier beats L4
    assert gm22["n_candidates"] == 6 and gm22["n_conflicts"] == 2
    vz = next(r for r in rows if r["doc_id"] == "vz")
    assert vz["value"] == 136835.0 and vz["n_conflicts"] == 1
    log = st.conflicts("c")
    assert {r["kind"] for r in log} == {"conflict", "duplicate"}  # logged, never silently dropped
    assert stats["n_conflicts"] == 3 and stats["n_duplicates"] == 4
    assert stats["n_canonical"] == 3 and stats["n_candidates"] == len(facts)
    # the raw evidence table is untouched
    assert len(st.facts("c")) == len(facts)
    # SUM over the canonical table counts each entity once; over raw facts it does not
    total = st.conn.execute("SELECT SUM(value) FROM facts_canonical WHERE field = 'revenue' "
                            "AND period = '2022'").fetchone()[0]
    assert total == pytest.approx(18992.8 + 136835.0)
    raw = st.conn.execute("SELECT SUM(value) FROM facts WHERE field = 'revenue' AND period = 'FY2022'"
                          ).fetchone()[0]
    assert raw != pytest.approx(total)
    # idempotent rebuild
    assert st.refresh_canonical("c") == stats and len(st.canonical_facts("c")) == 3


def test_canonical_selection_order_is_total_and_documented():
    from src.aggregation.store import CANONICAL_RULE, select_canonical
    base = dict(doc_id="d", field="capex", period="2022")
    f = lambda fid, **k: {**_fact(fid, "d", "capex", "2022", 100.0, "l3_textrow.v1"), **k}  # noqa: E731
    # tier: table > textrow > l4 > regex
    pick = lambda *fs: select_canonical([{**x, "corpus_id": "c"} for x in fs])[0][0]["fact_id"]  # noqa: E731
    assert pick(f("z", extractor="l4_llm.v1"), f("y"), f("x", extractor="l3_table.v1")) == "x"
    assert pick(f("z", extractor="l3_regex.v1", confidence="high"), f("y", extractor="l4_llm.v1")) == "y"
    # confidence next, then audit, then magnitude (consolidated total), then page, then fact_id
    assert pick(f("a", confidence="medium", value=900.0), f("b", confidence="high", value=1.0)) == "b"
    assert pick(f("a", audit_status="disputed"), f("b", audit_status="audited_ok")) == "b"
    assert pick(f("a", value=10.0), f("b", value=-500.0)) == "b"
    assert pick(f("a", page=9), f("b", page=3)) == "b"
    assert pick(f("b", page=3), f("a", page=3)) == "a"
    # order of arrival never matters
    assert pick(f("a", value=1.0), f("b", value=2.0)) == pick(f("b", value=2.0), f("a", value=1.0))
    assert CANONICAL_RULE.startswith("canonical.v1")
    assert base  # keys documented above


def test_canonical_normalises_period_and_unit():
    from src.aggregation.store import canonical_period, select_canonical
    assert canonical_period("FY2022") == canonical_period("2022") == canonical_period("Dec 31, 2022") == "2022"
    assert canonical_period(None) is None and canonical_period("None") is None
    rows, _ = select_canonical([{**_fact("k", "d", "revenue", "FY2022", 3_000_000.0, "l4_llm.v1",
                                         unit="USD_thousands"), "corpus_id": "c"}])
    assert rows[0]["value"] == pytest.approx(3000.0) and rows[0]["unit"] == "USD_millions"


def test_canonical_outflow_fields_are_positive():
    from src.aggregation.store import select_canonical
    rows, _ = select_canonical([
        {**_fact("a", "d", "capex", "2022", -4551.0, "l3_textrow.v1"), "corpus_id": "c"},
        {**_fact("b", "d", "net_income", "2022", -546.0, "l3_textrow.v1"), "corpus_id": "c"}])
    by = {r["field"]: r["value"] for r in rows}
    assert by == {"capex": 4551.0, "net_income": -546.0}  # a net loss keeps its sign


def test_sql_grammar_queries_canonical_table(on):
    st = _canon_store([_fact("a", "d", "revenue", "FY2022", 5.0, "l4_llm.v1"),
                       _fact("b", "d", "revenue", "2022", 5.0, "l3_textrow.v1")])
    st.refresh_canonical("c")
    cat = sql_grammar.catalog_from_canonical(st.canonical_facts("c"))
    assert cat.table == "facts_canonical" and cat.fields == ("revenue",)
    ok, _ = sql_grammar.validate_sql("SELECT SUM(value) FROM facts_canonical WHERE field = 'revenue'", cat)
    assert ok
    bad, why = sql_grammar.validate_sql("SELECT SUM(value) FROM facts WHERE field = 'revenue'", cat)
    assert not bad and "table" in why  # the raw evidence table is not queryable on this path
    assert "facts_canonical" in sql_grammar.build_sql_gbnf(cat)


_DOCS = [
    {"doc_id": "AES_2022_10K", "entity": "AES Corporation", "n_tokens_est": 231_844},
    {"doc_id": "COCACOLA_2022_10K", "entity": "Coca-Cola", "n_tokens_est": 171_908},
    {"doc_id": "CORNING_2022_10K", "entity": "Corning", "n_tokens_est": 120_830},
    {"doc_id": "JPMORGAN_2022_10K", "entity": "JPMorgan", "n_tokens_est": 390_496},
    {"doc_id": "PAYPAL_2022_10K", "entity": "Paypal", "n_tokens_est": 129_382},
]


def test_planner_scope_single_document_follows_size_rules(on):
    # Corpus >= 1M tokens: before the fix every question took the sql path.
    big = [dict(d, n_tokens_est=d["n_tokens_est"] * 4 if d["doc_id"] != "CORNING_2022_10K"
                else d["n_tokens_est"]) for d in _DOCS]
    assert sum(d["n_tokens_est"] for d in big) >= planner.LARGE_C
    f, why = planner.scoped_facts("Does Corning have positive working capital based on FY2022 data?",
                                  big, sql_fields_present=True)
    assert why == "single_entity" and f.scope_doc == "CORNING_2022_10K" and f.n_docs == 1
    assert f.corpus_tokens == 120_830 and f.sql_fields_present is False
    assert planner.plan("q", f).path == "whole_doc"                       # <= 200k at Q = 1
    f2, _ = planner.scoped_facts("Does Corning have positive working capital?", _DOCS, questions_so_far=2)
    assert planner.plan("q", f2).path == "repl_nav"                       # <= 200k at Q > 1
    f3, _ = planner.scoped_facts("What is AES's FY2022 return on assets?", _DOCS)
    assert f3.corpus_tokens == 231_844 and planner.plan("q", f3).path == "retrieval"  # > 200k


def test_planner_scope_aggregation_and_ambiguous_stay_corpus_wide(on):
    for q, why in [("Which company in the corpus reported the highest FY2022 total revenue?", "aggregation_cue"),
                   ("What were total dividends paid by companies other than JPMorgan?", "aggregation_cue"),
                   ("How many of them had positive working capital?", "aggregation_cue"),
                   ("Is Corning larger than Paypal by revenue?", "multi_entity"),
                   ("What is the highest margin?", "no_entity"),
                   ("What happened to the diseases business?", "no_entity")]:
        f, reason = planner.scoped_facts(q, _DOCS * 3, sql_fields_present=True)
        assert reason == why and f.scope_doc is None and f.n_docs == len(_DOCS) * 3, q
    big = [dict(d, n_tokens_est=300_000) for d in _DOCS * 3]
    f, _ = planner.scoped_facts("Which company had the highest revenue?", big, sql_fields_present=True)
    assert planner.plan("q", f).path == "sql"  # aggregation over a >= 1M corpus still goes to sql


def test_mentions_entity_is_token_exact():
    assert planner.mentions_entity("What is Coca Cola's payout?", "Coca-Cola")
    assert planner.mentions_entity("Coca-Cola dividends", "Coca-Cola")
    assert planner.mentions_entity("Does AES have debt?", "AES Corporation")
    assert not planner.mentions_entity("Tropical diseases", "AES Corporation")
    assert planner.mentions_entity("American Water Works liquidity", "American Water Works")
