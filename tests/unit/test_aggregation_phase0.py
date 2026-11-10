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
