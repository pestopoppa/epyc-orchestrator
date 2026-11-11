"""AGG-1.3 / AGG-1.4 orchestrator additions (zero model, zero server).

* ``CorpusStore.ingest`` records what a replace removes: the old document's facts in ``facts_superseded`` and one
  ``documents_superseded`` row, ``superseded_by`` = the new content hash; the live tables no longer hold them and
  the canonical facts and SQL catalog never see them.
* ``planner.question_fields`` / ``backfill_plan`` / ``plan(..., backfill=...)``: the deterministic ``sql(+backfill)``
  trigger and its place in the ``decision_receipt.v1``.
"""

from __future__ import annotations

import pytest

from src.aggregation import FLAG_ENV, planner, sql_grammar
from src.aggregation.store import CANONICAL_TABLE, CorpusStore
from src.models.document import DocumentPreprocessResult, Section


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")


def _doc(text: str) -> DocumentPreprocessResult:
    secs = [Section(id="s1", title="Statements", level=1, content=text, page_start=1, page_end=1)]
    return DocumentPreprocessResult(original_path="/x/a.pdf", sections=secs, figures=[], total_pages=1, tables=[])


def _fact(doc_hash: str, doc_id: str, fid: str, field: str, value: float, period: str = "2022",
          extractor: str = "l3_textrow.v1") -> dict:
    return {"fact_id": fid, "doc_hash": doc_hash, "doc_id": doc_id, "entity": doc_id, "period": period,
            "field": field, "value": value, "unit": "USD_millions", "quote": f"{field} {value}",
            "quote_start": 0, "quote_end": 5, "chunk_id": f"{doc_hash[:16]}:00000", "table_id": None, "page": 1,
            "line_start": 1, "line_end": 1, "confidence": "high", "extractor": extractor,
            "audit_status": "unaudited"}


def _two_docs(tmp_path):
    st = CorpusStore(tmp_path / "s.sqlite")
    r_a = st.ingest(_doc("Total revenue 100\n"), corpus_id="c", doc_id="A", entity="A Corp")
    r_b = st.ingest(_doc("Total revenue 200\n"), corpus_id="c", doc_id="B", entity="B Corp")
    st.write_facts([_fact(r_a.doc_hash, "A", "fa1", "revenue", 100.0), _fact(r_a.doc_hash, "A", "fa2", "capex", 7.0),
                    _fact(r_b.doc_hash, "B", "fb1", "revenue", 200.0)])
    st.refresh_canonical("c")
    return st, r_a, r_b


def test_replace_marks_old_facts_superseded_and_removes_them_from_every_read_path(on, tmp_path):
    st, r_a, r_b = _two_docs(tmp_path)
    assert st.superseded_documents("c") == [] and st.superseded_facts("c") == []
    new = st.ingest(_doc("Total revenue 150 (restated, other fiscal year)\n"), corpus_id="c", doc_id="A",
                    entity="A Corp")
    assert new.status == "replaced" and new.doc_hash != r_a.doc_hash and new.superseded == (r_a.doc_hash,)
    # the ledger
    docs = st.superseded_documents("c")
    assert len(docs) == 1
    d = docs[0]
    assert (d["doc_hash"], d["superseded_by"], d["doc_id"], d["n_facts"], d["n_canonical"]) == \
        (r_a.doc_hash, new.doc_hash, "A", 2, 2)
    sf = st.superseded_facts("c", doc_hash=r_a.doc_hash)
    assert {f["fact_id"] for f in sf} == {"fa1", "fa2"} and {f["superseded_by"] for f in sf} == {new.doc_hash}
    # nothing of the old hash is live anywhere
    for table in ("documents", "chunks", "tables_", "facts", CANONICAL_TABLE):
        assert st.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE doc_hash = ?", [r_a.doc_hash]).fetchone()[0] == 0
    st.write_facts([_fact(new.doc_hash, "A", "fa3", "revenue", 150.0)])
    st.refresh_canonical("c")
    canon = {(r["doc_id"], r["field"]): r for r in st.canonical_facts("c")}
    assert canon[("A", "revenue")]["fact_id"] == "fa3" and ("A", "capex") not in canon
    assert all(r["doc_hash"] != r_a.doc_hash for r in st.canonical_facts("c"))
    cat = sql_grammar.catalog_from_canonical(st.canonical_facts("c"))
    assert "capex" not in cat.fields  # the superseded-only field is not even nameable by the SQL grammar
    # B untouched
    assert st.conn.execute("SELECT COUNT(*) FROM facts WHERE doc_hash = ?", [r_b.doc_hash]).fetchone()[0] == 1


def test_unchanged_ingest_supersedes_nothing_and_is_a_hash_hit(on, tmp_path):
    st, r_a, _ = _two_docs(tmp_path)
    again = st.ingest(_doc("Total revenue 100\n"), corpus_id="c", doc_id="A", entity="A Corp")
    assert again.status == "unchanged" and again.doc_hash == r_a.doc_hash and again.superseded == ()
    assert st.superseded_documents() == [] and st.superseded_facts() == []
    assert st.conn.execute("SELECT COUNT(*) FROM facts WHERE doc_hash = ?", [r_a.doc_hash]).fetchone()[0] == 2


def test_first_ingest_has_no_superseded_and_ledger_is_idempotent(on, tmp_path):
    st = CorpusStore(tmp_path / "s.sqlite")
    r = st.ingest(_doc("x\n"), corpus_id="c", doc_id="A")
    assert (r.status, r.superseded) == ("ingested", ())
    st.write_facts([_fact(r.doc_hash, "A", "f1", "revenue", 1.0)])
    st._supersede_doc(r.doc_hash, "h" * 64, "c")
    st._supersede_doc(r.doc_hash, "h" * 64, "c")  # same (old, new) pair: replaced, not duplicated
    assert len(st.superseded_documents("c")) == 1 and len(st.superseded_facts("c")) == 1
    st._supersede_doc("no-such-hash", "h" * 64, "c")  # unknown hash: a no-op
    assert len(st.superseded_documents("c")) == 1


def test_an_old_store_without_the_ledger_tables_is_migrated_on_open(on, tmp_path):
    p = tmp_path / "old.sqlite"
    st = CorpusStore(p)
    st.conn.execute("DROP TABLE facts_superseded")
    st.conn.execute("DROP TABLE documents_superseded")
    st.conn.commit()
    st.close()
    st2 = CorpusStore(p)
    assert st2.superseded_documents() == [] and st2.superseded_facts() == []


# ----------------------------------------------------------------------------------------- planner
def test_question_fields_is_a_phrase_table_not_a_model():
    q = "Which company in the corpus reported the highest FY2022 long-term debt (non-current portion), in USD millions?"
    assert planner.question_fields(q) == ["long_term_debt"]
    assert planner.question_fields("Which company reported the highest FY2022 long term debt?") == ["long_term_debt"]
    assert planner.question_fields("Compare capital expenditures and dividends paid") == ["capex", "dividends_paid"]
    # a longer phrase hides the shorter one inside it
    assert planner.question_fields("highest net income attributable to shareholders") == ["net_income_attributable"]
    assert planner.question_fields("highest net income") == ["net_income"]
    assert planner.question_fields("How are you?") == []
    assert planner.question_period("FY2022 long-term debt") == "2022"
    assert planner.question_period("long-term debt in 2019 and 2020") == "2019"
    assert planner.question_period("long-term debt") is None


def test_backfill_plan_lists_the_documents_without_a_canonical_row():
    docs = [{"doc_id": d} for d in ("A", "B", "C")]
    canon = [{"doc_id": "A", "field": "long_term_debt", "period": "2022"},
             {"doc_id": "B", "field": "long_term_debt", "period": "2021"},
             {"doc_id": "B", "field": "revenue", "period": "2022"}]
    bp = planner.backfill_plan("highest FY2022 long-term debt", canon, docs)
    assert bp["trigger"] is True and bp["fields"] == ["long_term_debt"] and bp["period"] == "2022"
    assert bp["missing_docs"] == ["B", "C"] and bp["per_field"]["long_term_debt"]["n_present"] == 1
    # no period asked: any period counts
    assert planner.backfill_plan("highest long-term debt", canon, docs)["missing_docs"] == ["C"]
    # everything present: no trigger; a field no question needs is never planned
    full = canon + [{"doc_id": "B", "field": "long_term_debt", "period": "2022"},
                    {"doc_id": "C", "field": "long_term_debt", "period": "2022"}]
    assert planner.backfill_plan("highest FY2022 long-term debt", full, docs)["trigger"] is False
    assert planner.backfill_plan("how are you", canon, docs)["needed_fields"] == []


def test_the_plan_receipt_carries_the_backfill_trigger_on_the_sql_path(on):
    docs = [{"doc_id": d, "entity": d, "n_tokens_est": 300_000} for d in ("AAA", "BBB", "CCC", "DDD")]
    q = "Which company in the corpus reported the highest FY2022 long-term debt?"
    facts, why = planner.scoped_facts(q, docs, sql_fields_present=True)
    assert why == "aggregation_cue"
    bp = planner.backfill_plan(q, [{"doc_id": "AAA", "field": "long_term_debt", "period": "2022"}], docs)
    dec = planner.plan(q, facts, backfill=bp)
    assert (dec.path, dec.llm_called, dec.decided_by) == ("sql", False, "size_rule")
    r = dec.receipt
    assert r["schema"] == "decision_receipt.v1" and r["path_label"] == "sql+backfill"
    assert r["backfill"]["trigger"] is True and r["backfill"]["fields"] == ["long_term_debt"]
    assert r["backfill"]["missing_docs"] == ["BBB", "CCC", "DDD"] and dec.backfill == r["backfill"]
    # the plan without a backfill argument is exactly what it was
    plain = planner.plan(q, facts)
    assert "backfill" not in plain.receipt and plain.backfill == {} and plain.path == "sql"


def test_backfill_is_not_applicable_off_the_sql_path_or_when_nothing_is_missing(on):
    docs = [{"doc_id": "A", "entity": "A", "n_tokens_est": 40_000}]
    q = "What was FY2022 long-term debt?"
    bp = planner.backfill_plan(q, [], docs)
    assert bp["trigger"] is True
    facts, _ = planner.scoped_facts(q, docs, sql_fields_present=True)
    dec = planner.plan(q, facts, backfill=bp)
    assert dec.path == "whole_doc" and dec.backfill == {}
    assert dec.receipt["backfill"]["trigger"] is False and "does not read L4" in dec.receipt["backfill"]["reason"]
    big = [{"doc_id": "A", "entity": "A", "n_tokens_est": 700_000}, {"doc_id": "B", "entity": "B", "n_tokens_est": 700_000}]
    cq = "Which company has the highest FY2022 long-term debt across the corpus?"
    full = planner.backfill_plan(cq, [{"doc_id": d, "field": "long_term_debt", "period": "2022"} for d in "AB"], big)
    f2, _ = planner.scoped_facts(cq, big, sql_fields_present=True)
    d2 = planner.plan(cq, f2, backfill=full)
    assert d2.path == "sql" and d2.backfill == {} and d2.receipt["backfill"]["reason"] == "no needed field is missing"
