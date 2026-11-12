"""AGG-1.12 L3 period attribution (``l3_period.v2``, opt-in): no model, no server, no PDF.

The default (``period_version=None``) must stay byte-identical to the legacy extraction so Phase 0 / AGG-1.1 reproduce.
"""

from __future__ import annotations

import pytest

from src.aggregation import facts_det as fd
from src.aggregation.store import CorpusStore
from src.models.document import DocumentPreprocessResult, Section, TableRef

V2 = fd.L3_PERIOD_V2
H = "h" * 64


def _page(text, page=1, **kw):
    return fd.facts_from_page_text([(page, text)], doc_hash=H, doc_id="D", **kw)


def _tbl(rows, page=57, table_id="t:0001"):
    return {"table_id": table_id, "doc_hash": H, "doc_id": "D", "entity": "E", "page": page, "chunk_id": None,
            "caption": "(Dollars in millions)", "markdown": "", "rows": rows}


# ---------------------------------------------------------------- the default is the legacy behaviour
def test_default_is_legacy_even_when_a_fiscal_year_is_supplied():
    text = "Total assets $ 27,165 $ 27,241\n"
    legacy = _page(text)
    assert [f["period"] for f in legacy] == [None, None]
    assert _page(text, fiscal_year="2022") == legacy          # a fiscal year alone changes nothing
    rows = [["Total assets $137,100", "$138,552"]]
    assert fd.facts_from_table(_tbl(rows), "", fiscal_year="2022") == fd.facts_from_table(_tbl(rows))
    assert [f["value"] for f in fd.facts_from_table(_tbl(rows))] == [138552.0]      # the legacy 'other column' defect


def test_unknown_version_or_component_is_refused():
    with pytest.raises(ValueError):
        _page("Total assets 1 2\n", period_version="l3_period.v9")
    with pytest.raises(ValueError):
        _page("Total assets 1 2\n", period_version=V2, components=["no_such_component"])


def test_default_components_exclude_the_measured_regressions():
    assert "date_header" not in fd.L3_PERIOD_COMPONENTS and "inline_header" not in fd.L3_PERIOD_COMPONENTS
    assert set(fd.L3_PERIOD_OPTIONAL) == {"date_header", "inline_header"}


# ---------------------------------------------------------------- table rows
def test_table_split_reads_the_first_column_when_the_label_cell_carries_it():
    f = fd.facts_from_table(_tbl([["Total assets $137,100", "$138,552"]]), "", period_version=V2, fiscal_year="2022")
    assert [(x["value"], x["period"]) for x in f] == [(137100.0, "2022")]
    assert f[0]["field"] == "total_assets" and f[0]["extractor"] == "l3_table.v1" and "_inferred" in f[0]


def test_table_split_handles_paren_dollar_and_refuses_a_year_or_merged_cell():
    assert fd.split_label_number("Net loss attributable to Boeing Shareholders ($4,935)") == (
        "Net loss attributable to Boeing Shareholders", -4935.0)
    assert fd.split_label_number("Net sales 2022") == ("Net sales 2022", None)
    merged = "Total 36,776 Less: Current portion 399"
    assert fd.split_label_number(merged) == (merged, None)
    assert fd.split_label_number("Total assets") == ("Total assets", None)


def test_a_label_that_still_ends_in_a_number_is_not_given_a_period():
    # split disabled: the first printed number is inside the label, so the first column is unknown
    f = fd.facts_from_table(_tbl([["Total assets $137,100", "$138,552"]]), "", period_version=V2, fiscal_year="2022",
                            components=["first_column"])
    assert all(x["period"] is None for x in f)


def test_header_narrowing_ignores_a_long_cell_that_merely_mentions_a_year():
    rows = [["U.S. dollar notes due 2024-2093 $ 21,966 Euro notes due 2024-2041 12,485 Other, due through 2098 1,906"],
            ["Long-term debt $ 36,377", "$ 38,116"]]
    legacy = fd.facts_from_table(_tbl(rows))
    assert {x["period"] for x in legacy} == {"2024"}                      # the Coca-Cola defect: stored as '2024'
    f = fd.facts_from_table(_tbl(rows), "", period_version=V2, fiscal_year="2022")
    assert [(x["field"], x["value"], x["period"]) for x in f] == [("long_term_debt", 36377.0, "2022")]


def test_table_with_a_real_header_keeps_its_own_periods():
    rows = [["", "2022", "2021"], ["Total revenues", "$26,000", "$24,000"]]
    for kw in ({}, {"period_version": V2, "fiscal_year": "2099"}):
        f = fd.facts_from_table(_tbl(rows), "", **kw)
        assert {(x["period"], x["value"]) for x in f} == {("2022", 26000.0), ("2021", 24000.0)}
        assert not any("_inferred" in x for x in f)


def test_percent_rows_and_unknown_labels_get_no_inferred_period():
    f = fd.facts_from_table(_tbl([["Operating margin", "12.5 %"], ["Widgets sold", "5"]]), "", period_version=V2,
                            fiscal_year="2022")
    assert all(x["period"] is None for x in f)


# ---------------------------------------------------------------- text rows
def test_textrow_without_a_header_takes_the_fiscal_year_for_the_first_column_only():
    f = _page("Total assets $ 27,165 $ 27,241\n", period_version=V2, fiscal_year="2022")
    assert [(x["value"], x["period"]) for x in f] == [(27165.0, "2022"), (27241.0, None)]
    assert f[0]["extractor"] == "l3_textrow.v1" and f[0]["_inferred"] is True and "_inferred" not in f[1]


def test_textrow_with_no_fiscal_year_stays_none():
    f = _page("Total assets $ 27,165 $ 27,241\n", period_version=V2)
    assert [x["period"] for x in f] == [None, None]


def test_textrow_aligned_header_is_untouched_and_not_inferred():
    text = "Consolidated Balance Sheets\n2022 2021\nTotal assets $ 27,165 $ 27,241\n"
    legacy = _page(text)
    v2 = _page(text, period_version=V2, fiscal_year="1999")
    assert [(x["period"], x["value"]) for x in v2] == [(x["period"], x["value"]) for x in legacy] == [
        ("2022", 27165.0), ("2021", 27241.0)]
    assert not any("_inferred" in x for x in v2)


def test_textrow_fewer_numbers_than_header_years_takes_the_headers_first_year():
    text = "2022 2021 2020\nTotal assets $ 27,165 $ 27,241\n"
    assert [x["period"] for x in _page(text)] == [None, None]
    f = _page(text, period_version=V2, fiscal_year="2022")
    assert [(x["value"], x["period"]) for x in f] == [(27165.0, "2022"), (27241.0, None)]


def test_textrow_eps_and_non_canonical_rows_are_not_inferred():
    f = _page("Widget count 10 11\nBasic earnings per share: Net income $ 4.51 $ 6.96\n", period_version=V2,
              fiscal_year="2022")
    assert all(x["period"] is None for x in f)


def test_date_lines_header_is_optional_and_off_by_default():
    text = "December 31, 2022\n\nDecember 31, 2021\n\nTotal assets $ 27,165 $ 27,241\n"
    d = [(x["period"], x["value"]) for x in _page(text, period_version=V2, fiscal_year="2022")]
    assert d == [("2022", 27165.0), (None, 27241.0)]                      # default: fiscal-year fallback only
    o = _page(text, period_version=V2, fiscal_year="2022", components=list(fd.L3_PERIOD_COMPONENTS) + ["date_header"])
    assert [(x["period"], x["value"]) for x in o] == [("2022", 27165.0), ("2021", 27241.0)]


def test_inline_header_with_a_date_label_is_optional():
    text = "Year Ended December 31, (MILLIONS) 2022 2021 2020 Cash dividends paid (8,983) (8,729) (8,440)\n"
    base = {x["value"]: x["period"] for x in _page(text, period_version=V2, fiscal_year="2022")
            if x["field"] == "dividends_paid"}
    assert base == {-8983.0: "2022", -8729.0: None, -8440.0: None}        # header missed: fiscal-year fallback, first column only
    opt = _page(text, period_version=V2, fiscal_year="1999", components=list(fd.L3_PERIOD_COMPONENTS) + ["inline_header"])
    assert {x["value"]: x["period"] for x in opt if x["field"] == "dividends_paid"} == {
        -8983.0: "2022", -8729.0: "2021", -8440.0: "2020"}


# ---------------------------------------------------------------- primary pages and row selection
def test_primary_statement_pages_by_opening_title():
    pages = [(1, "Table of Contents\nCONSOLIDATED BALANCE SHEETS (In millions)\nASSETS"),
             (2, "NOTE 5. FAIR VALUE MEASUREMENTS Assets and Liabilities"),
             (3, "Notes to Consolidated Financial Statements\nConsolidated Statements of Income summary"),
             (4, "Consolidated Balance Sheets .... 53 Consolidated Statements of Income .... 54 "
                 "Consolidated Statements of Cash Flows .... 58"),
             (5, "Consolidated Statements of Cash Flows\n(in millions)")]
    assert fd.primary_statement_pages(pages) == {1, 5}


def _fact(page, quote, doc="d1", field="total_assets", period="2022", ex="l3_textrow.v1", **kw):
    f = {"doc_hash": doc, "field": field, "period": period, "extractor": ex, "page": page, "quote": quote, "value": 1.0,
         "_inferred": True, "fact_id": f"{doc}{page}", "table_id": "t", "_ridx": 0}
    f.update(kw)
    return f


def test_row_selection_prefers_the_earliest_primary_page_and_reverts_the_rest():
    facts = [_fact(71, "Total assets 5,650"), _fact(53, "Total assets 27,165"), _fact(72, "Total assets 5,297")]
    out = fd.select_inferred_periods(facts, {"d1": {53, 54}})
    assert [(f["page"], f["period"]) for f in facts] == [(71, None), (53, "2022"), (72, None)]
    assert out == {"inferred_kept": 1, "inferred_reverted": 2}


def test_row_selection_reverts_all_when_none_is_on_a_primary_page_and_keeps_a_single_row():
    many = [_fact(71, "a"), _fact(72, "b")]
    fd.select_inferred_periods(many, {"d1": {53}})
    assert [f["period"] for f in many] == [None, None]
    one = [_fact(71, "a"), _fact(71, "a")]                      # the same row twice is one row
    fd.select_inferred_periods(one, {"d1": {53}})
    assert [f["period"] for f in one] == ["2022", "2022"]


def test_row_selection_is_per_document_and_per_field():
    facts = [_fact(71, "a", doc="d1"), _fact(72, "b", doc="d1"), _fact(71, "c", doc="d2"), _fact(5, "z", field="capex")]
    fd.select_inferred_periods(facts, {"d1": {72}, "d2": set()})
    assert [f["period"] for f in facts] == [None, "2022", "2022", "2022"]


def test_a_reverted_table_fact_gets_its_legacy_fact_id_back():
    a = _fact(71, "a", ex="l3_table.v1", table_id="t9", _ridx=3, value=7.0, fact_id="stale")
    b = _fact(72, "b", ex="l3_table.v1")
    fd.select_inferred_periods([a, b], {})
    assert a["period"] is None and a["fact_id"] == fd._fact_id("t9", 3, None, 7.0)


# ---------------------------------------------------------------- the corpus extractor
def _doc():
    s = Section(id="s1", title="Consolidated Balance Sheets", level=1, content="Total assets $137,100 $138,552",
                page_start=57, page_end=57)
    t = TableRef(id="t1", page=57, caption="(Dollars in millions)", section_id="s1",
                 rows=[["Cash $14,614", "$8,052"], ["Total assets $137,100", "$138,552"]])
    return DocumentPreprocessResult(original_path="/x/b.pdf", sections=[s], figures=[], total_pages=100, tables=[t])


def _store():
    st = CorpusStore()
    st.ingest(_doc(), corpus_id="c", doc_id="BOEING_2022_10K", entity="BOEING")
    return st


def test_extract_corpus_facts_default_equals_the_legacy_call_and_v2_is_opt_in():
    st = _store()
    legacy = fd.extract_corpus_facts(st, "c")
    assert fd.extract_corpus_facts(st, "c", fiscal_years={"BOEING_2022_10K": "2022"}) == legacy
    assert [f["value"] for f in legacy if f["field"] == "total_assets"] == [138552.0]
    v2 = fd.extract_corpus_facts(st, "c", period_version=V2, fiscal_years={"BOEING_2022_10K": "2022"})
    ta = [f for f in v2 if f["field"] == "total_assets"]
    assert [(f["value"], f["period"]) for f in ta] == [(137100.0, "2022")]
    assert not any("_inferred" in f or "_ridx" in f for f in v2)
    assert not any(f["field"] == "total_assets" and f["period"] is None for f in v2)


def test_extract_corpus_facts_without_a_manifest_year_infers_nothing():
    v2 = fd.extract_corpus_facts(_store(), "c", period_version=V2, fiscal_years={})
    assert all(f["period"] is None for f in v2 if f["extractor"] == "l3_table.v1")


def test_row_selection_end_to_end_through_the_corpus_extractor(monkeypatch):
    st = _store()
    monkeypatch.setattr(st, "document_pages", lambda cid: [
        {"doc_hash": d["doc_hash"], "doc_id": d["doc_id"], "entity": d["entity"],
         "pages": [(42, "Financial Position selected data\nTotal assets $ 1,954 $ 2,182"),
                   (57, "Consolidated Statements of Financial Position\n(Dollars in millions)\n"
                        "Total assets $ 137,100 $ 138,552")]} for d in st.corpus_docs(cid)])
    v2 = fd.extract_corpus_facts(st, "c", period_version=V2, fiscal_years={"BOEING_2022_10K": "2022"})
    rows = {(f["extractor"], f["page"], f["value"]): f["period"] for f in v2 if f["field"] == "total_assets"}
    assert rows[("l3_textrow.v1", 57, 137100.0)] == "2022"
    assert rows[("l3_textrow.v1", 42, 1954.0)] is None            # a non-primary page's row does not get the fiscal year
    assert rows[("l3_table.v1", 57, 137100.0)] == "2022"


def test_default_v1_extraction_facts_never_carry_internal_keys():
    for f in fd.extract_corpus_facts(_store(), "c"):
        assert not any(k.startswith("_") for k in f)
