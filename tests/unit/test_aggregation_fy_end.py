"""AGG-1.13 amendment 1: the opt-in ``fy_end`` fiscal-year-end column labelling of the L3 layer (no model, no server, no PDF).

The rule reads only a printed column header and the manifest fiscal year. Default (``fy_end=None``) must stay byte-identical for both
``agg_pipeline.v1`` and ``agg_pipeline.v2`` so Phase 0, AGG-1.1 and AGG-1.12 stores reproduce. Header fixtures are the printed headers of
the four January-end filings (first column = the fiscal-2022 column), plus a calendar-year filer.
"""

from __future__ import annotations

import json

import pytest

from src.aggregation import FLAG_ENV as LAYER_FLAG
from src.aggregation import facts_det as fd
from src.aggregation import pipeline as pl
from src.aggregation.registry import IndexRegistry
from src.aggregation.store import CorpusStore
from src.models.document import DocumentPreprocessResult, Section, TableRef

V2 = fd.L3_PERIOD_V2
H = "h" * 64
CID = "c"

# printed first-column headers (the three-column statements print two for a balance sheet)
ULTA_BS = ["January 28, 2023", "January 29, 2022"]
ULTA_IS = ["January 28, 2023", "January 29, 2022", "January 30, 2021"]
BESTBUY = ["January 29, 2022", "January 30, 2021", "February 1, 2020"]
WALMART = ["2022", "2021", "2020"]
FOOTLOCKER_IS = ["2022", "2021", "2020"]
FOOTLOCKER_BS = ["January 28, 2023", "January 29, 2022"]
CALENDAR = ["December 31, 2022", "December 31, 2021"]
MARCH_FYE = ["March 31, 2023", "March 31, 2022"]


@pytest.fixture(autouse=True)
def _layer_on(monkeypatch):
    monkeypatch.setenv(LAYER_FLAG, "1")
    monkeypatch.delenv(pl.FLAG_ENV, raising=False)
    monkeypatch.delenv(fd.FY_END_ENV, raising=False)


def _tbl(hdr, label="Total assets", doc="D"):
    nums = [f"{1000 + 10 * i:,}" for i in range(len(hdr))]
    return {"table_id": "t:1", "doc_hash": H, "doc_id": doc, "entity": "E", "page": 57, "chunk_id": None,
            "caption": "(Dollars in millions)", "markdown": "", "rows": [[""] + list(hdr), [label] + nums]}


def _periods(hdr, *, fy="2022", fy_end=None, pv=None):
    kw = {"fiscal_year": fy}
    if pv:
        kw["period_version"] = pv
    if fy_end is not None:
        kw["fy_end"] = fy_end
    return [f["period"] for f in fd.facts_from_table(_tbl(hdr), "", **kw)]


def _page_periods(hdr_line, *, fy="2022", fy_end=None, pv=None):
    text = f"{hdr_line}\nTotal assets 1,000 990\n"
    kw = {"fiscal_year": fy}
    if pv:
        kw["period_version"] = pv
    if fy_end is not None:
        kw["fy_end"] = fy_end
    return [f["period"] for f in fd.facts_from_page_text([(1, text)], doc_hash=H, doc_id="D", **kw)]


# ---------------------------------------------------------------- the option itself
def test_fy_end_months_normalisation_and_env(monkeypatch):
    assert fd.fy_end_months(None) is None and fd.fy_end_months(False) is None and fd.fy_end_months("off") is None
    assert fd.fy_end_months(True) == fd.fy_end_months("on") == fd.fy_end_months("1") == (1, 2, 3)
    assert fd.fy_end_months("1,2") == fd.fy_end_months([2, 1]) == (1, 2)
    with pytest.raises(ValueError):
        fd.fy_end_months("13")
    assert fd.fy_end_months(environ={}) is None                                  # unset env = disabled
    assert fd.fy_end_months(environ={fd.FY_END_ENV: "on"}) == (1, 2, 3)
    monkeypatch.setenv(fd.FY_END_ENV, "1")
    assert fd.fy_end_months() == (1, 2, 3)


# ---------------------------------------------------------------- defaults are byte-identical
@pytest.mark.parametrize("pv", [None, V2])
@pytest.mark.parametrize("hdr", [ULTA_BS, ULTA_IS, BESTBUY, WALMART, FOOTLOCKER_BS, CALENDAR])
def test_default_and_explicit_none_are_the_unchanged_call(hdr, pv):
    kw = {"fiscal_year": "2022", **({"period_version": pv} if pv else {})}
    plain = fd.facts_from_table(_tbl(hdr), "", **kw)
    assert plain == fd.facts_from_table(_tbl(hdr), "", fy_end=None, **kw)
    assert plain == fd.facts_from_table(_tbl(hdr), "", fy_end=False, **kw)
    assert plain == fd.facts_from_table(_tbl(hdr), "", fy_end="off", **kw)


def test_the_legacy_defect_is_still_there_by_default_for_both_versions():
    assert _periods(ULTA_BS) == ["2023", "2022"]
    assert _periods(ULTA_BS, pv=V2) == ["2023", "2022"]


def test_pipeline_l3_kwargs_defaults_are_unchanged():
    v1, v2 = pl.pipeline_spec(pl.PIPELINE_V1), pl.pipeline_spec(pl.PIPELINE_V2)
    assert v1.l3_kwargs({"D": "2022"}) == {} and v1.l3_kwargs({"D": "2022"}, None) == {}
    assert v2.l3_kwargs({"D": "2022"}) == {"period_version": V2, "fiscal_years": {"D": "2022"}}
    assert v1.l3_kwargs({"D": "2022"}, fy_end=False) == {}                      # explicit off is the default
    assert v1.l3_kwargs({"D": "2022"}, fy_end=True) == {"fy_end": (1, 2, 3), "fiscal_years": {"D": "2022"}}
    assert v2.l3_kwargs({"D": "2022"}, fy_end="1,2,3") == {"period_version": V2, "fiscal_years": {"D": "2022"}, "fy_end": (1, 2, 3)}
    assert pl.pipeline_spec().version == pl.PIPELINE_V3                          # the flag does not touch the version default


# ---------------------------------------------------------------- the January-end filers: tables
@pytest.mark.parametrize("pv", [None, V2])
def test_ulta_headers_give_fiscal_2022_first_column_in_both_versions(pv):
    assert _periods(ULTA_BS, fy_end=True, pv=pv) == ["2022", "2021"]
    assert _periods(ULTA_IS, fy_end=True, pv=pv) == ["2022", "2021", "2020"]


@pytest.mark.parametrize("pv", [None, V2])
def test_foot_locker_balance_sheet_gives_2022_and_its_bare_year_income_header_is_untouched(pv):
    assert _periods(FOOTLOCKER_BS, fy_end=True, pv=pv) == ["2022", "2021"]
    assert _periods(FOOTLOCKER_IS, fy_end=True, pv=pv) == _periods(FOOTLOCKER_IS, pv=pv) == ["2022", "2021", "2020"]


@pytest.mark.parametrize("pv", [None, V2])
def test_best_buy_and_walmart_are_unchanged(pv):
    for hdr in (BESTBUY, WALMART):
        assert _periods(hdr, fy_end=True, pv=pv) == _periods(hdr, pv=pv)
    assert _periods(BESTBUY, fy_end=True, pv=pv) == ["2022", "2021", "2020"]   # first column 2022 == the manifest fiscal year: no shift


@pytest.mark.parametrize("pv", [None, V2])
def test_a_calendar_year_filer_is_unchanged(pv):
    assert _periods(CALENDAR, fy_end=True, pv=pv) == _periods(CALENDAR, pv=pv) == ["2022", "2021"]
    assert fd.facts_from_table(_tbl(CALENDAR), "", fiscal_year="2022", fy_end=True) == fd.facts_from_table(_tbl(CALENDAR), "", fiscal_year="2022")


def test_months_are_configurable_and_the_manifest_year_is_the_authority():
    assert _periods(MARCH_FYE, fy_end=True) == ["2022", "2021"]                   # March is in the default Jan-Mar
    assert _periods(MARCH_FYE, fy_end=(1, 2)) == ["2023", "2022"]                 # not in a narrowed month set
    assert _periods(ULTA_BS, fy_end=(1,)) == ["2022", "2021"]
    assert _periods(ULTA_BS, fy="2023", fy_end=True) == ["2023", "2022"]          # manifest says 2023: the printed 2023 column is fiscal 2023
    assert _periods(ULTA_BS, fy=None, fy_end=True) == ["2023", "2022"]            # no manifest fiscal year: nothing to anchor on
    assert _periods(ULTA_BS, fy="2021", fy_end=True) == ["2023", "2022"]          # first column is not manifest + 1: no shift


def test_mixed_or_non_date_header_cells_are_never_shifted():
    assert _periods(["January 28, 2023", "2022"], fy_end=True) == ["2023", "2022"]            # a bare-year cell in the header: leave
    assert _periods(["Fiscal 2023", "Fiscal 2022"], fy_end=True) == ["2023", "2022"]
    assert _periods(["January 28, 2023", "December 31, 2022"], fy_end=True) == ["2023", "2022"]  # a December date among them: leave
    # a label that merely contains a date is not a header cell: no shift of what the legacy code would read
    t = _tbl(["Outstanding on January 1, 2023 75 Granted"])
    assert fd.facts_from_table(t, "", fiscal_year="2022", fy_end=True) == fd.facts_from_table(t, "", fiscal_year="2022")


def test_prefixed_date_cells_are_dates():
    assert _periods(["As of January 28, 2023", "As of January 29, 2022"], fy_end=True) == ["2022", "2021"]
    assert _periods(["Fiscal Year Ended January 28, 2023"], fy_end=True) == ["2022"]


# ---------------------------------------------------------------- the same rule on the text-row path
@pytest.mark.parametrize("pv", [None, V2])
def test_text_row_header_dates_follow_the_same_rule(pv):
    assert _page_periods("January 28, 2023 January 29, 2022", fy_end=True, pv=pv) == ["2022", "2021"]
    assert _page_periods("January 29, 2022 January 30, 2021", fy_end=True, pv=pv) == ["2022", "2021"]       # Best Buy: unchanged
    assert _page_periods("January 29, 2022 January 30, 2021", pv=pv) == ["2022", "2021"]
    assert _page_periods("December 31, 2022 December 31, 2021", fy_end=True, pv=pv) == ["2022", "2021"]
    assert _page_periods("January 28, 2023 January 29, 2022", pv=pv) == ["2023", "2022"]                     # default: the defect
    assert _page_periods("2022 2021", fy_end=True, pv=pv) == ["2022", "2021"]


# ---------------------------------------------------------------- the corpus level: both versions, registry, defaults
def _store(hdr=ULTA_BS) -> CorpusStore:
    s = Section(id="s1", title="Consolidated Balance Sheets", level=1,
                content=f"{'  '.join(hdr)}\nTotal assets 1,000 990", page_start=57, page_end=57)
    t = TableRef(id="t1", page=57, caption="(Dollars in millions)", section_id="s1",
                 rows=[[""] + list(hdr), ["Total assets", "1,000", "990"]])
    st = CorpusStore()
    st.ingest(DocumentPreprocessResult(original_path="/x/u.pdf", sections=[s], figures=[], total_pages=100, tables=[t]),
              corpus_id=CID, doc_id="D", entity="E")
    return st


def _dump(facts):
    return json.dumps(sorted(facts, key=lambda f: f["fact_id"]), sort_keys=True, default=str)


def test_corpus_default_is_byte_identical_for_both_versions_and_fy_end_relabels_both():
    st = _store()
    for ver in (pl.PIPELINE_V1, pl.PIPELINE_V2):
        spec = pl.pipeline_spec(ver)
        base = spec.extract_l3(st, CID, {"D": "2022"})
        assert _dump(base) == _dump(spec.extract_l3(st, CID, {"D": "2022"}, None))
        on = spec.extract_l3(st, CID, {"D": "2022"}, True)
        assert {f["period"] for f in base if f["field"] == "total_assets"} >= {"2023"}
        got = {f["period"] for f in on if f["field"] == "total_assets"}
        assert "2022" in got and "2023" not in got, (ver, got)
        assert not any("_inferred" in f or "_ridx" in f for f in on)
    legacy = fd.extract_corpus_facts(st, CID)
    assert _dump(legacy) == _dump(pl.pipeline_spec(pl.PIPELINE_V1).extract_l3(st, CID, {"D": "2022"}))


def test_corpus_fy_end_needs_the_manifest_fiscal_year():
    st = _store()
    assert _dump(fd.extract_corpus_facts(st, CID, fy_end=True)) == _dump(fd.extract_corpus_facts(st, CID))      # no fiscal_years: no shift
    assert _dump(fd.extract_corpus_facts(st, CID, fiscal_years={"D": "2022"}, fy_end=True)) != _dump(fd.extract_corpus_facts(st, CID))


def test_registry_threads_fy_end_for_both_versions_and_defaults_are_unchanged():
    for ver in (None, pl.PIPELINE_V1, pl.PIPELINE_V2):
        spec = pl.pipeline_spec(ver) if ver else None
        off = IndexRegistry(_store(), pipeline=spec, fiscal_years={"D": "2022"})
        on = IndexRegistry(_store(), pipeline=spec, fiscal_years={"D": "2022"}, fy_end=True)
        f_off = off.get_or_build(CID, "l3_facts_det")
        f_on = on.get_or_build(CID, "l3_facts_det")
        assert "2023" in {f["period"] for f in f_off if f["field"] == "total_assets"}
        p_on = {f["period"] for f in f_on if f["field"] == "total_assets"}
        assert "2022" in p_on and "2023" not in p_on, (ver, p_on)
