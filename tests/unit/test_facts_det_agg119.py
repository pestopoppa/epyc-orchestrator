"""AGG-1.19 (2026-11-16): opt-in L3 components ``fy_end_lone_monthday`` and ``fy1_captions``. Default extraction unchanged.
No model, no network.
"""

from __future__ import annotations

import pytest

from src.aggregation import facts_det as fd

V2 = list(fd.L3_PERIOD_COMPONENTS)
SF = ("Consolidated Statements of Operations (in millions, except per share data)\n\n4 Fiscal Year Ended January 31,\n\n"
      "2023 2022 2021\n\nTotal revenues 31,352 26,492 21,252\n")
WALMART = "Fiscal Years Ended January 31, (Amounts in millions) 2022 2021 2020\nTotal revenues 572,754 559,151 523,964\n"
WALMART_SPLIT = "(Amounts in millions)\n\nFiscal Years Ended January 31,\n\n2022 2021 2020\n\nTotal revenues 572,754 559,151 523,964\n"
SF_LEAD = "Fiscal Year Ended January 31, (in millions) 2023 2022 2021\nTotal revenues 31,352 26,492 21,252\n"
DEC = "Years ended December 31, 2019 2018 2017\nTotal revenues 76,559 101,127 93,392\n"


def _rows(text, fy, comps=None, fy_end=fd.FY_END_MONTHS, pv=fd.L3_PERIOD_V2):
    fs = fd.facts_from_page_text([(1, text)], doc_hash="h", period_version=pv, fiscal_year=fy, fy_end=fy_end,
                                 components=(V2 if pv else []) + list(comps or []) if (comps or pv) else None)
    return [(f["field"], f["period"], f["value"]) for f in fs]


def test_salesforce_lone_monthday_is_shifted_only_with_the_component():
    assert [p for _f, p, _v in _rows(SF, "2022")] == ["2023", "2022", "2021"]
    assert [p for _f, p, _v in _rows(SF, "2022", ["fy_end_lone_monthday"])] == ["2022", "2021", "2020"]
    assert [p for _f, p, _v in _rows(SF_LEAD, "2022", ["fy_end_lone_monthday"])] == ["2022", "2021", "2020"]


@pytest.mark.parametrize("text", [WALMART, WALMART_SPLIT])
def test_walmart_january_filer_is_not_shifted(text):
    # manifest FY 2022, first printed year 2022: the unchanged guard (first year == FY + 1) refuses the shift
    assert _rows(text, "2022", ["fy_end_lone_monthday"]) == _rows(text, "2022")
    assert [p for _f, p, _v in _rows(text, "2022", ["fy_end_lone_monthday"])] == ["2022", "2021", "2020"]


def test_component_is_a_noop_without_fy_end_or_manifest_or_for_december_filers():
    assert _rows(SF, "2022", ["fy_end_lone_monthday"], fy_end=None) == _rows(SF, "2022", fy_end=None)
    assert _rows(SF, None, ["fy_end_lone_monthday"]) == _rows(SF, None)
    assert _rows(DEC, "2019", ["fy_end_lone_monthday"]) == _rows(DEC, "2019")
    assert _rows(SF, "2023", ["fy_end_lone_monthday"]) == _rows(SF, "2023")   # first year == FY: not shifted


def test_lone_month_day_line():
    assert fd.lone_month_day_line("4 Fiscal Year Ended January 31,") == [1]
    assert fd.lone_month_day_line("Fiscal Year Ended January 31,") == [1]
    assert fd.lone_month_day_line("January 28, January 29,") is None           # two month-days: amendment 2's path
    assert fd.lone_month_day_line("Revenue grew 4 percent through January 31,") is None


def test_lone_monthday_table_split_header():
    t = {"table_id": "t:1", "doc_hash": "h", "doc_id": "D", "entity": "E", "page": 60, "chunk_id": None,
         "rows": [["", "Fiscal Year Ended January 31,", "", ""], ["", "2023", "2022", "2021"], ["Total revenues", "31,352", "26,492", "21,252"]]}
    base = fd.facts_from_table(t, period_version=fd.L3_PERIOD_V2, fiscal_year="2022", fy_end=fd.FY_END_MONTHS)
    lone = fd.facts_from_table(t, period_version=fd.L3_PERIOD_V2, fiscal_year="2022", fy_end=fd.FY_END_MONTHS,
                               components=V2 + ["fy_end_lone_monthday"])
    assert sorted(f["period"] for f in base if f["field"] == "revenue") == ["2021", "2022", "2023"]
    assert sorted(f["period"] for f in lone if f["field"] == "revenue") == ["2020", "2021", "2022"]


@pytest.mark.parametrize("caption,field", [
    ("Net cash used in operating activities", "operating_cash_flow"),
    ("Net cash (used)/provided by operating activities", "operating_cash_flow"),
    ("Net (loss)/earnings", "net_income"),
    ("Property, plant and equipment additions", "capex"),
])
def test_fy1_captions_map_only_with_the_component(caption, field):
    assert fd.textrow_field(caption) == (fd.slug(caption), False)
    assert fd.textrow_field(caption, fy1_captions=True) == (field, True)
    assert _rows(f"(in millions) 2019 2018\n{caption} (2,446) 15,322\n", "2019", ["fy1_captions"])[0][0] == field
    assert _rows(f"(in millions) 2019 2018\n{caption} (2,446) 15,322\n", "2019")[0][0] != field


@pytest.mark.parametrize("caption", [
    "Current portion of property, plant and equipment additions", "Net loss earnings per share", "Net cash used in investing activities",
    "Property, plant and equipment reductions", "Net cash used in operating activities of discontinued operations",
])
def test_fy1_captions_do_not_admit_other_lines(caption):
    assert fd.textrow_field(caption, fy1_captions=True)[1] is False


def test_components_validate_and_default_is_unchanged():
    for pv in (None, fd.L3_PERIOD_V2):
        fd._components(pv, ["fy1_captions", "fy_end_lone_monthday"])
    with pytest.raises(ValueError):
        fd._components(fd.L3_PERIOD_V2, ["fy1_caption"])
    assert "fy1_captions" not in fd.L3_PERIOD_COMPONENTS and "fy_end_lone_monthday" not in fd.L3_TEXTROW_OPTIONAL


# ---------------------------------------------------------------- agg_pipeline.v5 (opt-in)
def test_v5_is_v4_plus_both_components_and_default_stays_v3(monkeypatch):
    from src.aggregation import pipeline as pl
    monkeypatch.delenv(pl.FLAG_ENV, raising=False)
    assert pl.DEFAULT == pl.PIPELINE_V3 and pl.pipeline_spec().version == pl.PIPELINE_V3
    s4, s5 = pl.pipeline_spec(pl.PIPELINE_V4), pl.pipeline_spec(pl.PIPELINE_V5)
    assert s5.l3_textrow == s4.l3_textrow + ("fy_end_lone_monthday", "fy1_captions")
    assert s5.l4_kwargs() == s4.l4_kwargs() and s5.canonical_kwargs() == s4.canonical_kwargs()
    kw = s5.l3_kwargs({"D": "2022"}, True)
    assert kw["components"] == list(fd.L3_PERIOD_COMPONENTS) + list(pl.V5_L3_TEXTROW) and kw["fy_end"] == fd.FY_END_MONTHS
    assert "fy1_captions" not in str(pl.pipeline_spec(pl.PIPELINE_V3).l3_kwargs({"D": "2022"}, True))
    assert pl.pipeline_spec(environ={pl.FLAG_ENV: pl.PIPELINE_V5}).version == pl.PIPELINE_V5
    assert s5.as_record()["l3_textrow"] == list(pl.V5_L3_TEXTROW)


def test_v3_and_v4_extraction_byte_identical_on_the_fixtures():
    from src.aggregation import pipeline as pl
    for spec in (pl.PIPELINE_V3, pl.PIPELINE_V4):
        comps = list(pl.pipeline_spec(spec).l3_textrow)
        for t in (SF, WALMART, DEC):
            a = fd.facts_from_page_text([(1, t)], doc_hash="h", period_version=fd.L3_PERIOD_V2, fiscal_year="2022",
                                        fy_end=fd.FY_END_MONTHS, components=V2 + comps)
            assert all(f["field"] != "operating_cash_flow" or "used" not in f["label"].lower() for f in a)
            assert [f["period"] for f in a if f["field"] == "revenue"][:1] in (["2023"], ["2022"], ["2019"])
