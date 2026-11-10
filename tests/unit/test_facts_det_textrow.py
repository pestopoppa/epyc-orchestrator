"""L3 text-row extractor: real-shaped lines from the AWK / generic 10-K statement pages."""

from __future__ import annotations

from src.aggregation import facts_det as fd

BALANCE = """Table of Contents
American Water Works Company, Inc. and Subsidiary Companies
Consolidated Balance Sheets
(In millions, except share and per share data)
December 31, 2022 December 31, 2021
ASSETS
Accumulated depreciation (6,513) (6,329)
Assets held for sale — 683
Accounts receivable, net of allowance for uncollectible accounts of $60 and $75, respectively 334 271
Total current assets 1,250 1,554
Total assets $ 27,787 $ 26,075
79"""

INCOME = """Consolidated Statements of Operations
(In millions, except per share data)
For the Years Ended December 31,
2022 2021 2020
Operating revenues $ 3,792 $ 3,930 $ 3,777
Gain on sale of businesses 19 747 —
Net income attributable to common shareholders $ 820 $ 1,263 $ 709
Basic earnings per share: (a)
Net income attributable to common shareholders $ 4.51 $ 6.96 $ 3.91
Weighted average common shares outstanding:
Basic 182 182 181"""

CASH = """(in thousands)
Fiscal 2022 2021 2020
Net cash provided by
operating activities 5,000 4,000 3,000
Capital expenditures $ (2,297) $ (1,764) $ (1,822)
Dividends paid(a) (400) (390) (380)
Footnoted widget 12 (b) 15 (b)"""


def _by(facts, field, period=None):
    return [f for f in facts if f["field"] == field and (period is None or f["period"] == period)]


def _run(text, page=1):
    return fd.facts_from_page_text([(page, text)], doc_hash="h" * 64, doc_id="D")


def test_two_year_balance_sheet_dash_parens_unit():
    f = _run(BALANCE)
    ta = {x["period"]: x for x in _by(f, "total_assets")}
    assert ta["2022"]["value"] == 27787.0 and ta["2021"]["value"] == 26075.0
    assert ta["2022"]["unit"] == "USD_millions" and ta["2022"]["confidence"] == "high"
    assert ta["2022"]["quote"] == "Total assets $ 27,787 $ 26,075"
    assert ta["2022"]["extractor"] == "l3_textrow.v1" and ta["2022"]["page"] == 1
    assert ta["2022"]["doc_hash"] == "h" * 64
    assert {x["period"]: x["value"] for x in _by(f, "total_current_assets")} == {
        "2022": 1250.0, "2021": 1554.0}
    dep = {x["period"]: x["value"] for x in _by(f, "accumulated_depreciation")}
    assert dep == {"2022": -6513.0, "2021": -6329.0}
    held = {x["period"]: x["value"] for x in _by(f, "assets_held_for_sale")}
    assert held == {"2022": 0.0, "2021": 683.0}  # em-dash is zero
    assert _by(f, "assets_held_for_sale")[0]["confidence"] == "low"  # label-only
    # "of $60 and $75, respectively 334 271": only the trailing two numbers are columns
    ar = [x for x in f if x["field"].startswith("accounts_receivable")]
    assert sorted(x["value"] for x in ar) == [271.0, 334.0]


def test_three_year_header_and_per_share_is_not_net_income():
    f = _run(INCOME)
    rev = {x["period"]: x["value"] for x in _by(f, "revenue")}
    assert rev == {"2022": 3792.0, "2021": 3930.0, "2020": 3777.0}
    ni = {x["period"]: x["value"] for x in _by(f, "net_income_attributable")}
    assert ni == {"2022": 820.0, "2021": 1263.0, "2020": 709.0}
    eps = {x["period"]: x for x in _by(f, "eps_basic")}
    assert eps["2022"]["value"] == 4.51 and eps["2022"]["unit"] == "USD_per_share"
    assert eps["2022"]["confidence"] != "high"
    gain = {x["period"]: x["value"] for x in _by(f, "gain_on_sale_of_businesses")}
    assert gain["2020"] == 0.0
    assert {x["period"]: x["value"] for x in _by(f, "basic")}["2020"] == 181.0


def test_wrapped_label_parentheses_footnotes_thousands():
    f = _run(CASH)
    ocf = {x["period"]: x for x in _by(f, "operating_cash_flow")}
    assert ocf["2022"]["value"] == 5000.0 and ocf["2022"]["unit"] == "USD_thousands"
    assert ocf["2022"]["quote"] == "Net cash provided by\noperating activities 5,000 4,000 3,000"
    assert ocf["2022"]["line_start"] == 3 and ocf["2022"]["line_end"] == 4
    capex = {x["period"]: x["value"] for x in _by(f, "capex")}
    assert capex == {"2022": -2297.0, "2021": -1764.0, "2020": -1822.0}
    div = {x["period"]: x["value"] for x in _by(f, "dividends_paid")}
    assert div == {"2022": -400.0, "2021": -390.0, "2020": -380.0}
    w = _by(f, "footnoted_widget")
    assert w == [] or all(x["confidence"] == "low" for x in w)


def test_exact_matching_no_fuzzy_and_no_headerless_period():
    assert fd.textrow_field("Total Assets:") == ("total_assets", True)
    assert fd.textrow_field("Total assets held in trust")[1] is False
    assert fd.textrow_field("Net earnings (loss) attributable to Acme") == (
        "net_income_attributable", True)
    f = _run("Total assets 10 20\nDeferred year 1 2022 5")
    assert f[0]["period"] is None and f[0]["unit"] == "unknown" and f[0]["confidence"] == "medium"
    assert _run("Total assets 10")==[]  # a single number is not a row


def test_stale_header_dropped_and_date_year_not_a_column():
    f = fd.facts_from_page_text(
        [(1, "(In millions)\n2022 2021\nTotal assets 1 2"), (9, "Total assets 3 4")],
        doc_hash="h")
    late = [x for x in f if x["page"] == 9]
    assert all(x["period"] is None and x["unit"] == "unknown" for x in late)
    g = _run("(In millions)\n2022 2021\nBalance at December 31, 2021 1,000 2,000")
    assert sorted(x["value"] for x in g) == [1000.0, 2000.0]
    assert len(g) == 2


ODL_PARAGRAPHS = """# American Water Works Company, Inc. Consolidated Balance Sheets (In millions, except share and per share data)

December 31, 2022 December 31, 2021

ASSETS Property, plant and equipment $ 29,736 $ 27,413 Accumulated depreciation (6,513) (6,329) Current assets:

Cash and cash equivalents 85 116 Restricted funds 32 20 Assets held for sale — 683

Total current assets 1,250 1,554 Regulatory and other long-term assets:

Goodwill 1,143 1,139 Total assets $ 27,787 $ 26,075 The accompanying notes are an integral part of these statements. 79"""


def test_odl_shape_rows_joined_into_one_paragraph():
    f = _run(ODL_PARAGRAPHS)
    by = {(x["field"], x["period"]): x for x in f}
    assert by[("total_assets", "2022")]["value"] == 27787.0
    assert by[("total_assets", "2021")]["quote"] == "Total assets $ 27,787 $ 26,075"
    assert by[("cash_and_equivalents", "2022")]["value"] == 85.0
    assert by[("total_current_assets", "2021")]["value"] == 1554.0
    assert by[("assets_held_for_sale", "2022")]["value"] == 0.0
    assert by[("accumulated_depreciation", "2021")]["value"] == -6329.0
    assert by[("total_assets", "2022")]["confidence"] == "high"
    assert by[("total_assets", "2022")]["unit"] == "USD_millions"
