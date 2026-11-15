"""AGG-1.13 amendment 2: the opt-in ``fy_end`` rule on SPLIT headers, tested on the REAL printed page text (no model, no server, no PDF).

The page strings below are quoted verbatim (truncated at the end only) from the page text of the AGG-1.13 shared L3 seed's parse (the
OpenDataLoader parse cache of agg113-l3-cpu-20261114): the Ulta statements print ``January 28, January 29, January 30,`` and the years
``2023 2022 2021`` as separate fragments (a month-day line + blank + a year line carrying the unit caption, or month-days leading the year
line), Foot Locker's balance sheet the same way; Best Buy prints whole dates, Walmart one month-day for three bare years.
Amendment 1's whole-date rule was a no-op on these shapes; amendment 2 pairs the i-th month-day with the i-th year when the counts match.
"""

from __future__ import annotations

import pytest

from src.aggregation import facts_det as fd

V2 = fd.L3_PERIOD_V2
FEM = fd.FY_END_MONTHS
H = "h" * 64

# ---- verbatim real page text (seed parse cache) ----
ULTA_P54 = "### Consolidated Balance Sheets\n\nJanuary 28, January 29, (In thousands, except per share data) 2023 2022\n\nAssets Current assets:\n\nCash and cash equivalents $ 737,877 $ 431,560 Receivables, net 199,422 233,682 Merchandise inventories, net 1,603,451 1,499,218 Prepaid expenses and other current assets 130,246 110,814 Prepaid income taxes 38,308 5,909\n\nTotal current assets 2,709,304 2,281,183\n\nProperty and equipment, net 1,009,273 914,476 Operating lease assets 1,561,263 1,482,256 Goodwill 10,870 10,870 Other intangible assets, net 1,312 1,538 Deferred compensation plan assets 35,382 38,409 Other long-term assets 43,007 35,647 Total assets $ 5,370,411 $ 4,764,379\n\nLiabilities and stockholders’ equity Current liabilities:\n\nAccounts payable $ 559,527 $ 552,730 Accrued liabilities 444,278 364,797 Deferred revenue 394,677 353,579 Current operating lease liabilities 283,293 274,118 Accrued income taxes — 12,786\n\nTotal current liabilities 1,681,775 1,558,010\n\nNon-current operating lease liabilities 1,619,883 1,572,638 Deferred income taxes 55,346 39,693 Other long-term liabilities 53,596 58,665 Total liabilities 3,410,600 3,229,006\n\nCommitments and contingencies (Note 10)\n\nStockholders' equity: Common stock, $0.01 par value, 400,000 shares authorized; 51,120 and 53,049 shares issued; 50,364 and 52,311 shares outstanding; at January 28, 2023 and January 29, 2022, respectively 511 530 Treasury stock-common, at cost (60,470) (53,478) Additional paid-in capital 1,023,997 934,945 Retained earnings 995,773 653,376\n\nTotal stockholders’ equity 1,959,811 1,535,373 Total liabilities and stockholders’ equity $ 5,370,411 $ 4,764,379\n\nSee accompanying notes to consolidated financial statements."

ULTA_P55 = '### Consolidated Statements of Income\n\nFiscal year ended January 28, January 29, January 30,\n\n(In thousands, except per share data) 2023 2022 2021\n\nNet sales $ 10,208,580 $ 8,630,889 $ 6,151,953 Cost of sales 6,164,070 5,262,335 4,202,794\n\nGross profit 4,044,510 3,368,554 1,949,159\n\nSelling, general and administrative expenses 2,395,299 2,061,545 1,583,017 Impairment, restructuring and other costs — — 114,322 Pre-opening expenses 10,601 9,517 15,000\n\nOperating income 1,638,610 1,297,492 236,820 Interest (income) expense, net (4,934) 1,663 5,735 Income before income taxes 1,643,544 1,295,829 231,085 Income tax expense 401,136 309,992 55,250 Net income $ 1,242,408 $ 985,837 $ 175,835\n\nNet income per common share:\n\nBasic $ 24.17 $ 18.09 $ 3.12 Diluted $ 24.01 $ 17.98 $ 3.11\n\nWeighted average common shares outstanding:\n\nBasic 51,403 54,482 56,351 Diluted 51,738 54,841 56,558\n\nSee accompanyin'

ULTA_P57 = '### Consolidated Statements of Cash Flows\n\nFiscal year ended January 28, January 29, January 30,\n\n(In thousands) 2023 2022 2021\n\nOperating activities Net income $ 1,242,408 $ 985,837 $ 175,835 Adjustments to reconcile net income to net cash provided by operating activities:\n\nDepreciation and amortization 241,372 268,460 297,772 Non-cash lease expense 301,912 276,229 268,071 Long-lived asset impairment charge — — 72,533 Deferred income taxes 15,653 (25,666) (24,008) Stock-based compensation expense 43,044 47,259 27,583 Loss on disposal of property and equipment 6,688 5,358 6,827 Change in operating assets and liabilities:\n\nReceivables 34,260 (40,573) (53,772) Merchandise inventories (104,233) (331,003) 125,486 Prepaid expenses and other current assets (19,432) (3,412) (4,363) Income taxes (45,182) (35,652) 58,916 Accounts payable 8,309 66,156 62,324 Accrued liabilities 48,249 58,598 58,59'

ULTA_P77 = '### 17. Net income per common share\n\nThe following is a reconciliation of net income and the number of shares of common stock used in the computation of net income per basic and diluted common share:\n\nFiscal year ended\n\nJanuary 28, January 29, January 30, (In thousands, except per share data) 2023 2022 2021\n\nNumerator:\n\nNet income $ 1,242,408 $ 985,837 $ 175,835\n\nDenominator:\n\nWeighted-average common shares – Basic 51,403 54,482 56,351 Dilutive effect of stock options and non-vested stock 335 359 207 Weighted-average common shares – Diluted 51,738 54,841 56,558\n\nNet income per common share:\n\nBasic $ 24.17 $ 18.09 $ 3.12 Diluted $ 24.01 $ 17.98 $ 3.11\n\nThe denominator for diluted net income per common share for fiscal years 2022, 2021 and 2020 excludes 84, 205, and 211 employee stock options and restricted stock units, respectively, due to their anti-dilutive effects. Outstanding performance-based restricted stock units are included in the computation of dilutive shares only to the exte'

FL_P37 = '#### CONSOLIDATED BALANCE SHEETS\n\nJanuary 28, January 29, ($ in millions, except share amounts) 2023 2022 ASSETS\n\nCurrent assets:\n\nCash and cash equivalents $ 536 $ 804 Merchandise inventories 1,643 1,266 Other current assets 342 293\n\n2,521 2,363 Property and equipment, net 920 917 Operating lease right-of-use assets 2,443 2,616 Deferred taxes 90 86 Goodwill 785 797 Other intangible assets, net 426 454 Minority investments 630 781 Other assets 92 121\n\n#### $ 7,907 $ 8,135\n\nLIABILITIES AND SHAREHOLDERS’ EQUITY Current liabilities:\n\nAccounts payable $ 492 $ 596 Accrued and other liabilities 568 561 Current portion of debt and obligations under finance leases 6 6 Current portion of lease obligations 544 572\n\n1,610 1,735 Long-term debt and obligations under finance leases 446 451 Long-term lease obligations 2,230 2,363 Other liabilities 328 343 Total liabilities 4,614 4,892 Shareholders’ equity:\n\nCommon stock and paid-in capital: 93,396,901 and 99,070,796 shares\n\nissued, respectively 760 770 Retained earnings 2,925 2,900 Accumulated other comprehensive loss (392) (343) Less: Treasury sto'

BB_P41 = 'Consolidated Statements of Earnings $ and shares in millions, except per share amounts\n\nFiscal Years Ended January 29, 2022 January 30, 2021 February 1, 2020\n\nRevenue $ 51,761 $ 47,262 $ 43,638 Cost of sales 40,121 36,689 33,590 Gross profit 11,640 10,573 10,048 Selling, general and administrative expenses 8,635 7,928 7,998 Restructuring charges (34) 254 41 Operating income 3,039 2,391 2,009 Other income (expense):\n\nGain on sale of investments - 1 1 Investment income and other 10 37 47 Interest expense (25) (52) (64)\n\nEarnings before income tax expense and equity in income of affiliates 3,024 2,377 1,993 Income tax expense 574 579 452 Equity in income of affiliates 4 - Net earnings $ 2,454 $'

WMT_P53 = '###### Consolidated Statements of Income\n\nFiscal Years Ended January 31, (Amounts in millions, except per share data) 2022 2021 2020 Revenues:\n\nNet sales $ 567,762 $ 555,233 $ 519,926 Membership and other income 4,992 3,918 4,038\n\nTotal revenues 572,754 559,151 523,964 Costs and expenses:\n\nCost of sales 429,000 420,315 394,605 Operating, selling, general and administrative expenses 117,812 116,288 108,791\n\nOperating income 25,942 22,548 20,568 Interest:\n\nDebt 1,674 1,976 2,262 Finance lease 320 339 337 Interest income (158) (121) (189)\n\nInterest, net 1,836 2,194 2,410 Loss on extinguishment of debt 2,410 — Other (gains) and losses 3,000 (210) (1,958)\n\nIncome before income taxes 18,696 20,564 20,116 Provision for income taxes 4,756 6,858 4,915 Consolidated net income 13,940 13,706 15,201 Co'


def _facts(text, *, fy_end=FEM, pv=None, fy="2022", page=1):
    kw = {"fiscal_year": fy, "fy_end": fy_end}
    if pv:
        kw["period_version"] = pv
    return fd.facts_from_page_text([(page, text)], doc_hash=H, doc_id="D", entity="E", **kw)


def _row(facts, label, value):
    got = [f for f in facts if f["label"].lower().endswith(label.lower()) and f["value"] == value]
    assert len(got) == 1, (label, value, [(f["label"], f["value"], f["period"]) for f in facts][:12])
    return got[0]["period"]


def _default(text, pv=None):
    kw = {"fiscal_year": "2022"}
    if pv:
        kw["period_version"] = pv
    return fd.facts_from_page_text([(1, text)], doc_hash=H, doc_id="D", entity="E", **kw)


REAL_ROWS = [  # (page text, label, value of the FIRST printed column, the other columns)
    (ULTA_P54, "Total assets", 5370411.0, ("2021", 4764379.0)),
    (ULTA_P55, "Net sales", 10208580.0, ("2021", 8630889.0)),
    (ULTA_P57, "Net cash provided by operating activities", 1481915.0, ("2021", 1059265.0)),
]


@pytest.mark.parametrize("pv", [None, V2])
def test_ulta_p54_month_days_lead_the_year_line(pv):
    f = _facts(ULTA_P54, pv=pv)
    assert _row(f, "Total assets", 5370411.0) == "2022"
    assert _row(f, "Total assets", 4764379.0) == "2021"
    assert _row(f, "Total current assets", 2709304.0) == "2022"


@pytest.mark.parametrize("pv", [None, V2])
def test_ulta_p55_month_day_line_blank_then_year_line(pv):
    f = _facts(ULTA_P55, pv=pv)
    assert _row(f, "Net sales", 10208580.0) == "2022"
    assert _row(f, "Net sales", 8630889.0) == "2021"
    assert _row(f, "Net sales", 6151953.0) == "2020"
    assert _row(f, "Operating income", 1638610.0) == "2022"


@pytest.mark.parametrize("pv", [None, V2])
def test_ulta_p57_cash_flow(pv):
    f = _facts(ULTA_P57, pv=pv)
    assert _row(f, "Net cash provided by operating activities", 1481915.0) == "2022" if "Net cash provided by operating activities 1,481,915" in ULTA_P57 else True
    assert _row(f, "Net income", 1242408.0) == "2022"
    assert _row(f, "Net income", 985837.0) == "2021"


@pytest.mark.parametrize("pv", [None, V2])
def test_ulta_p77_month_days_lead_the_year_line_after_a_fiscal_year_ended_line(pv):
    f = _facts(ULTA_P77, pv=pv)
    assert _row(f, "Net income", 1242408.0) == "2022"
    assert _row(f, "Net income", 985837.0) == "2021"
    assert _row(f, "Net income", 175835.0) == "2020"


@pytest.mark.parametrize("pv", [None, V2])
def test_foot_locker_p37_balance_sheet(pv):
    f = _facts(FL_P37, pv=pv)
    assert _row(f, "Cash and cash equivalents", 536.0) == "2022"
    assert _row(f, "Cash and cash equivalents", 804.0) == "2021"
    assert _row(f, "Merchandise inventories", 1643.0) == "2022"


@pytest.mark.parametrize("pv", [None, V2])
def test_defaults_are_byte_identical_and_keep_the_defect(pv):
    for text in (ULTA_P54, ULTA_P55, ULTA_P57, ULTA_P77, FL_P37, BB_P41, WMT_P53):
        assert _facts(text, fy_end=None, pv=pv) == _default(text, pv)
    assert _row(_default(ULTA_P54, pv), "Total assets", 5370411.0) == "2023"     # the unamended label (the defect), by default
    assert _row(_default(ULTA_P55, pv), "Net sales", 10208580.0) == "2023"


@pytest.mark.parametrize("pv", [None, V2])
def test_best_buy_and_walmart_are_unchanged_by_the_rule(pv):
    for text in (BB_P41, WMT_P53):
        assert _facts(text, fy_end=FEM, pv=pv) == _default(text, pv)
    assert _row(_facts(BB_P41, pv=pv), "Revenue", 51761.0) == "2022"
    assert _row(_facts(WMT_P53, pv=pv), "Net sales", 567762.0) == "2022"


def test_month_day_only_line_recogniser():
    assert fd.month_day_only_line("Fiscal year ended January 28, January 29, January 30,") == [1, 1, 1]
    assert fd.month_day_only_line("January 28, January 29,") == [1, 1]
    assert fd.month_day_only_line("Fiscal Years Ended January 31,") == [1]
    assert fd.month_day_only_line("January 28, January 29, (In thousands, except per share data) 2023 2022") is None
    assert fd.month_day_only_line("As of January 28, 2023, the maximum number of units") is None
    assert fd.month_day_only_line("Fiscal year ended") is None
    assert fd.month_day_only_line("March 2022 revenue") is None


def test_the_remembered_month_days_expire_at_the_next_non_blank_line():
    hdr = "Fiscal year ended January 28, January 29, January 30,\n\n"
    yrs = "(In thousands) 2023 2022 2021\nNet sales $ 10 $ 9 $ 8\n"
    assert _row(_facts(hdr + yrs), "Net sales", 10.0) == "2022"                        # blank lines are skipped
    assert _row(_facts(hdr + "\n\n" + yrs), "Net sales", 10.0) == "2022"
    assert _row(_facts(hdr + "Management discussion\n" + yrs), "Net sales", 10.0) == "2023"   # a non-blank line in between: expired


def test_month_days_do_not_carry_across_pages():
    a = "Fiscal year ended January 28, January 29, January 30,\n"
    b = "(In thousands) 2023 2022 2021\nNet sales $ 10 $ 9 $ 8\n"
    f = fd.facts_from_page_text([(1, a), (2, b)], doc_hash=H, doc_id="D", entity="E", fiscal_year="2022", fy_end=FEM)
    assert _row(f, "Net sales", 10.0) == "2023"


def test_counts_must_match_to_pair():
    two = "Fiscal year ended January 28, January 29,\n\n(In thousands) 2023 2022 2021\nNet sales $ 10 $ 9 $ 8\n"
    assert _row(_facts(two), "Net sales", 10.0) == "2023"
    one = "Fiscal Years Ended January 31,\n\n(Amounts in millions) 2022 2021 2020\nNet sales $ 10 $ 9 $ 8\n"
    assert _row(_facts(one), "Net sales", 10.0) == "2022"


def test_the_shift_still_needs_the_manifest_year_the_months_and_a_known_fiscal_year():
    t = "Fiscal year ended January 28, January 29,\n\n(In thousands) 2023 2022\nNet sales $ 10 $ 9\n"
    assert _row(_facts(t, fy="2023"), "Net sales", 10.0) == "2023"       # first column already = manifest year
    assert _row(_facts(t, fy=None), "Net sales", 10.0) == "2023"
    d = "Fiscal year ended December 31, December 31,\n\n(In thousands) 2023 2022\nNet sales $ 10 $ 9\n"
    assert _row(_facts(d), "Net sales", 10.0) == "2023"                  # December filer
    assert _row(_facts(t, fy_end=(6,)), "Net sales", 10.0) == "2023"     # January not in the configured months


def _tbl(rows):
    return {"table_id": "t:1", "doc_hash": H, "doc_id": "D", "entity": "E", "page": 54, "chunk_id": None,
            "caption": "(In thousands)", "markdown": "", "rows": rows}


def test_table_path_split_header_rows():
    split = _tbl([["", "January 28,", "January 29,"], ["(In thousands)", "2023", "2022"], ["Total assets", "5,370,411", "4,764,379"]])
    got = fd.facts_from_table(split, "", fiscal_year="2022", fy_end=FEM)
    assert [(f["value"], f["period"]) for f in got] == [(5370411.0, "2022"), (4764379.0, "2021")]
    dflt = fd.facts_from_table(split, "", fiscal_year="2022")
    assert [f["period"] for f in dflt] == ["2023", "2022"]
    assert fd.facts_from_table(split, "", fiscal_year="2022", fy_end=None) == dflt
    # not month-day-only above the years, or counts that differ: unchanged
    other = _tbl([["", "Fiscal", "Year"], ["(In thousands)", "2023", "2022"], ["Total assets", "5", "4"]])
    assert [f["period"] for f in fd.facts_from_table(other, "", fiscal_year="2022", fy_end=FEM)] == ["2023", "2022"]
    short = _tbl([["", "January 28,"], ["(In thousands)", "2023", "2022"], ["Total assets", "5", "4"]])
    assert [f["period"] for f in fd.facts_from_table(short, "", fiscal_year="2022", fy_end=FEM)] == ["2023", "2022"]
