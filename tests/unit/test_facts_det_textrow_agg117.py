"""AGG-1.17 opt-in ``l3_textrow`` label components (``L3_TEXTROW_OPTIONAL``): real-shaped rows of the held-out long_term_debt misses."""

from __future__ import annotations

import pytest

from src.aggregation import facts_det as fd

V2 = fd.L3_PERIOD_V2
DFLT = list(fd.L3_PERIOD_COMPONENTS)

BLOCK = ("CONSOLIDATED BALANCE SHEETS (In thousands, except share and per share data)\n\nDecember 31, 2022 2021\n\n"
         "Total current liabilities 8,434,516 6,535,700 Deferred tax liabilities 132,498 15,236 Warehouse funding facilities, non-current "
         "877,066 Long-term debt (Note 15) 4,109,829 4,559,208 Operating lease liabilities, non-current 357,419 395,017")
NIKE = ("## CONSOLIDATED BALANCE SHEETS\n\nMAY 31, (In millions) 2022 2021 ASSETS Current assets:\n\n"
        "| Cash and equivalents $ 8,574 | $ 9,889 |\n|---|---|\n| Long-term debt 8,920 | 9,413 |\n| Inventories 8,420 | 6,854 |")
QUALIFIED = ("(in millions) 2022 2021\n"
             "Long-term debt, classified as current 2,268 4,481\n"
             "Long-term debt (includes $10.31 billion and $1.82 billion related to VIEs at respective dates) 47,742 38,225\n"
             "Long-term debt, less current portion 6,340 6,186\n"
             "Long-term debt and obligations under finance leases 446 451\n"
             "Long-term debt due within one year 2,803 3,115\n"
             "Current portion of long-term debt 14 5")


def rows(text: str, comps: list[str] | None = None, *, version: str | None = V2) -> list[tuple]:
    c = None if comps is None else (DFLT if version else []) + comps
    fs = fd.facts_from_page_text([(1, text)], doc_hash="h", doc_id="D", period_version=version, components=c)
    return [(f["field"], f["period"], f["value"]) for f in fs]


def ltd(text: str, comps: list[str] | None = None, **kw) -> list[tuple]:
    return [r for r in rows(text, comps, **kw) if r[0] == "long_term_debt"]


def test_default_is_unchanged_and_the_components_are_not_in_any_default():
    assert ltd(QUALIFIED, None) == [] and ltd(BLOCK, None) == [] and ltd(NIKE, None) == []
    assert not set(fd.L3_TEXTROW_OPTIONAL) & (set(fd.L3_PERIOD_COMPONENTS) | set(fd.L3_PERIOD_OPTIONAL))


def test_ltd_qualifier_admits_the_agg14_qualifiers_and_a_parenthetical_never_a_current_portion():
    got = ltd(QUALIFIED, ["ltd_qualifier"])
    assert got == [("long_term_debt", "2022", 47742.0), ("long_term_debt", "2021", 38225.0),
                   ("long_term_debt", "2022", 6340.0), ("long_term_debt", "2021", 6186.0),
                   ("long_term_debt", "2022", 446.0), ("long_term_debt", "2021", 451.0)]


def test_textrow_field_qualifier_table():
    assert fd.textrow_field("Long-term debt (Note 15)", ltd_qualifier=True) == ("long_term_debt", True)
    assert fd.textrow_field("Long-term debt, net of current portion", ltd_qualifier=True) == ("long_term_debt", True)
    assert fd.textrow_field("Long-term debt, net of current portion")[1] is False            # opt-in only
    for cap in ("Long-term debt, classified as current", "Current portion of long-term debt", "Long-term debt due within one year",
                "Total long-term debt", "Investments in long-term debt securities"):
        assert fd.textrow_field(cap, ltd_qualifier=True)[0] != "long_term_debt", cap


def test_note_ref_and_tail_label_recover_the_block_row():
    assert ltd(BLOCK, ["ltd_qualifier"]) == []                                                  # "15)" is read as a column
    assert ltd(BLOCK, ["ltd_qualifier", "note_ref"]) == []                                       # label still carries "877,066"
    got = [f for f in fd.facts_from_page_text([(1, BLOCK)], doc_hash="h", period_version=V2,
                                              components=DFLT + ["ltd_qualifier", "note_ref", "tail_label"])
           if f["field"] == "long_term_debt"]
    assert [(f["period"], f["value"], f["unit"]) for f in got] == [("2022", 4109829.0, "USD_thousands"), ("2021", 4559208.0, "USD_thousands")]
    assert got[0]["quote"] == "Long-term debt (Note 15) 4,109,829 4,559,208"


def test_tail_label_needs_a_canonical_tail():
    t = "December 31, 2022 2021\nWidgets, net 5 Gadget reserve 7 8"
    assert [r for r in rows(t, ["tail_label"]) if r[0] == "gadget_reserve"] == []


def test_pipe_cells_and_pipe_cells_ltd():
    assert ltd(NIKE, ["pipe_cells"]) == [("long_term_debt", "2022", 8920.0), ("long_term_debt", "2021", 9413.0)]
    assert ("inventories", "2022", 8420.0) in rows(NIKE, ["pipe_cells"])
    narrow = rows(NIKE, ["pipe_cells_ltd"])
    assert [r for r in narrow if r[0] == "long_term_debt"] == [("long_term_debt", "2022", 8920.0), ("long_term_debt", "2021", 9413.0)]
    assert not [r for r in narrow if r[0] == "inventories"]                                      # other pipe rows are not emitted


def test_textrow_components_are_legal_on_the_legacy_extraction_and_others_are_refused():
    assert ltd(QUALIFIED, ["ltd_qualifier"], version=None)[:2] == [("long_term_debt", "2022", 47742.0), ("long_term_debt", "2021", 38225.0)]
    with pytest.raises(ValueError):
        fd.facts_from_page_text([(1, QUALIFIED)], doc_hash="h", period_version=None, components=["first_column"])
