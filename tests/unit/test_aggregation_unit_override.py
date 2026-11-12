"""L4d unit cross-check (2026-11-12): a model-chosen unit label is checked against the filing's declared scale.

Origin: AGG-1.4 r2 -- ``unit: "USD"`` on 13 of 13 ``long_term_debt`` facts whose value is the number printed in a table
declared ``(in millions)``; ``reconcile`` scaled them by 1e-6 and the scorer (no unit leniency) rejected all of them.
Fake backend only: no model, no network.
"""

from __future__ import annotations

import json

import pytest

from src.aggregation import FLAG_ENV, facts_llm
from src.aggregation.store import CorpusStore
from src.models.document import DocumentPreprocessResult, Section

# the six AGG-1.4 r2 facts (research store ``agg14-backfill-cpu-20261111-r2``): quote as stored, what the model labelled,
# the header the filing declares ahead of the line, and the gold value in USD millions (``agg14_cells.csv``).
R2_SIX = [
    ("LOCKHEEDMARTIN_2022_10K", "Long-term debt, net 15,429", 15429.0, "(In millions)", 15429.0),
    ("PEPSICO_2022_10K", "Long-Term Debt Obligations 35,657", 35657.0, "(in millions)", 35657.0),
    ("AMD_2022_10K", "Long-term debt, net of current portion 2,467", 2467.0, "(In millions)", 2467.0),
    ("COSTCO_2022_10K", "Long-term debt, excluding current portion $ 6,484", 6484.0, "(amounts in millions)", 6484.0),
    ("JOHNSON_JOHNSON_2022_10K", "Total long-term debt $ 26,888 $ 29,985", 26888.0, "(Dollars in Millions)", 26888.0),
    # MGM prints USD thousands (gold 7,432.817 millions). The r2 model had already restated it to millions
    # (value 7432.817) and still said "USD"; copying the printed 7,432,817 with that label is the same defect.
    ("MGMRESORTS_2022_10K", "Long-term debt, net 7,432,817", 7432.817, "(In thousands)", 7432.817),
    ("MGMRESORTS_2022_10K", "Long-term debt, net 7,432,817", 7432817.0, "(In thousands)", 7432.817),
]


class Backend:
    def __init__(self, rows):
        self.rows, self.calls = rows, 0

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        self.calls += 1
        if grammar == facts_llm.GATE_GBNF:
            return {"text": "yes", "prompt_tokens": 1, "cache_n": 0}
        return {"text": json.dumps(self.rows), "prompt_tokens": 100, "cache_n": 0}


def _chunk(header: str, quote: str, doc: str = "D") -> dict:
    text = f"CONSOLIDATED BALANCE SHEETS\n{header}\nTotal assets 1,000 900\n{quote} 1\nNotes 2 3\n"
    text = text.replace(f"{quote} 1", quote)
    return {"chunk_id": "c1#w0", "doc_hash": "h", "doc_id": doc, "entity": doc, "page_start": 5, "text": text, "sim": 1.0}


def _fill(chunk: dict, row: dict):
    be = Backend([row])
    rcpt = facts_llm.StageReceipt("x")
    return facts_llm.extract(chunk, "long_term_debt", "desc", be, n_predict=64, id_slot=None, receipt=rcpt)


@pytest.mark.parametrize("doc,quote,printed,header,gold", R2_SIX)
def test_r2_facts_label_usd_reconcile_to_gold(doc, quote, printed, header, gold):
    ch = _chunk(header, quote, doc)
    facts = _fill(ch, {"period": "2022-12-31", "value": printed, "unit": "USD", "quote": quote, "confidence": "high"})
    assert len(facts) == 1 and facts[0]["declared_scale"] in ("USD_millions", "USD_thousands")
    kept, conflicts = facts_llm.reconcile(facts)
    assert not conflicts and len(kept) == 1
    assert kept[0]["unit"] == "USD_millions"
    assert kept[0]["value"] == pytest.approx(gold, rel=1e-9)
    ov = kept[0]["unit_override"]
    assert ov["label"] == "USD" and ov["basis"] == "table_header" and ov["raw_value"] == printed
    assert ov["value_is"] == ("restated_to_millions" if printed == 7432.817 else "as_printed")


def test_r2_pre_fix_behaviour_is_what_the_diag_saw():
    """No declared scale (an old-style fact): reconcile is exactly the pre-fix path -> 2,467 becomes 0.002467."""
    f = {"fact_id": "a", "doc_id": "AMD", "field": "long_term_debt", "period": "2022", "value": 2467.0, "unit": "USD",
         "quote": "Long-term debt, net of current portion 2,467", "confidence": "high"}
    kept, _ = facts_llm.reconcile([f])
    assert kept[0]["value"] == pytest.approx(0.002467) and "unit_override" not in kept[0]


def test_correct_label_is_untouched():
    ch = _chunk("(in millions)", "Total long-term debt 26,888")
    facts = _fill(ch, {"period": "2022", "value": 26888, "unit": "USD_millions", "quote": "Total long-term debt 26,888",
                       "confidence": "high"})
    kept, _ = facts_llm.reconcile(facts)
    assert kept[0]["value"] == 26888 and "unit_override" not in kept[0]


def test_full_dollar_value_with_usd_label_is_not_overridden():
    """The model restated to dollars (value is NOT the printed number): the label is right, keep it."""
    ch = _chunk("(in millions)", "Total long-term debt 26,888")
    facts = _fill(ch, {"period": "2022", "value": 26888000000, "unit": "USD", "quote": "Total long-term debt 26,888",
                       "confidence": "high"})
    kept, _ = facts_llm.reconcile(facts)
    assert kept[0]["value"] == pytest.approx(26888.0) and "unit_override" not in kept[0]


def test_implausible_restatement_is_not_overridden():
    """Prose in full dollars after an (in millions) header: 6,590,000,000 'millions' is nonsense -> keep the label."""
    q = "long-term debt with fixed interest rates was $6,590,000,000"
    ch = _chunk("(in millions)", q)
    facts = _fill(ch, {"period": "2022", "value": 6590000000, "unit": "USD", "quote": q, "confidence": "high"})
    kept, _ = facts_llm.reconcile(facts)
    assert kept[0]["value"] == pytest.approx(6590.0) and "unit_override" not in kept[0]


def test_quote_span_scale_beats_a_table_header():
    q = "Long-term debt was $6.5 billion"
    ch = _chunk("(in millions)", q)
    facts = _fill(ch, {"period": "2022", "value": 6.5, "unit": "USD_millions", "quote": q, "confidence": "high"})
    assert facts[0]["declared_scale"] == "USD_billions" and facts[0]["declared_scale_basis"] == "quote_span"
    kept, _ = facts_llm.reconcile(facts)
    assert kept[0]["value"] == pytest.approx(6500.0)
    assert kept[0]["unit_override"]["declared"] == "USD_billions"


def test_no_header_means_no_override():
    q = "Long-term debt, net 15,429"
    ch = {"chunk_id": "c#w0", "doc_hash": "h", "doc_id": "D", "page_start": 1, "text": q + "\n", "sim": 1.0}
    facts = _fill(ch, {"period": "2022", "value": 15429, "unit": "USD", "quote": q, "confidence": "high"})
    assert facts[0]["declared_scale"] is None
    kept, _ = facts_llm.reconcile(facts)
    assert kept[0]["value"] == pytest.approx(0.015429) and "unit_override" not in kept[0]


def test_non_money_units_are_ignored():
    f = {"fact_id": "a", "doc_id": "d", "field": "f", "period": "2022", "value": 12.5, "unit": "percent",
         "quote": "12.5%", "confidence": "high", "declared_scale": "USD_millions"}
    kept, _ = facts_llm.reconcile([f])
    assert kept[0]["value"] == 12.5 and kept[0]["unit"] == "percent"


def test_header_after_the_quote_is_not_declared_for_it():
    q = "Long-term debt, net 15,429"
    scale, basis = facts_llm.declared_scale(f"{q}\n(In millions)\nx", 0, q)
    assert (scale, basis) == (None, None)


def test_fill_reads_the_header_from_the_parent_chunk_and_reports_overrides(monkeypatch):
    """The window starts BELOW the header (the real PepsiCo layout): ``fill`` looks the quote up in the stored chunk."""
    monkeypatch.setenv(FLAG_ENV, "1")
    quote = "Long-Term Debt Obligations 35,657"
    body = "(in millions)\n" + "filler line of table text\n" * 5 + quote + "\nother 1\n"
    st = CorpusStore()
    st.ingest(DocumentPreprocessResult(
        original_path="/x/a.pdf", figures=[], total_pages=10, tables=[],
        sections=[Section(id="s1", title="Contractual obligations", level=1, content=body, page_start=61, page_end=61)]),
        corpus_id="c", doc_id="PEPSICO_2022_10K", entity="PEPSICO")
    chunk = st.chunks("c")[0]
    cut = chunk["text"].index(quote)
    window = dict(chunk, chunk_id=f"{chunk['chunk_id']}#w1", text=chunk["text"][cut:], sim=1.0)
    be = Backend([{"period": "2022", "value": 35657, "unit": "USD", "quote": quote, "confidence": "high"}])
    out = facts_llm.LazyFieldBackfill(st, be).fill("c", "long_term_debt", "desc", [window])
    assert out["facts"][0]["value"] == pytest.approx(35657.0)
    assert out["unit_overrides"] == [{"fact_id": out["facts"][0]["fact_id"], "doc_id": window["doc_id"],
                                      "period": "2022", "label": "USD", "declared": "USD_millions",
                                      "to": "USD_millions", "value_is": "as_printed", "basis": "table_header",
                                      "raw_value": 35657}]
    stored = [f for f in st.facts("c") if f["field"] == "long_term_debt"]
    assert stored[0]["value"] == pytest.approx(35657.0) and stored[0]["unit"] == "USD_millions"
