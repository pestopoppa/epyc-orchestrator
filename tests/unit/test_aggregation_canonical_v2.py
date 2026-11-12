"""``canonical.v2`` (AGG-1.9, 2026-11-12): two narrow candidate filters over ``canonical.v1``.

Origin: AGG-1.8 -- 8 of the 211 verified AGG-1.1 cells had a correct candidate in ``facts`` that ``canonical.v1`` passed over,
4 of them net income (the attributable line lost to the consolidated line); a blanket l4-over-l3 tier flip fixed 7 and broke
19, so the guards never reorder tiers. ``canonical.v1`` is the default and must stay byte-identical. No model, no network.
"""

from __future__ import annotations

import pytest

from src.aggregation.store import (CANONICAL_RULE, CANONICAL_RULE_V2, CorpusStore, guard_attributable_net_income,
                                   guard_thousands_unit, is_attributable, select_canonical)


def _f(fid, doc, field, value, extractor, quote=None, *, unit="USD_millions", entity=None, page=1, conf="high",
       period="2022", audit="unaudited"):
    return {"corpus_id": "c", "fact_id": fid, "doc_hash": f"h:{doc}", "doc_id": doc, "entity": entity or doc.title(),
            "period": period, "field": field, "value": value, "unit": unit, "quote": quote or f"{field} {value}",
            "page": page, "confidence": conf, "extractor": extractor, "audit_status": audit}


def _pick(facts, **kw):
    rows, log = select_canonical(facts, **kw)
    return rows[0], log


def test_v1_is_the_default_and_unchanged():
    facts = [_f("a", "PepsiCo", "net_income", 8978.0, "l3_textrow.v1", "Net income 8,978", entity="PepsiCo"),
             _f("b", "PepsiCo", "net_income", 8910.0, "l4_llm.v1", "Net income attributable to PepsiCo $ 8,910",
                entity="PepsiCo")]
    row, log = _pick(facts)
    assert row["value"] == 8978.0 and row["rule"] == CANONICAL_RULE and CANONICAL_RULE.startswith("canonical.v1")
    assert _pick(facts, rule=CANONICAL_RULE)[0] == row
    assert all(r["rule"] == CANONICAL_RULE for r in log)


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        select_canonical([], rule="canonical.v9")


def test_attributable_net_income_beats_the_consolidated_line():
    facts = [_f("a", "Verizon", "net_income", 21748.0, "l3_textrow.v1", "Net Income $ 21,748 $ 22,618", entity="Verizon"),
             _f("b", "Verizon", "net_income", 21256.0, "l4_llm.v1", "Net income attributable to Verizon 21,256",
                entity="Verizon")]
    row, log = _pick(facts, rule=CANONICAL_RULE_V2)
    assert row["value"] == 21256.0 and row["rule"] == CANONICAL_RULE_V2
    assert [r["other_value"] for r in log] == [21748.0] and log[0]["kind"] == "conflict"   # the loser stays on record


def test_attributable_guard_ignores_other_fields_and_other_quantities():
    for quote, entity in [("Net income attributable to noncontrolling interests 12", "Acme"),
                          ("Net income attributable to common shareholders $ 7,400", "Acme"),          # after preferred dividends
                          ("Net income (loss) attributable to Adevinta $ 49", "eBay"),                 # another entity
                          ("Net income from discontinued operations attributable to Acme 5", "Acme"),
                          ("Net income 5", "Acme")]:
        assert not is_attributable({"quote": quote, "entity": entity}), quote
    assert is_attributable({"quote": "Net loss attributable to Boeing Shareholders ($4,935)", "entity": "Boeing"})
    assert is_attributable({"quote": "Net income attributable to MGM Resorts International $ 1,473,093",
                            "entity": "MGM Resorts International"})
    assert is_attributable({"quote": "Net earnings attributable to shareholders 9", "entity": "Acme"})
    rows = [_f("a", "d", "revenue", 100.0, "l3_textrow.v1"), _f("b", "d", "revenue", 90.0, "l4_llm.v1",
                                                               "revenue attributable to the company 90")]
    assert guard_attributable_net_income("revenue", rows)[1] is None          # net_income only


def test_attributable_guard_never_empties_and_leaves_groups_without_one_alone():
    facts = [_f("a", "d", "net_income", 50.0, "l3_textrow.v1", "Net income 50"),
             _f("b", "d", "net_income", 40.0, "l4_llm.v1", "Net income 40")]
    assert _pick(facts, rule=CANONICAL_RULE_V2)[0] == _pick(facts)[0] | {"rule": CANONICAL_RULE_V2}


def test_attributable_plurality_picks_the_agreed_value_among_attributable_candidates():
    q = "Net income (loss) attributable to MGM Resorts International"
    facts = [_f("a", "MGM", "net_income", 1487.247, "l4_llm.v1", q, entity="MGM Resorts International", audit="disputed"),
             _f("b", "MGM", "net_income", 1473.093, "l4_llm.v1", q, entity="MGM Resorts International", audit="disputed",
                page=73),
             _f("c", "MGM", "net_income", 1473.093, "l4_llm.v1", q, entity="MGM Resorts International", audit="disputed",
                page=92),
             _f("d", "MGM", "net_income", 206731.0, "l3_textrow.v1", "Net income (loss) 206,731", unit="USD_thousands",
                entity="MGM Resorts International")]
    assert _pick(facts)[0]["value"] == pytest.approx(206.731)                          # v1: tier first
    assert _pick(facts, rule=CANONICAL_RULE_V2)[0]["value"] == pytest.approx(1473.093)


def test_thousands_guard_drops_only_an_exact_thousandfold_slip_with_two_agreeing_peers():
    good = [_f("a", "d", "capex", 4009.0, "l3_textrow.v1", page=1), _f("b", "d", "capex", 4009.0, "l3_textrow.v1", page=2)]
    slip = _f("c", "d", "capex", 4.009, "l4_llm.v1")
    kept, note = guard_thousands_unit(good + [slip])
    assert note == "thousands_unit" and slip not in kept and len(kept) == 2
    big = _f("e", "d", "capex", 4009000.0, "l4_llm.v1")
    assert big not in guard_thousands_unit(good + [big])[0]
    # one peer is not consensus; a 100x outlier that is not exactly 1000x is NOT a thousands slip (the first draft broke 2
    # retained cells doing that); the group is never emptied
    assert guard_thousands_unit([good[0], slip, _f("x", "d", "capex", 7.0, "l4_llm.v1")])[1] is None
    assert guard_thousands_unit([_f("p", "d", "dividends_paid", 11682.0, "l4_llm.v1"),
                                 _f("q", "d", "dividends_paid", 4.45e-6, "l3_regex.v1", page=1),
                                 _f("r", "d", "dividends_paid", 4.45e-6, "l3_regex.v1", page=2)])[1] is None
    assert guard_thousands_unit(good)[1] is None


def test_v2_choice_flows_through_refresh_canonical_and_v1_is_restored():
    st = CorpusStore(":memory:")
    st.conn.execute("INSERT INTO documents (doc_hash, corpus_id, doc_id, entity, n_tokens_est) VALUES "
                    "('h:V', 'c', 'V', 'Verizon', 1)")
    st.conn.commit()
    st.write_facts([{k: v for k, v in f.items() if k != "corpus_id"} for f in [
        _f("a", "V", "net_income", 21748.0, "l3_textrow.v1", "Net Income $ 21,748", entity="Verizon"),
        _f("b", "V", "net_income", 21256.0, "l4_llm.v1", "Net income attributable to Verizon 21,256", entity="Verizon")]])
    for f in ("a", "b"):
        st.conn.execute("UPDATE facts SET doc_hash = 'h:V' WHERE fact_id = ?", [f])
    st.conn.commit()
    s2 = st.refresh_canonical("c", rule=CANONICAL_RULE_V2)
    assert s2["rule"] == CANONICAL_RULE_V2 and st.canonical_facts("c")[0]["value"] == 21256.0
    assert st.canonical_facts("c")[0]["rule"] == CANONICAL_RULE_V2
    s1 = st.refresh_canonical("c")
    assert s1["rule"] == CANONICAL_RULE and st.canonical_facts("c")[0]["value"] == 21748.0
