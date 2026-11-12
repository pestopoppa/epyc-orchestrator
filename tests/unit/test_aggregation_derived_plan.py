"""AGG-1.10 orchestrator additions (zero model, zero server): the ``agg110.v1`` derived-metric plan.

* ``sql_grammar``: pivot catalog, the TEMP ``facts_pivot`` view (nothing written to the database file), the bounded pivot GBNF
  (plain constructs only, checked by translating the grammar to a regex and matching statements), the deterministic mirror
  ``validate_pivot_sql``, the entity-conjunct guard and the null guard;
* ``planner.sql_shape``: the routing record;
* ``agg11.v0`` (``build_sql_gbnf``) is byte-identical to AGG-1.1.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3

import pytest

from src.aggregation import planner, sql_grammar as G


# ----------------------------------------------------------------------------- tiny GBNF -> regex (non-recursive grammars)
def gbnf_to_regex(gbnf: str, root: str = "root") -> "re.Pattern[str]":
    rules = {}
    for line in gbnf.splitlines():
        name, _, body = line.partition(" ::= ")
        rules[name.strip()] = body.strip()
    tok = re.compile(r'"(?:[^"\\]|\\.)*"|\[[^\]]*\]|[A-Za-z_][A-Za-z0-9_-]*|[()|?*+]')

    def conv(name: str, depth: int = 0) -> str:
        assert depth < 40, "recursive grammar"
        toks = tok.findall(rules[name])
        pos = 0

        def alt() -> str:
            nonlocal pos
            parts = [seq()]
            while pos < len(toks) and toks[pos] == "|":
                pos += 1
                parts.append(seq())
            return "(?:" + "|".join(parts) + ")"

        def seq() -> str:
            nonlocal pos
            out = []
            while pos < len(toks) and toks[pos] not in ("|", ")"):
                t = toks[pos]
                pos += 1
                if t == "(":
                    a = alt()
                    assert toks[pos] == ")"
                    pos += 1
                elif t.startswith('"'):
                    a = re.escape(bytes(t[1:-1], "utf-8").decode("unicode_escape"))
                elif t.startswith("["):
                    a = t
                else:
                    a = conv(t, depth + 1)
                while pos < len(toks) and toks[pos] in ("?", "*", "+"):
                    a = f"(?:{a}){toks[pos]}"
                    pos += 1
                out.append(a)
            return "".join(out)

        return alt()

    return re.compile(conv(root))


def _canon_rows():
    rows = []
    data = {"A Corp": {"revenue": 100.0, "net_income": 20.0, "capex": 5.0, "total_current_assets": 50.0,
                       "total_current_liabilities": 25.0},
            "B Corp": {"revenue": 200.0, "net_income": -10.0, "capex": 40.0, "total_current_assets": 30.0,
                       "total_current_liabilities": 60.0},
            "C Corp": {"revenue": 50.0, "net_income": 10.0},           # no capex, no balance sheet
            "McD's": {"revenue": 10.0, "net_income": 1.0, "capex": 1.0}}
    for ent, fl in data.items():
        doc = ent.split()[0].upper().replace("'", "") + "_2022_10K"
        for f, v in fl.items():
            rows.append({"corpus_id": "c", "doc_id": doc, "field": f, "period": "2022", "entity": ent, "value": v,
                         "unit": "USD_millions", "quote": f"{f} {v}", "page": 1})
        rows.append({"corpus_id": "c", "doc_id": doc, "field": "revenue", "period": "2021", "entity": ent, "value": 1.0,
                     "unit": "USD_millions", "quote": "old", "page": 1})
    rows.append({"corpus_id": "c", "doc_id": "A_2022_10K", "field": "12_gains_charges", "period": "2022",
                 "entity": "A Corp", "value": 9.0, "unit": "USD_millions", "quote": "x", "page": 1})
    return rows


@pytest.fixture
def db():
    c = sqlite3.connect(":memory:")
    c.execute("""CREATE TABLE facts_canonical (corpus_id TEXT, doc_id TEXT, field TEXT, period TEXT, fact_id TEXT,
        doc_hash TEXT, entity TEXT, value DOUBLE, unit TEXT, quote TEXT, page INTEGER)""")
    for r in _canon_rows():
        c.execute("INSERT INTO facts_canonical (corpus_id, doc_id, field, period, entity, value, unit, quote, page) "
                  "VALUES (?,?,?,?,?,?,?,?,?)", [r[k] for k in ("corpus_id", "doc_id", "field", "period", "entity", "value",
                                                                  "unit", "quote", "page")])
    return c


@pytest.fixture
def cat():
    return G.pivot_catalog(_canon_rows(), names=["A Corp", "B Corp", "C Corp", "McD's", "JPMorgan"])


def run(db, cat, sql):
    ok, why = G.validate_pivot_sql(sql, cat)
    assert ok, why
    G.create_pivot_view(db, "c", cat)
    return db.execute(G.null_guard(sql, cat)).fetchall()


# ----------------------------------------------------------------------------- catalog and view
def test_catalog_uses_known_fields_not_the_slug_tail_and_drops_quoted_names(cat):
    assert cat.fields == ("capex", "net_income", "revenue", "total_current_assets", "total_current_liabilities")
    assert "McD's" not in cat.entities and "A Corp" in cat.entities and cat.table == "facts_pivot"


def test_pivot_view_is_temp_and_does_not_touch_the_file(tmp_path, cat):
    p = tmp_path / "s.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE facts_canonical (corpus_id, doc_id, field, period, entity, value)")
    c.execute("INSERT INTO facts_canonical VALUES ('c','D','revenue','2022','D',5.0)")
    c.commit()
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    G.create_pivot_view(c, "c", G.PivotCatalog(("revenue",), ()))
    assert c.execute("SELECT revenue FROM facts_pivot").fetchall() == [(5.0,)]
    c.close()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before
    assert "facts_pivot" not in sqlite3.connect(p).execute("SELECT group_concat(name) FROM sqlite_master").fetchone()[0]


def test_pivot_has_one_row_per_document_and_year_with_nulls(db, cat):
    G.create_pivot_view(db, "c", cat)
    rows = {r[1]: r for r in db.execute("SELECT entity, doc_id, period, capex FROM facts_pivot WHERE period = '2022'")}
    assert len(rows) == 4 and rows["C_2022_10K"][3] is None and rows["A_2022_10K"][3] == 5.0
    assert db.execute("SELECT COUNT(*) FROM facts_pivot").fetchone()[0] == 8        # 4 documents x 2 years


# ----------------------------------------------------------------------------- grammar
def test_gbnf_uses_only_plain_constructs(cat):
    g = G.build_pivot_gbnf(cat)
    assert "{" not in g and "}" not in g
    assert g.startswith("root ::= ")


GOOD = [
    "SELECT entity FROM facts_pivot WHERE period = '2022' ORDER BY revenue DESC LIMIT 1",
    "SELECT entity, net_income / revenue FROM facts_pivot WHERE period = '2022' ORDER BY net_income / revenue DESC LIMIT 3",
    "SELECT COUNT(*) FROM facts_pivot WHERE period = '2022' AND net_income / revenue > 0.1",
    "SELECT SUM(net_income) FROM facts_pivot WHERE period = '2022' AND revenue > 90000",
    "SELECT SUM(revenue) FROM facts_pivot WHERE period = '2022' AND net_income < 0 AND entity != 'JPMorgan'",
    "SELECT COUNT(*) FROM facts_pivot WHERE period = '2022' AND total_current_assets / total_current_liabilities > 1.25",
    "SELECT entity, capex / revenue FROM facts_pivot WHERE period = '2022' ORDER BY capex / revenue ASC LIMIT 1",
]
BAD = [
    "SELECT SUM(entity) FROM facts_pivot WHERE period = '2022'",                                  # aggregate over text
    "SELECT SUM(quote) FROM facts_pivot WHERE period = '2022'",
    "SELECT COUNT(*) FROM facts_pivot WHERE period = '2022' AND entity = '3M'",                   # no entity = form
    "SELECT entity FROM facts_canonical WHERE period = '2022'",                                   # other table
    "SELECT entity FROM facts_pivot WHERE revenue > 5",                                           # period is mandatory
    "SELECT entity FROM facts_pivot WHERE period = '2022' ORDER BY revenue DESC LIMIT 1; DROP TABLE x",
    "SELECT COUNT(*) FROM facts_pivot WHERE period = '2022' AND revenue > 0.0500000000000000030000",   # digit runaway
    "SELECT entity, revenue, revenue, revenue, revenue FROM facts_pivot WHERE period = '2022'",    # 4 items
    "SELECT entity FROM facts_pivot WHERE period = '2022' AND revenue > 1 AND revenue > 2 AND revenue > 3 AND revenue > 4 "
    "AND revenue > 5",                                                                            # 5 extra conjuncts
    "SELECT entity FROM facts_pivot WHERE period = '2022' AND entity != 'Nobody Inc'",             # name not in catalog
    "SELECT salary FROM facts_pivot WHERE period = '2022'",
]


@pytest.mark.parametrize("sql", GOOD)
def test_good_statements_pass_both_validator_and_gbnf(sql, cat):
    assert G.validate_pivot_sql(sql, cat) == (True, "ok")
    assert gbnf_to_regex(G.build_pivot_gbnf(cat)).fullmatch(sql)


@pytest.mark.parametrize("sql", BAD)
def test_bad_statements_fail_both_validator_and_gbnf(sql, cat):
    assert not G.validate_pivot_sql(sql, cat)[0]
    assert not gbnf_to_regex(G.build_pivot_gbnf(cat)).fullmatch(sql)


def test_validator_and_grammar_agree_on_a_shared_sample(cat):
    gre = gbnf_to_regex(G.build_pivot_gbnf(cat))
    for sql in GOOD + BAD:
        if ";" in sql:                      # the grammar has no ';'; the validator strips a trailing one only
            continue
        assert (gre.fullmatch(sql) is not None) == G.validate_pivot_sql(sql, cat)[0], sql


def test_v0_grammar_is_byte_identical_to_agg11():
    c0 = G.Catalog(("entity", "doc_id", "period", "field", "value", "unit", "quote", "page"), ("capex", "revenue"),
                   table="facts_canonical")
    assert hashlib.sha256(G.build_sql_gbnf(c0).encode()).hexdigest() == \
        "76a40ac969d55cc2984e647df530a3f7833f567a4fd523f4c10ce69c111c161e"
    assert G.PLANNER_V0 == "agg11.v0" and G.PLANNER_V1 == "agg110.v1"
    assert (G.SQL_N_PREDICT_V0, G.SQL_N_PREDICT_V1) == (160, 256)


# ----------------------------------------------------------------------------- guards
def test_entity_conjunct_guard_rejects_the_aggv11_enumeration():
    enum = ("SELECT COUNT(*) FROM facts_canonical WHERE period = '2022' AND value > 0 AND entity = '3M' AND value > 0 "
            "AND entity = 'AMD' AND value > 0 AND entity = 'Adobe'")
    ok, why = G.entity_conjunct_guard(enum)
    assert not ok and why.startswith("entity_enumeration") and G.entity_conjunct_count(enum) == 3
    assert G.entity_conjunct_guard("SELECT x FROM t WHERE entity = 'A' AND entity != 'B'")[0]
    assert not G.validate_pivot_sql(enum, G.PivotCatalog(("revenue",), ("3M",)))[0]


def test_null_guard_adds_each_referenced_field_before_order_by_and_limit(cat):
    out = G.null_guard("SELECT entity FROM facts_pivot WHERE period = '2022' ORDER BY capex / revenue ASC LIMIT 1", cat)
    assert out == ("SELECT entity FROM facts_pivot WHERE period = '2022' AND capex IS NOT NULL AND revenue IS NOT NULL "
                   "ORDER BY capex / revenue ASC LIMIT 1")
    assert G.null_guard(out, cat) == out                                                           # idempotent


# ----------------------------------------------------------------------------- end to end on a toy store
def test_ratio_argmin_skips_companies_without_the_field(db, cat):
    # C Corp has no capex: without the null guard SQLite would rank its NULL ratio first under ASC
    sql = "SELECT entity FROM facts_pivot WHERE period = '2022' ORDER BY capex / revenue ASC LIMIT 1"
    G.create_pivot_view(db, "c", cat)
    assert db.execute(sql).fetchall()[0][0] != "A Corp"        # unguarded, the NULL ratio of C Corp ranks first
    assert run(db, cat, sql) == [("A Corp",)]


def test_difference_top_and_count_and_filtered_sum(db, cat):
    wc = "total_current_assets - total_current_liabilities"
    assert run(db, cat, f"SELECT entity, {wc} FROM facts_pivot WHERE period = '2022' ORDER BY {wc} DESC LIMIT 3") == \
        [("A Corp", 25.0), ("B Corp", -30.0)]
    assert run(db, cat, "SELECT COUNT(*) FROM facts_pivot WHERE period = '2022' AND net_income / revenue > 0.1") == [(2,)]
    assert run(db, cat, "SELECT SUM(revenue) FROM facts_pivot WHERE period = '2022' AND net_income < 0") == [(200.0,)]
    assert run(db, cat, "SELECT SUM(net_income) FROM facts_pivot WHERE period = '2022' AND revenue > 90 "
                        "AND entity != 'B Corp'") == [(20.0,)]


def test_ratio_is_real_division_not_integer(db, cat):
    assert run(db, cat, "SELECT entity, capex / revenue FROM facts_pivot WHERE period = '2022' ORDER BY capex / revenue "
                        "DESC LIMIT 1") == [("B Corp", 0.2)]


# ----------------------------------------------------------------------------- routing record
@pytest.fixture
def on(monkeypatch):
    from src.aggregation import FLAG_ENV
    monkeypatch.setenv(FLAG_ENV, "1")


@pytest.mark.parametrize("q,shape", [
    ("Which company in the corpus reported the highest FY2022 total revenue?", "single"),
    ("Which company had the highest FY2022 capital expenditures as a share of total revenue?", "pivot"),
    ("How many companies had an FY2022 net margin above 10%?", "pivot"),
    ("What is the combined FY2022 total revenue of the companies that reported a net loss attributable to shareholders?",
     "pivot"),
    ("Among companies with a classified balance sheet, how many had positive working capital at FY2022 year end?", "pivot"),
])
def test_sql_shape(q, shape, on):
    out = planner.sql_shape(q)
    assert out["shape"] == shape and set(out) == {"shape", "fields", "cues"}
