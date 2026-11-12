"""Catalog -> GBNF grammar for the ``sql`` path (schema hallucination 0 by construction).

The grammar admits one ``SELECT`` over the ``facts`` table whose column names
and ``field`` literals come only from the catalog. :func:`validate_sql` is the
deterministic mirror used to reject model output or hand-written SQL that the
grammar would not admit (tests exercise the validator; the GBNF string is what
the backend receives).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

AGGS = ("COUNT", "SUM", "AVG", "MIN", "MAX")
OPS = ("=", "!=", ">", "<", ">=", "<=")


@dataclass(frozen=True)
class Catalog:
    columns: tuple[str, ...]          # selectable / filterable facts columns
    fields: tuple[str, ...]           # values of facts.field present in the store
    table: str = "facts"


def catalog_from_facts(facts: Sequence[dict], columns: Sequence[str] = (
        "entity", "doc_id", "period", "field", "value", "unit", "quote", "page")) -> Catalog:
    return Catalog(tuple(columns), tuple(sorted({f["field"] for f in facts if f.get("field")})))


def catalog_from_canonical(rows: Sequence[dict], columns: Sequence[str] = (
        "entity", "doc_id", "period", "field", "value", "unit", "quote", "page")) -> Catalog:
    """Catalog over ``facts_canonical`` (one fact per doc, field and fiscal year).

    The SQL path queries this table, never the raw ``facts`` evidence table, so SUM / COUNT /
    top-k see each (entity, field, period) once. ``period`` is the 4-digit fiscal year and
    ``value`` is USD millions for money fields (``store.select_canonical``).
    """
    return Catalog(tuple(columns), tuple(sorted({r["field"] for r in rows if r.get("field")})),
                   table="facts_canonical")


def _alt(items: Sequence[str]) -> str:
    return " | ".join('"' + i.replace("\\", "\\\\").replace('"', '\\"') + '"' for i in items)


def build_sql_gbnf(cat: Catalog) -> str:
    fields = cat.fields or ("__none__",)
    return "\n".join([
        'root ::= "SELECT " sel " FROM ' + cat.table + '" where? group? order? limit?',
        'sel ::= item (", " item)*',
        'item ::= col | agg "(" (col | "*") ")"',
        f"agg ::= {_alt(AGGS)}",
        f"col ::= {_alt(cat.columns)}",
        'where ::= " WHERE " cond (" AND " cond)*',
        "cond ::= \"field = '\" fieldlit \"'\" | col \" \" op \" \" lit",
        f"fieldlit ::= {_alt(fields)}",
        f"op ::= {_alt(OPS)}",
        "lit ::= num | \"'\" [A-Za-z0-9 _.,&-]* \"'\"",
        'num ::= "-"? [0-9]+ ("." [0-9]+)?',
        'group ::= " GROUP BY " col (", " col)*',
        'order ::= " ORDER BY " (col | agg "(" col ")") (" ASC" | " DESC")?',
        'limit ::= " LIMIT " [0-9]+',
    ])


_TOKEN = re.compile(r"'[^']*'|[A-Za-z_][A-Za-z0-9_]*|-?[0-9]+(?:\.[0-9]+)?|>=|<=|!=|[=<>(),*]")
_KEYWORDS = {"SELECT", "FROM", "WHERE", "AND", "GROUP", "BY", "ORDER", "ASC", "DESC", "LIMIT"}


def validate_sql(sql: str, cat: Catalog) -> tuple[bool, str]:
    """True iff ``sql`` uses only catalog columns/fields, one table, no other statements."""
    s = sql.strip().rstrip(";")
    if ";" in s or "--" in s or "/*" in s:
        return False, "multiple statements or comments"
    toks = _TOKEN.findall(s)
    if "".join(toks).replace(" ", "") != re.sub(r"\s+", "", s):
        return False, "characters outside the grammar"
    if not toks or toks[0].upper() != "SELECT":
        return False, "not a SELECT"
    up = [t.upper() for t in toks]
    if up.count("FROM") != 1 or toks[up.index("FROM") + 1] != cat.table:
        return False, "table not in catalog"
    for i, t in enumerate(toks):
        if t.startswith("'"):
            if i >= 2 and toks[i - 2] == "field" and toks[i - 1] == "=" and t[1:-1] not in cat.fields:
                return False, f"field {t} not in catalog"
            continue
        if re.match(r"[A-Za-z_]", t):
            if t.upper() in _KEYWORDS or t.upper() in AGGS or t == cat.table:
                continue
            if t not in cat.columns:
                return False, f"column {t!r} not in catalog"
    return True, "ok"


# ------------------------------------------------------------------ AGG-1.10: derived-metric plan (planner ``agg110.v1``)
# ``agg11.v0`` (everything above) is the Phase 0 / AGG-1.1 behaviour and is NOT changed: one single-field SELECT over
# ``facts_canonical``. ``agg110.v1`` queries ``facts_pivot`` instead: a TEMP VIEW with one NUMERIC column per canonical field,
# so a ratio or a difference of two fields is one expression over one row (``net_income / revenue``), and the grammar is
# structurally bounded (at most 3 select items, 4 extra WHERE conjuncts, bounded digits, no ``entity =`` form, aggregates
# over numeric expressions only). Only plain GBNF constructs are used (no ``{m,n}``): bounds are explicit optional chains.
PLANNER_V0 = "agg11.v0"
PLANNER_V1 = "agg110.v1"
PLANNER_VERSIONS = (PLANNER_V0, PLANNER_V1)
PIVOT_TABLE = "facts_pivot"
PIVOT_ID_COLUMNS = ("entity", "doc_id", "period")
SQL_N_PREDICT_V0 = 160      # AGG-1.1 cap (three runaway statements of 541-566 characters hit it)
SQL_N_PREDICT_V1 = 256      # the longest other AGG-1.1 statement was 259 characters; the bounded grammar removes the runaways
_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CMP = ("=", "!=", ">", "<", ">=", "<=")
_AGGS_NUM = ("SUM", "AVG", "MIN", "MAX")
MAX_SELECT_ITEMS, MAX_EXTRA_CONDS = 3, 4
MAX_INT_DIGITS, MAX_FRAC_DIGITS = 12, 6


@dataclass(frozen=True)
class PivotCatalog:
    fields: tuple[str, ...]        # numeric pivot columns (canonical field names, identifiers)
    entities: tuple[str, ...]      # names usable in ``entity != '<name>'``
    table: str = PIVOT_TABLE


def pivot_catalog(canonical: Sequence[dict], names: Sequence[str] = (), *, fields: Sequence[str] | None = None
                  ) -> PivotCatalog:
    """Catalog over ``facts_canonical`` rows. ``fields`` (default: the canonical fields ``facts_det`` knows by name, present
    in the rows) become the pivot columns; the long tail of slugged table labels stays out of the view. Entity names with a
    quote or backslash are left out (the grammar has no escape)."""
    present = {r["field"] for r in canonical if r.get("field")}
    if fields is None:
        from src.aggregation.facts_det import FIELD_ALIASES  # noqa: PLC0415

        fields = sorted({f for _l, f in FIELD_ALIASES})
    cols = tuple(sorted(f for f in fields if f in present and _IDENT.match(f) and f not in PIVOT_ID_COLUMNS))
    ents = tuple(sorted({str(n) for n in (names or [r.get("entity") for r in canonical]) if n and not re.search(r"['\"\\]", str(n))}))
    return PivotCatalog(cols, ents)


def create_pivot_view(conn, corpus_id: str, cat: PivotCatalog, *, name: str = PIVOT_TABLE) -> None:
    """(Re)create the TEMP VIEW ``facts_pivot`` on ``conn``: one row per (document, fiscal year), one REAL column per
    catalog field, NULL where the document has no canonical row. TEMP: nothing is written to the database file."""
    if not cat.fields:
        raise ValueError("pivot catalog has no fields")
    for f in cat.fields:
        if not _IDENT.match(f):
            raise ValueError(f"unsafe field name {f!r}")
    cols = ",\n  ".join(f"CAST(MAX(CASE WHEN field = '{f}' THEN value END) AS REAL) AS {f}" for f in cat.fields)
    conn.execute(f"DROP VIEW IF EXISTS temp.{name}")
    conn.execute(
        f"CREATE TEMP VIEW {name} AS SELECT MAX(entity) AS entity, doc_id, period,\n  {cols}\n"
        f"FROM facts_canonical WHERE corpus_id = ? GROUP BY doc_id, period".replace("?", "'" + corpus_id.replace("'", "''") + "'"))


def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _opt_chain(cls: str, n: int) -> str:
    """``cls`` repeated 1..n times as nested optionals: ``[0-9] ([0-9] ([0-9])?)?``."""
    s = cls
    for _ in range(n - 1):
        s = f"{cls} ({s})?"
    return s


def build_pivot_gbnf(cat: PivotCatalog) -> str:
    fields = cat.fields or ("__none__",)
    ents = cat.entities or ("__none__",)
    return "\n".join([
        f"root ::= \"SELECT \" sel \" FROM {cat.table} WHERE period = '\" year \"'\" conds order? limit?",
        'year ::= [0-9] [0-9] [0-9] [0-9]',
        'sel ::= item (", " item (", " item)?)?',
        'item ::= "entity" | "COUNT(*)" | agg "(" expr ")" | expr',
        f"agg ::= {_alt(_AGGS_NUM)}",
        'expr ::= fld (" / " fld | " - " fld)?',
        f"fld ::= {_alt(fields)}",
        'conds ::= (" AND " cond (" AND " cond (" AND " cond (" AND " cond)?)?)?)?',
        "cond ::= expr \" \" op \" \" num | \"entity != '\" ent \"'\"",
        f"ent ::= {_alt(ents)}",
        f"op ::= {_alt(_CMP)}",
        f'num ::= "-"? int ("." frac)?',
        f"int ::= {_opt_chain('[0-9]', MAX_INT_DIGITS)}",
        f"frac ::= {_opt_chain('[0-9]', MAX_FRAC_DIGITS)}",
        'order ::= " ORDER BY " expr (" ASC" | " DESC")',
        'limit ::= " LIMIT " [0-9] [0-9]?',
    ])


def _pivot_regex(cat: PivotCatalog) -> "re.Pattern[str]":
    fld = "(?:" + "|".join(re.escape(f) for f in sorted(cat.fields, key=len, reverse=True)) + ")"
    expr = f"{fld}(?: [/-] {fld})?"
    ent = "(?:" + "|".join(re.escape(e) for e in sorted(cat.entities, key=len, reverse=True)) + ")" if cat.entities else "(?!)"
    num = rf"-?[0-9]{{1,{MAX_INT_DIGITS}}}(?:\.[0-9]{{1,{MAX_FRAC_DIGITS}}})?"
    cond = rf"(?:{expr} (?:>=|<=|!=|=|>|<) {num}|entity != '{ent}')"
    item = rf"(?:entity|COUNT\(\*\)|(?:SUM|AVG|MIN|MAX)\({expr}\)|{expr})"
    return re.compile(
        rf"SELECT {item}(?:, {item}(?:, {item})?)? FROM {cat.table} WHERE period = '[0-9]{{4}}'"
        rf"(?: AND {cond}){{0,{MAX_EXTRA_CONDS}}}(?: ORDER BY {expr} (?:ASC|DESC))?(?: LIMIT [0-9]{{1,2}})?")


_ENT_EQ = re.compile(r"\bentity\s*=\s*'", re.I)


def entity_conjunct_count(sql: str) -> int:
    """Number of ``entity = '...'`` conjuncts: the enumeration pattern of AGG-1.1 (``entity = '3M' AND ... entity = 'AMD'``)."""
    return len(_ENT_EQ.findall(sql))


def entity_conjunct_guard(sql: str, *, limit: int = 3) -> tuple[bool, str]:
    """(ok, reason): False when the statement carries ``limit`` or more ``entity =`` conjuncts. Deterministic, no model."""
    n = entity_conjunct_count(sql)
    return (n < limit, "ok" if n < limit else f"entity_enumeration:{n} entity = conjuncts")


def validate_pivot_sql(sql: str, cat: PivotCatalog) -> tuple[bool, str]:
    """True iff ``sql`` is exactly a statement the ``build_pivot_gbnf`` grammar admits (the deterministic mirror)."""
    s = sql.strip().rstrip(";")
    ok, why = entity_conjunct_guard(s)
    if not ok:
        return False, why
    if _pivot_regex(cat).fullmatch(s) is None:
        return False, "not a statement of the pivot plan grammar"
    return True, "ok"


def null_guard(sql: str, cat: PivotCatalog) -> str:
    """Add ``<field> IS NOT NULL`` for every pivot field the statement references (a company lacking a needed field is
    excluded; SQLite would otherwise rank NULL first under ``ORDER BY ... ASC``). Insert before ``ORDER BY`` / ``LIMIT``;
    a statement that already guards a field is not given a duplicate. Call on a statement that passed
    :func:`validate_pivot_sql`."""
    s = sql.strip().rstrip(";")
    head, sep, tail = s, "", ""
    for kw in (" ORDER BY ", " LIMIT "):
        i = head.find(kw)
        if i >= 0:
            head, tail = s[:i], s[i:]
            break
    used = sorted({f for f in cat.fields if re.search(rf"\b{re.escape(f)}\b", s)})
    extra = "".join(f" AND {f} IS NOT NULL" for f in used if f"{f} IS NOT NULL" not in head)
    return head + extra + tail
