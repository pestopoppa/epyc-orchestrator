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
