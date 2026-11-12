"""Content-hashed corpus store over ``DocumentPreprocessResult`` (AGG-0.2, L0).

One SQL file holds ``documents``, ``chunks``, ``tables_``, ``facts`` and
``index_manifest``. The plan names DuckDB; the orchestrator venv ships sqlite
only (DuckDB is a pending dependency decision), so the store uses stdlib
``sqlite3`` by default and DuckDB when ``engine="duckdb"`` and it is importable.
The DDL is the common subset of both dialects.

Ingest is idempotent per doc content hash: re-ingesting an unchanged document is
a no-op, a changed one replaces its rows (incremental re-ingest). The rows a replace
removes are not lost silently: their facts are copied to ``facts_superseded`` and the
document to ``documents_superseded`` (``superseded_by`` = the new content hash) before
the live tables drop them, so "marked superseded" and "absent from every read path"
both hold.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.models.document import DocumentPreprocessResult

SCHEMA_VERSION = "agg_store.v1"

DDL = (
    """CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)""",
    """CREATE TABLE IF NOT EXISTS documents (
        doc_hash TEXT PRIMARY KEY, corpus_id TEXT NOT NULL, doc_id TEXT NOT NULL,
        entity TEXT, path TEXT, total_pages INTEGER, n_tokens_est INTEGER,
        ingested_at DOUBLE, result_json TEXT)""",
    """CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY, doc_hash TEXT NOT NULL, ordinal INTEGER,
        section_id TEXT, title TEXT, level INTEGER, page_start INTEGER, page_end INTEGER,
        text TEXT, content_hash TEXT)""",
    """CREATE TABLE IF NOT EXISTS tables_ (
        table_id TEXT PRIMARY KEY, doc_hash TEXT NOT NULL, page INTEGER,
        section_id TEXT, chunk_id TEXT, caption TEXT, markdown TEXT, rows_json TEXT)""",
    """CREATE TABLE IF NOT EXISTS facts (
        fact_id TEXT PRIMARY KEY, doc_hash TEXT NOT NULL, doc_id TEXT, entity TEXT,
        period TEXT, field TEXT, value DOUBLE, unit TEXT, quote TEXT,
        quote_start INTEGER, quote_end INTEGER, chunk_id TEXT, table_id TEXT,
        page INTEGER, line_start INTEGER, line_end INTEGER, confidence TEXT,
        extractor TEXT, audit_status TEXT)""",
    """CREATE TABLE IF NOT EXISTS facts_canonical (
        corpus_id TEXT NOT NULL, doc_id TEXT NOT NULL, field TEXT NOT NULL, period TEXT NOT NULL,
        fact_id TEXT, doc_hash TEXT, entity TEXT, value DOUBLE, unit TEXT, quote TEXT, page INTEGER,
        confidence TEXT, extractor TEXT, audit_status TEXT, n_candidates INTEGER, n_conflicts INTEGER,
        rule TEXT, PRIMARY KEY (corpus_id, doc_id, field, period))""",
    """CREATE TABLE IF NOT EXISTS fact_conflicts (
        corpus_id TEXT NOT NULL, doc_id TEXT NOT NULL, field TEXT NOT NULL, period TEXT NOT NULL,
        chosen_fact_id TEXT, other_fact_id TEXT, chosen_value DOUBLE, other_value DOUBLE,
        other_extractor TEXT, kind TEXT, rule TEXT)""",
    """CREATE TABLE IF NOT EXISTS documents_superseded (
        doc_hash TEXT NOT NULL, corpus_id TEXT NOT NULL, doc_id TEXT NOT NULL, entity TEXT,
        total_pages INTEGER, n_tokens_est INTEGER, n_chunks INTEGER, n_tables INTEGER, n_facts INTEGER,
        n_canonical INTEGER, superseded_by TEXT NOT NULL, superseded_at DOUBLE NOT NULL,
        PRIMARY KEY (doc_hash, superseded_by))""",
    """CREATE TABLE IF NOT EXISTS facts_superseded (
        fact_id TEXT NOT NULL, doc_hash TEXT NOT NULL, doc_id TEXT, entity TEXT,
        period TEXT, field TEXT, value DOUBLE, unit TEXT, quote TEXT,
        quote_start INTEGER, quote_end INTEGER, chunk_id TEXT, table_id TEXT,
        page INTEGER, line_start INTEGER, line_end INTEGER, confidence TEXT,
        extractor TEXT, audit_status TEXT, corpus_id TEXT, superseded_by TEXT NOT NULL,
        superseded_at DOUBLE NOT NULL, PRIMARY KEY (fact_id, doc_hash, superseded_by))""",
    """CREATE TABLE IF NOT EXISTS index_manifest (
        corpus_id TEXT NOT NULL, content_hash TEXT NOT NULL, builder TEXT NOT NULL,
        model_id TEXT NOT NULL, layer TEXT, n_items INTEGER, built_at DOUBLE,
        origin TEXT, n_built INTEGER, n_loaded INTEGER,
        PRIMARY KEY (corpus_id, content_hash, builder, model_id))""",
)

FACT_COLUMNS = (
    "fact_id", "doc_hash", "doc_id", "entity", "period", "field", "value", "unit",
    "quote", "quote_start", "quote_end", "chunk_id", "table_id", "page",
    "line_start", "line_end", "confidence", "extractor", "audit_status",
)


# --------------------------------------------------------------- canonical facts
# One fact per (corpus, doc, field, fiscal year). ``facts`` stays the append-only evidence table
# (every extractor row, L4d conflicts included); ``facts_canonical`` is the read path the SQL
# grammar queries. Selection is a total order, so it is deterministic and reproducible:
#
#   1. source tier   l3_table (statement line item) < l3_textrow < l4_llm < l3_regex < other
#   2. confidence    high < medium < low
#   3. audit status  audited_ok < unaudited < disputed
#   4. magnitude     larger |value| in USD millions first (a consolidated total is never smaller
#                    than a segment or sub-line of the same field and year)
#   5. earliest page, then fact_id (stable final tie-break)
#
# Value: USD scales normalise to millions; fields the schema seed defines as positive amounts
# (``POSITIVE_FIELDS``: cash-flow outflow lines such as capex and dividends paid print as negatives
# in a cash-flow statement) are stored as |value|.
#
# Candidates whose normalised value differs from the chosen one by more than ``rel_tol`` are
# CONFLICTS and agreeing ones are DUPLICATES; both are written to ``fact_conflicts`` (never
# silently dropped). Facts with no period are not canonical (they cannot be keyed). The canonical period is the 4-digit fiscal year in the stored period text
# ('2022' and 'FY2022' are one key); text without a year keeps its stripped form.
CANONICAL_RULE = "canonical.v1:tier>conf>audit>|value|desc>page>fact_id"
CANONICAL_TABLE = "facts_canonical"
_SOURCE_RANK = {"l3_table.v1": 0, "l3_textrow.v1": 1, "l4_llm.v1": 2, "l3_regex.v1": 3}
_CONF_RANK = {"high": 0, "medium": 1, "low": 2}
_AUDIT_RANK = {"audited_ok": 0, "unaudited": 1, "disputed": 2}
_UNIT_TO_USD_M = {"USD_thousands": 1e-3, "USD_millions": 1.0, "USD_billions": 1e3, "USD": 1e-6}
POSITIVE_FIELDS = frozenset({"capex", "dividends_paid"})
_YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")

# ``canonical.v2`` (AGG-1.9, 2026-11-12): ``canonical.v1`` applied after two NARROW candidate filters. ``canonical.v1`` stays the
# default of every entry point, so Phase 0 / AGG-1.1 reproduce; v2 is selected by an explicit ``rule`` argument. A blanket
# "prefer l4 over l3" tier flip was measured to break 19 cells for 7 fixed (AGG-1.8), so the filters never reorder tiers: they
# only drop a candidate that is provably a different quantity (not the shareholders' net income) or a different scale.
CANONICAL_RULE_V2 = "canonical.v2:v1+attributable_net_income+thousands_unit"
CANONICAL_RULES = (CANONICAL_RULE, CANONICAL_RULE_V2)
THOUSANDS_RATIO = 1000.0           # a candidate 1/1000 of a value that >= 2 others agree on is a unit-slip candidate
_ATTRIBUTABLE_RE = re.compile(r"attributable\s+to", re.I)
_NOT_PARENT_RE = re.compile(r"non[-\s]?controlling|minority|preferred|redeemable|participating|discontinued|common", re.I)


def canonical_period(period: Any) -> str | None:
    """``'FY2022'`` / ``'2022'`` / ``'Dec 31, 2022'`` -> ``'2022'``; no period -> None."""
    text = str(period).strip() if period is not None else ""
    if not text or text.lower() in ("none", "null"):
        return None
    m = _YEAR_RE.search(text)
    return m.group(1) if m else text


def normalised_value(fact: dict[str, Any]) -> float:
    """Value in USD millions when its unit is a USD scale; otherwise unchanged."""
    v = float(fact["value"])
    return v * _UNIT_TO_USD_M.get(str(fact.get("unit")), 1.0)


def canonical_sort_key(fact: dict[str, Any]) -> tuple:
    return (
        _SOURCE_RANK.get(str(fact.get("extractor")), 4),
        _CONF_RANK.get(str(fact.get("confidence")), 3),
        _AUDIT_RANK.get(str(fact.get("audit_status")), 1),
        -abs(normalised_value(fact)),
        fact.get("page") if fact.get("page") is not None else 10**9,
        str(fact.get("fact_id")),
    )


def _agree(a: float, b: float, rel_tol: float) -> bool:
    return abs(a - b) <= max(abs(b) * rel_tol, 1e-9)


def _entity_token(entity: Any) -> str:
    """The first alphanumeric token of 3+ characters of the entity name, lower case ('MGM Resorts International' -> 'mgm',
    'The Coca-Cola Company' -> 'coca'); '' when none."""
    for t in re.findall(r"[a-z0-9]+", str(entity or "").lower()):
        if len(t) >= 3 and t not in ("the", "inc", "corp", "company", "corporation", "plc", "ltd"):
            return t
    return ""


def is_attributable(fact: dict[str, Any]) -> bool:
    """The quote says the figure is attributable to THE COMPANY'S shareholders: 'attributable to' followed by the company's own
    name (first entity token) or by 'shareholders' / 'stockholders' / 'owners', and not the noncontrolling-interest, preferred,
    common-shareholder (after preferred dividends), discontinued-operations or another-entity variants."""
    q = str(fact.get("quote") or "")
    m = _ATTRIBUTABLE_RE.search(q)
    if not m or _NOT_PARENT_RE.search(q):
        return False
    after = q[m.end():].lower()
    tok = _entity_token(fact.get("entity"))
    generic = re.match(r"\s*(the\s+)?(company'?s?\s+)?(share|stock)?holders?\b|\s*owners\b", after) is not None
    named = bool(tok) and re.search(r"\b" + re.escape(tok), after[:60]) is not None
    return generic or named


def guard_attributable_net_income(field: str, rows: list[dict[str, Any]], rel_tol: float = 0.005
                                  ) -> tuple[list[dict[str, Any]], str | None]:
    """``(preferred, note)``. For ``net_income`` only: when at least one candidate is attributable to the company's
    shareholders, only those candidates compete (the order inside is still ``canonical.v1``, led by the number of
    attributable candidates that agree with it). Never empties the group."""
    if field != "net_income":
        return rows, None
    att = [r for r in rows if is_attributable(r)]
    if not att:
        return rows, None
    vals = {id(r): normalised_value(r) for r in att}
    support = {id(r): sum(1 for o in att if _agree(vals[id(o)], vals[id(r)], rel_tol)) for r in att}
    att.sort(key=lambda r: (-support[id(r)],) + canonical_sort_key(r))
    return att, "attributable_net_income"


def guard_thousands_unit(rows: list[dict[str, Any]], rel_tol: float = 0.005
                         ) -> tuple[list[dict[str, Any]], str | None]:
    """``(kept, note)``. Drops a candidate that is a thousands-unit slip: its |value| is within 0.5 percent of
    ``THOUSANDS_RATIO`` times, or 1/``THOUSANDS_RATIO`` of, a value that at least two OTHER candidates agree on. Never empties
    the group. (The first draft also dropped any value 100x the median of the others; it broke 2 retained cells and is not
    in this rule, see PREDECLARED_AGG19_AMENDMENT1.)"""
    if len(rows) < 3:
        return rows, None
    vals = [abs(normalised_value(r)) for r in rows]
    drop: set[int] = set()
    for i, v in enumerate(vals):
        if v <= 0:
            continue
        others = [x for j, x in enumerate(vals) if j != i and x > 0]
        for target in (v * THOUSANDS_RATIO, v / THOUSANDS_RATIO):
            if sum(1 for x in others if _agree(x, target, 0.005)) >= 2:
                drop.add(i)
    kept = [r for i, r in enumerate(rows) if i not in drop]
    return (kept, "thousands_unit") if kept and drop else (rows, None)


def select_canonical(
    facts: Iterable[dict[str, Any]], *, rel_tol: float = 0.005, rule: str = CANONICAL_RULE,
    guards: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pure selection. ``facts`` carry ``corpus_id`` plus the ``facts`` columns.

    Returns ``(canonical_rows, conflict_log_rows)``; both lists are ordered by key. ``rule`` picks the rule:
    ``CANONICAL_RULE`` (``canonical.v1``, the default, AGG-1.1 behaviour) or ``CANONICAL_RULE_V2`` (the two AGG-1.9 guards;
    ``guards`` narrows v2 to a subset of ``("thousands_unit", "attributable_net_income")`` for measurement).
    """
    if rule not in CANONICAL_RULES:
        raise ValueError(f"unknown canonical rule {rule!r}; known: {CANONICAL_RULES}")
    active = tuple(guards) if guards is not None else ("thousands_unit", "attributable_net_income")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for f in facts:
        if f.get("value") is None or not f.get("field"):
            continue
        per = canonical_period(f.get("period"))
        if per is None:
            continue
        groups.setdefault((f.get("corpus_id"), f.get("doc_id"), f["field"], per), []).append(f)
    canon: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        corpus_id, doc_id, field, per = key
        rows = sorted(groups[key], key=canonical_sort_key)
        if rule == CANONICAL_RULE_V2:
            pre = rows
            if "thousands_unit" in active:
                pre, _g = guard_thousands_unit(pre, rel_tol)
            if "attributable_net_income" in active:
                pre, _g = guard_attributable_net_income(field, pre, rel_tol)
            if pre is not rows:   # the chosen candidate leads; every other candidate stays in the log, never silently dropped
                lead = {id(r) for r in pre}
                rows = pre + [r for r in rows if id(r) not in lead]
        best = rows[0]
        bv = normalised_value(best)
        n_conf = 0
        for other in rows[1:]:
            ov = normalised_value(other)
            conflict = abs(ov - bv) > max(abs(bv) * rel_tol, 1e-9)
            n_conf += conflict
            log.append({
                "corpus_id": corpus_id, "doc_id": doc_id, "field": field, "period": per,
                "chosen_fact_id": best["fact_id"], "other_fact_id": other["fact_id"],
                "chosen_value": bv, "other_value": ov, "other_extractor": other.get("extractor"),
                "kind": "conflict" if conflict else "duplicate", "rule": rule,
            })
        canon.append({
            "corpus_id": corpus_id, "doc_id": doc_id, "field": field, "period": per,
            "fact_id": best["fact_id"], "doc_hash": best.get("doc_hash"),
            "entity": best.get("entity"),
            "value": abs(bv) if field in POSITIVE_FIELDS else bv,
            "unit": "USD_millions" if str(best.get("unit")) in _UNIT_TO_USD_M else best.get("unit"),
            "quote": best.get("quote"), "page": best.get("page"),
            "confidence": best.get("confidence"), "extractor": best.get("extractor"),
            "audit_status": best.get("audit_status"), "n_candidates": len(rows),
            "n_conflicts": n_conf, "rule": rule,
        })
    return canon, log


CANONICAL_COLUMNS = ("corpus_id", "doc_id", "field", "period", "fact_id", "doc_hash", "entity",
                     "value", "unit", "quote", "page", "confidence", "extractor", "audit_status",
                     "n_candidates", "n_conflicts", "rule")
CONFLICT_COLUMNS = ("corpus_id", "doc_id", "field", "period", "chosen_fact_id", "other_fact_id",
                    "chosen_value", "other_value", "other_extractor", "kind", "rule")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_content_hash(result: DocumentPreprocessResult) -> str:
    """Hash of what the layers read: sections (title/level/content/pages) and tables.

    Independent of ``original_path`` and timing so the same parse of the same
    bytes at another path hashes identically.
    """
    payload = {
        "sections": [
            [s.title, s.level, s.content, s.page_start, s.page_end] for s in result.sections
        ],
        "tables": [[t.page, t.caption, t.markdown, t.rows] for t in result.tables],
        "total_pages": result.total_pages,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate (chars/4). Planner sizes use this, not a tokenizer."""
    return (len(text) + 3) // 4


@dataclass(frozen=True)
class IngestReceipt:
    doc_id: str
    doc_hash: str
    status: str  # "ingested" | "unchanged" | "replaced"
    n_chunks: int
    n_tables: int
    n_tokens_est: int
    superseded: tuple[str, ...] = ()  # content hashes this ingest replaced (their rows are in ``*_superseded``)


def _connect(path: str | Path, engine: str):
    if engine == "duckdb":
        import duckdb  # noqa: PLC0415 — optional dependency, decision pending

        return duckdb.connect(str(path))
    if engine != "sqlite":
        raise ValueError(f"unknown engine {engine!r}")
    conn = sqlite3.connect(str(path))
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


class CorpusStore:
    """Persistent L0 store; every index builder reads ``chunks``/``tables_`` from here."""

    def __init__(self, path: str | Path = ":memory:", *, engine: str = "sqlite") -> None:
        self.path = str(path)
        self.engine = engine
        self.conn = _connect(path, engine)
        for stmt in DDL:
            self.conn.execute(stmt)
        self._migrate_manifest()
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ["schema_version", SCHEMA_VERSION],
        )
        self.conn.commit()

    def _migrate_manifest(self) -> None:
        """Add the ``origin`` / ``n_built`` / ``n_loaded`` columns to a manifest created before they existed."""
        have = {d[0] for d in self.conn.execute("SELECT * FROM index_manifest LIMIT 0").description}
        for col, typ in (("origin", "TEXT"), ("n_built", "INTEGER"), ("n_loaded", "INTEGER")):
            if col not in have:
                self.conn.execute(f"ALTER TABLE index_manifest ADD COLUMN {col} {typ}")

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        result: DocumentPreprocessResult,
        *,
        corpus_id: str,
        doc_id: str,
        entity: str | None = None,
    ) -> IngestReceipt:
        doc_hash = document_content_hash(result)
        prior = self.conn.execute(
            "SELECT doc_hash FROM documents WHERE corpus_id = ? AND doc_id = ?",
            [corpus_id, doc_id],
        ).fetchall()
        if any(r[0] == doc_hash for r in prior):
            n_chunks = self.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_hash = ?", [doc_hash]
            ).fetchone()[0]
            n_tables = self.conn.execute(
                "SELECT COUNT(*) FROM tables_ WHERE doc_hash = ?", [doc_hash]
            ).fetchone()[0]
            tok = self.conn.execute(
                "SELECT n_tokens_est FROM documents WHERE doc_hash = ?", [doc_hash]
            ).fetchone()[0]
            return IngestReceipt(doc_id, doc_hash, "unchanged", n_chunks, n_tables, tok)
        for (old_hash,) in prior:
            self._supersede_doc(old_hash, doc_hash, corpus_id)
            self._delete_doc(old_hash)

        section_chunk: dict[str, str] = {}
        n_tok = 0
        for ordinal, s in enumerate(result.sections):
            text = f"{s.title}\n{s.content}" if s.title else s.content
            chash = sha256_text(text)
            chunk_id = f"{doc_hash[:16]}:{ordinal:05d}"
            section_chunk[s.id] = chunk_id
            n_tok += estimate_tokens(text)
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                [chunk_id, doc_hash, ordinal, s.id, s.title, s.level, s.page_start,
                 s.page_end, text, chash],
            )
        for i, t in enumerate(result.tables):
            table_id = f"{doc_hash[:16]}:t{i:04d}"
            chunk_id = section_chunk.get(t.section_id or "") or self._chunk_for_page(
                result, section_chunk, t.page
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO tables_ VALUES (?,?,?,?,?,?,?,?)",
                [table_id, doc_hash, t.page, t.section_id, chunk_id, t.caption,
                 t.markdown, json.dumps(t.rows, default=str)],
            )
        self.conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
            [doc_hash, corpus_id, doc_id, entity, result.original_path, result.total_pages,
             n_tok, time.time(), json.dumps(result.to_cache_dict(), default=str)],
        )
        self.conn.commit()
        return IngestReceipt(
            doc_id, doc_hash, "replaced" if prior else "ingested",
            len(result.sections), len(result.tables), n_tok,
            tuple(r[0] for r in prior),
        )

    @staticmethod
    def _chunk_for_page(
        result: DocumentPreprocessResult, section_chunk: dict[str, str], page: int
    ) -> str | None:
        for s in result.sections:
            if s.page_start <= page <= s.page_end:
                return section_chunk.get(s.id)
        return None

    def _supersede_doc(self, old_hash: str, new_hash: str, corpus_id: str) -> None:
        """Record what a replace is about to delete: the old document's facts go to ``facts_superseded`` and one
        ``documents_superseded`` row keeps its counts, both tagged with ``superseded_by`` = the new content hash.
        Idempotent (primary keys), and a no-op for a hash the store does not hold."""
        row = self.conn.execute(
            "SELECT doc_id, entity, total_pages, n_tokens_est FROM documents WHERE doc_hash = ?",
            [old_hash]).fetchone()
        if row is None:
            return
        now = time.time()
        cols = ", ".join(FACT_COLUMNS)
        self.conn.execute(
            f"INSERT OR REPLACE INTO facts_superseded ({cols}, corpus_id, superseded_by, superseded_at) "
            f"SELECT {cols}, ?, ?, ? FROM facts WHERE doc_hash = ?", [corpus_id, new_hash, now, old_hash])
        n = {t: self.conn.execute(f"SELECT COUNT(*) FROM {t} WHERE doc_hash = ?", [old_hash]).fetchone()[0]
             for t in ("chunks", "tables_", "facts", CANONICAL_TABLE)}
        self.conn.execute(
            "INSERT OR REPLACE INTO documents_superseded (doc_hash, corpus_id, doc_id, entity, total_pages, "
            "n_tokens_est, n_chunks, n_tables, n_facts, n_canonical, superseded_by, superseded_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [old_hash, corpus_id, row[0], row[1], row[2], row[3], n["chunks"], n["tables_"], n["facts"],
             n[CANONICAL_TABLE], new_hash, now])

    def _delete_doc(self, doc_hash: str) -> None:
        for table in ("chunks", "tables_", "facts", CANONICAL_TABLE, "documents"):
            self.conn.execute(f"DELETE FROM {table} WHERE doc_hash = ?", [doc_hash])

    # ------------------------------------------------------------------- reads
    def corpus_docs(self, corpus_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT doc_hash, doc_id, entity, n_tokens_est, total_pages FROM documents "
            "WHERE corpus_id = ? ORDER BY doc_id",
            [corpus_id],
        ).fetchall()
        return [
            dict(zip(("doc_hash", "doc_id", "entity", "n_tokens_est", "total_pages"), r))
            for r in rows
        ]

    def corpus_content_hash(self, corpus_id: str) -> str:
        return sha256_text("|".join(sorted(d["doc_hash"] for d in self.corpus_docs(corpus_id))))

    def corpus_tokens(self, corpus_id: str) -> int:
        return sum(d["n_tokens_est"] or 0 for d in self.corpus_docs(corpus_id))

    def document_pages(self, corpus_id: str) -> list[dict[str, Any]]:
        """Per-page OCR text of each document (from the stored result), for L3 text rows."""
        out = []
        for doc_hash, doc_id, entity, rj in self.conn.execute(
            "SELECT doc_hash, doc_id, entity, result_json FROM documents WHERE corpus_id = ? "
            "ORDER BY doc_id", [corpus_id]).fetchall():
            data = json.loads(rj) if rj else {}
            pages = [(p["page"], p.get("text") or "")
                     for p in ((data.get("ocr_result") or {}).get("pages") or [])]
            out.append({"doc_hash": doc_hash, "doc_id": doc_id, "entity": entity,
                        "pages": pages})
        return out

    def chunks(self, corpus_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT c.chunk_id, c.doc_hash, d.doc_id, c.ordinal, c.section_id, c.title, "
            "c.page_start, c.page_end, c.text FROM chunks c JOIN documents d "
            "ON c.doc_hash = d.doc_hash WHERE d.corpus_id = ? ORDER BY d.doc_id, c.ordinal",
            [corpus_id],
        ).fetchall()
        keys = ("chunk_id", "doc_hash", "doc_id", "ordinal", "section_id", "title",
                "page_start", "page_end", "text")
        return [dict(zip(keys, r)) for r in rows]

    def tables(self, corpus_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT t.table_id, t.doc_hash, d.doc_id, d.entity, t.page, t.chunk_id, "
            "t.caption, t.markdown, t.rows_json FROM tables_ t JOIN documents d "
            "ON t.doc_hash = d.doc_hash WHERE d.corpus_id = ? ORDER BY d.doc_id, t.table_id",
            [corpus_id],
        ).fetchall()
        out = []
        for r in rows:
            rec = dict(zip(("table_id", "doc_hash", "doc_id", "entity", "page", "chunk_id",
                            "caption", "markdown"), r[:8]))
            rec["rows"] = json.loads(r[8] or "[]")
            out.append(rec)
        return out

    def superseded_documents(self, corpus_id: str | None = None) -> list[dict[str, Any]]:
        """Documents a replace removed from the live tables, newest first (never part of any read path above)."""
        keys = ("doc_hash", "corpus_id", "doc_id", "entity", "total_pages", "n_tokens_est", "n_chunks",
                "n_tables", "n_facts", "n_canonical", "superseded_by", "superseded_at")
        sql = f"SELECT {', '.join(keys)} FROM documents_superseded"
        params: list[Any] = []
        if corpus_id is not None:
            sql += " WHERE corpus_id = ?"
            params.append(corpus_id)
        sql += " ORDER BY superseded_at DESC, doc_hash"
        return [dict(zip(keys, r)) for r in self.conn.execute(sql, params).fetchall()]

    def superseded_facts(self, corpus_id: str | None = None, *, doc_hash: str | None = None) -> list[dict[str, Any]]:
        """The ledger rows of ``facts_superseded`` (``FACT_COLUMNS`` + ``superseded_by``)."""
        cols = FACT_COLUMNS + ("superseded_by",)
        sql = f"SELECT {', '.join(cols)} FROM facts_superseded WHERE 1 = 1"
        params: list[Any] = []
        if corpus_id is not None:
            sql += " AND corpus_id = ?"
            params.append(corpus_id)
        if doc_hash is not None:
            sql += " AND doc_hash = ?"
            params.append(doc_hash)
        sql += " ORDER BY doc_hash, fact_id"
        return [dict(zip(cols, r)) for r in self.conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------- facts
    def write_facts(self, facts: Iterable[dict[str, Any]]) -> int:
        n = 0
        for f in facts:
            self.conn.execute(
                f"INSERT OR REPLACE INTO facts ({', '.join(FACT_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in FACT_COLUMNS)})",
                [f.get(c) for c in FACT_COLUMNS],
            )
            n += 1
        self.conn.commit()
        return n

    def facts(self, corpus_id: str, *, extractor: str | None = None) -> list[dict[str, Any]]:
        sql = (
            f"SELECT {', '.join('f.' + c for c in FACT_COLUMNS)} FROM facts f JOIN documents d "
            "ON f.doc_hash = d.doc_hash WHERE d.corpus_id = ?"
        )
        params: list[Any] = [corpus_id]
        if extractor:
            sql += " AND f.extractor = ?"
            params.append(extractor)
        return [dict(zip(FACT_COLUMNS, r)) for r in self.conn.execute(sql, params).fetchall()]

    # --------------------------------------------------------- canonical facts
    def refresh_canonical(self, corpus_id: str | None = None, *, rel_tol: float = 0.005,
                          rule: str = CANONICAL_RULE) -> dict[str, Any]:
        """Rebuild ``facts_canonical`` / ``fact_conflicts`` from ``facts`` (idempotent).

        Scope is one corpus, or every corpus when ``corpus_id`` is None. Returns counts for the
        run record: candidates, canonical rows, duplicates, conflicts.
        """
        sql = (f"SELECT d.corpus_id, {', '.join('f.' + c for c in FACT_COLUMNS)} FROM facts f "
               "JOIN documents d ON f.doc_hash = d.doc_hash")
        params: list[Any] = []
        if corpus_id is not None:
            sql += " WHERE d.corpus_id = ?"
            params.append(corpus_id)
        rows = [dict(zip(("corpus_id",) + FACT_COLUMNS, r))
                for r in self.conn.execute(sql, params).fetchall()]
        canon, log = select_canonical(rows, rel_tol=rel_tol, rule=rule)
        scope, sp = ("WHERE corpus_id = ?", [corpus_id]) if corpus_id is not None else ("", [])
        self.conn.execute(f"DELETE FROM {CANONICAL_TABLE} {scope}", sp)
        self.conn.execute(f"DELETE FROM fact_conflicts {scope}", sp)
        for r in canon:
            self.conn.execute(
                f"INSERT INTO {CANONICAL_TABLE} ({', '.join(CANONICAL_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in CANONICAL_COLUMNS)})",
                [r[c] for c in CANONICAL_COLUMNS])
        for r in log:
            self.conn.execute(
                f"INSERT INTO fact_conflicts ({', '.join(CONFLICT_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in CONFLICT_COLUMNS)})",
                [r[c] for c in CONFLICT_COLUMNS])
        self.conn.commit()
        return {"rule": rule, "n_candidates": len(rows), "n_canonical": len(canon),
                "n_duplicates": sum(1 for r in log if r["kind"] == "duplicate"),
                "n_conflicts": sum(1 for r in log if r["kind"] == "conflict")}

    def canonical_facts(self, corpus_id: str) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            f"SELECT {', '.join(CANONICAL_COLUMNS)} FROM {CANONICAL_TABLE} WHERE corpus_id = ? "
            "ORDER BY doc_id, field, period", [corpus_id])
        return [dict(zip(CANONICAL_COLUMNS, r)) for r in cur.fetchall()]

    def conflicts(self, corpus_id: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        sql = f"SELECT {', '.join(CONFLICT_COLUMNS)} FROM fact_conflicts WHERE corpus_id = ?"
        params: list[Any] = [corpus_id]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY doc_id, field, period, other_fact_id"
        return [dict(zip(CONFLICT_COLUMNS, r)) for r in self.conn.execute(sql, params).fetchall()]

    # ---------------------------------------------------------------- manifest
    def record_index(
        self, corpus_id: str, content_hash: str, builder: str, model_id: str,
        layer: str, n_items: int, origin: str = "built",
    ) -> None:
        """Record that an index for this key now exists in a process: ``origin`` is ``'built'`` (computed) or
        ``'loaded'`` (read back from its persisted file). ``n_built`` / ``n_loaded`` count both over the row's life;
        a load keeps the original ``built_at``. A row written before the counters existed counts as one build."""
        if origin not in ("built", "loaded"):
            raise ValueError(f"origin must be 'built' or 'loaded', got {origin!r}")
        key = [corpus_id, content_hash, builder, model_id]
        row = self.conn.execute(
            "SELECT n_built, n_loaded FROM index_manifest "
            "WHERE corpus_id = ? AND content_hash = ? AND builder = ? AND model_id = ?", key).fetchone()
        n_built = (row[0] if row[0] is not None else 1) if row else 0
        n_loaded = (row[1] or 0) if row else 0
        if origin == "built":
            n_built += 1
        else:
            n_loaded += 1
        if row and origin == "loaded":
            self.conn.execute(
                "UPDATE index_manifest SET origin = ?, n_built = ?, n_loaded = ?, n_items = ? "
                "WHERE corpus_id = ? AND content_hash = ? AND builder = ? AND model_id = ?",
                [origin, n_built, n_loaded, n_items, *key])
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO index_manifest "
                "(corpus_id, content_hash, builder, model_id, layer, n_items, built_at, origin, n_built, n_loaded) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [*key, layer, n_items, time.time(), origin, n_built, n_loaded])
        self.conn.commit()

    def manifest(self, corpus_id: str | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT corpus_id, content_hash, builder, model_id, layer, n_items, "
               "COALESCE(origin, 'built'), COALESCE(n_built, 1), COALESCE(n_loaded, 0) FROM index_manifest")
        params: Sequence[Any] = []
        if corpus_id is not None:
            sql += " WHERE corpus_id = ?"
            params = [corpus_id]
        keys = ("corpus_id", "content_hash", "builder", "model_id", "layer", "n_items",
                "origin", "n_built", "n_loaded")
        return [dict(zip(keys, r)) for r in self.conn.execute(sql, params).fetchall()]

    def built_layers(self, corpus_id: str) -> set[str]:
        """Layers with an index for the corpus' CURRENT content hash (M5 manifest check)."""
        h = self.corpus_content_hash(corpus_id)
        return {m["layer"] for m in self.manifest(corpus_id) if m["content_hash"] == h}

    def close(self) -> None:
        self.conn.close()
