"""Content-hashed corpus store over ``DocumentPreprocessResult`` (AGG-0.2, L0).

One SQL file holds ``documents``, ``chunks``, ``tables_``, ``facts`` and
``index_manifest``. The plan names DuckDB; the orchestrator venv ships sqlite
only (DuckDB is a pending dependency decision), so the store uses stdlib
``sqlite3`` by default and DuckDB when ``engine="duckdb"`` and it is importable.
The DDL is the common subset of both dialects.

Ingest is idempotent per doc content hash: re-ingesting an unchanged document is
a no-op, a changed one replaces its rows (incremental re-ingest).
"""

from __future__ import annotations

import hashlib
import json
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
    """CREATE TABLE IF NOT EXISTS index_manifest (
        corpus_id TEXT NOT NULL, content_hash TEXT NOT NULL, builder TEXT NOT NULL,
        model_id TEXT NOT NULL, layer TEXT, n_items INTEGER, built_at DOUBLE,
        PRIMARY KEY (corpus_id, content_hash, builder, model_id))""",
)

FACT_COLUMNS = (
    "fact_id", "doc_hash", "doc_id", "entity", "period", "field", "value", "unit",
    "quote", "quote_start", "quote_end", "chunk_id", "table_id", "page",
    "line_start", "line_end", "confidence", "extractor", "audit_status",
)


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
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ["schema_version", SCHEMA_VERSION],
        )
        self.conn.commit()

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
        )

    @staticmethod
    def _chunk_for_page(
        result: DocumentPreprocessResult, section_chunk: dict[str, str], page: int
    ) -> str | None:
        for s in result.sections:
            if s.page_start <= page <= s.page_end:
                return section_chunk.get(s.id)
        return None

    def _delete_doc(self, doc_hash: str) -> None:
        for table in ("chunks", "tables_", "facts", "documents"):
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

    # ---------------------------------------------------------------- manifest
    def record_index(
        self, corpus_id: str, content_hash: str, builder: str, model_id: str,
        layer: str, n_items: int,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO index_manifest VALUES (?,?,?,?,?,?,?)",
            [corpus_id, content_hash, builder, model_id, layer, n_items, time.time()],
        )
        self.conn.commit()

    def manifest(self, corpus_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT corpus_id, content_hash, builder, model_id, layer, n_items FROM index_manifest"
        params: Sequence[Any] = []
        if corpus_id is not None:
            sql += " WHERE corpus_id = ?"
            params = [corpus_id]
        keys = ("corpus_id", "content_hash", "builder", "model_id", "layer", "n_items")
        return [dict(zip(keys, r)) for r in self.conn.execute(sql, params).fetchall()]

    def built_layers(self, corpus_id: str) -> set[str]:
        """Layers with an index for the corpus' CURRENT content hash (M5 manifest check)."""
        h = self.corpus_content_hash(corpus_id)
        return {m["layer"] for m in self.manifest(corpus_id) if m["content_hash"] == h}

    def close(self) -> None:
        self.conn.close()
