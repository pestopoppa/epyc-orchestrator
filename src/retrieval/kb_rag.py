"""Internal Knowledge-Base RAG over the project's markdown corpus.

Architecture:
- Corpus: wiki/, handoffs/active/, handoffs/completed/, research/, progress/,
  + cross-repo docs/chapters/. Configured via kb_rag_config.yaml.
- Chunker: heading-aware (markdown_chunker.py).
- Encoder: shared ColBERT primitives (colbert_encoder.py).
- Storage: per-document .npz of token embeddings + SQLite catalog mapping
  (chunk_id, file_path, heading_path, line_range, mtime, content_hash).
- Query: top-K MaxSim ranking, returns chunk dicts with breadcrumbs.

Per handoffs/active/internal-kb-rag.md K3+K4.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval import colbert_encoder, cross_encoder
from src.retrieval.markdown_chunker import Chunk, chunk_file

logger = logging.getLogger(__name__)

# Default index location under data/kb_rag/ (gitignored).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = _REPO_ROOT / "data" / "kb_rag" / "index"

# Encoding bounds.
_QUERY_MAX_TOKENS = 48
_DOC_MAX_TOKENS = 256  # higher than reranker's 64; chunks are markdown sections


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# K10 (temporal recency) + K9 (cross-encoder rerank) hybrid-signal defaults.
# All default to a no-op so query() behaviour is unchanged until the K7 eval
# harness tunes them; env vars let the harness sweep without code edits.
# Source: agentmemory deep-dive, research/deep-dives/2026-05-27-agent-memory-cluster.md.
_RECENCY_WEIGHT_DEFAULT = _env_float("KB_RAG_RECENCY_WEIGHT", 0.0)  # 0 = MaxSim-only
_RECENCY_SIGMA_DAYS_DEFAULT = _env_float("KB_RAG_RECENCY_SIGMA_DAYS", 90.0)
_RERANK_DEFAULT = _env_flag("KB_RAG_RERANK")
_RERANK_WEIGHT_DEFAULT = _env_float("KB_RAG_RERANK_WEIGHT", 0.3)
_RERANK_POOL_MULT_DEFAULT = _env_float("KB_RAG_RERANK_POOL_MULT", 4.0)
_LEXICAL_WEIGHT_DEFAULT = _env_float("KB_RAG_LEXICAL_WEIGHT", 0.0)
_LEXICAL_POOL_MULT_DEFAULT = _env_float("KB_RAG_LEXICAL_POOL_MULT", 4.0)


def _recency_score(mtime: float, now: float, sigma_days: float) -> float:
    """Gaussian recency weight in (0, 1]: 1.0 at age 0, decaying with age.

    age_days = (now - mtime) / 86400; score = exp(-0.5 * (age/sigma)^2).
    Matches agentmemory's temporal-proximity signal (intake-611).
    """
    if sigma_days <= 0:
        return 1.0
    age_days = max(0.0, (now - mtime) / 86400.0)
    return math.exp(-0.5 * (age_days / sigma_days) ** 2)


def _sanitize_fts_query(text: str) -> str:
    """Normalize a free-form query into an FTS5-safe token string."""
    return " ".join(
        tok for tok in "".join(c if c.isalnum() else " " for c in text).split() if tok
    )

_CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk (
  chunk_id INTEGER PRIMARY KEY,
  file_path TEXT NOT NULL,
  heading_path TEXT NOT NULL,        -- JSON-encoded list[str]
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  mtime REAL NOT NULL,
  emb_path TEXT NOT NULL,            -- relative to index_dir
  text_preview TEXT,                  -- first 240 chars for snippet display
  token_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS chunk_file ON chunk(file_path);
CREATE INDEX IF NOT EXISTS chunk_hash ON chunk(content_hash);
"""


@dataclass
class CorpusConfig:
    """Roots + glob patterns for corpus walking."""

    roots: list[str]
    include_globs: list[str]
    exclude_patterns: list[str]
    max_chunk_chars: int = 4000

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CorpusConfig":
        import yaml  # lazy import

        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            roots=d.get("roots", []),
            include_globs=d.get("include_globs", ["**/*.md"]),
            exclude_patterns=d.get("exclude_patterns", []),
            max_chunk_chars=int(d.get("max_chunk_chars", 4000)),
        )


def _ensure_catalog(index_dir: Path) -> sqlite3.Connection:
    index_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_dir / "catalog.sqlite"))
    conn.executescript(_CATALOG_SCHEMA)
    conn.commit()
    return conn


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    """Create the optional lexical index if SQLite was built with FTS5."""
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
            USING fts5(file_path, heading_path, text, tokenize='porter')
            """
        )
        conn.commit()
        return True
    except sqlite3.OperationalError:
        logger.info("FTS5 not available — KB-RAG lexical signal disabled")
        return False


def _sync_chunk_fts_row(
    cur: sqlite3.Cursor,
    chunk_id: int,
    file_path: str,
    heading_path: list[str],
    text: str,
    fts_enabled: bool,
) -> None:
    if not fts_enabled:
        return
    cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
    cur.execute(
        "INSERT INTO chunk_fts(rowid, file_path, heading_path, text) VALUES (?, ?, ?, ?)",
        (chunk_id, file_path, json.dumps(heading_path), text),
    )


def _lexical_scores(
    conn: sqlite3.Connection,
    text: str,
    limit: int,
) -> dict[int, float]:
    """Return reciprocal-rank lexical scores from FTS5 matches."""
    query = _sanitize_fts_query(text)
    if not query:
        return {}
    try:
        rows = conn.execute(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? "
            "ORDER BY bm25(chunk_fts) LIMIT ?",
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {int(row[0]): 1.0 / float(idx + 1) for idx, row in enumerate(rows)}


def _walk_corpus(config: CorpusConfig) -> list[Path]:
    """Resolve all corpus markdown files per the config."""
    seen: set[Path] = set()
    files: list[Path] = []
    for root in config.roots:
        rp = Path(root).expanduser().resolve()
        if not rp.exists():
            logger.warning("corpus root %s does not exist — skipping", rp)
            continue
        for pattern in config.include_globs:
            for f in rp.glob(pattern):
                if not f.is_file():
                    continue
                # Apply exclude patterns (substring match against absolute path).
                rel = str(f)
                if any(ex in rel for ex in config.exclude_patterns):
                    continue
                if f in seen:
                    continue
                seen.add(f)
                files.append(f)
    return sorted(files)


def _emb_relative_path(file_path: str, content_hash: str) -> str:
    """Path of the per-chunk .npz under index_dir (relative)."""
    safe_name = Path(file_path).name.replace(".", "_")
    return f"emb/{safe_name}__{content_hash}.npz"


def build_index(
    config: CorpusConfig,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Build (or refresh) the KB-RAG index.

    For each markdown file:
    1. Read + chunk (heading-aware).
    2. For each chunk, check if `(file_path, content_hash)` already in catalog.
       If yes and force=False, skip (mtime-stable + content-stable).
    3. Otherwise, encode chunk text, write per-chunk .npz, upsert catalog row.

    Returns stats dict.
    """
    index_dir = Path(index_dir)
    (index_dir / "emb").mkdir(parents=True, exist_ok=True)

    if not colbert_encoder.is_available():
        return {
            "ok": False,
            "error": "encoder model not available on disk",
            "model_path": str(colbert_encoder._MODEL_DIR),
        }

    if not colbert_encoder.ensure_loaded():
        return {"ok": False, "error": "encoder failed to load"}

    conn = _ensure_catalog(index_dir)
    conn.row_factory = sqlite3.Row
    fts_enabled = _ensure_fts(conn)
    cur = conn.cursor()

    files = _walk_corpus(config)
    n_files = len(files)
    n_chunks_seen = 0
    n_chunks_encoded = 0
    n_chunks_skipped = 0
    started = time.perf_counter()

    for file_idx, f in enumerate(files, start=1):
        try:
            chunks = chunk_file(f, max_chars=config.max_chunk_chars)
        except Exception as e:  # noqa: BLE001
            logger.warning("chunker failed on %s: %s", f, e)
            continue

        mtime = f.stat().st_mtime
        for ch in chunks:
            n_chunks_seen += 1
            existing = cur.execute(
                "SELECT chunk_id, content_hash FROM chunk WHERE file_path=? AND content_hash=?",
                (str(f), ch.content_hash),
            ).fetchone()
            if existing and not force:
                n_chunks_skipped += 1
                _sync_chunk_fts_row(
                    cur,
                    int(existing["chunk_id"]),
                    str(f),
                    ch.heading_path,
                    ch.text,
                    fts_enabled,
                )
                continue

            emb = colbert_encoder.encode(ch.text, _DOC_MAX_TOKENS)
            if emb is None:
                continue

            emb_rel = _emb_relative_path(str(f), ch.content_hash)
            emb_abs = index_dir / emb_rel
            emb_abs.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(emb_abs, emb=emb)

            preview = ch.text.strip()[:240]
            cur.execute(
                "INSERT INTO chunk "
                "(file_path, heading_path, line_start, line_end, content_hash, "
                " mtime, emb_path, text_preview, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(f),
                    json.dumps(ch.heading_path),
                    ch.line_range[0],
                    ch.line_range[1],
                    ch.content_hash,
                    mtime,
                    emb_rel,
                    preview,
                    int(emb.shape[0]),
                ),
            )
            _sync_chunk_fts_row(
                cur,
                int(cur.lastrowid),
                str(f),
                ch.heading_path,
                ch.text,
                fts_enabled,
            )
            n_chunks_encoded += 1

        if file_idx % 25 == 0:
            conn.commit()
            logger.info(
                "kb_rag: %d/%d files, %d chunks encoded, %d skipped",
                file_idx, n_files, n_chunks_encoded, n_chunks_skipped,
            )

    # Optional cleanup: remove catalog rows for files that disappeared.
    catalog_files = {r[0] for r in cur.execute("SELECT DISTINCT file_path FROM chunk")}
    current_files = {str(f) for f in files}
    stale = catalog_files - current_files
    for stale_file in stale:
        if fts_enabled:
            for row in cur.execute(
                "SELECT chunk_id FROM chunk WHERE file_path = ?",
                (stale_file,),
            ).fetchall():
                cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (int(row["chunk_id"]),))
        cur.execute("DELETE FROM chunk WHERE file_path = ?", (stale_file,))
    conn.commit()
    conn.close()

    elapsed = time.perf_counter() - started
    return {
        "ok": True,
        "files": n_files,
        "chunks_seen": n_chunks_seen,
        "chunks_encoded": n_chunks_encoded,
        "chunks_skipped_unchanged": n_chunks_skipped,
        "stale_files_removed": len(stale),
        "elapsed_sec": round(elapsed, 2),
        "index_dir": str(index_dir),
    }


def update_files(
    paths: list[str],
    config: CorpusConfig,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
) -> dict[str, Any]:
    """Re-encode a specific list of files (e.g. from `git diff --name-only`).

    Each file's existing catalog rows are deleted before re-chunking +
    re-encoding. Files not matching any include glob are skipped silently.
    """
    index_dir = Path(index_dir)
    (index_dir / "emb").mkdir(parents=True, exist_ok=True)
    if not colbert_encoder.ensure_loaded():
        return {"ok": False, "error": "encoder failed to load"}

    conn = _ensure_catalog(index_dir)
    conn.row_factory = sqlite3.Row
    fts_enabled = _ensure_fts(conn)
    cur = conn.cursor()

    # Resolve which paths actually belong to corpus.
    corpus_files = {str(p) for p in _walk_corpus(config)}
    encoded = 0
    for raw_path in paths:
        p = Path(raw_path).resolve()
        if str(p) not in corpus_files:
            continue
        if fts_enabled:
            for row in cur.execute(
                "SELECT chunk_id, heading_path, line_start, line_end, text_preview "
                "FROM chunk WHERE file_path = ?",
                (str(p),),
            ).fetchall():
                cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (int(row["chunk_id"]),))
        cur.execute("DELETE FROM chunk WHERE file_path = ?", (str(p),))
        chunks = chunk_file(p, max_chars=config.max_chunk_chars)
        mtime = p.stat().st_mtime
        for ch in chunks:
            emb = colbert_encoder.encode(ch.text, _DOC_MAX_TOKENS)
            if emb is None:
                continue
            emb_rel = _emb_relative_path(str(p), ch.content_hash)
            emb_abs = index_dir / emb_rel
            emb_abs.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(emb_abs, emb=emb)
            cur.execute(
                "INSERT INTO chunk "
                "(file_path, heading_path, line_start, line_end, content_hash, "
                " mtime, emb_path, text_preview, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(p),
                    json.dumps(ch.heading_path),
                    ch.line_range[0],
                    ch.line_range[1],
                    ch.content_hash,
                    mtime,
                    emb_rel,
                    ch.text.strip()[:240],
                    int(emb.shape[0]),
                ),
            )
            _sync_chunk_fts_row(
                cur,
                int(cur.lastrowid),
                str(p),
                ch.heading_path,
                ch.text,
                fts_enabled,
            )
            encoded += 1
    conn.commit()
    conn.close()
    return {"ok": True, "files_processed": len(paths), "chunks_encoded": encoded}


def query(
    text: str,
    top_k: int = 8,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    recency_weight: float | None = None,
    recency_sigma_days: float | None = None,
    rerank: bool | None = None,
    rerank_weight: float | None = None,
    lexical_weight: float | None = None,
) -> list[dict[str, Any]]:
    """Top-K hybrid retrieval against the indexed corpus.

    Stage 1 (always): ColBERT MaxSim over every chunk.
    Stage 2 (K10, opt-in): blend a Gaussian temporal-recency weight in
      `final = (1 - recency_weight) * maxsim + recency_weight * recency`.
    Stage 3 (K9, opt-in): cross-encoder rerank of the top `top_k * pool_mult`
      candidates, blended `(1 - rerank_weight) * score + rerank_weight * ce`.
    Stage 4 (K11, opt-in): blend a lexical FTS5 signal over the chunk text,
      file path, and heading path. Default weight is zero, so the path remains
      MaxSim-only unless a caller explicitly opts in.

    All optional stages default to a no-op so the default result is identical
    to plain MaxSim; defaults come from env vars (KB_RAG_RECENCY_WEIGHT /
    _SIGMA_DAYS, KB_RAG_RERANK / _WEIGHT / _POOL_MULT, KB_RAG_LEXICAL_WEIGHT
    / _POOL_MULT) so the K7 eval harness can sweep without code edits. Tuning +
    win-validation is gated on K7 (inference). See
    research/deep-dives/2026-05-27-agent-memory-cluster.md.

    Returns list of dicts: {file, heading_path, line_range, snippet, score}
    (plus `recency` and `ce_score` when those stages are active).
    """
    recency_weight = _RECENCY_WEIGHT_DEFAULT if recency_weight is None else recency_weight
    recency_sigma_days = (
        _RECENCY_SIGMA_DAYS_DEFAULT if recency_sigma_days is None else recency_sigma_days
    )
    rerank = _RERANK_DEFAULT if rerank is None else rerank
    rerank_weight = _RERANK_WEIGHT_DEFAULT if rerank_weight is None else rerank_weight
    lexical_weight = _LEXICAL_WEIGHT_DEFAULT if lexical_weight is None else lexical_weight

    index_dir = Path(index_dir)
    catalog_path = index_dir / "catalog.sqlite"
    if not catalog_path.exists():
        logger.warning("no index at %s — run build_index first", index_dir)
        return []

    if not colbert_encoder.ensure_loaded():
        return []

    q_emb = colbert_encoder.encode(text, _QUERY_MAX_TOKENS)
    if q_emb is None:
        return []

    conn = sqlite3.connect(str(catalog_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, file_path, heading_path, line_start, line_end, "
        "       content_hash, emb_path, text_preview, mtime FROM chunk"
    ).fetchall()

    lexical_scores = (
        _lexical_scores(conn, text, max(top_k, int(top_k * _LEXICAL_POOL_MULT_DEFAULT)))
        if lexical_weight > 0.0
        else {}
    )

    now = time.time()
    scored: list[tuple[float, float, float, sqlite3.Row]] = []  # (blended, recency, lexical, row)
    for r in rows:
        emb_path = index_dir / r["emb_path"]
        if not emb_path.exists():
            continue
        try:
            data = np.load(emb_path)
            d_emb = data["emb"]
        except Exception:  # noqa: BLE001
            continue
        s = colbert_encoder.maxsim(q_emb, d_emb)
        rec = (
            _recency_score(r["mtime"], now, recency_sigma_days)
            if recency_weight > 0.0
            else 0.0
        )
        lexical = lexical_scores.get(int(r["chunk_id"]), 0.0)
        blended = s
        if recency_weight > 0.0:
            blended = (1.0 - recency_weight) * blended + recency_weight * rec
        if lexical_weight > 0.0:
            blended = (1.0 - lexical_weight) * blended + lexical_weight * lexical
        scored.append((blended, rec, lexical, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # K9: cross-encoder rerank a wider candidate pool, then truncate to top_k.
    pool_size = max(top_k, int(top_k * _RERANK_POOL_MULT_DEFAULT)) if rerank else top_k

    results = []
    for blended, rec, lexical, r in scored[:pool_size]:
        item = {
            "file": r["file_path"],
            "heading_path": json.loads(r["heading_path"]),
            "line_range": (r["line_start"], r["line_end"]),
            "snippet": r["text_preview"],
            "score": round(blended, 4),
            "content_hash": r["content_hash"],
        }
        if recency_weight > 0.0:
            item["recency"] = round(rec, 4)
        if lexical_weight > 0.0:
            item["lexical"] = round(lexical, 4)
        results.append(item)

    if rerank and rerank_weight > 0.0 and results:
        results = cross_encoder.rerank(
            text, results, text_key="snippet", weight=rerank_weight, base_key="score"
        )

    conn.close()
    return results[:top_k]


def stats(index_dir: Path | str = DEFAULT_INDEX_DIR) -> dict[str, Any]:
    """Index summary: chunk count, file count, total embedding bytes."""
    index_dir = Path(index_dir)
    catalog_path = index_dir / "catalog.sqlite"
    if not catalog_path.exists():
        return {"exists": False}
    conn = sqlite3.connect(str(catalog_path))
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    n_files = conn.execute("SELECT COUNT(DISTINCT file_path) FROM chunk").fetchone()[0]
    conn.close()
    emb_dir = index_dir / "emb"
    emb_bytes = sum(p.stat().st_size for p in emb_dir.glob("*.npz")) if emb_dir.exists() else 0
    return {
        "exists": True,
        "files": n_files,
        "chunks": n_chunks,
        "emb_bytes": emb_bytes,
        "emb_mib": round(emb_bytes / (1024 * 1024), 2),
        "index_dir": str(index_dir),
    }
