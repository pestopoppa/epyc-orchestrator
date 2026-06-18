"""Strategy Memory Store: retrievable strategy memory for AutoPilot species.

FAISS for vector similarity + SQLite for structured metadata. Reuses
FAISSEmbeddingStore and TaskEmbedder (with hash-based fallback for
environments without a running embedding model).

Usage:
    store = StrategyStore("/tmp/strategies")
    sid = store.store(
        description="Disable self-speculation for dense models",
        insight="HSD net-negative on Qwen3.5 hybrid; only viable for dense-only",
        source_trial_id=42,
        species="config_tuner",
    )
    results = store.retrieve("speculation configuration", k=3)
    store.close()

AP-28 upgrade: FTS5 keyword index + Reciprocal Rank Fusion with FAISS,
per-entry context-hash staleness, validity-weighted ranking, and
``entry_type`` (raw / pattern / convention) for the L1/L2/L3 hierarchy
consumed by ``knowledge_distiller`` (AP-29).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/strategies"
)

# AP-28: files whose contents define the configuration epoch. Hash of these
# files is recorded on every store(); entries whose stored hash differs from
# the current hash get a validity penalty at retrieve time.
DEFAULT_CONTEXT_FILES: tuple[Path, ...] = (
    Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml"),
    Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/prompts/frontdoor.md"),
    Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/prompts/roles/worker_general.md"),
)

# Reciprocal Rank Fusion default constant (Cormack et al. 2009).
_RRF_K = 60
_TITLE_MAX_CHARS = 96

_SPECIFICITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("absolute_path", re.compile(r"(?:/mnt/raid0|/workspace|/home/node)/[^\s,;:)]+")),
    ("repo_path", re.compile(r"\b(?:src|scripts|tests|orchestration|handoffs|docs)/[^\s,;:)]+")),
    ("trial_reference", re.compile(r"\btrial\s+#?\d+\b|(?<!\w)#\d+\b", re.IGNORECASE)),
    ("commit_hash", re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)),
)


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _derive_title(description: str) -> str:
    text = _compact_text(description)
    for separator in (". ", ": ", " - "):
        if separator in text:
            text = text.split(separator, 1)[0]
            break
    return text[:_TITLE_MAX_CHARS].rstrip(" .")


def _specificity_flags(*parts: str) -> list[str]:
    text = " ".join(_compact_text(part) for part in parts if part)
    flags = [name for name, pattern in _SPECIFICITY_PATTERNS if pattern.search(text)]
    return sorted(set(flags))


def _insight_format(
    *,
    title: str | None,
    description: str,
    generalized_content: str | None,
    insight: str,
) -> dict[str, Any]:
    formatted_title = _compact_text(title) or _derive_title(description)
    formatted_description = _compact_text(description)
    formatted_content = _compact_text(generalized_content if generalized_content is not None else insight)
    return {
        "version": 1,
        "title": formatted_title,
        "description": formatted_description,
        "generalized_content": formatted_content,
        "specificity_flags": _specificity_flags(
            formatted_title,
            formatted_description,
            formatted_content,
        ),
    }


def _journal_trial_id(entry: Any) -> int | None:
    try:
        return int(getattr(entry, "trial_id"))
    except (TypeError, ValueError):
        return None


def _journal_entry_excludes_strategy_evidence(entry: Any) -> bool:
    """True when a journal row should quarantine strategy evidence it cites."""
    if getattr(entry, "bug_corrupted_by", ""):
        return True
    if getattr(entry, "outcome_status", "ok") != "ok":
        return True
    if getattr(entry, "keep_revert_decision", "") == "excluded":
        return True
    eval_details = getattr(entry, "eval_details", {}) or {}
    return isinstance(eval_details, dict) and bool(eval_details.get("learning_exclusion"))


def excluded_strategy_evidence_trial_ids(journal: Any) -> set[int]:
    """Return trial IDs whose strategy evidence should not be retrieved.

    Prefer the folded append-only journal view when available so supersession
    events and learning exclusions quarantine downstream StrategyStore rows
    without mutating the persisted strategy database.
    """
    try:
        entries = (
            journal.entries_with_supersessions()
            if hasattr(journal, "entries_with_supersessions")
            else journal.all_entries()
        )
    except Exception:
        return set()

    excluded: set[int] = set()
    for entry in entries:
        trial_id = _journal_trial_id(entry)
        if trial_id is not None and _journal_entry_excludes_strategy_evidence(entry):
            excluded.add(trial_id)
    return excluded


@dataclass
class StrategyEntry:
    """A single strategy memory entry."""

    id: str
    description: str
    insight: str
    source_trial_id: int
    species: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0
    # AP-28 diagnostics (default 0/0/empty so to_dict round-trips cleanly)
    entry_type: str = "raw"
    validity_score: float = 0.5
    staleness: float = 1.0
    rrf_score: float = 0.0
    evidence_trial_ids: list[int] = field(default_factory=list)
    title: str = ""
    generalized_content: str = ""
    specificity_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrategyStore:
    """FAISS + SQLite strategy store for AutoPilot species memory.

    Stores strategy descriptions with embeddings for semantic retrieval.
    Reuses FAISSEmbeddingStore for vector storage and TaskEmbedder for
    embedding generation (hash-based fallback if no model available).
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_STRATEGY_PATH,
        embedding_dim: int = 1024,
        embedder: Any = None,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim

        # Initialize embedder (accepts mock/custom embedders for testing)
        self._embedder = embedder
        self._owns_embedder = False
        if self._embedder is None:
            try:
                from orchestration.repl_memory.embedder import TaskEmbedder
                self._embedder = TaskEmbedder()
                self._owns_embedder = True
            except Exception as e:
                logger.warning("Could not create TaskEmbedder: %s", e)

        # Initialize FAISS store
        from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore
        self._faiss = FAISSEmbeddingStore(
            path=self.path,
            dim=embedding_dim,
            index_filename="strategy_embeddings.faiss",
            id_map_filename="strategy_id_map.npy",
        )

        # Initialize SQLite
        self._db_path = self.path / "strategies.db"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                insight TEXT NOT NULL,
                source_trial_id INTEGER,
                species TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}'
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_strategies_species ON strategies(species)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_strategies_trial ON strategies(source_trial_id)"
        )

        # AP-28: additive columns for entry_type (raw/pattern/convention) and
        # per-entry context_hash. ALTER TABLE … ADD COLUMN with DEFAULT is a
        # zero-downtime migration; the existing tests / runtime keep working
        # because every read goes through ``SELECT *`` and old INSERT paths
        # rely on column defaults.
        for col, ddl in (
            ("entry_type", "ALTER TABLE strategies ADD COLUMN entry_type TEXT DEFAULT 'raw'"),
            ("context_hash", "ALTER TABLE strategies ADD COLUMN context_hash TEXT DEFAULT ''"),
            ("evidence_trial_ids", "ALTER TABLE strategies ADD COLUMN evidence_trial_ids TEXT DEFAULT '[]'"),
        ):
            try:
                self._conn.execute(ddl)
            except sqlite3.OperationalError:
                pass  # Column already present
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_strategies_entry_type ON strategies(entry_type)"
        )

        # AP-28: FTS5 keyword index parallel to ``strategies``. ``content=''``
        # contentless mode keeps ``strategies`` authoritative; we maintain the
        # FTS5 rows ourselves on store() / soft-delete paths instead of using
        # SQLite triggers, so the model fits in a single ``store()`` write.
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS strategies_fts
                USING fts5(
                    id UNINDEXED,
                    description,
                    insight,
                    species,
                    tokenize='porter unicode61'
                )
            """)
            self._fts_enabled = True
        except sqlite3.OperationalError as exc:
            # FTS5 missing is rare on stock CPython but degrade gracefully —
            # retrieve() falls back to FAISS-only.
            logger.warning("FTS5 not available, BM25 retrieval disabled: %s", exc)
            self._fts_enabled = False

        # NIB2-41: MDL conventions + Bayesian validity + content-hash staleness.
        # All additive; existing rows stay unaffected.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_conventions (
                id TEXT PRIMARY KEY,
                representative TEXT NOT NULL,
                member_ids TEXT NOT NULL,
                compression_ratio REAL NOT NULL,
                span_trials TEXT NOT NULL,
                promoted_at TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_validity (
                strategy_id TEXT PRIMARY KEY,
                alpha INTEGER NOT NULL DEFAULT 2,
                beta_fail INTEGER NOT NULL DEFAULT 0,
                quarantined INTEGER NOT NULL DEFAULT 0,
                last_checked_at TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategies(id)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS content_hashes (
                target_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    # ── NIB2-41 helpers ──────────────────────────────────────────

    def add_convention(
        self,
        representative: str,
        member_ids: list[str],
        compression_ratio: float,
        span_trials: tuple[int, int],
    ) -> str:
        """Persist a promoted MDL convention."""
        conv_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO strategy_conventions
               (id, representative, member_ids, compression_ratio, span_trials, promoted_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                conv_id,
                representative,
                json.dumps(member_ids),
                float(compression_ratio),
                json.dumps(list(span_trials)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return conv_id

    def list_conventions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, representative, member_ids, compression_ratio, span_trials, promoted_at "
            "FROM strategy_conventions ORDER BY promoted_at DESC"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "representative": r["representative"],
                "member_ids": json.loads(r["member_ids"]),
                "compression_ratio": r["compression_ratio"],
                "span_trials": json.loads(r["span_trials"]),
                "promoted_at": r["promoted_at"],
            }
            for r in rows
        ]

    def update_validity(
        self,
        strategy_id: str,
        *,
        failure: bool,
        quarantine_threshold: float = 0.40,
    ) -> tuple[float, bool]:
        """Bump Bayesian validity counters; return (validity, is_quarantined).

        Alpha starts at 2 (mild success prior); each failure increments beta_fail.
        Validity = alpha / (alpha + beta_fail). Below ``quarantine_threshold``
        we flip the quarantined flag so ``retrieve()`` can skip the entry.
        """
        self._conn.execute(
            """INSERT INTO strategy_validity (strategy_id, alpha, beta_fail, quarantined, last_checked_at)
               VALUES (?, 2, 0, 0, ?)
               ON CONFLICT(strategy_id) DO NOTHING""",
            (strategy_id, datetime.now(timezone.utc).isoformat()),
        )
        if failure:
            self._conn.execute(
                "UPDATE strategy_validity SET beta_fail = beta_fail + 1, last_checked_at = ? "
                "WHERE strategy_id = ?",
                (datetime.now(timezone.utc).isoformat(), strategy_id),
            )
        row = self._conn.execute(
            "SELECT alpha, beta_fail FROM strategy_validity WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        alpha = row["alpha"]
        beta = row["beta_fail"]
        validity = alpha / (alpha + beta)
        quarantine = validity < quarantine_threshold
        self._conn.execute(
            "UPDATE strategy_validity SET quarantined = ? WHERE strategy_id = ?",
            (1 if quarantine else 0, strategy_id),
        )
        self._conn.commit()
        return validity, quarantine

    def get_content_hash(self, target_path: str) -> str | None:
        row = self._conn.execute(
            "SELECT content_hash FROM content_hashes WHERE target_path = ?",
            (target_path,),
        ).fetchone()
        return row["content_hash"] if row else None

    def upsert_content_hash(self, target_path: str, content_hash: str) -> None:
        self._conn.execute(
            """INSERT INTO content_hashes (target_path, content_hash, last_seen_at)
               VALUES (?, ?, ?)
               ON CONFLICT(target_path) DO UPDATE SET
                   content_hash = excluded.content_hash,
                   last_seen_at = excluded.last_seen_at""",
            (target_path, content_hash, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def quarantined_ids(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT strategy_id FROM strategy_validity WHERE quarantined = 1"
        ).fetchall()
        return {r["strategy_id"] for r in rows}

    # ── AP-28 helpers ────────────────────────────────────────────

    def compute_context_hash(
        self, context_files: tuple[Path, ...] = DEFAULT_CONTEXT_FILES
    ) -> str:
        """SHA-256 of concatenated context-file contents, truncated to 16 hex.

        Files that don't exist are skipped (so a missing prompt file does not
        invalidate the entire store). 16 hex chars = 64 bits of collision
        resistance — safe for the small number of distinct configurations we
        ever see.
        """
        h = hashlib.sha256()
        for p in context_files:
            try:
                if p.exists():
                    h.update(p.read_bytes())
            except OSError:
                continue
        return h.hexdigest()[:16]

    def _validity_score(self, strategy_id: str) -> float:
        """Read Bayesian validity for a strategy as a 0–1 score.

        Uses the existing ``strategy_validity`` table (alpha/beta_fail). Falls
        back to a 0.5 prior for entries that have never been touched.
        """
        row = self._conn.execute(
            "SELECT alpha, beta_fail FROM strategy_validity WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            return 0.5
        alpha = row["alpha"] or 2
        beta = row["beta_fail"] or 0
        return alpha / (alpha + beta)

    def _retrieve_bm25(
        self, query_text: str, k: int, species: str | None = None
    ) -> list[tuple[str, float]]:
        """BM25 keyword retrieval via FTS5.

        Returns ``[(strategy_id, bm25_score), …]`` with the highest-relevance
        entry first. FTS5 ``rank`` is negative-by-convention, so we negate it
        to keep ``score`` monotonic with relevance.
        """
        if not getattr(self, "_fts_enabled", False):
            return []
        # Sanitise the query: FTS5 reserved punctuation can blow up the parser.
        sanitised = " ".join(
            tok for tok in "".join(c if c.isalnum() else " " for c in query_text).split() if tok
        )
        if not sanitised:
            return []
        sql = (
            "SELECT id, rank FROM strategies_fts WHERE strategies_fts MATCH ? "
            "ORDER BY rank LIMIT ?"
        )
        try:
            rows = self._conn.execute(sql, (sanitised, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        results: list[tuple[str, float]] = []
        for row in rows:
            sid = row[0]
            if species is not None:
                row2 = self._conn.execute(
                    "SELECT species FROM strategies WHERE id = ?", (sid,)
                ).fetchone()
                if row2 is None or row2["species"] != species:
                    continue
            results.append((sid, -float(row[1])))
        return results

    def backfill_fts(self) -> int:
        """One-time FTS5 backfill for entries created before AP-28 landed.

        Idempotent: skips entries already present in the FTS index. Returns
        the number of rows inserted.
        """
        if not getattr(self, "_fts_enabled", False):
            return 0
        existing = {
            row[0]
            for row in self._conn.execute("SELECT id FROM strategies_fts").fetchall()
        }
        rows = self._conn.execute(
            "SELECT id, description, insight, species FROM strategies"
        ).fetchall()
        inserted = 0
        for row in rows:
            if row["id"] in existing:
                continue
            self._conn.execute(
                "INSERT INTO strategies_fts(id, description, insight, species) "
                "VALUES (?, ?, ?, ?)",
                (row["id"], row["description"], row["insight"], row["species"]),
            )
            inserted += 1
        if inserted:
            self._conn.commit()
            logger.info("FTS5 backfill: inserted %d rows", inserted)
        return inserted

    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self._embedder is not None and hasattr(self._embedder, "embed_text"):
            return self._embedder.embed_text(text)
        # Hash fallback
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based pseudo-embedding (no semantic similarity)."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    def store(
        self,
        description: str,
        insight: str,
        source_trial_id: int,
        species: str,
        metadata: dict[str, Any] | None = None,
        entry_type: str = "raw",
        evidence_trial_ids: list[int] | None = None,
        title: str | None = None,
        generalized_content: str | None = None,
    ) -> str:
        """Store a strategy entry. Returns the UUID.

        AP-28: ``entry_type`` (default ``"raw"``) selects the L1/L2/L3 tier
        used by the knowledge distiller; the current configuration epoch's
        ``context_hash`` is recorded alongside the row so future retrievals
        can detect staleness.

        AP-32: new rows also carry normalized insight-format metadata:
        ``(title, description, generalized_content)`` plus specificity flags.
        The SQLite text columns remain backward-compatible retrieval fields.
        """
        entry_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = dict(metadata or {})
        format_meta = _insight_format(
            title=title,
            description=description,
            generalized_content=generalized_content,
            insight=insight,
        )
        metadata["insight_format"] = format_meta
        if generalized_content is not None:
            insight = format_meta["generalized_content"]
        context_hash = self.compute_context_hash()
        if evidence_trial_ids is None:
            evidence_trial_ids = [source_trial_id]
        evidence_trial_ids_json = json.dumps(
            [int(tid) for tid in evidence_trial_ids if tid is not None]
        )

        # Embed description + insight for retrieval
        embed_text = f"{format_meta['title']} {description} {insight}"
        embedding = self._embed(embed_text)

        # FAISS
        self._faiss.add(entry_id, embedding)
        self._faiss.save()

        # SQLite
        self._conn.execute(
            """INSERT INTO strategies
               (id, description, insight, source_trial_id, species, created_at,
                metadata_json, entry_type, context_hash, evidence_trial_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry_id, description, insight, source_trial_id, species,
             created_at, json.dumps(metadata), entry_type, context_hash,
             evidence_trial_ids_json),
        )
        if getattr(self, "_fts_enabled", False):
            try:
                self._conn.execute(
                    "INSERT INTO strategies_fts(id, description, insight, species) "
                    "VALUES (?, ?, ?, ?)",
                    (entry_id, description, insight, species),
                )
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 insert failed for %s: %s", entry_id, exc)
        self._conn.commit()

        return entry_id

    def _evidence_trial_ids_for_row(self, row: sqlite3.Row) -> list[int]:
        ids: list[int] = []
        try:
            raw = row["evidence_trial_ids"] or "[]"
        except (IndexError, KeyError):
            raw = "[]"
        try:
            decoded = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            decoded = []
        if isinstance(decoded, list):
            for item in decoded:
                try:
                    ids.append(int(item))
                except (TypeError, ValueError):
                    continue
        if ids:
            return ids
        try:
            source_trial_id = row["source_trial_id"]
        except (IndexError, KeyError):
            source_trial_id = None
        try:
            return [int(source_trial_id)]
        except (TypeError, ValueError):
            return []

    def retrieve(
        self,
        query_text: str,
        k: int = 5,
        species: Optional[str] = None,
        include_quarantined: bool = False,
        rrf_k: int = _RRF_K,
        stale_penalty: float = 0.5,
        excluded_trial_ids: set[int] | None = None,
    ) -> list[StrategyEntry]:
        """Retrieve strategies via Reciprocal Rank Fusion of FAISS + BM25.

        AP-28: hybrid retrieval. The FAISS rank captures semantic similarity;
        the FTS5 rank captures exact-term matches (species names, mutation
        types, file names) that the embedder cannot resolve. We fuse them
        with RRF (``score = Σ 1 / (rrf_k + rank_i)``) and weight the fused
        score by Bayesian validity and a content-hash staleness factor
        (``stale_penalty`` = 0.5 by default).

        Existing callers pass only ``query_text``/``k``/``species``/
        ``include_quarantined``; the new parameters default to values
        equivalent to the prior behaviour for those callers.
        """
        if self._faiss.count == 0:
            return []

        # Wider candidate pool to absorb species filtering, quarantine, and
        # FTS5/FAISS asymmetry.
        fetch_k = max(k * 3, k + 5) if species or not include_quarantined else k * 2

        # FAISS (vector similarity)
        embedding = self._embed(query_text)
        faiss_results = self._faiss.search(embedding, k=fetch_k)
        faiss_ranking: dict[str, int] = {}
        faiss_scores: dict[str, float] = {}
        for rank, (mid, score) in enumerate(faiss_results):
            faiss_ranking[mid] = rank
            faiss_scores[mid] = float(score)

        # BM25 (FTS5 keyword) — silently empty if FTS5 unavailable
        bm25_results = self._retrieve_bm25(query_text, k=fetch_k, species=species)
        bm25_ranking = {mid: rank for rank, (mid, _) in enumerate(bm25_results)}

        all_ids = set(faiss_ranking) | set(bm25_ranking)
        fused: list[tuple[str, float]] = []
        current_hash = self.compute_context_hash()
        quarantined = set() if include_quarantined else self.quarantined_ids()
        excluded_trial_ids = excluded_trial_ids or set()

        for sid in all_ids:
            score = 0.0
            if sid in faiss_ranking:
                score += 1.0 / (rrf_k + faiss_ranking[sid])
            if sid in bm25_ranking:
                score += 1.0 / (rrf_k + bm25_ranking[sid])
            fused.append((sid, score))
        fused.sort(key=lambda x: x[1], reverse=True)

        entries: list[StrategyEntry] = []
        for sid, rrf_score in fused:
            row = self._conn.execute(
                "SELECT * FROM strategies WHERE id = ?", (sid,)
            ).fetchone()
            if row is None:
                continue
            if species and row["species"] != species:
                continue
            if sid in quarantined:
                continue
            evidence_trial_ids = self._evidence_trial_ids_for_row(row)
            if excluded_trial_ids and excluded_trial_ids.intersection(evidence_trial_ids):
                continue

            validity = self._validity_score(sid)
            # Stored row predates the ``context_hash`` column on legacy DBs.
            try:
                stored_hash = row["context_hash"] or ""
            except (IndexError, KeyError):
                stored_hash = ""
            staleness = 1.0 if (not stored_hash or stored_hash == current_hash) else stale_penalty

            adjusted = rrf_score * (0.5 + validity) * staleness

            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            try:
                entry_type = row["entry_type"] or "raw"
            except (IndexError, KeyError):
                entry_type = "raw"
            format_meta = meta.get("insight_format") if isinstance(meta, dict) else {}
            if not isinstance(format_meta, dict):
                format_meta = {}
            title = _compact_text(format_meta.get("title")) or _derive_title(row["description"])
            generalized_content = (
                _compact_text(format_meta.get("generalized_content"))
                or _compact_text(row["insight"])
            )
            flags = format_meta.get("specificity_flags")
            specificity_flags = (
                sorted({str(flag) for flag in flags})
                if isinstance(flags, list)
                else _specificity_flags(title, row["description"], generalized_content)
            )

            entries.append(StrategyEntry(
                id=row["id"],
                description=row["description"],
                insight=row["insight"],
                source_trial_id=row["source_trial_id"],
                species=row["species"],
                created_at=row["created_at"],
                metadata=meta,
                similarity_score=adjusted,
                entry_type=entry_type,
                validity_score=validity,
                staleness=staleness,
                rrf_score=rrf_score,
                evidence_trial_ids=evidence_trial_ids,
                title=title,
                generalized_content=generalized_content,
                specificity_flags=specificity_flags,
            ))
            if len(entries) >= k:
                break

        return entries

    def audit_insight_specificity(self) -> list[dict[str, Any]]:
        """Return stored strategies whose insight text looks task-specific."""
        rows = self._conn.execute(
            "SELECT id, description, insight, source_trial_id, species, metadata_json "
            "FROM strategies ORDER BY created_at ASC"
        ).fetchall()
        findings: list[dict[str, Any]] = []
        for row in rows:
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            format_meta = meta.get("insight_format") if isinstance(meta, dict) else {}
            if not isinstance(format_meta, dict):
                format_meta = {}
            generalized = _compact_text(format_meta.get("generalized_content")) or row["insight"]
            flags = _specificity_flags(row["description"], row["insight"], generalized)
            if not flags:
                continue
            findings.append({
                "id": row["id"],
                "source_trial_id": row["source_trial_id"],
                "species": row["species"],
                "specificity_flags": flags,
                "title": _compact_text(format_meta.get("title")) or _derive_title(row["description"]),
            })
        return findings

    def count(self) -> int:
        """Number of strategies in the store."""
        row = self._conn.execute("SELECT COUNT(*) FROM strategies").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Persist FAISS index and close connections."""
        try:
            self._faiss.save()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
        if self._owns_embedder and hasattr(self._embedder, "close"):
            try:
                self._embedder.close()
            except Exception:
                pass
