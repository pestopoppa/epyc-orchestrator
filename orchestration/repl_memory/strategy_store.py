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
"""

from __future__ import annotations

import json
import logging
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
    ) -> str:
        """Store a strategy entry. Returns the UUID."""
        entry_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = metadata or {}

        # Embed description + insight for retrieval
        embed_text = f"{description} {insight}"
        embedding = self._embed(embed_text)

        # FAISS
        self._faiss.add(entry_id, embedding)
        self._faiss.save()

        # SQLite
        self._conn.execute(
            """INSERT INTO strategies
               (id, description, insight, source_trial_id, species, created_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entry_id, description, insight, source_trial_id, species,
             created_at, json.dumps(metadata)),
        )
        self._conn.commit()

        return entry_id

    def retrieve(
        self,
        query_text: str,
        k: int = 5,
        species: Optional[str] = None,
        include_quarantined: bool = False,
    ) -> list[StrategyEntry]:
        """Retrieve strategies by semantic similarity, optionally filtered by species.

        Quarantined entries (NIB2-41 staleness) are excluded by default.
        """
        if self._faiss.count == 0:
            return []

        embedding = self._embed(query_text)
        # Retrieve more candidates than k to account for species filtering.
        # Quarantine may also drop entries, so widen the pool again when active.
        fetch_k = k * 3 if (species or not include_quarantined) else k
        faiss_results = self._faiss.search(embedding, k=fetch_k)

        quarantined = set() if include_quarantined else self.quarantined_ids()

        entries: list[StrategyEntry] = []
        for memory_id, score in faiss_results:
            row = self._conn.execute(
                "SELECT * FROM strategies WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                continue
            if species and row["species"] != species:
                continue
            if row["id"] in quarantined:
                continue
            entries.append(StrategyEntry(
                id=row["id"],
                description=row["description"],
                insight=row["insight"],
                source_trial_id=row["source_trial_id"],
                species=row["species"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata_json"]),
                similarity_score=float(score),
            ))
            if len(entries) >= k:
                break

        return entries

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
