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
        self._conn.commit()

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
    ) -> list[StrategyEntry]:
        """Retrieve strategies by semantic similarity, optionally filtered by species."""
        if self._faiss.count == 0:
            return []

        embedding = self._embed(query_text)
        # Retrieve more candidates than k to account for species filtering
        fetch_k = k * 3 if species else k
        faiss_results = self._faiss.search(embedding, k=fetch_k)

        entries: list[StrategyEntry] = []
        for memory_id, score in faiss_results:
            row = self._conn.execute(
                "SELECT * FROM strategies WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                continue
            if species and row["species"] != species:
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
