"""Frecency-based file access scoring with SQLite persistence.

Tracks file access frequency and recency to produce a combined score
useful for ranking files in search results and suggestions.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "file_recency.db"

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS file_access (
    path TEXT PRIMARY KEY,
    access_count INTEGER DEFAULT 0,
    last_access REAL,
    frecency_score REAL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS combo_access (
    query TEXT,
    path TEXT,
    count INTEGER DEFAULT 0,
    last_seen REAL,
    PRIMARY KEY (query, path)
);
"""


class FrecencyStore:
    """SQLite-backed frecency scorer for file access patterns.

    Scoring formula:
        score = freq_weight * log(access_count + 1)
              + recency_weight * exp(-age_hours / half_life)

    Combo boost (for repeated query+file pairs):
        boosted = score * (1 + combo_weight * log(pair_count + 1))
    """

    def __init__(
        self,
        db_path: Union[str, Path, None] = None,
        freq_weight: float = 1.0,
        recency_weight: float = 2.0,
        half_life: float = 24.0,
        combo_weight: float = 0.5,
    ) -> None:
        self.freq_weight = freq_weight
        self.recency_weight = recency_weight
        self.half_life = half_life
        self.combo_weight = combo_weight

        resolved = str(db_path) if db_path is not None else str(_DEFAULT_DB_PATH)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(resolved, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_access(self, path: str, query: str | None = None) -> None:
        """Record a file access. Increments count, updates timestamp, recomputes score."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "SELECT access_count FROM file_access WHERE path = ?", (path,)
            )
            row = cur.fetchone()
            if row is None:
                access_count = 1
                self._conn.execute(
                    "INSERT INTO file_access (path, access_count, last_access, frecency_score) "
                    "VALUES (?, ?, ?, ?)",
                    (path, access_count, now, self._compute_score(access_count, now)),
                )
            else:
                access_count = row[0] + 1
                self._conn.execute(
                    "UPDATE file_access SET access_count = ?, last_access = ?, frecency_score = ? "
                    "WHERE path = ?",
                    (access_count, now, self._compute_score(access_count, now), path),
                )

            if query is not None:
                cur2 = self._conn.execute(
                    "SELECT count FROM combo_access WHERE query = ? AND path = ?",
                    (query, path),
                )
                combo_row = cur2.fetchone()
                if combo_row is None:
                    self._conn.execute(
                        "INSERT INTO combo_access (query, path, count, last_seen) "
                        "VALUES (?, ?, 1, ?)",
                        (query, path, now),
                    )
                else:
                    self._conn.execute(
                        "UPDATE combo_access SET count = count + 1, last_seen = ? "
                        "WHERE query = ? AND path = ?",
                        (now, query, path),
                    )
            self._conn.commit()
        logger.debug("Recorded access: path=%s query=%s count=%d", path, query, access_count)

    def get_score(self, path: str) -> float:
        """Get current frecency score for a path. Returns 0.0 if not tracked."""
        row = self._conn.execute(
            "SELECT access_count, last_access FROM file_access WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return 0.0
        return self._compute_score(row[0], row[1])

    def get_scores(self, paths: list[str]) -> dict[str, float]:
        """Batch lookup of frecency scores. Returns {path: score} for all paths."""
        if not paths:
            return {}
        placeholders = ",".join("?" for _ in paths)
        rows = self._conn.execute(
            f"SELECT path, access_count, last_access FROM file_access "  # noqa: S608
            f"WHERE path IN ({placeholders})",
            paths,
        ).fetchall()
        scores: dict[str, float] = {p: 0.0 for p in paths}
        for row_path, access_count, last_access in rows:
            scores[row_path] = self._compute_score(access_count, last_access)
        return scores

    def get_combo_boost(self, query: str, path: str) -> float:
        """Get combo boost multiplier for a (query, file) pair. Returns 1.0 if not tracked."""
        row = self._conn.execute(
            "SELECT count FROM combo_access WHERE query = ? AND path = ?",
            (query, path),
        ).fetchone()
        if row is None:
            return 1.0
        pair_count = row[0]
        return 1.0 + self.combo_weight * math.log(pair_count + 1)

    def top_files(self, limit: int = 20) -> list[tuple[str, float]]:
        """Return the top-K files ranked by current frecency score.

        Recomputes scores at query time to account for recency decay.

        Args:
            limit: Maximum number of files to return.

        Returns:
            List of (path, score) tuples, highest score first.
        """
        rows = self._conn.execute(
            "SELECT path, access_count, last_access FROM file_access"
        ).fetchall()
        if not rows:
            return []
        scored = [(path, self._compute_score(count, last_access))
                  for path, count, last_access in rows]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _compute_score(self, access_count: int, last_access: float) -> float:
        """Compute the frecency score from raw data."""
        age_hours = (time.time() - last_access) / 3600.0
        return (
            self.freq_weight * math.log(access_count + 1)
            + self.recency_weight * math.exp(-age_hours / self.half_life)
        )

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
