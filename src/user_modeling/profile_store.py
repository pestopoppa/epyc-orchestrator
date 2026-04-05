"""SQLite-backed user profile store.

Stores user preferences and facts as § -delimited entries in a bounded
text blob (following Hermes Agent's MEMORY.md / USER.md pattern).
SQLite provides cross-session persistence without external dependencies.

Design decisions:
  - Bounded size (configurable, default 4KB) prevents unbounded growth
  - § delimiter for entry separation (Hermes convention)
  - Frozen snapshot injected at session start for prefix cache stability
  - Thread-safe via SQLite WAL mode + module-level singleton
  - Injection scanning on all writes (B7 dependency)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Entry delimiter (Hermes convention: § section sign)
ENTRY_DELIMITER = " § "

# Default maximum profile size in characters
DEFAULT_MAX_PROFILE_CHARS = 4096

# Default database path
DEFAULT_DB_DIR = os.environ.get(
    "ORCHESTRATOR_DATA_DIR",
    os.path.expanduser("~/.epyc-orchestrator"),
)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id    TEXT PRIMARY KEY,
    profile    TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0.0,
    entry_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    fact       TEXT NOT NULL,
    category   TEXT NOT NULL DEFAULT 'general',
    created_at REAL NOT NULL,
    source     TEXT NOT NULL DEFAULT 'user',
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON user_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON user_facts(user_id, category);
"""


@dataclass
class UserFact:
    """A single extracted user preference or fact."""

    fact: str
    category: str = "general"  # general, format, workflow, style, domain
    source: str = "user"       # user, deriver, session
    created_at: float = 0.0


@dataclass
class UserProfile:
    """Bounded user profile (frozen snapshot for prompt injection)."""

    user_id: str
    profile_text: str
    facts: list[UserFact] = field(default_factory=list)
    entry_count: int = 0
    updated_at: float = 0.0


class ProfileStore:
    """SQLite-backed user profile store with bounded entries.

    Usage::

        store = ProfileStore()
        store.add_fact("default", UserFact("prefers box-drawing tables", "format"))
        profile = store.get_profile("default")
        snapshot = store.frozen_snapshot("default")  # for system prompt
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        max_profile_chars: int = DEFAULT_MAX_PROFILE_CHARS,
    ):
        if db_path is None:
            db_dir = Path(DEFAULT_DB_DIR)
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "user_profiles.db"
        self._db_path = str(db_path)
        self._max_chars = max_profile_chars
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")
        return conn

    def add_fact(self, user_id: str, fact: UserFact) -> bool:
        """Add a fact to the user profile.

        Performs injection scanning before persisting. If the profile
        would exceed the size limit, the oldest general-category fact
        is evicted.

        Args:
            user_id: User identifier (default: "default").
            fact: The fact to add.

        Returns:
            True if fact was added, False if rejected (injection scan).
        """
        from src.security.injection_scanner import scan_content

        scan = scan_content(fact.fact, source=f"user_fact:{user_id}")
        if not scan.safe:
            logger.warning(
                "Rejected user fact for %s: injection threats %s",
                user_id, scan.threats,
            )
            return False

        now = time.time()
        with self._lock, self._connect() as conn:
            # Ensure profile row exists
            conn.execute(
                "INSERT OR IGNORE INTO user_profiles (user_id, updated_at) VALUES (?, ?)",
                (user_id, now),
            )

            # Insert fact
            conn.execute(
                "INSERT INTO user_facts (user_id, fact, category, created_at, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, fact.fact, fact.category, fact.created_at or now, fact.source),
            )

            # Rebuild profile text and enforce size limit
            self._rebuild_profile(conn, user_id, now)

        return True

    def get_profile(self, user_id: str) -> UserProfile:
        """Get the full user profile with all facts.

        Args:
            user_id: User identifier.

        Returns:
            UserProfile (empty if user not found).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile, updated_at, entry_count FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if not row:
                return UserProfile(user_id=user_id, profile_text="")

            facts_rows = conn.execute(
                "SELECT fact, category, source, created_at FROM user_facts "
                "WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()

            facts = [
                UserFact(fact=r[0], category=r[1], source=r[2], created_at=r[3])
                for r in facts_rows
            ]

            return UserProfile(
                user_id=user_id,
                profile_text=row[0],
                facts=facts,
                entry_count=row[2],
                updated_at=row[1],
            )

    def frozen_snapshot(self, user_id: str) -> str:
        """Get a frozen profile text for system prompt injection.

        This text is injected at session start and not mutated mid-session,
        preserving prefix cache stability (Hermes pattern).

        Args:
            user_id: User identifier.

        Returns:
            Profile text string (empty string if no profile).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row[0] if row else ""

    def search_facts(self, user_id: str, query: str) -> list[UserFact]:
        """Search facts by substring match.

        Args:
            user_id: User identifier.
            query: Search query (case-insensitive substring).

        Returns:
            Matching facts ordered by recency.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fact, category, source, created_at FROM user_facts "
                "WHERE user_id = ? AND fact LIKE ? ORDER BY created_at DESC LIMIT 20",
                (user_id, f"%{query}%"),
            ).fetchall()
            return [
                UserFact(fact=r[0], category=r[1], source=r[2], created_at=r[3])
                for r in rows
            ]

    def remove_fact(self, user_id: str, fact_text: str) -> bool:
        """Remove a fact by exact text match.

        Args:
            user_id: User identifier.
            fact_text: Exact text of the fact to remove.

        Returns:
            True if a fact was removed.
        """
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM user_facts WHERE user_id = ? AND fact = ?",
                (user_id, fact_text),
            )
            if cursor.rowcount > 0:
                self._rebuild_profile(conn, user_id, now)
                return True
            return False

    def _rebuild_profile(
        self, conn: sqlite3.Connection, user_id: str, now: float
    ) -> None:
        """Rebuild the profile text from facts, enforcing size limit."""
        rows = conn.execute(
            "SELECT id, fact, category FROM user_facts "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()

        # Build profile text with § delimiter, newest first
        parts: list[str] = []
        total_chars = 0
        evict_ids: list[int] = []

        for row_id, fact, category in rows:
            entry = f"[{category}] {fact}" if category != "general" else fact
            if total_chars + len(entry) + len(ENTRY_DELIMITER) > self._max_chars:
                evict_ids.append(row_id)
            else:
                parts.append(entry)
                total_chars += len(entry) + len(ENTRY_DELIMITER)

        # Evict overflow entries (oldest first since we sorted DESC)
        if evict_ids:
            placeholders = ",".join("?" * len(evict_ids))
            conn.execute(
                f"DELETE FROM user_facts WHERE id IN ({placeholders})",
                evict_ids,
            )

        profile_text = ENTRY_DELIMITER.join(parts)
        conn.execute(
            "UPDATE user_profiles SET profile = ?, updated_at = ?, entry_count = ? "
            "WHERE user_id = ?",
            (profile_text, now, len(parts), user_id),
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy, thread-safe)
# ---------------------------------------------------------------------------

_store: ProfileStore | None = None
_store_lock = threading.Lock()


def get_profile_store() -> ProfileStore:
    """Get the global ProfileStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ProfileStore()
    return _store


def reset_profile_store() -> None:
    """Reset the global ProfileStore (for tests)."""
    global _store
    with _store_lock:
        _store = None
