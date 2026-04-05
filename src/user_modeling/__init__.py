"""User modeling subsystem (B1).

Cherry-picked from Hermes Agent's Honcho integration and memory system.

Provides:
  - ProfileStore: SQLite-backed persistent user profile with bounded entries
  - Deriver: Background extraction of user preferences from session logs
  - Tool functions: user_profile, user_search, user_context, user_conclude

Guarded by ``features().user_modeling``.
"""

from src.user_modeling.profile_store import ProfileStore
from src.user_modeling.deriver import derive_preferences
from src.user_modeling.tools import (
    user_conclude,
    user_context,
    user_profile,
    user_search,
)

__all__ = [
    "ProfileStore",
    "derive_preferences",
    "user_conclude",
    "user_context",
    "user_profile",
    "user_search",
]
