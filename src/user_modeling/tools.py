"""User modeling tool functions for the REPL environment.

Four tools mirroring Honcho's interface:
  - user_profile: Fast retrieval of the current user profile
  - user_search: Semantic search over stored facts
  - user_context: LLM-synthesized answer about the user
  - user_conclude: Write a persistent fact about the user

These are registered in the tool registry when ``features().user_modeling``
is enabled.
"""

from __future__ import annotations

import logging

from src.user_modeling.profile_store import (
    ProfileStore,
    UserFact,
    get_profile_store,
)

logger = logging.getLogger(__name__)

# Default user ID for single-user deployments
DEFAULT_USER_ID = "default"


def user_profile(user_id: str = DEFAULT_USER_ID) -> str:
    """Retrieve the current user profile.

    Returns the § -delimited profile text with all known preferences
    and facts about the user.

    Args:
        user_id: User identifier (default: "default").

    Returns:
        Profile text string, or "No user profile found." if empty.
    """
    store = get_profile_store()
    profile = store.get_profile(user_id)
    if not profile.profile_text:
        return "No user profile found."
    return (
        f"User profile ({profile.entry_count} entries):\n"
        f"{profile.profile_text}"
    )


def user_search(query: str, user_id: str = DEFAULT_USER_ID) -> str:
    """Search user facts by keyword.

    Args:
        query: Search query (case-insensitive substring match).
        user_id: User identifier.

    Returns:
        Formatted list of matching facts, or "No matching facts."
    """
    store = get_profile_store()
    facts = store.search_facts(user_id, query)
    if not facts:
        return f'No facts matching "{query}".'
    lines = [f"Found {len(facts)} matching fact(s):"]
    for f in facts:
        cat = f"[{f.category}] " if f.category != "general" else ""
        lines.append(f"  - {cat}{f.fact}")
    return "\n".join(lines)


def user_context(question: str, user_id: str = DEFAULT_USER_ID) -> str:
    """Get LLM-synthesized answer about the user.

    Retrieves the full profile and formats it as context for answering
    the question. In production, the calling model uses this to tailor
    its response.

    Args:
        question: Question about the user's preferences or background.
        user_id: User identifier.

    Returns:
        Contextual answer based on stored profile.
    """
    store = get_profile_store()
    profile = store.get_profile(user_id)
    if not profile.facts:
        return "No user context available — no preferences have been recorded yet."

    # Group facts by category
    by_cat: dict[str, list[str]] = {}
    for f in profile.facts:
        by_cat.setdefault(f.category, []).append(f.fact)

    lines = [f"User context for question: {question}", ""]
    for cat, facts in sorted(by_cat.items()):
        lines.append(f"**{cat.title()}**:")
        for fact in facts[:5]:  # cap per category
            lines.append(f"  - {fact}")
    return "\n".join(lines)


def user_conclude(
    fact: str,
    category: str = "general",
    user_id: str = DEFAULT_USER_ID,
) -> str:
    """Persist a fact or preference about the user.

    The fact is scanned for injection attacks before storage.
    Oldest facts are evicted if the profile exceeds its size limit.

    Args:
        fact: The preference or fact to store.
        category: One of: general, format, workflow, style, domain.
        user_id: User identifier.

    Returns:
        Confirmation message or rejection reason.
    """
    valid_categories = {"general", "format", "workflow", "style", "domain"}
    if category not in valid_categories:
        return f"Invalid category '{category}'. Use one of: {', '.join(sorted(valid_categories))}"

    store = get_profile_store()
    user_fact = UserFact(fact=fact, category=category, source="user")
    if store.add_fact(user_id, user_fact):
        return f"Saved: [{category}] {fact}"
    else:
        return "Rejected: fact failed injection safety scan."
