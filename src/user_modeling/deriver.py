"""Background deriver for extracting user preferences from session logs.

Reviews completed session logs and extracts durable preferences, corrections,
and workflow patterns. Extracted facts are persisted via ProfileStore.

In production, runs as a post-session background task using worker_explore
(Qwen2.5-7B) as the auxiliary LLM. In tests, can be called synchronously
with a mock LLM.

Cherry-picked from Hermes Agent's Honcho Deriver/Dreamer pattern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.user_modeling.profile_store import ProfileStore, UserFact

logger = logging.getLogger(__name__)

# Extraction prompt template for the deriver LLM
_DERIVE_PROMPT = """\
You are a preference extraction engine. Given a session transcript, extract
durable user preferences — things the user would want remembered across sessions.

Focus on:
- Output format preferences (tables, markdown, code style)
- Workflow preferences (don't run X without asking, always show Y)
- Domain knowledge indicators (expertise level, role)
- Explicit corrections ("don't do X", "I prefer Y")

Ignore:
- Transient task details (specific file names, one-time requests)
- Information derivable from the codebase
- Anything the user explicitly said to forget

Output each preference on its own line in the format:
PREF [category] preference text

Categories: format, workflow, style, domain, general

Example output:
PREF [format] Prefers box-drawing Unicode tables over ASCII
PREF [workflow] Always show TPS metrics after benchmark runs
PREF [domain] Senior systems engineer with NUMA optimization expertise

Session transcript:
{transcript}
"""

_PREF_RE = re.compile(r"^PREF\s+\[(\w+)\]\s+(.+)$", re.MULTILINE)


@dataclass
class DeriverResult:
    """Result of a preference extraction run."""

    facts_extracted: list[UserFact]
    facts_added: int
    facts_rejected: int


def extract_preferences_from_text(text: str) -> list[UserFact]:
    """Parse LLM output into UserFact objects.

    Args:
        text: Raw LLM output following the PREF [category] format.

    Returns:
        List of extracted UserFact objects.
    """
    facts = []
    for match in _PREF_RE.finditer(text):
        category = match.group(1).lower()
        fact_text = match.group(2).strip()
        if fact_text and len(fact_text) > 5:  # skip trivially short
            facts.append(UserFact(
                fact=fact_text,
                category=category,
                source="deriver",
            ))
    return facts


def derive_preferences(
    store: ProfileStore,
    user_id: str,
    transcript: str,
    llm_call: callable | None = None,
) -> DeriverResult:
    """Extract and persist user preferences from a session transcript.

    If ``llm_call`` is None, only parses pre-formatted text (for testing).
    In production, pass ``primitives.llm_call`` with role="worker_explore".

    Args:
        store: ProfileStore to persist facts to.
        user_id: User identifier.
        transcript: Session transcript text.
        llm_call: Optional LLM function(prompt, role) -> str.

    Returns:
        DeriverResult with extraction statistics.
    """
    if llm_call is not None:
        prompt = _DERIVE_PROMPT.format(transcript=transcript[:8000])
        try:
            raw_output = llm_call(prompt, role="worker_explore")
        except Exception as exc:
            logger.warning("Deriver LLM call failed: %s", exc)
            return DeriverResult(facts_extracted=[], facts_added=0, facts_rejected=0)
    else:
        raw_output = transcript  # assume pre-formatted for testing

    facts = extract_preferences_from_text(raw_output)

    added = 0
    rejected = 0
    for fact in facts:
        if store.add_fact(user_id, fact):
            added += 1
        else:
            rejected += 1

    logger.info(
        "Deriver for %s: extracted=%d, added=%d, rejected=%d",
        user_id, len(facts), added, rejected,
    )

    return DeriverResult(
        facts_extracted=facts,
        facts_added=added,
        facts_rejected=rejected,
    )
