"""Research-like prompt detector (NIB2-45 MindDR MD-2).

Pure-Python detector with no MemRL dependency. The orchestrator pipeline
calls ``is_research_like(prompt)`` at routing time; when ``True`` AND
``features.deep_research_mode`` is set, the request enters the three-
agent research pipeline (Planning → DeepSearch → Report).

Heuristics combine three cheap signals — research keywords, comparison
patterns, and multi-clause structure — into a single boolean. A prompt
passes if any of:

1. Research verbs like "investigate", "deep dive", "survey" appear.
2. "X vs Y" / "X or Y or Z" comparison structure at the clause level.
3. Multi-question structure (≥2 "?") or explicit enumeration
   ("compare ... across") suggesting decomposition will help.

The detector intentionally prefers recall over precision — false
positives route to deep-research mode (more compute, slower) but not
to wrong answers. False negatives skip the richer pipeline silently.
MemRL can refine this later via the ``research_like`` exemplars in
``classifier_config.yaml``, wired through ``ClassificationRetriever``.
"""

from __future__ import annotations

import re

_RESEARCH_VERBS = (
    "deep dive",
    "deep-dive",
    "investigate",
    "survey",
    "research and",
    "research the",
    "comprehensive",
    "landscape",
    "synthesize",
    "synthesise",
    "walk me through",
    "report on",
    "overview of",
    "review the",
    "state of the",
    "state-of-the-art",
    "current state",
    "literature",
)

_COMPARISON_PATTERNS = (
    re.compile(r"\b\S+\s+vs\.?\s+\S+", re.IGNORECASE),
    re.compile(r"\bcompare\s+\S+\s+(?:and|to|with|against)\s+\S+", re.IGNORECASE),
    # "Compare X, Y, Z across A, B" — explicit axis list.
    re.compile(r"\bcompare\s+[\w\-]+(?:\s*,\s*[\w\-]+){1,}\s+(?:and|across)\b", re.IGNORECASE),
    # Looser "compare ... across" — anything between compare and across.
    re.compile(r"\bcompare\s+.+?\s+across\b", re.IGNORECASE),
    re.compile(r"\btradeoffs?\b", re.IGNORECASE),
    re.compile(r"\bhow\s+do\s+.+?\s+differ\b", re.IGNORECASE),
    re.compile(r"\bhow\s+do\s+.+?\s+compare\b", re.IGNORECASE),
    re.compile(r"\bwhen\s+should\s+(?:each|one|you|we)\b", re.IGNORECASE),
    re.compile(r"\bacross\s+(?:multiple|several|various)\b", re.IGNORECASE),
    # "across X, Y, and Z" or "across A loss, B footprint, C speed" — cross-axis list.
    re.compile(r"\bacross\s+[\w\-]+(?:\s*(?:loss|footprint|speed|quality|cost|latency|scales|sizes|families))?(?:\s*,\s*\S+){2,}", re.IGNORECASE),
)

_DECOMPOSITION_PATTERNS = (
    re.compile(r"\bcompare\s+\S+\s+across\b", re.IGNORECASE),
    re.compile(r"\band\s+their\s+(?:tradeoffs?|limits|strengths|weaknesses)\b", re.IGNORECASE),
)


def is_research_like(prompt: str) -> bool:
    """Return ``True`` when the prompt looks like a research-style query.

    Prefers recall — when in doubt, return True so the multi-agent
    pipeline gets a chance to structure the work. The router then
    checks ``features.deep_research_mode`` before actually switching.
    """
    if not prompt or not prompt.strip():
        return False

    lower = prompt.lower()

    # 1. Direct research verbs.
    for verb in _RESEARCH_VERBS:
        if verb in lower:
            return True

    # 2. Comparison patterns.
    for pat in _COMPARISON_PATTERNS:
        if pat.search(prompt):
            return True

    # 3. Multi-question structure.
    if prompt.count("?") >= 2:
        return True

    # 4. Explicit decomposition hints.
    for pat in _DECOMPOSITION_PATTERNS:
        if pat.search(prompt):
            return True

    return False


def score_research_like(prompt: str) -> float:
    """Return a 0-1 score reflecting how research-like the prompt is.

    Useful for telemetry / logit_probe-style downstream calibration. The
    hard boolean is the routing signal; the score helps autopilot tune
    thresholds offline.
    """
    if not prompt or not prompt.strip():
        return 0.0
    lower = prompt.lower()
    score = 0.0
    verb_hits = sum(1 for v in _RESEARCH_VERBS if v in lower)
    score += min(verb_hits * 0.25, 0.50)
    score += min(sum(1 for p in _COMPARISON_PATTERNS if p.search(prompt)) * 0.15, 0.30)
    if prompt.count("?") >= 2:
        score += 0.20
    return min(score, 1.0)
