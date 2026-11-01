"""Bounded SEARCH command facade for read-only scouts.

This module deliberately does not build an index or resolve an embedding backend. The owner
passes an already constructed ``ContextSearchIndex`` or ``ContextBundle``. Passing a search
index without an embedder selects its real lexical BM25 path; omitting the provider returns an
explicit unavailable result. SEARCH results remain the existing pointer records, so consumers
must use their authorized read path to fetch text.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Collection
from dataclasses import dataclass
from itertools import islice
from typing import Any

SEARCH_DEFAULT_K = 8
SEARCH_MAX_K = 50
SEARCH_MAX_QUERY_CHARS = 2000
_POINTER_KEYS = ("section", "start", "end", "line", "score", "via", "mode")
_RANK_SOURCES = frozenset({"bm25", "dense"})
_SEARCH_MODES = frozenset({"hybrid", "lexical"})


def _valid_mode(value: Any) -> bool:
    return isinstance(value, str) and (
        value in _SEARCH_MODES
        or (value.startswith("lexical_fallback:") and bool(value[len("lexical_fallback:") :]))
    )


@dataclass(frozen=True)
class SearchCommand:
    """A parsed command; query text is preserved except surrounding whitespace."""

    query: str


def parse_search_command(line: str) -> SearchCommand | None:
    """Parse a complete ``SEARCH <query>`` line, or return None for another verb.

    The query may contain spaces. Empty queries and malformed command lines raise ValueError,
    allowing the caller to report a command error without accidentally searching.
    """
    if not isinstance(line, str):
        raise TypeError("SEARCH command must be text")
    parts = line.strip().split(None, 1)
    if not parts or parts[0].upper() != "SEARCH":
        return None
    if len(parts) != 2 or not parts[1].strip():
        raise ValueError("SEARCH requires a non-empty query")
    return SearchCommand(parts[1].strip())


class ScoutSearch:
    """Execute bounded queries through an existing search provider.

    ``provider`` is either an existing ``ContextSearchIndex`` (whose ``search`` accepts
    ``embedder``) or a ``ContextBundle`` (whose ``search`` resolves its already-configured
    embedder). ``allowed_sections`` is an optional allowlist of opaque section IDs. When set,
    each query must name one allowed section; no pointer is dereferenced here.
    """

    def __init__(
        self,
        provider: Any | None,
        *,
        embedder: Any | None = None,
        allowed_sections: Collection[str] | None = None,
    ) -> None:
        self.provider = provider
        self.embedder = embedder
        if allowed_sections is not None:
            if isinstance(allowed_sections, (str, bytes)) or not isinstance(
                allowed_sections, Collection
            ):
                raise TypeError("allowed_sections must be a collection of opaque string IDs")
            if any(not isinstance(section, str) or not section for section in allowed_sections):
                raise ValueError("allowed_sections must contain non-empty opaque string IDs")
        self.allowed_sections = None if allowed_sections is None else frozenset(allowed_sections)

    def execute(
        self,
        query: str,
        *,
        k: int = SEARCH_DEFAULT_K,
        section: str | None = None,
    ) -> dict[str, Any]:
        """Return deterministic status/mode metadata and bounded pointer hits.

        The provider's hit record schema is preserved. Query size and result count follow the
        existing context-search limits; an absent backend is explicit and never emulated.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("SEARCH query must be a non-empty string")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("SEARCH k must be an integer")
        if section is not None and not isinstance(section, str):
            raise TypeError("SEARCH section must be an opaque string ID")
        if self.allowed_sections is not None:
            if section is None:
                raise ValueError("SEARCH section is required for this scoped search")
            if section not in self.allowed_sections:
                raise KeyError(f"SEARCH section is outside the authorized scope: {section}")

        normalized_query = query.strip()[:SEARCH_MAX_QUERY_CHARS]
        bounded_k = max(1, min(k, SEARCH_MAX_K))
        if self.provider is None:
            return {
                "status": "unavailable",
                "query": normalized_query,
                "k": bounded_k,
                "section": section,
                "mode": "unavailable",
                "reason": "search_backend_unavailable",
                "hits": [],
            }

        search = getattr(self.provider, "search", None)
        if not callable(search):
            return {
                "status": "unavailable",
                "query": normalized_query,
                "k": bounded_k,
                "section": section,
                "mode": "unavailable",
                "reason": "search_backend_missing_interface",
                "hits": [],
            }

        # ContextSearchIndex returns (hits, meta); ContextBundle.search returns hits and logs
        # mode/provenance internally. Support both without constructing or retaining an index.
        params = inspect.signature(search).parameters
        if "embedder" in params:
            result = search(
                normalized_query,
                bounded_k,
                section=section,
                embedder=self.embedder,
            )
        else:
            # ContextBundle.search owns embedder resolution and has no embedder parameter.
            result = search(normalized_query, bounded_k, section=section)
        meta: dict[str, Any] = {}
        if isinstance(result, tuple) and len(result) == 2:
            raw_hits, raw_meta = result
            meta = raw_meta if isinstance(raw_meta, dict) else {}
        else:
            raw_hits = result

        hits = []
        for raw in islice(raw_hits if raw_hits is not None else (), bounded_k):
            if not isinstance(raw, dict) or any(key not in raw for key in _POINTER_KEYS):
                raise ValueError("SEARCH backend returned an invalid pointer record")
            if not isinstance(raw["section"], str) or not raw["section"]:
                raise ValueError("SEARCH backend returned an invalid section ID")
            for name, minimum in (("start", 0), ("end", 1), ("line", 1)):
                value = raw[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"SEARCH backend returned an invalid {name}")
            if raw["end"] <= raw["start"]:
                raise ValueError("SEARCH backend returned an empty or reversed pointer range")
            score = raw["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("SEARCH backend returned a non-numeric score")
            try:
                finite_score = math.isfinite(score)
            except OverflowError:
                finite_score = False
            if not finite_score or score <= 0:
                raise ValueError("SEARCH backend returned a non-finite or invalid score")
            if (
                not isinstance(raw["via"], list)
                or not raw["via"]
                or any(not isinstance(source, str) or source not in _RANK_SOURCES for source in raw["via"])
                or len(set(raw["via"])) != len(raw["via"])
            ):
                raise ValueError("SEARCH backend returned invalid ranking sources")
            if not _valid_mode(raw["mode"]):
                raise ValueError("SEARCH backend returned an unknown mode")
            if section is not None and raw["section"] != section:
                # Treat a provider that ignores the requested scope as a contract violation.
                raise ValueError("SEARCH backend returned a pointer outside the requested section")
            if self.allowed_sections is not None and raw["section"] not in self.allowed_sections:
                raise ValueError("SEARCH backend returned a pointer outside the authorized scope")
            hits.append({key: raw[key] for key in _POINTER_KEYS})

        mode = meta.get("mode")
        status = "ok"
        if not _valid_mode(mode):
            if mode == "unavailable":
                status = "unavailable"
            elif "mode" in meta:
                # An explicit but unrecognized provider claim cannot be repaired from its hits.
                status = "unknown"
            else:
                modes = {hit["mode"] for hit in hits}
                if len(modes) == 1:
                    mode = next(iter(modes))
                else:
                    status = "unknown"
            mode = status if status != "ok" else mode
        return {
            "status": status,
            "query": normalized_query,
            "k": bounded_k,
            "section": section,
            "mode": mode,
            "reason": meta.get("reason"),
            "hits": hits,
        }
