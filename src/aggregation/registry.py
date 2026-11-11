"""Persistent index registry over the corpus store (AGG-0.2, L1-L4 identity).

Indexes are keyed by (corpus_id, content_hash, builder, model_id) — the
REPL-EMB-2.5 index-identity rule. A key whose content hash no longer matches the
corpus is stale and is rebuilt; an unchanged corpus is never rebuilt.

Builders:
    l1_bm25       L1, reuses ``context_search.BM25Index`` over store chunks (one BM25 document per stored chunk).
    l1_passage_bm25
                  L1, the ``context.search`` engine (``ContextSearchIndex``: line-aware 800-character passages +
                  BM25, lexical mode) over the corpus text rebuilt from the store's chunks, built ONCE per
                  (corpus, content hash). Same chunker, tokenizer, BM25 parameters and ranking as a per-request
                  ``context.search`` over that text; hits are passage spans, so the caller reads and shows the
                  matching text, not the head of a multi-thousand-character stored chunk.
    l3_facts_det  L3, ``facts_det.extract_corpus_facts`` written to ``facts``.
    l2_dense      L2, reserved: Phase 0 here builds no embeddings. Raises.
    l4_facts_llm  L4, built lazily per field by ``facts_llm``; never via ``build``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

from src.aggregation import require_enabled
from src.aggregation.store import CorpusStore
from src.repl_environment.context_search import (
    QUERY_MAX_CHARS,
    SEARCH_MAX_K,
    BM25Index,
    ContextSearchIndex,
)

LAYER_OF = {
    "l1_bm25": "L1",
    "l1_passage_bm25": "L1",
    "l2_dense": "L2",
    "l3_facts_det": "L3",
    "l4_facts_llm": "L4",
}

#: Joins the stored chunks of a corpus (ordered by doc_id, ordinal) into the one text the passage index chunks.
CORPUS_TEXT_SEP = "\n\n"


class IndexNotAvailable(RuntimeError):
    pass


@dataclass
class BM25Handle:
    index: BM25Index
    chunk_ids: list[str]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return [(self.chunk_ids[i], s) for i, s in self.index.scores(query)[:k]]


@dataclass
class _Section:
    """The two attributes ``chunk_sections`` reads from a section."""

    name: str
    text: str


@dataclass
class PassageHandle:
    """A passage-granularity L1 index: ``ContextSearchIndex`` over the corpus text, plus the map from a
    passage back to the stored chunk it starts in (provenance; the span itself is a corpus-text offset)."""

    index: ContextSearchIndex
    text: str
    chunk_ids: list[str]
    chunk_starts: list[int]

    def chunk_id_at(self, offset: int) -> str:
        i = bisect.bisect_right(self.chunk_starts, max(0, offset)) - 1
        return self.chunk_ids[max(i, 0)]

    def search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Ranked passage hits ``{chunk_id, start, end, line, score}`` (lexical mode, dense off), exactly the
        ranking ``ContextBundle.search`` gives over the same text."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = max(1, min(int(k), SEARCH_MAX_K))
        hits, _meta = self.index.search(query[:QUERY_MAX_CHARS], k, section=None, embedder=None)
        return [{"chunk_id": self.chunk_id_at(h["start"]), "start": h["start"], "end": h["end"],
                 "line": h["line"], "score": h["score"]} for h in hits]

    def span(self, start: int, end: int) -> str:
        return self.text[start:end]


class IndexRegistry:
    def __init__(self, store: CorpusStore) -> None:
        self.store = store
        self._live: dict[tuple[str, str, str, str], Any] = {}

    def key(self, corpus_id: str, builder: str, model_id: str = "") -> tuple[str, str, str, str]:
        return (corpus_id, self.store.corpus_content_hash(corpus_id), builder, model_id)

    def get_or_build(self, corpus_id: str, builder: str, model_id: str = "") -> Any:
        require_enabled()
        key = self.key(corpus_id, builder, model_id)
        if key in self._live:
            return self._live[key]
        if builder == "l1_bm25":
            chunks = self.store.chunks(corpus_id)
            handle: Any = BM25Handle(BM25Index([c["text"] for c in chunks]),
                                     [c["chunk_id"] for c in chunks])
            n = len(chunks)
        elif builder == "l1_passage_bm25":
            chunks = self.store.chunks(corpus_id)
            starts, pos = [], 0
            for c in chunks:
                starts.append(pos)
                pos += len(c["text"] or "") + len(CORPUS_TEXT_SEP)
            text = CORPUS_TEXT_SEP.join(c["text"] or "" for c in chunks)
            index = ContextSearchIndex([_Section(corpus_id, text)])
            handle = PassageHandle(index, text, [c["chunk_id"] for c in chunks], starts)
            n = len(index.chunks)
        elif builder == "l3_facts_det":
            from src.aggregation.facts_det import extract_corpus_facts  # noqa: PLC0415

            facts = extract_corpus_facts(self.store, corpus_id)
            self.store.write_facts(facts)
            handle, n = facts, len(facts)
        elif builder in ("l2_dense", "l4_facts_llm"):
            raise IndexNotAvailable(
                f"{builder} is not built by the registry in Phase 0 (no embedding / lazy L4)"
            )
        else:
            raise KeyError(f"unknown builder {builder!r}")
        self.store.record_index(*key, layer=LAYER_OF[builder], n_items=n)
        self._live[key] = handle
        return handle
