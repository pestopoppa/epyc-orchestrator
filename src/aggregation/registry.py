"""Persistent index registry over the corpus store (AGG-0.2, L1-L4 identity).

Indexes are keyed by (corpus_id, content_hash, builder, model_id) — the
REPL-EMB-2.5 index-identity rule. A key whose content hash no longer matches the
corpus is stale and is rebuilt; an unchanged corpus is never rebuilt.

Builders:
    l1_bm25       L1, reuses ``context_search.BM25Index`` over store chunks.
    l3_facts_det  L3, ``facts_det.extract_corpus_facts`` written to ``facts``.
    l2_dense      L2, reserved: Phase 0 here builds no embeddings. Raises.
    l4_facts_llm  L4, built lazily per field by ``facts_llm``; never via ``build``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.aggregation import require_enabled
from src.aggregation.store import CorpusStore
from src.repl_environment.context_search import BM25Index

LAYER_OF = {"l1_bm25": "L1", "l2_dense": "L2", "l3_facts_det": "L3", "l4_facts_llm": "L4"}


class IndexNotAvailable(RuntimeError):
    pass


@dataclass
class BM25Handle:
    index: BM25Index
    chunk_ids: list[str]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return [(self.chunk_ids[i], s) for i, s in self.index.scores(query)[:k]]


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
