"""Versioned passage ranker for the ``l1_passage_bm25`` index (AGG-1.7).

``l1_passage_bm25.v1`` (the default) is the ranking ``PassageHandle.search`` gives: ``ContextSearchIndex.search`` in
lexical mode. It is not re-implemented here; v1 callers keep calling ``PassageHandle.search`` unchanged.

``l1_passage_bm25.v2`` is a re-ranker over the SAME index (no extra build): take the BM25 top
``CANDIDATES_PER_RANKER`` (64) of the query and order them by ``score * (1 + numeric_frac(passage))``, ties by
passage id. It is candidate ``c4_numeric_row`` of ``epyc-inference-research``
``benchmarks/agg_phase1/agg17_recall.py`` (0becfd52b), ported with its exact logic and parameters; on the AGG-1.5
labels it raised recall@8 from 16/20 to 19/20 with no label or trace query lost (agg17_result.json, 5e4808d39).

Selection: ``ranker_version(version)`` with an explicit version, else the environment variable
``ORCHESTRATOR_AGG_PASSAGE_RANKER`` (``l1_passage_bm25.v1`` | ``l1_passage_bm25.v2``), else v1. An unknown value is
refused, never silently mapped to the default (the ``ORCHESTRATOR_AGG_PIPELINE`` pattern of ``pipeline.py``).
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

from src.repl_environment.context_search import CANDIDATES_PER_RANKER, QUERY_MAX_CHARS, SEARCH_MAX_K

RANKER_V1 = "l1_passage_bm25.v1"
RANKER_V2 = "l1_passage_bm25.v2"
RANKERS = (RANKER_V1, RANKER_V2)
FLAG_ENV = "ORCHESTRATOR_AGG_PASSAGE_RANKER"
DEFAULT = RANKER_V1

#: A numeric table-cell token (== agg17_recall.NUM_TOKEN): 1,234  (1,234)  $1,234.5  12%
NUM_TOKEN = re.compile(r"^\(?\$?\(?[\d,]+(\.\d+)?\)?%?$")


def ranker_version(version: str | None = None, *, environ: Mapping[str, str] | None = None) -> str:
    """``version`` (explicit wins), else ``ORCHESTRATOR_AGG_PASSAGE_RANKER``, else ``l1_passage_bm25.v1``."""
    if version is None:
        env = os.environ if environ is None else environ
        version = (env.get(FLAG_ENV) or "").strip() or DEFAULT
    if version not in RANKERS:
        raise ValueError(f"unknown passage ranker {version!r}; known: {RANKERS}")
    return version


def numeric_frac(text: str) -> float:
    """Share of whitespace tokens of ``text`` that are numeric cells (== agg17_recall.numeric_frac)."""
    toks = text.split()
    return sum(1 for t in toks if NUM_TOKEN.match(t)) / max(1, len(toks))


def rank_v2(index: Any, query: str, k: int = 8) -> list[int]:
    """Passage ids of ``index`` (a ``ContextSearchIndex``) in v2 order, top ``k`` (== agg17_recall.rank c4)."""
    pool = index.bm25.scores(query)[:CANDIDATES_PER_RANKER]
    key = [(0, -(s * (1.0 + numeric_frac(index.chunks[i].text))), i) for i, s in pool]
    return [t[2] for t in sorted(key)][:k]


def search(handle: Any, query: str, k: int = 8, version: str | None = None) -> list[dict[str, Any]]:
    """Ranked passage hits ``{chunk_id, start, end, line, score}`` of a ``PassageHandle`` under the selected ranker.
    v1 delegates to ``handle.search`` unchanged; v2's ``score`` is the boosted re-rank score."""
    v = ranker_version(version)
    if v == RANKER_V1:
        return handle.search(query, k)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    k = max(1, min(int(k), SEARCH_MAX_K))
    q = query[:QUERY_MAX_CHARS]
    ix = handle.index
    scores = dict(ix.bm25.scores(q)[:CANDIDATES_PER_RANKER])
    out = []
    for i in rank_v2(ix, q, k):
        c = ix.chunks[i]
        out.append({"chunk_id": handle.chunk_id_at(c.start), "start": c.start, "end": c.end, "line": c.line,
                    "score": scores[i] * (1.0 + numeric_frac(c.text))})
    return out
