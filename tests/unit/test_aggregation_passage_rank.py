"""AGG-1.7: ``l1_passage_bm25.v2`` (c4_numeric_row re-ranker) is opt-in; v1 is unchanged. Zero model, CPU only."""

from __future__ import annotations

import re

import pytest

from src.aggregation import passage_rank as P
from src.aggregation.registry import PassageHandle, _Section
from src.repl_environment.context_search import ContextSearchIndex

PROSE = "Total revenue discussion. Management believes total revenue growth reflects pricing and volume.\n"
ROW = "Total revenue 86,392 79,474 70,372\nNet income 8,910 7,618 7,120\n"


def _handle() -> PassageHandle:
    text = PROSE * 12 + "\n" + ROW + "\n" + PROSE * 12
    ix = ContextSearchIndex([_Section("c", text)])
    return PassageHandle(ix, text, ["c0"], [0])


# The runner's c4, copied verbatim from agg17_recall.py (0becfd52b) as the oracle.
_NUM = re.compile(r"^\(?\$?\(?[\d,]+(\.\d+)?\)?%?$")


def _c4(ix, q, k=8, pool_n=64):
    def nf(text):
        toks = text.split()
        return sum(1 for t in toks if _NUM.match(t)) / max(1, len(toks))
    pool = ix.bm25.scores(q)[:pool_n]
    key = [(0, -(s * (1.0 + nf(ix.chunks[i].text))), i) for i, s in pool]
    return [t[2] for t in sorted(key)][:k]


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv(P.FLAG_ENV, raising=False)
    assert P.ranker_version() == P.RANKER_V1 == P.DEFAULT
    assert P.ranker_version(environ={}) == P.RANKER_V1
    assert P.ranker_version(environ={P.FLAG_ENV: " "}) == P.RANKER_V1
    assert P.ranker_version(environ={P.FLAG_ENV: P.RANKER_V2}) == P.RANKER_V2
    with pytest.raises(ValueError):
        P.ranker_version(environ={P.FLAG_ENV: "v2"})


def test_v1_output_unchanged(monkeypatch):
    monkeypatch.delenv(P.FLAG_ENV, raising=False)
    h = _handle()
    for q in ("Total revenue", "net income", "pricing volume"):
        direct, _ = h.index.search(q, 8, section=None, embedder=None)
        want = [{"chunk_id": "c0", "start": d["start"], "end": d["end"], "line": d["line"], "score": d["score"]}
                for d in direct]
        assert h.search(q, 8) == want
        assert P.search(h, q, 8) == want
        assert P.search(h, q, 8, version=P.RANKER_V1) == want


def test_v2_reproduces_c4():
    h = _handle()
    ix = h.index
    row = next(i for i, c in enumerate(ix.chunks) if "86,392" in c.text)
    for q in ("Total revenue", "net income", "pricing volume", "Total revenue 2022"):
        assert P.rank_v2(ix, q, 8) == _c4(ix, q)
        hits = P.search(h, q, 8, version=P.RANKER_V2)
        by = {(c.start, c.end): i for i, c in enumerate(ix.chunks)}
        assert [by[(x["start"], x["end"])] for x in hits] == _c4(ix, q)
    assert P.rank_v2(ix, "Total revenue", 8)[0] == row


def test_numeric_frac():
    assert P.numeric_frac("Total 1,234 (56) $7.5 12% x") == pytest.approx(4 / 6)
    assert P.numeric_frac("") == 0.0
