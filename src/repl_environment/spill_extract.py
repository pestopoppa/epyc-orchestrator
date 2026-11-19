"""REPL-EMB-3.1: extractive spill summary (no worker call).

Replaces the 512-token worker summary of ``REPLEnvironment._spill_output`` with
verbatim material only: head, tail, and the top-k chunks most relevant to the
task query (lexical token overlap, IDF-weighted over the output's own chunks).
Pure and deterministic; nothing here talks to a model or an embedder.

Default OFF behind the ``spill_extractive_summary`` feature flag.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d{2,}")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def split_chunks(text: str, chunk_chars: int) -> list[tuple[int, str]]:
    """Split on line boundaries into ~chunk_chars chunks. Returns (start_offset, text)."""
    chunks: list[tuple[int, str]] = []
    cur_start = 0
    cur_len = 0
    for line in text.splitlines(keepends=True):
        if cur_len and cur_len + len(line) > chunk_chars:
            chunks.append((cur_start, text[cur_start : cur_start + cur_len]))
            cur_start += cur_len
            cur_len = 0
        cur_len += len(line)
    if cur_len:
        chunks.append((cur_start, text[cur_start : cur_start + cur_len]))
    return chunks


def top_k_chunks(
    output: str,
    query: str,
    *,
    k: int,
    chunk_chars: int,
    exclude_before: int = 0,
    exclude_after: int | None = None,
) -> list[tuple[int, str, float]]:
    """Rank chunks of ``output`` by IDF-weighted overlap with ``query``.

    Chunks that start before ``exclude_before`` or at/after ``exclude_after`` (already shown
    verbatim as head/tail) are skipped. Zero-score chunks are never returned.
    Result is ordered by offset, not score, so it reads in document order.
    """
    q = set(_tokens(query))
    if not q or k <= 0:
        return []
    chunks = split_chunks(output, chunk_chars)
    n = len(chunks)
    if n == 0:
        return []
    chunk_toks = [set(_tokens(c)) for _, c in chunks]
    df: Counter[str] = Counter()
    for toks in chunk_toks:
        df.update(toks & q)
    scored: list[tuple[float, int]] = []
    for i, (off, _c) in enumerate(chunks):
        if off < exclude_before:
            continue
        if exclude_after is not None and off >= exclude_after:
            continue
        s = sum(math.log(1.0 + n / df[t]) for t in chunk_toks[i] & q)
        if s > 0:
            scored.append((s, i))
    scored.sort(key=lambda p: (-p[0], p[1]))
    picked = sorted(scored[:k], key=lambda p: p[1])
    return [(chunks[i][0], chunks[i][1], s) for s, i in picked]


def extractive_summary(
    output: str,
    query: str,
    *,
    head_chars: int = 600,
    tail_chars: int = 400,
    top_k: int = 3,
    chunk_chars: int = 400,
) -> str:
    """Head + tail + top-k task-relevant chunks, all verbatim, with char offsets."""
    n = len(output)
    if n <= head_chars + tail_chars:
        return output
    head = output[:head_chars]
    tail = output[-tail_chars:]
    parts = [head, f"...[{n - head_chars - tail_chars} chars omitted]..."]
    hits = top_k_chunks(
        output,
        query,
        k=top_k,
        chunk_chars=chunk_chars,
        exclude_before=head_chars,
        exclude_after=n - tail_chars,
    )
    if hits:
        parts.append(f"[{len(hits)} task-relevant excerpt(s)]")
        for off, text, _score in hits:
            parts.append(f"[@char {off}]\n{text.rstrip()}")
        parts.append("...")
    parts.append(tail)
    return "\n".join(parts)
