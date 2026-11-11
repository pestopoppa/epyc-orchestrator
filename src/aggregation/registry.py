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
                  matching text, not the head of a multi-thousand-character stored chunk. PERSISTED next to the
                  store (``<store>.indexes/``), keyed by (corpus, content hash, builder version): a fresh process
                  LOADS it (manifest ``origin='loaded'``) instead of rebuilding; a content-hash change, a builder
                  version change, a changed engine (tokenizer / chunker / BM25) or a corrupt file all rebuild.
    l3_facts_det  L3, ``facts_det.extract_corpus_facts`` written to ``facts``.
    l2_dense      L2, reserved: Phase 0 here builds no embeddings. Raises.
    l4_facts_llm  L4, built lazily per field by ``facts_llm``; never via ``build``.
"""

from __future__ import annotations

import bisect
import functools
import gzip
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.aggregation import require_enabled
from src.aggregation.store import CorpusStore
from src.repl_environment.context_search import (
    BM25_B,
    BM25_K1,
    CHUNK_MAX_CHARS,
    QUERY_MAX_CHARS,
    SEARCH_MAX_K,
    BM25Index,
    Chunk,
    ContextSearchIndex,
    tokenize,
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


#: On-disk builder version. Bump it when the persisted payload layout or the meaning of a builder changes; a file
#: written under another version is invalid and the index is rebuilt (never read).
BUILDER_VERSIONS = {"l1_passage_bm25": "l1_passage_bm25.v1"}
INDEX_FILE_SCHEMA = "agg_index_file.v1"
INDEX_DIR_SUFFIX = ".indexes"

#: How an index handle came to be in this process.
ORIGIN_BUILT = "built"
ORIGIN_LOADED = "loaded"


class IndexNotAvailable(RuntimeError):
    pass


class _IndexFileInvalid(Exception):
    """A persisted index file that must not be used (the reason is recorded, the index is rebuilt)."""


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
    origin: str = ORIGIN_BUILT  # "built" in this process, or "loaded" from the persisted file
    seconds: float = 0.0  # wall time of that build / load

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


# ------------------------------------------------------------------ persisted passage index
# The persisted state is exactly what the build spends its time on: the per-passage term frequencies (the output of
# ``tokenize``) and the passage spans (the output of ``chunk_sections``). The passage text is NOT stored: it is rebuilt
# from the store's chunk table on every load and pinned by its sha256, so a file can never be paired with other text.
# idf / length / average are derived from the term frequencies by the same formula ``BM25Index`` uses.

_PROBE_TEXT = "".join(
    f"Line {i}: Total_Current Assets {i * 137 % 997:,} and net_income (1,{i:03d}) cash-equivalents {i % 7}\n"
    + ("Funding. We make contributions to pension trusts; Note %d to the statements.\n" % i if i % 3 == 0 else "")
    for i in range(60)
)
_PROBE_QUERIES = ("Total Current Assets", "net_income cash", "pension trusts contributions", "no_such_token_zzz")


def _export_state(index: ContextSearchIndex) -> tuple[list[list[int]], list[dict[str, int]]]:
    return ([[c.start, c.end, c.line] for c in index.chunks],
            [dict(tf) for tf in index.bm25._tf])  # noqa: SLF001 — guarded by ``_engine_fingerprint``


def _restore_index(corpus_id: str, text: str, spans: list[list[int]],
                   tfs: list[dict[str, int]]) -> ContextSearchIndex:
    """A ``ContextSearchIndex`` from persisted state, no tokenizing and no chunking.

    ``ContextSearchIndex([])`` supplies every dense-side default (so a new attribute of the real class is never
    missed); the lexical side is then set from the state."""
    idx = ContextSearchIndex([])
    idx.chunks = [Chunk(corpus_id, int(s), int(e), int(ln), text[int(s):int(e)]) for s, e, ln in spans]
    bm = BM25Index([])
    bm._tf = [Counter({str(t): int(n) for t, n in tf.items()}) for tf in tfs]  # noqa: SLF001
    bm._len = [sum(tf.values()) for tf in bm._tf]  # noqa: SLF001
    bm._avg = (sum(bm._len) / len(bm._len)) if bm._len else 0.0  # noqa: SLF001
    df: Counter[str] = Counter()
    for tf in bm._tf:  # noqa: SLF001
        df.update(tf.keys())
    n = len(bm._tf)  # noqa: SLF001
    bm._idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}  # noqa: SLF001
    idx.bm25 = bm
    idx._by_section = {corpus_id: list(range(len(idx.chunks)))} if idx.chunks else {}  # noqa: SLF001
    return idx


@functools.lru_cache(maxsize=1)
def _engine_fingerprint() -> str | None:
    """Hash of what the persisted state depends on (tokenizer, chunker, BM25 constants and ranking) on a fixed probe,
    or None when a restored index does not reproduce a live one exactly (persistence is then off: build every time).

    A file written under another fingerprint is invalid, so a change to ``tokenize`` / ``chunk_sections`` /
    ``BM25Index`` invalidates every file instead of silently serving a stale index."""
    live = ContextSearchIndex([_Section("probe", _PROBE_TEXT)])
    spans, tfs = _export_state(live)
    restored = _restore_index("probe", _PROBE_TEXT, spans, tfs)
    ranks = []
    for q in _PROBE_QUERIES:
        a, b = live.bm25.scores(q), restored.bm25.scores(q)
        if a != b or [(c.start, c.end, c.line, c.text) for c in live.chunks] != \
                [(c.start, c.end, c.line, c.text) for c in restored.chunks]:
            return None
        ranks.append([[i, repr(sc)] for i, sc in a])
    blob = json.dumps({"tokens": tokenize(_PROBE_TEXT[:600]), "spans": spans, "ranks": ranks,
                       "chunk_max_chars": CHUNK_MAX_CHARS, "k1": BM25_K1, "b": BM25_B,
                       "sep": CORPUS_TEXT_SEP}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _slug(corpus_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", corpus_id)[:48] + "-" + _sha256_bytes(corpus_id.encode())[:10]


def _write_atomic(path: Path, data: bytes) -> None:
    """Write-then-rename in the target directory: a reader sees the old file or the complete new one, never a part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    try:  # make the rename durable; best-effort (not every filesystem allows a directory fd)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


class IndexRegistry:
    """Builds and persists indexes over a ``CorpusStore``.

    ``persist`` (default: on for a file-backed store, off for ``:memory:``) writes the ``l1_passage_bm25`` index to
    ``<store path>.indexes/`` and loads it in a later process. Counters: ``built`` / ``loaded`` count handles this
    registry produced (a handle served again from memory counts as neither); ``events`` has one record per
    produced handle, ``invalid`` one per persisted file that was refused (and then rebuilt)."""

    def __init__(self, store: CorpusStore, *, persist: bool | None = None) -> None:
        self.store = store
        self._live: dict[tuple[str, str, str, str], Any] = {}
        file_backed = store.path not in (":memory:", "")
        self.persist = file_backed if persist is None else bool(persist and file_backed)
        self.built = 0
        self.loaded = 0
        self.events: list[dict[str, Any]] = []
        self.invalid: list[dict[str, Any]] = []
        self.persist_errors: list[dict[str, Any]] = []

    def key(self, corpus_id: str, builder: str, model_id: str = "") -> tuple[str, str, str, str]:
        return (corpus_id, self.store.corpus_content_hash(corpus_id), builder, model_id)

    # -- persisted passage index ----------------------------------------------------------------
    def index_dir(self) -> Path | None:
        if not self.persist:
            return None
        p = Path(self.store.path)
        return p.with_name(p.name + INDEX_DIR_SUFFIX)

    def index_path(self, corpus_id: str, builder: str, content_hash: str) -> Path | None:
        d = self.index_dir()
        if d is None or builder not in BUILDER_VERSIONS:
            return None
        return d / f"{BUILDER_VERSIONS[builder]}.{_slug(corpus_id)}.{content_hash[:16]}.json.gz"

    def _header(self, corpus_id: str, builder: str, content_hash: str, text: str, n_chunks: int,
                payload: bytes, fp: str) -> dict[str, Any]:
        return {"schema": INDEX_FILE_SCHEMA, "builder": builder, "builder_version": BUILDER_VERSIONS[builder],
                "corpus_id": corpus_id, "content_hash": content_hash, "engine_fp": fp,
                "text_sha256": _sha256_bytes(text.encode()), "text_len": len(text), "n_chunks": n_chunks,
                "payload_sha256": _sha256_bytes(payload), "payload_bytes": len(payload), "created_at": time.time()}

    def _load_passage(self, path: Path, corpus_id: str, content_hash: str, text: str, fp: str):
        """The persisted index at ``path``, or ``_IndexFileInvalid`` (reason) / ``FileNotFoundError`` (none)."""
        try:
            raw = gzip.decompress(path.read_bytes())
        except FileNotFoundError:
            raise
        except (OSError, EOFError, ValueError) as e:
            raise _IndexFileInvalid(f"unreadable: {type(e).__name__}") from e
        head_raw, sep, payload = raw.partition(b"\n")
        try:
            if not sep:
                raise ValueError("no header")
            h = json.loads(head_raw)
        except ValueError as e:
            raise _IndexFileInvalid("bad_header") from e
        want = {"schema": INDEX_FILE_SCHEMA, "builder": "l1_passage_bm25",
                "builder_version": BUILDER_VERSIONS["l1_passage_bm25"], "corpus_id": corpus_id,
                "content_hash": content_hash, "engine_fp": fp, "text_sha256": _sha256_bytes(text.encode()),
                "text_len": len(text)}
        for k, v in want.items():
            if h.get(k) != v:
                raise _IndexFileInvalid(f"{k}_mismatch")
        if len(payload) != h.get("payload_bytes") or _sha256_bytes(payload) != h.get("payload_sha256"):
            raise _IndexFileInvalid("payload_integrity")
        try:
            body = json.loads(payload)
            spans, tfs = body["spans"], body["tf"]
            if not (len(spans) == len(tfs) == h.get("n_chunks")):
                raise ValueError("n_chunks")
            prev = 0
            for s, e, ln in spans:
                if not (prev <= s < e <= len(text)) or ln < 1:
                    raise ValueError("span")
                prev = e
            return _restore_index(corpus_id, text, spans, tfs)
        except (ValueError, KeyError, TypeError) as e:
            raise _IndexFileInvalid(f"payload_invalid: {e}") from e

    def _save_passage(self, path: Path, corpus_id: str, content_hash: str, text: str, fp: str,
                      index: ContextSearchIndex) -> None:
        spans, tfs = _export_state(index)
        payload = json.dumps({"spans": spans, "tf": tfs}, sort_keys=True, separators=(",", ":")).encode()
        head = json.dumps(self._header(corpus_id, "l1_passage_bm25", content_hash, text, len(spans), payload, fp),
                          sort_keys=True, separators=(",", ":")).encode()
        _write_atomic(path, gzip.compress(head + b"\n" + payload, mtime=0))
        for old in path.parent.glob(f"{BUILDER_VERSIONS['l1_passage_bm25']}.{_slug(corpus_id)}.*.json.gz"):
            if old != path:  # a previous content hash of this corpus: stale for good
                try:
                    old.unlink()
                except OSError:
                    pass

    def _passage(self, corpus_id: str, content_hash: str) -> tuple[PassageHandle, int]:
        builder = "l1_passage_bm25"
        t0 = time.perf_counter()
        chunks = self.store.chunks(corpus_id)
        starts, pos = [], 0
        for c in chunks:
            starts.append(pos)
            pos += len(c["text"] or "") + len(CORPUS_TEXT_SEP)
        text = CORPUS_TEXT_SEP.join(c["text"] or "" for c in chunks)
        ids = [c["chunk_id"] for c in chunks]
        fp = _engine_fingerprint() if self.persist else None
        path = self.index_path(corpus_id, builder, content_hash) if fp else None
        if path is not None:
            try:
                index = self._load_passage(path, corpus_id, content_hash, text, fp)
                return PassageHandle(index, text, ids, starts, ORIGIN_LOADED,
                                     time.perf_counter() - t0), len(index.chunks)
            except FileNotFoundError:
                pass
            except _IndexFileInvalid as e:
                self.invalid.append({"corpus_id": corpus_id, "path": str(path), "reason": str(e)})
        index = ContextSearchIndex([_Section(corpus_id, text)])
        if path is not None:
            try:
                self._save_passage(path, corpus_id, content_hash, text, fp, index)
            except OSError as e:  # the index is good; only its persistence failed
                self.persist_errors.append({"corpus_id": corpus_id, "path": str(path), "error": repr(e)})
        return PassageHandle(index, text, ids, starts, ORIGIN_BUILT, time.perf_counter() - t0), len(index.chunks)

    def get_or_build(self, corpus_id: str, builder: str, model_id: str = "") -> Any:
        require_enabled()
        key = self.key(corpus_id, builder, model_id)
        if key in self._live:
            return self._live[key]
        origin, t0 = ORIGIN_BUILT, time.perf_counter()
        if builder == "l1_bm25":
            chunks = self.store.chunks(corpus_id)
            handle: Any = BM25Handle(BM25Index([c["text"] for c in chunks]),
                                     [c["chunk_id"] for c in chunks])
            n = len(chunks)
        elif builder == "l1_passage_bm25":
            handle, n = self._passage(corpus_id, key[1])
            origin = handle.origin
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
        if origin == ORIGIN_LOADED:
            self.loaded += 1
        else:
            self.built += 1
        self.events.append({"corpus_id": corpus_id, "builder": builder, "content_hash": key[1],
                            "origin": origin, "seconds": round(time.perf_counter() - t0, 6), "n_items": n})
        self.store.record_index(*key, layer=LAYER_OF[builder], n_items=n, origin=origin)
        self._live[key] = handle
        return handle
