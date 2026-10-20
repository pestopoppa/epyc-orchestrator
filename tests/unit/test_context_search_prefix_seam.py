"""REPL-EMB-2.2 follow-on M12: the model-keyed query/document prefix seam of ``context_search``.

Offline and inference-free. Acceptance:
  * BGE is BYTE-IDENTICAL to the pre-seam code: same query strings, same clip order (prefix, then
    clip), documents unprefixed;
  * EmbeddingGemma 2 (EG2) gets ``task: search result | query: `` (default), ``task: code
    retrieval | query: `` (explicit ``code_intent=True`` only) and ``title: none | text: `` for
    documents, with the clip applied to the raw text BEFORE the prefix;
  * ``ensure_dense`` embeds ``prefix + chunk[:800]`` for EG2 and the bare chunk for BGE;
  * the frozen benchmark harness (``repl_emb_arms``) never trips the new markers: loaded through
    ``repl_emb_arms.load_context_search()`` its ids yield unchanged strings, and its adapters
    still prefix exactly once, with the same strings the production seam produces.

Run: .venv/bin/python -m pytest tests/unit/test_context_search_prefix_seam.py -q
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
import repl_emb_arms as arms  # noqa: E402
import repl_emb_eval as ev  # noqa: E402

from src.repl_environment import context_search as prod  # noqa: E402

harness_cs = arms.load_context_search()  # the same file, loaded by path the way the harness does

BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EG2_Q = "task: search result | query: "
EG2_CQ = "task: code retrieval | query: "
EG2_D = "title: none | text: "

BGE_IDS = [
    "bge-large-en-v1.5",
    "bge-base-en-v1.5",
    "bge-small-en-v1.5",
    "bge-large-en-v1.5-f16.gguf",
    "BAAI/BGE-Large-EN-v1.5#0123456789ab",
]
# what a production pool reports (Path(model_path).name) plus the explicit token
EG2_IDS = [
    "embeddinggemma-2-BF16.gguf",
    "embeddinggemma-2-Q8_0.gguf",
    "EmbeddingGemma-2",
    "egemma2-test",
]
UNKNOWN_IDS = ["fake-embedder", "scripted", "granite-97m-r2", "", None]


def _legacy_query_text(model_id, query):
    """The pre-seam ``query_text`` (BGE-only table), copied verbatim."""
    table = (
        ("bge-large-en-v1.5", BGE_INSTRUCTION),
        ("bge-base-en-v1.5", BGE_INSTRUCTION),
        ("bge-small-en-v1.5", BGE_INSTRUCTION),
    )
    low = (model_id or "").lower()
    for marker, prefix in table:
        if marker in low:
            return prefix + query
    return query


class RecordingEmbedder:
    """Records every text list it is asked to embed; deterministic unit vectors."""

    def __init__(self, model_id: str, dim: int = 8) -> None:
        self.model_id = model_id
        self.dim = dim
        self.calls: list[dict] = []

    def embed(self, texts, *, timeout_s, admission_wait_s):
        self.calls.append({"texts": list(texts), "admission_wait_s": admission_wait_s})
        rows = []
        for t in texts:
            rng = np.random.default_rng(sum(map(ord, t[:64])) + len(t))
            v = rng.normal(size=self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return prod.DenseResult(np.stack(rows), self.model_id)

    @property
    def doc_texts(self) -> list[str]:
        return [
            t
            for c in self.calls
            if c["admission_wait_s"] == prod.INDEX_ADMISSION_WAIT_S
            for t in c["texts"]
        ]

    @property
    def query_texts(self) -> list[str]:
        return [
            t
            for c in self.calls
            if c["admission_wait_s"] == prod.QUERY_ADMISSION_WAIT_S
            for t in c["texts"]
        ]


def _sections(line_len: int = 60, lines: int = 40):
    text = "\n".join(f"row {i}: alpha beta {'x' * line_len}" for i in range(lines))
    return [SimpleNamespace(name="s", text=text)]


# ───────────────────────────────────────────────────────────────────────────── BGE unchanged


@pytest.mark.parametrize("mid", BGE_IDS)
def test_bge_query_string_is_byte_identical_to_the_pre_seam_code(mid):
    assert prod.query_text(mid, "find x") == BGE_INSTRUCTION + "find x"
    for q in ("", "find x", "y" * 799, "y" * 800, "y" * 2000):
        assert prod.query_text(mid, q) == _legacy_query_text(mid, q)
        # what search sends: the already-prefixed string, clipped (legacy order kept for BGE)
        assert (
            prod.dense_query_input(mid, q)
            == _legacy_query_text(mid, q)[: prod.DENSE_INPUT_MAX_CHARS]
        )
        # no code form exists for BGE: code_intent changes nothing
        assert prod.query_text(mid, q, code_intent=True) == prod.query_text(mid, q)
        assert prod.dense_query_input(mid, q, code_intent=True) == prod.dense_query_input(mid, q)


def test_bge_long_query_still_loses_its_tail_not_its_prefix():
    s = prod.dense_query_input("bge-large-en-v1.5", "q" * 2000)
    assert len(s) == prod.DENSE_INPUT_MAX_CHARS and s.startswith(BGE_INSTRUCTION)
    assert s == (BGE_INSTRUCTION + "q" * 2000)[:800]


@pytest.mark.parametrize("mid", BGE_IDS + UNKNOWN_IDS)
def test_bge_and_unknown_documents_are_unprefixed(mid):
    for t in ("", "chunk text", "z" * 800):
        assert prod.document_text(mid, t) == t


@pytest.mark.parametrize("mid", UNKNOWN_IDS)
def test_unknown_models_are_untouched_on_both_sides(mid):
    assert prod.query_text(mid, "q") == "q"
    assert prod.query_text(mid, "q", code_intent=True) == "q"
    assert prod.dense_query_input(mid, "q" * 900) == "q" * 800


def test_prefix_table_shape_the_harness_reads_is_preserved():
    # repl_emb_eval.bge_query_prefix() and the speed harness iterate / read this table
    assert all(isinstance(m, str) and isinstance(p, str) for m, p in prod._QUERY_PREFIXES)
    assert ev.bge_query_prefix() == BGE_INSTRUCTION
    assert prod.query_text("bge-large-en-v1.5", "") == BGE_INSTRUCTION
    assert [m for m, _ in prod._QUERY_PREFIXES[:3]] == [
        "bge-large-en-v1.5",
        "bge-base-en-v1.5",
        "bge-small-en-v1.5",
    ]


def test_dense_embedder_protocol_is_unchanged():
    params = list(inspect.signature(prod.DenseEmbedder.embed).parameters)
    assert params == ["self", "texts", "timeout_s", "admission_wait_s"]
    assert "model_id" in prod.DenseEmbedder.__annotations__


# ───────────────────────────────────────────────────────────────────────────── EG2 strings


def test_eg2_strings_are_exactly_the_model_card_prompts():
    assert (prod.EG2_QUERY_PREFIX, prod.EG2_CODE_QUERY_PREFIX, prod.EG2_DOCUMENT_PREFIX) == (
        EG2_Q,
        EG2_CQ,
        EG2_D,
    )
    # and the frozen harness's own spec strings agree with them
    assert (arms.EG2_QUERY_PROMPT, arms.EG2_CODE_QUERY_PROMPT, arms.EG2_DOC_PROMPT) == (
        EG2_Q,
        EG2_CQ,
        EG2_D,
    )


@pytest.mark.parametrize("mid", EG2_IDS)
def test_eg2_query_code_and_document_strings(mid):
    assert prod.query_text(mid, "where is f") == EG2_Q + "where is f"
    assert prod.query_text(mid, "where is f", code_intent=False) == EG2_Q + "where is f"
    assert prod.query_text(mid, "where is f", code_intent=True) == EG2_CQ + "where is f"
    assert prod.document_text(mid, "def f(): pass") == EG2_D + "def f(): pass"
    assert prod.dense_query_input(mid, "where is f") == EG2_Q + "where is f"
    assert prod.dense_query_input(mid, "where is f", code_intent=True) == EG2_CQ + "where is f"


@pytest.mark.parametrize("mid", EG2_IDS)
def test_eg2_clip_is_applied_before_the_prefix(mid):
    n = prod.DENSE_INPUT_MAX_CHARS
    long_q = "q" * 2000
    # the query: 800 raw chars survive, the prefix is whole and on top of the budget
    assert prod.dense_query_input(mid, long_q) == EG2_Q + "q" * n
    assert prod.dense_query_input(mid, long_q, code_intent=True) == EG2_CQ + "q" * n
    assert len(prod.dense_query_input(mid, long_q)) == len(EG2_Q) + n
    # the document: clip, then prefix (the seam never clips; the caller clips first)
    assert prod.document_text(mid, "d" * 2000) == EG2_D + "d" * 2000
    assert prod.document_text(mid, ("d" * 2000)[:n]) == EG2_D + "d" * n
    # BGE, for contrast, clips the prefixed string
    assert len(prod.dense_query_input("bge-large-en-v1.5", long_q)) == n


# ───────────────────────────────────────────────────────────────────────────── ensure_dense


def _long_chunk_index():
    """chunks up to 1200 chars, one line of 1100: the 800-char clip must engage."""
    sections = [SimpleNamespace(name="s", text="head line\n" + "w" * 1100 + "\ntail line\n")]
    idx = prod.ContextSearchIndex(sections, chunk_max_chars=1200)
    assert max(len(c.text) for c in idx.chunks) > prod.DENSE_INPUT_MAX_CHARS
    return idx


def test_ensure_dense_embeds_prefix_plus_clipped_chunk_for_eg2():
    idx = _long_chunk_index()
    emb = RecordingEmbedder("embeddinggemma-2-BF16.gguf")
    assert idx.ensure_dense(emb) is None
    assert emb.doc_texts == [EG2_D + c.text[: prod.DENSE_INPUT_MAX_CHARS] for c in idx.chunks]
    # the prefix did not eat the budget: the chunk part is a full 800 chars where it was clipped
    assert any(len(t) == len(EG2_D) + prod.DENSE_INPUT_MAX_CHARS for t in emb.doc_texts)
    assert all(t.startswith(EG2_D) and not t.startswith(EG2_D + EG2_D) for t in emb.doc_texts)


@pytest.mark.parametrize("mid", ["bge-large-en-v1.5", "fake-embedder"])
def test_ensure_dense_embeds_bare_clipped_chunks_for_bge_and_unknown(mid):
    idx = _long_chunk_index()
    emb = RecordingEmbedder(mid)
    assert idx.ensure_dense(emb) is None
    assert emb.doc_texts == [c.text[: prod.DENSE_INPUT_MAX_CHARS] for c in idx.chunks]


def test_ensure_dense_batches_keep_the_prefix_on_every_batch():
    sections = _sections(line_len=300, lines=400)  # > DENSE_BATCH_CHUNKS chunks
    idx = prod.ContextSearchIndex(sections)
    assert len(idx.chunks) > prod.DENSE_BATCH_CHUNKS
    emb = RecordingEmbedder("egemma2-test")
    assert idx.ensure_dense(emb) is None
    assert len([c for c in emb.calls if c["admission_wait_s"] == prod.INDEX_ADMISSION_WAIT_S]) >= 2
    assert emb.doc_texts == [EG2_D + c.text[:800] for c in idx.chunks]


# ───────────────────────────────────────────────────────────────────────────── search


def test_search_sends_the_search_result_query_by_default_and_code_form_on_request():
    pytest.importorskip("src.trace.navigation")
    idx = prod.ContextSearchIndex(_sections())
    emb = RecordingEmbedder("embeddinggemma-2-Q8_0.gguf")
    hits, meta = idx.search("alpha beta", 3, section=None, embedder=emb)
    assert meta["mode"] == prod.MODE_HYBRID and hits
    assert emb.query_texts == [EG2_Q + "alpha beta"]
    hits, meta = idx.search("alpha beta", 3, section=None, embedder=emb, code_intent=True)
    assert meta["mode"] == prod.MODE_HYBRID
    assert emb.query_texts == [EG2_Q + "alpha beta", EG2_CQ + "alpha beta"]
    # the index was built once, documents prefixed
    assert all(t.startswith(EG2_D) for t in emb.doc_texts)


def test_search_with_a_long_query_keeps_the_eg2_prefix_whole():
    pytest.importorskip("src.trace.navigation")
    idx = prod.ContextSearchIndex(_sections())
    emb = RecordingEmbedder("embeddinggemma-2-BF16.gguf")
    idx.search("alpha " * 400, 3, section=None, embedder=emb)
    (sent,) = emb.query_texts
    assert sent == EG2_Q + ("alpha " * 400)[:800]


def test_search_is_unchanged_for_bge_even_with_code_intent():
    pytest.importorskip("src.trace.navigation")
    idx = prod.ContextSearchIndex(_sections())
    emb = RecordingEmbedder("bge-large-en-v1.5-f16.gguf")
    q = "alpha beta " * 100
    idx.search(q, 3, section=None, embedder=emb)
    idx.search(q, 3, section=None, embedder=emb, code_intent=True)
    legacy = _legacy_query_text("bge-large-en-v1.5-f16.gguf", q)[:800]
    assert emb.query_texts == [legacy, legacy]
    assert emb.doc_texts == [c.text[:800] for c in idx.chunks]


# ───────────────────────────────────────────────────────────────────────────── frozen harness


def _all_harness_specs():
    specs = list(arms.arm_specs("st", "ll", "m2v"))
    specs += list(ev.eval_specs("llama-test", "m2v-test").values())
    return specs


def test_the_markers_cannot_match_a_harness_arm_id():
    assert "eg2" not in prod._EG2_MARKERS  # the harness arm-id stem is deliberately not a marker
    assert all(m.startswith(("embeddinggemma-2", "egemma2")) for m in prod._EG2_MARKERS)
    ids = {s.model_id() for s in _all_harness_specs()}
    assert any(i.startswith("eg2-") for i in ids)  # the ids do contain `eg2` ...
    for i in ids:
        # ... and still no id the harness can produce contains an EG2 marker
        assert not any(m in i.lower() for m in prod._EG2_MARKERS), i


def test_harness_ids_yield_unchanged_strings_through_the_path_loaded_module():
    cs = arms.load_context_search()  # exactly what the frozen harness runs
    bge_id = ev.eval_specs("llama-test", "m2v-test")["bge-large"].model_id()
    n = cs.DENSE_INPUT_MAX_CHARS
    for spec in _all_harness_specs():
        mid = spec.model_id()
        if mid == bge_id:
            assert cs.query_text(mid, "q") == BGE_INSTRUCTION + "q"
            assert cs.dense_query_input(mid, "q" * 900) == (BGE_INSTRUCTION + "q" * 900)[:n]
        else:
            assert cs.query_text(mid, "q") == "q", mid
            assert cs.query_text(mid, "q", code_intent=True) == "q", mid
            assert cs.dense_query_input(mid, "q" * 900) == "q" * n, mid
        assert cs.document_text(mid, "d") == "d", mid  # documents stay unprefixed for the adapters
    # the loaded copy is the file under test, and the two copies agree on every probe
    assert Path(cs.__file__).resolve() == Path(prod.__file__).resolve()
    for mid in BGE_IDS + EG2_IDS + UNKNOWN_IDS:
        assert cs.query_text(mid, "x") == prod.query_text(mid, "x")
        assert cs.document_text(mid, "x") == prod.document_text(mid, "x")


class _FakeHarnessAdapter(arms._AdapterBase):
    """A harness adapter with a recording ``_encode`` (what the model would have been fed)."""

    def __init__(self, spec):
        super().__init__(spec)
        self.seen: list[list[str]] = []

    def _encode(self, texts):
        self.seen.append(list(texts))
        rng = np.random.default_rng(len(self.seen))
        return rng.normal(size=(len(texts), self.spec.native_dim)).astype("float32")


def test_harness_adapters_still_prefix_exactly_once_and_match_the_production_seam():
    pytest.importorskip("src.trace.navigation")
    cs = arms.load_context_search()
    spec = next(s for s in arms.arm_specs("st", "ll", "m2v") if s.arm_id == "eg2-gguf-q8_0-256")
    assert spec.model_id().startswith("eg2-gguf-q8_0-256#")
    ad = _FakeHarnessAdapter(spec)
    idx = cs.ContextSearchIndex(_sections())
    hits, meta = idx.search("alpha beta " * 100, 3, section=None, embedder=ad)
    assert meta["mode"] == cs.MODE_HYBRID and hits
    docs = [t for batch in ad.seen[:-1] for t in batch]
    (query,) = ad.seen[-1]
    # exactly one document prefix on each chunk, none doubled
    assert docs == [EG2_D + c.text[:800] for c in idx.chunks]
    assert query == EG2_Q + ("alpha beta " * 100)[:800]
    # ... and those are the very strings the production seam produces for a production EG2 id
    prod_id = "embeddinggemma-2-Q8_0.gguf"
    assert docs == [cs.document_text(prod_id, c.text[:800]) for c in idx.chunks]
    assert query == cs.dense_query_input(prod_id, "alpha beta " * 100)
    # code-intent through the adapter's own seam is unchanged too
    with ad.query_intent(True):
        idx.search("find def", 3, section=None, embedder=ad)
    assert ad.seen[-1] == [EG2_CQ + "find def"]
    assert ad.seen[-1] == [cs.dense_query_input(prod_id, "find def", code_intent=True)]


def test_harness_bge_arm_is_unchanged_end_to_end():
    pytest.importorskip("src.trace.navigation")
    cs = arms.load_context_search()
    spec = ev.eval_specs("llama-test", "m2v-test")["bge-large"]
    ad = _FakeHarnessAdapter(spec)
    idx = cs.ContextSearchIndex(_sections())
    q = "alpha beta " * 100
    idx.search(q, 3, section=None, embedder=ad)
    docs = [t for batch in ad.seen[:-1] for t in batch]
    assert docs == [c.text[:800] for c in idx.chunks]
    assert ad.seen[-1] == [(BGE_INSTRUCTION + q)[:800]]
