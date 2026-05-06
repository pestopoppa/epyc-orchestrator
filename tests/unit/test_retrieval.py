"""Unit tests for src/retrieval/.

Encoder is mocked — onnxruntime is not always available in devcontainers.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import numpy as np
import pytest

from src.retrieval.markdown_chunker import (
    DEFAULT_MAX_CHARS,
    Chunk,
    chunk_file,
    chunk_markdown,
)


# ─── chunker ──────────────────────────────────────────────────────────────────

def test_chunk_empty_returns_empty() -> None:
    assert chunk_markdown("", "/p") == []
    assert chunk_markdown("   \n  \n", "/p") == []


def test_chunk_no_headings_single_chunk() -> None:
    text = "just some prose without any headings"
    chunks = chunk_markdown(text, "/p")
    assert len(chunks) == 1
    assert chunks[0].heading_path == []
    assert chunks[0].text == text


def test_chunk_h1_h2_h3_breadcrumb() -> None:
    text = dedent(
        """
        # Top

        intro

        ## Mid A

        body A

        ### Sub

        sub body

        ## Mid B

        body B
        """
    ).strip()
    chunks = chunk_markdown(text, "/p")
    headings = [c.heading_path for c in chunks]
    assert ["Top"] in headings
    assert ["Top", "Mid A"] in headings
    assert ["Top", "Mid A", "Sub"] in headings
    assert ["Top", "Mid B"] in headings
    # Sub closes when Mid B opens — so Mid B should NOT include Sub.
    mid_b = next(c for c in chunks if c.heading_path == ["Top", "Mid B"])
    assert "sub body" not in mid_b.text


def test_chunk_long_section_splits_at_paragraphs() -> None:
    # 6 paragraphs, each 800 chars; total ~5000 chars -> must split.
    para = "x" * 800
    body = "\n\n".join(para for _ in range(6))
    text = f"# Top\n\n{body}"
    chunks = chunk_markdown(text, "/p", max_chars=2000)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 2200  # max_chars + small slack for boundary
        assert c.heading_path == ["Top"]


def test_chunk_content_hash_stable() -> None:
    text = "# h\n\nbody"
    a = chunk_markdown(text, "/p")[0]
    b = chunk_markdown(text, "/p")[0]
    assert a.content_hash == b.content_hash
    # Different text -> different hash
    c = chunk_markdown("# h\n\nother body", "/p")[0]
    assert a.content_hash != c.content_hash


def test_chunk_file_absent_returns_empty(tmp_path: Path) -> None:
    assert chunk_file(tmp_path / "nope.md") == []


def test_chunk_breadcrumb_property() -> None:
    c = Chunk(file_path="/x", heading_path=["A", "B"], line_range=(1, 2), text="t")
    assert c.heading_breadcrumb == "A > B"
    c2 = Chunk(file_path="/x", heading_path=[], line_range=(1, 2), text="t")
    assert c2.heading_breadcrumb == "(no headings)"


def test_chunk_line_ranges_cover_file(tmp_path: Path) -> None:
    md = tmp_path / "f.md"
    md.write_text("# A\n\nbody1\n\n## B\n\nbody2\n")
    chunks = chunk_file(md)
    # Every line should fall within at least one chunk's range.
    line_count = md.read_text().count("\n") + 1
    covered = set()
    for c in chunks:
        for ln in range(c.line_range[0], c.line_range[1] + 1):
            covered.add(ln)
    # Allow off-by-one slack at the file tail.
    assert len(covered) >= line_count - 2


# ─── kb_rag (mocked encoder) ──────────────────────────────────────────────────

def test_kb_rag_build_with_mock_encoder(tmp_path: Path) -> None:
    from src.retrieval import kb_rag

    # Set up tiny corpus.
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("# Doc A\n\nFirst doc, mentions cats.\n")
    (corpus_root / "b.md").write_text("# Doc B\n\n## Section\n\nSecond doc, dogs and birds.\n")

    cfg = kb_rag.CorpusConfig(
        roots=[str(corpus_root)],
        include_globs=["*.md"],
        exclude_patterns=[],
        max_chunk_chars=4000,
    )
    index_dir = tmp_path / "idx"

    # Mock encoder to return deterministic embeddings keyed by text length.
    def fake_encode(text: str, max_tokens: int):
        # 3 tokens of dim-4: deterministic hash-based vector.
        arr = np.zeros((3, 4), dtype=np.float32)
        arr[0, 0] = (len(text) % 7) * 0.1
        arr[1, 1] = 0.5
        arr[2, 2] = 0.7
        # L2 normalize per token.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return arr / norms

    with patch.object(kb_rag.colbert_encoder, "is_available", return_value=True), \
         patch.object(kb_rag.colbert_encoder, "ensure_loaded", return_value=True), \
         patch.object(kb_rag.colbert_encoder, "encode", side_effect=fake_encode):
        result = kb_rag.build_index(cfg, index_dir=index_dir)

    assert result["ok"] is True
    assert result["files"] == 2
    assert result["chunks_encoded"] >= 2
    assert (index_dir / "catalog.sqlite").exists()
    assert (index_dir / "emb").exists()


def test_kb_rag_build_skips_unchanged_on_rebuild(tmp_path: Path) -> None:
    from src.retrieval import kb_rag

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("# Doc\n\nbody\n")
    cfg = kb_rag.CorpusConfig(roots=[str(corpus_root)], include_globs=["*.md"], exclude_patterns=[])
    index_dir = tmp_path / "idx"

    def fake_encode(text, max_tokens):
        arr = np.eye(3, 4, dtype=np.float32)
        return arr

    with patch.object(kb_rag.colbert_encoder, "is_available", return_value=True), \
         patch.object(kb_rag.colbert_encoder, "ensure_loaded", return_value=True), \
         patch.object(kb_rag.colbert_encoder, "encode", side_effect=fake_encode):
        first = kb_rag.build_index(cfg, index_dir=index_dir)
        second = kb_rag.build_index(cfg, index_dir=index_dir)

    assert first["chunks_encoded"] >= 1
    # Content unchanged -> all chunks skipped on rebuild.
    assert second["chunks_encoded"] == 0
    assert second["chunks_skipped_unchanged"] == first["chunks_encoded"]


def test_kb_rag_query_returns_ranked_results(tmp_path: Path) -> None:
    from src.retrieval import kb_rag

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "cats.md").write_text("# cats\n\nfeline body\n")
    (corpus_root / "dogs.md").write_text("# dogs\n\ncanine body\n")
    cfg = kb_rag.CorpusConfig(roots=[str(corpus_root)], include_globs=["*.md"], exclude_patterns=[])
    index_dir = tmp_path / "idx"

    # Encoder returns vectors that prefer the "cats" file when query mentions feline.
    def fake_encode(text, max_tokens):
        text_l = text.lower()
        v = np.zeros((2, 4), dtype=np.float32)
        if "feline" in text_l or "cats" in text_l:
            v[0, 0] = 1.0
            v[1, 1] = 0.5
        else:
            v[0, 2] = 1.0
            v[1, 3] = 0.5
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-8)

    with patch.object(kb_rag.colbert_encoder, "is_available", return_value=True), \
         patch.object(kb_rag.colbert_encoder, "ensure_loaded", return_value=True), \
         patch.object(kb_rag.colbert_encoder, "encode", side_effect=fake_encode):
        kb_rag.build_index(cfg, index_dir=index_dir)
        results = kb_rag.query("tell me about cats", top_k=2, index_dir=index_dir)

    assert len(results) == 2
    # Top result should be the cats file.
    assert "cats.md" in results[0]["file"]
    assert results[0]["score"] >= results[1]["score"]


def test_kb_rag_update_files_replaces_specific_files(tmp_path: Path) -> None:
    from src.retrieval import kb_rag

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    f = corpus_root / "a.md"
    f.write_text("# v1\n\nbody v1\n")
    cfg = kb_rag.CorpusConfig(roots=[str(corpus_root)], include_globs=["*.md"], exclude_patterns=[])
    index_dir = tmp_path / "idx"

    def fake_encode(text, max_tokens):
        return np.eye(3, 4, dtype=np.float32)

    with patch.object(kb_rag.colbert_encoder, "is_available", return_value=True), \
         patch.object(kb_rag.colbert_encoder, "ensure_loaded", return_value=True), \
         patch.object(kb_rag.colbert_encoder, "encode", side_effect=fake_encode):
        kb_rag.build_index(cfg, index_dir=index_dir)
        # Edit file.
        f.write_text("# v2\n\ncompletely different body\n")
        result = kb_rag.update_files([str(f)], cfg, index_dir=index_dir)

    assert result["ok"] is True
    s = kb_rag.stats(index_dir=index_dir)
    assert s["files"] == 1
    # Old chunks for f are gone, replaced.
    assert s["chunks"] >= 1


def test_kb_rag_query_returns_empty_when_no_index(tmp_path: Path) -> None:
    from src.retrieval import kb_rag

    assert kb_rag.query("anything", index_dir=tmp_path / "no_index") == []


# ─── encoder back-compat ──────────────────────────────────────────────────────

def test_existing_reranker_imports_still_work() -> None:
    """The K1 refactor must not break src.tools.web.colbert_reranker."""
    # Import-only test — no runtime call.
    from src.tools.web.colbert_reranker import is_available, rerank_snippets

    assert callable(is_available)
    assert callable(rerank_snippets)


def test_new_retrieval_exports_match_handoff() -> None:
    """Public API per handoffs/active/internal-kb-rag.md K1."""
    from src.retrieval import encode, ensure_loaded, is_available, maxsim

    assert callable(encode)
    assert callable(ensure_loaded)
    assert callable(is_available)
    assert callable(maxsim)
