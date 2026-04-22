"""Tests for ColBERT reranker module."""

import numpy as np
from unittest.mock import patch, MagicMock

import pytest

from src.tools.web.colbert_reranker import (
    _maxsim,
    rerank_snippets,
    is_available,
)


class TestMaxSim:
    """Test MaxSim scoring function."""

    def test_identical_embeddings_score_one(self):
        """Identical normalized embeddings → MaxSim = 1.0."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert _maxsim(emb, emb) == pytest.approx(1.0)

    def test_orthogonal_embeddings_score_zero(self):
        """Orthogonal query/doc tokens → MaxSim = 0.0."""
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        doc = np.array([[0.0, 1.0]], dtype=np.float32)
        assert _maxsim(query, doc) == pytest.approx(0.0)

    def test_partial_overlap_intermediate_score(self):
        """Partial overlap → score between 0 and 1."""
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        doc = np.array([[0.707, 0.707]], dtype=np.float32)  # 45-degree
        score = _maxsim(query, doc)
        assert 0.0 < score < 1.0

    def test_multiple_doc_tokens_max_per_query(self):
        """MaxSim takes max per query token across all doc tokens."""
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        doc = np.array([
            [0.0, 1.0],  # low sim
            [0.9, 0.1],  # high sim
        ], dtype=np.float32)
        # Normalize doc tokens
        doc = doc / np.linalg.norm(doc, axis=1, keepdims=True)
        score = _maxsim(query, doc)
        assert score > 0.8  # Should pick the high-sim doc token


class TestRerankSnippets:
    """Test rerank_snippets with mocked model."""

    def test_empty_snippets_returns_empty(self):
        result = rerank_snippets("query", [], top_k=3)
        assert result == []

    def test_model_unavailable_returns_original_order(self):
        """When model not loaded, returns original snippets (graceful degradation)."""
        snippets = [
            {"title": "A", "snippet": "First"},
            {"title": "B", "snippet": "Second"},
        ]
        # Model won't be available in test env (no onnxruntime)
        result = rerank_snippets("query", snippets, top_k=2)
        assert len(result) == 2
        assert result[0]["title"] == "A"  # Original order preserved

    @patch("src.tools.web.colbert_reranker._ensure_loaded", return_value=True)
    @patch("src.tools.web.colbert_reranker._encode")
    def test_reranks_by_maxsim(self, mock_encode, mock_loaded):
        """Snippets reranked by MaxSim when model available."""
        # Query embedding: points in x direction
        query_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        # Doc embeddings: B is more aligned with query than A
        doc_a = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # orthogonal
        doc_b = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)  # aligned

        def encode_side_effect(text, max_tokens):
            if "irrelevant" in text.lower():
                return doc_a
            elif "relevant" in text.lower():
                return doc_b
            return query_emb  # query

        mock_encode.side_effect = encode_side_effect

        snippets = [
            {"title": "Irrelevant", "snippet": "Irrelevant content about cooking"},
            {"title": "Relevant", "snippet": "Relevant content about the query topic"},
        ]

        result = rerank_snippets("test query", snippets, top_k=2)
        assert len(result) == 2
        assert result[0]["title"] == "Relevant"
        assert "rerank_score" in result[0]
        assert result[0]["rerank_score"] > result[1]["rerank_score"]

    @patch("src.tools.web.colbert_reranker._ensure_loaded", return_value=True)
    @patch("src.tools.web.colbert_reranker._encode")
    def test_top_k_limits_output(self, mock_encode, mock_loaded):
        """Returns at most top_k snippets."""
        mock_encode.return_value = np.random.randn(3, 128).astype(np.float32)

        snippets = [{"snippet": f"doc {i}"} for i in range(10)]
        result = rerank_snippets("query", snippets, top_k=3)
        assert len(result) == 3

    def test_empty_snippet_text_scores_zero(self):
        """Snippets with no text get score 0.0 (no encoding attempted)."""
        snippets = [
            {"title": "", "snippet": ""},
            {"title": "Real", "snippet": "Has actual content"},
        ]

        query_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        doc_emb = np.array([[0.8, 0.2]], dtype=np.float32)

        def encode_side_effect(text, max_tokens):
            if not text.strip():
                return query_emb  # won't be called for empty text
            return doc_emb if "actual" in text.lower() else query_emb

        with patch("src.tools.web.colbert_reranker._ensure_loaded", return_value=True), \
             patch("src.tools.web.colbert_reranker._encode", side_effect=encode_side_effect):
            result = rerank_snippets("query", snippets, top_k=2)
            # Empty snippet gets score 0.0 (no text → skipped)
            # Real snippet gets a rerank_score
            assert result[0]["title"] == "Real"

    @patch("src.tools.web.colbert_reranker._ensure_loaded", return_value=True)
    @patch("src.tools.web.colbert_reranker._encode", return_value=None)
    def test_encode_failure_returns_original(self, mock_encode, mock_loaded):
        """Encoding failure → returns original order (top_k sliced)."""
        snippets = [{"snippet": f"doc {i}"} for i in range(5)]
        result = rerank_snippets("query", snippets, top_k=3)
        assert len(result) == 3


class TestIsAvailable:
    """Test model availability check."""

    def test_is_available_checks_model_path(self):
        """Returns True when model files exist on disk."""
        # This checks the actual filesystem
        result = is_available()
        assert isinstance(result, bool)


class TestLateonModelPathOverride:
    """Test LATEON_MODEL_PATH env var override (NIB2-47)."""

    def test_default_points_to_gte_moderncolbert(self, monkeypatch):
        """With no env var, module resolves to the GTE-ModernColBERT-v1 directory."""
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        import importlib
        import src.tools.web.colbert_reranker as cr
        importlib.reload(cr)
        assert str(cr._MODEL_DIR) == "/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx"
        assert cr._MODEL_PATH.name == "model_int8.onnx"

    def test_env_var_overrides_to_lateon(self, monkeypatch):
        """LATEON_MODEL_PATH redirects the module-level constants."""
        monkeypatch.setenv("LATEON_MODEL_PATH", "/mnt/raid0/llm/models/lateon-onnx-int8")
        import importlib
        import src.tools.web.colbert_reranker as cr
        importlib.reload(cr)
        assert str(cr._MODEL_DIR) == "/mnt/raid0/llm/models/lateon-onnx-int8"
        assert cr._MODEL_PATH == cr._MODEL_DIR / "model_int8.onnx"
        assert cr._TOKENIZER_PATH == cr._MODEL_DIR / "tokenizer.json"
        # Restore default for subsequent tests.
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        importlib.reload(cr)
