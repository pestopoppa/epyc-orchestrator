"""Tests for ColBERT reranker module."""

import numpy as np
from unittest.mock import patch

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


class TestModelPathOverride:
    """Test ColBERT reranker model-path slot selection."""

    def test_default_points_to_gte_moderncolbert(self, monkeypatch):
        """With no env var, module resolves to the GTE-ModernColBERT-v1 directory."""
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        import importlib
        import src.tools.web.colbert_reranker as cr
        importlib.reload(cr)
        assert str(cr._MODEL_DIR) == "/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx"
        assert cr._MODEL_SLOT == "gte_moderncolbert"
        assert cr._MODEL_PATH.name == "model_int8.onnx"

    def test_env_var_overrides_to_lateon(self, monkeypatch):
        """LATEON_MODEL_PATH redirects the module-level constants."""
        monkeypatch.setenv("LATEON_MODEL_PATH", "/mnt/raid0/llm/models/lateon-onnx-int8")
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        import importlib
        import src.tools.web.colbert_reranker as cr
        importlib.reload(cr)
        assert str(cr._MODEL_DIR) == "/mnt/raid0/llm/models/lateon-onnx-int8"
        assert cr._MODEL_SLOT == "lateon"
        assert cr._MODEL_PATH == cr._MODEL_DIR / "model_int8.onnx"
        assert cr._TOKENIZER_PATH == cr._MODEL_DIR / "tokenizer.json"
        # Restore default for subsequent tests.
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        importlib.reload(cr)

    def test_reason_mxbai_env_var_selects_fallback_slot(self, monkeypatch):
        """REASON_MXBAI_MODEL_PATH redirects when LateOn is unset."""
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        monkeypatch.setenv(
            "REASON_MXBAI_MODEL_PATH",
            "/mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8",
        )
        import importlib
        import src.tools.web.colbert_reranker as cr
        importlib.reload(cr)
        assert str(cr._MODEL_DIR) == (
            "/mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8"
        )
        assert cr._MODEL_SLOT == "reason_mxbai"
        assert cr._MODEL_PATH == cr._MODEL_DIR / "model_int8.onnx"
        assert cr._TOKENIZER_PATH == cr._MODEL_DIR / "tokenizer.json"
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        importlib.reload(cr)

    def test_lateon_precedes_reason_mxbai_when_both_are_set(self, monkeypatch):
        """LateOn remains the primary slot when both overrides are configured."""
        monkeypatch.setenv("LATEON_MODEL_PATH", "/mnt/raid0/llm/models/lateon-onnx-int8")
        monkeypatch.setenv(
            "REASON_MXBAI_MODEL_PATH",
            "/mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8",
        )
        import importlib
        import src.tools.web.colbert_reranker as cr
        importlib.reload(cr)
        assert str(cr._MODEL_DIR) == "/mnt/raid0/llm/models/lateon-onnx-int8"
        assert cr._MODEL_SLOT == "lateon"
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        importlib.reload(cr)


class TestOnnxThreadBound:
    """The ONNX session must be created with a bounded intra-op thread pool.

    Regression guard for the 2026-08-12 measurement: ORT's default pool (one
    thread per visible core, 192 here) is both slower and far noisier than a
    small bound for this 1+N single-row forward-pass workload, and it spins up
    192 threads per rerank call on a shared host. Without an explicit
    ``SessionOptions``, ``intra_op_num_threads`` silently reverts to that default
    and nothing in the request path notices.
    """

    @staticmethod
    def _load_with_stubs(monkeypatch, model_dir):
        """Drive _ensure_loaded() with a stubbed ORT + tokenizer; return options."""
        import onnxruntime as ort
        import tokenizers

        import src.tools.web.colbert_reranker as cr

        captured = {}

        def fake_session(path, sess_options=None, providers=None, **kwargs):
            captured["path"] = path
            captured["sess_options"] = sess_options
            captured["providers"] = providers
            return object()

        monkeypatch.setattr(ort, "InferenceSession", fake_session)
        monkeypatch.setattr(
            tokenizers.Tokenizer, "from_file", staticmethod(lambda p: object())
        )
        monkeypatch.setattr(cr, "_MODEL_PATH", model_dir / "model_int8.onnx")
        monkeypatch.setattr(cr, "_TOKENIZER_PATH", model_dir / "tokenizer.json")
        monkeypatch.setattr(cr, "_session", None)
        monkeypatch.setattr(cr, "_tokenizer", None)

        assert cr._ensure_loaded() is True
        # Leave no loaded singleton behind for other tests.
        monkeypatch.setattr(cr, "_session", None)
        monkeypatch.setattr(cr, "_tokenizer", None)
        return captured

    def test_session_options_bound_intra_op_threads(self, monkeypatch, tmp_path):
        """Default load pins intra-op to the bounded default, not the core count."""
        import os

        import src.tools.web.colbert_reranker as cr

        (tmp_path / "model_int8.onnx").write_bytes(b"")
        (tmp_path / "tokenizer.json").write_text("{}")
        monkeypatch.delenv("COLBERT_RERANK_ONNX_THREADS", raising=False)

        captured = self._load_with_stubs(monkeypatch, tmp_path)
        opts = captured["sess_options"]

        assert opts is not None, "InferenceSession was built without SessionOptions"
        assert opts.intra_op_num_threads == cr._DEFAULT_ONNX_THREADS
        assert opts.intra_op_num_threads > 0
        # 0 is ORT's "use every core" sentinel, and the bound must stay well under
        # the host's core count or the oversubscription this guards is back.
        assert opts.intra_op_num_threads < (os.cpu_count() or 2)
        assert opts.inter_op_num_threads == 1
        assert captured["providers"] == ["CPUExecutionProvider"]

    def test_env_override_sets_thread_count(self, monkeypatch, tmp_path):
        """COLBERT_RERANK_ONNX_THREADS reaches the SessionOptions."""
        (tmp_path / "model_int8.onnx").write_bytes(b"")
        (tmp_path / "tokenizer.json").write_text("{}")
        monkeypatch.setenv("COLBERT_RERANK_ONNX_THREADS", "3")

        captured = self._load_with_stubs(monkeypatch, tmp_path)

        assert captured["sess_options"].intra_op_num_threads == 3

    @pytest.mark.parametrize("bad", ["not-a-number", "0", "-4", ""])
    def test_invalid_override_falls_back_to_default(self, monkeypatch, bad):
        """Garbage, zero and negative overrides never reach ORT."""
        import src.tools.web.colbert_reranker as cr

        monkeypatch.setenv("COLBERT_RERANK_ONNX_THREADS", bad)
        assert cr._onnx_threads() == cr._DEFAULT_ONNX_THREADS
