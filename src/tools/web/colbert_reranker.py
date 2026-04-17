"""ColBERT snippet reranker using ONNX Runtime.

Provides semantic reranking of search snippets via late-interaction
MaxSim scoring. Uses GTE-ModernColBERT-v1 ONNX (128-dim per-token
embeddings, INT8 quantized, 144MB).

Model loaded lazily on first call, cached as module-level singleton.
ONNX inference session is thread-safe for prediction.

Usage:
    from src.tools.web.colbert_reranker import rerank_snippets

    ranked = rerank_snippets(
        query="What causes aurora borealis?",
        snippets=[
            {"title": "Aurora", "snippet": "Northern lights are caused by..."},
            {"title": "Cooking", "snippet": "Best pasta recipes..."},
        ],
        top_k=3,
    )
    # ranked[0] = highest relevance snippet

Feature-gated: ORCHESTRATOR_WEB_RESEARCH_RERANK=1
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default model path (GTE-ModernColBERT-v1 INT8 ONNX, already on disk)
_MODEL_DIR = Path("/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx")
_MODEL_PATH = _MODEL_DIR / "model_int8.onnx"
_TOKENIZER_PATH = _MODEL_DIR / "tokenizer.json"

# Module-level singleton (lazy-loaded)
_session = None
_tokenizer = None

# Encoding parameters
_MAX_QUERY_TOKENS = 48
_MAX_DOC_TOKENS = 64


def _ensure_loaded() -> bool:
    """Lazily load ONNX session and tokenizer on first call.

    Returns:
        True if model is ready for inference.
    """
    global _session, _tokenizer

    if _session is not None and _tokenizer is not None:
        return True

    if not _MODEL_PATH.exists():
        logger.warning("ColBERT ONNX model not found at %s", _MODEL_PATH)
        return False

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        start = time.perf_counter()

        _session = ort.InferenceSession(
            str(_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        _tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "ColBERT reranker loaded: %s (%.0fms)",
            _MODEL_PATH.name, elapsed_ms,
        )
        return True

    except ImportError as e:
        logger.warning("ColBERT reranker dependencies missing: %s", e)
        return False
    except Exception as e:
        logger.error("ColBERT reranker load failed: %s", e)
        return False


def _encode(text: str, max_tokens: int) -> np.ndarray | None:
    """Encode text into per-token ColBERT embeddings.

    Args:
        text: Input text to encode.
        max_tokens: Maximum token length.

    Returns:
        Array of shape (n_tokens, 128) or None on failure.
    """
    if _session is None or _tokenizer is None:
        return None

    try:
        _tokenizer.enable_truncation(max_length=max_tokens)
        _tokenizer.enable_padding(length=max_tokens)
        encoded = _tokenizer.encode(text)

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        outputs = _session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )

        # Output shape: (1, max_tokens, hidden_dim)
        # We need per-token embeddings masked by attention
        embeddings = outputs[0][0]  # (max_tokens, hidden_dim)
        mask = attention_mask[0]  # (max_tokens,)

        # Only keep real token embeddings (where attention_mask == 1)
        token_embeddings = embeddings[mask == 1]

        # L2 normalize per-token embeddings
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        token_embeddings = token_embeddings / norms

        return token_embeddings

    except Exception as e:
        logger.debug("ColBERT encode failed: %s", e)
        return None


def _maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Compute MaxSim score between query and document embeddings.

    MaxSim: for each query token, find the maximum cosine similarity
    to any document token, then average across query tokens.

    Args:
        query_emb: Query token embeddings (n_q, dim).
        doc_emb: Document token embeddings (n_d, dim).

    Returns:
        MaxSim score in [0, 1].
    """
    # Similarity matrix: (n_q, n_d)
    sim_matrix = query_emb @ doc_emb.T

    # Max similarity per query token, then average
    max_per_query = sim_matrix.max(axis=1)
    return float(max_per_query.mean())


def rerank_snippets(
    query: str,
    snippets: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rerank search snippets by ColBERT MaxSim relevance.

    Each snippet dict should have at least 'snippet' or 'title' keys.
    Returns snippets sorted by relevance, with 'rerank_score' added.

    If the model is not loaded or reranking fails, returns the original
    snippets unchanged (graceful degradation).

    Args:
        query: Search query.
        snippets: List of snippet dicts from web search.
        top_k: Number of top snippets to return.

    Returns:
        Top-k snippets sorted by relevance, with 'rerank_score' field.
    """
    if not snippets:
        return []

    if not _ensure_loaded():
        logger.debug("ColBERT reranker not available, returning original order")
        return snippets[:top_k]

    start = time.perf_counter()

    # Encode query
    query_emb = _encode(query, _MAX_QUERY_TOKENS)
    if query_emb is None:
        return snippets[:top_k]

    # Score each snippet
    scored = []
    for snippet_dict in snippets:
        # Combine title + snippet text for encoding
        text = ""
        if "title" in snippet_dict:
            text += snippet_dict["title"] + ". "
        if "snippet" in snippet_dict:
            text += snippet_dict["snippet"]
        elif "body" in snippet_dict:
            text += snippet_dict["body"]

        # Skip snippets with no meaningful text (only punctuation/whitespace)
        cleaned = text.strip().strip(".")
        if not cleaned.strip():
            scored.append((snippet_dict, 0.0))
            continue

        doc_emb = _encode(text, _MAX_DOC_TOKENS)
        if doc_emb is None:
            scored.append((snippet_dict, 0.0))
            continue

        score = _maxsim(query_emb, doc_emb)
        scored.append((snippet_dict, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "ColBERT rerank: %d snippets in %.0fms, scores=[%s]",
        len(snippets), elapsed_ms,
        ", ".join(f"{s:.3f}" for _, s in scored[:5]),
    )

    # Add scores and return top-k
    result = []
    for snippet_dict, score in scored[:top_k]:
        enriched = dict(snippet_dict)
        enriched["rerank_score"] = round(score, 4)
        result.append(enriched)

    return result


def is_available() -> bool:
    """Check if the reranker model is available on disk."""
    return _MODEL_PATH.exists() and _TOKENIZER_PATH.exists()
