"""Shared ColBERT encoder primitives (ONNX Runtime + MaxSim).

Exposes corpus-agnostic encode/maxsim/ensure_loaded primitives so multiple
consumers (web_research reranker, internal KB-RAG) reuse one model load
and one tokenizer.

Default model: GTE-ModernColBERT-v1 ONNX INT8 (128-dim per-token, ~144 MB).
Override via `LATEON_MODEL_PATH` env var to point at LightOn LateOn (same
ModernBERT backbone, +2.55 pp BEIR per intake-430).

Public API (corpus-agnostic, max-token configurable per call):
    is_available() -> bool
    ensure_loaded() -> bool
    encode(text: str, max_tokens: int) -> np.ndarray | None
    maxsim(query_emb, doc_emb) -> float

ONNX session and tokenizer are module-level singletons, lazy-loaded on first
call. ONNX inference is thread-safe for prediction.

Per handoffs/active/internal-kb-rag.md K1.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Model path resolution: LATEON_MODEL_PATH overrides to the LateOn drop-in.
DEFAULT_MODEL_DIR = Path("/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx")
_MODEL_DIR = Path(os.environ.get("LATEON_MODEL_PATH") or DEFAULT_MODEL_DIR)
_MODEL_PATH = _MODEL_DIR / "model_int8.onnx"
_TOKENIZER_PATH = _MODEL_DIR / "tokenizer.json"

# Module-level singletons (lazy-loaded).
_session = None
_tokenizer = None


def is_available() -> bool:
    """Return True iff model files exist on disk. Does not load."""
    return _MODEL_PATH.exists() and _TOKENIZER_PATH.exists()


def ensure_loaded() -> bool:
    """Lazily load ONNX session + tokenizer. Returns True on success.

    Subsequent calls are no-ops when already loaded. Returns False if
    dependencies are missing or model files cannot be opened.
    """
    global _session, _tokenizer

    if _session is not None and _tokenizer is not None:
        return True

    if not is_available():
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
            "ColBERT encoder loaded: %s (%.0fms)",
            _MODEL_PATH.name,
            elapsed_ms,
        )
        return True
    except ImportError as e:
        logger.warning("ColBERT encoder dependencies missing: %s", e)
        return False
    except Exception as e:  # noqa: BLE001 — defensive; caller checks return.
        logger.error("ColBERT encoder load failed: %s", e)
        return False


def encode(text: str, max_tokens: int) -> np.ndarray | None:
    """Encode text into per-token L2-normalized ColBERT embeddings.

    Args:
        text: Input text.
        max_tokens: Max tokens; tokenizer truncates beyond.

    Returns:
        Array shape (n_real_tokens, 128) or None on failure.
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

        embeddings = outputs[0][0]  # (max_tokens, hidden_dim)
        mask = attention_mask[0]  # (max_tokens,)
        token_embeddings = embeddings[mask == 1]

        # L2 normalize.
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        token_embeddings = token_embeddings / norms
        return token_embeddings
    except Exception as e:  # noqa: BLE001
        logger.debug("ColBERT encode failed: %s", e)
        return None


def maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """ColBERT MaxSim score: avg over query tokens of max cosine to any doc token.

    Both inputs must be L2-normalized along the last axis (encode() does this).
    """
    sim_matrix = query_emb @ doc_emb.T  # (n_q, n_d)
    return float(sim_matrix.max(axis=1).mean())
