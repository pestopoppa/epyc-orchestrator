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
    encode(text: str, max_tokens: int, *, role: str) -> np.ndarray | None
    maxsim(query_emb, doc_emb) -> float

ColBERT role prefixes (OP-24, 2026-08-12)
-----------------------------------------
Both GTE-ModernColBERT-v1 and the LateOn candidate declare
``query_prefix "[Q] "`` / ``document_prefix "[D] "`` in
``config_sentence_transformers.json``, backed by DEDICATED TRAINED token ids
(50368 / 50369 for the deployed model) that pylate inserts at index 1, right
after ``[CLS]``. Encoding without them is off-distribution: measured on the
reference model with no ONNX in the loop, the missing prefix moves MaxSim by
max |delta| 1.63e-01 and flips top-1 on 37.5% of queries — roughly 25x more
perturbing than the INT8 quantization we already accept (6.60e-03, top-1
agreement 100%).

`role` is therefore REQUIRED and keyword-only. There is deliberately no
default: a silent default is exactly how the prefix-less path survived
undetected, so every call site must state its intent. `ROLE_NONE` is the
explicit legacy escape hatch for stores built before this change — it is a
choice a caller makes visibly, not a fallback.

Prepending the prefix STRING is equivalent to pylate's id insertion because
"[Q] " / "[D] " (trailing space included) are single added tokens in the
tokenizer vocabulary; `ensure_loaded()` verifies that and refuses the prefixed
roles if the loaded model does not carry them.

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

# ONNX Runtime intra-op thread bound.
#
# `encode()` runs ONE single-row forward pass over <=max_tokens padded tokens, so
# the parallel work per call is tiny and ORT's default pool (one thread per visible
# core) oversubscribes badly on this 192-thread host. Measured 2026-08-12 on EPYC
# 9655, model_int8.onnx, batch=1, 20 calls x 3 interleaved rounds, medians in ms:
#
#     2t=20.54  4t=17.34  8t=17.88  16t=18.58  32t=23.92  192t=33.94
#
# Unbounded costs 1.96x. 4 measured nominally best, but 8 is within 3% — inside
# shared-host run-to-run variance — and matches the sibling reranker, which has the
# same single-row shape. Chosen for shape-consistency over a third magic number.
#
# Do NOT copy this value to a BATCHED consumer: cross_encoder.score_pairs() feeds N
# rows in one run() and measures best at 16 (40% faster than 8 at batch=50). The
# optimum is call-shape dependent, so measure before reusing.
_DEFAULT_ONNX_THREADS = 8


def _onnx_threads() -> int:
    """Resolve the ONNX intra-op thread count (env-overridable, positive int)."""
    raw = os.environ.get("COLBERT_ENCODE_ONNX_THREADS")
    if not raw:
        return _DEFAULT_ONNX_THREADS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "COLBERT_ENCODE_ONNX_THREADS=%r is not an integer; using %d",
            raw, _DEFAULT_ONNX_THREADS,
        )
        return _DEFAULT_ONNX_THREADS
    if value <= 0:
        logger.warning(
            "COLBERT_ENCODE_ONNX_THREADS=%d must be positive; using %d",
            value, _DEFAULT_ONNX_THREADS,
        )
        return _DEFAULT_ONNX_THREADS
    return value


# ── ColBERT role prefixes ────────────────────────────────────────────────────
#
# Fallbacks only; ensure_loaded() prefers whatever the model's own
# config_sentence_transformers.json declares, so a model swap cannot silently
# keep the wrong convention.
_FALLBACK_QUERY_PREFIX = "[Q] "
_FALLBACK_DOCUMENT_PREFIX = "[D] "

ROLE_QUERY = "query"
ROLE_DOCUMENT = "document"
ROLE_NONE = "none"
VALID_ROLES = (ROLE_QUERY, ROLE_DOCUMENT, ROLE_NONE)

# Identifier stamped alongside stored vectors so an index can always say which
# convention produced it. Bump when the prefix semantics change.
PREFIX_CONVENTION = "qd-v1"
# Convention of every embedding written before OP-24 (2026-08-12).
LEGACY_CONVENTION = "none"

# Module-level singletons (lazy-loaded).
_session = None
_tokenizer = None
_query_prefix = _FALLBACK_QUERY_PREFIX
_document_prefix = _FALLBACK_DOCUMENT_PREFIX
_prefix_tokens_ok = False


def _load_declared_prefixes(model_dir: Path) -> tuple[str, str]:
    """Read query/document prefixes from the model's sentence-transformers config."""
    cfg_path = model_dir / "config_sentence_transformers.json"
    try:
        import json

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(
            "ColBERT: could not read %s (%s); using fallback prefixes %r / %r",
            cfg_path, e, _FALLBACK_QUERY_PREFIX, _FALLBACK_DOCUMENT_PREFIX,
        )
        return _FALLBACK_QUERY_PREFIX, _FALLBACK_DOCUMENT_PREFIX
    q = cfg.get("query_prefix")
    d = cfg.get("document_prefix")
    if not isinstance(q, str) or not q or not isinstance(d, str) or not d:
        logger.warning(
            "ColBERT: %s declares no usable query/document prefix; using fallbacks",
            cfg_path,
        )
        return _FALLBACK_QUERY_PREFIX, _FALLBACK_DOCUMENT_PREFIX
    return q, d


def prefix_for_role(role: str) -> str:
    """Return the literal prefix string for `role` ("" for ROLE_NONE)."""
    if role == ROLE_QUERY:
        return _query_prefix
    if role == ROLE_DOCUMENT:
        return _document_prefix
    if role == ROLE_NONE:
        return ""
    raise ValueError(f"unknown ColBERT role {role!r}; expected one of {VALID_ROLES}")


def prefix_tokens_available() -> bool:
    """True iff the loaded tokenizer maps both prefixes to single trained tokens."""
    return _prefix_tokens_ok


def is_available() -> bool:
    """Return True iff model files exist on disk. Does not load."""
    return _MODEL_PATH.exists() and _TOKENIZER_PATH.exists()


def ensure_loaded() -> bool:
    """Lazily load ONNX session + tokenizer. Returns True on success.

    Subsequent calls are no-ops when already loaded. Returns False if
    dependencies are missing or model files cannot be opened.
    """
    global _session, _tokenizer, _query_prefix, _document_prefix, _prefix_tokens_ok

    if _session is not None and _tokenizer is not None:
        return True

    if not is_available():
        logger.warning("ColBERT ONNX model not found at %s", _MODEL_PATH)
        return False

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        start = time.perf_counter()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = _onnx_threads()
        sess_options.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            str(_MODEL_PATH),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        _tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))

        _query_prefix, _document_prefix = _load_declared_prefixes(_MODEL_DIR)
        q_id = _tokenizer.token_to_id(_query_prefix)
        d_id = _tokenizer.token_to_id(_document_prefix)
        _prefix_tokens_ok = q_id is not None and d_id is not None
        if not _prefix_tokens_ok:
            logger.error(
                "ColBERT: prefixes %r/%r are NOT single tokens in %s (ids %r/%r) — "
                "prefixed roles will be refused; only ROLE_NONE can be encoded",
                _query_prefix, _document_prefix, _TOKENIZER_PATH, q_id, d_id,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "ColBERT encoder loaded: %s (%.0fms), prefixes %r=%r %r=%r",
            _MODEL_PATH.name,
            elapsed_ms,
            _query_prefix, q_id, _document_prefix, d_id,
        )
        return True
    except ImportError as e:
        logger.warning("ColBERT encoder dependencies missing: %s", e)
        return False
    except Exception as e:  # noqa: BLE001 — defensive; caller checks return.
        logger.error("ColBERT encoder load failed: %s", e)
        return False


def encode(text: str, max_tokens: int, *, role: str) -> np.ndarray | None:
    """Encode text into per-token L2-normalized ColBERT embeddings.

    Args:
        text: Input text.
        max_tokens: Max tokens; tokenizer truncates beyond. The role prefix
            occupies one of them, matching pylate.
        role: REQUIRED. `ROLE_QUERY`, `ROLE_DOCUMENT`, or `ROLE_NONE`.
            Queries and documents MUST use the same convention as the store
            they are compared against — see the module docstring. There is no
            default on purpose.

    Returns:
        Array shape (n_real_tokens, 128) or None on failure.

    Raises:
        ValueError: `role` is not one of `VALID_ROLES`, or a prefixed role was
            requested from a model whose tokenizer lacks the trained prefix
            tokens (encoding it would silently emit garbage sub-word tokens).
    """
    prefix = prefix_for_role(role)  # raises on an unknown role
    if prefix and not _prefix_tokens_ok:
        raise ValueError(
            f"ColBERT role {role!r} needs prefix {prefix!r}, but the loaded "
            f"tokenizer has no such token ({_TOKENIZER_PATH})"
        )

    if _session is None or _tokenizer is None:
        return None

    try:
        _tokenizer.enable_truncation(max_length=max_tokens)
        _tokenizer.enable_padding(length=max_tokens)
        encoded = _tokenizer.encode(prefix + text)

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
