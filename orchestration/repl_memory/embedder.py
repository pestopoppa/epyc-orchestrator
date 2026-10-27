"""
TaskEmbedder: Generate embeddings for TaskIR and related content.

Uses BGE-large-en-v1.5 via llama-server /embedding endpoint for efficient embedding.
Subprocess (llama-embedding) requires explicit opt-in and the production kernel store.
Falls back to hash-based pseudo-embeddings if model is unavailable.

For parallel fan-out to multiple servers, see parallel_embedder.py.

Performance:
- HTTP server: 2-5ms per embedding (40x faster)
- Subprocess: 50-200ms per embedding
- Hash fallback: <1ms but no semantic similarity
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from src.embedding_dims import (
    LEGACY_EMBEDDING_DIM,
    SideRequiredError as SideRequiredError,
    apply_side as _apply_side,
    default_truncate_dim,
    normalise_embedding,
)
from src.inference_lock import inference_lock

from .memory_record import embedding_text_for, join_embedding_segments

# Default model path (BGE-large-en-v1.5 for embeddings)
# BGE-large produces 1024-dim embeddings, purpose-built for similarity search
DEFAULT_MODEL_PATH = Path("/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf")
# Subprocess embeddings are opt-in and may only use the production CPU kernel store.
KERNEL_STORE_CPU = Path("/mnt/raid0/llm/kernels/production/cpu")


def resolve_embedding_binary(store: Path = KERNEL_STORE_CPU) -> Path:
    """Name the store binary without probing it during import or server-only use."""
    return store / "llama-embedding"


DEFAULT_EMBEDDING_BINARY = resolve_embedding_binary()
DEFAULT_SERVER_URL = "http://127.0.0.1:8090"

# Per-field truncation for the non-task embedding conventions (EPD-3-R8). These
# are part of each convention's meaning; the joined text is additionally bounded
# by memory_record.EMBED_TEXT_MAX_CHARS inside join_embedding_segments.
FAILURE_MESSAGE_EMBED_CHARS = 200
EXPLORATION_PREVIEW_EMBED_CHARS = 500
CLASSIFICATION_PROMPT_EMBED_CHARS = 300


def _head(value: Any, n: int) -> Any:
    """First ``n`` chars of a string; ``None``/empty pass through unchanged."""
    if not value:
        return value
    return value[:n]


# ── Degenerate-embedding detection ───────────────────────────────────────────
#
# The hash fallback below (`TaskEmbedder._generate_embedding_fallback`) is not a
# usable embedding by any reading, and measurement is worse than its docstring
# admits. Over 5,000 real task texts:
#
#     all-zero          89.0%   norm overflows float32 -> inf; v/inf == 0
#     contains NaN       2.8%   permanently unretrievable (FAISS scores it -inf)
#     unit-normalised     8.1%   well-formed but semantically meaningless
#
# `np.frombuffer(sha256_bytes, float32)` reinterprets hash bytes as floats with
# exponents up to ~1e38, so `np.linalg.norm` overflows; `norm > 0` is True for
# inf, and the divide silently yields zeros.
#
# `use_fallback` defaults to **True** in both EmbeddingConfig and
# EmbedderPoolConfig, and every live site that builds a TaskEmbedder uses the
# bare constructor. So an embedder outage does not fail a write — it fills the
# episodic store with these. These helpers exist so the episodic write
# chokepoint can refuse them; detection is exact rather than heuristic because
# the fallback is a pure function of the text.


def hash_fallback_embedding(text: str, embedding_dim: int = LEGACY_EMBEDDING_DIM) -> np.ndarray:
    """Reproduce the hash pseudo-embedding for ``text``, bug-for-bug.

    Deliberately mirrors ``TaskEmbedder._generate_embedding_fallback`` including
    its float32 norm overflow — the point is to reproduce what was *actually
    stored*, not what a correct implementation would store.
    """
    hash_bytes = hashlib.sha256(text.encode()).digest()
    repeats = (embedding_dim * 4 // len(hash_bytes)) + 1
    expanded = (hash_bytes * repeats)[: embedding_dim * 4]
    embedding = np.frombuffer(expanded, dtype=np.float32)[:embedding_dim].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    return embedding


def is_hash_fallback_embedding(text: str, vec: np.ndarray) -> bool:
    """True if ``vec`` is exactly the hash pseudo-embedding of ``text``.

    Exact comparison, not cosine: the fallback is deterministic, so an exact
    match is both available and far safer. A cosine test at 0.99 gave a **45%
    false-positive rate** against random unit vectors, because the unnormalised
    (overflowed) reference vector has a huge magnitude and dots high with
    anything.

    Note this only catches the 8.1% well-formed case; the all-zero and NaN
    cases are text-independent and are caught by ``is_degenerate_embedding``.
    """
    if vec is None or not text:
        return False
    vec = np.asarray(vec, dtype=np.float32).ravel()
    if vec.size == 0:
        return False
    ref = hash_fallback_embedding(text, embedding_dim=vec.shape[0])
    with np.errstate(over="ignore", invalid="ignore"):
        return bool(np.allclose(vec, ref, rtol=1e-5, atol=1e-6, equal_nan=True))


def is_degenerate_embedding(vec: np.ndarray) -> str | None:
    """Return a reason string if ``vec`` must never be persisted, else None.

    Text-independent checks, so they hold at any write site regardless of which
    embedder produced the vector:

    * ``non_finite`` — NaN/Inf. FAISS scores the row -inf, so it exists in
      SQLite with a valid ``embedding_idx`` and can never be retrieved.
    * ``all_zero`` — carries no information and, being identical across rows,
      collapses distinct tasks onto one point. This is the 89% case above.
    """
    if vec is None:
        return "missing"
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.size == 0:
        return "empty"
    if not np.all(np.isfinite(v)):
        return "non_finite"
    if not np.any(v):
        return "all_zero"
    return None


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_path: Path = DEFAULT_MODEL_PATH
    embedding_binary: Path = DEFAULT_EMBEDDING_BINARY
    embedding_dim: int = LEGACY_EMBEDDING_DIM  # BGE-large; EG2 flag: the truncate_dim
    threads: int = 8  # Use fewer threads for embedding (fast operation)
    timeout: int = 30  # Seconds
    use_fallback: bool = True  # Fall back to hash-based if model unavailable
    server_url: str = DEFAULT_SERVER_URL  # Embedding server URL
    use_server: bool = True  # Try HTTP server first (40x faster)
    use_parallel: bool = True  # Use parallel embedder client (probe-first to 6 servers)
    allow_subprocess: bool = False  # Model subprocesses require explicit opt-in
    # REPL-EMB-2.9 EG2 seam. None (OFF) = legacy BGE behaviour, byte-identical. When set (e.g.
    # 256, or via EPYC_EMBEDDING_TRUNCATE_DIM) every raw server vector is sliced to its leading
    # columns and L2 re-normalised, `embedding_dim` follows it, and the EG2 document/query
    # prefixes become mandatory (`side=` on the embed_* helpers).
    truncate_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.truncate_dim is None:
            self.truncate_dim = default_truncate_dim()
        if self.truncate_dim and self.embedding_dim == LEGACY_EMBEDDING_DIM:
            self.embedding_dim = int(self.truncate_dim)


class TaskEmbedder:
    """
    Generate embeddings for TaskIR and other orchestration content.

    Embedding strategy (in priority order):
    1. Parallel HTTP servers (probe-first to ports 8090-8095) - 5-15ms with redundancy
    2. Single HTTP server fallback (port 8090) - 2-5ms
    3. Explicitly enabled subprocess (kernel-store llama-embedding) - 50-200ms
    4. Hash-based pseudo-embeddings - fallback

    Fallback:
    - If model unavailable, use hash-based pseudo-embeddings
    - These preserve semantic locality for identical inputs
    - But lose generalization (similar != identical)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model_available = self._check_model()
        self._server_available: Optional[bool] = None  # Lazy check
        self._http_client = None  # Lazy httpx client
        self._parallel_client = None  # Lazy parallel client

    def _check_model(self) -> bool:
        """Check if embedding model and binary are available."""
        return self.config.model_path.exists() and self.config.embedding_binary.exists()

    def _check_server(self) -> bool:
        """Check if embedding server is available (lazy, cached)."""
        if self._server_available is not None:
            return self._server_available

        if not self.config.use_server:
            self._server_available = False
            return False

        try:
            import httpx

            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.config.server_url}/health")
                self._server_available = resp.status_code == 200
        except Exception as e:
            logger.debug("Embedding server health check failed: %s", e)
            self._server_available = False

        if self._server_available:
            logger.debug("Embedding server available at %s", self.config.server_url)
        else:
            logger.debug("Embedding server not available")

        return self._server_available

    def _get_http_client(self):
        """Get or create httpx client (lazy initialization)."""
        if self._http_client is None:
            import httpx

            self._http_client = httpx.Client(timeout=10.0)
        return self._http_client

    def _generate_embedding_http(self, text: str) -> np.ndarray:
        """Generate embedding using HTTP server (2-5ms)."""
        client = self._get_http_client()

        # llama-server /embedding endpoint format
        resp = client.post(
            f"{self.config.server_url}/embedding",
            json={"content": text},
        )
        resp.raise_for_status()
        data = resp.json()

        # llama-server returns:
        # - {"embedding": [[...]], ...}
        # - {"data": [{"embedding": [...]}], ...}
        # - or list payload: [{"embedding": [[...]], ...}]
        if isinstance(data, list) and data:
            data = data[0]

        if "embedding" in data:
            # Direct embedding format
            embedding_data = data["embedding"]
            if isinstance(embedding_data[0], list):
                embedding = np.array(embedding_data[0], dtype=np.float32)
            else:
                embedding = np.array(embedding_data, dtype=np.float32)
        elif "data" in data and len(data["data"]) > 0:
            # OpenAI-compatible format
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
        else:
            raise ValueError(
                "Unexpected embedding response format: "
                f"{list(data.keys()) if hasattr(data, 'keys') else type(data)}"
            )

        # Normalize to unit vector (REPL-EMB-2.9: shared helper; truncates first when EG2 is on)
        return normalise_embedding(embedding, self.config.truncate_dim)

    def _serialize_task_ir(self, task_ir: Dict[str, Any]) -> str:
        """Serialize a TaskIR into the ONE canonical embedding convention.

        Delegates to ``memory_record.embedding_text_for`` — this is the live
        query/write path for every ``embed_task_ir`` caller, and it must produce
        byte-identical text to what the store was published with, or similarity
        search matches queries against a different convention.

        It used to re-spell the convention here and had drifted three ways
        (EPD-3-R2): key PRESENCE instead of truthiness (so a TaskIR carrying
        ``priority: None`` emitted the literal ``priority:None``), no
        ``.strip()`` and no length cap, plus ``constraints:``/``input_types:``
        segments no other write site emits. Any field beyond the task triple is
        telemetry as far as the embedding is concerned and is stored, never
        embedded (see memory_record's THE ONE INVARIANT).
        """
        return embedding_text_for(
            objective=task_ir.get("objective"),
            task_type=task_ir.get("task_type"),
            priority=task_ir.get("priority"),
        )

    def _serialize_failure_context(self, failure_context: Dict[str, Any]) -> str:
        """Serialize failure context for escalation memory.

        Assembled by ``join_embedding_segments`` (EPD-3-R8), so a key that is
        present but ``None`` is dropped instead of embedding ``error:None``.
        """
        return join_embedding_segments(
            (
                ("error", failure_context.get("error_type")),
                ("gate", failure_context.get("gate_name")),
                ("tier", failure_context.get("agent_tier")),
                # Truncate message to avoid embedding noise
                (
                    "message",
                    _head(failure_context.get("failure_message"), FAILURE_MESSAGE_EMBED_CHARS),
                ),
            )
        )

    def _serialize_exploration(self, query: str, context_preview: str) -> str:
        """Serialize exploration context for REPL memory (via the shared builder)."""
        # Truncate to focus on structure; both segments are mandatory.
        return join_embedding_segments(
            (
                ("query", query if query is not None else ""),
                ("preview", _head(context_preview, EXPLORATION_PREVIEW_EMBED_CHARS) or ""),
            )
        )

    def _serialize_classification_prompt(self, prompt: str, classification_type: str) -> str:
        """Serialize a prompt for classification lookup (via the shared builder).

        Args:
            prompt: User prompt to classify.
            classification_type: Type of classification (e.g., "routing", "summarization").

        Returns:
            Serialized string for embedding.
        """
        # Truncate long prompts (first 300 chars capture intent)
        return join_embedding_segments(
            (
                ("classify", classification_type),
                ("prompt", _head(prompt, CLASSIFICATION_PROMPT_EMBED_CHARS) or ""),
            )
        )

    def embed_classification_prompt(
        self,
        prompt: str,
        classification_type: str = "routing",
        side: Optional[str] = None,
    ) -> np.ndarray:
        """Generate embedding for a classification prompt.

        Used by ClassificationRetriever to find similar classification exemplars.

        Args:
            prompt: User prompt to classify.
            classification_type: Type of classification (routing, summarization, etc.).

        Returns:
            Embedding vector (1024-dim for BGE-large).
        """
        text = self._serialize_classification_prompt(prompt, classification_type)
        return self._generate_embedding(text, side)

    def _subprocess_binary(self) -> Path:
        """Validate explicit subprocess admission before any fallback can hide failure."""
        if not self.config.allow_subprocess:
            raise RuntimeError(
                "Embedding subprocess disabled; set allow_subprocess=True explicitly"
            )
        expected = resolve_embedding_binary(KERNEL_STORE_CPU)
        binary = expected.resolve()
        if Path(self.config.embedding_binary).resolve() != binary:
            raise RuntimeError(f"Embedding binary must come from the CPU kernel store: {expected}")
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(
                f"CPU kernel-store embedding binary missing or not executable: {expected}"
            )
        if not self.config.model_path.is_file():
            raise RuntimeError(f"Embedding subprocess model missing: {self.config.model_path}")
        return binary

    def _generate_embedding_llama(self, text: str) -> np.ndarray:
        """Generate embedding with the explicitly admitted CPU kernel-store binary."""
        binary = self._subprocess_binary()
        cmd = [
            str(binary),
            "-m",
            str(self.config.model_path),
            "-p",
            text,
            "-t",
            str(self.config.threads),
            "--embd-output-format",
            "json",
        ]
        if self.config.truncate_dim:
            # EG2 is mean-pooled; the legacy BGE command (CLS from the GGUF header) is untouched.
            cmd.extend(["--pooling", "mean"])
        # The binary and its own ggml libraries must come from the same resolved store.
        run_kwargs: Dict[str, Any] = {"env": {**os.environ, "LD_LIBRARY_PATH": str(binary.parent)}}
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                **run_kwargs,
            )

            if result.returncode != 0:
                raise RuntimeError(f"llama-embedding failed: {result.stderr}")

            # Parse JSON output
            output = json.loads(result.stdout)

            # llama-embedding returns embeddings in different formats:
            # Format 1: {'object': 'list', 'data': [{'embedding': [...]}]}
            # Format 2: [{'embedding': [...]}]
            if isinstance(output, dict) and "data" in output:
                # Format 1: OpenAI-compatible format
                data = output["data"]
                if data and len(data) > 0:
                    embedding = np.array(data[0]["embedding"], dtype=np.float32)
                else:
                    raise ValueError(f"Empty data in output: {output}")
            elif isinstance(output, list) and len(output) > 0:
                # Format 2: Simple list format
                embedding = np.array(output[0]["embedding"], dtype=np.float32)
            else:
                raise ValueError(f"Unexpected output format: {output}")

            # Normalize to unit vector (REPL-EMB-2.9: shared helper; truncates first when EG2 is on)
            return normalise_embedding(embedding, self.config.truncate_dim)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"llama-embedding timed out after {self.config.timeout}s")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse embedding output: {e}")

    def _generate_embedding_fallback(self, text: str) -> np.ndarray:
        """
        Generate hash-based pseudo-embedding as fallback.

        Uses SHA-256 to create deterministic embeddings.
        Preserves identity (same input = same embedding).
        Does NOT preserve similarity (similar != close).
        """
        # Create deterministic hash
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Expand hash to embedding dimension
        # Repeat hash bytes to fill embedding
        repeats = (self.config.embedding_dim * 4 // len(hash_bytes)) + 1
        expanded = (hash_bytes * repeats)[: self.config.embedding_dim * 4]

        # Convert to float32 array
        embedding = np.frombuffer(expanded, dtype=np.float32)[: self.config.embedding_dim]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_task_ir(self, task_ir: Dict[str, Any], side: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for a TaskIR.

        Args:
            task_ir: TaskIR dictionary

        Returns:
            Embedding vector (normalized)
        """
        text = self._serialize_task_ir(task_ir)
        return self._generate_embedding(text, side)

    def embed_failure_context(
        self, failure_context: Dict[str, Any], side: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for failure context (escalation memory).

        Args:
            failure_context: Failure context dictionary

        Returns:
            Embedding vector (normalized)
        """
        text = self._serialize_failure_context(failure_context)
        return self._generate_embedding(text, side)

    def embed_exploration(
        self, query: str, context_preview: str, side: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for exploration context (REPL memory).

        Args:
            query: User query
            context_preview: Preview of context being explored

        Returns:
            Embedding vector (normalized)
        """
        text = self._serialize_exploration(query, context_preview)
        return self._generate_embedding(text, side)

    def embed_text(self, text: str, side: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for arbitrary text.

        Args:
            text: Input text

        Returns:
            Embedding vector (normalized)
        """
        return self._generate_embedding(text, side)

    def _get_parallel_client(self):
        """Get or create parallel embedder client (lazy initialization)."""
        if self._parallel_client is None:
            from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

            if self.config.truncate_dim:
                from orchestration.repl_memory.parallel_embedder import EmbedderPoolConfig

                self._parallel_client = ParallelEmbedderClient(
                    EmbedderPoolConfig(truncate_dim=self.config.truncate_dim)
                )
            else:
                self._parallel_client = ParallelEmbedderClient()
        return self._parallel_client

    def _generate_embedding(self, text: str, side: Optional[str] = None) -> np.ndarray:
        """Generate embedding using available method (parallel > single HTTP > subprocess > hash).

        ``side`` ('document' | 'query' | None) only matters with the EG2 seam on, where it selects
        the prompt prefix and is mandatory; with the seam off it is validated and ignored.
        """
        raw_text = text
        text = _apply_side(text, side, bool(self.config.truncate_dim))
        with inference_lock("embedder", shared=True):
            # Try parallel embedder first (probe-first to 6 servers)
            if self.config.use_parallel and self.config.use_server:
                try:
                    client = self._get_parallel_client()
                    if side is None:
                        return client.embed_sync(raw_text)
                    return client.embed_sync(raw_text, side=side)
                except Exception as e:
                    logger.warning(
                        "Parallel embedder failed (%s), falling back to single server", e
                    )

            # Try single HTTP server (2-5ms)
            if self._check_server():
                try:
                    return self._generate_embedding_http(text)
                except Exception as e:
                    logger.warning("Embedding server failed (%s)", e)
                    self._server_available = False  # Reset to retry later

            # An explicit subprocess request must fail loudly on invalid store/model state.
            if self.config.allow_subprocess:
                self._subprocess_binary()
                try:
                    return self._generate_embedding_llama(text)
                except Exception as e:
                    if self.config.use_fallback:
                        logger.warning("llama-embedding failed (%s), using hash fallback", e)
                        return self._generate_embedding_fallback(text)
                    raise

            # Hash fallback
            if self.config.use_fallback:
                return self._generate_embedding_fallback(text)
            else:
                raise RuntimeError(
                    "Embedding servers unavailable; subprocess and hash fallback disabled"
                )

    def embed_batch(self, texts: List[str], side: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Array of embedding vectors (N x embedding_dim)
        """
        embeddings = []
        for text in texts:
            embeddings.append(self._generate_embedding(text, side))
        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.config.embedding_dim

    @property
    def is_model_available(self) -> bool:
        """Check if the neural embedding model is available."""
        return self._model_available

    @property
    def is_server_available(self) -> bool:
        """Check if the embedding server is available."""
        return self._check_server()

    def close(self) -> None:
        """Close HTTP client connections."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        if self._parallel_client is not None:
            self._parallel_client.close_sync()
            self._parallel_client = None
