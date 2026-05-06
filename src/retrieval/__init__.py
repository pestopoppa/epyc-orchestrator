"""Retrieval primitives shared across web_research and internal KB-RAG.

`colbert_encoder` exposes ONNX-runtime-backed ColBERT encoding + MaxSim
scoring as a corpus-agnostic library. Two consumers use it:

- `src.tools.web.colbert_reranker` (existing, web search snippet reranking)
- `src.retrieval.kb_rag` (this package, internal markdown KB)

Per handoffs/active/internal-kb-rag.md K1.
"""

from src.retrieval.colbert_encoder import (
    DEFAULT_MODEL_DIR,
    encode,
    ensure_loaded,
    is_available,
    maxsim,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "encode",
    "ensure_loaded",
    "is_available",
    "maxsim",
]
