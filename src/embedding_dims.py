"""The one place the embedding dimension and the Matryoshka truncation are defined (REPL-EMB-2.9).

Before this module every raw-vector path (``repl_memory/embedder.py`` http and llama-subprocess,
``parallel_embedder.py``, the pooled client) normalised whatever the server returned at its full
size and every store/config carried its own ``1024`` literal. EmbeddingGemma 2 returns 768-d
vectors that the adoption decision serves at 256 (slice the leading columns, then L2
re-normalise), so the truncation must live in ONE helper called from all four paths.

FLAG OFF is the default and is byte-identical to the pre-change code: ``truncate_dim`` is ``None``
unless ``EPYC_EMBEDDING_TRUNCATE_DIM`` (or an explicit argument / policy key) is set, and with it
``None`` :func:`normalise_embedding` performs exactly the old ``v / norm if norm > 0`` step.

The slice+renormalise definition is the one the REPL-EMB-2.2 harness froze,
``scripts/benchmark/repl_emb_arms.py::truncate_renormalise``; it is loaded by path (that module
imports stdlib only) rather than copied, so there is no second definition to drift.

Dimension single source: :data:`LEGACY_EMBEDDING_DIM` (BGE-large, what every store holds today)
and :data:`EG2_SERVED_DIM`. A migrated store directory carries ``embedding_model.json`` whose
``dim`` is authoritative for that directory (:func:`marker_dim`).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

LEGACY_EMBEDDING_DIM = 1024  # BGE-large-en-v1.5: the dimension of every store written before EG2
EG2_NATIVE_DIM = 768  # EmbeddingGemma 2 server reply
EG2_SERVED_DIM = 256  # adopted Matryoshka cut (REPL-EMB-2.2 decision, 2026-10-20)
# Asymmetric prompts of the adopted model (config_sentence_transformers.json of embeddinggemma-2;
# the same strings scripts/benchmark/repl_emb_arms.py and the migration tool use). Documents carry
# the document prompt at write time, lookups the query prompt.
SIDES = ("document", "query")
EG2_SIDE_PREFIXES = {
    "document": "title: none | text: ",
    "query": "task: search result | query: ",
}
TRUNCATE_DIM_ENV = "EPYC_EMBEDDING_TRUNCATE_DIM"
MARKER_NAME = "embedding_model.json"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARMS_PATH = _REPO_ROOT / "scripts" / "benchmark" / "repl_emb_arms.py"
_ARMS_MODULE = "epyc_repl_emb_arms_shared"


def _arms() -> Any:
    mod = sys.modules.get(_ARMS_MODULE)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(_ARMS_MODULE, _ARMS_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - the file ships with the repo
        raise ImportError(f"cannot load {_ARMS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_ARMS_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


def truncate_renormalise(mat: Any, dim: int) -> np.ndarray:
    """Keep the leading ``dim`` columns then L2-normalise (the harness-frozen definition)."""
    return np.asarray(_arms().truncate_renormalise(mat, dim), dtype=np.float32)


def default_truncate_dim() -> int | None:
    """``None`` (flag OFF) unless ``EPYC_EMBEDDING_TRUNCATE_DIM`` holds a positive integer."""
    raw = os.environ.get(TRUNCATE_DIM_ENV, "").strip()
    if not raw or raw == "0":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{TRUNCATE_DIM_ENV}={raw!r} is not an integer") from None
    if value < 0:
        raise ValueError(f"{TRUNCATE_DIM_ENV} must be positive, got {value}")
    return value


def effective_embedding_dim(truncate_dim: int | None = None) -> int:
    """The dimension a consumer should expect: the truncation when on, else the legacy BGE dim."""
    td = truncate_dim if truncate_dim else default_truncate_dim()
    return int(td) if td else LEGACY_EMBEDDING_DIM


def normalise_embedding(embedding: np.ndarray, truncate_dim: int | None = None) -> np.ndarray:
    """The shared post-processing step for a raw server vector.

    ``truncate_dim`` falsy: the legacy step, bug-for-bug (divide by the L2 norm when ``norm > 0``,
    no dimension check). Truthy: slice to the leading ``truncate_dim`` columns then L2
    re-normalise; a reply shorter than ``truncate_dim`` raises ``ValueError``.
    """
    if truncate_dim:
        return truncate_renormalise(embedding, int(truncate_dim))
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


class SideRequiredError(ValueError):
    """EG2 vectors are asymmetric: a stored text is embedded as a ``document`` and a lookup as a
    ``query``. With the EG2 seam on, an embed call that names neither is refused instead of
    silently producing a lookup that cannot match the prefixed documents (REPL-EMB-2.10 B-5)."""


def apply_side(text: str, side: str | None, eg2: bool) -> str:
    """The text a model should see for ``side``. Identity unless the EG2 seam is on; a bad
    ``side`` always raises, and ``side=None`` raises ``SideRequiredError`` when the seam is on."""
    if side is not None and side not in SIDES:
        raise ValueError(f"side must be one of {SIDES} or None, got {side!r}")
    if not eg2:
        return text
    if side is None:
        raise SideRequiredError(
            "embedding seam is on (truncate_dim set): pass side='document' for stored text or "
            "side='query' for a lookup"
        )
    return EG2_SIDE_PREFIXES[side] + text


class StoreDimensionMismatch(ValueError):
    """A store directory's ``embedding_model.json`` pins a dimension the consumer is not
    configured for (REPL-EMB-2.10 B-4). Raised before the store is opened; nothing is touched."""


def check_store_marker(
    store_dir: str | os.PathLike[str], expected_dim: int, *, consumer: str = "store"
) -> None:
    """Refuse a marked store whose ``dim`` differs from the consumer's ``expected_dim``.

    A directory WITHOUT a marker is not refused here (its dimension is whatever the files say and
    the consumer's own checks apply): only a positive, migration-written pin can contradict.
    """
    dim = marker_dim(store_dir)
    if dim is not None and dim != expected_dim:
        raise StoreDimensionMismatch(
            f"{consumer} at {store_dir}: {MARKER_NAME} says dim={dim} (written by "
            f"scripts/migration/reembed_stores_eg2.py) but the consumer is configured for "
            f"dim={expected_dim}. Configure the consumer for dim={dim} together with the EG2 "
            f"embedder (EPYC_EMBEDDING_TRUNCATE_DIM), or roll the store back with "
            f"`reembed_stores_eg2.py rollback`. Nothing was opened or modified."
        )


def marker_dim(store_dir: str | os.PathLike[str]) -> int | None:
    """The ``dim`` recorded in ``<store_dir>/embedding_model.json``; ``None`` for a legacy store."""
    path = Path(store_dir) / MARKER_NAME
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    dim = data.get("dim") if isinstance(data, dict) else None
    return int(dim) if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 else None


__all__ = [
    "EG2_NATIVE_DIM",
    "EG2_SIDE_PREFIXES",
    "EG2_SERVED_DIM",
    "LEGACY_EMBEDDING_DIM",
    "MARKER_NAME",
    "SIDES",
    "SideRequiredError",
    "StoreDimensionMismatch",
    "check_store_marker",
    "apply_side",
    "TRUNCATE_DIM_ENV",
    "default_truncate_dim",
    "effective_embedding_dim",
    "marker_dim",
    "normalise_embedding",
    "truncate_renormalise",
]
