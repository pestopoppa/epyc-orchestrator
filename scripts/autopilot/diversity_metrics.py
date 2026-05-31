"""Diversity metrics for autopilot evaluation trials (EV-8 / NIB2-42).

Thin adapter that delegates to the canonical implementation in
``src.tools.diversity.metrics`` when the orchestrator package root is on
``sys.path`` (normal autopilot runtime), and provides a self-contained
fallback for environments where only ``scripts/autopilot`` is imported.

Public API (mirrors src.tools.diversity.metrics)
------------------------------------------------
distinct_2(texts)                      → float
type_token_ratio(texts)                → float
self_bleu(texts)                       → float
entropy(texts)                         → float
semantic_embedding_agreement(texts, embed_fn=None) → float | None
compute_diversity(texts, embed_fn=None) → dict[str, float]

All functions accept ``list[str]`` and require ≥2 completions. A
single-completion list returns 0.0 for ratio-based metrics and ``None``
(or ``math.nan``) for pair-based metrics (self-BLEU, semantic agreement)
that are undefined on a singleton.

Semantic embedding agreement is **inference-gated**: pass an embedder
object with ``.encode(list[str]) -> np.ndarray`` to activate it; omit
(or pass ``None``) and it returns ``None`` so SafetyGate treats the
signal as unavailable for this trial.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Try the canonical implementation first (works when ORCH_ROOT is on path).
# ---------------------------------------------------------------------------
try:
    from src.tools.diversity.metrics import (  # type: ignore[import]
        entropy as _entropy,
        distinct_n as _distinct_n,
        self_bleu as _self_bleu,
        type_token_ratio as _ttr,
        semantic_embedding_agreement as _sea,
        compute_all as _compute_all,
    )
    _USING_CANONICAL = True
except ImportError:
    _USING_CANONICAL = False


# ---------------------------------------------------------------------------
# Self-contained fallback (identical logic, no deps beyond stdlib + numpy).
# ---------------------------------------------------------------------------

if not _USING_CANONICAL:
    import logging
    from collections import Counter

    log = logging.getLogger(__name__)

    def _tokenize(text: str) -> list[str]:
        return [t for t in text.lower().split() if t]

    def _entropy(completions: list[str]) -> float:  # type: ignore[misc]
        tokens = [tok for c in completions for tok in _tokenize(c)]
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        total = len(tokens)
        return -sum((c / total) * math.log(c / total) for c in counts.values())

    def _distinct_n(completions: list[str], n: int = 2) -> float:  # type: ignore[misc]
        all_ngrams: list[tuple[str, ...]] = []
        for c in completions:
            tokens = _tokenize(c)
            if len(tokens) < n:
                continue
            all_ngrams.extend(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)

    def _bleu4(hyp_tokens: list[str], refs_tokens: list[list[str]]) -> float:
        if len(hyp_tokens) < 4 or not refs_tokens:
            return 0.0
        precisions: list[float] = []
        for n in range(1, 5):
            hyp_ngrams = Counter(
                tuple(hyp_tokens[i: i + n]) for i in range(len(hyp_tokens) - n + 1)
            )
            if not hyp_ngrams:
                return 0.0
            max_ref: Counter = Counter()
            for ref in refs_tokens:
                ref_ngrams = Counter(
                    tuple(ref[i: i + n]) for i in range(len(ref) - n + 1)
                )
                for ng, count in ref_ngrams.items():
                    max_ref[ng] = max(max_ref[ng], count)
            clipped = sum(min(c, max_ref.get(ng, 0)) for ng, c in hyp_ngrams.items())
            total = sum(hyp_ngrams.values())
            precisions.append((clipped / total) if clipped > 0 else 1e-9 / total)
        log_mean = sum(math.log(p) for p in precisions) / 4
        bleu = math.exp(log_mean)
        hyp_len = len(hyp_tokens)
        closest_ref = min(refs_tokens, key=lambda r: (abs(len(r) - hyp_len), len(r)))
        ref_len = len(closest_ref)
        if hyp_len > ref_len:
            bp = 1.0
        elif hyp_len == 0:
            bp = 0.0
        else:
            bp = math.exp(1.0 - ref_len / hyp_len)
        return bleu * bp

    def _self_bleu(completions: list[str]) -> float:  # type: ignore[misc]
        if len(completions) < 2:
            return math.nan
        tokenized = [_tokenize(c) for c in completions]
        scores = []
        for i, hyp in enumerate(tokenized):
            refs = [t for j, t in enumerate(tokenized) if j != i]
            if not refs or not hyp:
                continue
            scores.append(_bleu4(hyp, refs))
        return sum(scores) / len(scores) if scores else math.nan

    def _ttr(completions: list[str]) -> float:  # type: ignore[misc]
        tokens = [tok for c in completions for tok in _tokenize(c)]
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _sea(  # type: ignore[misc]
        completions: list[str],
        embedder: Any | None = None,
    ) -> float:
        if len(completions) < 2:
            return math.nan
        if embedder is None:
            log.debug("semantic_embedding_agreement: no embedder; returning NaN")
            return math.nan
        try:
            import numpy as np
        except ImportError:
            log.warning("semantic_embedding_agreement requires numpy; returning NaN")
            return math.nan
        vectors = embedder.encode(completions)
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(completions):
            log.warning("unexpected embedder shape %s", getattr(arr, "shape", None))
            return math.nan
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit = arr / norms
        sim = unit @ unit.T
        n = sim.shape[0]
        mask = np.triu_indices(n, k=1)
        pairs = sim[mask]
        return float(pairs.mean()) if pairs.size else math.nan

    def _compute_all(  # type: ignore[misc]
        completions: list[str],
        embedder: Any | None = None,
    ) -> dict[str, float]:
        return {
            "diversity_entropy": _entropy(completions),
            "diversity_distinct2": _distinct_n(completions, n=2),
            "diversity_self_bleu": _self_bleu(completions),
            "diversity_ttr": _ttr(completions),
            "diversity_semantic_embedding_agreement": _sea(completions, embedder),
        }


# ---------------------------------------------------------------------------
# Public API — stable names regardless of which backend was loaded.
# ---------------------------------------------------------------------------

def distinct_2(texts: list[str]) -> float:
    """Distinct bigram ratio across all texts (higher = more surface variety)."""
    return _distinct_n(texts, n=2)


def type_token_ratio(texts: list[str]) -> float:
    """Unique tokens / total tokens across all texts (higher = more lexical variety)."""
    return _ttr(texts)


def self_bleu(texts: list[str]) -> float:
    """Mean pairwise BLEU-4 (lower = more diverse generations)."""
    return _self_bleu(texts)


def entropy(texts: list[str]) -> float:
    """Shannon entropy of token unigrams in nats (higher = more variety)."""
    return _entropy(texts)


def semantic_embedding_agreement(
    texts: list[str],
    embed_fn: Any | None = None,
) -> float:
    """Mean pairwise cosine similarity of embeddings (lower = more semantic diversity).

    ``embed_fn`` must expose ``.encode(list[str]) -> np.ndarray``.  When
    ``None``, returns ``math.nan`` — the signal is inference-gated and
    SafetyGate treats NaN as "unavailable" rather than zero agreement.
    """
    return _sea(texts, embedder=embed_fn)


def compute_diversity(
    texts: list[str],
    embed_fn: Any | None = None,
) -> dict[str, float]:
    """Compute all 5 EV-8 diversity metrics in one pass.

    Returns a dict with keys:
      ``diversity_entropy``, ``diversity_distinct2``, ``diversity_self_bleu``,
      ``diversity_ttr``, ``diversity_semantic_embedding_agreement``.

    ``diversity_semantic_embedding_agreement`` is ``math.nan`` when no
    embedder is supplied (inference-gated; never blocks a trial result).
    """
    return _compute_all(texts, embedder=embed_fn)
