"""Pure-Python diversity metrics over a set of model completions.

Five metrics implemented (NIB2-42 / EV-8):

- ``diversity_entropy``  — Shannon entropy of token unigrams (natural log),
  reported in nats. Higher = more token variety.
- ``diversity_distinct2`` — distinct bigrams / total bigrams. Higher = more
  diverse surface phrasings.
- ``diversity_self_bleu``  — mean BLEU-4 of each completion against the
  others. **Lower = more diverse.**
- ``diversity_ttr`` — type-token ratio (unique unigrams / total). Higher
  = more diverse.
- ``diversity_semantic_embedding_agreement`` — mean pairwise cosine across
  an embedder's outputs. **Lower = more diverse.** Accepts an optional
  embedder (any object with ``encode(list_of_texts) -> np.ndarray``);
  returns ``math.nan`` with a log if no embedder is available (anti-gaming
  surface-level distinct-2 per the DD4 deep-dive L205).

All functions accept ``list[str]`` and require ≥2 completions. A single-
completion list returns 0.0 for entropy/distinct-2/TTR and ``math.nan``
for self-BLEU / semantic agreement (undefined).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Protocol

log = logging.getLogger(__name__)


class _Embedder(Protocol):
    def encode(self, texts: list[str]) -> Any: ...


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def _bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


def entropy(completions: list[str]) -> float:
    """Shannon entropy over unigram distribution across all completions (nats)."""
    tokens = [tok for c in completions for tok in _tokenize(c)]
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def distinct_n(completions: list[str], n: int = 2) -> float:
    """distinct-n = |unique n-grams| / |total n-grams|. n=2 is the EV-8 signal."""
    all_ngrams: list[tuple[str, ...]] = []
    for c in completions:
        tokens = _tokenize(c)
        if len(tokens) < n:
            continue
        all_ngrams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def _bleu4(hyp_tokens: list[str], refs_tokens: list[list[str]]) -> float:
    """Brevity-penalty BLEU-4 (cumulative geometric mean of 1..4 gram precisions).

    Returns 0.0 if the hypothesis has <4 tokens (bleu-4 undefined).
    """
    if len(hyp_tokens) < 4 or not refs_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, 5):
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        if not hyp_ngrams:
            return 0.0
        max_ref = Counter()
        for ref in refs_tokens:
            ref_ngrams = Counter(
                tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)
            )
            for ng, count in ref_ngrams.items():
                max_ref[ng] = max(max_ref[ng], count)
        clipped = sum(min(c, max_ref.get(ng, 0)) for ng, c in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        if clipped == 0:
            # Smoothing: add epsilon so log doesn't blow up; still a near-zero score.
            precisions.append(1e-9 / total)
        else:
            precisions.append(clipped / total)

    # Geometric mean.
    log_mean = sum(math.log(p) for p in precisions) / 4
    bleu = math.exp(log_mean)

    # Brevity penalty against closest-length reference.
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


def self_bleu(completions: list[str]) -> float:
    """Mean BLEU-4 of each completion against the others. Lower = more diverse."""
    if len(completions) < 2:
        return math.nan
    tokenized = [_tokenize(c) for c in completions]
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [t for j, t in enumerate(tokenized) if j != i]
        if not refs or not hyp:
            continue
        scores.append(_bleu4(hyp, refs))
    if not scores:
        return math.nan
    return sum(scores) / len(scores)


def type_token_ratio(completions: list[str]) -> float:
    tokens = [tok for c in completions for tok in _tokenize(c)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def semantic_embedding_agreement(
    completions: list[str],
    embedder: _Embedder | None = None,
) -> float:
    """Mean pairwise cosine across ``embedder.encode(completions)``.

    ``embedder`` is any object with ``encode(list[str]) -> array``. When
    absent, returns ``math.nan`` with a warning — SafetyGate must treat
    NaN as "signal unavailable" rather than "zero agreement".
    """
    if len(completions) < 2:
        return math.nan
    if embedder is None:
        log.debug("semantic_embedding_agreement: no embedder supplied; returning NaN")
        return math.nan
    try:
        import numpy as np
    except ImportError:
        log.warning("semantic_embedding_agreement requires numpy; returning NaN")
        return math.nan

    vectors = embedder.encode(completions)
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(completions):
        log.warning("semantic_embedding_agreement: unexpected embedder shape %s",
                    getattr(arr, "shape", None))
        return math.nan
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = arr / norms
    sim = unit @ unit.T
    n = sim.shape[0]
    # Upper triangle (exclude diagonal).
    mask = np.triu_indices(n, k=1)
    pairs = sim[mask]
    return float(pairs.mean()) if pairs.size else math.nan


def compute_all(
    completions: list[str],
    embedder: _Embedder | None = None,
) -> dict[str, float]:
    """Compute the full 5-metric bundle. Used by SafetyGate."""
    return {
        "diversity_entropy": entropy(completions),
        "diversity_distinct2": distinct_n(completions, n=2),
        "diversity_self_bleu": self_bleu(completions),
        "diversity_ttr": type_token_ratio(completions),
        "diversity_semantic_embedding_agreement": semantic_embedding_agreement(
            completions, embedder=embedder
        ),
    }
