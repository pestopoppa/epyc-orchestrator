"""DeepConf (intake-603, arXiv:2508.15260) — offline confidence-filtered self-consistency.

Proxy-layer reimplementation of the *offline* DeepConf variant: generate N reasoning
traces, score each by a group-confidence signal derived from per-token top-k
log-probabilities, keep the top-η% most-confident traces, then take a
confidence-weighted majority vote over their extracted answers.

This module is PURE LOGIC + scoring — it never calls a model. The N traces (each with
its per-token top-k probabilities) are supplied by the caller, which keeps the whole
scoring path unit-testable without inference. Wiring it onto N parallel llama-server
completions (payload ``n_probs=K`` — the same llama.cpp param ``logit_probe`` already
uses, surfaced as ``completion_probabilities`` in the response) is the follow-up step
(P21.A3), gated behind a live-server sanity check (P21.A2). Default-OFF feature flag:
``Features.deepconf``.

Confidence convention (HIGHER = more confident):
    token confidence  C_t = -(1/k) Σ_{j∈top-k} log p_j
    A peaked distribution makes the non-top candidates very improbable (large -log p),
    so C_t is large; a flat distribution gives a small C_t. Group confidence is a
    sliding-window mean of token confidences; a trace is scored by its *weakest link*
    (mean of its lowest-`bottom_fraction` group confidences, default 10%). We keep the
    traces with the HIGHEST trace score and weight each kept trace's vote by that score.

    NOTE (P21.A2): cross-check this polarity and the exact aggregation against the
    reference impl (github.com/facebookresearch/deepconf, vLLM PR #23201) on a live
    server before trusting the numbers. The functions are parameterised so flipping or
    swapping the aggregation is a one-line change.

References:
    - Paper: arXiv:2508.15260 (Fu, Wang, Tian, Zhao — Meta FAIR)
    - Deep dive + autopilot scope: research/deep-dives/optillm-test-time-techniques.md (P21.A)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# Floor for log() so a zero/negative probability from the backend can't blow up.
_PROB_FLOOR = 1e-12


@dataclass(frozen=True)
class DeepConfConfig:
    """Knobs for offline DeepConf. These are the P21.A3 NumericSwarm sweep surface.

    Attributes:
        n_traces: How many traces the caller generates (recorded for telemetry; the
            scorer works on whatever it is handed).
        keep_percent: Percentage of traces to KEEP, ranked by trace score (DeepConf-low
            ≈ 10 keeps the top 10%; DeepConf-high ≈ 90). Always keeps at least one.
        window: Sliding-window size (in tokens) for group confidence.
        warmup: Warmup-trace count for threshold calibration. Unused by the pure offline
            path (kept for parity with the online variant / future P21.A online work).
        group_metric: Trace-scoring aggregation — "bottom_pct" (weakest-link, paper
            default), "tail" (last window only), or "mean" (all groups).
        bottom_fraction: Fraction of lowest groups averaged when group_metric="bottom_pct".
        top_k: Number of candidate logprobs per token to read (the llama.cpp n_probs value).
    """

    n_traces: int = 16
    keep_percent: float = 10.0
    window: int = 2048
    warmup: int = 16
    group_metric: str = "bottom_pct"
    bottom_fraction: float = 0.10
    top_k: int = 20

    def __post_init__(self) -> None:
        if self.group_metric not in {"bottom_pct", "tail", "mean"}:
            raise ValueError(f"unknown group_metric: {self.group_metric!r}")
        if not 0.0 < self.keep_percent <= 100.0:
            raise ValueError(f"keep_percent must be in (0, 100], got {self.keep_percent}")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if not 0.0 < self.bottom_fraction <= 1.0:
            raise ValueError("bottom_fraction must be in (0, 1]")


@dataclass
class Trace:
    """One generated reasoning trace plus its per-token top-k probabilities.

    Attributes:
        text: The full decoded completion.
        token_top_probs: One list of top-k *linear* probabilities per generated token
            (e.g. ``[[0.9, 0.05, ...], [0.4, 0.3, ...], ...]``). May be empty if the
            backend returned no probabilities (such a trace scores 0.0).
    """

    text: str
    token_top_probs: list[list[float]] = field(default_factory=list)


@dataclass
class DeepConfResult:
    """Outcome of an offline DeepConf pass.

    Attributes:
        answer: Winning answer key (None if there were no usable traces).
        votes: answer-key -> summed confidence weight across kept traces.
        winning_weight: Confidence weight behind ``answer``.
        kept_indices: Indices (into the input traces) that survived filtering.
        trace_scores: Per-input-trace confidence score (parallel to the input list).
    """

    answer: str | None
    votes: dict[str, float]
    winning_weight: float
    kept_indices: list[int]
    trace_scores: list[float]


# ── token / group / trace confidence ──────────────────────────────────────


def token_confidence(top_probs: Sequence[float], top_k: int | None = None) -> float:
    """Mean negative log-prob over the top-k candidate probabilities for one token.

    Higher = more confident (a peaked distribution drives the improbable alternatives'
    -log p up). Returns 0.0 for an empty candidate list.
    """
    probs = list(top_probs) if top_k is None else list(top_probs)[:top_k]
    if not probs:
        return 0.0
    return -sum(math.log(max(p, _PROB_FLOOR)) for p in probs) / len(probs)


def group_confidences(token_confs: Sequence[float], window: int) -> list[float]:
    """Sliding-window means of token confidences (one value per window position).

    For a trace shorter than ``window`` this returns a single group spanning the whole
    trace, matching the paper's behaviour on short completions.
    """
    confs = list(token_confs)
    if not confs:
        return []
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(confs) <= window:
        return [sum(confs) / len(confs)]
    out: list[float] = []
    running = sum(confs[:window])
    out.append(running / window)
    for i in range(window, len(confs)):
        running += confs[i] - confs[i - window]
        out.append(running / window)
    return out


def trace_confidence(
    token_confs: Sequence[float],
    *,
    window: int,
    metric: str = "bottom_pct",
    bottom_fraction: float = 0.10,
) -> float:
    """Aggregate a trace's per-token confidences into a single score (higher = better)."""
    groups = group_confidences(token_confs, window)
    if not groups:
        return 0.0
    if metric == "mean":
        return sum(groups) / len(groups)
    if metric == "tail":
        return groups[-1]
    if metric == "bottom_pct":
        ordered = sorted(groups)
        n = max(1, math.ceil(bottom_fraction * len(ordered)))
        worst = ordered[:n]
        return sum(worst) / len(worst)
    raise ValueError(f"unknown metric: {metric!r}")


def score_trace(trace: Trace, config: DeepConfConfig) -> float:
    """Confidence score for a whole Trace under ``config`` (higher = better)."""
    token_confs = [token_confidence(tp, config.top_k) for tp in trace.token_top_probs]
    return trace_confidence(
        token_confs,
        window=config.window,
        metric=config.group_metric,
        bottom_fraction=config.bottom_fraction,
    )


# ── filtering + voting ─────────────────────────────────────────────────────


def select_top(scores: Sequence[float], keep_percent: float) -> list[int]:
    """Indices of the top ``keep_percent`` % of scores (rank-based, ties broken by index).

    Always keeps at least one. Rank-based rather than percentile-on-values so it behaves
    sensibly with tiny N and tied scores.
    """
    n = len(scores)
    if n == 0:
        return []
    keep = max(1, round((keep_percent / 100.0) * n))
    ranked = sorted(range(n), key=lambda i: (-scores[i], i))
    return sorted(ranked[:keep])


def normalize_answer(text: str) -> str:
    r"""Default answer-key extractor for voting.

    Prefers a ``\boxed{...}`` span, else the last number, else the last non-empty line.
    Lower-cased and whitespace-collapsed. Tasks with structured outputs should pass a
    custom extractor to ``run_offline`` instead of relying on this.
    """
    if not text or not text.strip():
        return ""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip().lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    last_line = [ln.strip() for ln in text.splitlines() if ln.strip()][-1]
    return re.sub(r"\s+", " ", last_line).lower()


def weighted_vote(
    answers: Sequence[str], weights: Sequence[float]
) -> tuple[str | None, dict[str, float]]:
    """Confidence-weighted majority vote. Ties broken by first appearance order."""
    tally: dict[str, float] = defaultdict(float)
    order: dict[str, int] = {}
    for idx, (ans, w) in enumerate(zip(answers, weights, strict=True)):
        if ans not in order:
            order[ans] = idx
        tally[ans] += w
    if not tally:
        return None, {}
    winner = max(tally, key=lambda a: (tally[a], -order[a]))
    return winner, dict(tally)


# ── top-level offline pass ─────────────────────────────────────────────────


def run_offline(
    traces: Sequence[Trace],
    config: DeepConfConfig | None = None,
    answer_extractor: Callable[[str], str] = normalize_answer,
) -> DeepConfResult:
    """Score → filter (keep top-η%) → confidence-weighted vote over already-generated traces.

    No inference happens here; ``traces`` are produced by the caller. This is the
    unit-testable core of P21.A.
    """
    config = config or DeepConfConfig()
    if not traces:
        return DeepConfResult(answer=None, votes={}, winning_weight=0.0,
                              kept_indices=[], trace_scores=[])

    scores = [score_trace(t, config) for t in traces]
    kept = select_top(scores, config.keep_percent)
    answers = [answer_extractor(traces[i].text) for i in kept]
    weights = [scores[i] for i in kept]
    winner, votes = weighted_vote(answers, weights)
    winning_weight = votes.get(winner, 0.0) if winner is not None else 0.0
    return DeepConfResult(
        answer=winner,
        votes=votes,
        winning_weight=winning_weight,
        kept_indices=kept,
        trace_scores=scores,
    )


# ── adapter: llama.cpp completion_probabilities -> Trace ────────────────────


def trace_from_completion_probabilities(
    text: str, completion_probabilities: Sequence[dict]
) -> Trace:
    """Build a Trace from a llama.cpp ``completion_probabilities`` list.

    Handles both shapes llama.cpp has shipped (confirmed against the production
    Qwen3.6 server 2026-05-24, P21.A2):

      * OpenAI-style (current builds): each token is
        ``{"token": t, "logprob": lp, "top_logprobs": [{"token": t, "logprob": lp}, ...]}``
        — log-probabilities.
      * legacy: ``{"content": t, "probs": [{"tok_str": t, "prob": p}, ...]}`` — linear probs.

    Linear probabilities are produced either way (logprobs are exp'd); the caller's
    ``token_confidence`` takes the log again. Robust to missing / non-dict entries.
    """
    token_top_probs: list[list[float]] = []
    for tok in completion_probabilities or []:
        if not isinstance(tok, dict):
            token_top_probs.append([])
            continue
        cands = tok.get("top_logprobs") or tok.get("probs") or []
        row: list[float] = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            if "logprob" in c:
                row.append(math.exp(c["logprob"]))
            elif "prob" in c:
                row.append(float(c["prob"]))
        token_top_probs.append(row)
    return Trace(text=text, token_top_probs=token_top_probs)
