"""Native candidate-scoring fast path for the typed decision plane (TD-1b).

``run_typed_decisions_native`` answers a whole ``Question`` catalogue in ONE
grammar-constrained ``LLMPrimitives.llm_call``: it generates exactly one token
per question, in catalogue order, under a GBNF grammar whose position ``i``
only accepts question ``i``'s candidate TOKEN IDS. The token probabilities of
the generated positions (``n_probs=K``) are then sliced to the question's
declared candidates, renormalized, and the argmax becomes the ``Decision``
value.

Why native mode exists:
    * JSON mode asks the model to write out a probability vector per question.
      That spends output tokens on numbers the sampler already has, and the
      written numbers are free-form text that the runner can only trust.
    * Native mode reads the sampler's own distribution at each generated
      position: the same one generation, but the probabilities are captured
      instead of re-typed, and the grammar makes an out-of-catalogue answer
      impossible at the token level.

TD-1b — what changed from TD-1a:
    TD-1a guessed which candidate strings are single tokens with a
    conservative text predicate, emitted quoted label literals in the
    grammar, and matched captured rows by token text. Measured live
    (2026-09-17, 27B): a quoted GBNF literal is parsed as a sequence of
    CHARACTER elements (``llama-grammar.cpp::parse_sequence`` -- every char
    becomes a ``LLAMA_GRETYPE_CHAR``), so the grammar could be satisfied by a
    partial token piece (``fal`` for ``false``); the emitted piece then
    matched no declared candidate and the question failed typed
    (``native_unknown_candidate``, 23/24). TD-1b tokenizes for real:

    * Eligibility: a candidate is native-eligible iff llama-server
      ``POST /tokenize`` returns EXACTLY ONE token id for the candidate's
      declared text or for the same text with a leading space (llama.cpp's
      pre-tokenizer distinguishes ``false`` from `` false``; every
      single-token variant found is kept). Everything else is
      ``native_unsupported_candidates`` -> JSON-mode fallback. No guessing.
    * Grammar: each position rule is an alternation of exact-token terminals
      ``<[id]>`` (``LLAMA_GRETYPE_TOKEN``, parsed by
      ``llama-grammar.cpp::parse_token``), one alternative per eligible
      candidate token variant, in declaration order. A quoted literal would
      be character-level again (the TD-1a bug); the id terminal pins exactly
      one generated token, preserving the one-token-per-question contract.
    * Matching: production-consolidated-v9 rows carry token ids
      (``server-task.cpp::probs_vector_to_json``), so the emitted token and
      every top-probability entry are matched BY ID; when several variants of
      one label are captured, their weights SUM. Text matching is retained
      ONLY as a documented fallback for rows that lack ``id`` ALTOGETHER
      (legacy ``content``/``probs``/``tok_str`` shapes); an id-bearing row
      never text-matches, and fallback text must equal one of the exact
      variant strings sent to ``/tokenize``.
    * ``n_probs`` sizes to the true token-alternative count per question
      after tokenization (variants included), plus the near-miss buffer,
      capped at 128.

Tokenizer seam:
    ``tokenize_fn: Callable[[str], Sequence[int] | None]`` is injectable on
    this runner and forwarded by ``runner.run_typed_decisions`` in native mode
    only. ``None`` (or an exception) from a call means "this text could not be
    tokenized" and fails the affected question closed. When not injected,
    ``_resolve_tokenize_fn`` derives the role's backend base URL from the
    primitives object and POSTs to ``{base_url}/tokenize`` via httpx (client
    pattern: ``src/backends/llama_server.py``).
    ``src/llm_primitives/tokenizer.py::LlamaTokenizer`` is deliberately NOT
    reused as the instrument: it returns counts only and silently falls back
    to a ``len(text) // 4`` heuristic on error, which cannot satisfy the
    exact-single-token-id contract (a heuristic count of 1 would fabricate
    eligibility). Its ``base_url`` is used as a last-resort resolution source.

Contract:
    * ONE call, exactly ``len(native_questions)`` generated tokens (the
      default ``n_tokens``), temperature ``0.0`` and seed ``0``.
    * ``n_probs`` defaults to the largest per-question token-alternative count
      plus ``_N_PROBS_BUFFER``, capped at ``_MAX_N_PROBS`` (128 — the
      llama-server payload cap in ``src/backends/llama_server.py``). The
      buffer exists because the captured top-K is global (all tokens), not
      per-candidate.
    * At each position the returned top-probability entries are matched to
      that question's candidate token ids (text fallback only for a whole row
      without ids), giving every declared candidate a weight (absent
      candidates weigh ``0.0``), then
      ``confidence.normalize_probabilities`` renormalizes. The argmax of the
      renormalized distribution is the ``Decision.value``;
      ``Decision.token_logprob`` is the raw model log-probability of that
      value's token (``None`` when the payload only carried linear probs).
      ``Decision.confidence`` uses ``choice_confidence`` for choice/noul and
      ``score_confidence`` for score, exactly like the JSON runner.
    * A position whose emitted token id/text is not one of the declared
      candidates, whose row is missing, or whose top-probability slice
      contains no declared candidate token yields a ``ParseFailure``
      (``native_unknown_candidate``) — never a default, and never a uniform
      fallback over an empty slice.
    * Questions that cannot be tokenized into single tokens — or whose
      candidates cannot be tokenized at all — are NEVER forced into the
      native batch. Multi-token candidates are returned as
      ``ParseFailure(native_unsupported_candidates)``; an unresolvable or
      unresponsive tokenizer yields ``ParseFailure(native_tokenizer_unavailable)``
      for every affected question. Both are emitted alongside the native
      decisions so the caller can re-ask exactly those questions in JSON
      mode, which remains the correctness fallback. This module never falls
      back itself: a silent mode switch would make the receipt lie about how
      the answer was produced. With no resolvable tokenizer NOTHING is sent
      to the model — no tokenizer, no call, no guess.
    * Transport failures (``llm_call`` returning an ``"[ERROR: ...]"`` string)
      short-circuit to a single ``transport_error`` failure; the stale
      ``_last_inference_meta`` is deliberately NOT read on that path.

Row-shape note (production-consolidated-v9):
    ``/completion`` with ``n_probs`` set returns ``completion_probabilities``
    rows shaped ``{"id": int, "token": str, "bytes": [...], "logprob": float,
    "top_logprobs": [{"id", "token", "bytes", "logprob"}, ...]}``
    (``tools/server/server-task.cpp::probs_vector_to_json`` with
    ``post_sampling_probs=false``, the default). The row and every entry carry
    the token ``id``, which is the primary match key. Older builds shipped the
    legacy ``{"content": str, "probs": [{"tok_str", "prob"}, ...]}`` shape and
    ``post_sampling_probs=true`` ships ``top_probs`` with linear ``prob`` and
    no ids; for a row without ``id`` the match falls back to the exact token
    text against the variant strings sent to ``/tokenize`` (see above).

Concurrency caveat:
    ``primitives._last_inference_meta`` is an INSTANCE-level attribute, not a
    request-scoped return value (``src/llm_primitives/inference.py``). This
    module reads it immediately after its single call, but a concurrent
    ``llm_call`` on the SAME primitives object can overwrite it between the
    call and the read. Native scoring therefore requires serialized use of
    one primitives object; sharing one across threads is unsupported.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from src.typed_decisions.confidence import (
    choice_confidence,
    normalize_probabilities,
    score_confidence,
)
from src.typed_decisions.runner import REASON_TRANSPORT_ERROR, _validated_catalogue
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    ParseFailure,
    Question,
    QuestionKind,
)

__all__ = [
    "REASON_NATIVE_TOKENIZER_UNAVAILABLE",
    "REASON_NATIVE_UNKNOWN_CANDIDATE",
    "REASON_NATIVE_UNSUPPORTED_CANDIDATES",
    "TokenizeFn",
    "run_typed_decisions_native",
]

logger = logging.getLogger(__name__)

# Text -> token ids, or ``None`` when the text could not be tokenized (the
# endpoint failed, or the response shape was unusable). ``None`` is never
# coerced into a count: the caller fails closed instead.
TokenizeFn = Callable[[str], Sequence[int] | None]

# Deterministic decode, identical policy to the JSON runner: temperature 0.0
# plus a pinned seed. The grammar removes most sampling freedom anyway (the
# token still comes from the model's masked distribution).
_DECODE_SEED = 0

# The captured top-K list is global over the vocabulary, so K must exceed the
# per-question token-alternative count for every declared candidate token to be
# captured; the buffer covers the near-miss tokens between candidates.
_N_PROBS_BUFFER = 4

# Mirrors the payload clamp in src/backends/llama_server.py
# (``payload["n_probs"] = min(128, int(request.n_probs))``).
_MAX_N_PROBS = 128

# /tokenize is a local, CPU-cheap endpoint; a short timeout keeps an
# unresponsive server from stalling the batch.
_TOKENIZE_TIMEOUT_S = 2.0

REASON_NATIVE_UNSUPPORTED_CANDIDATES = "native_unsupported_candidates"
REASON_NATIVE_UNKNOWN_CANDIDATE = "native_unknown_candidate"
REASON_NATIVE_TOKENIZER_UNAVAILABLE = "native_tokenizer_unavailable"

_NATIVE_INSTRUCTIONS = """\
Answer the question sequence below by emitting EXACTLY ONE token per question.
The decoder is grammar-constrained: position 1 may only be one of the first
question's candidates, position 2 one of the second question's candidates, and
so on. Emit the candidate labels verbatim, in order, with no separators,
whitespace, punctuation or explanation."""


@dataclass(frozen=True)
class _NativeCandidate:
    """One declared label bound to its exact token id(s).

    ``token_ids`` / ``token_texts`` hold one entry per single-token variant
    found by the tokenizer (the declared text and, when distinct, its
    space-prefixed form), in probe order. More than one id means the label has
    two tokenizations; their captured weights sum.
    """

    label: str
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]

    @property
    def alternatives(self) -> int:
        """Number of grammar alternatives this candidate contributes."""
        return len(self.token_ids)


@dataclass(frozen=True)
class _NativeQuestion:
    """A question whose every candidate is bound to exact token ids."""

    question: Question
    candidates: tuple[_NativeCandidate, ...]

    @property
    def alternatives(self) -> int:
        """Total token alternatives in this question's grammar position."""
        return sum(candidate.alternatives for candidate in self.candidates)


class _CandidateTokenizationError(Exception):
    """A question's candidates could not be bound to exact token ids.

    Carries the ``ParseFailure`` reason so the catalogue split can record it
    verbatim: ``native_unsupported_candidates`` (multi-token label or an
    id collision) or ``native_tokenizer_unavailable`` (the tokenizer could not
    answer for a candidate text).
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def run_typed_decisions_native(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    n_tokens: int | None = None,
    n_probs: int | None = None,
    tokenize_fn: TokenizeFn | None = None,
) -> DecisionResult:
    """Score one question catalogue in a single constrained generation.

    Args:
        primitives: The ``LLMPrimitives`` seam. Only the serial
            ``llm_call(prompt, role=..., n_tokens=..., grammar=...,
            temperature=..., seed=..., n_probs=...)`` contract is used, and
            ``_last_inference_meta`` is read immediately after the call (see
            the module docstring's concurrency caveat). Also the source of the
            default tokenizer's base URL when ``tokenize_fn`` is not given.
        state: Task/context state injected after the stable prefix.
        questions: The catalogue; ids must be unique and non-empty.
        role: Registry role the call is charged to.
        n_tokens: Output budget; defaults to exactly one token per
            native-capable question. A smaller explicit value truncates the
            batch (the missing positions fail typed); a larger one is capped
            by the grammar itself, which ends after the last position.
        n_probs: Top-K probability capture override. Defaults to the largest
            per-question token-alternative count (after tokenization) plus
            ``_N_PROBS_BUFFER``; always clamped to ``[1, _MAX_N_PROBS]``.
            Values below 1 raise ``ValueError``.
        tokenize_fn: Text -> token ids seam used to bind candidates to exact
            tokens (see module docstring). When ``None``, a default resolver
            derives the role's backend base URL from ``primitives`` and uses
            its ``POST /tokenize`` endpoint. When no tokenizer can be
            resolved, no model call is made and every question fails with
            ``native_tokenizer_unavailable``.

    Returns:
        ``DecisionResult`` with ``mode="native"``. ``decisions`` holds the
        native-capable questions in catalogue order; ``failures`` holds the
        transport error first (when the call failed), then one
        ``native_unsupported_candidates`` / ``native_tokenizer_unavailable``
        failure per excluded question, then one ``native_unknown_candidate``
        failure per unresolved position, all in catalogue order. Questions are
        never silently defaulted.
    """
    catalogue = _validated_catalogue(questions)
    tokenize = tokenize_fn if tokenize_fn is not None else _resolve_tokenize_fn(primitives, role)
    own_tokenizer = (
        tokenize if tokenize_fn is None and isinstance(tokenize, _HttpTokenizer) else None
    )
    try:
        return _score_native_batch(
            primitives,
            state=state,
            catalogue=catalogue,
            role=role,
            n_tokens=n_tokens,
            n_probs=n_probs,
            tokenize=tokenize,
        )
    finally:
        if own_tokenizer is not None:
            own_tokenizer.close()


def _score_native_batch(
    primitives: Any,
    *,
    state: str,
    catalogue: Sequence[Question],
    role: str,
    n_tokens: int | None,
    n_probs: int | None,
    tokenize: TokenizeFn | None,
) -> DecisionResult:
    """Tokenize, generate and slice one catalogue (tokenizer already resolved)."""
    if tokenize is None:
        native_questions: list[_NativeQuestion] = []
        tokenizer_failures = [
            ParseFailure(
                REASON_NATIVE_TOKENIZER_UNAVAILABLE,
                f"question {question.id!r}: no tokenizer could be resolved from the "
                "primitives object; ask this question in JSON mode",
            )
            for question in catalogue
        ]
    else:
        native_questions, tokenizer_failures = _tokenize_catalogue(catalogue, tokenize)

    prompt = _build_native_prompt(state, native_questions)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    if not native_questions:
        # Nothing is grammar-forceable; do not call the model at all.
        return DecisionResult(
            decisions=(),
            failures=tuple(tokenizer_failures),
            raw_text="",
            mode="native",
            elapsed_ms=0.0,
            prompt_sha256=prompt_sha256,
        )

    if n_tokens is None:
        n_tokens = len(native_questions)
    if n_probs is None:
        n_probs = min(
            _MAX_N_PROBS,
            max(native.alternatives for native in native_questions) + _N_PROBS_BUFFER,
        )
    else:
        n_probs = int(n_probs)
        if n_probs < 1:
            raise ValueError(f"n_probs must be >= 1, got {n_probs}")
        n_probs = min(_MAX_N_PROBS, n_probs)

    grammar = _build_native_grammar(native_questions)
    started = time.perf_counter()
    raw_text = str(
        primitives.llm_call(
            prompt,
            role=role,
            n_tokens=n_tokens,
            grammar=grammar,
            temperature=0.0,
            seed=_DECODE_SEED,
            n_probs=n_probs,
        )
        or ""
    )
    # Read the instance-level meta BEFORE anything else: it is overwritten by
    # any other llm_call on this primitives object (module docstring).
    meta = getattr(primitives, "_last_inference_meta", None)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if raw_text.strip().startswith("[ERROR:"):
        # A dead transport cannot be fixed by slicing stale probabilities.
        transport_failure = ParseFailure(REASON_TRANSPORT_ERROR, raw_text.strip())
        return DecisionResult(
            decisions=(),
            failures=tuple([transport_failure, *tokenizer_failures]),
            raw_text=raw_text,
            mode="native",
            elapsed_ms=elapsed_ms,
            prompt_sha256=prompt_sha256,
        )

    decisions, position_failures = _decisions_from_rows(meta, native_questions)
    return DecisionResult(
        decisions=tuple(decisions),
        failures=tuple(tokenizer_failures + position_failures),
        raw_text=raw_text,
        mode="native",
        elapsed_ms=elapsed_ms,
        prompt_sha256=prompt_sha256,
    )


# ── tokenizer resolution and the /tokenize seam ───────────────────────────


class _HttpTokenizer:
    """Best-effort llama-server ``/tokenize`` client returning token ids.

    Mirrors the pooled ``httpx.Client`` pattern of
    ``src/backends/llama_server.py``, but unlike
    ``src/llm_primitives/tokenizer.py::LlamaTokenizer`` it NEVER substitutes a
    character-count heuristic: any transport/shape failure returns ``None`` so
    the caller fails closed instead of fabricating eligibility.
    """

    def __init__(self, base_url: str, timeout: float = _TOKENIZE_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def __call__(self, text: str) -> list[int] | None:
        try:
            response = self._client.post(
                f"{self.base_url}/tokenize",
                # add_special=False: no BOS is prepended, so the ids describe
                # exactly ``text``. parse_special stays at the server default.
                json={"content": text, "add_special": False, "with_pieces": False},
            )
            response.raise_for_status()
            tokens = response.json().get("tokens")
        except Exception as exc:  # noqa: BLE001 - every failure is "no answer"
            logger.debug("native /tokenize failed for %r: %s", text, exc)
            return None
        if not isinstance(tokens, list):
            return None
        ids: list[int] = []
        for token in tokens:
            if not _is_token_id(token):
                return None
            ids.append(token)
        return ids

    def close(self) -> None:
        self._client.close()


def _resolve_tokenize_fn(primitives: Any, role: str) -> TokenizeFn | None:
    """Best-effort default tokenizer for this primitives object.

    Resolution order (first usable base URL wins): the role's backend
    (unwrapping CachingBackend / RoundRobinBackend / ConcurrencyAwareBackend
    wrappers), ``primitives.server_urls[role]`` (or the sole entry), then a
    ``LlamaTokenizer`` already attached by ``LLMPrimitives.__init__``. The
    endpoint is probed once with an empty content; a dead endpoint resolves no
    tokenizer, which makes the caller fail closed with zero model calls.
    """
    base_url = _resolve_base_url(primitives, role)
    if base_url is None:
        return None
    tokenizer = _HttpTokenizer(base_url)
    if tokenizer("") is None:
        tokenizer.close()
        logger.warning("native tokenizer unavailable at %s/tokenize", base_url)
        return None
    return tokenizer


def _resolve_base_url(primitives: Any, role: str) -> str | None:
    backend: Any = None
    get_backend = getattr(primitives, "get_backend", None)
    if callable(get_backend):
        try:
            backend = get_backend(role)
        except Exception:  # noqa: BLE001 - resolution is best-effort
            backend = None
    backends = getattr(primitives, "_backends", None)
    if backend is None and isinstance(backends, Mapping):
        backend = backends.get(role)
        if backend is None and len(backends) == 1:
            backend = next(iter(backends.values()))
    if backend is not None:
        url = _backend_base_url(backend)
        if url:
            return url

    server_urls = getattr(primitives, "server_urls", None)
    if isinstance(server_urls, Mapping) and server_urls:
        raw = server_urls.get(role)
        if raw is None and len(server_urls) == 1:
            raw = next(iter(server_urls.values()))
        url = _normalize_server_url(raw)
        if url:
            return url

    attached = getattr(primitives, "_tokenizer", None)
    url = getattr(attached, "base_url", None)
    if isinstance(url, str) and url:
        return url
    return None


def _backend_base_url(backend: Any) -> str | None:
    """Unwrap backend wrappers to the first concrete ``config.base_url``.

    Mirrors and extends ``src/backends/concurrency_aware.py::_get_base_url``
    (which only handles CachingBackend): RoundRobinBackend lists and
    ConcurrencyAwareBackend full/quarter members are traversed too. Cyclic
    wrappers are impossible in practice but guarded with a visited set.
    """
    queue: list[Any] = [backend]
    visited: set[int] = set()
    while queue:
        node = queue.pop(0)
        if node is None or id(node) in visited:
            continue
        visited.add(id(node))
        config = getattr(node, "config", None)
        base_url = getattr(config, "base_url", None)
        if isinstance(base_url, str) and base_url:
            return base_url
        for attr in ("backend", "_backend", "_full", "full_backend"):
            inner = getattr(node, attr, None)
            if inner is not None:
                queue.append(inner)
        for attr in ("backends", "_quarters", "quarter_backends"):
            members = getattr(node, attr, None)
            if isinstance(members, (list, tuple)):
                queue.extend(member for member in members if member is not None)
    return None


def _normalize_server_url(raw: Any) -> str | None:
    """First URL of a (possibly comma-separated, possibly ``full:``) role URL."""
    if not isinstance(raw, str):
        return None
    first = raw.split(",", 1)[0].strip()
    if first.startswith("full:"):
        first = first[len("full:") :].strip()
    return first or None


# ── catalogue tokenization ────────────────────────────────────────────────


def _tokenize_catalogue(
    questions: Sequence[Question],
    tokenize: TokenizeFn,
) -> tuple[list[_NativeQuestion], list[ParseFailure]]:
    """Partition the catalogue into token-bound questions and typed failures.

    Tokens are memoized per run (the same label appears in many questions), and
    every failure is recorded in catalogue order with the reason that keeps the
    question out of the native batch.
    """
    cache: dict[str, Sequence[int] | None] = {}

    def tokenize_cached(text: str) -> Sequence[int] | None:
        if text not in cache:
            try:
                cache[text] = tokenize(text)
            except Exception as exc:  # noqa: BLE001 - any failure is "no answer"
                logger.warning("native tokenizer raised for %r: %s", text, exc)
                cache[text] = None
        return cache[text]

    native: list[_NativeQuestion] = []
    failures: list[ParseFailure] = []
    for question in questions:
        try:
            native.append(_tokenize_question(question, tokenize_cached))
        except _CandidateTokenizationError as exc:
            failures.append(ParseFailure(exc.reason, f"question {question.id!r}: {exc.detail}"))
    return native, failures


def _tokenize_question(question: Question, tokenize: TokenizeFn) -> _NativeQuestion:
    """Bind every candidate label of one question to exact token id(s).

    Raises:
        _CandidateTokenizationError: when a candidate text cannot be
            tokenized at all (``native_tokenizer_unavailable``), or when any
            label is not a single token / two labels collide on one token id
            (``native_unsupported_candidates``). Partial eligibility is not a
            thing: an answer that could have chosen an unsupported label must
            go to the JSON fallback as a whole question.
    """
    candidates: list[_NativeCandidate] = []
    unsupported: list[str] = []
    for label in _candidate_labels(question):
        token_ids: list[int] = []
        token_texts: list[str] = []
        for text in _candidate_variants(label):
            ids = tokenize(text)
            if ids is None:
                raise _CandidateTokenizationError(
                    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
                    f"the tokenizer returned no ids for candidate text {text!r}; "
                    "ask this question in JSON mode",
                )
            if len(ids) == 1:
                token_ids.append(ids[0])
                token_texts.append(text)
        if not token_ids:
            unsupported.append(label)
            continue
        candidates.append(
            _NativeCandidate(
                label=label,
                token_ids=tuple(token_ids),
                token_texts=tuple(token_texts),
            )
        )
    if unsupported:
        raise _CandidateTokenizationError(
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
            f"candidate labels {unsupported!r} do not tokenize to exactly one token "
            "(with or without a leading space); ask this question in JSON mode",
        )
    _reject_token_id_collisions(candidates)
    return _NativeQuestion(question=question, candidates=tuple(candidates))


def _candidate_variants(label: str) -> tuple[str, ...]:
    """The candidate texts probed for single-token tokenization.

    llama.cpp's pre-tokenizer distinguishes a word at a boundary (``false``)
    from the same word after a space (`` false``); either can be the token the
    decoder is about to emit, so both are probed (and both kept when both are
    single tokens). A label declared with a leading space is probed as declared
    and with that space removed.
    """
    if label.startswith(" "):
        stripped = label[1:]
        return (label, stripped) if stripped else (label,)
    return (label, " " + label)


def _reject_token_id_collisions(candidates: Sequence[_NativeCandidate]) -> None:
    """Two labels bound to the same token id are indistinguishable natively."""
    seen: dict[int, str] = {}
    for candidate in candidates:
        for token_id in candidate.token_ids:
            other = seen.get(token_id)
            if other is not None and other != candidate.label:
                raise _CandidateTokenizationError(
                    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
                    f"candidate labels {other!r} and {candidate.label!r} both tokenize "
                    f"to token id {token_id}; the native grammar cannot distinguish "
                    "them, ask this question in JSON mode",
                )
            seen[token_id] = candidate.label


# ── grammar and prompt ────────────────────────────────────────────────────


def _build_native_grammar(questions: Sequence[_NativeQuestion]) -> str:
    """Build the bare-token-sequence grammar for the native batch.

    ``root`` concatenates one rule per question in catalogue order; rule ``i``
    is the alternation of question ``i``'s token alternatives as exact-token
    terminals (``<[id]>``). This is NOT ``schema.build_gbnf`` (that one builds
    the JSON-object grammar): native mode never emits JSON, it emits one forced
    token per question, so the grammar is the sequence itself.

    Token-id terminals are load-bearing: a quoted literal would be parsed as
    character elements by ``llama-grammar.cpp::parse_sequence``, and the
    decoder could satisfy it with a partial piece (the TD-1a ``fal`` bug).
    ``<[id]>`` can only be advanced by exactly that token, so each position
    consumes exactly one generated token.
    """
    rules = ["root ::= " + " ".join(f"position-{index}" for index in range(len(questions)))]
    for index, native in enumerate(questions):
        alternatives = " | ".join(
            f"<[{token_id}]>" for candidate in native.candidates for token_id in candidate.token_ids
        )
        rules.append(f"position-{index} ::= {alternatives}")
    return "\n".join(rules) + "\n"


def _build_native_prompt(state: str, questions: Sequence[_NativeQuestion]) -> str:
    """Build the deterministic native prompt for the native batch in order."""
    lines = [f"Emit exactly {len(questions)} tokens.", "", _NATIVE_INSTRUCTIONS, ""]
    lines.append(f"STATE:\n{state}")
    lines.append("")
    lines.append("QUESTION SEQUENCE:")
    for index, native in enumerate(questions, start=1):
        question = native.question
        lines.append(f"{index}. id={question.id} kind={question.kind.value}")
        lines.append(f"   question: {question.text}")
        lines.append(f"   candidates: {' | '.join(_candidate_labels(question))}")
        for criterion in question.criteria:
            lines.append(f"   criterion: {criterion}")
    return "\n".join(lines) + "\n"


def _candidate_labels(question: Question) -> list[str]:
    """Candidate labels in declaration order (the grammar's alternative order)."""
    if question.kind is QuestionKind.CHOICE:
        return list(question.options)
    if question.kind is QuestionKind.SCORE:
        return [str(level) for level in question.levels]
    return ["true", "false"]


# ── completion_probabilities slicing ──────────────────────────────────────


def _decisions_from_rows(
    meta: Any,
    questions: Sequence[_NativeQuestion],
) -> tuple[list[Decision], list[ParseFailure]]:
    """Turn captured probability rows into typed decisions / per-position failures."""
    rows = _rows_from_meta(meta)
    decisions: list[Decision] = []
    failures: list[ParseFailure] = []

    for index, native in enumerate(questions):
        question = native.question
        if index >= len(rows):
            failures.append(
                ParseFailure(
                    REASON_NATIVE_UNKNOWN_CANDIDATE,
                    f"position {index + 1} question {question.id!r}: no "
                    "completion_probabilities row was captured",
                )
            )
            continue
        row = rows[index]
        id_to_label = {
            token_id: candidate.label
            for candidate in native.candidates
            for token_id in candidate.token_ids
        }
        text_to_label = {
            text: candidate.label
            for candidate in native.candidates
            for text in candidate.token_texts
        }
        emitted_label = _match_row_label(row, id_to_label, text_to_label)
        if emitted_label is None:
            failures.append(
                ParseFailure(
                    REASON_NATIVE_UNKNOWN_CANDIDATE,
                    f"position {index + 1} question {question.id!r}: emitted token "
                    f"({_describe_row_token(row)}) is not one of the declared "
                    f"candidates {_candidate_labels(question)!r}",
                )
            )
            continue
        entries = _row_entries(row)
        # Text fallback is scoped to a row that lacks ``id`` altogether: when
        # the row carries an id the capture is a modern one, and every entry
        # must resolve by id (a text fallback there could resurrect a stale
        # piece from a mixed shape).
        text_fallback = not _is_token_id(row.get("id"))
        weights = _candidate_weights(entries, native, text_fallback=text_fallback)
        if not any(weight > 0.0 for weight in weights.values()):
            # normalize_probabilities would otherwise fabricate a uniform
            # distribution over an all-zero slice; the capture is unusable.
            failures.append(
                ParseFailure(
                    REASON_NATIVE_UNKNOWN_CANDIDATE,
                    f"position {index + 1} question {question.id!r}: none of the "
                    "declared candidate tokens appears in the captured top probabilities",
                )
            )
            continue
        decisions.append(_decision_from_weights(native, weights, entries, text_fallback))
    return decisions, failures


def _match_row_label(
    row: Mapping[str, Any],
    id_to_label: Mapping[int, str],
    text_to_label: Mapping[str, str],
) -> str | None:
    """Resolve the emitted row to a candidate label.

    The v9 row carries ``id``, which is authoritative: when an ``id`` is
    present the text is NOT consulted (the text is derived from the id, and
    mixing keys would let a stale text override the sampler's token). Text
    matching is the documented fallback for legacy rows that lack ``id``.
    """
    token_id = row.get("id")
    if _is_token_id(token_id):
        return id_to_label.get(token_id)
    text = _row_token_text(row)
    if isinstance(text, str):
        return text_to_label.get(text)
    return None


def _describe_row_token(row: Mapping[str, Any]) -> str:
    """Human-readable identity of the emitted row token for failure details."""
    token_id = row.get("id")
    text = _row_token_text(row)
    parts = []
    if _is_token_id(token_id):
        parts.append(f"id={token_id}")
    parts.append(f"text={text!r}")
    return " ".join(parts)


def _decision_from_weights(
    native: _NativeQuestion,
    weights: Mapping[str, float],
    entries: Sequence[Mapping[str, Any]],
    text_fallback: bool,
) -> Decision:
    """Renormalize candidate weights and build the typed ``Decision``.

    ``weights`` carries one entry per declared label (absent candidates at
    ``0.0``), so the renormalized distribution always covers the full
    candidate set. The value is the argmax of the slice (ties resolve to the
    first declared candidate); under the grammar's masked greedy decode the
    emitted token and the argmax coincide, which is why the emitted token was
    only used as an alignment check upstream.
    """
    question = native.question
    probabilities = normalize_probabilities(weights)
    value_label = max(weights, key=lambda label: weights[label])
    value_candidate = next(
        candidate for candidate in native.candidates if candidate.label == value_label
    )
    token_logprob = _logprob_for(entries, value_candidate, text_fallback=text_fallback)

    if question.kind is QuestionKind.SCORE:
        typed_probabilities: Mapping[str | int, float] = {
            int(label): probability for label, probability in probabilities.items()
        }
        value: object = int(value_label)
        confidence = score_confidence(typed_probabilities)
    elif question.kind is QuestionKind.NOUL:
        typed_probabilities = probabilities
        value = value_label == "true"
        confidence = choice_confidence(typed_probabilities)
    else:
        typed_probabilities = probabilities
        value = value_label
        confidence = choice_confidence(typed_probabilities)

    return Decision(
        question_id=question.id,
        kind=question.kind,
        value=value,
        probabilities=typed_probabilities,
        confidence=confidence,
        mode="native",
        token_logprob=token_logprob,
    )


def _rows_from_meta(meta: Any) -> list[Mapping[str, Any]]:
    """Extract the ``completion_probabilities`` rows from an inference meta dict.

    Missing / malformed meta yields ``[]``, which every position then reports
    as a typed failure — never as a default answer.
    """
    if not isinstance(meta, Mapping):
        return []
    rows = meta.get("completion_probabilities")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _row_token_text(row: Mapping[str, Any]) -> str | None:
    """The emitted token's text: ``token`` (v9) or ``content`` (legacy)."""
    for key in ("token", "content"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def _row_entries(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The row's top-probability entries: logprob- or prob-keyed, any generation."""
    for key in ("top_logprobs", "top_probs", "probs"):
        entries = row.get(key)
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, Mapping)]
    return []


def _entry_token_text(entry: Mapping[str, Any]) -> str | None:
    """The candidate token's text: ``token`` (v9) or ``tok_str`` (legacy)."""
    for key in ("token", "tok_str"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def _entry_weight(entry: Mapping[str, Any]) -> float | None:
    """One entry's linear probability weight (``prob`` directly, ``logprob`` exp'd).

    Non-finite or unconvertible values yield ``None`` (the entry is skipped),
    never a fabricated ``0.0``-vs-missing ambiguity.
    """
    probability = entry.get("prob")
    if isinstance(probability, (int, float)) and not isinstance(probability, bool):
        value = float(probability)
        return value if math.isfinite(value) else None
    logprob = entry.get("logprob")
    if isinstance(logprob, (int, float)) and not isinstance(logprob, bool):
        value = float(logprob)
        if not math.isfinite(value):
            return None
        try:
            return math.exp(value)
        except OverflowError:
            return None
    return None


def _entry_logprob(entry: Mapping[str, Any]) -> float | None:
    """One entry's log-probability, re-derived from ``prob`` when necessary."""
    logprob = entry.get("logprob")
    if isinstance(logprob, (int, float)) and not isinstance(logprob, bool):
        value = float(logprob)
        if math.isfinite(value):
            return value
        return None
    probability = _entry_weight(entry)
    if probability is None or probability <= 0.0:
        return None
    return math.log(probability)


def _candidate_weights(
    entries: Sequence[Mapping[str, Any]],
    native: _NativeQuestion,
    *,
    text_fallback: bool,
) -> dict[str, float]:
    """Slice entries to the declared candidates; ids first, then text.

    Every declared label gets an entry (``0.0`` when its token is absent from
    the captured top-K, i.e. below the capture cutoff), so the caller's
    renormalization always covers the full candidate set. Entries carrying an
    ``id`` are matched by id only (an id outside the candidate set is ignored,
    never reinterpreted as text); each distinct id contributes once. Ids-less
    entries match by exact token text against the variant strings sent to
    ``/tokenize`` (each distinct text contributes once) ONLY when
    ``text_fallback`` is set, i.e. when the enclosing row carries no id. A
    label with two single-token variants has its captured weights summed.
    """
    id_to_label = {
        token_id: candidate.label
        for candidate in native.candidates
        for token_id in candidate.token_ids
    }
    text_to_label = {
        text: candidate.label for candidate in native.candidates for text in candidate.token_texts
    }
    weights = {candidate.label: 0.0 for candidate in native.candidates}
    seen_ids: set[int] = set()
    seen_texts: set[str] = set()
    for entry in entries:
        weight = _entry_weight(entry)
        if weight is None:
            continue
        token_id = entry.get("id")
        if _is_token_id(token_id):
            if token_id in seen_ids:
                continue
            seen_ids.add(token_id)
            label = id_to_label.get(token_id)
            if label is not None:
                weights[label] += weight
            continue
        if not text_fallback:
            continue
        text = _entry_token_text(entry)
        if text is None or text in seen_texts:
            continue
        seen_texts.add(text)
        label = text_to_label.get(text)
        if label is not None:
            weights[label] += weight
    return weights


def _logprob_for(
    entries: Sequence[Mapping[str, Any]],
    candidate: _NativeCandidate,
    *,
    text_fallback: bool,
) -> float | None:
    """Raw log-probability of a candidate's token in the captured row, if present.

    A label with two single-token variants returns the first listed match
    (llama.cpp lists top entries in descending probability), i.e. the most
    probable captured variant. Text matching follows the same row-scoped
    ``text_fallback`` rule as ``_candidate_weights``.
    """
    token_ids = set(candidate.token_ids)
    token_texts = set(candidate.token_texts)
    for entry in entries:
        token_id = entry.get("id")
        if _is_token_id(token_id):
            if token_id in token_ids:
                return _entry_logprob(entry)
            continue
        if not text_fallback:
            continue
        text = _entry_token_text(entry)
        if text is not None and text in token_texts:
            return _entry_logprob(entry)
    return None


def _is_token_id(value: Any) -> bool:
    """True for a real integer token id (``bool`` is not a token id)."""
    return isinstance(value, int) and not isinstance(value, bool)
