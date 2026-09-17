"""Native candidate-scoring fast path for the typed decision plane (TD-1a).

``run_typed_decisions_native`` answers a whole ``Question`` catalogue in ONE
grammar-constrained ``LLMPrimitives.llm_call``: it generates exactly one token
per question, in catalogue order, under a GBNF grammar whose position ``i``
only accepts question ``i``'s candidate tokens. The token probabilities of the
generated positions (``n_probs=K``) are then sliced to the question's declared
candidates, renormalized, and the argmax becomes the ``Decision`` value.

Why native mode exists:
    * JSON mode asks the model to write out a probability vector per question.
      That spends output tokens on numbers the sampler already has, and the
      written numbers are free-form text that the runner can only trust.
    * Native mode reads the sampler's own distribution at each generated
      position: the same one generation, but the probabilities are captured
      instead of re-typed, and the grammar makes an out-of-catalogue answer
      impossible at the token level.

Contract:
    * ONE call, exactly ``len(native_questions)`` generated tokens (the
      default ``n_tokens``), temperature ``0.0`` and seed ``0``.
    * ``n_probs`` defaults to ``max candidate count + _N_PROBS_BUFFER``,
      capped at ``_MAX_N_PROBS`` (128 — the llama-server payload cap in
      ``src/backends/llama_server.py``). The buffer exists because the
      captured top-K is global (all tokens), not per-candidate.
    * At each position the returned top-probability entries are matched to
      that question's candidate tokens BY TOKEN TEXT (``token`` / ``tok_str``
      — see the row-shape note below), giving every declared candidate a
      weight (absent candidates weigh ``0.0``), then
      ``confidence.normalize_probabilities`` renormalizes. The argmax of the
      renormalized distribution is the ``Decision.value``;
      ``Decision.token_logprob`` is the raw model log-probability of that
      value's token (``None`` when the payload only carried linear probs).
      ``Decision.confidence`` uses ``choice_confidence`` for choice/noul and
      ``score_confidence`` for score, exactly like the JSON runner.
    * A position whose emitted token is not one of the declared candidates,
      whose row is missing, or whose top-probability slice contains no
      declared candidate token yields a ``ParseFailure``
      (``native_unknown_candidate``) — never a default, and never a uniform
      fallback over an empty slice.
    * Questions whose candidate labels are not presumptively single tokens
      (same conservative predicate as ``schema.build_gbnf``: printable ASCII,
      no whitespace/quotes/backslash, at most 16 chars) are NEVER forced into
      the native batch. They are returned as
      ``ParseFailure(native_unsupported_candidates)`` alongside the native
      decisions so the caller can re-ask exactly those questions in JSON mode,
      which remains the correctness fallback. This module never falls back
      itself: a silent mode switch would make the receipt lie about how the
      answer was produced.
    * Transport failures (``llm_call`` returning an ``"[ERROR: ...]"`` string)
      short-circuit to a single ``transport_error`` failure; the stale
      ``_last_inference_meta`` is deliberately NOT read on that path.

Row-shape note (production-consolidated-v9):
    ``/completion`` with ``n_probs`` set returns ``completion_probabilities``
    rows shaped ``{"id": int, "token": str, "bytes": [...], "logprob": float,
    "top_logprobs": [{"id", "token", "bytes", "logprob"}, ...]}``
    (``tools/server/server-task.cpp::probs_vector_to_json`` with
    ``post_sampling_probs=false``, the default). Older builds shipped the
    legacy ``{"content": str, "probs": [{"tok_str", "prob"}, ...]}`` shape and
    ``post_sampling_probs=true`` ships ``top_probs`` with linear ``prob``;
    all three are accepted. The label -> token-id map is not available
    offline (there is no tokenizer at build time) and ``token`` is the text
    the grammar matched, so slicing matches on token text.

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
from collections.abc import Mapping, Sequence
from typing import Any

from src.typed_decisions.confidence import (
    choice_confidence,
    normalize_probabilities,
    score_confidence,
)
from src.typed_decisions.runner import REASON_TRANSPORT_ERROR, _validated_catalogue
from src.typed_decisions.schema import _is_presumptive_single_token
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    ParseFailure,
    Question,
    QuestionKind,
)

__all__ = [
    "REASON_NATIVE_UNKNOWN_CANDIDATE",
    "REASON_NATIVE_UNSUPPORTED_CANDIDATES",
    "run_typed_decisions_native",
]

logger = logging.getLogger(__name__)

# Deterministic decode, identical policy to the JSON runner: temperature 0.0
# plus a pinned seed. The grammar removes most sampling freedom anyway (the
# token still comes from the model's masked distribution).
_DECODE_SEED = 0

# The captured top-K list is global over the vocabulary, so K must exceed the
# per-question candidate count for every declared candidate token to be
# captured; the buffer covers the near-miss tokens between candidates.
_N_PROBS_BUFFER = 4

# Mirrors the payload clamp in src/backends/llama_server.py
# (``payload["n_probs"] = min(128, int(request.n_probs))``).
_MAX_N_PROBS = 128

REASON_NATIVE_UNSUPPORTED_CANDIDATES = "native_unsupported_candidates"
REASON_NATIVE_UNKNOWN_CANDIDATE = "native_unknown_candidate"

_NATIVE_INSTRUCTIONS = """\
Answer the question sequence below by emitting EXACTLY ONE token per question.
The decoder is grammar-constrained: position 1 may only be one of the first
question's candidates, position 2 one of the second question's candidates, and
so on. Emit the candidate labels verbatim, in order, with no separators,
whitespace, punctuation or explanation."""


def run_typed_decisions_native(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    n_tokens: int | None = None,
    n_probs: int | None = None,
) -> DecisionResult:
    """Score one question catalogue in a single constrained generation.

    Args:
        primitives: The ``LLMPrimitives`` seam. Only the serial
            ``llm_call(prompt, role=..., n_tokens=..., grammar=...,
            temperature=..., seed=..., n_probs=...)`` contract is used, and
            ``_last_inference_meta`` is read immediately after the call (see
            the module docstring's concurrency caveat).
        state: Task/context state injected after the stable prefix.
        questions: The catalogue; ids must be unique and non-empty.
        role: Registry role the call is charged to.
        n_tokens: Output budget; defaults to exactly one token per
            native-capable question. A smaller explicit value truncates the
            batch (the missing positions fail typed); a larger one is capped
            by the grammar itself, which ends after the last position.
        n_probs: Top-K probability capture override. Defaults to
            ``max candidate count + _N_PROBS_BUFFER``; always clamped to
            ``[1, _MAX_N_PROBS]``. Values below 1 raise ``ValueError``.

    Returns:
        ``DecisionResult`` with ``mode="native"``. ``decisions`` holds the
        native-capable questions in catalogue order; ``failures`` holds the
        transport error first (when the call failed), then one
        ``native_unsupported_candidates`` failure per excluded question, then
        one ``native_unknown_candidate`` failure per unresolved position, all
        in catalogue order. Questions are never silently defaulted.
    """
    catalogue = _validated_catalogue(questions)
    native_questions, unsupported_failures = _split_native_capable(catalogue)
    prompt = _build_native_prompt(state, native_questions)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    if not native_questions:
        # Nothing is grammar-forceable; do not call the model at all.
        return DecisionResult(
            decisions=(),
            failures=tuple(unsupported_failures),
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
            max(len(_candidate_labels(question)) for question in native_questions)
            + _N_PROBS_BUFFER,
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
            failures=tuple([transport_failure, *unsupported_failures]),
            raw_text=raw_text,
            mode="native",
            elapsed_ms=elapsed_ms,
            prompt_sha256=prompt_sha256,
        )

    decisions, position_failures = _decisions_from_rows(meta, native_questions)
    return DecisionResult(
        decisions=tuple(decisions),
        failures=tuple(unsupported_failures + position_failures),
        raw_text=raw_text,
        mode="native",
        elapsed_ms=elapsed_ms,
        prompt_sha256=prompt_sha256,
    )


# ── catalogue split and grammar ───────────────────────────────────────────


def _split_native_capable(
    questions: Sequence[Question],
) -> tuple[list[Question], list[ParseFailure]]:
    """Partition the catalogue into grammar-forceable questions and failures.

    A question qualifies only when every candidate label is presumptively a
    single token under the same conservative predicate the JSON-mode GBNF
    builder uses (``schema._is_presumptive_single_token``), so the two
    builders can never disagree about tokenizability.
    """
    native: list[Question] = []
    failures: list[ParseFailure] = []
    for question in questions:
        labels = _candidate_labels(question)
        unsupported = [label for label in labels if not _is_presumptive_single_token(label)]
        if unsupported:
            failures.append(
                ParseFailure(
                    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
                    f"question {question.id!r}: candidate labels {unsupported!r} are "
                    "not presumptively single tokens; ask this question in JSON mode",
                )
            )
            continue
        native.append(question)
    return native, failures


def _build_native_grammar(questions: Sequence[Question]) -> str:
    """Build the bare-token-sequence grammar for the native batch.

    ``root`` concatenates one rule per question in catalogue order; rule ``i``
    is the alternation of question ``i``'s candidate labels. This is NOT
    ``schema.build_gbnf`` (that one builds the JSON-object grammar): native
    mode never emits JSON, it emits one forced token per question, so the
    grammar is the sequence itself. The single-token presumption is shared
    (see ``_split_native_capable``), and labels are emitted as quoted GBNF
    terminals — the predicate excludes ``"`` and ``\\``, so no escaping is
    required.
    """
    rules = ["root ::= " + " ".join(f"position-{index}" for index in range(len(questions)))]
    for index, question in enumerate(questions):
        alternatives = " | ".join(f'"{label}"' for label in _candidate_labels(question))
        rules.append(f"position-{index} ::= {alternatives}")
    return "\n".join(rules) + "\n"


def _build_native_prompt(state: str, questions: Sequence[Question]) -> str:
    """Build the deterministic native prompt for the native batch in order."""
    lines = [f"Emit exactly {len(questions)} tokens.", "", _NATIVE_INSTRUCTIONS, ""]
    lines.append(f"STATE:\n{state}")
    lines.append("")
    lines.append("QUESTION SEQUENCE:")
    for index, question in enumerate(questions, start=1):
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
    questions: Sequence[Question],
) -> tuple[list[Decision], list[ParseFailure]]:
    """Turn captured probability rows into typed decisions / per-position failures."""
    rows = _rows_from_meta(meta)
    decisions: list[Decision] = []
    failures: list[ParseFailure] = []

    for index, question in enumerate(questions):
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
        labels = _candidate_labels(question)
        emitted = _row_token_text(row)
        if emitted is None or emitted not in labels:
            failures.append(
                ParseFailure(
                    REASON_NATIVE_UNKNOWN_CANDIDATE,
                    f"position {index + 1} question {question.id!r}: emitted token "
                    f"{emitted!r} is not one of the declared candidates {labels!r}",
                )
            )
            continue
        entries = _row_entries(row)
        weights = _candidate_weights(entries, labels)
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
        decisions.append(_decision_from_weights(question, weights, entries))
    return decisions, failures


def _decision_from_weights(
    question: Question,
    weights: Mapping[str, float],
    entries: Sequence[Mapping[str, Any]],
) -> Decision:
    """Renormalize candidate weights and build the typed ``Decision``.

    ``weights`` carries one entry per declared label (absent candidates at
    ``0.0``), so the renormalized distribution always covers the full
    candidate set. The value is the argmax of the slice (ties resolve to the
    first declared candidate); under the grammar's masked greedy decode the
    emitted token and the argmax coincide, which is why the emitted token was
    only used as an alignment check upstream.
    """
    probabilities = normalize_probabilities(weights)
    value_label = max(weights, key=lambda label: weights[label])
    token_logprob = _logprob_for(entries, value_label)

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
    labels: Sequence[str],
) -> dict[str, float]:
    """Slice entries to the declared candidates; first matching entry wins.

    Every declared label gets an entry (``0.0`` when its token is absent from
    the captured top-K, i.e. below the capture cutoff), so the caller's
    renormalization always covers the full candidate set.
    """
    label_set = set(labels)
    weights = {label: 0.0 for label in labels}
    seen: set[str] = set()
    for entry in entries:
        token = _entry_token_text(entry)
        if token is None or token not in label_set or token in seen:
            continue
        weight = _entry_weight(entry)
        if weight is None:
            continue
        seen.add(token)
        weights[token] = weight
    return weights


def _logprob_for(entries: Sequence[Mapping[str, Any]], label: str) -> float | None:
    """Raw log-probability of ``label``'s token in the captured row, if present."""
    for entry in entries:
        if _entry_token_text(entry) == label:
            return _entry_logprob(entry)
    return None
