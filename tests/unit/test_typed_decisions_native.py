"""Unit tests for the TD-1b native candidate-scoring path.

Mirrors the fake-primitives pattern of ``tests/unit/test_typed_decisions.py``
(one canned response, captured call kwargs) but adds the instance-level
``_last_inference_meta`` that carries synthetic ``completion_probabilities``
rows, plus a fake tokenizer seam.

The primary row shape pinned here is the production-consolidated-v9
``/completion`` shape emitted by
``tools/server/server-task.cpp::probs_vector_to_json`` with
``post_sampling_probs=false``::

    {"id": int, "token": str, "bytes": [int],
     "logprob": float, "top_logprobs": [{"id", "token", "bytes", "logprob"}, ...]}

The legacy ``{"content", "probs": [{"tok_str", "prob"}]}`` shape and the
``top_probs`` linear-probability variant (both WITHOUT token ids) are pinned
alongside it: they exercise the documented text fallback. No model/server
call.

The fake tokenizer makes both the bare and the space-prefixed form of every
fixture label a single token with distinct ids — as llama.cpp tokenizers
commonly do — so eligibility, grammar construction and id-based slicing are
all exercised against realistic tokenizer output. Texts outside the fake
vocabulary fall back to one id per character (multi-token).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from src.typed_decisions import (
    DecisionResult,
    Question,
    QuestionKind,
    run_typed_decisions,
    run_typed_decisions_native,
)
from src.typed_decisions.confidence import choice_confidence, score_confidence
from src.typed_decisions.native import (
    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
    REASON_NATIVE_UNKNOWN_CANDIDATE,
    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
)

ROLE = "worker"
STATE = "unit-test state: native candidate scoring over a three-question catalogue."


# ── fake tokenizer ────────────────────────────────────────────────────────


def _default_vocab() -> dict[str, tuple[int, ...]]:
    """Bare and space-prefixed single-token ids for every fixture label."""
    labels = (
        "red",
        "blue",
        "green",
        "0",
        "1",
        "2",
        "3",
        "true",
        "false",
        "yes",
        "no",
        "purple",
        "x",
        "y",
        "fal",
    )
    vocab: dict[str, tuple[int, ...]] = {}
    for index, label in enumerate(labels):
        vocab[label] = (1000 + 2 * index,)
        vocab[" " + label] = (1001 + 2 * index,)
    return vocab


_DEFAULT_VOCAB = _default_vocab()
_DEFAULT_IDS = {text: ids[0] for text, ids in _DEFAULT_VOCAB.items()}


class _FakeTokenizer:
    """Text -> token ids with a recorded call log.

    ``failing`` texts return ``None`` (the seam's "no answer" signal). Texts
    absent from ``vocab`` tokenize to one id per character, so multi-token
    candidates are representable without special-casing.
    """

    def __init__(
        self,
        vocab: Mapping[str, tuple[int, ...]] | None = None,
        failing: Sequence[str] = (),
    ) -> None:
        self.vocab = {text: tuple(ids) for text, ids in (vocab or _DEFAULT_VOCAB).items()}
        self.failing = set(failing)
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[int] | None:
        self.calls.append(text)
        if text in self.failing:
            return None
        if text in self.vocab:
            return list(self.vocab[text])
        return [ord(char) for char in text]


class _FakePrimitives:
    """Canned-response stand-in for ``LLMPrimitives`` with inference meta.

    ``responses`` is consumed in order; the last entry repeats. The meta is
    instance-level exactly as in ``src/llm_primitives/inference.py``, so the
    tests can also prove that a transport failure does not fall back to stale
    probability rows. No ``_backends`` / ``server_urls`` / ``_tokenizer`` is
    configured, so the default tokenizer resolver must fail closed.
    """

    def __init__(self, responses: str | Sequence[str], meta: dict[str, Any] | None = None):
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.calls: list[dict] = []
        self._last_inference_meta = meta if meta is not None else {}

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


# ── fixtures ──────────────────────────────────────────────────────────────

CHOICE = Question(
    id="choice",
    kind=QuestionKind.CHOICE,
    text="Pick a colour.",
    options=("red", "blue", "green"),
)
SCORE = Question(
    id="score",
    kind=QuestionKind.SCORE,
    text="Rate the change.",
    levels=(0, 1, 2, 3),
)
NOUL = Question(id="noul", kind=QuestionKind.NOUL, text="Ship it?")
QUESTIONS = (CHOICE, SCORE, NOUL)

MULTI_TOKEN = Question(
    id="multi",
    kind=QuestionKind.CHOICE,
    text="Pick a multi-token option.",
    options=("do the thing", "do nothing"),
)
SINGLE_TOKEN = Question(
    id="single",
    kind=QuestionKind.CHOICE,
    text="Pick a single-token option.",
    options=("yes", "no"),
)


# ── Production-v9 row shape ───────────────────────────────────────────────


def _v9_row(
    emitted: str,
    logprob: float,
    top: Sequence[tuple[str, float]],
    *,
    ids: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """One ``completion_probabilities`` row in the pinned v9 shape."""
    table = _DEFAULT_IDS if ids is None else ids

    def token_id(text: str) -> int:
        return table[text]

    row = {
        "id": token_id(emitted),
        "token": emitted,
        "bytes": list(emitted.encode("utf-8")),
        "logprob": logprob,
        "top_logprobs": [
            {
                "id": token_id(token),
                "token": token,
                "bytes": list(token.encode("utf-8")),
                "logprob": token_logprob,
            }
            for token, token_logprob in top
        ],
    }
    return row


def _main_meta() -> dict[str, Any]:
    """Three rows: choice (renormalizes 0.5/0.2/0.1 -> sum 0.8), score, noul."""
    return {
        "role": ROLE,
        "transport": "batch",
        "tokens": 3,
        "completion_reason": "stop",
        "completion_probabilities": [
            _v9_row(
                "blue",
                math.log(0.5),
                [("blue", math.log(0.5)), ("red", math.log(0.2)), ("green", math.log(0.1))],
            ),
            _v9_row(
                "2",
                math.log(0.6),
                [("2", math.log(0.6)), ("1", math.log(0.25)), ("0", math.log(0.1))],
            ),
            _v9_row(
                "true",
                math.log(0.9),
                [("true", math.log(0.9)), ("false", math.log(0.1))],
            ),
        ],
    }


def _by_id(result: DecisionResult) -> dict[str, Any]:
    return {decision.question_id: decision for decision in result.decisions}


def _run(primitives: _FakePrimitives, questions: Sequence[Question], **kwargs):
    return run_typed_decisions_native(
        primitives,
        state=STATE,
        questions=questions,
        role=ROLE,
        tokenize_fn=kwargs.pop("tokenize_fn", _FakeTokenizer()),
        **kwargs,
    )


_MAIN_GRAMMAR = (
    "root ::= position-0 position-1 position-2\n"
    "position-0 ::= <[1000]> | <[1001]> | <[1002]> | <[1003]> | <[1004]> | <[1005]>\n"
    "position-1 ::= <[1006]> | <[1007]> | <[1008]> | <[1009]> | <[1010]> | <[1011]> | <[1012]> | <[1013]>\n"
    "position-2 ::= <[1014]> | <[1015]> | <[1016]> | <[1017]>\n"
)


# ── 1. Pinned row shape, slicing, renormalization, argmax ─────────────────


class TestNativeSlicing:
    def test_production_v9_row_shape_is_pinned(self):
        row = _v9_row(
            "blue",
            math.log(0.5),
            [("blue", math.log(0.5)), ("red", math.log(0.2)), ("green", math.log(0.1))],
        )

        assert set(row) == {"id", "token", "bytes", "logprob", "top_logprobs"}
        assert isinstance(row["id"], int)
        assert isinstance(row["bytes"], list)
        for entry in row["top_logprobs"]:
            assert set(entry) == {"id", "token", "bytes", "logprob"}

        primitives = _FakePrimitives("", meta={"completion_probabilities": [row]})
        question = Question(
            id="colour",
            kind=QuestionKind.CHOICE,
            text="Pick a colour.",
            options=("red", "blue", "green"),
        )

        result = _run(primitives, [question])

        assert len(result.decisions) == 1
        assert result.failures == ()

    def test_slice_is_renormalized_and_argmax_is_reported(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = _run(primitives, QUESTIONS)

        assert isinstance(result, DecisionResult)
        assert result.mode == "native"
        assert result.raw_text == ""
        assert result.failures == ()
        decisions = _by_id(result)
        assert set(decisions) == {"choice", "score", "noul"}

        # choice: candidate weights 0.5/0.2/0.1 (sum 0.8) -> renormalized.
        choice = decisions["choice"]
        assert choice.value == "blue"
        assert dict(choice.probabilities) == {
            "red": pytest.approx(0.25),
            "blue": pytest.approx(0.625),
            "green": pytest.approx(0.125),
        }
        assert sum(choice.probabilities.values()) == pytest.approx(1.0)
        assert choice.confidence == pytest.approx(0.4375)
        assert choice.token_logprob == pytest.approx(math.log(0.5))
        assert choice.mode == "native"

        # score: level 3 absent from the top-K -> explicit zero, no default.
        score = decisions["score"]
        assert score.value == 2
        expected_score = {0: 0.1 / 0.95, 1: 0.25 / 0.95, 2: 0.6 / 0.95, 3: 0.0}
        assert dict(score.probabilities) == {
            level: pytest.approx(probability) for level, probability in expected_score.items()
        }
        assert sum(score.probabilities.values()) == pytest.approx(1.0)
        assert score.confidence == pytest.approx(score_confidence(expected_score))
        assert score.token_logprob == pytest.approx(math.log(0.6))

        # noul: bool value, "true"/"false" labels.
        noul = decisions["noul"]
        assert noul.value is True
        assert dict(noul.probabilities) == {
            "true": pytest.approx(0.9),
            "false": pytest.approx(0.1),
        }
        assert noul.confidence == pytest.approx(choice_confidence({"true": 0.9, "false": 0.1}))
        assert noul.token_logprob == pytest.approx(math.log(0.9))

    def test_legacy_content_probs_shape_is_accepted(self):
        meta = {
            "completion_probabilities": [
                {
                    "content": "red",
                    "probs": [
                        {"tok_str": "red", "prob": 0.6},
                        {"tok_str": "blue", "prob": 0.3},
                        {"tok_str": "green", "prob": 0.1},
                    ],
                }
            ]
        }
        primitives = _FakePrimitives("", meta=meta)
        question = Question(
            id="colour",
            kind=QuestionKind.CHOICE,
            text="Pick a colour.",
            options=("red", "blue", "green"),
        )

        result = _run(primitives, [question])

        decision = result.decisions[0]
        assert decision.value == "red"
        assert dict(decision.probabilities) == {
            "red": pytest.approx(0.6),
            "blue": pytest.approx(0.3),
            "green": pytest.approx(0.1),
        }
        assert decision.token_logprob == pytest.approx(math.log(0.6))

    def test_post_sampling_top_probs_shape_is_accepted(self):
        meta = {
            "completion_probabilities": [
                {
                    "token": "false",
                    "prob": 0.7,
                    "top_probs": [
                        {"token": "false", "prob": 0.7},
                        {"token": "true", "prob": 0.3},
                    ],
                }
            ]
        }
        primitives = _FakePrimitives("", meta=meta)
        question = Question(id="flag", kind=QuestionKind.NOUL, text="Ship it?")

        result = _run(primitives, [question])

        decision = result.decisions[0]
        assert decision.value is False
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.3),
            "false": pytest.approx(0.7),
        }
        assert decision.token_logprob == pytest.approx(math.log(0.7))


# ── 2. Call contract: grammar, n_probs, n_tokens, determinism ─────────────


class TestNativeCallContract:
    def test_captured_kwargs_carry_grammar_and_probability_capture(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        _run(primitives, QUESTIONS)

        assert len(primitives.calls) == 1
        call = primitives.calls[0]
        assert call["role"] == ROLE
        assert call["n_tokens"] == 3
        # score contributes the most alternatives: 4 labels x 2 variants + buffer.
        assert call["n_probs"] == 12
        assert call["temperature"] == 0.0
        assert call["seed"] == 0
        assert "json_schema" not in call
        assert call["grammar"] == _MAIN_GRAMMAR

    def test_n_probs_accounts_for_tokenized_alternatives(self):
        # yes: bare + spaced single; no: bare single only -> 3 alternatives.
        vocab = {"yes": (41,), " yes": (42,), "no": (43,)}
        tokenizer = _FakeTokenizer(vocab)
        meta = {
            "completion_probabilities": [
                _v9_row(
                    "yes",
                    math.log(0.8),
                    [("yes", math.log(0.8)), ("no", math.log(0.2))],
                    ids={"yes": 41, "no": 43},
                )
            ]
        }
        primitives = _FakePrimitives("", meta=meta)

        _run(primitives, [SINGLE_TOKEN], tokenize_fn=tokenizer)

        call = primitives.calls[0]
        assert call["n_probs"] == 7  # max alternatives (3) + buffer 4
        assert call["grammar"] == "root ::= position-0\nposition-0 ::= <[41]> | <[42]> | <[43]>\n"

    def test_prompt_is_deterministic_and_hashed(self):
        first = _FakePrimitives("", meta=_main_meta())
        second = _FakePrimitives("", meta=_main_meta())

        first_result = _run(first, QUESTIONS)
        second_result = _run(second, QUESTIONS)

        prompt = first.calls[0]["prompt"]
        assert prompt == second.calls[0]["prompt"]
        assert first_result.prompt_sha256 == second_result.prompt_sha256
        assert first_result.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert "candidates: red | blue | green" in prompt
        assert prompt.index("id=choice") < prompt.index("id=score") < prompt.index("id=noul")

    def test_explicit_n_probs_is_forwarded_and_capped(self):
        low = _FakePrimitives("", meta=_main_meta())
        high = _FakePrimitives("", meta=_main_meta())

        _run(low, QUESTIONS, n_probs=7)
        _run(high, QUESTIONS, n_probs=1000)

        assert low.calls[0]["n_probs"] == 7
        assert high.calls[0]["n_probs"] == 128

    def test_non_positive_n_probs_is_rejected(self):
        with pytest.raises(ValueError, match="n_probs must be >= 1"):
            _run(_FakePrimitives("", meta=_main_meta()), QUESTIONS, n_probs=0)

    def test_explicit_n_tokens_is_forwarded(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        _run(primitives, QUESTIONS, n_tokens=16)

        assert primitives.calls[0]["n_tokens"] == 16


# ── 3. Tokenizer-aware eligibility and grammar ────────────────────────────


class TestTokenizedEligibility:
    def test_each_candidate_binds_to_its_single_token_variant(self):
        # true: only the bare form is one token; false: only the spaced form.
        vocab = {"true": (11,), " false": (23,)}
        tokenizer = _FakeTokenizer(vocab)
        meta = {
            "completion_probabilities": [
                _v9_row(
                    " false",
                    math.log(0.7),
                    [(" false", math.log(0.7)), ("true", math.log(0.3))],
                    ids={" false": 23, "true": 11},
                )
            ]
        }
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, [NOUL], tokenize_fn=tokenizer)

        call = primitives.calls[0]
        assert call["grammar"] == "root ::= position-0\nposition-0 ::= <[11]> | <[23]>\n"
        assert call["n_probs"] == 6  # 2 alternatives + buffer 4
        decision = result.decisions[0]
        assert decision.value is False
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.3),
            "false": pytest.approx(0.7),
        }
        assert decision.token_logprob == pytest.approx(math.log(0.7))
        # The multi-token variants were probed, not guessed.
        assert " true" in tokenizer.calls
        assert "false" in tokenizer.calls

    def test_two_id_candidates_are_deferred_to_json_mode(self):
        vocab = {
            "true": (11,),
            " true": (12,),
            "false": (21,),
            " false": (22,),
            "lock": (31, 32),
            " lock": (33, 34),
            "unlock": (35, 36),
            " unlock": (37, 38),
        }
        tokenizer = _FakeTokenizer(vocab)
        choice = Question(
            id="lock", kind=QuestionKind.CHOICE, text="Lock it?", options=("lock", "unlock")
        )
        noul = Question(id="confirm", kind=QuestionKind.NOUL, text="Confirm?")
        meta = {
            "completion_probabilities": [
                _v9_row(
                    " true",
                    math.log(0.8),
                    [(" true", math.log(0.8)), ("false", math.log(0.2))],
                    ids={" true": 12, "false": 21},
                )
            ]
        }
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, [choice, noul], tokenize_fn=tokenizer)

        # Only the single-token noul question entered the native batch, in order.
        call = primitives.calls[0]
        assert call["n_tokens"] == 1
        assert call["grammar"] == (
            "root ::= position-0\nposition-0 ::= <[11]> | <[12]> | <[21]> | <[22]>\n"
        )
        assert call["n_probs"] == 8
        assert [decision.question_id for decision in result.decisions] == ["confirm"]
        # id-based slicing: the spaced " true" variant carries the 0.8.
        confirm = result.decisions[0]
        assert confirm.value is True
        assert dict(confirm.probabilities) == {
            "true": pytest.approx(0.8),
            "false": pytest.approx(0.2),
        }
        assert confirm.token_logprob == pytest.approx(math.log(0.8))

        failure = result.failures[0]
        assert failure.reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "lock" in failure.detail and "unlock" in failure.detail
        assert "two tokens" not in failure.detail  # never a fabricated reason

    def test_token_id_collision_between_labels_is_unsupported(self):
        vocab = {"red": (5,), " red": (6,), "blue": (5,), " blue": (6,)}
        tokenizer = _FakeTokenizer(vocab)
        question = Question(
            id="colour", kind=QuestionKind.CHOICE, text="Pick.", options=("red", "blue")
        )
        primitives = _FakePrimitives("", meta=_main_meta())

        result = _run(primitives, [question], tokenize_fn=tokenizer)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "both tokenize" in result.failures[0].detail


# ── 4. Multi-token candidates -> JSON-mode fallback failures ──────────────


class TestMultiTokenFallback:
    def test_unsupported_question_is_excluded_and_recorded(self):
        questions = (SINGLE_TOKEN, MULTI_TOKEN, NOUL)
        meta = {
            "completion_probabilities": [
                _v9_row("yes", math.log(0.8), [("yes", math.log(0.8)), ("no", math.log(0.2))]),
                _v9_row(
                    "false",
                    math.log(0.6),
                    [("false", math.log(0.6)), ("true", math.log(0.4))],
                ),
            ]
        }
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, questions)

        assert [decision.question_id for decision in result.decisions] == ["single", "noul"]
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "multi" in failure.detail
        assert "JSON mode" in failure.detail

        call = primitives.calls[0]
        assert call["n_tokens"] == 2
        assert call["grammar"] == (
            "root ::= position-0 position-1\n"
            "position-0 ::= <[1018]> | <[1019]> | <[1020]> | <[1021]>\n"
            "position-1 ::= <[1014]> | <[1015]> | <[1016]> | <[1017]>\n"
        )
        assert "do the thing" not in call["grammar"]
        assert call["n_probs"] == 8  # max alternatives (4) + buffer 4

    def test_all_unsupported_questions_make_no_call_at_all(self):
        primitives = _FakePrimitives("", meta=_main_meta())
        other_multi = Question(
            id="multi-2",
            kind=QuestionKind.CHOICE,
            text="Pick another multi-token option.",
            options=("do the other thing", "do nothing"),
        )
        questions = (MULTI_TOKEN, other_multi)

        result = _run(primitives, questions)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.raw_text == ""
        assert result.elapsed_ms == 0.0
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
        ]
        assert len(result.prompt_sha256) == 64

    def test_partial_piece_label_is_ineligible_before_generation(self):
        # The TD-1a live failure: the grammar forced a partial piece ("fal")
        # of a two-token "false". With real tokenization the label is excluded
        # up front and the partial piece is never generated or accepted.
        vocab = {"true": (11,), " true": (12,), "false": (901, 902), " false": (903, 904)}
        tokenizer = _FakeTokenizer(vocab)
        meta = {
            "completion_probabilities": [
                {
                    "id": 901,
                    "token": "fal",
                    "bytes": [102, 97, 108],
                    "logprob": math.log(0.9),
                    "top_logprobs": [
                        {"id": 901, "token": "fal", "bytes": [], "logprob": math.log(0.9)}
                    ],
                }
            ]
        }
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, [NOUL], tokenize_fn=tokenizer)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "false" in result.failures[0].detail
        assert REASON_NATIVE_UNKNOWN_CANDIDATE not in {
            failure.reason for failure in result.failures
        }


# ── 5. Tokenizer unavailable -> fail closed, zero model calls ─────────────


class TestTokenizerUnavailable:
    def test_unresolvable_tokenizer_fails_every_question_without_a_call(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.raw_text == ""
        assert result.elapsed_ms == 0.0
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
        ] * 3
        assert all(
            "no tokenizer could be resolved" in failure.detail for failure in result.failures
        )
        assert len(result.prompt_sha256) == 64

    def test_injected_tokenizer_returning_none_fails_closed(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = _run(primitives, QUESTIONS, tokenize_fn=lambda text: None)

        assert primitives.calls == []
        assert result.decisions == ()
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
        ] * 3
        assert all("tokenizer returned no ids" in failure.detail for failure in result.failures)

    def test_raising_tokenizer_is_treated_as_unavailable(self):
        def boom(text: str) -> list[int]:
            raise RuntimeError("tokenizer exploded")

        primitives = _FakePrimitives("", meta=_main_meta())

        result = _run(primitives, QUESTIONS, tokenize_fn=boom)

        assert primitives.calls == []
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
        ] * 3

    def test_unavailable_question_does_not_block_eligible_questions(self):
        tokenizer = _FakeTokenizer(failing={"true", " true"})
        meta = {
            "completion_probabilities": [
                _v9_row("yes", math.log(0.8), [("yes", math.log(0.8)), ("no", math.log(0.2))]),
            ]
        }
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, (SINGLE_TOKEN, NOUL), tokenize_fn=tokenizer)

        assert len(primitives.calls) == 1
        call = primitives.calls[0]
        assert call["n_tokens"] == 1
        assert call["grammar"] == (
            "root ::= position-0\nposition-0 ::= <[1018]> | <[1019]> | <[1020]> | <[1021]>\n"
        )
        assert [decision.question_id for decision in result.decisions] == ["single"]
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE
        ]
        assert "noul" in result.failures[0].detail


# ── 6. Default resolver: role backend base URL -> /tokenize ───────────────


class _BackendConfig:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


class _InnerBackend:
    def __init__(self, base_url: str) -> None:
        self.config = _BackendConfig(base_url)


class _WrappedBackend:
    """CachingBackend-shaped wrapper (``.backend.config.base_url``)."""

    def __init__(self, base_url: str) -> None:
        self.backend = _InnerBackend(base_url)


class _ResolverPrimitives(_FakePrimitives):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._backends = {"worker": _WrappedBackend(base_url)}


def _install_fake_http_client(monkeypatch, responder):
    """Replace ``httpx.Client`` with a recorder; returns the created clients."""
    created: list[Any] = []

    class _Response:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return self._payload

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.requests: list[tuple[str, dict[str, Any]]] = []
            self.closed = False
            created.append(self)

        def post(self, url: str, json: dict[str, Any]):
            self.requests.append((url, json))
            return _Response(responder(json))

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("src.typed_decisions.native.httpx.Client", _Client)
    return created


class TestDefaultTokenizerResolver:
    def test_role_backend_url_is_used_and_owned_client_is_closed(self, monkeypatch):
        ids = {"true": 7, " true": 8, "false": 9, " false": 10}

        def responder(payload: dict[str, Any]) -> dict[str, Any]:
            text = payload.get("content", "")
            return {"tokens": [] if text == "" else [ids[text]]}

        created = _install_fake_http_client(monkeypatch, responder)
        meta = {
            "completion_probabilities": [
                _v9_row(
                    " false",
                    math.log(0.7),
                    [(" false", math.log(0.7)), ("true", math.log(0.3))],
                    ids=ids,
                )
            ]
        }
        primitives = _ResolverPrimitives("http://test-host:8123", meta=meta)

        result = run_typed_decisions_native(
            primitives,
            state=STATE,
            questions=[NOUL],
            role=ROLE,
            # tokenize_fn omitted: the default resolver must do the work.
        )

        assert len(created) == 1
        client = created[0]
        assert client.requests[0][0] == "http://test-host:8123/tokenize"
        assert client.requests[0][1]["add_special"] is False
        assert client.requests[0][1]["content"] == ""  # probe
        probe_and_candidates = {request[1]["content"] for request in client.requests}
        assert {"", "true", " true", "false", " false"} <= probe_and_candidates
        assert client.closed is True  # the owned tokenizer is closed after the run

        assert primitives.calls[0]["grammar"] == (
            "root ::= position-0\nposition-0 ::= <[7]> | <[8]> | <[9]> | <[10]>\n"
        )
        decision = result.decisions[0]
        assert decision.value is False
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.3),
            "false": pytest.approx(0.7),
        }

    def test_unreachable_tokenize_endpoint_fails_closed(self, monkeypatch):
        def responder(payload: dict[str, Any]) -> dict[str, Any]:
            raise ConnectionError("connection refused")

        created = _install_fake_http_client(monkeypatch, responder)
        primitives = _ResolverPrimitives("http://test-host:8123", meta=_main_meta())

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert primitives.calls == []  # zero model calls
        assert result.decisions == ()
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
        ] * 3
        assert created[0].closed is True

    def test_server_urls_are_used_when_no_backend_is_registered(self, monkeypatch):
        ids = {"true": 7, " true": 8, "false": 9, " false": 10}

        def responder(payload: dict[str, Any]) -> dict[str, Any]:
            text = payload.get("content", "")
            return {"tokens": [] if text == "" else [ids[text]]}

        created = _install_fake_http_client(monkeypatch, responder)
        primitives = _FakePrimitives("", meta=_main_meta())
        primitives.server_urls = {"worker": "http://fallback-host:9000,http://other:9001"}

        run_typed_decisions_native(primitives, state=STATE, questions=[NOUL], role=ROLE)

        assert created[0].requests[0][0] == "http://fallback-host:9000/tokenize"


# ── 7. Typed failure paths (never a default) ──────────────────────────────


class TestNativeFailurePaths:
    def test_emitted_token_outside_candidates_is_typed_failure(self):
        meta = _main_meta()
        meta["completion_probabilities"][0] = _v9_row(
            "purple",
            math.log(0.9),
            [("purple", math.log(0.9)), ("red", math.log(0.1))],
        )
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, QUESTIONS)

        decisions = _by_id(result)
        assert set(decisions) == {"score", "noul"}
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "purple" in result.failures[0].detail
        assert "id=" in result.failures[0].detail

    def test_idless_row_outside_candidates_still_fails(self):
        meta = {
            "completion_probabilities": [
                {
                    "content": "purple",
                    "probs": [{"tok_str": "purple", "prob": 0.9}, {"tok_str": "red", "prob": 0.1}],
                }
            ]
        }
        primitives = _FakePrimitives("", meta=meta)
        question = Question(
            id="colour", kind=QuestionKind.CHOICE, text="Pick.", options=("red", "blue")
        )

        result = _run(primitives, [question])

        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "purple" in result.failures[0].detail

    def test_id_bearing_row_does_not_text_match_ids_less_entries(self):
        # The text fallback is scoped to rows that lack an id altogether.
        meta = {
            "completion_probabilities": [
                {
                    "id": _DEFAULT_IDS["red"],
                    "token": "red",
                    "bytes": [114, 101, 100],
                    "logprob": math.log(0.9),
                    "probs": [{"tok_str": "red", "prob": 0.9}],
                }
            ]
        }
        primitives = _FakePrimitives("", meta=meta)
        question = Question(
            id="colour", kind=QuestionKind.CHOICE, text="Pick.", options=("red", "blue")
        )

        result = _run(primitives, [question])

        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "none of the declared candidate tokens" in result.failures[0].detail

    def test_no_candidate_token_in_the_capture_is_failure_not_uniform(self):
        meta = _main_meta()
        meta["completion_probabilities"][0] = _v9_row(
            "blue",
            math.log(0.9),
            [("x", math.log(0.5)), ("y", math.log(0.5))],
        )
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, QUESTIONS)

        assert "choice" not in _by_id(result)
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "none of the declared candidate tokens" in result.failures[0].detail

    def test_missing_meta_yields_typed_failures_for_every_position(self):
        primitives = _FakePrimitives("", meta=None)

        result = _run(primitives, QUESTIONS)

        assert result.decisions == ()
        assert len(result.failures) == 3
        assert all(failure.reason == REASON_NATIVE_UNKNOWN_CANDIDATE for failure in result.failures)
        assert all(
            "no completion_probabilities row" in failure.detail for failure in result.failures
        )

    def test_short_row_count_fails_only_the_missing_positions(self):
        meta = _main_meta()
        meta["completion_probabilities"] = meta["completion_probabilities"][:2]
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, QUESTIONS)

        assert [decision.question_id for decision in result.decisions] == ["choice", "score"]
        assert len(result.failures) == 1
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "noul" in result.failures[0].detail

    def test_transport_error_short_circuits_and_ignores_stale_meta(self):
        primitives = _FakePrimitives("[ERROR: connection refused]", meta=_main_meta())

        result = _run(primitives, QUESTIONS)

        assert len(primitives.calls) == 1  # no retry
        assert result.decisions == ()
        assert result.failures[0].reason == "transport_error"
        assert result.raw_text == "[ERROR: connection refused]"

    def test_transport_failure_precedes_unsupported_question_failures(self):
        questions = (MULTI_TOKEN, SINGLE_TOKEN)
        primitives = _FakePrimitives("[ERROR: timeout]", meta={"completion_probabilities": []})

        result = _run(primitives, questions)

        assert [failure.reason for failure in result.failures] == [
            "transport_error",
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
        ]


# ── 7. Runner dispatch and JSON-mode non-regression ───────────────────────


class TestRunnerDispatch:
    def test_runner_native_mode_dispatches_to_native_runner(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = run_typed_decisions(
            primitives,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            mode="native",
            tokenize_fn=_FakeTokenizer(),
        )

        assert result.mode == "native"
        assert [decision.value for decision in result.decisions] == ["blue", 2, True]
        assert primitives.calls[0]["grammar"] == _MAIN_GRAMMAR
        assert "json_schema" not in primitives.calls[0]

    def test_runner_native_mode_forwards_explicit_n_tokens(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        run_typed_decisions(
            primitives,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            mode="native",
            n_tokens=11,
            tokenize_fn=_FakeTokenizer(),
        )

        assert primitives.calls[0]["n_tokens"] == 11

    def test_runner_native_mode_fails_closed_without_a_tokenizer(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = run_typed_decisions(
            primitives, state=STATE, questions=QUESTIONS, role=ROLE, mode="native"
        )

        assert primitives.calls == []
        assert result.decisions == ()
        assert all(
            failure.reason == REASON_NATIVE_TOKENIZER_UNAVAILABLE for failure in result.failures
        )

    def test_json_mode_still_uses_the_schema_path_only(self):
        response = (
            '{"answers": {"noul": {"noul": true, "probabilities": '
            '{"true": 0.8, "false": 0.2}, "confidence": 0.8}}}'
        )
        primitives = _FakePrimitives(response, meta=_main_meta())

        result = run_typed_decisions(
            primitives, state=STATE, questions=[NOUL], role=ROLE, mode="json"
        )

        assert result.mode == "json"
        assert result.decisions[0].value is True
        call = primitives.calls[0]
        assert "json_schema" in call
        assert "grammar" not in call
        assert "n_probs" not in call
