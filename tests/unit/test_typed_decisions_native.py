"""Unit tests for the TD-1a native candidate-scoring path.

Mirrors the fake-primitives pattern of ``tests/unit/test_typed_decisions.py``
(one canned response, captured call kwargs) but adds the instance-level
``_last_inference_meta`` that carries synthetic ``completion_probabilities``
rows. The primary row shape pinned here is the production-consolidated-v9
``/completion`` shape emitted by
``tools/server/server-task.cpp::probs_vector_to_json`` with
``post_sampling_probs=false``::

    {"id": int, "token": str, "bytes": [int],
     "logprob": float, "top_logprobs": [{"id", "token", "bytes", "logprob"}, ...]}

The legacy ``{"content", "probs": [{"tok_str", "prob"}]}`` shape and the
``top_probs`` linear-probability variant are pinned alongside it. No
model/server call.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
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
    REASON_NATIVE_UNKNOWN_CANDIDATE,
    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
)

ROLE = "worker"
STATE = "unit-test state: native candidate scoring over a three-question catalogue."

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


class _FakePrimitives:
    """Canned-response stand-in for ``LLMPrimitives`` with inference meta.

    ``responses`` is consumed in order; the last entry repeats. The meta is
    instance-level exactly as in ``src/llm_primitives/inference.py``, so the
    tests can also prove that a transport failure does not fall back to stale
    probability rows.
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


# ── Production-v9 row shape ───────────────────────────────────────────────


def _v9_row(
    emitted: str,
    logprob: float,
    top: Sequence[tuple[str, float]],
) -> dict[str, Any]:
    """One ``completion_probabilities`` row in the pinned v9 shape."""
    row = {
        "id": 1001,
        "token": emitted,
        "bytes": list(emitted.encode("utf-8")),
        "logprob": logprob,
        "top_logprobs": [
            {
                "id": 2000 + index,
                "token": token,
                "bytes": list(token.encode("utf-8")),
                "logprob": token_logprob,
            }
            for index, (token, token_logprob) in enumerate(top)
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

        result = run_typed_decisions_native(
            primitives, state=STATE, questions=[question], role=ROLE
        )

        assert len(result.decisions) == 1
        assert result.failures == ()

    def test_slice_is_renormalized_and_argmax_is_reported(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

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

        result = run_typed_decisions_native(
            primitives, state=STATE, questions=[question], role=ROLE
        )

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

        result = run_typed_decisions_native(
            primitives, state=STATE, questions=[question], role=ROLE
        )

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

        run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert len(primitives.calls) == 1
        call = primitives.calls[0]
        assert call["role"] == ROLE
        assert call["n_tokens"] == 3
        assert call["n_probs"] == 8  # max candidates (4, score) + buffer 4
        assert call["temperature"] == 0.0
        assert call["seed"] == 0
        assert "json_schema" not in call
        assert call["grammar"] == (
            "root ::= position-0 position-1 position-2\n"
            'position-0 ::= "red" | "blue" | "green"\n'
            'position-1 ::= "0" | "1" | "2" | "3"\n'
            'position-2 ::= "true" | "false"\n'
        )

    def test_prompt_is_deterministic_and_hashed(self):
        first = _FakePrimitives("", meta=_main_meta())
        second = _FakePrimitives("", meta=_main_meta())

        first_result = run_typed_decisions_native(
            first, state=STATE, questions=QUESTIONS, role=ROLE
        )
        second_result = run_typed_decisions_native(
            second, state=STATE, questions=QUESTIONS, role=ROLE
        )

        prompt = first.calls[0]["prompt"]
        assert prompt == second.calls[0]["prompt"]
        assert first_result.prompt_sha256 == second_result.prompt_sha256
        assert first_result.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert "candidates: red | blue | green" in prompt
        assert prompt.index("id=choice") < prompt.index("id=score") < prompt.index("id=noul")

    def test_explicit_n_probs_is_forwarded_and_capped(self):
        low = _FakePrimitives("", meta=_main_meta())
        high = _FakePrimitives("", meta=_main_meta())

        run_typed_decisions_native(low, state=STATE, questions=QUESTIONS, role=ROLE, n_probs=7)
        run_typed_decisions_native(high, state=STATE, questions=QUESTIONS, role=ROLE, n_probs=1000)

        assert low.calls[0]["n_probs"] == 7
        assert high.calls[0]["n_probs"] == 128

    def test_non_positive_n_probs_is_rejected(self):
        with pytest.raises(ValueError, match="n_probs must be >= 1"):
            run_typed_decisions_native(
                _FakePrimitives("", meta=_main_meta()),
                state=STATE,
                questions=QUESTIONS,
                role=ROLE,
                n_probs=0,
            )

    def test_explicit_n_tokens_is_forwarded(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        run_typed_decisions_native(
            primitives, state=STATE, questions=QUESTIONS, role=ROLE, n_tokens=16
        )

        assert primitives.calls[0]["n_tokens"] == 16


# ── 3. Multi-token candidates -> JSON-mode fallback failures ──────────────


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

        result = run_typed_decisions_native(primitives, state=STATE, questions=questions, role=ROLE)

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
            'position-0 ::= "yes" | "no"\n'
            'position-1 ::= "true" | "false"\n'
        )
        assert "do the thing" not in call["grammar"]
        assert call["n_probs"] == 6  # max candidates (2/2) + buffer 4

    def test_all_unsupported_questions_make_no_call_at_all(self):
        primitives = _FakePrimitives("", meta=_main_meta())
        other_multi = Question(
            id="multi-2",
            kind=QuestionKind.CHOICE,
            text="Pick another multi-token option.",
            options=("do the other thing", "do nothing"),
        )
        questions = (MULTI_TOKEN, other_multi)

        result = run_typed_decisions_native(primitives, state=STATE, questions=questions, role=ROLE)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.raw_text == ""
        assert result.elapsed_ms == 0.0
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
        ]
        assert len(result.prompt_sha256) == 64


# ── 4. Typed failure paths (never a default) ──────────────────────────────


class TestNativeFailurePaths:
    def test_emitted_token_outside_candidates_is_typed_failure(self):
        meta = _main_meta()
        meta["completion_probabilities"][0] = _v9_row(
            "purple",
            math.log(0.9),
            [("purple", math.log(0.9)), ("red", math.log(0.1))],
        )
        primitives = _FakePrimitives("", meta=meta)

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        decisions = _by_id(result)
        assert set(decisions) == {"score", "noul"}
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "purple" in result.failures[0].detail

    def test_no_candidate_token_in_the_capture_is_failure_not_uniform(self):
        meta = _main_meta()
        meta["completion_probabilities"][0] = _v9_row(
            "blue",
            math.log(0.9),
            [("x", math.log(0.5)), ("y", math.log(0.5))],
        )
        primitives = _FakePrimitives("", meta=meta)

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert "choice" not in _by_id(result)
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "none of the declared candidate tokens" in result.failures[0].detail

    def test_missing_meta_yields_typed_failures_for_every_position(self):
        primitives = _FakePrimitives("", meta=None)

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

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

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert [decision.question_id for decision in result.decisions] == ["choice", "score"]
        assert len(result.failures) == 1
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "noul" in result.failures[0].detail

    def test_transport_error_short_circuits_and_ignores_stale_meta(self):
        primitives = _FakePrimitives("[ERROR: connection refused]", meta=_main_meta())

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert len(primitives.calls) == 1  # no retry
        assert result.decisions == ()
        assert result.failures[0].reason == "transport_error"
        assert result.raw_text == "[ERROR: connection refused]"

    def test_transport_failure_precedes_unsupported_question_failures(self):
        questions = (MULTI_TOKEN, SINGLE_TOKEN)
        primitives = _FakePrimitives("[ERROR: timeout]", meta={"completion_probabilities": []})

        result = run_typed_decisions_native(primitives, state=STATE, questions=questions, role=ROLE)

        assert [failure.reason for failure in result.failures] == [
            "transport_error",
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
        ]


# ── 5. Runner dispatch and JSON-mode non-regression ───────────────────────


class TestRunnerDispatch:
    def test_runner_native_mode_dispatches_to_native_runner(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = run_typed_decisions(
            primitives, state=STATE, questions=QUESTIONS, role=ROLE, mode="native"
        )

        assert result.mode == "native"
        assert [decision.value for decision in result.decisions] == ["blue", 2, True]
        assert primitives.calls[0]["grammar"] is not None
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
        )

        assert primitives.calls[0]["n_tokens"] == 11

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
