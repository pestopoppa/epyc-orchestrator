"""TD-31: prefix-stable ordering is enforced on the real prompt builder; per-kind rollup."""
from src.typed_decisions.prefill_metrics import (
    prefix_ordering_violations, rollup_by_kind, shared_prefix_chars,
)
from src.typed_decisions.runner import _build_prompt
from src.typed_decisions.schema import build_response_schema
from src.typed_decisions.types import Question, QuestionKind


def _q(text):
    return Question(id="q1", kind=QuestionKind.NOUL, text=text)


def _prompt(state, text):
    qs = [_q(text)]
    return _build_prompt(state, qs, build_response_schema(qs))


def test_real_builder_orders_head_state_then_questions():
    p = _prompt("shared state", "is it ready?")
    assert prefix_ordering_violations(p) == []
    assert prefix_ordering_violations("STATE:\nx\nQUESTION CATALOG:\ny") != []


def test_question_text_change_keeps_head_and_state_in_the_shared_prefix():
    a, b = _prompt("same state", "first question"), _prompt("same state", "second question")
    cut = shared_prefix_chars(a, b)
    assert cut >= a.index("QUESTION CATALOG:")
    c = _prompt("other state", "first question")
    assert shared_prefix_chars(a, c) < a.index("STATE:") + len("STATE:\nsame state")


def test_rollup_keeps_unknown_unknown_and_splits_by_kind():
    calls = [
        {"kind": "choice", "prompt_ms": 10.0, "prompt_n": 20, "cache_n": 80},
        {"kind": "choice", "prompt_ms": 30.0, "prompt_n": 100, "cache_n": 0},
        {"kind": "noul", "prompt_ms": None, "prompt_n": None, "cache_n": None},
    ]
    out = rollup_by_kind(calls)
    assert out["choice"]["prefix_reuse_rate"] == 80 / 200 and out["choice"]["prompt_ms_total"] == 40.0
    assert out["noul"]["prefix_reuse_rate"] is None and out["noul"]["cache_n_total"] is None
