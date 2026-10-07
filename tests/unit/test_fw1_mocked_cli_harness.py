"""FW-1's bounded mocked authoring and worker-routing example.

This is a bounded, non-runtime harness: the mock UI supplies the one documented
example, while the CLI path composes the existing error classifier, typed
decision runner, and gate predicates. It never starts a server, calls a model,
writes a ledger, or creates a ClaimTuple. The in-memory hook payload is marked
synthetic and non-grading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from src.escalation import ErrorCategory
from src.graph.decision_gates import _should_escalate, _should_retry
from src.graph.error_classifier import classify_error
from src.graph.think_harder import _should_think_harder
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.types import Question, QuestionKind


CATEGORY_OPTIONS = tuple(
    category.value for category in ErrorCategory if category is not ErrorCategory.UNKNOWN
)
QUESTIONS = (
    Question(
        id="q_cat",
        kind=QuestionKind.CHOICE,
        text="Classify this previously-unrecognized worker error.",
        options=CATEGORY_OPTIONS,
    ),
    Question(
        id="q_done",
        kind=QuestionKind.NOUL,
        text="Does the prior output already contain a complete task answer?",
    ),
)


@dataclass(frozen=True)
class _MockWorkflow:
    """Small explicit graph from the FW-1 sketch, returned by a mocked GUI."""

    workflow_id: str = "fw1-worker-failure-example"
    start: str = "C0"
    nodes: tuple[str, ...] = (
        "C0", "F1", "G1", "G2", "G3", "G4", "V1", "WORKER",
        "End_success", "End_failure",
    )
    edges: tuple[tuple[str, str, str], ...] = (
        ("C0", "F1", "classified-error"),
        ("F1", "G1", "parsed"),
        ("F1", "G1", "fallback-after-parse-budget"),
        ("G1", "V1", "answer-already-complete"),
        ("G1", "G2", "answer-incomplete"),
        ("G2", "WORKER", "think-harder-available"),
        ("G2", "G3", "think-harder-unavailable"),
        ("G3", "WORKER", "retry-available"),
        ("G3", "G4", "retry-exhausted-escalation-available"),
        ("G3", "End_failure", "retry-exhausted-no-escalation"),
        ("G4", "End_failure", "approval-refused-or-budget-exhausted"),
        ("G4", "End_success", "approval-granted"),
        ("V1", "End_success", "validation-accepted"),
        ("V1", "End_failure", "validation-rejected"),
    )
    fallback_target: str = "G1"
    expensive_node: str = "WORKER"
    fuzzy_targets: tuple[str, ...] = ()  # Confidence/probability values have no edges.


class _MockGUI:
    def author(self) -> _MockWorkflow:
        return _MockWorkflow()


class _FakePrimitives:
    """Sentinel for the existing typed runner's llm_call interface; never inference."""

    def __init__(self, emissions: list[str]):
        self.emissions = list(emissions)
        self.calls: list[dict[str, Any]] = []

    def llm_call(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.emissions.pop(0)


def _emission(category: str, *, confidence: float, done: bool = False) -> str:
    probabilities = {label: 0.0 for label in CATEGORY_OPTIONS}
    probabilities[category] = confidence
    remainder = (1.0 - confidence) / (len(probabilities) - 1)
    for label in probabilities:
        if label != category:
            probabilities[label] = remainder
    return json.dumps({"answers": {
        "q_cat": {"choice": category, "probabilities": probabilities,
                  "confidence": confidence},
        "q_done": {"noul": done, "probabilities": {"true": 0.1, "false": 0.9},
                   "confidence": 0.8},
    }})


def _context(*, failures: int = 1, escalations: int = 0, max_retries: int = 2,
             max_escalations: int = 2):
    state = SimpleNamespace(
        consecutive_failures=failures,
        escalation_count=escalations,
        last_error="unfamiliar worker failure",
        role_history=[],
        think_harder_attempted=False,
        current_role="frontdoor",
        think_harder_roi_by_role={},
        turns=1,
        think_harder_cooldown_turns=0,
        think_harder_min_samples=100,
    )
    config = SimpleNamespace(
        max_retries=max_retries,
        max_escalations=max_escalations,
        no_escalate_categories={ErrorCategory.FORMAT},
    )
    return SimpleNamespace(state=state, deps=SimpleNamespace(config=config)), state


def _run_cli(*, raw_error: str, primitives: _FakePrimitives,
             capture_hook: Any | None, failures: int = 1, escalations: int = 0,
             max_retries: int = 2, max_escalations: int = 2,
             approval_granted: bool = True) -> dict[str, Any]:
    """Execute the documented UNKNOWN-residue path against existing pure seams."""
    if capture_hook is None:
        raise RuntimeError("FW-1 refuses to run without its prospective capture hook")
    workflow = _MockGUI().author()
    assert workflow.start in workflow.nodes
    assert workflow.fallback_target in workflow.nodes
    assert not workflow.fuzzy_targets
    edge_set = {(source, target) for source, target, _ in workflow.edges}
    assert all(source in workflow.nodes and target in workflow.nodes
               for source, target, _ in workflow.edges)

    initial = classify_error(raw_error)
    if initial is not ErrorCategory.UNKNOWN:
        raise ValueError("this example exercises only the UNKNOWN residue")
    context, state = _context(failures=failures, escalations=escalations,
                              max_retries=max_retries, max_escalations=max_escalations)
    result = run_typed_decisions(
        primitives,
        state=raw_error[:512],
        questions=QUESTIONS,
        role="frontdoor",
        max_retries=1,
    )
    decisions = {decision.question_id: decision for decision in result.decisions}
    category_decision = decisions.get("q_cat")
    done_decision = decisions.get("q_done")
    category = (ErrorCategory(category_decision.value)
                if category_decision is not None else ErrorCategory.UNKNOWN)
    done = bool(done_decision.value) if done_decision is not None else False

    # Fuzzy parse/retry uses B_parse only. The worker's failure budget is the
    # one increment performed by the preceding failed worker call.
    failures_before_gates = state.consecutive_failures
    prefix = (("C0", "F1"), ("F1", "G1"))
    if done:
        action, route = "V1_VALIDATE", prefix + (
            ("G1", "V1"), ("V1", "End_success")
        )
    elif _should_think_harder(context, category):
        action, route = "WORKER_THINK_HARDER", prefix + (
            ("G1", "G2"), ("G2", "WORKER")
        )
    elif _should_retry(context, category):
        action, route = "WORKER_RETRY", prefix + (
            ("G1", "G2"), ("G2", "G3"), ("G3", "WORKER")
        )
    elif _should_escalate(context, category, "coder_escalation"):
        action, route = "G4_APPROVAL", prefix + (
            ("G1", "G2"), ("G2", "G3"), ("G3", "G4")
        )
        if not approval_granted:
            action, route = "END_FAILURE_APPROVAL_REFUSED", prefix + (
                ("G1", "G2"), ("G2", "G3"), ("G3", "G4"),
                ("G4", "End_failure")
            )
        else:
            route += (("G4", "End_success"),)
    else:
        action, route = "END_FAILURE", prefix + (
            ("G1", "G2"), ("G2", "G3"), ("G3", "End_failure")
        )
    assert all(edge in edge_set for edge in route), f"undeclared workflow transition: {route}"
    assert state.consecutive_failures == failures_before_gates

    record = {
        "evidence_kind": "synthetic_mock_only",
        "workflow_id": workflow.workflow_id,
        "worker_error_category": category.value,
        "q_cat_confidence_recorded_only": (
            category_decision.confidence if category_decision is not None else None
        ),
        "q_done": done,
        "typed_parse_failures": [failure.reason for failure in result.failures],
        "gate_action": action,
        "workflow_path": [list(edge) for edge in route],
        "worker_failure_budget_before_after_fuzzy_node": [
            failures_before_gates, state.consecutive_failures
        ],
        "real_model_call": False,
        "canonical_store_authority": False,
    }
    capture_hook(record)
    return record


def _synthetic_hook(records: list[dict[str, Any]]):
    def capture(record: dict[str, Any]) -> None:
        assert record["evidence_kind"] == "synthetic_mock_only"
        assert record["canonical_store_authority"] is False
        records.append(record)
    return capture


@pytest.mark.parametrize("confidence", [0.55, 0.99], ids=["lower-confidence", "higher-confidence"])
def test_mocked_gui_cli_routes_by_typed_value_not_confidence(confidence: float):
    records: list[dict[str, Any]] = []
    primitives = _FakePrimitives([_emission("code", confidence=confidence)])
    record = _run_cli(
        raw_error="a novel tool returned an unrecognized status",
        primitives=primitives,
        capture_hook=_synthetic_hook(records),
    )

    assert classify_error("a novel tool returned an unrecognized status") is ErrorCategory.UNKNOWN
    assert len(primitives.calls) == 1
    assert primitives.calls[0]["temperature"] == 0.0
    assert record["worker_error_category"] == ErrorCategory.CODE.value
    assert record["gate_action"] == "WORKER_THINK_HARDER"
    assert record["q_cat_confidence_recorded_only"] == confidence
    assert records == [record]


def test_parse_retry_and_failure_budget_are_independent():
    records: list[dict[str, Any]] = []
    primitives = _FakePrimitives(["not-json", _emission("logic", confidence=0.8)])
    record = _run_cli(
        raw_error="a novel worker error",
        primitives=primitives,
        capture_hook=_synthetic_hook(records),
    )

    assert len(primitives.calls) == 2  # F1's one corrective retry, not a worker retry.
    assert record["typed_parse_failures"] == ["no_json"]
    assert record["worker_failure_budget_before_after_fuzzy_node"] == [1, 1]


def test_exhausted_typed_budget_uses_declared_unknown_fallback():
    records: list[dict[str, Any]] = []
    primitives = _FakePrimitives(["not-json", "not-json"])
    record = _run_cli(
        raw_error="a novel worker error",
        primitives=primitives,
        capture_hook=_synthetic_hook(records),
    )

    assert record["worker_error_category"] == ErrorCategory.UNKNOWN.value
    assert record["gate_action"] == "WORKER_THINK_HARDER"
    assert record["typed_parse_failures"] == ["no_json", "no_json"]
    assert record["worker_failure_budget_before_after_fuzzy_node"] == [1, 1]


@pytest.mark.parametrize("done", [False, True], ids=["incomplete", "complete"])
def test_q_done_routes_to_worker_or_validation(done: bool):
    primitives = _FakePrimitives([_emission("code", confidence=0.8, done=done)])
    record = _run_cli(raw_error="unrecognized worker error", primitives=primitives,
                      capture_hook=_synthetic_hook([]))
    assert record["q_done"] is done
    assert record["workflow_path"] == (
        [["C0", "F1"], ["F1", "G1"], ["G1", "V1"], ["V1", "End_success"]]
        if done else [["C0", "F1"], ["F1", "G1"], ["G1", "G2"],
                      ["G2", "WORKER"]]
    )


def test_format_failure_retries_but_never_escalates():
    primitives = _FakePrimitives([_emission("format", confidence=0.8)])
    record = _run_cli(raw_error="unrecognized worker error", primitives=primitives,
                      capture_hook=_synthetic_hook([]))
    assert record["gate_action"] == "WORKER_RETRY"
    assert record["workflow_path"] == [
        ["C0", "F1"], ["F1", "G1"], ["G1", "G2"], ["G2", "G3"],
        ["G3", "WORKER"],
    ]


def test_exhausted_retry_escalation_and_approval_end_routes():
    granted = _run_cli(raw_error="unrecognized worker error",
                       primitives=_FakePrimitives([_emission("code", confidence=0.8)]),
                       capture_hook=_synthetic_hook([]), failures=2, max_retries=2)
    refused = _run_cli(raw_error="unrecognized worker error",
                       primitives=_FakePrimitives([_emission("code", confidence=0.8)]),
                       capture_hook=_synthetic_hook([]), failures=2, max_retries=2,
                       approval_granted=False)
    capped = _run_cli(raw_error="unrecognized worker error",
                      primitives=_FakePrimitives([_emission("code", confidence=0.8)]),
                      capture_hook=_synthetic_hook([]), failures=2, max_retries=2,
                      escalations=2, max_escalations=2)
    prefix = [["C0", "F1"], ["F1", "G1"], ["G1", "G2"], ["G2", "G3"]]
    assert granted["workflow_path"] == prefix + [["G3", "G4"], ["G4", "End_success"]]
    assert refused["workflow_path"] == prefix + [["G3", "G4"], ["G4", "End_failure"]]
    assert capped["workflow_path"] == prefix + [["G3", "End_failure"]]


def test_missing_capture_hook_refuses_before_any_typed_call():
    primitives = _FakePrimitives([_emission("code", confidence=0.8)])
    with pytest.raises(RuntimeError, match="capture hook"):
        _run_cli(raw_error="a novel worker error", primitives=primitives, capture_hook=None)
    assert primitives.calls == []
