"""FW-1's bounded mocked authoring and worker-routing example.

This is a bounded, non-runtime harness: the mock UI supplies the one documented
example, while the CLI path composes the existing error classifier, typed
decision runner, and gate predicates. It never starts a server, calls a model,
writes a ledger, or creates a ClaimTuple. The in-memory hook payload is marked
synthetic and non-grading.
"""
from __future__ import annotations

import json
import hashlib
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
        "C0", "F1", "V1", "G1", "G2", "G3", "G4", "WORKER",
        "End_success", "End_failure",
    )
    node_types: tuple[tuple[str, str], ...] = (
        ("C0", "code"), ("F1", "fuzzy"), ("V1", "code"),
        ("G1", "gate"), ("G2", "gate"), ("G3", "gate"),
        ("G4", "gate"), ("WORKER", "expensive"),
        ("End_success", "terminal"), ("End_failure", "terminal"),
    )
    edges: tuple[tuple[str, str, str], ...] = (
        ("C0", "F1", "classified-error"),
        ("F1", "G1", "parsed-category-or-fallback"),
        ("F1", "V1", "q_done-true"),
        ("G1", "WORKER", "think-harder-pass"),
        ("G1", "G2", "think-harder-reject"),
        ("G2", "WORKER", "retry-pass"),
        ("G2", "G3", "retry-reject"),
        ("G3", "G4", "escalation-pass"),
        ("G3", "End_failure", "escalation-reject"),
        ("G4", "WORKER", "approval-granted"),
        ("G4", "End_failure", "approval-refused"),
        ("WORKER", "V1", "mock-worker-output"),
        ("V1", "End_success", "validation-accepted"),
        ("V1", "End_failure", "validation-rejected"),
    )
    budgets: tuple[tuple[str, int], ...] = (
        ("B_parse", 1), ("B_think", 1), ("B_retry", 2), ("B_esc", 2),
    )
    fallback_target: str = "G1"
    expensive_node: str = "WORKER"
    fuzzy_targets: tuple[str, ...] = ()  # Confidence/probability values have no edges.
    fuzzy_model_catalog_id: str = "synthetic-mock-model+fw1-questions-v1"


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
        "q_done": {"noul": done, "probabilities": (
            {"true": 0.9, "false": 0.1} if done else {"true": 0.1, "false": 0.9}
        ),
                   "confidence": 0.8},
    }})


class _MockWorker:
    """One explicit synthetic resource-step stand-in; never a model/server call."""

    def __init__(self, output: str):
        self.output = output
        self.calls: list[dict[str, str]] = []

    def run(self, *, role: str, category: ErrorCategory, action: str) -> str:
        self.calls.append({"role": role, "category": category.value, "action": action})
        return self.output


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
             approval_granted: bool = True, output_valid: bool = True,
             prior_output_valid: bool = True) -> dict[str, Any]:
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
    escalations_before_gates = state.escalation_count
    prefix = (("C0", "F1"),)
    gate_outcomes: dict[str, dict[str, Any]] = {
        "B_parse": {"used": True, "retries": len(result.failures)},
        "q_done": {"value": done, "confidence_recorded_only": (
            done_decision.confidence if done_decision is not None else None
        )},
        "G1": {"evaluated": False}, "G2": {"evaluated": False},
        "G3": {"evaluated": False}, "G4": {"evaluated": False},
    }
    worker = _MockWorker("synthetic worker answer")
    if done:
        candidate = "synthetic prior answer"
        valid = prior_output_valid
        action, route = "V1_VALIDATE_PRIOR", prefix + (
            ("F1", "V1"), ("V1", "End_success" if valid else "End_failure")
        )
    else:
        think = _should_think_harder(context, category)
        gate_outcomes["G1"] = {
            "evaluated": True, "pass": think,
            "reject_reason": None if think else "already-tried-or-category-or-budget",
            "reject_target": None if think else "G2",
            "budget": "B_think",
        }
        if think:
            action = "WORKER_THINK_HARDER"
            route = prefix + (("F1", "G1"), ("G1", "WORKER"))
        else:
            retry = _should_retry(context, category)
            gate_outcomes["G2"] = {
                "evaluated": True, "pass": retry,
                "reject_reason": None if retry else "timeout-or-retry-budget-exhausted",
                "reject_target": None if retry else "G3",
                "budget": "B_retry",
            }
            if retry:
                action = "WORKER_RETRY"
                route = prefix + (("F1", "G1"), ("G1", "G2"), ("G2", "WORKER"))
            else:
                escalate = _should_escalate(context, category, "coder_escalation")
                gate_outcomes["G3"] = {
                    "evaluated": True, "pass": escalate,
                    "reject_reason": None if escalate else (
                        "no-escalate-category-or-target-or-escalation-budget-or-cycle"
                    ),
                    "reject_target": None if escalate else "End_failure",
                    "budget": "B_esc",
                }
                if escalate and approval_granted:
                    gate_outcomes["G4"] = {
                        "evaluated": True, "pass": True,
                        "reject_target": "End_failure", "budget": "no-budget",
                        "evidence": "synthetic approval decision",
                    }
                    action = "G4_APPROVAL_GRANTED"
                    route = prefix + (
                        ("F1", "G1"), ("G1", "G2"), ("G2", "G3"),
                        ("G3", "G4"), ("G4", "WORKER"),
                    )
                elif escalate:
                    gate_outcomes["G4"] = {
                        "evaluated": True, "pass": False,
                        "reject_target": "End_failure", "budget": "no-budget",
                        "evidence": "synthetic approval refusal",
                    }
                    action, route = "END_FAILURE_APPROVAL_REFUSED", prefix + (
                        ("F1", "G1"), ("G1", "G2"), ("G2", "G3"),
                        ("G3", "G4"), ("G4", "End_failure"),
                    )
                else:
                    action, route = "END_FAILURE", prefix + (
                        ("F1", "G1"), ("G1", "G2"), ("G2", "G3"),
                        ("G3", "End_failure"),
                    )

        if route[-1][1] == "WORKER":
            candidate = worker.run(role="frontdoor", category=category, action=action)
            valid = output_valid
            gate_outcomes["V1"] = {
                "evaluated": True, "accepted": valid,
                "reject_target": None if valid else "End_failure",
            }
            route += (("WORKER", "V1"),
                      ("V1", "End_success" if valid else "End_failure"))
        else:
            candidate = None
            valid = False

    # q_done validation uses the same explicit deterministic validation node.
    if done:
        gate_outcomes["V1"] = {
            "evaluated": True, "accepted": valid,
            "reject_target": None if valid else "End_failure",
        }
    assert all(edge in edge_set for edge in route), f"undeclared workflow transition: {route}"
    assert state.consecutive_failures == failures_before_gates
    assert state.escalation_count == escalations_before_gates

    document = {
        "workflow_id": workflow.workflow_id, "nodes": workflow.nodes,
        "node_types": workflow.node_types, "edges": workflow.edges,
        "budgets": workflow.budgets, "model_catalog_id": workflow.fuzzy_model_catalog_id,
    }
    document_sha256 = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    record = {
        "evidence_kind": "synthetic_mock_only",
        "workflow_id": workflow.workflow_id,
        "workflow_document_sha256": document_sha256,
        "executed_node_types": [dict(workflow.node_types)[node] for node in
                                [workflow.start] + [target for _, target in route]],
        "fuzzy_model_catalog_id": workflow.fuzzy_model_catalog_id,
        "budgets": dict(workflow.budgets),
        "gate_outcomes": gate_outcomes,
        "worker_error_category": category.value,
        "q_cat_confidence_recorded_only": (
            category_decision.confidence if category_decision is not None else None
        ),
        "q_done": done,
        "typed_parse_failures": [failure.reason for failure in result.failures],
        "gate_action": action,
        "workflow_path": [list(edge) for edge in route],
        "worker_invocation_count": len(worker.calls),
        "candidate_output": candidate,
        "worker_output_is_synthetic": True,
        "output_validation_accepted": valid,
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
    # The real reader records the categorical margin above uniform, rather
    # than the emission's top probability or its untrusted confidence field.
    uniform = 1.0 / len(CATEGORY_OPTIONS)
    expected_margin = (confidence - uniform) / (1.0 - uniform)
    assert record["q_cat_confidence_recorded_only"] == pytest.approx(expected_margin)
    assert record["q_cat_confidence_recorded_only"] != confidence
    assert record["workflow_path"] == [
        ["C0", "F1"], ["F1", "G1"], ["G1", "WORKER"],
        ["WORKER", "V1"], ["V1", "End_success"],
    ]
    assert record["worker_invocation_count"] == 1
    assert record["candidate_output"] == "synthetic worker answer"
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
    assert record["gate_outcomes"]["B_parse"] == {"used": True, "retries": 1}


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
    assert record["gate_outcomes"]["B_parse"] == {"used": True, "retries": 2}
    assert record["worker_invocation_count"] == 1


@pytest.mark.parametrize("done", [False, True], ids=["incomplete", "complete"])
def test_q_done_routes_to_worker_or_validation(done: bool):
    primitives = _FakePrimitives([_emission("code", confidence=0.8, done=done)])
    record = _run_cli(raw_error="unrecognized worker error", primitives=primitives,
                      capture_hook=_synthetic_hook([]))
    assert record["q_done"] is done
    assert record["workflow_path"] == (
        [["C0", "F1"], ["F1", "V1"], ["V1", "End_success"]]
        if done else [["C0", "F1"], ["F1", "G1"], ["G1", "WORKER"],
                      ["WORKER", "V1"], ["V1", "End_success"]]
    )


def test_format_failure_retries_but_never_escalates():
    primitives = _FakePrimitives([_emission("format", confidence=0.8)])
    record = _run_cli(raw_error="unrecognized worker error", primitives=primitives,
                      capture_hook=_synthetic_hook([]))
    assert record["gate_action"] == "WORKER_RETRY"
    assert record["workflow_path"] == [
        ["C0", "F1"], ["F1", "G1"], ["G1", "G2"], ["G2", "WORKER"],
        ["WORKER", "V1"], ["V1", "End_success"],
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
    escalation_prefix = [
        ["C0", "F1"], ["F1", "G1"], ["G1", "G2"],
        ["G2", "G3"], ["G3", "G4"],
    ]
    assert granted["workflow_path"] == escalation_prefix + [
        ["G4", "WORKER"], ["WORKER", "V1"], ["V1", "End_success"]
    ]
    assert refused["workflow_path"] == escalation_prefix + [["G4", "End_failure"]]
    assert refused["worker_invocation_count"] == 0
    assert capped["workflow_path"] == escalation_prefix[:-1] + [["G3", "End_failure"]]
    assert granted["worker_invocation_count"] == 1


def test_worker_output_validation_rejection_has_declared_end_and_mock_call():
    record = _run_cli(
        raw_error="unrecognized worker error",
        primitives=_FakePrimitives([_emission("code", confidence=0.8)]),
        capture_hook=_synthetic_hook([]), output_valid=False,
    )
    assert record["worker_invocation_count"] == 1
    assert record["output_validation_accepted"] is False
    assert record["workflow_path"][-2:] == [["WORKER", "V1"], ["V1", "End_failure"]]


def test_missing_capture_hook_refuses_before_any_typed_call():
    primitives = _FakePrimitives([_emission("code", confidence=0.8)])
    with pytest.raises(RuntimeError, match="capture hook"):
        _run_cli(raw_error="a novel worker error", primitives=primitives, capture_hook=None)
    assert primitives.calls == []
