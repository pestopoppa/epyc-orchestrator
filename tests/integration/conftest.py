"""Shared fixtures for graph integration tests.

Provides a GraphRunContext factory with real REPL execution, mock LLM
primitives (controllable responses), and stub observability graphs.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.graph.state import GraphConfig, TaskDeps, TaskState
from src.repl_environment.environment import REPLEnvironment
from src.repl_environment.types import REPLConfig
from src.roles import Role

# ---------------------------------------------------------------------------
# Stub implementations for FailureGraph and HypothesisGraph protocols
# ---------------------------------------------------------------------------


class StubFailureGraph:
    """In-memory FailureGraph that records calls for assertion."""

    def __init__(self):
        self.failures: list[dict[str, Any]] = []
        self.mitigations: list[dict[str, Any]] = []

    def get_failure_risk(self, action: str) -> float:
        return 0.0

    def record_failure(
        self,
        memory_id: str,
        symptoms: list[str],
        description: str = "",
        severity: int = 3,
        previous_failure_id: str | None = None,
    ) -> str:
        fid = f"f-{len(self.failures)}"
        self.failures.append({
            "id": fid,
            "memory_id": memory_id,
            "symptoms": symptoms,
            "description": description,
            "severity": severity,
        })
        return fid

    def record_mitigation(self, failure_id: str, action: str, worked: bool) -> str:
        mid = f"m-{len(self.mitigations)}"
        self.mitigations.append({
            "id": mid,
            "failure_id": failure_id,
            "action": action,
            "worked": worked,
        })
        return mid

    def find_matching_failures(self, symptoms: list[str]) -> list[Any]:
        return [f for f in self.failures if set(symptoms) & set(f["symptoms"])]

    def get_effective_mitigations(self, symptoms: list[str]) -> list[dict[str, Any]]:
        matching = self.find_matching_failures(symptoms)
        fids = {f["id"] for f in matching}
        return [m for m in self.mitigations if m["failure_id"] in fids and m["worked"]]


class StubHypothesisGraph:
    """In-memory HypothesisGraph that records calls for assertion."""

    def __init__(self):
        self.evidence: list[dict[str, Any]] = []
        self._confidence: float = 0.5

    def add_evidence(self, hypothesis_id: str, outcome: str, source: str) -> float:
        self.evidence.append({
            "hypothesis_id": hypothesis_id,
            "outcome": outcome,
            "source": source,
        })
        if outcome == "success":
            self._confidence = min(1.0, self._confidence + 0.1)
        else:
            self._confidence = max(0.0, self._confidence - 0.1)
        return self._confidence

    def get_confidence(self, action: str, task_type: str) -> float:
        return self._confidence


# ---------------------------------------------------------------------------
# Mock LLM Primitives (controllable responses)
# ---------------------------------------------------------------------------


def make_mock_primitives(responses: list[str] | None = None) -> MagicMock:
    """Create a MagicMock LLM primitives that returns canned responses.

    Args:
        responses: List of responses to return in sequence.
            If None, returns 'FINAL("mock_answer")' on every call.
            When the list is exhausted, repeats the last response.
    """
    if responses is None:
        responses = ['FINAL("mock_answer")']

    call_index = {"i": 0}

    def _llm_call(prompt, role="worker", **kwargs):
        idx = min(call_index["i"], len(responses) - 1)
        call_index["i"] += 1
        return responses[idx]

    mock = MagicMock()
    mock.llm_call = MagicMock(side_effect=_llm_call)
    mock._early_stop_check = None
    mock._last_inference_meta = {"tokens": 50, "prompt_tokens": 100}
    mock._backends = True
    mock.total_tokens_generated = 0
    mock.total_prompt_eval_ms = 0.0
    mock.total_generation_ms = 0.0
    mock._last_predicted_tps = 25.0
    mock.total_http_overhead_ms = 0.0
    mock.get_cache_stats = MagicMock(return_value={"hits": 0, "misses": 0})
    return mock


# ---------------------------------------------------------------------------
# GraphRunContext factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_ctx(tmp_path):
    """Create a minimal but functional GraphRunContext factory.

    Returns a callable that builds (state, deps) tuples. The REPL is
    real (executes actual Python), but LLM calls are mocked.

    Usage::

        state, deps = graph_ctx(
            prompt="What is 2+2?",
            responses=['FINAL("4")'],
        )
    """

    def _factory(
        prompt: str = "Test prompt",
        context: str = "Test context",
        task_id: str | None = None,
        role: Role | str = Role.FRONTDOOR,
        responses: list[str] | None = None,
        max_turns: int = 15,
        with_failure_graph: bool = False,
        with_hypothesis_graph: bool = False,
    ) -> tuple[TaskState, TaskDeps]:
        tid = task_id or f"test-{uuid.uuid4().hex[:8]}"

        state = TaskState(
            task_id=tid,
            prompt=prompt,
            context=context,
            current_role=role,
            max_turns=max_turns,
        )

        repl = REPLEnvironment(
            context=context,
            config=REPLConfig(timeout_seconds=10, spill_dir=str(tmp_path)),
            role=str(role),
            task_id=tid,
        )

        deps = TaskDeps(
            primitives=make_mock_primitives(responses),
            repl=repl,
            failure_graph=StubFailureGraph() if with_failure_graph else None,
            hypothesis_graph=StubHypothesisGraph() if with_hypothesis_graph else None,
            config=GraphConfig(max_turns=max_turns),
        )

        return state, deps

    return _factory
