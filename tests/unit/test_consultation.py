"""Unit tests for one-shot internal consultation scaffolding."""

from __future__ import annotations

import json

import pytest

from src.orchestration.consultation import (
    ConsultationDenied,
    build_consult_prompt,
    consult,
    load_interaction_skill,
)
from src.scheduling.contention_gate import ContentionDenied as GateContentionDenied


class _FakePrimitives:
    def __init__(self, response: str | None = None, raise_contention: bool = False):
        self.response = response or json.dumps(
            {
                "risks": ["syntax drift"],
                "blocking_issues": [],
                "confidence": 0.7,
                "recommended_delta": "keep the draft",
            }
        )
        self.raise_contention = raise_contention
        self.calls: list[dict] = []
        self.contexts: list[dict] = []

    def request_context(self, **kwargs):
        self.contexts.append(kwargs)

        class _Ctx:
            def __enter__(_self):
                return None

            def __exit__(_self, exc_type, exc, tb):
                return False

        return _Ctx()

    def llm_call(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.raise_contention:
            raise GateContentionDenied("busy")
        return self.response


def test_load_review_before_commit_skill() -> None:
    skill = load_interaction_skill("architect_general", "review_before_commit")

    assert skill.kind == "consult"
    assert skill.max_output_tokens == 400
    assert skill.scheduler_defaults["priority"] == "background"
    assert skill.output_schema["required"] == [
        "risks",
        "blocking_issues",
        "confidence",
        "recommended_delta",
    ]
    assert len(skill.schema_hash) == 16


def test_build_consult_prompt_names_roles_and_schema() -> None:
    skill = load_interaction_skill("architect_general", "review_before_commit")

    prompt = build_consult_prompt(
        requester_role="coder_escalation",
        consultant_role="architect_general",
        skill=skill,
        context="draft",
    )

    assert "coder_escalation" in prompt
    assert "architect_general" in prompt
    assert "review_before_commit" in prompt
    assert "JSON schema" in prompt


def test_consult_calls_llm_with_schema_and_scheduler_context() -> None:
    primitives = _FakePrimitives()

    advisory, stats = consult(
        "architect_general",
        "coder_escalation",
        "review_before_commit",
        "draft context",
        primitives,
    )

    assert advisory["recommended_delta"] == "keep the draft"
    assert stats["interaction_type"] == "consult"
    assert stats["skill"] == "review_before_commit"
    assert primitives.contexts == [
        {
            "priority": "background",
            "workload_class": "consult",
            "max_queue_wait_ms": 2000,
        }
    ]
    assert primitives.calls[0]["role"] == "architect_general"
    assert primitives.calls[0]["n_tokens"] == 400
    assert primitives.calls[0]["json_schema"]["type"] == "object"


def test_consult_rejects_schema_violations() -> None:
    primitives = _FakePrimitives(response=json.dumps({"risks": []}))

    with pytest.raises(ConsultationDenied) as exc_info:
        consult(
            "architect_general",
            "coder_escalation",
            "review_before_commit",
            "draft context",
            primitives,
        )

    assert exc_info.value.reason == "schema_violation"


def test_consult_translates_contention_to_consultation_denied() -> None:
    primitives = _FakePrimitives(raise_contention=True)

    with pytest.raises(ConsultationDenied) as exc_info:
        consult(
            "architect_general",
            "coder_escalation",
            "review_before_commit",
            "draft context",
            primitives,
        )

    assert exc_info.value.reason == "contention_skip"
