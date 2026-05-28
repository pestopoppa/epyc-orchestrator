"""Tests for AutoPilot provider coordination."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

planner_coordinator = importlib.import_module("planner_coordinator")
planner_providers = importlib.import_module("planner_providers")

PlannerProviderResult = planner_providers.PlannerProviderResult
PlannerSettings = planner_coordinator.PlannerSettings


@pytest.fixture(autouse=True)
def _no_planner_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner_coordinator, "_append_planner_archive", lambda _record: None)


class FakeProvider:
    def __init__(
        self,
        name: str,
        responses: list[PlannerProviderResult],
        *,
        supports_resume: bool = False,
    ) -> None:
        self.name = name
        self.supports_resume = supports_resume
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        prompt: str,
        *,
        role: str,
        session_id: str | None = None,
        timeout: int = 300,
        cwd: Path | str | None = None,
    ) -> PlannerProviderResult:
        self.calls.append(
            {
                "prompt": prompt,
                "role": role,
                "session_id": session_id,
                "timeout": timeout,
                "cwd": cwd,
            }
        )
        result = self.responses.pop(0)
        result.provider = self.name
        result.role = role
        return result


def _factory(providers: dict[str, FakeProvider]):
    def _get(name: str) -> FakeProvider:
        return providers[name]

    return _get


def _action_text(action: dict[str, Any]) -> str:
    import json

    return (
        "```json:autopilot_actions\n"
        f"{json.dumps(action)}\n"
        "```\n\n"
        "```json:autopilot_rationale\n"
        '{"falsifier":"x","rubric_scores":{"info_gain":3}}\n'
        "```\n"
    )


def _critique_text(payload: dict[str, Any]) -> str:
    import json

    return f"```json:autopilot_critique\n{json.dumps(payload)}\n```\n"


def test_primary_failure_falls_back_to_secondary() -> None:
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json")],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text({"type": "seed_batch", "n_questions": 10}),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state={},
        settings=PlannerSettings(mode="fallback"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == {"type": "seed_batch", "n_questions": 10}
    assert decision.draft_provider == "codex"
    assert "claude draft failed" in decision.fallback_reason
    assert len(claude.calls) == 1
    assert len(codex.calls) == 1


def test_shadow_critique_does_not_apply_revision() -> None:
    original = {"type": "code_mutation", "file": "src/escalation.py"}
    revised = {"type": "seed_batch", "n_questions": 10}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "revise",
                        "confidence": 0.8,
                        "issues": ["too risky"],
                        "revised_action": revised,
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="shadow_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == original
    assert decision.critique is not None
    assert decision.critique.decision == "revise"


def test_active_critique_applies_valid_revision() -> None:
    original = {"type": "structural_experiment", "flags": {"a": True}}
    revised = {"type": "seed_batch", "n_questions": 10}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "revise",
                        "confidence": 0.9,
                        "issues": ["missing validation"],
                        "revised_action": revised,
                        "revised_rationale": {"falsifier": "x", "rubric_scores": {}},
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == revised
    assert decision.canonical_text.startswith("```json:autopilot_actions")


def test_active_reject_without_revision_uses_safe_seed_batch() -> None:
    original = {"type": "rollback", "to_checkpoint": "production_best"}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "reject",
                        "confidence": 0.95,
                        "issues": ["unsupported rollback"],
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action is not None
    assert decision.action["type"] == "seed_batch"
    assert decision.action["n_questions"] == 10


def test_open_primary_circuit_routes_directly_to_fallback() -> None:
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text="unused")],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text({"type": "seed_batch", "n_questions": 10}),
            )
        ],
    )
    state = {"claude": {"circuit_open_until": time.time() + 60}}

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state=state,
        settings=PlannerSettings(mode="fallback"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.fallback_reason == "claude circuit open"
    assert claude.calls == []
    assert len(codex.calls) == 1
