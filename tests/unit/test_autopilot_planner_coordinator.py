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


def test_unparseable_critique_fails_closed_not_open() -> None:
    """Regression: a critic invoke that 'succeeds' (ok=True) but returns text
    that is NOT a valid json:autopilot_critique block (e.g. Codex emitting a
    file-read error or prose) must NOT silently auto-approve the risky draft.
    It must be treated as a FAILED critique → reject → safe seed_batch fallback,
    and mark the critic degraded for the circuit breaker."""
    original = {"type": "structural_experiment", "flags": {"a": True}}
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
            # ok=True but garbage payload — exactly the file-read-error case:
            # "Unable to read /mnt/raid0/llm/tmp/tmp....txt".
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text="Unable to read /mnt/raid0/llm/tmp/tmpXXXX.txt: file not found",
            )
        ],
    )

    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    # Fail-closed: the unsafe structural_experiment must NOT be admitted; it is
    # routed to the safe seed_batch fallback.
    assert decision.action is not None
    assert decision.action["type"] == "seed_batch"
    # The critique is recorded with its parse_error, not a clean approve.
    assert decision.critique is not None
    assert decision.critique.parse_error
    assert decision.degraded is True
    # Critic marked failed (feeds the circuit breaker).
    assert state.get("codex", {}).get("failures", 0) >= 1


def test_failed_critique_invoke_fails_closed_not_open() -> None:
    """A critic process failure (timeout, nonzero exit, empty response) must
    fail closed in active draft_critique mode, not silently admit the draft."""
    original = {"type": "structural_experiment", "flags": {"a": True}}
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
                ok=False,
                text="",
                error="timeout after 300s",
            )
        ],
    )

    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action is not None
    assert decision.action["type"] == "seed_batch"
    assert decision.critique is not None
    assert decision.critique.decision == "reject"
    assert decision.critique.parse_error == "timeout after 300s"
    assert decision.degraded is True
    assert state.get("codex", {}).get("failures", 0) >= 1


def test_unparseable_critique_shadow_mode_keeps_draft() -> None:
    """In shadow_critique (non-binding) mode, an unparseable critique still must
    not crash or fabricate approval — the draft stands (shadow never revises),
    but the critic is still marked degraded/failed."""
    original = {"type": "seed_batch", "n_questions": 10}
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text=_action_text(original))],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="critique", ok=True, text="garbage not-json")],
    )
    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(mode="shadow_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )
    assert decision.action == original  # shadow never revises
    assert decision.critique is not None and decision.critique.parse_error
    assert decision.degraded is True


def test_default_planner_mode_is_active_draft_critique() -> None:
    """The shipped default must be the BINDING critic (the shadow→active flip)."""
    assert planner_coordinator.PlannerSettings().mode == "draft_critique"
    import os as _os
    saved = _os.environ.pop("AUTOPILOT_PLANNER_MODE", None)
    try:
        assert planner_coordinator.load_planner_settings_from_env().mode == "draft_critique"
    finally:
        if saved is not None:
            _os.environ["AUTOPILOT_PLANNER_MODE"] = saved


# ----- _reconcile matrix (gates the active-mode default, req #1) -----

PlannerCritique = planner_coordinator.PlannerCritique
_reconcile = planner_coordinator._reconcile


def test_reconcile_inactive_is_passthrough_even_on_reject() -> None:
    action = {"type": "structural_experiment", "flags": {"a": True}}
    crit = PlannerCritique(decision="reject", issues=["x"])
    out_action, out_rat, out_text = _reconcile(action, {"r": 1}, "draft", crit, active=False)
    assert out_action == action
    assert out_rat == {"r": 1}


def test_reconcile_active_approve_is_passthrough() -> None:
    action = {"type": "seed_batch", "n_questions": 10}
    crit = PlannerCritique(decision="approve")
    out_action, _, _ = _reconcile(action, {}, "draft", crit, active=True)
    assert out_action == action


def test_reconcile_active_revise_applies_valid_revision() -> None:
    action = {"type": "structural_experiment", "flags": {"a": True}}
    revised = {"type": "seed_batch", "n_questions": 12}
    crit = PlannerCritique(decision="revise", revised_action=revised,
                           revised_rationale={"falsifier": "y", "rubric_scores": {}})
    out_action, out_rat, out_text = _reconcile(action, {}, "draft", crit, active=True)
    assert out_action == revised
    assert out_rat == {"falsifier": "y", "rubric_scores": {}}
    assert out_text.startswith("```json:autopilot_actions")


def test_reconcile_active_revise_with_invalid_revision_keeps_original() -> None:
    """A revise whose revised_action fails validation must not be dispatched;
    the original draft stands (it is still subject to dispatch-time validation)."""
    action = {"type": "seed_batch", "n_questions": 10}
    bad_revision = {"type": "not_a_real_action"}
    crit = PlannerCritique(decision="revise", revised_action=bad_revision)
    out_action, _, _ = _reconcile(action, {}, "draft", crit, active=True)
    assert out_action == action  # invalid revision rejected, original retained


def test_reconcile_active_reject_without_revision_is_safe_seed_batch() -> None:
    action = {"type": "rollback", "to_checkpoint": "production_best"}
    crit = PlannerCritique(decision="reject", issues=["unsupported"])
    out_action, _, _ = _reconcile(action, {}, "draft", crit, active=True)
    assert out_action["type"] == "seed_batch"


def test_decision_carries_original_draft_action_after_substitution() -> None:
    """draft_action must preserve the planner's ORIGINAL action even when the
    binding critic substitutes it (so the loop can record the rejected draft)."""
    original = {"type": "structural_experiment", "flags": {"graph_router": True}}
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text=_action_text(original))],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="critique", ok=True,
                               text=_critique_text({"decision": "reject", "issues": ["deps unmet"]}))],
    )
    decision = planner_coordinator.plan_with_providers(
        "prompt", session_id=None, planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )
    assert decision.action["type"] == "seed_batch"          # substituted
    assert decision.draft_action == original                # original preserved


def test_critique_prompt_surfaces_flag_and_feedback_context() -> None:
    """The critic must SEE the flag schema, current flags, last-invalid action,
    and blacklist (embedded via the planner prompt) and be told to use them."""
    planner_prompt = (
        "### Feature Flags (live state + dependency rules)\n"
        "  - graph_router (currently OFF) requires [specialist_routing=OFF]\n"
        "### Last Non-Executing Action (validator/dispatch feedback)\n"
        "  reason: graph_router feature requires specialist_routing feature\n"
        "### Blacklisted Configurations\n  some-entry\n"
    )
    out = planner_coordinator.build_critique_prompt(
        planner_prompt, "draft text",
        {"type": "structural_experiment", "flags": {"graph_router": True}}, {},
    )
    # Context embedded:
    assert "Feature Flags" in out
    assert "Last Non-Executing Action" in out
    assert "Blacklisted Configurations" in out
    assert "specialist_routing" in out
    # Instructions tell the critic to use them + that the verdict is binding:
    assert "BINDING" in out
    assert "dependencies are not all currently ON" in out


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
