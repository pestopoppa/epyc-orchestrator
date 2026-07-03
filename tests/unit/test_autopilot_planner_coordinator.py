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


def test_codex_primary_can_use_distinct_codex_critic_alias() -> None:
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text(
                    {
                        "type": "structural_prune",
                        "file": "frontdoor.md",
                        "block": "## Examples",
                    }
                ),
            )
        ],
    )
    codex_critic = FakeProvider(
        "codex_critic",
        [
            PlannerProviderResult(
                provider="codex_critic",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "approve",
                        "confidence": 0.8,
                        "issues": [],
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(
            primary="codex",
            critic="codex_critic",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory({"codex": codex, "codex_critic": codex_critic}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == "codex_critic"
    assert decision.degraded is False
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == ""


def test_fallback_draft_gets_independent_primary_critique() -> None:
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json"),
            PlannerProviderResult(
                provider="claude",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "approve",
                        "confidence": 0.86,
                        "issues": ["fallback draft is independently reviewed"],
                    }
                ),
            ),
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text({"type": "numeric_trial", "surface": "memrl_retrieval"}),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == "claude"
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert [call["role"] for call in claude.calls] == ["draft", "critique"]
    assert [call["role"] for call in codex.calls] == ["draft"]
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == ""


def test_fallback_draft_gets_primary_critique_even_when_draft_failure_opens_circuit() -> None:
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json"),
            PlannerProviderResult(
                provider="claude",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "approve",
                        "confidence": 0.86,
                        "issues": [],
                    }
                ),
            ),
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text(
                    {"type": "structural_experiment", "flags": {"user_modeling": True}}
                ),
            )
        ],
    )

    state = {"claude": {"failures": 1, "circuit_open_until": 0.0}}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state=state,
        settings=PlannerSettings(
            mode="draft_critique",
            critique_policy="always",
            circuit_failures=2,
        ),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == "claude"
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert [call["role"] for call in claude.calls] == ["draft", "critique"]
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == ""


def test_fallback_draft_pauses_when_independent_primary_critique_fails() -> None:
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json"),
            PlannerProviderResult(
                provider="claude",
                role="critique",
                ok=False,
                text="",
                error="empty response",
            ),
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text({"type": "numeric_trial", "surface": "memrl_retrieval"}),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == "claude"
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == "critic_unavailable"


def test_nonresumable_primary_clears_persisted_session_id() -> None:
    action = {"type": "seed_batch", "n_questions": 10}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(action),
                session_id="new-session",
            )
        ],
        supports_resume=False,
    )
    codex = FakeProvider("codex", [])

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old-session",
        planner_state={},
        settings=PlannerSettings(mode="single"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == action
    assert decision.session_id is None
    assert claude.calls[0]["session_id"] is None


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
    assert decision.action["n_questions"] == planner_coordinator.SAFE_FALLBACK_SEED_N


def test_unparseable_critique_fails_closed_not_open() -> None:
    """Regression: a critic invoke that 'succeeds' (ok=True) but returns text
    that is NOT a valid json:autopilot_critique block (e.g. Codex emitting a
    file-read error or prose) must NOT silently auto-approve the risky draft.
    It must be treated as a FAILED REVIEW: verdict "unavailable", the trusted
    primary draft KEPT (not swapped for seed_batch), critic marked degraded for
    the circuit breaker. Because the draft is HIGH-risk, the dispatch gate then
    fails CLOSED by PAUSING (critic_unavailable) — the unsafe draft is still not
    admitted unreviewed. (Behavior refined 2026-06-10.)"""
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

    # New contract (2026-06-10): a FAILED review does NOT substitute seed_batch —
    # the trusted-primary draft is KEPT and the verdict is "unavailable".
    assert decision.action is not None
    assert decision.action["type"] == "structural_experiment"
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert decision.critique.parse_error  # recorded, not a clean approve
    assert decision.degraded is True
    # Critic marked failed (feeds the circuit breaker).
    assert state.get("codex", {}).get("failures", 0) >= 1
    # Fail-closed for HIGH-risk: the unsafe uncritiqued draft is NOT admitted —
    # the dispatch gate pauses for operator review.
    assert (
        planner_coordinator.uncritiqued_dispatch_block_reason(decision)
        == "critic_unavailable"
    )


def test_failed_critique_invoke_fails_closed_not_open() -> None:
    """A critic process failure (timeout, nonzero exit, empty response) on a
    HIGH-risk draft must fail closed: keep the trusted draft + verdict
    "unavailable", and the dispatch gate PAUSES (critic_unavailable) rather than
    admit it unreviewed or substitute a stale seed_batch. (Refined 2026-06-10.)"""
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
    # Trusted draft KEPT (not seed_batch); verdict "unavailable".
    assert decision.action["type"] == "structural_experiment"
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert decision.critique.parse_error == "timeout after 300s"
    assert decision.degraded is True
    assert state.get("codex", {}).get("failures", 0) >= 1
    # HIGH-risk + critic unavailable => gate pauses (fails closed).
    assert (
        planner_coordinator.uncritiqued_dispatch_block_reason(decision)
        == "critic_unavailable"
    )


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
    assert out_action["n_questions"] == planner_coordinator.SAFE_FALLBACK_SEED_N


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
    """The critic sees the measurement/constraint view, not the full prompt."""
    planner_prompt = (
        "## Original Planning Instructions\n"
        "FULL_PLANNER_MARKER_UNIQUE should never reach the critic.\n"
        "### Evidence Power and Sequential Candidate Status\n"
        "  - quality MDE: 0.070; seq verdict: accumulating\n"
        "### System Health\n"
        "  - nominal; no host contention detected\n"
        "### Action Availability\n"
        "  - graph_router unavailable until specialist_routing is OFF\n"
        "### Blacklisted Configurations\n  some-entry\n"
        "### Feature Flags (live state + dependency rules)\n"
        "  - graph_router (currently OFF) requires [specialist_routing=OFF]\n"
        "### Last Non-Executing Action (validator/dispatch feedback)\n"
        "  reason: graph_router feature requires specialist_routing feature\n"
        "### Experiment Journal\n"
        "  SHOULD_NOT_REACH_CRITIC\n"
    )
    out = planner_coordinator.build_critique_prompt(
        planner_prompt, "draft text",
        {"type": "structural_experiment", "flags": {"graph_router": True}}, {},
    )
    # Selected context embedded:
    assert "Selected Measurement and Constraint Context" in out
    assert "Evidence Power and Sequential Candidate Status" in out
    assert "quality MDE: 0.070" in out
    assert "System Health" in out
    assert "Action Availability" in out
    assert "Feature Flags" in out
    assert "Last Non-Executing Action" in out
    assert "Blacklisted Configurations" in out
    assert "specialist_routing" in out
    # Full planner context not embedded:
    assert "Original Planner Context" not in out
    assert "FULL_PLANNER_MARKER_UNIQUE" not in out
    assert "SHOULD_NOT_REACH_CRITIC" not in out
    # Instructions tell the critic to use them + that the verdict is binding:
    assert "BINDING" in out
    assert "dependencies are not all currently ON" in out
    assert "below-MDE" in out
    assert "unrelated dirty files" in out
    assert "target-path dirty fence" in out


def test_critique_prompt_caps_selected_sections() -> None:
    long_feature_section = "x" * 2000
    planner_prompt = (
        "### Feature Flags\n"
        f"{long_feature_section}\n"
        "TAIL_MARKER_AFTER_LIMIT\n"
    )
    out = planner_coordinator.build_critique_prompt(
        planner_prompt, "draft text", {"type": "seed_batch"}, {},
    )
    assert "... [truncated for critic context]" in out
    assert "TAIL_MARKER_AFTER_LIMIT" not in out


def test_critique_prompt_uses_safe_fallback_when_selected_sections_absent() -> None:
    out = planner_coordinator.build_critique_prompt(
        "## Planner internals\nFULL_PLANNER_MARKER_UNIQUE\n",
        "draft text",
        {"type": "seed_batch"},
        {},
    )
    assert "selected planner context unavailable" in out
    assert "FULL_PLANNER_MARKER_UNIQUE" not in out


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


def _uncritiqued_decision(action: Any, *, degraded: bool, critique: Any,
                          rationale: Any = None, mode: str = "draft_critique"):
    return planner_coordinator.PlannerDecision(
        action=action, rationale=rationale or {}, session_id=None, canonical_text="",
        draft_text="", draft_provider="codex", mode=mode,
        degraded=degraded, critique=critique,
    )


def test_uncritiqued_degraded_nonobservational_action_pauses() -> None:
    """Degraded with NO critic verdict + non-observational action => critic_unavailable.
    seed_batch is explicitly NOT safe (the @708 critic_reject_loop failure mode)."""
    ubr = planner_coordinator.uncritiqued_dispatch_block_reason
    assert ubr(_uncritiqued_decision({"type": "seed_batch", "n_questions": 12},
               degraded=True, critique=None)) == "critic_unavailable"
    # OBSERVATIONAL_ACTIONS is empty => every non-observational action blocks.
    assert ubr(_uncritiqued_decision({"type": "structural_experiment"},
               degraded=True, critique=None)) == "critic_unavailable"


def test_uncritiqued_gate_allows_when_critiqued_or_not_degraded() -> None:
    ubr = planner_coordinator.uncritiqued_dispatch_block_reason
    crit = planner_coordinator.PlannerCritique(decision="approve", provider="codex")
    # real critic verdict => not uncritiqued => no pause
    assert ubr(_uncritiqued_decision({"type": "seed_batch", "n_questions": 12},
               degraded=True, critique=crit)) == ""
    # not degraded => no pause
    assert ubr(_uncritiqued_decision({"type": "seed_batch", "n_questions": 12},
               degraded=False, critique=None)) == ""
    # no dict action => handled by the separate no-action path => no pause
    assert ubr(_uncritiqued_decision(None, degraded=True, critique=None)) == ""


def test_draft_validation_error_is_surfaced_not_opaque() -> None:
    """#776 root cause (2026-06-11): a draft that PARSES but violates a schema cap
    (train_routing_models min_memories > 100000) must surface the EXACT validator
    error in fallback_reason, not an opaque 'invalid action' — so the next trial's
    feedback can self-correct."""
    bad = {"type": "train_routing_models", "min_memories": 250000}
    good = {"type": "train_routing_models", "min_memories": 500}
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="draft", ok=True,
                               text=_action_text(bad))],
    )
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True,
                               text=_action_text(good))],
        supports_resume=True,
    )
    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt", session_id=None, planner_state=state,
        settings=PlannerSettings(primary="codex", critic="claude", mode="fallback"),
        provider_factory=_factory({"codex": codex, "claude": claude}),
    )
    assert decision.fallback_reason
    assert "invalid action" not in decision.fallback_reason
    assert "min_memories" in decision.fallback_reason
    assert "100000" in decision.fallback_reason
    assert decision.action == good  # the valid fallback draft was adopted


def test_uncritiqued_unavailable_dispatch_rule() -> None:
    """Tightened Case-B rule (2026-06-12): verdict 'unavailable' on a TRUSTED
    primary draft. Risk class alone is NOT the guard (the @708 failure was
    'low-risk' seed looping):
      - HIGH risk => pause.
      - seed_batch / passive low-risk => pause (loop-prone class).
      - MEDIUM experiment => proceed ONLY IF novel + non-looping (not blacklisted,
        not repeated, carries a falsifier); else pause.
      - shadow (non-binding) mode => never blocks."""
    ubr = planner_coordinator.uncritiqued_dispatch_block_reason
    unavail = planner_coordinator.PlannerCritique(
        decision="unavailable", provider="claude", parse_error="timeout"
    )
    falsifier = {"falsifier": "tps drops >5% if X regresses"}

    # HIGH risk => pause
    assert ubr(_uncritiqued_decision({"type": "structural_experiment"},
               degraded=True, critique=unavail, rationale=falsifier)) == "critic_unavailable"
    # seed_batch (loop-prone low-risk) => pause even with a falsifier
    assert ubr(_uncritiqued_decision({"type": "seed_batch", "n_questions": 12},
               degraded=True, critique=unavail, rationale=falsifier)) == "critic_unavailable"
    # passive low-risk (deep_eval) => pause (not in OBSERVATIONAL_ACTIONS)
    assert ubr(_uncritiqued_decision({"type": "deep_eval", "tier": 2},
               degraded=True, critique=unavail, rationale=falsifier)) == "critic_unavailable"

    # MEDIUM experiment, novel + falsifier => PROCEED
    assert ubr(_uncritiqued_decision({"type": "numeric_trial"},
               degraded=True, critique=unavail, rationale=falsifier)) == ""
    # MEDIUM but no falsifier (loop-keeping) => pause
    assert ubr(_uncritiqued_decision({"type": "numeric_trial"},
               degraded=True, critique=unavail, rationale={})) == "critic_unavailable"
    # MEDIUM + blacklisted => pause
    assert ubr(_uncritiqued_decision({"type": "numeric_trial"},
               degraded=True, critique=unavail, rationale=falsifier),
               is_blacklisted=True) == "critic_unavailable"
    # MEDIUM + repeated (recurring invalid signature) => pause
    assert ubr(_uncritiqued_decision({"type": "numeric_trial"},
               degraded=True, critique=unavail, rationale=falsifier),
               is_repeated=True) == "critic_unavailable"

    # shadow (non-binding) mode never blocks, even for HIGH risk
    assert ubr(_uncritiqued_decision({"type": "structural_experiment"},
               degraded=True, critique=unavail, rationale=falsifier,
               mode="shadow_critique")) == ""


def test_failed_critique_keeps_draft_seed_batch_still_pauses() -> None:
    """End-to-end (2026-06-12): a critic failure on a seed_batch draft KEEPS the
    draft (verdict 'unavailable', not a stale seed_batch *substitution*) — but the
    tightened gate still PAUSES it, because seed looping is the exact @708 failure
    the critic exists to catch. Risk class alone must not let it through."""
    original = {"type": "seed_batch", "n_questions": 12}  # loop-prone low-risk
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True,
                               text=_action_text(original))],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="critique", ok=False,
                               text="", error="timeout after 600s")],
    )
    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt", session_id=None, planner_state=state,
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )
    # Draft KEPT (not the seed_batch-fallback substitution), verdict 'unavailable'.
    assert decision.action == original
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert decision.degraded is True
    assert state.get("codex", {}).get("failures", 0) >= 1
    # seed_batch is loop-prone => the gate pauses for operator review.
    assert (
        planner_coordinator.uncritiqued_dispatch_block_reason(decision)
        == "critic_unavailable"
    )


def test_cross_model_failover_codex_offline_crosses_to_claude() -> None:
    # PRIMARY=codex + CRITIC=codex_critic both resolve to the codex binary. When
    # codex is offline the failover draft MUST cross to a DIFFERENT model (claude),
    # not re-hit codex. (cross-model failover, 2026-06-12)
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="draft", ok=False,
                               text="", error="timeout after 600s")],
    )
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(
            provider="claude", role="draft", ok=True,
            text=_action_text({"type": "seed_batch", "n_questions": 10}),
        )],
    )

    def factory(name: str) -> FakeProvider:
        # Both configured codex roles map to the codex stub; claude to claude.
        if name in ("codex", "codex_critic"):
            return codex
        return claude

    decision = planner_coordinator.plan_with_providers(
        "prompt", session_id=None, planner_state={},
        settings=PlannerSettings(primary="codex", critic="codex_critic",
                                 mode="draft_critique"),
        provider_factory=factory,
    )

    assert decision.action == {"type": "seed_batch", "n_questions": 10}
    assert decision.draft_provider == "claude"
    assert decision.providers_unavailable is False


def test_both_models_unavailable_sets_providers_unavailable() -> None:
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="draft", ok=False,
                               text="", error="timeout")],
    )
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=False,
                               text="", error="empty response")],
    )

    def factory(name: str) -> FakeProvider:
        if name in ("codex", "codex_critic"):
            return codex
        return claude

    decision = planner_coordinator.plan_with_providers(
        "prompt", session_id=None, planner_state={},
        settings=PlannerSettings(primary="codex", critic="codex_critic",
                                 mode="draft_critique"),
        provider_factory=factory,
    )

    assert decision.action is None
    assert decision.providers_unavailable is True
    assert "both planner models unavailable" in decision.fallback_reason


def test_content_failure_is_not_providers_unavailable() -> None:
    # Both models RESPOND ok but neither emits a parseable action block. This is a
    # CONTENT failure, not an availability failure → providers_unavailable False.
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="draft", ok=True,
                               text="no json")],
    )
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True,
                               text="also no json")],
    )

    def factory(name: str) -> FakeProvider:
        if name in ("codex", "codex_critic"):
            return codex
        return claude

    decision = planner_coordinator.plan_with_providers(
        "prompt", session_id=None, planner_state={},
        settings=PlannerSettings(primary="codex", critic="codex_critic",
                                 mode="draft_critique"),
        provider_factory=factory,
    )

    assert decision.action is None
    assert decision.providers_unavailable is False


def test_open_primary_circuit_routes_to_other_model() -> None:
    # A pre-open primary (codex) circuit must route the draft to the OTHER model
    # (claude), even though the configured critic (codex_critic) is also codex.
    import time as _time

    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(
            provider="claude", role="draft", ok=True,
            text=_action_text({"type": "seed_batch", "n_questions": 10}),
        )],
    )
    codex = FakeProvider("codex", [])  # must NOT be invoked

    def factory(name: str) -> FakeProvider:
        if name in ("codex", "codex_critic"):
            return codex
        return claude

    state = {"codex": {"circuit_open_until": _time.time() + 600.0, "failures": 2}}
    decision = planner_coordinator.plan_with_providers(
        "prompt", session_id=None, planner_state=state,
        settings=PlannerSettings(primary="codex", critic="codex_critic",
                                 mode="draft_critique"),
        provider_factory=factory,
    )

    assert decision.draft_provider == "claude"
    assert decision.action == {"type": "seed_batch", "n_questions": 10}
    assert len(codex.calls) == 0
