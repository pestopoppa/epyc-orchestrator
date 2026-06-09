"""Draft, critique, fallback, and reconciliation for AutoPilot planning."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import peaf
from controller_io import (
    _append_planner_archive,
    extract_action,
    extract_rationale,
    validate_single_variable,
)
from planner_providers import (
    PlannerProvider,
    PlannerProviderResult,
    get_planner_provider,
)

log = logging.getLogger("autopilot")

KNOWN_ACTION_TYPES = {
    "seed_batch",
    "numeric_trial",
    "prompt_mutation",
    "gepa_optimize",
    "code_mutation",
    "structural_experiment",
    "structural_prune",
    "slot_compact",
    "train_routing_models",
    "distill_skillbank",
    "reset_memories",
    "deep_eval",
    "rollback",
    "distill_knowledge",
}

LOW_RISK_ACTIONS = {"seed_batch", "deep_eval", "distill_knowledge"}
MEDIUM_RISK_ACTIONS = {
    "numeric_trial",
    "slot_compact",
    "train_routing_models",
    "distill_skillbank",
}
HIGH_RISK_ACTIONS = KNOWN_ACTION_TYPES - LOW_RISK_ACTIONS - MEDIUM_RISK_ACTIONS

# Actions safe to DISPATCH WITHOUT a critic verdict (degraded/uncritiqued mode).
# Intentionally EMPTY: when the binding critic is unavailable we PAUSE by default
# rather than run anything unreviewed. NOTE this is deliberately stricter than
# LOW_RISK_ACTIONS — seed_batch is NOT here, because low-risk seed looping was the
# exact failure the critic exists to catch (critic_reject_loop @708). Extend ONLY
# with genuinely observational, non-mutating, non-looping action types.
OBSERVATIONAL_ACTIONS: frozenset[str] = frozenset()

# Claude planner drafts run a ~80KB prompt WITH tool access (Read/Grep/Glob) and
# take 60-228s in practice; the old hard 300s default intermittently KILLED a
# slow-but-valid draft → "empty response" → degraded (and, since 2026-06-07, a
# critic_unavailable pause that stalled the run, e.g. @711). Give generous
# headroom; operator-tunable via AUTOPILOT_PLANNER_TIMEOUT. (2026-06-09)
DEFAULT_PLANNER_TIMEOUT = int(os.environ.get("AUTOPILOT_PLANNER_TIMEOUT", "600"))


@dataclass
class PlannerSettings:
    primary: str = "claude"
    critic: str = "codex"
    # 2026-06-04: default flipped shadow_critique -> draft_critique. The critic is
    # now BINDING: a valid revised_action is dispatched, and a reject routes to the
    # safe fallback (see _reconcile). Shadow mode logged 320 reject/revise verdicts
    # that were all ignored (264 dispatched anyway) while the run dead-locked.
    # Gated on the expanded _reconcile test matrix + the rejected-draft feedback
    # path in autopilot.py (a reject can no longer bypass blacklist/quota/skip).
    mode: str = "draft_critique"
    critique_policy: str = "medium_plus"
    circuit_failures: int = 2
    circuit_cooldown_s: float = 900.0


@dataclass
class PlannerCritique:
    decision: str = "approve"
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    revised_action: dict[str, Any] | None = None
    revised_rationale: dict[str, Any] | None = None
    raw_text: str = ""
    provider: str = ""
    parse_error: str = ""


@dataclass
class PlannerDecision:
    action: dict[str, Any] | None
    rationale: dict[str, Any]
    session_id: str | None
    canonical_text: str
    draft_text: str
    draft_provider: str
    mode: str
    degraded: bool = False
    fallback_reason: str = ""
    critic_provider: str = ""
    critique: PlannerCritique | None = None
    predicted_objectives: dict[str, float] = field(default_factory=dict)
    # The planner's ORIGINAL parsed action, before _reconcile may substitute it
    # (revised_action in revise, or the safe fallback in reject). The main loop
    # uses this to record a critic-rejected/revised draft into the invalid-action
    # feedback + blacklist so a substituted draft cannot silently escape the
    # feedback loop (draft_critique authority change, 2026-06-04).
    draft_action: dict[str, Any] | None = None


ProviderFactory = Callable[[str], PlannerProvider]


def load_planner_settings_from_env() -> PlannerSettings:
    return PlannerSettings(
        primary=os.environ.get("AUTOPILOT_PLANNER_PRIMARY", "claude"),
        critic=os.environ.get("AUTOPILOT_PLANNER_CRITIC", "codex"),
        mode=os.environ.get("AUTOPILOT_PLANNER_MODE", "draft_critique"),
        critique_policy=os.environ.get(
            "AUTOPILOT_PLANNER_CRITIQUE_POLICY",
            "medium_plus",
        ),
        circuit_failures=_env_int("AUTOPILOT_PLANNER_CIRCUIT_FAILURES", 2),
        circuit_cooldown_s=_env_float("AUTOPILOT_PLANNER_CIRCUIT_COOLDOWN_S", 900.0),
    )


def plan_with_providers(
    prompt: str,
    *,
    session_id: str | None,
    timeout: int = DEFAULT_PLANNER_TIMEOUT,
    cwd: Path | str | None = None,
    planner_state: dict[str, Any] | None = None,
    stagnation_signal: str = "",
    settings: PlannerSettings | None = None,
    provider_factory: ProviderFactory = get_planner_provider,
) -> PlannerDecision:
    """Draft a canonical planner action with optional secondary critique."""
    settings = settings or load_planner_settings_from_env()
    planner_state = planner_state if planner_state is not None else {}
    session_update = session_id

    primary_name = _normalize_provider(settings.primary)
    critic_name = _normalize_provider(settings.critic)
    fallback_name = critic_name if critic_name != primary_name else _other_provider(primary_name)
    allow_fallback = settings.mode.strip().lower() != "single"

    draft_provider_name = primary_name
    fallback_reason = ""
    if allow_fallback and _circuit_is_open(planner_state, primary_name):
        draft_provider_name = fallback_name
        fallback_reason = f"{primary_name} circuit open"

    draft_provider = provider_factory(draft_provider_name)
    draft = draft_provider.invoke(
        prompt,
        role="draft",
        session_id=session_id if draft_provider.supports_resume else None,
        timeout=timeout,
        cwd=cwd,
    )
    if draft.provider == "claude":
        session_update = draft.session_id

    action = extract_action(draft.text)
    if not _draft_is_usable(draft, action):
        _mark_failure(planner_state, draft.provider, settings)
        if (
            allow_fallback
            and draft_provider_name != fallback_name
            and not _circuit_is_open(
                planner_state,
                fallback_name,
            )
        ):
            fallback_provider = provider_factory(fallback_name)
            fallback = fallback_provider.invoke(
                prompt,
                role="draft",
                session_id=session_update if fallback_provider.supports_resume else None,
                timeout=timeout,
                cwd=cwd,
            )
            if fallback.provider == "claude":
                session_update = fallback.session_id
            fallback_action = extract_action(fallback.text)
            if _draft_is_usable(fallback, fallback_action):
                fallback_reason = (
                    fallback_reason
                    or f"{draft.provider} draft failed: {draft.error or 'invalid action'}"
                )
                _mark_success(planner_state, fallback.provider)
                draft = fallback
                action = fallback_action
            else:
                _mark_failure(planner_state, fallback.provider, settings)
        else:
            fallback_reason = fallback_reason or (
                f"{draft.provider} draft failed: {draft.error or 'invalid action'}"
            )
    else:
        _mark_success(planner_state, draft.provider)

    rationale = extract_rationale(draft.text)
    canonical_text = draft.text
    critique: PlannerCritique | None = None
    degraded = False
    # Snapshot the planner's ORIGINAL action before _reconcile can substitute it
    # (a dict copy so later mutation of `action` cannot alias it).
    draft_action = dict(action) if isinstance(action, dict) else action

    if not action:
        decision = PlannerDecision(
            action=None,
            rationale=rationale,
            session_id=session_update,
            canonical_text=canonical_text,
            draft_text=draft.text,
            draft_provider=draft.provider,
            mode=settings.mode,
            degraded=True,
            fallback_reason=fallback_reason or "no usable draft action",
            predicted_objectives=peaf.extract_predicted_objectives(draft.text),
        )
        _archive_decision(decision, planner_state)
        return decision

    if _should_critique(settings, action, stagnation_signal):
        if draft.provider == critic_name:
            degraded = True
        elif _circuit_is_open(planner_state, critic_name):
            degraded = True
        else:
            active_critique = settings.mode.strip().lower() == "draft_critique"
            critic_provider = provider_factory(critic_name)
            critique_prompt = build_critique_prompt(prompt, draft.text, action, rationale)
            critique_result = critic_provider.invoke(
                critique_prompt,
                role="critique",
                session_id=None,
                timeout=timeout,
                cwd=cwd,
            )
            if critique_result.ok:
                critique = extract_critique(critique_result.text)
                critique.provider = critique_result.provider
                if critique.parse_error:
                    # The critic invoke "succeeded" (nonzero text) but the text
                    # is NOT a valid critique block — e.g. Codex emitted prose
                    # or an error message instead of the json:autopilot_critique
                    # fence. Do NOT let the dataclass's default decision="approve"
                    # become a silent rubber-stamp (fail-open). Treat it as a
                    # FAILED critique: mark failure (feeds circuit breaker),
                    # degrade, and in draft_critique mode force decision="reject"
                    # so _reconcile routes the risky action to the safe fallback
                    # rather than admitting it unreviewed.
                    _mark_failure(planner_state, critique_result.provider, settings)
                    degraded = True
                    if settings.mode.strip().lower() == "draft_critique":
                        critique.decision = "reject"
                    action, rationale, canonical_text = _reconcile(
                        action,
                        rationale,
                        draft.text,
                        critique,
                        active=active_critique,
                    )
                else:
                    _mark_success(planner_state, critique_result.provider)
                    action, rationale, canonical_text = _reconcile(
                        action,
                        rationale,
                        draft.text,
                        critique,
                        active=active_critique,
                    )
            else:
                _mark_failure(planner_state, critique_result.provider, settings)
                degraded = True
                critique = PlannerCritique(
                    decision="reject" if active_critique else "approve",
                    raw_text=critique_result.text,
                    provider=critique_result.provider,
                    parse_error=critique_result.error or "critique failed",
                )
                action, rationale, canonical_text = _reconcile(
                    action,
                    rationale,
                    draft.text,
                    critique,
                    active=active_critique,
                )

    decision = PlannerDecision(
        action=action,
        rationale=rationale,
        session_id=session_update,
        canonical_text=canonical_text,
        draft_text=draft.text,
        draft_provider=draft.provider,
        mode=settings.mode,
        degraded=degraded or bool(fallback_reason),
        fallback_reason=fallback_reason,
        critic_provider=critique.provider if critique else "",
        critique=critique,
        predicted_objectives=peaf.extract_predicted_objectives(canonical_text),
        draft_action=draft_action,
    )
    _archive_decision(decision, planner_state)
    return decision


def uncritiqued_dispatch_block_reason(decision: PlannerDecision) -> str:
    """Pause reason if a DEGRADED decision must NOT dispatch its action.

    When the planner ran degraded with NO critic verdict (e.g. the draft provider
    returned empty and the fallback draft IS the critic provider, so no critique
    ran — `degraded and critique is None`), the action is uncritiqued. It may
    dispatch ONLY if it is explicitly observational (OBSERVATIONAL_ACTIONS, empty
    by default); otherwise return "critic_unavailable" so the caller PAUSES for
    operator review instead of running it unreviewed. seed_batch is deliberately
    not observational (low-risk seed looping was the failure the critic catches).

    Returns "" when a real critique exists, when not degraded, or when there is no
    dict action (the no-action path is handled separately by the caller).
    """
    if not decision.degraded or decision.critique is not None:
        return ""
    action = decision.action
    if not isinstance(action, dict):
        return ""
    if action.get("type") in OBSERVATIONAL_ACTIONS:
        return ""
    return "critic_unavailable"


def build_critique_prompt(
    planner_prompt: str,
    draft_text: str,
    action: dict[str, Any],
    rationale: dict[str, Any],
) -> str:
    return f"""\
You are the secondary AutoPilot planner reviewer.

Review the draft below. Do not invent an independent competing plan. Your job
is to find missing constraints, unsafe assumptions, weak attribution, missing
validation, stale context, or an action that violates the single-variable rule.

Your verdict is BINDING (draft_critique mode): a `reject` routes the action to a
safe fallback, and a `revise` with a valid `revised_action` REPLACES the draft.
Reject or revise (preferring a concrete `revised_action`) if the draft does any
of the following — the relevant evidence is in the Original Planner Context:
  - re-proposes a feature flag whose dependencies are not all currently ON (see
    the "Feature Flags" section: live state + dependency rules). Prefer a
    `revised_action` that enables the missing dependency first (one flag/trial).
  - repeats the "Last Non-Executing Action" (the validator/dispatch already
    rejected it; re-proposing it burns a trial and will be auto-blacklisted).
  - matches a "Blacklisted Configurations" entry.
Do NOT manufacture host-noise / contention narratives when System Health is
nominal, and do NOT propose operator-domain actions (widening safety-gate
thresholds, baseline refresh) — those are outside the autopilot action space.

Return JSON ONLY in this fenced block:

```json:autopilot_critique
{{"decision":"approve|revise|reject",
 "confidence":0.0,
 "issues":["short issue"],
 "revised_action":null,
 "revised_rationale":null}}
```

Use revise only when you can provide a better single canonical action. Use
reject only when the draft is unsafe or unsupported and no safe revision is
available.

## Original Planner Context

{planner_prompt}

## Draft Text

{draft_text}

## Parsed Draft Action

{json.dumps(action, indent=2, sort_keys=True)}

## Parsed Draft Rationale

{json.dumps(rationale, indent=2, sort_keys=True)}
"""


def extract_critique(text: str) -> PlannerCritique:
    raw = text or ""
    data, error = _extract_json_payload(raw, "```json:autopilot_critique")
    if data is None:
        return PlannerCritique(raw_text=raw, parse_error=error or "no critique JSON")

    decision = str(data.get("decision", "approve")).strip().lower()
    if decision not in {"approve", "revise", "reject"}:
        decision = "approve"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    issues_raw = data.get("issues", [])
    if isinstance(issues_raw, list):
        issues = [str(i) for i in issues_raw if str(i).strip()]
    elif issues_raw:
        issues = [str(issues_raw)]
    else:
        issues = []

    revised_action = data.get("revised_action")
    if not isinstance(revised_action, dict):
        revised_action = None
    revised_rationale = data.get("revised_rationale")
    if not isinstance(revised_rationale, dict):
        revised_rationale = None

    return PlannerCritique(
        decision=decision,
        confidence=confidence,
        issues=issues,
        revised_action=revised_action,
        revised_rationale=revised_rationale,
        raw_text=raw,
    )


def action_risk(action: dict[str, Any]) -> str:
    action_type = action.get("type", "")
    if action_type == "seed_batch":
        try:
            n_questions = int(action.get("n_questions", 0))
        except (TypeError, ValueError):
            n_questions = 0
        return "medium" if n_questions > 50 else "low"
    if action_type in LOW_RISK_ACTIONS:
        return "low"
    if action_type in MEDIUM_RISK_ACTIONS:
        return "medium"
    if action_type in HIGH_RISK_ACTIONS:
        return "high"
    return "high"


def _should_critique(
    settings: PlannerSettings,
    action: dict[str, Any],
    stagnation_signal: str,
) -> bool:
    mode = settings.mode.strip().lower()
    if mode in {"single", "fallback"}:
        return False
    if mode not in {"draft_critique", "shadow_critique"}:
        return False

    policy = settings.critique_policy.strip().lower()
    risk = action_risk(action)
    stagnating = bool(stagnation_signal and stagnation_signal != "none")
    if policy == "always":
        return True
    if policy == "stagnation":
        return stagnating
    if policy == "high_risk":
        return risk == "high" or stagnating
    if policy == "medium_plus":
        return risk in {"medium", "high"} or stagnating
    return False


def _reconcile(
    action: dict[str, Any],
    rationale: dict[str, Any],
    draft_text: str,
    critique: PlannerCritique,
    *,
    active: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not active or critique.decision == "approve":
        return action, rationale, draft_text

    revised = critique.revised_action
    if revised and _action_validation_error(revised) is None:
        new_rationale = critique.revised_rationale or rationale
        return revised, new_rationale, _canonical_text_from_parts(revised, new_rationale)

    if critique.decision == "reject":
        safe_action = {
            "type": "seed_batch",
            "n_questions": 10,
            "suites": ["coder", "math"],
        }
        safe_rationale = {
            "falsifier": "safe fallback fails to improve trustworthy evidence",
            "rubric_scores": {},
        }
        return (
            safe_action,
            safe_rationale,
            _canonical_text_from_parts(
                safe_action,
                safe_rationale,
            ),
        )

    return action, rationale, draft_text


def _canonical_text_from_parts(
    action: dict[str, Any],
    rationale: dict[str, Any],
) -> str:
    return (
        "```json:autopilot_actions\n"
        f"{json.dumps(action, indent=2, sort_keys=True)}\n"
        "```\n\n"
        "```json:autopilot_rationale\n"
        f"{json.dumps(rationale, indent=2, sort_keys=True)}\n"
        "```\n"
    )


def _draft_is_usable(
    result: PlannerProviderResult,
    action: dict[str, Any] | None,
) -> bool:
    return result.ok and action is not None and _action_validation_error(action) is None


def _action_validation_error(action: dict[str, Any] | None) -> str | None:
    if not action:
        return "missing action"
    action_type = action.get("type")
    if action_type not in KNOWN_ACTION_TYPES:
        return f"unknown action type: {action_type}"
    return validate_single_variable(action)


def _extract_json_payload(
    text: str,
    marker: str,
) -> tuple[dict[str, Any] | None, str]:
    if marker in text:
        start = text.index(marker) + len(marker)
        try:
            end = text.index("```", start)
        except ValueError:
            return None, "unclosed fenced block"
        payload = text[start:end].strip()
    elif "```json" in text:
        start = text.index("```json") + len("```json")
        try:
            end = text.index("```", start)
        except ValueError:
            return None, "unclosed json block"
        payload = text[start:end].strip()
    else:
        payload = text.strip()

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "payload is not an object"
    return data, ""


def _circuit_is_open(planner_state: dict[str, Any], provider: str) -> bool:
    state = planner_state.get(provider, {})
    try:
        return float(state.get("circuit_open_until", 0.0)) > time.time()
    except (TypeError, ValueError):
        return False


def _mark_success(planner_state: dict[str, Any], provider: str) -> None:
    state = planner_state.setdefault(provider, {})
    state["last_success"] = time.time()
    state["failures"] = 0
    state["circuit_open_until"] = 0.0


def _mark_failure(
    planner_state: dict[str, Any],
    provider: str,
    settings: PlannerSettings,
) -> None:
    state = planner_state.setdefault(provider, {})
    failures = int(state.get("failures", 0)) + 1
    state["failures"] = failures
    state["last_failure"] = time.time()
    if failures >= settings.circuit_failures:
        state["circuit_open_until"] = time.time() + settings.circuit_cooldown_s


def _archive_decision(
    decision: PlannerDecision,
    planner_state: dict[str, Any],
) -> None:
    critique = decision.critique
    _append_planner_archive(
        {
            "ts": time.time(),
            "type": "planner_coordinator",
            "mode": decision.mode,
            "draft_provider": decision.draft_provider,
            "critic_provider": decision.critic_provider,
            "degraded": decision.degraded,
            "fallback_reason": decision.fallback_reason,
            "action_type": (decision.action or {}).get("type"),
            "critique_decision": critique.decision if critique else "",
            "critique_confidence": critique.confidence if critique else 0.0,
            "critique_issues": critique.issues if critique else [],
            "planner_state": planner_state,
        }
    )


def _normalize_provider(name: str) -> str:
    return (name or "claude").strip().lower()


def _other_provider(name: str) -> str:
    return "codex" if name == "claude" else "claude"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
