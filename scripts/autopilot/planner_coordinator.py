"""Draft, critique, fallback, and reconciliation for AutoPilot planning."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import peaf

from controller_io import (
    _append_planner_archive,
    _loads_json_payload,
    _open_planner_tap,
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
SAFE_FALLBACK_NUMERIC_SURFACE = "memrl_retrieval"

# Actions safe to DISPATCH WITHOUT a critic verdict (degraded/uncritiqued mode).
# Intentionally EMPTY: when the binding critic is unavailable we PAUSE by default
# rather than run anything unreviewed. NOTE this is deliberately stricter than
# LOW_RISK_ACTIONS — seed_batch is NOT here, because low-risk seed looping was the
# exact failure the critic exists to catch (critic_reject_loop @708). Extend ONLY
# with genuinely observational, non-mutating, non-looping action types.
OBSERVATIONAL_ACTIONS: frozenset[str] = frozenset()
_CRITIQUE_CONTEXT_HEADINGS = (
    "### Evidence Power and Sequential Candidate Status",
    "### System Health",
    "### Action Availability",
    "### Blacklisted Configurations",
    "### Feature Flags",
    "### Last Non-Executing Action",
)
_CRITIQUE_SECTION_CHAR_LIMIT = 1600

# Claude planner drafts run a ~80KB prompt WITH tool access (Read/Grep/Glob) and
# take 60-228s in practice; the old hard 300s default intermittently KILLED a
# slow-but-valid draft → "empty response" → degraded (and, since 2026-06-07, a
# critic_unavailable pause that stalled the run, e.g. @711). Give generous
# headroom; operator-tunable via AUTOPILOT_PLANNER_TIMEOUT. (2026-06-09)
DEFAULT_PLANNER_TIMEOUT = int(os.environ.get("AUTOPILOT_PLANNER_TIMEOUT", "600"))
DEFAULT_SPEND_BREAKER_LOCAL_PRIMARY = "local_frontdoor"
DEFAULT_SPEND_BREAKER_LOCAL_CRITIC = "local_worker"
DEFAULT_CODEX_CRITIC_FALLBACK = "claude"


@dataclass
class PlannerSettings:
    primary: str = "claude"
    critic: str = "codex"
    # 2026-06-04: default flipped shadow_critique -> draft_critique. The critic is
    # now BINDING: a valid revised_action is dispatched, and a reject routes to a
    # metric-bearing fallback (see _reconcile). Shadow mode logged 320 reject/revise
    # verdicts that were all ignored (264 dispatched anyway) while the run dead-locked.
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
    # (revised_action in revise, or the metric fallback in reject). The main loop
    # uses this to record a critic-rejected/revised draft into the invalid-action
    # feedback + blacklist so a substituted draft cannot silently escape the
    # feedback loop (draft_critique authority change, 2026-06-04).
    draft_action: dict[str, Any] | None = None
    # True iff `action is None` AND every model we attempted was an AVAILABILITY
    # failure (timeout / empty / rc / not-found / exception, or circuit-open meant
    # we could not even attempt) — i.e. NO attempted model produced a usable
    # RESPONSE at all. False when some model responded ok but with bad/unparseable
    # content. The main loop uses this to pick deterministic-fallback vs pause when
    # both planner models are offline (cross-model failover, 2026-06-12).
    providers_unavailable: bool = False
    # Ordered provider attempt telemetry for planner_archive.jsonl. This is
    # observability only; dispatch semantics stay derived from action/critique.
    provider_trace: list[dict[str, Any]] = field(default_factory=list)


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


def _apply_planner_spend_breaker(
    settings: PlannerSettings,
    planner_state: dict[str, Any],
) -> tuple[PlannerSettings, bool]:
    if not _env_bool("AUTOPILOT_PLANNER_SPEND_BREAKER", True):
        planner_state.pop("_spend_breaker", None)
        return settings, False

    try:
        from scripts.economics.ledger import summarize_economics

        ledger = summarize_economics(days=_env_int("AUTOPILOT_PLANNER_SPEND_DAYS", 7))
    except Exception as exc:
        log.warning("Planner spend breaker unavailable: %s", exc)
        planner_state["_spend_breaker"] = {
            "active": False,
            "error": str(exc),
        }
        return settings, False

    review = ledger.review
    rules = ledger.rules
    active = bool(review.planner_spend_triggered)
    state = {
        "active": active,
        "projected_monthly_planner_spend_usd": round(
            review.projected_monthly_planner_spend_usd,
            6,
        ),
        "threshold_usd": round(rules.planner_monthly_spend_threshold_usd, 6),
        "planner_spend_usd": round(ledger.planner.total_usd, 6),
        "days": ledger.days,
    }
    planner_state["_spend_breaker"] = state
    if not active:
        return settings, False

    local_primary = _normalize_provider(
        os.environ.get(
            "AUTOPILOT_PLANNER_SPEND_BREAKER_PRIMARY",
            DEFAULT_SPEND_BREAKER_LOCAL_PRIMARY,
        )
    )
    spend_breaker_critic = os.environ.get("AUTOPILOT_PLANNER_SPEND_BREAKER_CRITIC")
    configured_critic = _normalize_provider(settings.critic)
    local_critic = _normalize_provider(
        spend_breaker_critic
        if spend_breaker_critic is not None
        else (
            configured_critic
            if _model_of(configured_critic) == "local"
            else DEFAULT_SPEND_BREAKER_LOCAL_CRITIC
        )
    )
    state["local_primary"] = local_primary
    state["local_critic"] = local_critic
    state["previous_primary"] = _normalize_provider(settings.primary)
    state["previous_critic"] = _normalize_provider(settings.critic)
    log.warning(
        "Planner spend breaker active: projected monthly planner spend $%.2f "
        "exceeds threshold $%.2f; using local providers %s/%s",
        review.projected_monthly_planner_spend_usd,
        rules.planner_monthly_spend_threshold_usd,
        local_primary,
        local_critic,
    )
    return (
        replace(
            settings,
            primary=local_primary,
            critic=local_critic,
        ),
        True,
    )


def _write_trial_planning_banner(trial_id: int | None) -> None:
    """Emit a big, scannable ``>>>> TRIAL N <<<<`` header into the planner tap.

    Best-effort and self-contained: opens its own append handle so it never
    disturbs the per-provider tap writes. The wide arrow rule makes the start of
    each trial's planning trivial to spot while scrolling the raw log or the
    dashboard planner panel.
    """
    if trial_id is None:
        return
    label = f"TRIAL {trial_id}"
    ts = datetime.now().isoformat(timespec="seconds")
    banner = (
        f"\n{'>' * 24} {label} {'<' * 24}\n"
        f"[{ts}] planning cycle start\n"
    )
    tap = _open_planner_tap()
    if tap is None:
        return
    try:
        tap.write(banner)
        tap.flush()
    finally:
        try:
            tap.close()
        except Exception:
            pass


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
    allowed_action_types: Iterable[str] | None = None,
    trial_id: int | None = None,
) -> PlannerDecision:
    """Draft a canonical planner action with optional secondary critique."""
    # One deterministic, unmistakable banner per live planning cycle. Tests and
    # helper calls omit trial_id, so they do not write to the shared planner tap.
    _write_trial_planning_banner(trial_id)
    settings = settings or load_planner_settings_from_env()
    planner_state = planner_state if planner_state is not None else {}
    settings, spend_breaker_active = _apply_planner_spend_breaker(settings, planner_state)
    session_update = None
    allowed_actions = (
        frozenset(str(item) for item in allowed_action_types)
        if allowed_action_types is not None
        else None
    )

    primary_name = _normalize_provider(settings.primary)
    critic_name = _normalize_provider(settings.critic)
    # Compare MODELS, not provider names: the failover draft MUST target a
    # different underlying model. With PRIMARY=codex + CRITIC=codex_critic the
    # names differ but both resolve to the codex binary — a name-based fallback
    # would re-hit codex when codex is offline → "no usable draft action".
    fallback_name = _draft_fallback_provider_name(
        primary_name,
        critic_name,
        spend_breaker_active=spend_breaker_active,
    )
    allow_fallback = settings.mode.strip().lower() != "single"

    draft_provider_name = primary_name
    fallback_reason = ""
    primary_circuit_open_before_draft = _circuit_is_open(planner_state, primary_name)
    if allow_fallback and primary_circuit_open_before_draft:
        draft_provider_name = fallback_name
        fallback_reason = f"{primary_name} circuit open"

    draft_provider = provider_factory(draft_provider_name)
    draft_resume_id = session_id if draft_provider.supports_resume else None
    draft = draft_provider.invoke(
        prompt,
        role="draft",
        session_id=draft_resume_id,
        timeout=timeout,
        cwd=cwd,
    )
    if draft_provider.supports_resume:
        session_update = draft.session_id

    # Track whether ANY attempted model produced a usable RESPONSE (result.ok),
    # regardless of whether its content parsed. Distinguishes "both planner models
    # unavailable" (deterministic fallback / pause) from "models reachable but
    # drafted bad content" (keep the seed_batch default). (2026-06-12)
    any_response_ok = bool(draft.ok)
    provider_trace: list[dict[str, Any]] = []

    action = extract_action(draft.text)
    draft_unusable = _draft_unusable_reason(
        draft,
        action,
        allowed_action_types=allowed_actions,
    )
    provider_trace.append(
        _draft_provider_event(
            stage="draft_primary",
            result=draft,
            action=action,
            unusable_reason=draft_unusable,
        )
    )
    if draft_unusable:
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
            any_response_ok = any_response_ok or bool(fallback.ok)
            fallback_action = extract_action(fallback.text)
            fallback_unusable = _draft_unusable_reason(
                fallback,
                fallback_action,
                allowed_action_types=allowed_actions,
            )
            provider_trace.append(
                _draft_provider_event(
                    stage="draft_fallback",
                    result=fallback,
                    action=fallback_action,
                    unusable_reason=fallback_unusable,
                )
            )
            if not fallback_unusable:
                session_update = fallback.session_id if fallback_provider.supports_resume else None
                fallback_reason = (
                    fallback_reason or f"{draft.provider} draft failed: {draft_unusable}"
                )
                _mark_success(planner_state, fallback.provider)
                draft = fallback
                action = fallback_action
            else:
                _mark_failure(planner_state, fallback.provider, settings)
        else:
            fallback_reason = fallback_reason or (
                f"{draft.provider} draft failed: {draft_unusable}"
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
        # providers_unavailable iff NO attempted model produced a usable response
        # at all (every attempt was an availability failure / circuit-open). If any
        # model responded ok but with bad content, this is a CONTENT failure → leave
        # it False (the loop keeps its seed_batch default). (2026-06-12)
        providers_unavailable = not any_response_ok
        no_action_reason = (
            "both planner models unavailable (no usable response from any attempted model)"
            if providers_unavailable
            else (fallback_reason or "no usable draft action")
        )
        decision = PlannerDecision(
            action=None,
            rationale=rationale,
            session_id=session_update,
            canonical_text=canonical_text,
            draft_text=draft.text,
            draft_provider=draft.provider,
            mode=settings.mode,
            degraded=True,
            fallback_reason=no_action_reason,
            predicted_objectives=peaf.extract_predicted_objectives(draft.text),
            providers_unavailable=providers_unavailable,
            provider_trace=provider_trace,
        )
        _archive_decision(decision, planner_state)
        return decision

    if _should_critique(settings, action, stagnation_signal):
        active_critique = settings.mode.strip().lower() == "draft_critique"
        critique_provider_name = critic_name
        allow_primary_fallback_critique = False
        if draft.provider == critic_name:
            # A fallback draft from the configured critic provider is not
            # independently reviewed yet. Try the original primary as the
            # reviewer if it is available; otherwise keep fail-closed behavior.
            allow_primary_fallback_critique = (
                fallback_reason
                and primary_name != draft.provider
                and not primary_circuit_open_before_draft
            )
            if allow_primary_fallback_critique and (
                _model_of(primary_name) != "local"
                or spend_breaker_active
                or _distinct_local_roles(primary_name, draft.provider)
            ):
                critique_provider_name = primary_name
            else:
                degraded = True

        if not (degraded and critique is None):
            critic_fallback_name = _critic_fallback_provider_name(
                critique_provider_name,
                draft.provider,
            )
            if (
                _circuit_is_open(planner_state, critique_provider_name)
                and not allow_primary_fallback_critique
                and critic_fallback_name
                and not _circuit_is_open(planner_state, critic_fallback_name)
            ):
                critique_provider_name = critic_fallback_name
            elif (
                _circuit_is_open(planner_state, critique_provider_name)
                and not allow_primary_fallback_critique
            ):
                # Critic circuit is open (it failed repeatedly and is cooling down),
                # but the PRIMARY draft succeeded. Treat as critic-unavailable on a
                # TRUSTED draft: keep the draft and let the dispatch gate proceed for
                # low/medium-risk actions (pause only for HIGH risk) rather than
                # discarding a good primary draft for a stale seed_batch fallback.
                # (2026-06-10)
                degraded = True
                critique = PlannerCritique(
                    decision="unavailable",
                    provider=critique_provider_name,
                    parse_error="critic circuit open",
                )
            else:
                critic_provider = provider_factory(critique_provider_name)
                critique_prompt = build_critique_prompt(
                    prompt,
                    draft.text,
                    action,
                    rationale,
                    allowed_action_types=allowed_actions,
                )
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
                    provider_trace.append(
                        _critique_provider_event(
                            stage="critique_primary",
                            result=critique_result,
                            critique=critique,
                        )
                    )
                    if critique.parse_error:
                        # The critic invoke "succeeded" (nonzero text) but the text
                        # is NOT a valid critique block — e.g. Codex emitted prose
                        # or an error message instead of the json:autopilot_critique
                        # fence. This is a FAILED REVIEW, not a rejection of the draft.
                        # Mark failure (feeds circuit breaker) and degrade. In binding
                        # (draft_critique) mode flag the verdict "unavailable" and KEEP
                        # the trusted-primary draft: the dispatch gate then proceeds for
                        # low/medium-risk actions and pauses only for HIGH-risk ones.
                        # We deliberately do NOT force a `reject` → seed_batch fallback:
                        # a critic *failure* must not discard a good primary draft, and
                        # the stale-seed substitution is what re-triggered the
                        # critic_reject_loop halt @708. A genuine parsed `reject` (no
                        # parse_error) still routes to the safe fallback below. (2026-06-10)
                        _mark_failure(planner_state, critique_result.provider, settings)
                        fallback_critique = _try_fallback_critic(
                            provider_factory=provider_factory,
                            fallback_name=critic_fallback_name,
                            failed_provider=critique_result.provider,
                            critique_prompt=critique_prompt,
                            timeout=timeout,
                            cwd=cwd,
                            planner_state=planner_state,
                            settings=settings,
                            provider_trace=provider_trace,
                        )
                        if fallback_critique and not fallback_critique.parse_error:
                            critique = fallback_critique
                            action, rationale, canonical_text = _reconcile(
                                action,
                                rationale,
                                draft.text,
                                critique,
                                active=active_critique,
                                allowed_action_types=allowed_actions,
                            )
                        else:
                            degraded = True
                            if fallback_critique:
                                critique = fallback_critique
                            if active_critique:
                                critique.decision = "unavailable"
                            # keep draft action/rationale/canonical_text (no _reconcile)
                    else:
                        _mark_success(planner_state, critique_result.provider)
                        action, rationale, canonical_text = _reconcile(
                            action,
                            rationale,
                            draft.text,
                            critique,
                            active=active_critique,
                            allowed_action_types=allowed_actions,
                        )
                else:
                    # The critic invoke FAILED outright (timeout / empty / nonzero rc).
                    # Same principle as the parse_error branch: a failed *review* must
                    # not discard the trusted-primary draft. Binding mode → verdict
                    # "unavailable" + KEEP the draft (gate proceeds for low/medium risk,
                    # pauses for HIGH risk). Shadow mode → non-binding, fail-open to the
                    # draft. Neither substitutes the stale seed_batch fallback. (2026-06-10)
                    _mark_failure(planner_state, critique_result.provider, settings)
                    provider_trace.append(
                        _critique_provider_event(
                            stage="critique_primary",
                            result=critique_result,
                            critique=None,
                        )
                    )
                    fallback_critique = _try_fallback_critic(
                        provider_factory=provider_factory,
                        fallback_name=critic_fallback_name,
                        failed_provider=critique_result.provider,
                        critique_prompt=critique_prompt,
                        timeout=timeout,
                        cwd=cwd,
                        planner_state=planner_state,
                        settings=settings,
                        provider_trace=provider_trace,
                    )
                    if fallback_critique and not fallback_critique.parse_error:
                        critique = fallback_critique
                        action, rationale, canonical_text = _reconcile(
                            action,
                            rationale,
                            draft.text,
                            critique,
                            active=active_critique,
                            allowed_action_types=allowed_actions,
                        )
                    else:
                        degraded = True
                        critique = fallback_critique or PlannerCritique(
                            decision="unavailable" if active_critique else "approve",
                            raw_text=critique_result.text,
                            provider=critique_result.provider,
                            parse_error=critique_result.error or "critique failed",
                        )
                        if active_critique:
                            critique.decision = "unavailable"
                        # keep draft action/rationale/canonical_text (no _reconcile)

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
        provider_trace=provider_trace,
    )
    _archive_decision(decision, planner_state)
    return decision


def uncritiqued_dispatch_block_reason(
    decision: PlannerDecision,
    *,
    is_blacklisted: bool = False,
    is_repeated: bool = False,
) -> str:
    """Pause reason if a DEGRADED decision must NOT dispatch its action.

    Two distinct degraded shapes, handled differently (2026-06-10):

    * NO critique object at all (`critique is None`): the PRIMARY failed to draft
      and the draft fell back to the critic provider (or the critic circuit opened
      before any draft existed). The action is uncritiqued AND not from the trusted
      primary → dispatch ONLY if explicitly observational (OBSERVATIONAL_ACTIONS,
      empty by default), else pause `critic_unavailable` for operator review.
      seed_batch is deliberately not observational (low-risk seed looping was the
      failure the critic catches — critic_reject_loop @708).

    * Critic verdict `"unavailable"`: the PRIMARY drafted fine but the binding
      critic could not render a verdict (timeout / empty / unparseable / circuit
      open). A failed *review* must not discard a good primary draft — but RISK
      CLASS ALONE IS NOT A SUFFICIENT GUARD (the @708 failure was "low-risk" seed
      looping). Tightened dispatch rule:
        - HIGH risk → pause.
        - seed_batch / passive low-risk actions (the loop-prone class) → pause,
          unless explicitly observational + one-shot (OBSERVATIONAL_ACTIONS).
        - MEDIUM-risk experiment → proceed ONLY IF novel and non-looping:
          not `is_blacklisted`, not `is_repeated` (a recurring invalid signature),
          and carrying a real `falsifier` hypothesis; else pause.
      `is_blacklisted` / `is_repeated` are supplied by the caller (which holds the
      blacklist + invalid-signature state). Shadow (non-binding) mode never blocks.

    Returns "" when a real verdict exists (approve/reject/revise, already
    reconciled), when not degraded, or when there is no dict action (the no-action
    path is handled separately by the caller).
    """
    if not decision.degraded:
        return ""
    action = decision.action
    if not isinstance(action, dict):
        return ""
    crit = decision.critique
    # Case A — no trusted primary draft (no critique object).
    if crit is None:
        if decision.fallback_reason and _model_of(decision.draft_provider) == "codex":
            # Operator-approved rollout mode: if the local drafter fails but Codex
            # produces a schema-valid fallback draft, dispatch it and learn from
            # the visible provider/fallback telemetry instead of pausing the loop.
            return ""
        if action.get("type") in OBSERVATIONAL_ACTIONS:
            return ""
        return "critic_unavailable"
    # Case B — trusted primary draft, but the binding critic could not review it.
    if crit.decision == "unavailable":
        # A fallback draft whose independent reviewer was also unavailable is not
        # a trusted-primary draft. Keep the stricter Case-A gate.
        if decision.fallback_reason:
            return "" if action.get("type") in OBSERVATIONAL_ACTIONS else "critic_unavailable"
        if decision.mode.strip().lower() != "draft_critique":
            return ""  # shadow / non-binding: advisory critic, never blocks
        atype = action.get("type", "")
        risk = action_risk(action)
        # HIGH risk → never auto-run an unreviewed structural/code/registry change.
        if risk == "high":
            return "critic_unavailable"
        # seed_batch / passive low-risk actions are the LOOP-PRONE class the critic
        # exists to catch (the @708 failure was "low-risk" seed looping). Risk class
        # alone is NOT enough — pause unless the action is EXPLICITLY observational
        # and one-shot (membership in OBSERVATIONAL_ACTIONS, empty by default).
        if atype in LOW_RISK_ACTIONS:
            return "" if atype in OBSERVATIONAL_ACTIONS else "critic_unavailable"
        # MEDIUM-risk experiment may proceed ONLY IF it is genuinely novel and not
        # loop-keeping: not blacklisted, not a repeated (recurring-invalid) action,
        # and carrying a real falsifiable hypothesis. Otherwise pause for review.
        if risk == "medium":
            if is_blacklisted or is_repeated:
                return "critic_unavailable"
            rationale = decision.rationale if isinstance(decision.rationale, dict) else {}
            if not str(rationale.get("falsifier", "")).strip():
                return "critic_unavailable"  # no hypothesis → treat as loop-keeping
            return ""
        # Any other class → pause (safe default).
        return "critic_unavailable"
    # A real verdict was already reconciled upstream → never blocks here.
    return ""


def build_critique_prompt(
    planner_prompt: str,
    draft_text: str,
    action: dict[str, Any],
    rationale: dict[str, Any],
    *,
    allowed_action_types: Iterable[str] | None = None,
) -> str:
    selected_context = _selected_critique_context(planner_prompt)
    action_type = str(action.get("type", ""))
    known_actions_text = ", ".join(sorted(KNOWN_ACTION_TYPES))
    if allowed_action_types is None:
        selectable_actions_text = "(not supplied; use the Action Availability section)"
    else:
        selectable_actions_text = ", ".join(sorted({str(item) for item in allowed_action_types}))
    return f"""\
You are the secondary AutoPilot planner reviewer.

Review the draft below. Do not invent an independent competing plan. Your job
is to find missing constraints, unsafe assumptions, weak attribution, missing
validation, stale context, or an action that violates the single-variable rule.

Your verdict is BINDING (draft_critique mode): a `reject` routes the action to a
metric-bearing fallback, and a `revise` with a valid `revised_action` REPLACES the draft.
Reject or revise (preferring a concrete `revised_action`) if the draft does any
of the following — the relevant evidence is in the selected context below:
  - re-proposes a feature flag whose dependencies are not all currently ON (see
    the "Feature Flags" section: live state + dependency rules). Prefer a
    `revised_action` that enables the missing dependency first (one flag/trial).
  - repeats the "Last Non-Executing Action" (the validator/dispatch already
    rejected it; re-proposing it burns a trial and will be auto-blacklisted).
  - matches a "Blacklisted Configurations" entry.
  - cites below-MDE or single-trial evidence as decisive without a reproduction
    plan; use the "Evidence Power" section to reject unmeasurable proposals.
Do NOT reject a `numeric_trial` solely because `"params": {{}}`. In the action
schema, empty params means "ask NumericSwarm/Optuna for concrete values"; the
dispatcher writes the applied params into the trial action and eval details
before W8 replay/promotion accounting. Historical logged rows that stayed empty
are not replayable, but a new empty-params numeric request is a valid way to
produce a replayable candidate if the selected surface is otherwise allowed.
Do NOT manufacture host-noise / contention narratives when System Health is
nominal, and do NOT propose operator-domain actions (widening safety-gate
thresholds, baseline refresh) — those are outside the autopilot action space.
Do NOT reject solely because the shared worktree has unrelated dirty files.
The dispatcher has a target-path dirty fence: cite dirty state only when the
parsed action would write/stage that same target path or prompt directory.
Do NOT reject a draft by claiming its parsed action type is unrecognized,
non-standard, or not in the AutoPilot schema when it appears in the
authoritative action-type list below. If the selected context says a known
action is temporarily unavailable or cannot satisfy the current W8 replay
pressure, name that concrete availability/evidence reason instead.
`seed_batch` and `deep_eval` are valid known action types. They may be the wrong
choice for a replayable W8 candidate, but they are not schema errors.

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

## Selected Measurement and Constraint Context

{selected_context}

## Authoritative Action-Type Check

- parsed_action_type: `{action_type}`
- known_action_types: `{known_actions_text}`
- currently_selectable_action_types: `{selectable_actions_text}`

## Draft Text

{draft_text}

## Parsed Draft Action

{json.dumps(action, indent=2, sort_keys=True)}

## Parsed Draft Rationale

{json.dumps(rationale, indent=2, sort_keys=True)}
"""


def _critic_fallback_provider_name(
    critic_provider_name: str,
    draft_provider_name: str,
) -> str:
    configured = os.environ.get("AUTOPILOT_PLANNER_CRITIC_FALLBACK")
    if configured is None:
        if _model_of(critic_provider_name) != "codex":
            return ""
        configured = DEFAULT_CODEX_CRITIC_FALLBACK

    fallback = _normalize_provider(configured)
    if fallback in {"", "0", "false", "no", "none", "off"}:
        return ""
    if fallback == _normalize_provider(critic_provider_name):
        return ""
    if _model_of(fallback) == _model_of(draft_provider_name):
        return ""
    return fallback


def _try_fallback_critic(
    *,
    provider_factory: ProviderFactory,
    fallback_name: str,
    failed_provider: str,
    critique_prompt: str,
    timeout: int,
    cwd: Path | str | None,
    planner_state: dict[str, Any],
    settings: PlannerSettings,
    provider_trace: list[dict[str, Any]] | None = None,
) -> PlannerCritique | None:
    if not fallback_name or fallback_name == _normalize_provider(failed_provider):
        return None
    if _circuit_is_open(planner_state, fallback_name):
        return None

    fallback_provider = provider_factory(fallback_name)
    fallback_result = fallback_provider.invoke(
        critique_prompt,
        role="critique",
        session_id=None,
        timeout=timeout,
        cwd=cwd,
    )
    if not fallback_result.ok:
        _mark_failure(planner_state, fallback_result.provider, settings)
        if provider_trace is not None:
            provider_trace.append(
                _critique_provider_event(
                    stage="critique_fallback",
                    result=fallback_result,
                    critique=None,
                )
            )
        return PlannerCritique(
            decision="unavailable",
            raw_text=fallback_result.text,
            provider=fallback_result.provider,
            parse_error=(
                f"{failed_provider} critique failed; "
                f"{fallback_result.provider} fallback failed: "
                f"{fallback_result.error or 'critique failed'}"
            ),
        )

    fallback_critique = extract_critique(fallback_result.text)
    fallback_critique.provider = fallback_result.provider
    if provider_trace is not None:
        provider_trace.append(
            _critique_provider_event(
                stage="critique_fallback",
                result=fallback_result,
                critique=fallback_critique,
            )
        )
    if fallback_critique.parse_error:
        _mark_failure(planner_state, fallback_result.provider, settings)
        fallback_critique.decision = "unavailable"
        fallback_critique.parse_error = (
            f"{failed_provider} critique failed; "
            f"{fallback_result.provider} fallback unparseable: "
            f"{fallback_critique.parse_error}"
        )
        return fallback_critique

    _mark_success(planner_state, fallback_result.provider)
    return fallback_critique


def _selected_critique_context(planner_prompt: str) -> str:
    sections = [
        section
        for heading in _CRITIQUE_CONTEXT_HEADINGS
        if (section := _extract_markdown_section(planner_prompt, heading))
    ]
    if not sections:
        return (
            "(selected planner context unavailable; critique only the parsed draft and rationale)"
        )
    return "\n\n".join(sections)


def _extract_markdown_section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    tail = text[start:]
    next_heading = re.search(r"\n#{2,3}\s+", tail[len(heading) :])
    end = len(heading) + next_heading.start() if next_heading else len(tail)
    section = tail[:end].strip()
    if len(section) <= _CRITIQUE_SECTION_CHAR_LIMIT:
        return section
    return (
        section[:_CRITIQUE_SECTION_CHAR_LIMIT].rstrip() + "\n  ... [truncated for critic context]"
    )


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
    allowed_action_types: Iterable[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not active or critique.decision == "approve":
        return action, rationale, draft_text

    revised = critique.revised_action
    if (
        revised
        and _action_validation_error(
            revised,
            allowed_action_types=allowed_action_types,
        )
        is None
    ):
        new_rationale = critique.revised_rationale or rationale
        return revised, new_rationale, _canonical_text_from_parts(revised, new_rationale)

    if critique.decision == "reject":
        safe_action = {
            "type": "numeric_trial",
            "surface": SAFE_FALLBACK_NUMERIC_SURFACE,
            "params": {},
        }
        safe_rationale = {
            "falsifier": "critic reject numeric fallback fails to produce replayable evidence",
            "rubric_scores": {},
            "critic_reject_numeric_fallback": True,
            "critic_reject_issues": list(critique.issues or []),
            "critic_reject_original_action": dict(action),
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
    return not _draft_unusable_reason(result, action)


def _draft_unusable_reason(
    result: PlannerProviderResult,
    action: dict[str, Any] | None,
    *,
    allowed_action_types: Iterable[str] | None = None,
) -> str:
    """Why a draft is unusable ("" if usable). Surfaces the EXACT schema-validation
    error (e.g. an out-of-range min_memories) rather than an opaque 'invalid action'
    — so the planner fallback log + the next trial's 'Last Non-Executing Action'
    feedback can self-correct instead of re-drafting the same hidden-cap violation
    (root cause of the #776 train_routing_models pause, 2026-06-11)."""
    if not result.ok:
        return result.error or "provider error / empty response"
    if action is None:
        return "no parseable json:autopilot_actions block"
    return _action_validation_error(
        action,
        allowed_action_types=allowed_action_types,
    ) or ""


def _action_validation_error(
    action: dict[str, Any] | None,
    *,
    allowed_action_types: Iterable[str] | None = None,
) -> str | None:
    if not action:
        return "missing action"
    action_type = action.get("type")
    if action_type not in KNOWN_ACTION_TYPES:
        return f"unknown action type: {action_type}"
    if allowed_action_types is not None:
        allowed = {str(item) for item in allowed_action_types}
        if action_type not in allowed:
            return f"action type currently unavailable: {action_type}"
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
        data = _loads_json_payload(payload)
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
            "draft_action": decision.draft_action,
            "final_action": decision.action,
            "provider_trace": decision.provider_trace,
            "planner_state": planner_state,
        }
    )


def _clip(value: Any, limit: int = 300) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit].rstrip() + "...[truncated]"


def _draft_provider_event(
    *,
    stage: str,
    result: PlannerProviderResult,
    action: dict[str, Any] | None,
    unusable_reason: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "role": result.role or "draft",
        "provider": result.provider,
        "ok": bool(result.ok),
        "text_chars": len(result.text or ""),
        "error": _clip(result.error),
        "parse_ok": action is not None,
        "action_type": (action or {}).get("type", ""),
        "unusable_reason": unusable_reason,
    }


def _critique_provider_event(
    *,
    stage: str,
    result: PlannerProviderResult,
    critique: PlannerCritique | None,
) -> dict[str, Any]:
    parse_error = critique.parse_error if critique else ""
    return {
        "stage": stage,
        "role": result.role or "critique",
        "provider": result.provider,
        "ok": bool(result.ok),
        "text_chars": len(result.text or ""),
        "error": _clip(result.error),
        "parse_ok": bool(result.ok and critique and not parse_error),
        "critique_decision": critique.decision if critique else "",
        "critique_confidence": critique.confidence if critique else 0.0,
        "parse_error": _clip(parse_error),
    }


def _normalize_provider(name: str) -> str:
    return (name or "claude").strip().lower()


def _other_provider(name: str) -> str:
    return "codex" if name == "claude" else "claude"


def _distinct_local_roles(left: str, right: str) -> bool:
    return (
        _model_of(left) == "local"
        and _model_of(right) == "local"
        and _normalize_provider(left) != _normalize_provider(right)
    )


def _draft_fallback_provider_name(
    primary_name: str,
    critic_name: str,
    *,
    spend_breaker_active: bool,
) -> str:
    primary = _normalize_provider(primary_name)
    critic = _normalize_provider(critic_name)
    if spend_breaker_active:
        return critic if critic != primary else primary
    if _model_of(critic) != _model_of(primary):
        return critic
    if _distinct_local_roles(primary, critic):
        return critic
    return _other_provider(primary)


# Provider-role names that resolve to the same underlying MODEL binary. Mirrors
# the dispatch set in planner_providers.get_planner_provider: codex_critic /
# codex-critic / codex_reviewer / codex-reviewer all launch the codex binary.
_CODEX_ROLE_NAMES = {"codex", "codex_critic", "codex-critic", "codex_reviewer", "codex-reviewer"}
_LOCAL_ROLE_NAMES = {
    "local",
    "local_frontdoor",
    "frontdoor_local",
    "local_chat",
    "local_chat_planner",
    "chat_local",
    "local_worker",
    "local_worker_general",
    "worker_general_local",
    "local_ingest",
    "local_ingest_long_context",
    "ingest_local",
    "local_brief_frontdoor",
    "local_ingest_frontdoor",
    "local_two_stage",
    "local_brief_worker",
    "local_ingest_worker",
}


def _model_of(name: str) -> str:
    """Canonical underlying model for a provider-role name.

    Cross-model failover compares MODELS, not provider names: with PRIMARY=codex
    and CRITIC=codex_critic both names differ but resolve to the SAME codex binary,
    so a name-based fallback would re-hit codex when codex is offline. (2026-06-12)
    """
    normalized = (name or "").strip().lower()
    if normalized == "claude":
        return "claude"
    if normalized in _CODEX_ROLE_NAMES:
        return "codex"
    if normalized in _LOCAL_ROLE_NAMES:
        return "local"
    return normalized


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}
