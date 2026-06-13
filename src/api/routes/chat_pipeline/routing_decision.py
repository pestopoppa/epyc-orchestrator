"""Routing decision helpers for the chat pipeline."""

from __future__ import annotations

import logging

from src.api.models import ChatRequest
from src.api.routes.chat_routing import _classify_and_route
from src.api.routes.chat_utils import DEFAULT_TIMEOUT_S, ROLE_TIMEOUTS
from src.api.structured_logging import task_extra
from src.features import features
from src.roles import Role

log = logging.getLogger(__name__)

_TIER_COST_WEIGHTS: dict[str, float] = {
    "A": 10.0,
    "B": 3.0,
    "C": 1.0,
    "D": 0.2,
}

_INGRESS_ROLE_ALIASES = {
    "coder": "coder_escalation",
    "worker_coder": "worker_general",
    "worker_code": "worker_general",
    "worker_fast": "worker_general",
}


def normalize_ingress_role(role: object) -> object:
    """Normalize externally supplied role labels before config lookup."""
    if not isinstance(role, str):
        return role
    return _INGRESS_ROLE_ALIASES.get(role, role)


def assess_factual_risk(prompt: str, role: str, task_id: str) -> tuple[float, str]:
    """Return factual-risk score and band, falling back to no-risk on failure."""
    try:
        from src.classifiers.factual_risk import assess_risk, get_mode as _fr_mode

        if _fr_mode() == "off":
            return 0.0, ""
        result = assess_risk(prompt, role=role)
        log.info(
            "Factual risk: score=%.3f band=%s (raw=%.3f adj=%.1f)",
            result.adjusted_risk_score,
            result.risk_band,
            result.risk_score,
            result.role_adjustment,
            extra=task_extra(
                task_id=task_id,
                stage="routing",
                strategy="factual_risk",
            ),
        )
        return result.adjusted_risk_score, result.risk_band
    except Exception as exc:
        log.debug("Factual risk scoring skipped: %s", exc)
        return 0.0, ""


def select_initial_route(
    request: ChatRequest,
    state,
    task_ir: dict,
    use_mock: bool,
    heuristic_priors: dict[str, float],
    classify_and_route=_classify_and_route,
) -> tuple[list, str, str]:
    """Select initial route before risk veto and telemetry enrichment."""
    skill_context = ""
    if use_mock:
        role = normalize_ingress_role(request.role) if request.role else Role.FRONTDOOR
        return [role], "mock", skill_context
    if request.force_role:
        return [normalize_ingress_role(request.force_role)], "forced", skill_context
    if request.role and request.role not in ("", "frontdoor"):
        return [normalize_ingress_role(request.role)], "explicit", skill_context
    if request.image_path or request.image_base64:
        return ["worker_vision"], "vision_input", skill_context
    if state.hybrid_router and request.real_mode:
        if hasattr(state.hybrid_router, "route_with_skills") and features().skillbank:
            routing_decision, routing_strategy, skill_context = (
                state.hybrid_router.route_with_skills(task_ir)
            )
            return routing_decision, routing_strategy, skill_context
        routing_decision, routing_strategy = state.hybrid_router.route(
            task_ir,
            priors=heuristic_priors,
        )
        return routing_decision, routing_strategy, skill_context

    classified_role, routing_strategy = classify_and_route(
        request.prompt,
        request.context or "",
        has_image=bool(request.image_path or request.image_base64),
    )
    return [classified_role], routing_strategy, skill_context


def apply_failure_veto(
    state,
    routing_decision: list,
    routing_strategy: str,
    factual_risk_band: str,
    task_id: str,
) -> tuple[list, str]:
    """Apply failure-graph veto to risky specialist routes."""
    veto_thresholds = {"high": 0.3, "medium": 0.5, "low": 0.7, "": 0.5}
    if not (
        state.failure_graph
        and routing_decision
        and str(routing_decision[0]) != str(Role.FRONTDOOR)
        and routing_strategy not in ("mock", "forced")
    ):
        return routing_decision, routing_strategy

    try:
        risk = state.failure_graph.get_failure_risk(str(routing_decision[0]))
        veto_threshold = veto_thresholds.get(factual_risk_band, 0.5)
        if risk > veto_threshold:
            log.warning(
                "Failure veto: %s risk=%.2f > %.1f (factual_risk=%s), reverting to frontdoor",
                routing_decision[0],
                risk,
                veto_threshold,
                factual_risk_band or "none",
                extra=task_extra(
                    task_id=task_id,
                    role=str(routing_decision[0]),
                    stage="routing",
                    strategy="failure_vetoed",
                ),
            )
            return [str(Role.FRONTDOOR)], "failure_vetoed"
    except Exception as exc:
        log.debug("Failure risk veto check failed: %s", exc)
    return routing_decision, routing_strategy


# Ingest-triviality guard thresholds (chars; ~4 chars/token).
# Long-context payloads (big context) and genuinely hard prompts are never
# demoted — the guard only catches trivially-easy short prompts that the
# learned router (MemRL) leaks onto the 80B accuracy/long-context specialist.
_INGEST_GUARD_MAX_CONTEXT_CHARS = 2000   # above this we assume a real long-context task
_INGEST_GUARD_EASY_MAX_PROMPT_CHARS = 4000      # difficulty band == "easy"
_INGEST_GUARD_UNKNOWN_MAX_PROMPT_CHARS = 400    # difficulty signal off/unavailable: be strict


def apply_ingest_triviality_guard(
    request: ChatRequest,
    routing_decision: list,
    routing_strategy: str,
    difficulty_band: str,
    task_id: str,
) -> tuple[list, str]:
    """Demote trivially-easy short prompts off ``ingest_long_context``.

    ``ingest_long_context`` (Qwen3-Next-80B @ ~6.4 t/s) is a router-target
    accuracy/long-context specialist, but the learned MemRL router leaks some
    trivial short prompts (e.g. one-line arithmetic) onto it — paying a ~19×
    latency tax for work a cheap role answers identically. This guard redirects
    only *positively trivial* requests to ``worker_general``; it never touches
    long-context payloads or prompts the difficulty signal calls medium/hard,
    so short-but-hard reasoning (which ingest legitimately wins) is preserved.

    Opt-in: no-op unless the ``ingest_triviality_guard`` feature flag is on.
    Reuses the already-computed ``difficulty_band`` (no extra classifier call).
    """
    if not features().ingest_triviality_guard:
        return routing_decision, routing_strategy
    if not routing_decision or str(routing_decision[0]) != str(Role.INGEST_LONG_CONTEXT):
        return routing_decision, routing_strategy

    band = (difficulty_band or "").strip().lower()
    # Positive hard/medium evidence wins — never demote possibly-hard reasoning.
    if band in ("medium", "hard"):
        return routing_decision, routing_strategy

    context_chars = len(request.context or "")
    if context_chars > _INGEST_GUARD_MAX_CONTEXT_CHARS:
        return routing_decision, routing_strategy  # genuine long-context work

    prompt_chars = len(request.prompt or "")
    prompt_limit = (
        _INGEST_GUARD_EASY_MAX_PROMPT_CHARS
        if band == "easy"
        else _INGEST_GUARD_UNKNOWN_MAX_PROMPT_CHARS
    )
    if prompt_chars > prompt_limit:
        return routing_decision, routing_strategy

    target = str(Role.WORKER_GENERAL)
    log.info(
        "Ingest-triviality guard: ingest_long_context → %s "
        "(prompt_chars=%d, context_chars=%d, band=%s)",
        target,
        prompt_chars,
        context_chars,
        band or "none",
        extra=task_extra(
            task_id=task_id,
            role=target,
            stage="routing",
            strategy="ingest_triviality_guard",
        ),
    )
    return [target], f"{routing_strategy}:ingest_triviality_guard"


def assess_difficulty(prompt: str, role: str, task_id: str) -> tuple[float, str]:
    """Return difficulty score and band, falling back to no signal on failure."""
    try:
        from src.classifiers.difficulty_signal import assess_difficulty, get_mode as _ds_mode

        if _ds_mode() == "off":
            return 0.0, ""
        result = assess_difficulty(prompt, role=role)
        log.info(
            "Difficulty signal: score=%.3f band=%s",
            result.difficulty_score,
            result.difficulty_band,
            extra=task_extra(
                task_id=task_id,
                stage="routing",
                strategy="difficulty_signal",
            ),
        )
        return result.difficulty_score, result.difficulty_band
    except Exception as exc:
        log.debug("Difficulty signal scoring skipped: %s", exc)
        return 0.0, ""


def estimate_routing_cost(request: ChatRequest, state, routing_decision: list) -> float:
    """Estimate relative route cost for routing telemetry."""
    try:
        role_str = str(routing_decision[0]) if routing_decision else "frontdoor"
        if state.registry:
            role_cfg = state.registry.get_role(role_str)
            tier = getattr(role_cfg, "tier", "C") if role_cfg else "C"
        else:
            tier = "C"
        est_tokens = len(request.prompt) // 4 + len(request.context or "") // 4
        return _TIER_COST_WEIGHTS.get(tier, 1.0) * est_tokens / 1_000_000
    except Exception:
        return 0.0


def routing_meta(
    request: ChatRequest,
    state,
    routing_strategy: str,
    heuristic_priors: dict[str, float],
    factual_risk_score: float,
    factual_risk_band: str,
    difficulty_score: float,
    difficulty_band: str,
    estimated_cost: float,
    assigned_role: str,
) -> dict:
    """Build progress-log routing metadata."""
    meta = {
        "decision_source": routing_strategy,
        "was_forced": bool(request.force_role),
        "heuristic_priors": {
            k: round(v, 4)
            for k, v in sorted(heuristic_priors.items(), key=lambda kv: -kv[1])[:4]
        },
        "factual_risk_score": round(factual_risk_score, 4),
        "factual_risk_band": factual_risk_band,
        "difficulty_score": round(difficulty_score, 4),
        "difficulty_band": difficulty_band,
        "assigned_role": assigned_role,
        "estimated_cost": round(estimated_cost, 6),
    }
    if state.llm_primitives and hasattr(state.llm_primitives, "get_stats"):
        try:
            stats = state.llm_primitives.get_stats()
            meta["active_requests"] = state.active_requests
            if "per_role" in stats:
                meta["queue_depth"] = {
                    role: rs.get("total_active", rs.get("round_robin_requests", 0))
                    for role, rs in stats["per_role"].items()
                    if isinstance(rs, dict)
                }
        except Exception:
            pass
    if state.registry and hasattr(state.registry, "roles"):
        try:
            meta["stack_state"] = {
                name: {
                    "model": str(getattr(cfg.model, "name", cfg.model))[:60],
                    "tier": cfg.tier,
                    "instances": getattr(cfg, "numa_instances", 1),
                }
                for name, cfg in state.registry.roles.items()
                if not name.startswith("draft_")
            }
        except Exception:
            pass
    if state.hybrid_router and hasattr(state.hybrid_router, "last_decision_meta"):
        try:
            router_meta = dict(state.hybrid_router.last_decision_meta or {})
            # Defensive remap: legacy episodic-memory seeds tagged actions as
            # `<role>:react` (a mode unified into REPL 2026-05-25). Normalize
            # here so the routing-decision event reflects current vocabulary
            # even if a stale seed slips through. The hybrid_router writes
            # last_decision_meta from ~7 inline sites; centralizing the
            # rewrite here is cheaper than patching each writer.
            ca = router_meta.get("chosen_action")
            if isinstance(ca, str) and ca.endswith(":react"):
                router_meta["chosen_action"] = ca[:-len(":react")] + ":repl"
            meta.update(router_meta)
        except Exception:
            pass
    # J10 (URE-1): persist uncertainty into the canonical progress event when
    # shadow logging is enabled. The sidecar remains for older ingest jobs, but
    # W7 requires the QScorer/replay path to see these fields directly.
    try:
        from src.features import features
        if features().ure_uncertainty_shadow_log:
            from src.uncertainty_shadow import (
                compute_routing_uncertainty,
                emit_uncertainty_shadow,
            )
            uncertainty = compute_routing_uncertainty(meta)
            meta["uncertainty_score"] = uncertainty["score"]
            meta["uncertainty_components"] = uncertainty["components"]
            meta["uncertainty_n_alternatives"] = uncertainty["n_alternatives"]
            emit_uncertainty_shadow(meta, request_id=getattr(request, "session_id", None))
    except Exception:
        pass
    return meta


def log_routing_start(
    request: ChatRequest,
    state,
    task_id: str,
    task_ir: dict,
    routing_decision: list,
    routing_strategy: str,
    heuristic_priors: dict[str, float],
    factual_risk_score: float,
    factual_risk_band: str,
    difficulty_score: float,
    difficulty_band: str,
    estimated_cost: float,
    assigned_role: str,
) -> None:
    """Log routing start to MemRL progress logger when available."""
    if not state.progress_logger:
        return
    state.progress_logger.log_task_started(
        task_id=task_id,
        task_ir=task_ir,
        routing_decision=routing_decision,
        routing_strategy=routing_strategy,
        routing_meta=routing_meta(
            request,
            state,
            routing_strategy,
            heuristic_priors,
            factual_risk_score,
            factual_risk_band,
            difficulty_score,
            difficulty_band,
            estimated_cost,
            assigned_role,
        ),
    )


def resolve_timeout(request: ChatRequest, routing_decision: list) -> int:
    """Compute role-specific timeout with request-level clamp."""
    role_str = str(routing_decision[0]) if routing_decision else str(Role.FRONTDOOR)
    timeout_s = ROLE_TIMEOUTS.get(role_str, DEFAULT_TIMEOUT_S)
    if request.timeout_s is not None:
        timeout_s = max(1, min(timeout_s, int(request.timeout_s)))
    return timeout_s


def extract_skill_ids(skill_context: str, state) -> list[str]:
    """Extract skill IDs from skill-augmented routing state."""
    if not (skill_context and hasattr(state, "hybrid_router") and state.hybrid_router):
        return []
    try:
        if hasattr(state.hybrid_router, "_last_skill_ids"):
            return list(state.hybrid_router._last_skill_ids)
    except Exception:
        pass
    return []


def derive_task_type_from_route(task_ir: dict, routing_decision: list) -> None:
    """Mutate task_ir task_type from routed role for tool-policy checks."""
    role_to_task_type = {
        "worker_math": "math",
        "coder_escalation": "coder",
        "coder_primary": "coder",
        "thinking_reasoning": "thinking",
    }
    routed_role = str(routing_decision[0]) if routing_decision else ""
    derived_task_type = role_to_task_type.get(routed_role)
    if derived_task_type:
        task_ir["task_type"] = derived_task_type


def classify_trinity_role(request: ChatRequest, routing_decision: list, task_id: str) -> str:
    """Return shadow Trinity role classification."""
    try:
        from src.classifiers.role_classifier import classify_role as _classify_trinity_role

        result = _classify_trinity_role(
            request.prompt,
            routing_decision=routing_decision,
            force_role=request.force_role,
            thinking_budget=request.thinking_budget,
            context=request.context or "",
        )
        log.info(
            "Trinity role classified: role=%s reason=%s",
            result.role,
            result.reason,
            extra=task_extra(
                task_id=task_id,
                stage="routing",
                strategy="trinity_role_shadow",
            ),
        )
        return result.role
    except Exception as exc:
        log.debug("Trinity role classification skipped: %s", exc)
        return "worker"
