"""Pipeline stages 1-3 + 5: routing, preprocessing, backend init, plan review."""

from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException

from src.api.models import ChatRequest
from src.api.services.memrl import ensure_memrl_initialized
from src.config import get_config
from src.constants import TASK_IR_OBJECTIVE_LEN
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.roles import Role
from src.task_ir import canonicalize_task_ir

from src.api.routes.chat_review import (
    _apply_plan_review,
    _architect_plan_review,
    _needs_plan_review,
    _store_plan_review_episode,
)
from src.api.routes.chat_routing import _classify_and_route
from src.api.routes.chat_routing import _heuristic_role_priors
from src.api.routes.chat_utils import (
    DEFAULT_TIMEOUT_S,
    ROLE_TIMEOUTS,
    RoutingResult,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)

# Relative cost weights by model tier (arbitrary units for Pareto comparison)
_TIER_COST_WEIGHTS: dict[str, float] = {
    "A": 10.0,   # 235B+ architect
    "B": 3.0,    # 32B-70B specialist
    "C": 1.0,    # 7B-14B worker
    "D": 0.2,    # draft/TTS
}


# ── Stage 1: Routing ────────────────────────────────────────────────────


def _route_request(request: ChatRequest, state) -> RoutingResult:
    """Determine routing decision, strategy, and task metadata.

    Produces a RoutingResult that captures all routing decisions made
    before execution begins. Includes failure graph veto and MemRL logging.
    """
    task_id = f"chat-{uuid.uuid4().hex[:8]}"
    task_ir = canonicalize_task_ir({
        "task_type": "chat",
        "objective": request.prompt[:TASK_IR_OBJECTIVE_LEN],
        "priority": "interactive",
        "context_preview": request.context or "",
    })

    use_mock = request.mock_mode and not request.real_mode
    skill_context = ""  # Populated by SkillAugmentedRouter when skillbank is enabled
    has_image = bool(request.image_path or request.image_base64)
    heuristic_priors = _heuristic_role_priors(
        request.prompt,
        request.context or "",
        has_image=has_image,
    )

    # Initialize MemRL early for real_mode to enable HybridRouter
    if request.real_mode and not use_mock:
        ensure_memrl_initialized(state)

    # Determine routing using HybridRouter if available, otherwise rules
    if use_mock:
        routing_decision = [request.role or Role.FRONTDOOR]
        routing_strategy = "mock"
    elif request.force_role:
        routing_decision = [request.force_role]
        routing_strategy = "forced"
    elif request.role and request.role not in ("", "frontdoor"):
        routing_decision = [request.role]
        routing_strategy = "explicit"
    elif state.hybrid_router and request.real_mode:
        # Use skill-augmented routing if available (SkillAugmentedRouter)
        if hasattr(state.hybrid_router, "route_with_skills") and features().skillbank:
            routing_decision, routing_strategy, skill_context = (
                state.hybrid_router.route_with_skills(task_ir)
            )
        else:
            routing_decision, routing_strategy = state.hybrid_router.route(
                task_ir,
                priors=heuristic_priors,
            )
            skill_context = ""

    else:
        classified_role, routing_strategy = _classify_and_route(
            request.prompt,
            request.context or "",
            has_image=has_image,
        )
        routing_decision = [classified_role]

    # Factual-risk scoring (shadow/enforce mode only — no-op when mode is "off")
    # RI-5: Moved BEFORE failure graph veto so risk band can modulate veto threshold.
    try:
        from src.classifiers.factual_risk import assess_risk, get_mode as _fr_mode
        if _fr_mode() != "off":
            _fr_result = assess_risk(
                request.prompt,
                role=str(routing_decision[0]) if routing_decision else "",
            )
            _factual_risk_score = _fr_result.adjusted_risk_score
            _factual_risk_band = _fr_result.risk_band
            log.info(
                "Factual risk: score=%.3f band=%s (raw=%.3f adj=%.1f)",
                _fr_result.adjusted_risk_score,
                _fr_result.risk_band,
                _fr_result.risk_score,
                _fr_result.role_adjustment,
                extra=task_extra(
                    task_id=task_id,
                    stage="routing",
                    strategy="factual_risk",
                ),
            )
        else:
            _factual_risk_score = 0.0
            _factual_risk_band = ""
    except Exception as _fr_exc:
        log.debug("Factual risk scoring skipped: %s", _fr_exc)
        _factual_risk_score = 0.0
        _factual_risk_band = ""

    # Failure graph veto — revert high-risk specialists to frontdoor
    # RI-5: Veto threshold modulated by factual-risk band.
    # High factual risk → lower veto threshold (more conservative routing).
    # Low factual risk → higher threshold (allow specialist attempts).
    _VETO_THRESHOLDS = {"high": 0.3, "medium": 0.5, "low": 0.7, "": 0.5}
    if (
        state.failure_graph
        and routing_decision
        and str(routing_decision[0]) != str(Role.FRONTDOOR)
        and routing_strategy not in ("mock", "forced")
    ):
        try:
            risk = state.failure_graph.get_failure_risk(str(routing_decision[0]))
            veto_threshold = _VETO_THRESHOLDS.get(_factual_risk_band, 0.5)
            if risk > veto_threshold:
                log.warning(
                    "Failure veto: %s risk=%.2f > %.1f (factual_risk=%s), reverting to frontdoor",
                    routing_decision[0],
                    risk,
                    veto_threshold,
                    _factual_risk_band or "none",
                    extra=task_extra(
                        task_id=task_id,
                        role=str(routing_decision[0]),
                        stage="routing",
                        strategy="failure_vetoed",
                    ),
                )
                routing_decision = [str(Role.FRONTDOOR)]
                routing_strategy = "failure_vetoed"
        except Exception as exc:
            log.debug("Failure risk veto check failed: %s", exc)

    # Difficulty-signal scoring (shadow/enforce mode only — no-op when mode is "off")
    try:
        from src.classifiers.difficulty_signal import assess_difficulty, get_mode as _ds_mode
        if _ds_mode() != "off":
            _ds_result = assess_difficulty(
                request.prompt,
                role=str(routing_decision[0]) if routing_decision else "",
            )
            _difficulty_score = _ds_result.difficulty_score
            _difficulty_band = _ds_result.difficulty_band
            log.info(
                "Difficulty signal: score=%.3f band=%s",
                _ds_result.difficulty_score,
                _ds_result.difficulty_band,
                extra=task_extra(
                    task_id=task_id,
                    stage="routing",
                    strategy="difficulty_signal",
                ),
            )
        else:
            _difficulty_score, _difficulty_band = 0.0, ""
    except Exception as _ds_exc:
        log.debug("Difficulty signal scoring skipped: %s", _ds_exc)
        _difficulty_score, _difficulty_band = 0.0, ""

    # Estimated cost (tier weight × prompt tokens / 1M — relative units for Pareto)
    _estimated_cost = 0.0
    try:
        _role_str = str(routing_decision[0]) if routing_decision else "frontdoor"
        if state.registry:
            _role_cfg = state.registry.get_role(_role_str)
            _tier = getattr(_role_cfg, "tier", "C") if _role_cfg else "C"
        else:
            _tier = "C"
        _est_tokens = len(request.prompt) // 4 + len(request.context or "") // 4
        _estimated_cost = _TIER_COST_WEIGHTS.get(_tier, 1.0) * _est_tokens / 1_000_000
    except Exception:
        pass

    # Log task start (MemRL integration)
    if state.progress_logger:
        routing_meta = {
            "decision_source": routing_strategy,
            "was_forced": bool(request.force_role),
            "heuristic_priors": {
                k: round(v, 4) for k, v in sorted(heuristic_priors.items(), key=lambda kv: -kv[1])[:4]
            },
            "factual_risk_score": round(_factual_risk_score, 4),
            "factual_risk_band": _factual_risk_band,
            "difficulty_score": round(_difficulty_score, 4),
            "difficulty_band": _difficulty_band,
            "estimated_cost": round(_estimated_cost, 6),
        }
        # DS-1: Inject queue depth telemetry from backend stats
        if state.llm_primitives and hasattr(state.llm_primitives, "get_stats"):
            try:
                _stats = state.llm_primitives.get_stats()
                routing_meta["active_requests"] = state.active_requests
                if "per_role" in _stats:
                    routing_meta["queue_depth"] = {
                        role: rs.get("total_active", rs.get("round_robin_requests", 0))
                        for role, rs in _stats["per_role"].items()
                        if isinstance(rs, dict)
                    }
            except Exception:
                pass
        # DS-4: Log stack state (models loaded, instance counts) alongside routing
        if state.registry and hasattr(state.registry, "roles"):
            try:
                routing_meta["stack_state"] = {
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
                routing_meta.update(state.hybrid_router.last_decision_meta or {})
            except Exception:
                pass
        state.progress_logger.log_task_started(
            task_id=task_id,
            task_ir=task_ir,
            routing_decision=routing_decision,
            routing_strategy=routing_strategy,
            routing_meta=routing_meta,
        )

    # Compute role-specific timeout
    role_str = str(routing_decision[0]) if routing_decision else str(Role.FRONTDOOR)
    timeout_s = ROLE_TIMEOUTS.get(role_str, DEFAULT_TIMEOUT_S)
    if request.timeout_s is not None:
        timeout_s = max(1, min(timeout_s, int(request.timeout_s)))

    # Detect tool requirement for forced tool use
    from src.api.routes.chat_routing import detect_tool_requirement

    tool_required, tool_hint = detect_tool_requirement(request.prompt)

    # Extract skill IDs from skill-augmented routing results
    skill_ids: list[str] = []
    if skill_context and hasattr(state, "hybrid_router") and state.hybrid_router:
        try:
            if hasattr(state.hybrid_router, "_last_skill_ids"):
                skill_ids = list(state.hybrid_router._last_skill_ids)
        except Exception:
            pass

    return RoutingResult(
        task_id=task_id,
        task_ir=task_ir,
        use_mock=use_mock,
        routing_decision=routing_decision,
        routing_strategy=routing_strategy,
        timeout_s=timeout_s,
        tool_required=tool_required,
        tool_hint=tool_hint,
        skill_context=skill_context,
        skill_ids=skill_ids,
        factual_risk_score=_factual_risk_score,
        factual_risk_band=_factual_risk_band,
        difficulty_score=_difficulty_score,
        difficulty_band=_difficulty_band,
        estimated_cost=_estimated_cost,
    )


# ── Stage 2: Preprocessing ──────────────────────────────────────────────


def _preprocess(request: ChatRequest, state, routing: RoutingResult) -> None:
    """Apply input formalization if enabled. Mutates request.context and routing."""
    if (
        features().input_formalizer
        and request.real_mode
        and not routing.use_mock
        and routing.routing_strategy not in ("mock",)
    ):
        from src.formalizer import should_formalize_input, formalize_prompt, inject_formalization

        should_fml, problem_hint = should_formalize_input(request.prompt)
        if should_fml:
            fml_result = formalize_prompt(request.prompt, problem_hint, state.registry)
            if fml_result.success:
                request.context = inject_formalization(
                    request.prompt, request.context or "", fml_result.ir_json
                )
                routing.formalization_applied = True
                log.info(
                    "Input formalization: %s (%.1fs, %s)",
                    problem_hint,
                    fml_result.elapsed_seconds,
                    fml_result.model_role,
                    extra=task_extra(
                        task_id=routing.task_id,
                        stage="preprocess",
                        latency_ms=fml_result.elapsed_seconds * 1000,
                    ),
                )


# ── Stage 3: Backend initialization ─────────────────────────────────────


def _init_primitives(request: ChatRequest, state) -> LLMPrimitives:
    """Initialize LLM backends for real inference.

    Reuses shared LLMPrimitives instance for connection pooling when possible.
    Raises HTTPException(503) if backends unavailable.
    """
    if request.real_mode:
        server_urls = request.server_urls or get_config().server_urls.as_dict()

        if (
            hasattr(state, "_real_primitives")
            and state._real_primitives is not None
            and not request.server_urls
        ):
            primitives = state._real_primitives
            primitives.reset_counters()
        else:
            try:
                primitives = LLMPrimitives(
                    mock_mode=False,
                    server_urls=server_urls,
                    registry=state.registry,
                    health_tracker=getattr(state, "health_tracker", None),
                    admission_controller=getattr(state, "admission", None),
                    num_slots=get_config().server.num_slots,
                )
                if not request.server_urls:
                    state._real_primitives = primitives
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize real mode backends: {e}",
                )

        primitives.cache_prompt = request.cache_prompt

        if not primitives._backends:
            raise HTTPException(
                status_code=503,
                detail="No backends available. Ensure llama-server is running on configured ports.",
            )
    else:
        primitives = LLMPrimitives(mock_mode=False, registry=state.registry)
        if primitives.model_server is None:
            raise HTTPException(
                status_code=503,
                detail="Real inference not available: no model server configured",
            )

    return primitives


# ── Stage 5: Plan review gate ───────────────────────────────────────────


def _plan_review_gate(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
) -> list | None:
    """Run architect plan review if applicable. Returns modified routing_decision or None."""
    plan_review_result = None
    # RI-3: Force plan review when factual risk is high, regardless of complexity heuristics.
    # High-risk prompts need architect oversight to catch factual errors.
    from src.classifiers.factual_risk import get_mode as _fr_get_mode
    risk_forced = routing.factual_risk_band == "high" and _fr_get_mode() == "enforce"
    needs_review = _needs_plan_review(routing.task_ir, routing.routing_decision, state)
    if (
        request.real_mode
        and features().plan_review
        and (needs_review or risk_forced)
    ):
        plan_review_result = _architect_plan_review(
            routing.task_ir, routing.routing_decision, primitives, state, routing.task_id
        )
        if plan_review_result and plan_review_result.decision != "ok":
            routing.routing_decision = _apply_plan_review(
                routing.routing_decision, plan_review_result
            )
        if plan_review_result:
            _store_plan_review_episode(state, routing.task_id, routing.task_ir, plan_review_result)
    return plan_review_result
