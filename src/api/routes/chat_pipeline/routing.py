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
from src.task_ir import canonicalize_task_ir

from src.api.routes.chat_pipeline.routing_decision import (
    apply_failure_veto,
    apply_ingest_triviality_guard,
    assess_difficulty,
    assess_factual_risk,
    classify_trinity_role,
    derive_task_type_from_route,
    estimate_routing_cost,
    extract_skill_ids,
    log_routing_start,
    resolve_timeout,
    select_initial_route,
)
from src.api.routes.chat_review import (
    _apply_plan_review,
    _architect_plan_review,
    _needs_plan_review,
    _store_plan_review_episode,
)
from src.api.routes.chat_routing import _classify_and_route
from src.api.routes.chat_routing import _heuristic_role_priors
from src.api.routes.chat_utils import (
    RoutingResult,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)

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
    routing_decision, routing_strategy, skill_context = select_initial_route(
        request,
        state,
        task_ir,
        use_mock,
        heuristic_priors,
        _classify_and_route,
    )

    role_for_signals = str(routing_decision[0]) if routing_decision else ""
    _factual_risk_score, _factual_risk_band = assess_factual_risk(
        request.prompt,
        role_for_signals,
        task_id,
    )

    # Failure graph veto — revert high-risk specialists to frontdoor
    # RI-5: Veto threshold modulated by factual-risk band.
    # High factual risk → lower veto threshold (more conservative routing).
    # Low factual risk → higher threshold (allow specialist attempts).
    routing_decision, routing_strategy = apply_failure_veto(
        state,
        routing_decision,
        routing_strategy,
        _factual_risk_band,
        task_id,
    )

    # Difficulty-signal scoring (shadow/enforce mode only — no-op when mode is "off")
    role_for_signals = str(routing_decision[0]) if routing_decision else ""
    _difficulty_score, _difficulty_band = assess_difficulty(
        request.prompt,
        role_for_signals,
        task_id,
    )

    # Ingest-triviality guard (opt-in): keep trivially-easy short prompts off the
    # ingest_long_context 80B specialist. Reuses _difficulty_band; no-op when the
    # ingest_triviality_guard feature flag is off. Placed after the failure veto
    # and difficulty scoring so it sees the final pre-execution route.
    routing_decision, routing_strategy = apply_ingest_triviality_guard(
        request,
        routing_decision,
        routing_strategy,
        _difficulty_band,
        task_id,
    )

    # Estimated cost (tier weight × prompt tokens / 1M — relative units for Pareto)
    _estimated_cost = estimate_routing_cost(request, state, routing_decision)

    # Log task start (MemRL integration)
    log_routing_start(
        request,
        state,
        task_id,
        task_ir,
        routing_decision,
        routing_strategy,
        heuristic_priors,
        _factual_risk_score,
        _factual_risk_band,
        _difficulty_score,
        _difficulty_band,
        _estimated_cost,
    )

    # Compute role-specific timeout
    timeout_s = resolve_timeout(request, routing_decision)

    # Detect tool requirement for forced tool use
    from src.api.routes.chat_routing import detect_tool_requirement

    tool_required, tool_hint = detect_tool_requirement(request.prompt)

    # Extract skill IDs from skill-augmented routing results
    skill_ids = extract_skill_ids(skill_context, state)

    # WS-3: Derive task_type from routed role so cascading tool policy can deny
    # web tools for reasoning domains. Without this, task_type stays "chat" and
    # NO_WEB_TASK_TYPES never matches.
    derive_task_type_from_route(task_ir, routing_decision)

    # TR-3.2: Trinity tri-role classification (shadow mode). Always populates
    # `assigned_role` regardless of the ROLE_AWARE_ROUTING flag — TR-4 gates
    # acting on the role; TR-3.3 uses shadow telemetry to decide promotion.
    _assigned_role = classify_trinity_role(request, routing_decision, task_id)

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
        assigned_role=_assigned_role,
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
    _current_role = routing.routing_decision[0] if routing.routing_decision else ""
    risk_forced = routing.factual_risk_band == "high" and _fr_get_mode(role=_current_role) == "enforce"
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
