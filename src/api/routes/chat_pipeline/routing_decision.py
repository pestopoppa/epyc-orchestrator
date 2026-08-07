"""Routing decision helpers for the chat pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from src.api.models import ChatRequest
from src.api.routes.chat_routing import _classify_and_route
from src.api.routes.chat_utils import LONG_CONTEXT_CONFIG, role_timeout_for
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

_INGRESS_ROLE_ALIASES = frozenset({"worker_coder", "worker_code"})

# Upper bound on the image reference persisted into the routing-decision
# progress event. Real image paths are ~90 chars; the cap only fences a
# pathological caller from writing an unbounded string into every log line.
MAX_IMAGE_PATH_LEN = 512

_DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[4] / "orchestration" / "derived" / "stack_priors.yaml"
)

# Declarative-source fallback ONLY. The authoritative set of vision-only roles
# is read from generated stack priors (launch mode == "vision") via
# ``vision_serving.vision_roles``; these two known VL-only role names are used
# solely when that artifact is unreadable (mirrors vision_serving's own
# LEGACY_VISION_ROLES). Do not treat this literal as the source of truth.
_FALLBACK_VISION_ONLY_ROLES = frozenset({"worker_vision", "vision_escalation"})


def _vision_only_roles() -> frozenset[str]:
    """Return roles that serve ONLY multimodal VL traffic (declarative source).

    A VL-only llama-server rejects a text-only ``/completion`` with HTTP 400
    (observed: a longbench TEXT question misrouted to ``worker_vision:8086``).
    Both directions of the modality fence key on this set: text requests are
    steered away from these roles, and image requests routed to them are exempt
    from the failure veto (they have no valid text fallback).
    """
    try:
        from src.api.routes.vision_serving import vision_roles as _stack_prior_vision_roles

        return _stack_prior_vision_roles(_DEFAULT_STACK_PRIORS_PATH)
    except Exception:
        return _FALLBACK_VISION_ONLY_ROLES


def _fence_text_off_vision_roles(
    routing_decision: list, routing_strategy: str
) -> tuple[list, str]:
    """Strip vision-only roles from a TEXT request's candidate route.

    The learned/hybrid router and the rules classifier can both emit a
    vision-only role for a text-only prompt; routing text there yields an HTTP
    400 from the VL server (blind/failed answers that poison the eval). Callers
    invoke this ONLY for requests with no image data and no explicit/forced role
    (forced/explicit routing is the caller's responsibility and bypasses the
    fence). If stripping empties the route, fall back to the frontdoor.
    """
    if not routing_decision:
        return routing_decision, routing_strategy
    vision_only = _vision_only_roles()
    filtered = [r for r in routing_decision if str(r) not in vision_only]
    if len(filtered) == len(routing_decision):
        return routing_decision, routing_strategy
    if not filtered:
        log.warning(
            "Modality fence: text request routed to vision-only %s → frontdoor",
            [str(r) for r in routing_decision],
        )
        return [str(Role.FRONTDOOR)], f"{routing_strategy}:vision_fenced"
    log.warning(
        "Modality fence: stripped vision-only role(s) from text route %s",
        [str(r) for r in routing_decision],
    )
    return filtered, f"{routing_strategy}:vision_fenced"


def normalize_ingress_role(role: object) -> object:
    """Normalize externally supplied role labels before config lookup."""
    if not isinstance(role, str):
        return role
    normalized = Role.from_string(role)
    if normalized is not None:
        return normalized
    if role in _INGRESS_ROLE_ALIASES:
        return Role.WORKER_GENERAL
    return role


def assess_factual_risk(prompt: str, role: str, task_id: str) -> tuple[float, str]:
    """Return factual-risk score and band, falling back to no-risk on failure."""
    try:
        from src.classifiers.factual_risk import assess_risk, get_configured_mode

        if get_configured_mode() == "off":
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
    # Capacity/specialist fence: learned routing only sees a bounded task-IR
    # objective and may therefore choose frontdoor for a request whose full
    # prompt is hundreds of thousands of characters.  That failure mode sends
    # long-context prefill to a latency-oriented worker and can outlive the
    # request budget.  Preserve explicit/forced and image routing above, but
    # keep ordinary oversized text on the role provisioned for it.
    long_context_enabled = bool(LONG_CONTEXT_CONFIG.get("enabled", True))
    long_context_threshold = max(
        1,
        int(LONG_CONTEXT_CONFIG.get("threshold_chars", 20_000)),
    )
    effective_chars = len(request.prompt or "") + len(request.context or "")
    if long_context_enabled and effective_chars > long_context_threshold:
        log.info(
            "Long-context routing guard: %d chars > %d → ingest_long_context",
            effective_chars,
            long_context_threshold,
        )
        return [str(Role.INGEST_LONG_CONTEXT)], "long_context_guard", skill_context
    # Below this point the request has NO image data and NO forced/explicit
    # role (those returned above). The learned/hybrid router and the rules
    # classifier can still emit a vision-only role for a text prompt, which a
    # VL server rejects with HTTP 400 — fence text traffic off those roles.
    if state.hybrid_router and request.real_mode:
        if hasattr(state.hybrid_router, "route_with_skills") and features().skillbank:
            routing_decision, routing_strategy, skill_context = (
                state.hybrid_router.route_with_skills(task_ir)
            )
            routing_decision, routing_strategy = _fence_text_off_vision_roles(
                routing_decision, routing_strategy
            )
            return routing_decision, routing_strategy, skill_context
        routing_decision, routing_strategy = state.hybrid_router.route(
            task_ir,
            priors=heuristic_priors,
        )
        routing_decision, routing_strategy = _fence_text_off_vision_roles(
            routing_decision, routing_strategy
        )
        return routing_decision, routing_strategy, skill_context

    classified_role, routing_strategy = classify_and_route(
        request.prompt,
        request.context or "",
        has_image=bool(request.image_path or request.image_base64),
    )
    routing_decision, routing_strategy = _fence_text_off_vision_roles(
        [classified_role], routing_strategy
    )
    return routing_decision, routing_strategy, skill_context


def apply_failure_veto(
    state,
    routing_decision: list,
    routing_strategy: str,
    factual_risk_band: str,
    task_id: str,
    has_image: bool = False,
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

    # Modality fence (image direction): an image-carrying request routed to a
    # vision-only role has NO valid text fallback. Reverting it to the text
    # frontdoor makes the VL model answer BLIND (the same poisoning the vision
    # stage now fails visibly on) — worse than attempting the risky VL role.
    # Exempt it; if the VL backend is truly down the vision stage emits an
    # honest in-band vision_unavailable marker.
    if has_image and str(routing_decision[0]) in _vision_only_roles():
        log.debug(
            "Failure veto exempt: image request on vision-only role %s (no text fallback)",
            routing_decision[0],
            extra=task_extra(
                task_id=task_id,
                role=str(routing_decision[0]),
                stage="routing",
                strategy="vision_veto_exempt",
            ),
        )
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
    factual_risk_mode: str,
    difficulty_score: float,
    difficulty_band: str,
    estimated_cost: float,
    assigned_role: str,
    xmas_meta: dict | None = None,
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
        "factual_risk_mode": factual_risk_mode,
        "difficulty_score": round(difficulty_score, 4),
        "difficulty_band": difficulty_band,
        "assigned_role": assigned_role,
        "estimated_cost": round(estimated_cost, 6),
    }
    if xmas_meta:
        meta["xmas"] = xmas_meta
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

    # ── Vision capture ────────────────────────────────────────────────────
    # A request carrying image data must leave a DURABLE reference in the
    # progress JSONL. Until this, `image_path` was forwarded to the VL backend
    # but written to no progress event at all: every offline consumer (the
    # dashboard completed-tasks panel, replay, vl-suite triage) saw a vision
    # task as an ordinary text task with no way to tell which image produced
    # the answer. Verified against stored rows — `grep -c image_path` over
    # every logs/progress/*.jsonl returned 0, including for genuinely
    # successful `Vision multimodal` completions (e.g. chat-94e59d8f,
    # 2026-06-13, worker_vision).
    #
    # Written LAST so neither `meta.update(router_meta)` nor the uncertainty
    # block above can clobber or be perturbed by these keys.
    #
    # Only the PATH is stored — NEVER `image_base64`, whose payload is
    # megabytes per row and would balloon the progress log. A base64-only
    # request records the flag and the source so the panel can still label it.
    try:
        image_path = str(getattr(request, "image_path", "") or "").strip()
        if image_path:
            meta["image_path"] = image_path[:MAX_IMAGE_PATH_LEN]
            meta["has_image"] = True
            meta["image_source"] = "path"
        elif getattr(request, "image_base64", None):
            meta["has_image"] = True
            meta["image_source"] = "base64"
    except Exception:
        pass
    # Request-lifecycle join fields for the in-process live telemetry reducer.
    # These are observability-only and do not alter the routing decision.
    meta["request_id"] = getattr(request, "request_id", None)
    meta["batch_id"] = getattr(request, "batch_id", None)
    meta["workload_class"] = getattr(request, "workload_class", None)
    meta["request_priority"] = getattr(request, "request_priority", None)
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
    factual_risk_mode: str,
    difficulty_score: float,
    difficulty_band: str,
    estimated_cost: float,
    assigned_role: str,
    xmas_meta: dict | None = None,
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
            factual_risk_mode,
            difficulty_score,
            difficulty_band,
            estimated_cost,
            assigned_role,
            xmas_meta,
        ),
    )


def resolve_timeout(request: ChatRequest, routing_decision: list) -> int:
    """Compute role-specific timeout with request-level budget.

    Standard traffic clamps DOWN to the role SLA: a request-supplied
    ``timeout_s`` may only *shorten* the per-request budget below the role's
    interactive latency guarantee, never lengthen it. Interactive latency
    guarantees are therefore unchanged by this function.

    Exception — 2026-07-21 EV-11c incident. Eval-batch traffic
    (``workload_class == "eval_batch"``) that declares an explicit
    ``timeout_s`` may EXTEND its budget *beyond* the role SLA, because the
    request itself is declaring how long its work legitimately takes. The
    incident: 2,048-token MATH-tail generations served at 4-wide shared
    bandwidth need >60s, but the interactive worker SLA (60s) clamped every
    such call DOWN to 60s. Those doomed calls 504'd, tripped the production
    circuit breaker, and the breaker then served in-band ``[ERROR: ...]`` text
    as answers plus a silent role fallback. Honoring the eval batch's
    self-declared budget lets the long-but-legitimate call finish instead of
    being force-failed. Only self-declared ``eval_batch`` requests can lengthen
    their budget; all other traffic keeps the exact DOWN-only ``min`` clamp.
    """
    role_str = str(routing_decision[0]) if routing_decision else str(Role.FRONTDOOR)
    timeout_s = role_timeout_for(role_str)
    if request.timeout_s is not None:
        if str(getattr(request, "workload_class", "") or "") == "eval_batch":
            # Self-declared eval budget: may EXTEND beyond the role SLA.
            timeout_s = max(1, int(request.timeout_s))
        else:
            # Interactive / all other traffic: DOWN-only clamp (unchanged).
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
