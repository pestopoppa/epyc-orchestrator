"""Request models for the orchestrator API."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Role strings that legitimately appear on the wire but are pipeline-stage
# markers rather than serving roles. They must not be rewritten to "" (that
# would change routing), but they are also not expected to resolve to a Role.
_NON_ROLE_SENTINELS = frozenset({"mock", "plan", "stream_init", "proactive_delegation"})


def _normalize_role_field(value: str | None, field_name: str) -> str | None:
    """Coerce a client-supplied role string to a canonical role name.

    ``role`` and ``force_role`` were bare ``str`` fields with no validation, so
    any client string flowed through the pipeline into the TASK_COMPLETED
    telemetry as ``producer_role``. A rescore of 117,074 historical completions
    (2026-07-21) found 157 rows with non-role values, including uppercase
    variants from 3-way seeding (``SELF``, ``WORKER``) and one row whose role
    field contained a prompt fragment ("Provide full Python snippet with
    variable name.").

    That is not cosmetic. ``compute_reward`` looks the role up in
    ``baseline_tps_by_role``; an unresolvable role misses every cost dimension
    and the task scores the full base reward. An unvalidated client string could
    therefore suppress the entire cost/speed penalty — a latent reward-hacking
    surface, and the same silent-miss shape as the ``role``/``producer_role``
    bug it was found alongside.

    Behaviour: resolve via ``Role.from_string`` (which already handles legacy
    aliases), accept a case-insensitive match, pass known non-role sentinels
    through untouched, and downgrade anything else to "" (auto-route) with a
    warning rather than letting it reach telemetry.
    """
    if value is None or value == "":
        return value

    from src.roles import Role

    if Role.from_string(value) is not None:
        return value
    if value in _NON_ROLE_SENTINELS:
        return value

    lowered = value.strip().lower()
    if Role.from_string(lowered) is not None:
        logger.warning(
            "Normalized non-canonical %s=%r to %r", field_name, value, lowered
        )
        return lowered

    logger.warning(
        "Rejected unrecognized %s=%r (%d chars); falling back to auto-route. "
        "Unresolvable roles silently disable the reward cost penalty.",
        field_name,
        value[:80],
        len(value),
    )
    return ""


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    prompt: str = Field(..., description="The user prompt to process")
    context: str = Field(default="", description="Optional context to include")
    mock_mode: bool = Field(
        default=True, description="Use mock responses instead of real inference"
    )
    real_mode: bool = Field(
        default=False, description="Enable real inference with RadixAttention caching"
    )
    max_turns: int = Field(default=15, ge=1, le=50, description="Maximum orchestration turns")
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=32768,
        description=(
            "Optional response-token cap for direct chat execution. When set, "
            "stage defaults such as MCQ/code budgets are clamped to this value."
        ),
    )
    n_probs: int | None = Field(
        default=None,
        ge=0,
        le=128,
        description=(
            "Optional llama.cpp top-k token probability capture for calibration "
            "instrumentation. Omitted by normal chat traffic."
        ),
    )
    role: str = Field(
        default="", description="Initial role to use (empty = auto-route via _classify_and_route)"
    )
    force_role: str | None = Field(
        default=None,
        description="Force routing to a specific role, bypassing all routing logic. "
        "Used by comparative seeding to test specialist quality.",
    )

    @field_validator("role", "force_role")
    @classmethod
    def _validate_role_fields(cls, value, info):
        """Reject free text in role fields — see _normalize_role_field."""
        return _normalize_role_field(value, info.field_name)
    force_mode: str | None = Field(
        default=None,
        description="Force execution mode ('direct', 'react', 'repl', 'delegated', or 'edit'), "
        "bypassing _select_mode heuristics. 'delegated' enables architect delegation "
        "where the architect formulates investigation briefs for faster specialists. "
        "'edit' runs a one-shot transactional file edit (flag-gated ORCHESTRATOR_EDIT_TRANSACTION=1 "
        "+ scoped ORCHESTRATOR_EDIT_ROOT; bypasses the multi-turn REPL loop for routine file edits).",
    )
    allow_delegation: bool | None = Field(
        default=None,
        description="Override delegation capability. True=allow delegation to workers, "
        "False=disable delegation (model handles alone). None=use feature flag default. "
        "Used by 3-way seeding to isolate SELF vs ARCHITECT behavior.",
    )
    server_urls: dict[str, str] | None = Field(
        default=None,
        description="Server URLs for real mode (e.g., {'frontdoor': 'http://localhost:8080'})",
    )
    # Vision support — when set, routes to VL workers (8086/8087)
    image_path: str | None = Field(
        default=None, description="Path to image file for vision tasks (routes to VL worker)"
    )
    image_base64: str | None = Field(
        default=None, description="Base64-encoded image data for vision tasks"
    )
    files: list[str] | None = Field(
        default=None,
        description="List of file paths for multi-file vision/document tasks (archives auto-extracted)",
    )
    # Per-request cache control
    cache_prompt: bool | None = Field(
        default=None,
        description="Override cache_prompt for this request (None=backend default True). "
        "Set to False for benchmark seeding where prefix caching adds overhead.",
    )
    # Extended thinking support (Claude Code parity)
    thinking_budget: int = Field(
        default=0,
        ge=0,
        le=32000,
        description="Token budget for internal reasoning (0=disabled, max=32000)",
    )
    permission_mode: str = Field(
        default="normal", description="Permission mode: 'normal', 'auto-accept', or 'plan'"
    )
    timeout_s: int | None = Field(
        default=None,
        ge=1,
        # 600->1800->3600: giant (~90K-token) eval prompts can spend nearly
        # 30 minutes in half-instance prefill. Policy (which workload may
        # extend how far) is governed by resolve_timeout's eval_batch scoping,
        # not this shape-validation ceiling; non-eval traffic remains clamped
        # to its role SLA.
        le=3600,
        description="Optional per-request server-side timeout budget in seconds. "
        "When set, orchestration deadlines and lock waits are bounded to this value.",
    )
    client_deadline_unix_s: float | None = Field(
        default=None,
        description="Optional client wall-clock deadline in Unix seconds. "
        "When set, server execution budget is additionally clamped to this deadline.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for cross-request REPL globals restore.",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional caller request id for tracing. The live inference tap "
        "records this as parent_request_id and still creates one unique id per model call.",
    )
    trial_id: int | str | None = Field(
        default=None,
        description="Optional autopilot/benchmark trial id for inference tap attribution.",
    )
    batch_id: int | str | None = Field(
        default=None,
        description="Optional concurrency/eval batch id for inference tap attribution.",
    )
    request_priority: str = Field(
        default="interactive",
        description="Admission priority: 'interactive' (default) or 'background'. "
        "Interactive requests are prioritized at backend admission gates.",
    )
    workload_class: Literal["interactive", "eval_batch", "campaign"] | None = Field(
        default=None,
        description="Optional workload traffic class for attribution: "
        "'interactive', 'eval_batch', or 'campaign'. When unset, the server "
        "infers it from existing request metadata without changing admission priority.",
    )
    batch_placement_mode: Literal[
        "auto", "homogeneous_native_batch", "mixed_role_split"
    ] | None = Field(
        default=None,
        description="Optional burst placement intent. Homogeneous cohorts may share "
        "one full server's certified native slots; mixed routed pipelines start on "
        "sub-full instances so different CPU roles can occupy disjoint regions.",
    )
    x_orchestrator_prompt_root: str | None = Field(
        default=None,
        description=(
            "Internal AutoPilot/GEPA override for resolving prompt files from "
            "a scratch prompt tree. Accepted only from configured scratch roots."
        ),
    )
    routing_preferences: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional DAR-4b routing scalarization weights. Keys 'perf'/'performance' "
            "and 'cost' are normalized onto the performance-cost simplex. "
            "Absent means the existing retrieval score is preserved."
        ),
    )
    max_queue_wait_ms: int | None = Field(
        default=None,
        description="Maximum time the cross-role contention gate may queue "
        "this request before rejecting it (HTTP 503 + Retry-After). When unset, "
        "the gate uses 5s for interactive and 90s for background. Foreground "
        "callers with tight SLO budgets should set this explicitly.",
    )
    migration_budget_ms: int | None = Field(
        default=None,
        description="Phase E (cross-role-bw-aware-routing): maximum acceptable "
        "KV-save+restore latency for migrating an existing session from full "
        "to quarter on concurrent arrival. Short interactive turns should set "
        "this low (e.g. 200) to skip migration and queue/cold-start instead. "
        "Long conversations + background probes can amortize a longer budget. "
        "Honored by the per-region-locks session-handover migration transaction "
        "when one is attempted; see "
        "ConcurrencyAwareBackend.kv_migration_status() for runtime state.",
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Additional stop sequences to halt generation. "
        "Merged with any pipeline-default stop sequences (e.g. QWEN_STOP). "
        "Used by benchmark seeding to stop after answer tags.",
    )
    tools: list[dict] | None = Field(
        default=None,
        description="Optional OpenAI-compatible tool schemas for callers that route "
        "through the chat API. Function tools are exposed to the REPL as CALL(name, **kwargs).",
    )
    tool_choice: str | dict | None = Field(
        default=None,
        description="Optional OpenAI-compatible tool choice policy for the provided tools.",
    )
    output_schema: dict | None = Field(
        default=None,
        description="Optional JSON Schema for the agent's FINAL() value. "
        "When set AND features().final_schema_validation is True, the agent receives "
        "the schema in its initial prompt and must call FINAL(json.dumps(value)). "
        "Validation failure injects a retry-with-error message into the next turn.",
    )


class RewardRequest(BaseModel):
    """Request model for injecting external rewards into MemRL."""

    task_description: str = Field(..., description="Description of the task that was scored")
    action: str = Field(..., description="Action taken, e.g. 'frontdoor:direct'")
    reward: float = Field(..., ge=-1.0, le=1.0, description="Reward value (-1.0 to 1.0)")
    context: dict | None = Field(
        default=None, description="Optional metadata (suite, tier, scoring_method)"
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Precomputed embedding for task_description (avoids re-embedding)",
    )


class GateRequest(BaseModel):
    """Request model for running gates."""

    gate_names: list[str] | None = Field(
        default=None, description="Specific gates to run (None = all)"
    )
    stop_on_first_failure: bool = Field(
        default=True, description="Stop after first required gate fails"
    )
    required_only: bool = Field(default=False, description="Only run required gates")
