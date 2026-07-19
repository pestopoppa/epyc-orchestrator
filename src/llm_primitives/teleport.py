"""Default-off policy helpers for CPU-to-GPU teleport cutover experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TELEPORT_EVENTS = (
    "teleport_candidate",
    "gpu_lease_acquired",
    "gpu_prefill_start",
    "gpu_prefill_end",
    "cutover",
    "fallback",
    "lease_released",
)

SUPPORTED_TELEPORT_MODE = "v1_reprefill_cutover_only"


@dataclass(frozen=True)
class TeleportPolicy:
    enabled: bool = False
    mode: str = "v1_reprefill_cutover_only"
    quant_policy: str = "same_quant_only"
    long_running_trigger_tokens: int = 128
    rate_window_tokens: int = 64
    min_resident_remaining_tokens: int = 150
    min_cold_remaining_tokens: int = 350
    min_speedup: float = 1.05
    lease_interactive_weight: float = 1.0
    lease_batch_weight: float = 0.25
    lease_eval_weight: float = 0.1
    allowed_roles: frozenset[str] = field(default_factory=frozenset)
    allowed_quant_change_roles: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class TeleportInputs:
    role: str
    generated_tokens: int
    estimated_remaining_tokens: int
    cpu_tps: float
    gpu_tps: float
    gpu_available: bool
    gpu_resident: bool
    cpu_quant: str | None = None
    gpu_quant: str | None = None
    catch_up_supported: bool = False
    rate_window_observed_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeleportDecision:
    should_cutover: bool
    reason: str
    catch_up_supported: bool = False
    catch_up_reason: str = "llama_server_verify_api_unavailable"
    threshold_tokens: int | None = None
    estimated_speedup: float | None = None
    long_running_trigger_tokens: int | None = None
    rate_window_tokens: int | None = None
    rate_window_observed_tokens: int | None = None
    mode: str | None = None
    quant_policy: str | None = None
    quant_transition: str | None = None


def _normalize_quant(value: str | None) -> str:
    return str(value or "").strip().lower()


def _quant_transition(inputs: TeleportInputs) -> str:
    cpu = _normalize_quant(inputs.cpu_quant) or "unknown"
    gpu = _normalize_quant(inputs.gpu_quant) or "unknown"
    return f"{cpu}->{gpu}"


def _quant_policy_rejection(policy: TeleportPolicy, inputs: TeleportInputs) -> str | None:
    cpu = _normalize_quant(inputs.cpu_quant)
    gpu = _normalize_quant(inputs.gpu_quant)
    policy_name = str(policy.quant_policy or "").strip().lower()

    if policy_name == "same_quant_only":
        if not cpu or not gpu:
            return "missing_quant_context"
        if cpu != gpu:
            return "quant_change_not_allowed"
        return None

    if policy_name == "operator_approved_tail_roles":
        if not cpu or not gpu:
            return "missing_quant_context"
        if cpu == gpu:
            return None
        if inputs.role not in policy.allowed_quant_change_roles:
            return "quant_change_role_not_allowed"
        return None

    return "invalid_quant_policy"


def decide_teleport(policy: TeleportPolicy, inputs: TeleportInputs) -> TeleportDecision:
    """Return a default-off AXA-2 teleport decision.

    v1 is a re-prefill cutover. Speculative catch-up is intentionally reported
    as unsupported until llama-server exposes a token-verification API.
    """

    threshold = (
        policy.min_resident_remaining_tokens
        if inputs.gpu_resident
        else policy.min_cold_remaining_tokens
    )
    speedup = (
        inputs.gpu_tps / inputs.cpu_tps
        if inputs.cpu_tps > 0 and inputs.gpu_tps > 0
        else None
    )
    observed_rate_window = (
        inputs.rate_window_observed_tokens
        if inputs.rate_window_observed_tokens is not None
        else inputs.generated_tokens
    )
    common = {
        "catch_up_supported": False,
        "catch_up_reason": "llama_server_verify_api_unavailable",
        "threshold_tokens": threshold,
        "estimated_speedup": speedup,
        "long_running_trigger_tokens": policy.long_running_trigger_tokens,
        "rate_window_tokens": policy.rate_window_tokens,
        "rate_window_observed_tokens": observed_rate_window,
        "mode": policy.mode,
        "quant_policy": policy.quant_policy,
        "quant_transition": _quant_transition(inputs),
    }

    if not policy.enabled:
        return TeleportDecision(False, "disabled", **common)
    if policy.mode != SUPPORTED_TELEPORT_MODE:
        return TeleportDecision(False, "invalid_teleport_mode", **common)
    if inputs.catch_up_supported:
        return TeleportDecision(False, "catch_up_not_supported_in_v1", **common)
    if policy.allowed_roles and inputs.role not in policy.allowed_roles:
        return TeleportDecision(False, "role_not_allowed", **common)
    quant_rejection = _quant_policy_rejection(policy, inputs)
    if quant_rejection:
        return TeleportDecision(False, quant_rejection, **common)
    if not inputs.gpu_available:
        return TeleportDecision(False, "gpu_unavailable", **common)
    if inputs.generated_tokens < policy.long_running_trigger_tokens:
        return TeleportDecision(False, "below_long_running_trigger", **common)
    if observed_rate_window < policy.rate_window_tokens:
        return TeleportDecision(False, "insufficient_rate_window", **common)
    if inputs.estimated_remaining_tokens < threshold:
        return TeleportDecision(False, "below_break_even_tokens", **common)
    if speedup is None:
        return TeleportDecision(False, "missing_speed_estimate", **common)
    if speedup < policy.min_speedup:
        return TeleportDecision(False, "below_speedup_threshold", **common)
    return TeleportDecision(True, "cutover", **common)


def lease_weight_for_workload(policy: TeleportPolicy, workload_class: str) -> float:
    """Return the class-3 lease priority weight for a workload class."""

    normalized = str(workload_class or "").strip().lower()
    if normalized in {"eval", "evaluation", "benchmark", "bench"}:
        return policy.lease_eval_weight
    if normalized in {"batch", "offline", "background"}:
        return policy.lease_batch_weight
    return policy.lease_interactive_weight
