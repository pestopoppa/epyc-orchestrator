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


@dataclass(frozen=True)
class TeleportPolicy:
    enabled: bool = False
    min_resident_remaining_tokens: int = 150
    min_cold_remaining_tokens: int = 350
    min_speedup: float = 1.05
    allowed_roles: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class TeleportInputs:
    role: str
    generated_tokens: int
    estimated_remaining_tokens: int
    cpu_tps: float
    gpu_tps: float
    gpu_available: bool
    gpu_resident: bool
    catch_up_supported: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeleportDecision:
    should_cutover: bool
    reason: str
    catch_up_supported: bool = False
    catch_up_reason: str = "llama_server_verify_api_unavailable"
    threshold_tokens: int | None = None
    estimated_speedup: float | None = None


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
    common = {
        "catch_up_supported": inputs.catch_up_supported,
        "catch_up_reason": (
            ""
            if inputs.catch_up_supported
            else "llama_server_verify_api_unavailable"
        ),
        "threshold_tokens": threshold,
        "estimated_speedup": speedup,
    }

    if not policy.enabled:
        return TeleportDecision(False, "disabled", **common)
    if policy.allowed_roles and inputs.role not in policy.allowed_roles:
        return TeleportDecision(False, "role_not_allowed", **common)
    if not inputs.gpu_available:
        return TeleportDecision(False, "gpu_unavailable", **common)
    if inputs.estimated_remaining_tokens < threshold:
        return TeleportDecision(False, "below_break_even_tokens", **common)
    if speedup is None:
        return TeleportDecision(False, "missing_speed_estimate", **common)
    if speedup < policy.min_speedup:
        return TeleportDecision(False, "below_speedup_threshold", **common)
    return TeleportDecision(True, "cutover", **common)
