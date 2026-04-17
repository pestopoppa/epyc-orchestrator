"""KV Compression control for autopilot.

Wraps the llama-server /slots/{id}?action=compact endpoint with telemetry
logging and auto-trigger support for Expected Attention KV compression.

Usage by autopilot controller:
    from kv_compress import compress_slot, auto_compress_if_needed

    # Explicit compression with telemetry
    result = compress_slot(port=8070, slot_id=0, keep_ratio=0.5)

    # Auto-trigger: compress when utilization > threshold
    auto_compress_if_needed(port=8070, slot_id=0, threshold=0.80, keep_ratio=0.5)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclass
class CompressResult:
    """Result from a KV compression operation."""
    success: bool
    n_evicted: int = 0
    keep_ratio: float = 0.0
    scorer: str = ""
    pos_max_after: int = 0
    elapsed_ms: float = 0.0
    port: int = 0
    slot_id: int = 0
    error: str = ""

    def to_journal_dict(self) -> dict[str, Any]:
        """Format for experiment journal eval_details."""
        return {
            "kv_compress": {
                "n_evicted": self.n_evicted,
                "keep_ratio": self.keep_ratio,
                "scorer": self.scorer,
                "pos_max_after": self.pos_max_after,
                "elapsed_ms": self.elapsed_ms,
                "port": self.port,
                "slot_id": self.slot_id,
            }
        }


def compress_slot(
    port: int,
    slot_id: int = 0,
    keep_ratio: float = 0.50,
    scorer: str = "expected_attention",
    keep_first: int = 4,
    n_future: int = 128,
    use_covariance: bool = True,
    layer_weights: list[float] | None = None,
    timeout: float = 30.0,
) -> CompressResult:
    """Compress KV cache for a slot via the server endpoint.

    Args:
        port: llama-server port (e.g., 8070 for frontdoor)
        slot_id: Slot ID (usually 0)
        keep_ratio: Fraction of KV entries to KEEP (0.5 = evict 50%)
        scorer: "expected_attention" (default) or "knorm" (legacy)
        keep_first: Number of sink tokens to protect
        n_future: Future positions for RoPE averaging
        use_covariance: Use full EA with covariance (True) or mean-only (False)
        layer_weights: Per-layer importance weights (None = uniform)
        timeout: Request timeout in seconds

    Returns:
        CompressResult with eviction details and timing
    """
    url = f"http://localhost:{port}/slots/{slot_id}?action=compact"
    body: dict[str, Any] = {
        "keep_ratio": keep_ratio,
        "scorer": scorer,
        "keep_first": keep_first,
        "n_future": n_future,
        "use_covariance": use_covariance,
    }
    if layer_weights:
        body["layer_weights"] = layer_weights

    t0 = time.perf_counter()
    try:
        resp = httpx.post(url, json=body, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            error_msg = resp.text[:200]
            log.warning("KV compress failed on port %d slot %d: %s", port, slot_id, error_msg)
            return CompressResult(
                success=False, port=port, slot_id=slot_id,
                elapsed_ms=elapsed_ms, error=error_msg,
            )

        data = resp.json()
        result = CompressResult(
            success=True,
            n_evicted=data.get("n_evicted", 0),
            keep_ratio=keep_ratio,
            scorer=data.get("scorer", scorer),
            pos_max_after=data.get("pos_max_after", 0),
            elapsed_ms=elapsed_ms,
            port=port,
            slot_id=slot_id,
        )

        log.info(
            "KV compress: port=%d slot=%d evicted=%d keep=%.0f%% scorer=%s time=%.1fms",
            port, slot_id, result.n_evicted, keep_ratio * 100, scorer, elapsed_ms,
        )
        return result

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.error("KV compress error on port %d: %s", port, e)
        return CompressResult(
            success=False, port=port, slot_id=slot_id,
            elapsed_ms=elapsed_ms, error=str(e),
        )


def get_slot_utilization(port: int, slot_id: int = 0, timeout: float = 5.0) -> float:
    """Query slot KV utilization as a fraction of context window.

    Returns 0.0 on error or if no tokens are cached.
    """
    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=timeout)
        if resp.status_code != 200:
            return 0.0
        slots = resp.json()
        for slot in slots:
            if slot.get("id") == slot_id:
                n_ctx = slot.get("n_ctx", 1)
                n_past = slot.get("n_past", 0)
                return n_past / n_ctx if n_ctx > 0 else 0.0
    except Exception:
        pass
    return 0.0


def check_gap_ratio(port: int, slot_id: int = 0, timeout: float = 5.0) -> dict[str, Any]:
    """Check the gap ratio in a slot's KV cache.

    After EA eviction without gap compaction, position gaps accumulate.
    A high gap ratio (>50%) means over half the context window is wasted on gaps.

    Returns:
        {"gap_ratio": float, "pos_max": int, "n_past": int, "n_ctx": int, "warning": str}
    """
    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=timeout)
        if resp.status_code != 200:
            return {"gap_ratio": 0.0, "warning": ""}
        slots = resp.json()
        for slot in slots:
            if slot.get("id") == slot_id:
                n_ctx = slot.get("n_ctx", 1)
                n_past = slot.get("n_past", 0)  # actual KV entries
                # pos_max would tell us the highest position, but /slots doesn't expose it.
                # n_past is the server's tracked token count; after eviction without gap
                # compaction, the KV cache has fewer entries than pos_max suggests.
                # We can't compute the exact gap ratio from /slots alone, but we can
                # detect the symptom: if n_past / n_ctx is high and we've done compressions,
                # gaps are likely accumulating.
                return {
                    "gap_ratio": 0.0,  # Can't compute without pos_max from KV cache
                    "n_past": n_past,
                    "n_ctx": n_ctx,
                    "utilization": n_past / n_ctx if n_ctx > 0 else 0.0,
                    "warning": "",
                }
    except Exception:
        pass
    return {"gap_ratio": 0.0, "warning": ""}


GAP_WARN_THRESHOLD = 0.70  # Warn when context utilization > 70% after compression


def auto_compress_if_needed(
    port: int,
    slot_id: int = 0,
    threshold: float = 0.80,
    keep_ratio: float = 0.50,
    gap_warn_threshold: float = GAP_WARN_THRESHOLD,
    role: str = "",
    layer_adaptive_profile: str = "",
    **kwargs,
) -> CompressResult | None:
    """Auto-trigger compression when KV utilization exceeds threshold.

    Args:
        port: llama-server port
        slot_id: Slot ID
        threshold: Utilization fraction that triggers compression (0.80 = 80%)
        keep_ratio: Target keep ratio when compressing
        gap_warn_threshold: Warn if post-compress utilization still above this
        role: Model role name (enables layer-adaptive compression when set)
        layer_adaptive_profile: Layer weight profile ("balanced", "aggressive", "conservative").
            Empty string disables layer-adaptive; uses uniform weights.
        **kwargs: Additional args passed to compress_slot()

    Returns:
        CompressResult if compression was triggered, None if below threshold
    """
    utilization = get_slot_utilization(port, slot_id)
    if utilization < threshold:
        return None

    log.info(
        "Auto-compress triggered: port=%d slot=%d util=%.1f%% > threshold=%.1f%%",
        port, slot_id, utilization * 100, threshold * 100,
    )

    # Use layer-adaptive compression when role is known and profile specified
    if role and layer_adaptive_profile and role in MODEL_LAYER_COUNTS:
        result = compress_slot_adaptive(
            port, role, slot_id=slot_id, keep_ratio=keep_ratio,
            profile=layer_adaptive_profile, **kwargs,
        )
    else:
        result = compress_slot(port, slot_id, keep_ratio=keep_ratio, **kwargs)

    # Gap accumulation guardrail: if post-compress utilization is still high,
    # gaps are consuming the context window. Recommend slot erase + re-prefill.
    if result.success:
        post_util = get_slot_utilization(port, slot_id)
        if post_util > gap_warn_threshold:
            msg = (
                f"GAP ACCUMULATION WARNING: port={port} slot={slot_id} "
                f"post-compress util={post_util:.1%} > {gap_warn_threshold:.0%}. "
                f"Position gaps consuming context window. "
                f"Recommend: erase slot and re-prefill, or use knorm scorer (serialize/restore)."
            )
            log.warning(msg)
            result.error = msg  # surface in result for journal logging

    return result


# ── Layer-Adaptive Compression ──────────────────────────────────

# Layer-adaptive keep ratios from AM P2 validation (Qwen2.5-7B):
#   Early layers (L0): tolerate 10x compression (keep_ratio ≈ 0.10)
#   Middle layers (L14): tolerate 5x compression (keep_ratio ≈ 0.20)
#   Deep layers (L27): tolerate 2x compression (keep_ratio ≈ 0.50)
#
# The EA scorer uses layer_weights to bias which layers' importance
# scores contribute most to the eviction decision. Higher weight =
# that layer's per-token scores contribute more to the aggregated
# ranking. Deep layers get higher weight because their tokens are
# harder to compress without quality loss.

# Preset profiles mapping layer depth to importance weight
LAYER_PROFILES = {
    "conservative": {"early": 1.0, "mid": 2.0, "deep": 5.0},
    "aggressive": {"early": 0.5, "mid": 1.0, "deep": 5.0},
    "balanced": {"early": 1.0, "mid": 3.0, "deep": 10.0},
}


def compute_layer_adaptive_weights(
    n_layers: int,
    profile: str = "balanced",
) -> list[float]:
    """Compute per-layer importance weights for EA scoring.

    Divides layers into three zones (early/mid/deep, each ~1/3) and
    assigns importance weights from the named profile. Deep layers
    get higher weights because their attention patterns are more
    sensitive to token eviction.

    Args:
        n_layers: Number of attention layers in the model.
        profile: One of "conservative", "aggressive", "balanced".

    Returns:
        List of n_layers floats — per-layer importance weights.
    """
    if profile not in LAYER_PROFILES:
        log.warning("Unknown layer profile %r, using balanced", profile)
        profile = "balanced"

    p = LAYER_PROFILES[profile]
    third = max(1, n_layers // 3)

    weights = []
    for i in range(n_layers):
        if i < third:
            weights.append(p["early"])
        elif i < 2 * third:
            weights.append(p["mid"])
        else:
            weights.append(p["deep"])

    return weights


# Known model layer counts (attention layers only, from model_registry.yaml)
MODEL_LAYER_COUNTS = {
    "frontdoor": 28,          # Qwen3.5-35B-A3B (28 attention layers)
    "coder_escalation": 64,   # Qwen2.5-Coder-32B
    "architect_general": 64,  # Qwen3.5-122B-A10B
    "architect_coding": 64,   # REAP-246B
    "worker_explore": 28,     # Qwen3-Coder-30B-A3B
    "ingest_long_context": 32,  # Qwen3-Next-80B-A3B (SSM layers excluded)
}


def compress_slot_adaptive(
    port: int,
    role: str,
    slot_id: int = 0,
    keep_ratio: float = 0.50,
    profile: str = "balanced",
    **kwargs,
) -> CompressResult:
    """Compress KV cache with layer-adaptive importance weighting.

    Computes per-layer weights based on the model's layer count and
    the specified profile, then delegates to compress_slot().

    Args:
        port: llama-server port.
        role: Model role name (for layer count lookup).
        slot_id: Slot ID (usually 0).
        keep_ratio: Fraction of KV entries to keep.
        profile: Layer weight profile ("conservative", "aggressive", "balanced").
        **kwargs: Additional args for compress_slot().

    Returns:
        CompressResult with eviction details.
    """
    n_layers = MODEL_LAYER_COUNTS.get(role)
    if n_layers is None:
        log.info("No layer count for role %r, using uniform weights", role)
        return compress_slot(port, slot_id, keep_ratio=keep_ratio, **kwargs)

    weights = compute_layer_adaptive_weights(n_layers, profile)
    log.info(
        "Layer-adaptive compress: role=%s n_layers=%d profile=%s weights=[%.1f..%.1f..%.1f]",
        role, n_layers, profile, weights[0], weights[len(weights) // 2], weights[-1],
    )
    return compress_slot(
        port, slot_id, keep_ratio=keep_ratio,
        layer_weights=weights, **kwargs,
    )


# Production port assignments (from orchestrator_stack.py)
PRODUCTION_PORTS = {
    "frontdoor": 8070,
    "coder": 8071,
    "worker": 8072,
    "architect_general": 8083,
    "architect_coding": 8084,
}


def auto_compress_all(
    threshold: float = 0.80,
    keep_ratio: float = 0.50,
    layer_adaptive_profile: str = "",
    **kwargs,
) -> dict[str, CompressResult | None]:
    """Auto-compress all production slots above threshold.

    Args:
        layer_adaptive_profile: If non-empty, use layer-adaptive weights for all roles.

    Returns dict of role → CompressResult (or None if below threshold).
    """
    results = {}
    for role, port in PRODUCTION_PORTS.items():
        results[role] = auto_compress_if_needed(
            port, slot_id=0, threshold=threshold, keep_ratio=keep_ratio,
            role=role, layer_adaptive_profile=layer_adaptive_profile, **kwargs,
        )
    return results
