"""Versioned phase observations shared by the serving-call writer and readers."""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "epyc.orchestrator.serving_phases.v1"
NATIVE_COUNTERS = {
    "fresh_prompt_tokens": "prompt_n",
    "cached_prompt_tokens": "cache_n",
    "generated_tokens": "predicted_n",
    "prefill_ms": "prompt_ms",
    "decode_ms": "predicted_ms",
    "draft_proposed_tokens": "draft_n",
    "draft_accepted_tokens": "draft_n_accepted",
}


def nonnegative(value: Any) -> int | float | None:
    if type(value) is int and 0 <= value <= 2**53:
        return value
    if type(value) is float and math.isfinite(value) and value >= 0:
        return value
    return None


def phase_breakdown(
    timings: dict[str, Any],
    *,
    trial_id: int | str | None,
    batch_id: str | None,
    request_id: str | None,
    stream_id: str,
    streamed: bool | None,
    wall_ms: float | None,
    delivered_first_ms: float | None,
) -> dict[str, Any]:
    """Project native counters; phase durations alone cannot locate intervals."""
    values = {name: nonnegative(timings.get(key)) for name, key in NATIVE_COUNTERS.items()}
    for name in (key for key in values if key.endswith("_tokens")):
        if type(values[name]) is not int:
            values[name] = None
    return {
        "schema": SCHEMA,
        "trial_id": trial_id,
        "batch_id": batch_id,
        "request_id": request_id,
        "stream_id": stream_id,
        "streamed": streamed if isinstance(streamed, bool) else None,
        "timing_source": "server_native_counters" if timings else "absent",
        **values,
        "client_wall_ms": nonnegative(wall_ms),
        "delivered_first_ms": nonnegative(delivered_first_ms) if streamed is True else None,
        "first_delta_scope": "backend_client_after_dispatch"
        if streamed is True and nonnegative(delivered_first_ms) is not None
        else None,
        "wall_scope": "recorded_backend_call",
    }
