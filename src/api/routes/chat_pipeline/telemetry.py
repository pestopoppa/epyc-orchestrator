"""Shared telemetry helpers for chat pipeline progress records."""

from __future__ import annotations

from typing import Any


def llm_completion_meta(primitives: Any) -> dict[str, Any]:
    """Return token/timing metadata from an LLMPrimitives-like object."""
    return {
        "tokens_generated": int(getattr(primitives, "total_tokens_generated", 0) or 0),
        "prompt_eval_ms": float(getattr(primitives, "total_prompt_eval_ms", 0.0) or 0.0),
        "generation_ms": float(getattr(primitives, "total_generation_ms", 0.0) or 0.0),
        "http_overhead_ms": float(getattr(primitives, "total_http_overhead_ms", 0.0) or 0.0),
    }
