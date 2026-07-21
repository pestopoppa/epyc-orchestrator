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


def llm_completion_probabilities(primitives: Any) -> list[dict[str, Any]]:
    """Return last-call llama.cpp probability rows when explicitly requested."""
    meta = getattr(primitives, "_last_inference_meta", {}) or {}
    rows = meta.get("completion_probabilities") or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]
