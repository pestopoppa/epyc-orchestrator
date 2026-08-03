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


def work_completion_meta(
    *,
    answer: Any = None,
    tool_calls: Any = None,
    repl_steps: Any = None,
    reasoning: Any = None,
) -> dict[str, Any]:
    """Return the ``work`` payload for a TASK_COMPLETED progress record.

    M-11a2b. This is the only channel from the request-scoped pipeline to the
    episodic writer: scoring runs asynchronously in a thread pool keyed by
    task_id (``memrl.score_completed_task``), and ``QScorer._score_task``
    reconstructs everything it knows from the progress trajectory. So work that
    is not put on the TASK_COMPLETED entry cannot reach ``memories.context``.

    Redaction and size bounds come from
    ``orchestration.repl_memory.memory_record.build_work_payload`` — the same
    single policy the episodic write site applies — so the progress JSONL and the
    episodic store never diverge on what was considered safe to persist. The
    progress log already carries up to ``PROGRESS_OBJECTIVE_LOG_LEN`` (2,000)
    chars of raw prompt, so this adds no new category of content to it.

    Returns {} when there is no work, so callers can splat it unconditionally.
    """
    from orchestration.repl_memory.memory_record import build_work_payload

    payload = build_work_payload(
        answer=answer,
        tool_calls=tool_calls,
        repl_steps=repl_steps,
        reasoning=reasoning,
    )
    return {"work": payload} if payload else {}
