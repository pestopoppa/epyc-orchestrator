"""Normalised trace model. Only declared fields are read; ``from_dict`` drops everything else."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STEP_FIELDS = (
    "idx", "kind", "tool", "args", "args_raw", "observation", "exit_status", "error",
    "truncated", "finish_reason", "tokens_in", "is_final", "mutating", "verifying",
    # HIL-13a capture (src/trace/step_events.py): arguments/observations are not stored, so these
    # carry what the detectors need instead of the raw text.
    "args_hash", "state_digest", "invocation_error", "output_bytes", "latency_ms", "error_class",
)
TRACE_FIELDS = (
    "trace_id", "task_key", "role", "model_pin", "arm", "source", "outcome", "status",
    "exception", "step_cap", "ctx_limit", "tool_required",
)


@dataclass
class L2Step:
    idx: int
    kind: str = "tool_call"            # tool_call | final | message
    tool: str | None = None
    args: dict | None = None
    args_raw: str | None = None        # raw argument text when it did not parse
    observation: str | None = None
    exit_status: int | None = None
    error: bool | None = None          # structured flag; None = unknown (never inferred here)
    truncated: bool | None = None
    finish_reason: str | None = None
    tokens_in: int | None = None
    is_final: bool = False
    mutating: bool | None = None
    verifying: bool | None = None
    args_hash: str | None = None         # args unobserved: schema checks are skipped, see detect()
    state_digest: str | None = None      # precomputed state_hash(tool, args, normalised observation)
    invocation_error: str | None = None  # registry verdict at call time: unknown_tool | invalid_arguments
    output_bytes: int | None = None
    latency_ms: float | None = None
    error_class: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "L2Step":
        return cls(**{k: d[k] for k in STEP_FIELDS if k in d})


@dataclass
class L2Trace:
    trace_id: str
    task_key: str
    outcome: str = "unknown"           # pass | fail | unknown
    role: str | None = None
    model_pin: str | None = None
    arm: str | None = None
    source: str | None = None
    status: str | None = None
    exception: str | None = None
    step_cap: int | None = None
    ctx_limit: int | None = None
    tool_required: bool | None = None  # None = not captured (unobserved); only True supports zero_step claims
    steps: list[L2Step] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "L2Trace":
        kw = {k: d[k] for k in TRACE_FIELDS if k in d}
        steps = [L2Step.from_dict(s) for s in d.get("steps", [])]
        return cls(steps=steps, **kw)

    def tool_steps(self) -> list[L2Step]:
        return [s for s in self.steps if s.kind == "tool_call"]
