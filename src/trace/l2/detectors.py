"""Rule detectors for the mechanical L2 categories. No model, no free-text judgement."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.trace.l2.model import L2Step, L2Trace

CATEGORIES = (
    "tool_error", "truncation", "loop", "premature_wrap_up",
    "zero_step_violation", "invalid_invocation",
)
# Named, deliberately unimplemented: they need a judge.
JUDGE_ONLY = ("fabricated_fact", "instruction_not_followed")

_VOLATILE = re.compile(
    r"\d{4}-\d\d-\d\d[T ][\d:.]+Z?|0x[0-9a-fA-F]+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\d+\.\d+s\b"
)


def _loop_guard_noprogress(made_progress: bool, prev_count: int) -> int:
    """The live REPL guard's counter (src/graph/helpers.py), reused when importable.

    The import is lazy and exact; the identical one-line fallback keeps the scan usable in a
    process that cannot import the graph package. ``counter_source()`` reports which was used.
    """
    return _COUNTER(made_progress, prev_count)


def _resolve_counter():
    try:
        from src.graph.helpers import _loop_guard_noprogress as live
        return live, "src.graph.helpers"
    except Exception:  # pragma: no cover - environment dependent
        return (lambda made, prev: 0 if made else prev + 1), "local-fallback"


_COUNTER, _COUNTER_SOURCE = _resolve_counter()


def counter_source() -> str:
    return _COUNTER_SOURCE


@dataclass
class DetectorConfig:
    loop_threshold: int = 2                 # live guard halts at >= 2 non-advancing turns
    byte_cap: int | None = None             # observation byte cap that implies truncation
    ctx_frac: float = 0.85                  # wrap-up near the context limit
    mutating_re: str = r"write|edit|patch|create|apply|delete|commit|replace|insert|append"
    verifying_re: str = r"test|verify|check|lint|build|compile|assert|validate"
    schemas: dict[str, dict] = field(default_factory=dict)   # tool -> JSON schema
    truncation_status_re: str = r"trunc"


@dataclass(frozen=True)
class Detection:
    category: str
    step_idx: int | None        # None = trace-level (status / exception field)
    signature: str


def state_hash(step: L2Step) -> str:
    obs = _VOLATILE.sub("#", " ".join((step.observation or "").split()))
    args = json.dumps(step.args, sort_keys=True, default=str) if step.args is not None else (step.args_raw or "")
    return hashlib.sha1(f"{step.tool}\x1f{args}\x1f{obs}".encode("utf-8", "replace")).hexdigest()


def _is_mutating(s: L2Step, cfg: DetectorConfig) -> bool:
    if s.mutating is not None:
        return s.mutating
    return bool(s.tool and re.search(cfg.mutating_re, s.tool, re.I))


def _is_verifying(s: L2Step, cfg: DetectorConfig) -> bool:
    if s.verifying is not None:
        return s.verifying
    return bool(s.tool and re.search(cfg.verifying_re, s.tool, re.I))


def _validate(schema: dict, args: Any) -> str | None:
    from jsonschema import Draft202012Validator
    errs = sorted(Draft202012Validator(schema).iter_errors(args), key=lambda e: list(e.absolute_path))
    return errs[0].validator if errs else None


def detect(t: L2Trace, cfg: DetectorConfig | None = None) -> list[Detection]:
    cfg = cfg or DetectorConfig()
    out: list[Detection] = []
    tools = t.tool_steps()

    # zero-step / first-turn protocol violation
    if not tools:
        out.append(Detection("zero_step_violation", None, "zero_step"))
    elif t.tool_required and t.steps and t.steps[0].kind != "tool_call":
        out.append(Detection("zero_step_violation", t.steps[0].idx, "first_turn_no_tool_call"))

    prev, count, fired = None, 0, False
    for s in t.steps:
        if s.kind != "tool_call":
            continue
        if s.error is True or (s.exit_status not in (None, 0)):
            out.append(Detection("tool_error", s.idx, "error_flag" if s.error else f"exit_status"))
        if (s.truncated is True or s.finish_reason in ("length", "max_tokens")
                or (cfg.byte_cap and s.observation is not None
                    and len(s.observation.encode("utf-8", "replace")) >= cfg.byte_cap)):
            out.append(Detection("truncation", s.idx,
                                 "flag" if s.truncated else ("finish_reason" if s.finish_reason else "byte_cap")))
        # invalid_invocation
        if s.args is None and s.args_raw:
            out.append(Detection("invalid_invocation", s.idx, "unparseable_args"))
        elif cfg.schemas and s.tool is not None:
            sch = cfg.schemas.get(s.tool)
            if sch is None:
                out.append(Detection("invalid_invocation", s.idx, "unknown_tool"))
            elif s.args is not None:
                v = _validate(sch, s.args)
                if v:
                    out.append(Detection("invalid_invocation", s.idx, f"schema:{v}"))
        # loop: state = hash(tool, args, normalised observation)
        h = state_hash(s)
        count = _loop_guard_noprogress(h != prev, count) if prev is not None else 0
        prev = h
        if count >= cfg.loop_threshold and not fired:
            out.append(Detection("loop", s.idx, f"repeat_state>={cfg.loop_threshold}"))
            fired = True

    # trace-level truncation from the status / exception field
    if re.search(cfg.truncation_status_re, f"{t.status or ''} {t.exception or ''}", re.I):
        out.append(Detection("truncation", None, "status_field"))

    # premature wrap-up
    final = next((s for s in t.steps if s.is_final or s.kind == "final"), None)
    if final is not None:
        muts = [s for s in tools if _is_mutating(s, cfg) and s.idx < final.idx]
        if muts and not any(_is_verifying(s, cfg) and s.idx > muts[-1].idx and s.idx < final.idx for s in tools):
            out.append(Detection("premature_wrap_up", final.idx, "no_verify_after_mutation"))
        if t.ctx_limit and final.tokens_in and final.tokens_in >= cfg.ctx_frac * t.ctx_limit:
            out.append(Detection("premature_wrap_up", final.idx, "near_context_limit"))
    return out


def first_visible_deviation(dets: list[Detection]) -> Detection | None:
    """Earliest mechanical event. Step-level events order by step; trace-level ones come last."""
    if not dets:
        return None
    return min(dets, key=lambda d: (d.step_idx is None, d.step_idx if d.step_idx is not None else 0, d.category))


def loop_ratio(t: L2Trace) -> float | None:
    tools = t.tool_steps()
    if not tools:
        return None
    return 1.0 - len({state_hash(s) for s in tools}) / len(tools)


def baseline_first_error_step(t: L2Trace) -> int | None:
    for s in t.tool_steps():
        if s.error is True or s.exit_status not in (None, 0):
            return s.idx
    return None


def baseline_last_step(t: L2Trace) -> int | None:
    return t.steps[-1].idx if t.steps else None
