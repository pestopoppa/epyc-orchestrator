"""Trace feedback helpers for EvalTower.

Pure formatting and trace-bank logic lives here so eval_tower.py does not own
PromptForge trace IR, trace-bank retention, and eval execution in one module.
The public EvalTower methods remain compatibility wrappers.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def trim_trace_text(trace_text: Any, max_chars: int) -> str:
    text = str(trace_text or "").strip()
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return "[trace truncated]\n" + text[-max_chars:]


def trace_ir_steps(
    trace_text: str, *, max_steps: int = 12, preview_chars: int = 240
) -> list[dict[str, Any]]:
    """Convert a tap tail into compact ROLE/PROMPT/RESPONSE steps."""
    text = str(trace_text or "").strip()
    if not text:
        return []

    sections: list[tuple[str, str]] = []
    current_kind = "trace"
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        upper = line.upper()
        if upper.startswith("ROLE"):
            if current_lines:
                sections.append((current_kind, "\n".join(current_lines).strip()))
            sections.append(("role", line.strip()))
            current_kind = "trace"
            current_lines = []
        elif upper in {"PROMPT:", "PROMPT"}:
            if current_lines:
                sections.append((current_kind, "\n".join(current_lines).strip()))
            current_kind = "prompt"
            current_lines = []
        elif upper in {"RESPONSE:", "RESPONSE"}:
            if current_lines:
                sections.append((current_kind, "\n".join(current_lines).strip()))
            current_kind = "response"
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_kind, "\n".join(current_lines).strip()))

    steps: list[dict[str, Any]] = []
    for kind, content in sections:
        cleaned = content.strip()
        if not cleaned:
            continue
        steps.append(
            {
                "step_id": f"s{len(steps) + 1}",
                "kind": kind,
                "line_count": len(cleaned.splitlines()),
                "content_hash": hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12],
                "content_preview": cleaned[:preview_chars],
            }
        )
        if len(steps) >= max_steps:
            break
    return steps


def build_critic_trace_ir(
    *,
    trace_bank: list[dict[str, Any]] | None = None,
    raw_trace_text: str = "",
    trial_id: int | None = None,
    failure_summary: str = "",
    k_success: int = 2,
    k_failure: int = 2,
    max_trace_chars: int = 1600,
) -> dict[str, Any]:
    """Build a deterministic, observe-only trace IR for critic/prompt context.

    This is a structured companion to the legacy formatted trace text. It is
    intentionally not consumed by any score, safety, or acceptance gate.
    """

    def selected(outcome: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        matches = [
            item
            for item in trace_bank or []
            if isinstance(item, dict)
            and str(item.get("outcome") or "").lower() == outcome
            and str(item.get("trace") or "").strip()
        ]
        return matches[-limit:]

    examples: list[dict[str, Any]] = []
    for outcome, limit in (("success", int(k_success)), ("failure", int(k_failure))):
        for raw in selected(outcome, limit):
            trace = trim_trace_text(raw.get("trace", ""), max_trace_chars)
            if not trace:
                continue
            examples.append(
                {
                    "outcome": outcome,
                    "trial_id": raw.get("trial_id"),
                    "species": str(raw.get("species") or ""),
                    "action_type": str(raw.get("action_type") or ""),
                    "reason": str(raw.get("reason") or "")[:500],
                    "trace_hash": str(
                        raw.get("trace_hash")
                        or hashlib.sha256(trace.encode("utf-8")).hexdigest()[:12]
                    ),
                    "steps": trace_ir_steps(trace),
                }
            )

    raw_tail = ""
    if not examples:
        raw_tail = trim_trace_text(raw_trace_text, max_trace_chars)
        if raw_tail:
            examples.append(
                {
                    "outcome": "unlabeled",
                    "trial_id": trial_id,
                    "species": "",
                    "action_type": "",
                    "reason": "raw_recent_trace_fallback",
                    "trace_hash": hashlib.sha256(raw_tail.encode("utf-8")).hexdigest()[:12],
                    "steps": trace_ir_steps(raw_tail),
                }
            )

    return {
        "schema_version": "harness_trace_ir.v1",
        "observe_only": True,
        "acceptance_effect": "none_observe_only",
        "trial_id": trial_id,
        "failure_summary": str(failure_summary or "")[:500],
        "source": "contrastive_trace_bank"
        if trace_bank and examples and not raw_tail
        else "raw_recent_traces",
        "trace_examples": examples,
    }


def format_critic_trace_ir(trace_ir: dict[str, Any] | None) -> str:
    """Render critic trace IR as a prompt-safe JSON block."""
    if not isinstance(trace_ir, dict) or not trace_ir.get("trace_examples"):
        return ""
    return (
        "## Harness Trace IR (MH-11 observe-only)\n"
        "This structured trace evidence is diagnostic context only; it is not "
        "an acceptance score or quality gate.\n"
        "```json\n"
        f"{json.dumps(trace_ir, sort_keys=True, indent=2)}\n"
        "```"
    )


def update_contrastive_trace_bank(
    trace_bank: list[dict[str, Any]] | None,
    *,
    trace_text: str,
    outcome: str,
    trial_id: int | None = None,
    species: str = "",
    action_type: str = "",
    reason: str = "",
    max_examples_per_outcome: int = 8,
    max_trace_chars: int = 1600,
) -> list[dict[str, Any]]:
    """Append one labeled trace example and cap the in-state contrastive bank."""
    normalized_outcome = str(outcome or "").strip().lower()
    if normalized_outcome not in {"success", "failure"}:
        return list(trace_bank or [])
    trace = trim_trace_text(trace_text, max_trace_chars)
    if not trace:
        return list(trace_bank or [])

    normalized: list[dict[str, Any]] = []
    for raw in trace_bank or []:
        if not isinstance(raw, dict):
            continue
        raw_outcome = str(raw.get("outcome") or "").strip().lower()
        if raw_outcome not in {"success", "failure"}:
            continue
        raw_trace = trim_trace_text(raw.get("trace", ""), max_trace_chars)
        if not raw_trace:
            continue
        normalized.append(
            {
                "outcome": raw_outcome,
                "trial_id": raw.get("trial_id"),
                "species": str(raw.get("species") or ""),
                "action_type": str(raw.get("action_type") or ""),
                "reason": str(raw.get("reason") or ""),
                "trace": raw_trace,
                "trace_hash": str(
                    raw.get("trace_hash")
                    or hashlib.sha256(raw_trace.encode("utf-8")).hexdigest()[:12]
                ),
            }
        )

    trace_hash = hashlib.sha256(trace.encode("utf-8")).hexdigest()[:12]
    normalized = [
        item
        for item in normalized
        if not (
            item.get("outcome") == normalized_outcome
            and item.get("trial_id") == trial_id
            and item.get("trace_hash") == trace_hash
        )
    ]
    normalized.append(
        {
            "outcome": normalized_outcome,
            "trial_id": trial_id,
            "species": str(species or ""),
            "action_type": str(action_type or ""),
            "reason": str(reason or ""),
            "trace": trace,
            "trace_hash": trace_hash,
        }
    )

    capped: list[dict[str, Any]] = []
    for bucket in ("success", "failure"):
        capped.extend(
            [item for item in normalized if item.get("outcome") == bucket][
                -max_examples_per_outcome:
            ]
        )
    return capped


def format_contrastive_traces(
    *,
    k_success: int = 2,
    k_failure: int = 2,
    trace_bank: list[dict[str, Any]] | None = None,
) -> str:
    """Format labeled success/failure trace examples for PromptForge."""
    if not trace_bank:
        return ""

    def selected(outcome: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        matches = [
            item
            for item in trace_bank
            if isinstance(item, dict)
            and str(item.get("outcome") or "").lower() == outcome
            and str(item.get("trace") or "").strip()
        ]
        return matches[-limit:]

    success_examples = selected("success", int(k_success))
    failure_examples = selected("failure", int(k_failure))
    if not success_examples and not failure_examples:
        return ""

    def append_entry(lines: list[str], idx: int, entry: dict[str, Any]) -> None:
        trial = entry.get("trial_id")
        label_parts = []
        if trial is not None:
            label_parts.append(f"trial #{trial}")
        species = str(entry.get("species") or "").strip()
        action_type = str(entry.get("action_type") or "").strip()
        if species or action_type:
            label_parts.append("/".join(part for part in (species, action_type) if part))
        label = ", ".join(label_parts) or "unlabeled trial"
        lines.append(f"[{idx}] {label}")
        reason = str(entry.get("reason") or "").strip()
        if reason:
            lines.append(f"Reason: {reason}")
        lines.append("Trace:")
        lines.append(str(entry.get("trace") or "").strip())

    lines: list[str] = ["## Contrastive Execution Traces"]
    if success_examples:
        lines.append("### Success Examples")
        for idx, entry in enumerate(success_examples, start=1):
            append_entry(lines, idx, entry)
    if failure_examples:
        lines.append("### Failure Examples")
        for idx, entry in enumerate(failure_examples, start=1):
            append_entry(lines, idx, entry)
    return "\n".join(lines)
