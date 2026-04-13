"""Task IR processing helpers for orchestration graph nodes.

Extracts candidate files from task_ir plans, auto-seeds task managers, gathers
file context via REPL peek, and detects recurring failure anti-patterns from the
FailureGraph. Extracted from graph/helpers.py — all callers continue to import
from helpers via compatibility re-exports.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_graph import GraphRunContext

from src.graph.state import TaskDeps, TaskState

log = logging.getLogger(__name__)

Ctx = GraphRunContext[TaskState, TaskDeps]


def _extract_candidate_files_from_task_ir(state: TaskState) -> list[str]:
    """Extract candidate file paths from task_ir plan steps."""
    task_ir = state.task_ir if isinstance(state.task_ir, dict) else {}
    plan = task_ir.get("plan", {}) if isinstance(task_ir, dict) else {}
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    file_paths: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        for raw_path in step.get("files", []):
            text = str(raw_path).strip()
            if text and text not in file_paths:
                file_paths.append(text)
    return file_paths[:10]


def _auto_seed_tasks_from_task_ir(state: TaskState) -> None:
    """Auto-populate task manager from TaskIR plan on first turn."""
    manager = getattr(state, "task_manager", None)
    if manager is None or manager.has_tasks():
        return
    task_ir = state.task_ir if isinstance(state.task_ir, dict) else {}
    plan = task_ir.get("plan", {}) if isinstance(task_ir, dict) else {}
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    if not isinstance(steps, list):
        return
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).strip()
        if not action:
            continue
        manager.create(
            subject=action,
            description=action,
            active_form=f"Working on step {idx + 1}",
            metadata={"source": "task_ir", "step_id": step.get("id", "")},
            task_type=state.task_type,
        )


def _auto_gather_context(ctx: Ctx, files: list[str]) -> str:
    """Gather file snippets into prompt context using REPL peek."""
    repl = ctx.deps.repl
    if repl is None or not files:
        return ""
    seen = set(ctx.state.gathered_files or [])
    gathered: list[str] = []
    for path in files[:10]:
        if path in seen:
            continue
        try:
            content = repl._peek(200, file_path=path)  # noqa: SLF001
            gathered.append(f"### {path}\n```\n{content}\n```")
            seen.add(path)
        except Exception:
            gathered.append(f"### {path}\n[Could not read]")
    ctx.state.gathered_files = list(seen)
    return "\n\n".join(gathered[:10])


def _check_anti_pattern(ctx: Ctx) -> str | None:
    """Return anti-pattern warning from FailureGraph when recurring failures are detected."""
    fg = ctx.deps.failure_graph
    if fg is None:
        return None
    if ctx.state.consecutive_failures < 2 and not ctx.state.last_error:
        return None
    symptoms: list[str] = []
    if ctx.state.last_error:
        symptoms.append(ctx.state.last_error[:100])
    if ctx.state.consecutive_failures >= 2:
        symptoms.append(f"{ctx.state.current_role}:consecutive_fail_{ctx.state.consecutive_failures}")
    if not symptoms:
        return None
    try:
        matches = fg.find_matching_failures(symptoms)
        if not matches:
            return None
        best = matches[0]
        if int(best.severity) < 3:
            return None
        mitigations = fg.get_effective_mitigations(symptoms)
        if mitigations:
            top = mitigations[0]
            action = str(top.get("action", "unknown"))
            success_rate = float(top.get("success_rate", 0.0))
            return (
                f"Recurring pattern seen before. Prior mitigation: {action} "
                f"(success={success_rate:.0%})."
            )
        return f"Recurring pattern: {str(best.description)[:140]}"
    except Exception as exc:
        log.debug("anti-pattern check failed: %s", exc)
        return None
