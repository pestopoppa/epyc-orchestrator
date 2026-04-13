"""Workspace-state helpers for graph execution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.graph.state import TaskState
from src.roles import Role


def _workspace_prompt_block(state: TaskState) -> str:
    """Build a compact workspace block to keep specialists aligned."""
    ws = state.workspace_state or {}
    objective = ws.get("objective") or state.prompt[:240]
    constraints = ws.get("constraints", [])[:4]
    invariants = ws.get("invariants", [])[:4]
    commitments = ws.get("commitments", [])[-3:]
    decisions = ws.get("decisions", [])[-3:]
    open_questions = ws.get("open_questions", [])[-3:]

    lines = [
        "[Workspace State]",
        f"- objective: {objective}",
    ]
    if constraints:
        lines.append(f"- constraints: {constraints}")
    if invariants:
        lines.append(f"- invariants: {invariants}")
    if commitments:
        lines.append(f"- commitments: {commitments}")
    if decisions:
        lines.append(f"- decisions: {decisions}")
    if open_questions:
        lines.append(f"- open_questions: {open_questions}")
    task_manager = getattr(state, "task_manager", None)
    if task_manager and task_manager.has_tasks():
        task_lines = task_manager.summary_block(limit=8)
        if task_lines:
            lines.append("- task_progress:")
            for line in task_lines:
                lines.append(f"  {line}")
    if state.anti_pattern_warning:
        lines.append(f"- warning: {state.anti_pattern_warning[:240]}")
    return "\n".join(lines)


def _update_workspace_from_turn(
    state: TaskState,
    role: Role | str,
    output: str,
    error: str | None,
) -> None:
    """Update workspace via proposal -> selection -> broadcast cycle."""
    ws = state.workspace_state
    if not ws:
        return
    if not ws.get("objective"):
        ws["objective"] = state.prompt[:240]

    role_name = str(role)
    proposal = None
    if error:
        proposal = {
            "id": f"p{state.turns}",
            "kind": "open_question",
            "owner": role_name,
            "text": error[:180],
            "priority": "high",
        }
    elif output and output.strip():
        proposal = {
            "id": f"p{state.turns}",
            "kind": "commitment",
            "owner": role_name,
            "text": output[:180],
            "priority": "normal",
        }

    if proposal:
        proposals = ws.setdefault("proposals", [])
        proposals.append(proposal)
        if len(proposals) > 12:
            ws["proposals"] = proposals[-12:]
        _select_and_broadcast_workspace_delta(ws)

    for key in ("proposals", "commitments", "open_questions", "decisions"):
        vals = ws.get(key, [])
        if isinstance(vals, list) and len(vals) > 12:
            ws[key] = vals[-12:]
    ws["updated_at"] = datetime.now(timezone.utc).isoformat()


def _select_and_broadcast_workspace_delta(ws: dict[str, Any]) -> None:
    """Controller step: select proposals and broadcast merged deltas."""
    proposals = ws.get("proposals", [])
    if not isinstance(proposals, list) or not proposals:
        return

    def _priority_rank(p: dict[str, Any]) -> int:
        kind = str(p.get("kind", ""))
        prio = str(p.get("priority", "normal"))
        if kind == "open_question":
            return 0
        if prio == "high":
            return 1
        return 2

    indexed = list(enumerate(proposals))
    indexed.sort(key=lambda item: (_priority_rank(item[1]), -item[0]))
    selected = [p for _, p in indexed[:2]]
    if not selected:
        return

    broadcast_items: list[dict[str, Any]] = []
    for proposal in selected:
        kind = str(proposal.get("kind", ""))
        owner = str(proposal.get("owner", ""))
        text = str(proposal.get("text", "")).strip()
        if not text:
            continue

        if kind == "open_question":
            target_key = "open_questions"
            prefix = "q"
        else:
            target_key = "commitments"
            prefix = "c"

        target = ws.setdefault(target_key, [])
        if target_key == "commitments":
            target[:] = [x for x in target if str(x.get("owner", "")) != owner]

        if not any(str(item.get("text", "")).strip() == text for item in target):
            entry = {
                "id": f"{prefix}{ws.get('broadcast_version', 0) + len(broadcast_items) + 1}",
                "owner": owner,
                "text": text,
            }
            target.append(entry)
            broadcast_items.append(entry)

        if target_key == "commitments":
            open_questions = ws.get("open_questions", [])
            resolved = ws.setdefault("resolved_questions", [])
            for q in list(open_questions):
                q_text = str(q.get("text", "")).strip().lower()
                if q_text and q_text in text.lower():
                    open_questions.remove(q)
                    resolved.append(q)

    if broadcast_items:
        ws["broadcast_version"] = int(ws.get("broadcast_version", 0)) + 1
        b_log = ws.setdefault("broadcast_log", [])
        b_log.append(
            {
                "version": ws["broadcast_version"],
                "items": broadcast_items,
            }
        )
        if len(b_log) > 20:
            ws["broadcast_log"] = b_log[-20:]
