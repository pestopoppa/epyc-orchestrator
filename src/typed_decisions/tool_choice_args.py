"""TD-12: one-call tool + argument arm (operation x conditional target), shadow-only.

One typed-decision call asks, over the role's tool catalogue, (a) which tool to use (a
``CHOICE`` question, id ``__op__``) and (b) one argument head set PER candidate tool, each
head's text naming the tool it assumes. Only the resolved operation's branch is consumed:
that branch must be complete and pass ``assemble_arguments`` (the same schema validation as
TD-4). Everything else is kept strict:

* the response envelope is validated by the general runner (all-questions schema) exactly as
  for every other typed call; this module adapts nothing there;
* a per-question failure record in an UNUSED branch (reason ``invalid_value`` naming a
  catalogue question of another tool) is ignored and listed in ``ignored_unused_failures``;
  it is NEVER repaired or re-run;
* any other failure record (``no_json`` / ``schema_violation`` / ``transport_error``, an
  unknown id, an unparsable detail), a failure on the operation question or on the selected
  branch, an unknown operation value, or a selected branch that does not assemble, abstains
  (``status="abstained"``, fallback to the incumbent). Nothing is guessed.

The proposal never changes the executed tool: ``operation_differs`` is recorded for the
TD-29 shadow receipt; host-owned prepared-action authority, the eligible menu, freshness and
read-set revalidation stay with the host (this module sees only schemas). Joint
probabilities are NOT presumed independent (TD-13 tests sibling dependence; adoption is gated
on it).

Default off: ``tool_choice_enabled()`` reads ``TYPED_DECISIONS_TOOL_CHOICE_ARGS``. No live
path imports this module. Code and unit tests are zero-inference; the live comparison against
REPL generation is the champion-sidecar CPU-window gate. Hierarchical codes / chunked
candidate tables (absorbed TD-15) are not built here: they are needed only past the TD-9
36-symbol map or the 128 ``n_probs`` cap, which this JSON-mode arm does not hit.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from src.typed_decisions.tool_args import (
    ToolArgumentError,
    assemble_arguments,
    tool_schema_to_questions,
)
from src.typed_decisions.types import Decision, DecisionResult, Question, QuestionKind

ENV_FLAG = "TYPED_DECISIONS_TOOL_CHOICE_ARGS"
OP_ID = "__op__"
NAMESPACE_SEP = "::"
_FAILURE_ID_RE = re.compile(r"^question (['\"])(.*)\1: ")
_REASON_INVALID_VALUE = "invalid_value"


def tool_choice_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(ENV_FLAG, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class JointCatalogue:
    questions: tuple[Question, ...]
    tools: tuple[str, ...]  # operation options, in the order given
    branches: Mapping[str, tuple[tuple[Question, ...], Mapping[str, Any]]]  # tool -> (orig qs, schema)
    id_map: Mapping[str, tuple[str, str]]  # namespaced id -> (tool, original id)
    excluded: Mapping[str, str] = field(default_factory=dict)  # tool -> reason


@dataclass(frozen=True)
class JointProposal:
    status: str  # "proposed" | "abstained"
    tool: str | None
    arguments: dict[str, Any] | None
    reason: str | None
    operation_differs: bool | None = None
    ignored_unused_failures: tuple[str, ...] = ()


def _single_line(text: str) -> str:
    return " ".join(str(text).split())


def build_joint_catalogue(
    tools: Mapping[str, Mapping[str, Any]],
    *,
    descriptions: Mapping[str, str] | None = None,
    operation_text: str = "Which tool should be called next?",
) -> JointCatalogue:
    """``tools``: tool name -> complete JSON-schema ``parameters`` object.

    A tool whose REQUIRED argument cannot be mapped to a closed set cannot be completed by
    heads, so it is excluded from the operation options (reason recorded), not guessed.
    Fewer than two usable tools is an error: the arm only means something over a real choice.
    """
    usable: dict[str, tuple[tuple[Question, ...], Mapping[str, Any]]] = {}
    excluded: dict[str, str] = {}
    for name, schema in tools.items():
        if NAMESPACE_SEP in name or not name or name == OP_ID:
            excluded[name] = "tool name unusable as a namespace"
            continue
        try:
            mapping = tool_schema_to_questions(name, dict(schema))
        except ToolArgumentError as exc:
            excluded[name] = f"schema not an object: {exc}"
            continue
        required = set(schema.get("required", ()))
        blocked = sorted(
            a for a in required if any(s.startswith(f"{a}: ") for s in mapping.skipped)
            or a not in (schema.get("properties") or {})
        )
        if blocked:
            excluded[name] = f"required argument(s) not closed-set mappable: {blocked}"
            continue
        usable[name] = (tuple(mapping.questions), dict(schema))
    if len(usable) < 2:
        raise ValueError(
            f"joint tool+argument arm needs >= 2 usable tools, got {len(usable)} "
            f"(excluded: {excluded})"
        )
    names = tuple(usable)
    descs = descriptions or {}
    op = Question(
        id=OP_ID,
        kind=QuestionKind.CHOICE,
        text=operation_text,
        options=names,
        option_descriptions=tuple(
            _single_line(descs[n]) if descs.get(n, "").strip() else None for n in names
        ),
    )
    questions: list[Question] = [op]
    id_map: dict[str, tuple[str, str]] = {}
    for name in names:
        for q in usable[name][0]:
            nid = f"{name}{NAMESPACE_SEP}{q.id}"
            id_map[nid] = (name, q.id)
            questions.append(
                replace(q, id=nid, text=f"Assuming the chosen tool is {name!r}: {q.text}")
            )
    return JointCatalogue(tuple(questions), names, usable, id_map, excluded)


def _failure_question_id(detail: str) -> str | None:
    m = _FAILURE_ID_RE.match(detail or "")
    return m.group(2) if m else None


def resolve_joint(
    catalogue: JointCatalogue,
    result: DecisionResult,
    *,
    executed_tool: str | None = None,
) -> JointProposal:
    """Consume only the resolved operation's complete, schema-valid branch."""

    def abstain(reason: str, ignored: Sequence[str] = ()) -> JointProposal:
        return JointProposal("abstained", None, None, reason, None, tuple(ignored))

    decisions = {d.question_id: d for d in result.decisions}
    op = decisions.get(OP_ID)
    ignored_candidates: list[tuple[str, str]] = []
    for failure in result.failures:
        qid = _failure_question_id(failure.detail)
        if failure.reason != _REASON_INVALID_VALUE or qid is None:
            return abstain(f"malformed_or_envelope_failure:{failure.reason}")
        if qid == OP_ID:
            return abstain("operation_failed")
        if qid not in catalogue.id_map:
            return abstain("failure_names_unknown_question")
        ignored_candidates.append((qid, failure.detail))
    if op is None:
        return abstain("operation_unresolved")
    tool = op.value
    if not isinstance(tool, str) or tool not in catalogue.branches:
        return abstain("operation_not_in_catalogue")
    ignored: list[str] = []
    for qid, detail in ignored_candidates:
        if catalogue.id_map[qid][0] == tool:
            return abstain("selected_branch_failure")
        ignored.append(detail)
    questions, schema = catalogue.branches[tool]
    branch: dict[str, Decision] = {}
    for nid, (t, orig) in catalogue.id_map.items():
        if t == tool and nid in decisions:
            branch[orig] = replace(decisions[nid], question_id=orig)
    try:
        args = assemble_arguments(questions, branch, schema)
    except ToolArgumentError as exc:
        return abstain("selected_branch_invalid: " + "; ".join(exc.reasons), ignored)
    return JointProposal(
        "proposed", tool, args, None,
        None if executed_tool is None else tool != executed_tool,
        tuple(ignored),
    )


def propose_joint_tool_call(
    *,
    tools: Mapping[str, Mapping[str, Any]],
    primitives: Any,
    state: str,
    role: str,
    executed_tool: str | None = None,
    descriptions: Mapping[str, str] | None = None,
    mode: str = "json",
    environ: Mapping[str, str] | None = None,
) -> JointProposal:
    """Shadow-only: one typed call, then ``resolve_joint``. Never executes anything.

    Returns an abstention when the flag is off, the state is empty, the catalogue cannot be
    built, or the call raises. A native-mode run keeps the runner's own eligibility rules."""
    if not tool_choice_enabled(environ):
        return JointProposal("abstained", None, None, "flag_off")
    if not isinstance(state, str) or not state.strip():
        return JointProposal("abstained", None, None, "invalid_state")
    try:
        catalogue = build_joint_catalogue(tools, descriptions=descriptions)
    except ValueError as exc:
        return JointProposal("abstained", None, None, f"catalogue_unbuildable: {exc}")
    try:
        from src.typed_decisions.runner import run_typed_decisions

        result = run_typed_decisions(
            primitives, state=state, questions=catalogue.questions, role=role,
            mode=mode, value_only=(mode == "json"),
        )
    except Exception as exc:  # fail open to the incumbent
        return JointProposal("abstained", None, None, f"{type(exc).__name__}: {exc}")
    return resolve_joint(catalogue, result, executed_tool=executed_tool)


def shadow_receipt_fields(proposal: JointProposal, executed_tool: str | None) -> dict[str, Any]:
    """Fields for the TD-29 shadow receipt: the typed pick is written BESIDE the executed
    tool; nothing here feeds dispatch."""
    return {
        "joint_status": proposal.status,
        "joint_reason": proposal.reason,
        "proposed_tool": proposal.tool,
        "executed_tool": executed_tool,
        "operation_differs": proposal.operation_differs,
        "proposed_arguments": proposal.arguments,
        "ignored_unused_failures": list(proposal.ignored_unused_failures),
    }
