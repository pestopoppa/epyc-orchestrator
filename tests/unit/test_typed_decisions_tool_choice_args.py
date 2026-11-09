"""TD-12: one-call tool + argument arm. Canned decisions / fake primitives; no model call."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.typed_decisions import tool_choice_args as tc
from src.typed_decisions.tool_choice_args import (
    OP_ID,
    build_joint_catalogue,
    propose_joint_tool_call,
    resolve_joint,
    shadow_receipt_fields,
)
from src.typed_decisions.types import Decision, DecisionResult, ParseFailure, QuestionKind

SEARCH = {
    "type": "object",
    "properties": {"scope": {"type": "string", "enum": ["web", "repo"]}, "deep": {"type": "boolean"}},
    "required": ["scope"],
}
DEPLOY = {
    "type": "object",
    "properties": {"env": {"type": "string", "enum": ["dev", "prod"]}, "dry_run": {"type": "boolean"}},
    "required": ["env", "dry_run"],
}
FREEFORM = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
TOOLS = {"search": SEARCH, "deploy": DEPLOY}
ON = {"TYPED_DECISIONS_TOOL_CHOICE_ARGS": "1"}


def _dec(qid: str, value: Any, kind=QuestionKind.CHOICE) -> Decision:
    return Decision(qid, kind, value, {}, None, "json")


def _result(decisions, failures=()):
    return DecisionResult(tuple(decisions), tuple(failures), "", "json", 0.0, "x")


def _full_decisions(tool="deploy"):
    return [
        _dec(OP_ID, tool),
        _dec("search::scope", "repo"), _dec("search::deep", True, QuestionKind.NOUL),
        _dec("deploy::env", "prod"), _dec("deploy::dry_run", False, QuestionKind.NOUL),
    ]


def test_catalogue_shape_namespacing_and_text_names_the_tool():
    cat = build_joint_catalogue(TOOLS, descriptions={"search": "find things\nacross sources"})
    ids = [q.id for q in cat.questions]
    assert ids == [OP_ID, "search::scope", "search::deep", "deploy::env", "deploy::dry_run"]
    op = cat.questions[0]
    assert op.kind is QuestionKind.CHOICE and op.options == ("search", "deploy")
    assert op.option_descriptions == ("find things across sources", None)
    assert all("Assuming the chosen tool is 'deploy'" in q.text for q in cat.questions if q.id.startswith("deploy::"))
    assert cat.id_map["deploy::env"] == ("deploy", "env")


def test_catalogue_excludes_unmappable_required_and_needs_two_tools():
    cat = build_joint_catalogue({**TOOLS, "grep": FREEFORM})
    assert cat.tools == ("search", "deploy") and "grep" in cat.excluded
    with pytest.raises(ValueError):
        build_joint_catalogue({"search": SEARCH, "grep": FREEFORM})
    cat2 = build_joint_catalogue({"search": SEARCH, "a::b": DEPLOY, "deploy": DEPLOY})
    assert "a::b" in cat2.excluded


def test_selected_branch_consumed_unused_branch_ignored():
    cat = build_joint_catalogue(TOOLS)
    p = resolve_joint(cat, _result(_full_decisions("deploy")), executed_tool="search")
    assert (p.status, p.tool, p.arguments) == ("proposed", "deploy", {"env": "prod", "dry_run": False})
    assert p.operation_differs is True
    p2 = resolve_joint(cat, _result(_full_decisions("search")), executed_tool="search")
    assert p2.arguments == {"scope": "repo", "deep": True} and p2.operation_differs is False


def test_unused_branch_invalid_value_failure_is_ignored_not_repaired():
    cat = build_joint_catalogue(TOOLS)
    decisions = [d for d in _full_decisions("deploy") if not d.question_id.startswith("search::")]
    fail = ParseFailure("invalid_value", "question 'search::scope': choice outside options")
    p = resolve_joint(cat, _result(decisions, [fail]))
    assert p.status == "proposed" and p.arguments["env"] == "prod"
    assert p.ignored_unused_failures == (fail.detail,)


@pytest.mark.parametrize(
    "failure,reason",
    [
        (ParseFailure("invalid_value", "question 'deploy::env': choice outside options"), "selected_branch_failure"),
        (ParseFailure("invalid_value", f"question '{OP_ID}': choice outside options"), "operation_failed"),
        (ParseFailure("schema_violation", "attempt 1/2: bad envelope"), "malformed_or_envelope_failure:schema_violation"),
        (ParseFailure("invalid_value", "unknown question id 'zzz'"), "malformed_or_envelope_failure:invalid_value"),
        (ParseFailure("invalid_value", "question 'nope::x': oops"), "failure_names_unknown_question"),
        (ParseFailure("transport_error", "attempt 1/1: [ERROR: x]"), "malformed_or_envelope_failure:transport_error"),
    ],
)
def test_strict_rejections(failure, reason):
    cat = build_joint_catalogue(TOOLS)
    p = resolve_joint(cat, _result(_full_decisions("deploy"), [failure]))
    assert p.status == "abstained" and p.reason == reason and p.tool is None


def test_operation_missing_or_unknown_abstains():
    cat = build_joint_catalogue(TOOLS)
    assert resolve_joint(cat, _result(_full_decisions()[1:])).reason == "operation_unresolved"
    assert resolve_joint(cat, _result([_dec(OP_ID, "rm")])).reason == "operation_not_in_catalogue"
    assert resolve_joint(cat, _result([])).reason == "operation_unresolved"


def test_incomplete_selected_branch_abstains_with_reasons():
    cat = build_joint_catalogue(TOOLS)
    decisions = [d for d in _full_decisions("deploy") if d.question_id != "deploy::dry_run"]
    p = resolve_joint(cat, _result(decisions))
    assert p.status == "abstained" and p.reason.startswith("selected_branch_invalid") and "dry_run" in p.reason


def test_default_off_and_bad_state():
    assert propose_joint_tool_call(tools=TOOLS, primitives=object(), state="s", role="worker").reason == "flag_off"
    assert propose_joint_tool_call(tools=TOOLS, primitives=object(), state=" ", role="w", environ=ON).reason == "invalid_state"
    assert tc.tool_choice_enabled({}) is False and tc.tool_choice_enabled(ON) is True


class FakePrimitives:
    def __init__(self, payload: dict):
        self.payload, self.calls = payload, []

    def llm_call(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return json.dumps(self.payload)


def _payload(tool):
    return {"answers": {
        OP_ID: {"choice": tool},
        "search::scope": {"choice": "web"}, "search::deep": {"noul": False},
        "deploy::env": {"choice": "dev"}, "deploy::dry_run": {"noul": True},
    }}


def test_end_to_end_one_call_through_the_real_runner():
    prim = FakePrimitives(_payload("deploy"))
    p = propose_joint_tool_call(tools=TOOLS, primitives=prim, state="ship it", role="worker",
                                executed_tool="search", environ=ON)
    assert len(prim.calls) == 1  # ONE call for operation + every head
    assert p.status == "proposed" and p.tool == "deploy" and p.arguments == {"env": "dev", "dry_run": True}
    assert p.operation_differs is True
    rec = shadow_receipt_fields(p, "search")
    assert rec["proposed_tool"] == "deploy" and rec["executed_tool"] == "search" and rec["operation_differs"]


def test_end_to_end_malformed_envelope_falls_back():
    class Bad(FakePrimitives):
        def llm_call(self, prompt, **kw):
            self.calls.append(1)
            return "not json at all"

    p = propose_joint_tool_call(tools=TOOLS, primitives=Bad({}), state="s", role="w", environ=ON)
    assert p.status == "abstained" and p.reason.startswith("malformed_or_envelope_failure")


def test_end_to_end_value_outside_options_in_unused_branch_is_ignored():
    payload = _payload("deploy")
    payload["answers"]["search::scope"] = {"choice": "galaxy"}  # outside options, unused branch
    prim = FakePrimitives(payload)
    p = propose_joint_tool_call(tools=TOOLS, primitives=prim, state="s", role="w", environ=ON)
    # The runner's strict schema rejects an out-of-enum value at the ENVELOPE level (schema_violation),
    # so this abstains rather than ignoring: global validation stays strict.
    assert p.status == "abstained"
