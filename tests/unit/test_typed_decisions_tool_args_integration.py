"""Unit tests for the TD-4 tool-argument integration (fail-open, flag-gated).

Covers the integration contract in
``src/typed_decisions/tool_args_integration.py``: the flag gate, the
all-arguments-mappable gate, the exact assembled dict from canned primitives,
typed/transport/assembly failures, and the call-site wiring in
``src/repl_environment/context.py`` (typed path skipped when off, typed
arguments used when on). No model/server call: the primitives seam is a
canned-response fake, mirroring ``tests/unit/test_typed_decisions_tool_args.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from src.features import Features, reset_features, set_features
from src.typed_decisions import tool_args_integration as integration
from src.typed_decisions.native import (
    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
    REASON_NATIVE_UNKNOWN_CANDIDATE,
    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
)
from src.typed_decisions.tool_args import tool_schema_to_questions
from src.typed_decisions.tool_args_integration import (
    maybe_typed_arguments,
    registry_parameters_to_schema,
)
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    ParseFailure,
    Question,
    QuestionKind,
)

ROLE = "worker"
STATE = "unit-test state: deploy the service safely"

_FULLY_MAPPABLE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["fast", "safe"]},
        "dry_run": {"type": "boolean"},
    },
    "required": ["mode", "dry_run"],
}

# One unmappable optional argument -> the whole schema must decline, because
# the assembled dict replaces the model's argument object wholesale.
_PARTIAL: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["fast", "safe"]},
        "notes": {"type": "string"},
    },
    "required": ["mode"],
}

# The required argument itself is unmappable -> decline without a call.
_REQUIRED_UNMAPPABLE: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

# Valid decisions, invalid assembly: both array-enum members true, which the
# documented single-select approximation rejects.
_MULTI_SELECT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tags": {"type": "array", "items": {"type": "string", "enum": ["alpha", "beta"]}},
    },
    "required": ["tags"],
}

_REGISTRY_PARAMETERS: dict[str, Any] = {
    "mode": {"type": "string", "enum": ["fast", "safe"], "required": True},
    "dry_run": {"type": "boolean", "required": True},
}


@pytest.fixture
def enabled():
    set_features(Features(typed_decisions_tool_args=True))
    try:
        yield
    finally:
        reset_features()


@pytest.fixture
def disabled():
    set_features(Features(typed_decisions_tool_args=False))
    try:
        yield
    finally:
        reset_features()


class _FakePrimitives:
    """Canned-response stand-in for ``LLMPrimitives`` (last response repeats)."""

    def __init__(self, responses: str | Sequence[str] = "{}"):
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls: list[dict] = []

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def _questions(parameters: Mapping[str, Any], tool_name: str = "deploy") -> list[Question]:
    return tool_schema_to_questions(tool_name, parameters).questions


def _canned_response(questions: Sequence[Question], values: Mapping[str, object]) -> str:
    answers: dict[str, dict] = {}
    for question in questions:
        value = values[question.id]
        if question.kind is QuestionKind.CHOICE:
            labels = list(question.options)
            answers[question.id] = {
                "choice": value,
                "probabilities": {label: 1.0 / len(labels) for label in labels},
                "confidence": 1.0,
            }
        elif question.kind is QuestionKind.SCORE:
            labels = [str(level) for level in question.levels]
            answers[question.id] = {
                "score": value,
                "probabilities": {label: 1.0 / len(labels) for label in labels},
                "confidence": 1.0,
            }
        else:
            answers[question.id] = {
                "noul": value,
                "probabilities": {"true": 1.0, "false": 0.0}
                if value
                else {"true": 0.0, "false": 1.0},
                "confidence": 1.0,
            }
    return json.dumps({"answers": answers})


def _decision(question: Question, value: object, mode: str = "native") -> Decision:
    return Decision(
        question_id=question.id,
        kind=question.kind,
        value=value,
        probabilities={},
        confidence=1.0,
        mode=mode,
    )


def _result(decisions: Sequence[Decision], failures: Sequence[ParseFailure], mode: str):
    return DecisionResult(
        decisions=tuple(decisions),
        failures=tuple(failures),
        raw_text="",
        mode=mode,
        elapsed_ms=0.0,
        prompt_sha256="deadbeef",
    )


# ── 1. Flag gate ──────────────────────────────────────────────────────────


class TestFlagGate:
    def test_flag_off_returns_none_without_calling_the_model(self, disabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_flag_off_returns_none_even_when_everything_else_is_ready(self, disabled):
        mapping = _questions(_FULLY_MAPPABLE)
        primitives = _FakePrimitives(_canned_response(mapping, {"mode": "safe", "dry_run": True}))

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []


# ── 2. Engagement gate ────────────────────────────────────────────────────


class TestEngagementGate:
    def test_partially_mappable_schema_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_PARTIAL,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_required_unmappable_argument_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_REQUIRED_UNMAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_non_object_schema_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters={"type": "array", "items": {"type": "string"}},
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_missing_state_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state="   ",
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_missing_primitives_declines(self, enabled):
        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=None,
            role=ROLE,
        )

        assert result is None

    def test_unusable_primitives_decline(self, enabled):
        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=object(),
            role=ROLE,
        )

        assert result is None


# ── 3. Happy path through the real runner ─────────────────────────────────


class TestHappyPath:
    def test_fully_mappable_schema_returns_the_validated_arguments(self, enabled):
        mapping = tool_schema_to_questions("deploy", _FULLY_MAPPABLE)
        primitives = _FakePrimitives(
            _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
        )

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result == {"mode": "safe", "dry_run": True}
        # One JSON call: native mode could not resolve a tokenizer from the
        # fake primitives (no base URL), so it failed closed with zero model
        # calls and the JSON rerun produced the answer.
        assert len(primitives.calls) == 1
        assert "grammar" not in primitives.calls[0]

    def test_json_preferred_mode_skips_the_native_attempt(self, enabled):
        mapping = tool_schema_to_questions("deploy", _FULLY_MAPPABLE)
        primitives = _FakePrimitives(
            _canned_response(mapping.questions, {"mode": "fast", "dry_run": False})
        )

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
            mode="json",
        )

        assert result == {"mode": "fast", "dry_run": False}
        assert len(primitives.calls) == 1


# ── 4. Failure contract ───────────────────────────────────────────────────


class TestFailureContract:
    def test_typed_failure_returns_none(self, enabled):
        primitives = _FakePrimitives("this is not a JSON object")

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls

    def test_transport_failure_returns_none(self, enabled):
        primitives = _FakePrimitives("[ERROR: backend is down]")

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert len(primitives.calls) == 1

    def test_assembly_error_returns_none(self, enabled):
        mapping = tool_schema_to_questions("deploy", _MULTI_SELECT)
        primitives = _FakePrimitives(
            _canned_response(
                mapping.questions,
                {"tags__alpha": True, "tags__beta": True},
            )
        )

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_MULTI_SELECT,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None

    def test_unknown_preferred_mode_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
            mode="batch",
        )

        assert result is None
        assert primitives.calls == []


# ── 5. Mode preference (native first, JSON on eligibility failure) ────────


class TestModePreference:
    def test_complete_native_pass_wins_without_a_json_rerun(self, monkeypatch, enabled):
        mapping = tool_schema_to_questions("deploy", _FULLY_MAPPABLE)
        values = {"mode": "safe", "dry_run": True}
        native_result = _result(
            [_decision(question, values[question.id]) for question in mapping.questions],
            [],
            "native",
        )
        calls: list[str] = []

        def fake_run(primitives, *, state, questions, role, mode="json", **kwargs):
            calls.append(mode)
            return native_result

        monkeypatch.setattr(integration, "run_typed_decisions", fake_run)

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=_FakePrimitives(),
            role=ROLE,
        )

        assert result == {"mode": "safe", "dry_run": True}
        assert calls == ["native"]

    @pytest.mark.parametrize(
        "reason",
        [REASON_NATIVE_UNSUPPORTED_CANDIDATES, REASON_NATIVE_TOKENIZER_UNAVAILABLE],
    )
    def test_native_eligibility_failure_reruns_in_json(self, monkeypatch, enabled, reason):
        values = {"mode": "fast", "dry_run": False}
        calls: list[str] = []

        def fake_run(primitives, *, state, questions, role, mode="json", **kwargs):
            calls.append(mode)
            if mode == "native":
                return _result([], [ParseFailure(reason, "not eligible")], "native")
            return _result(
                [_decision(question, values[question.id]) for question in questions],
                [],
                "json",
            )

        monkeypatch.setattr(integration, "run_typed_decisions", fake_run)

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=_FakePrimitives(),
            role=ROLE,
        )

        assert result == {"mode": "fast", "dry_run": False}
        assert calls == ["native", "json"]

    def test_other_native_failure_declines_without_a_json_rerun(self, monkeypatch, enabled):
        calls: list[str] = []

        def fake_run(primitives, *, state, questions, role, mode="json", **kwargs):
            calls.append(mode)
            return _result([], [ParseFailure(REASON_NATIVE_UNKNOWN_CANDIDATE, "no row")], mode)

        monkeypatch.setattr(integration, "run_typed_decisions", fake_run)

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=_FakePrimitives(),
            role=ROLE,
        )

        assert result is None
        assert calls == ["native"]


# ── 6. Registry-shape projection ──────────────────────────────────────────


class TestRegistryProjection:
    def test_projection_carries_closed_sets_and_required(self):
        schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)

        assert schema["type"] == "object"
        assert schema["required"] == ["mode", "dry_run"]
        assert schema["properties"]["mode"]["enum"] == ["fast", "safe"]
        assert schema["properties"]["dry_run"]["type"] == "boolean"

    def test_projection_carries_bounds_and_items(self):
        schema = registry_parameters_to_schema(
            {
                "retries": {"type": "integer", "minimum": 0, "maximum": 3},
                "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
            }
        )

        assert schema["properties"]["retries"]["minimum"] == 0
        assert schema["properties"]["tags"]["items"]["enum"] == ["a", "b"]
        assert "required" not in schema

    def test_projection_declines_malformed_maps(self):
        assert registry_parameters_to_schema(None) == {}
        assert registry_parameters_to_schema({}) == {}
        assert registry_parameters_to_schema({"bad": "not-a-spec"}) == {}

    def test_projection_defaults_missing_type_to_string(self):
        schema = registry_parameters_to_schema({"query": {"description": "free text"}})

        assert schema["properties"]["query"]["type"] == "string"


# ── 7. Call-site wiring (src/repl_environment/context.py) ─────────────────


class _Tool:
    def __init__(self, name: str, parameters: Mapping[str, Any]):
        self.name = name
        self.parameters = parameters

    def validate_args(self, args):
        return []


class _FakeRegistry:
    def __init__(self, tool: _Tool):
        self._tools = {tool.name: tool}
        self.invocations: list[dict] = []
        self.allowed = True

    def can_use_tool(self, role, tool_name, context=None):
        return self.allowed and tool_name in self._tools

    def invoke(
        self,
        tool_name,
        role,
        *,
        caller_type=None,
        chain_id=None,
        chain_index=0,
        context=None,
        **kwargs,
    ):
        if not self.can_use_tool(role, tool_name, context=context):
            raise PermissionError("tool denied")
        self.invocations.append({"tool_name": tool_name, "role": role, "kwargs": dict(kwargs)})
        return "tool-result"


class _Repl:
    def __init__(self, *, registry, primitives, context, role="worker"):
        self.tool_registry = registry
        self.llm_primitives = primitives
        self.context = context
        self.role = role
        self._active_tool_chain_id = None
        self._active_tool_chain_index = 0


def _make_repl(primitives, registry) -> Any:
    from src.repl_environment.context import _ContextMixin

    class _TestRepl(_ContextMixin, _Repl):
        pass

    return _TestRepl(registry=registry, primitives=primitives, context=STATE)


def test_call_site_skips_typed_path_when_off(monkeypatch, disabled):
    """Flag off: the typed-args path is never entered (no dead work per dispatch)."""
    recorded: list[dict] = []
    helper_calls: list[str] = []

    def _recording(**kwargs):
        recorded.append(kwargs)
        return None

    monkeypatch.setattr(integration, "maybe_typed_arguments", _recording)

    primitives = _FakePrimitives("{}")
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(primitives, registry)
    original_helper = repl._typed_tool_arguments

    def _helper_spy(tool_name):
        helper_calls.append(tool_name)
        return original_helper(tool_name)

    monkeypatch.setattr(repl, "_typed_tool_arguments", _helper_spy)

    result = repl._dispatch_tool("deploy", mode="fast", dry_run=False)

    assert result == "tool-result"
    assert helper_calls == []
    assert recorded == []
    assert primitives.calls == []
    assert getattr(repl, "_last_decision_receipt", None) is None
    assert registry.invocations == [
        {"tool_name": "deploy", "role": "worker", "kwargs": {"mode": "fast", "dry_run": False}}
    ]


def _shadow_receipt(tmp_path):
    from src.typed_decisions.shadow import drain_shadow

    drain_shadow()
    path = tmp_path / "typed_decisions" / "decision_receipts.jsonl"
    return json.loads(path.read_text().strip())


def _set_receipt_path(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import src.config

    monkeypatch.setattr(
        src.config,
        "get_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )


def test_call_site_shadow_records_proposal_but_executes_original(enabled, monkeypatch, tmp_path):
    _set_receipt_path(monkeypatch, tmp_path)
    schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)
    mapping = tool_schema_to_questions("deploy", schema)
    primitives = _FakePrimitives(
        _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
    )
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(primitives, registry)

    assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
    assert registry.invocations[0]["kwargs"] == {"mode": "fast", "dry_run": False}
    receipt = _shadow_receipt(tmp_path)
    assert receipt["schema_version"] == "decision_receipt.v1"
    assert receipt["validation_result"] == "accepted"
    assert receipt["selector_status"] == "proposed"
    assert receipt["proposed_arguments"] == {"mode": "safe", "dry_run": True}
    assert receipt["executed_arguments"] == {"mode": "fast", "dry_run": False}
    assert receipt["observed_downstream_outcome"]["status"] == "returned"
    assert receipt["shadow_mode"] == "json"


def test_call_site_rechecks_task_revision_without_selector(enabled, monkeypatch, tmp_path):
    _set_receipt_path(monkeypatch, tmp_path)
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    primitives = _FakePrimitives()
    repl = _make_repl(primitives, registry)
    original = repl._tool_decision_snapshot
    calls = 0

    def changed_snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:
            repl.context = "changed after preparation"
        return original()

    monkeypatch.setattr(repl, "_tool_decision_snapshot", changed_snapshot)
    assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
    receipt = _shadow_receipt(tmp_path)
    assert receipt["validation_result"] == "stale"
    assert receipt["selector_status"] == "abstained"
    assert primitives.calls == []
    assert registry.invocations[0]["kwargs"] == {"mode": "fast", "dry_run": False}


def test_call_site_rechecks_catalog_and_skips_shadow(enabled, monkeypatch, tmp_path):
    _set_receipt_path(monkeypatch, tmp_path)
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS.copy()))
    primitives = _FakePrimitives()
    repl = _make_repl(primitives, registry)
    original = repl._tool_decision_snapshot
    calls = 0

    def changed_snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:
            registry._tools["deploy"].parameters = {"changed": {"type": "string"}}
        return original()

    monkeypatch.setattr(repl, "_tool_decision_snapshot", changed_snapshot)
    assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
    receipt = _shadow_receipt(tmp_path)
    assert receipt["validation_result"] == "stale"
    assert receipt["selector_status"] == "abstained"
    assert primitives.calls == []


def test_call_site_revoked_permission_cannot_be_granted_by_shadow(enabled, monkeypatch, tmp_path):
    _set_receipt_path(monkeypatch, tmp_path)
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(_FakePrimitives(), registry)
    original = registry.can_use_tool
    calls = 0

    def revoke_after_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls >= 2:
            registry.allowed = False
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "can_use_tool", revoke_after_prepare)
    with pytest.raises(PermissionError, match="tool denied"):
        repl._dispatch_tool("deploy", mode="fast", dry_run=False)
    receipt = _shadow_receipt(tmp_path)
    assert receipt["validation_result"] == "unauthorized"
    assert receipt["authorization_result"] == "denied"
    assert receipt["selector_status"] == "abstained"
    assert receipt["observed_downstream_outcome"]["detail"] == "PermissionError"


def test_shadow_preserves_unmappable_optional_fields_on_original_td4_cohort():
    from src.typed_decisions.tool_args_pilot import build_cases

    for case in build_cases():
        mapping = tool_schema_to_questions(case.tool, case.parameters)
        answers = {}
        for question in mapping.questions:
            if "__" in question.id:
                name, label = question.id.split("__", 1)
                answers[question.id] = label in case.expected[name]
            else:
                answers[question.id] = case.expected[question.id]
        optional = next(iter(set(case.parameters["properties"]) - set(case.expected)))
        original = {**case.expected, optional: f"original-{case.case_id}"}
        primitives = _FakePrimitives(_canned_response(mapping.questions, answers))
        proposal = integration.propose_shadow_typed_arguments(
            tool_name=case.tool,
            full_schema=case.parameters,
            original_kwargs=original,
            primitives=primitives,
            state=case.state,
            role=ROLE,
            mode="json",
        )
        assert proposal.status == "proposed", (case.case_id, proposal.reason)
        assert proposal.proposed == original
        assert proposal.proposed[optional] == original[optional]
        assert len(primitives.calls) == 1


def test_shadow_required_unmappable_declines_without_model_call():
    primitives = _FakePrimitives()
    result = integration.propose_shadow_typed_arguments(
        tool_name="search",
        full_schema=_REQUIRED_UNMAPPABLE,
        original_kwargs={"query": "exact"},
        primitives=primitives,
        state=STATE,
        role=ROLE,
    )
    assert result.status == "abstained"
    assert result.reason == "required_argument_unmappable"
    assert primitives.calls == []


def test_shadow_full_validation_rejects_extra_and_constraint_violations():
    schema = _FULLY_MAPPABLE
    mapping = tool_schema_to_questions("deploy", schema)
    response = _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
    result = integration.propose_shadow_typed_arguments(
        tool_name="deploy",
        full_schema=schema,
        original_kwargs={"mode": "fast", "dry_run": False, "extra": "unregistered"},
        primitives=_FakePrimitives(response),
        state=STATE,
        role=ROLE,
    )
    assert result.status == "abstained"
    assert result.proposed is None
    assert "ValidationError" in result.reason

    constrained = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 2, "maximum": 4},
        },
        "required": ["count"],
        "additionalProperties": False,
    }
    questions = tool_schema_to_questions("count", constrained).questions
    result = integration.propose_shadow_typed_arguments(
        tool_name="count",
        full_schema=constrained,
        original_kwargs={"count": 3},
        primitives=_FakePrimitives(_canned_response(questions, {"count": 1})),
        state=STATE,
        role=ROLE,
    )
    assert result.status == "abstained"
    assert result.proposed is None


def test_shadow_refuses_unknown_registry_constraint_including_nested():
    assert (
        integration.registry_parameters_to_full_schema(
            {"name": {"type": "string", "x-secret-rule": True}}
        )
        is None
    )
    assert (
        integration.registry_parameters_to_full_schema(
            {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "x-secret-rule": True,
                    },
                }
            }
        )
        is None
    )
    assert integration.registry_parameters_to_full_schema(
        {"mode": {"type": "string", "enum": ["fast", "safe"], "required": True}}
    )["properties"]["mode"]["enum"] == ["fast", "safe"]


def test_dispatch_never_waits_for_blocked_shadow(enabled, monkeypatch, tmp_path):
    import threading

    _set_receipt_path(monkeypatch, tmp_path)
    schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)
    mapping = tool_schema_to_questions("deploy", schema)
    started, release = threading.Event(), threading.Event()

    class BlockingPrimitives(_FakePrimitives):
        def llm_call(self, prompt, **kwargs):
            started.set()
            assert release.wait(3)
            return super().llm_call(prompt, **kwargs)

    primitives = BlockingPrimitives(
        _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
    )
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(primitives, registry)
    try:
        assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
        assert registry.invocations[0]["kwargs"] == {"mode": "fast", "dry_run": False}
        assert started.wait(2)
    finally:
        release.set()
    assert _shadow_receipt(tmp_path)["selector_status"] == "proposed"


def test_shadow_worker_observes_cancel_and_keeps_cost_separate(enabled, monkeypatch, tmp_path):
    _set_receipt_path(monkeypatch, tmp_path)

    class CountedPrimitives(_FakePrimitives):
        def __init__(self, response):
            super().__init__(response)
            self.total_calls = 0
            self.cancelled = False

        def llm_call(self, prompt, **kwargs):
            self.total_calls += 1
            return super().llm_call(prompt, **kwargs)

        def get_request_cancel_check(self):
            return lambda: self.cancelled

    schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)
    mapping = tool_schema_to_questions("deploy", schema)
    primitives = CountedPrimitives(
        _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
    )
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(primitives, registry)
    assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
    assert _shadow_receipt(tmp_path)["selector_status"] == "proposed"
    assert primitives.total_calls == 0  # model work ran on an isolated primitives copy

    other_path = tmp_path / "cancelled"
    _set_receipt_path(monkeypatch, other_path)
    primitives.cancelled = True
    assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
    receipt = _shadow_receipt(other_path)
    assert receipt["selector_reason"] == "request_cancelled"
    assert receipt["selector_calls"] == []
    assert primitives.total_calls == 0


def test_bounded_shadow_executor_saturates_without_calling_fifth():
    import threading

    from src.typed_decisions.shadow import (
        _reset_for_tests,
        drain_shadow,
        submit_bounded_shadow_work,
    )

    _reset_for_tests()
    release = threading.Event()
    calls = []

    def block(i):
        calls.append(i)
        release.wait(3)

    try:
        assert all(submit_bounded_shadow_work(block, i) for i in range(4))
        assert not submit_bounded_shadow_work(block, 4)
    finally:
        release.set()
        drain_shadow()
    assert sorted(calls) == list(range(4))


def test_shadow_failure_and_malformed_response_never_change_dispatch(
    enabled, monkeypatch, tmp_path
):
    _set_receipt_path(monkeypatch, tmp_path)
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    primitives = _FakePrimitives("not-json")
    repl = _make_repl(primitives, registry)
    assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
    receipt = _shadow_receipt(tmp_path)
    assert receipt["selector_status"] == "abstained"
    assert receipt["proposed_arguments"] is None
    assert receipt["executed_arguments"] == {"mode": "fast", "dry_run": False}
    assert registry.invocations[0]["kwargs"] == receipt["executed_arguments"]

    class FailingPrimitives(_FakePrimitives):
        def llm_call(self, prompt, **kwargs):
            raise RuntimeError("selector unavailable")

    other = tmp_path / "failure"
    _set_receipt_path(monkeypatch, other)
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    assert (
        _make_repl(FailingPrimitives(), registry)._dispatch_tool(
            "deploy", mode="fast", dry_run=False
        )
        == "tool-result"
    )
    receipt = _shadow_receipt(other)
    assert receipt["selector_status"] == "abstained"
    assert "selector unavailable" in receipt["selector_reason"]
    assert registry.invocations[0]["kwargs"] == {"mode": "fast", "dry_run": False}


def test_shadow_carries_request_context_across_worker_delay(enabled, monkeypatch, tmp_path):
    import contextvars
    import threading

    _set_receipt_path(monkeypatch, tmp_path)
    marker = contextvars.ContextVar("td29_request_marker", default="missing")
    started, release = threading.Event(), threading.Event()
    seen = []

    def delayed_proposal(**kwargs):
        started.set()
        assert release.wait(3)
        seen.append((marker.get(), kwargs["state"], kwargs["original_kwargs"]))
        return integration.ShadowArgumentProposal("abstained", None, "test")

    monkeypatch.setattr(integration, "propose_shadow_typed_arguments", delayed_proposal)
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(_FakePrimitives(), registry)
    token = marker.set("request-A")
    try:
        assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
        assert started.wait(2)
    finally:
        marker.reset(token)
        release.set()
    receipt = _shadow_receipt(tmp_path)
    assert seen == [("request-A", STATE, {"mode": "fast", "dry_run": False})]
    assert receipt["selector_status"] == "abstained"


def test_shadow_rejects_foreground_instance_llm_wrapper(enabled, monkeypatch, tmp_path):
    _set_receipt_path(monkeypatch, tmp_path)
    primitives = _FakePrimitives()
    foreground_calls = []
    primitives.llm_call = lambda *args, **kwargs: foreground_calls.append((args, kwargs))
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    assert _make_repl(primitives, registry)._dispatch_tool(
        "deploy", mode="fast", dry_run=False
    ) == "tool-result"
    receipt = _shadow_receipt(tmp_path)
    assert receipt["selector_reason"] == "instance_llm_call_override"
    assert receipt["selector_calls"] == []
    assert foreground_calls == []
    assert registry.invocations[0]["kwargs"] == {"mode": "fast", "dry_run": False}


def test_concurrent_foreground_and_shadow_isolate_meta_and_cost(enabled, monkeypatch, tmp_path):
    import contextvars
    import threading

    _set_receipt_path(monkeypatch, tmp_path)
    schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)
    mapping = tool_schema_to_questions("deploy", schema)
    response = _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
    started, release = threading.Event(), threading.Event()

    class ConcurrentPrimitives(_FakePrimitives):
        def __init__(self):
            super().__init__(response)
            self.total_calls = 0
            self.call_log = []
            self._request_trace_keys = {}
            self._meta = contextvars.ContextVar("td29_concurrent_meta", default=None)
            self.bound_request_id = "foreground"

        def bind_request_usage(self, request_id=None):
            self.bound_request_id = request_id

        def get_last_inference_meta(self):
            return self._meta.get()

        def llm_call(self, prompt, **kwargs):
            self.total_calls += 1
            self.call_log.append({"prompt": prompt})
            if self.bound_request_id != "foreground":
                started.set()
                assert release.wait(3)
                self._meta.set({"prompt_n": 11, "cache_n": 0, "prompt_tokens": 11})
            else:
                self._meta.set({"prompt_n": 99, "cache_n": 0, "prompt_tokens": 99})
            return response

    primitives = ConcurrentPrimitives()
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(primitives, registry)
    try:
        assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
        assert started.wait(2)
        primitives.llm_call("foreground")
        assert primitives.get_last_inference_meta()["prompt_n"] == 99
    finally:
        release.set()
    receipt = _shadow_receipt(tmp_path)
    assert receipt["selector_status"] == "proposed"
    assert receipt["selector_calls"][0]["prompt_n"] == 11
    assert primitives.total_calls == 1
    assert len(primitives.call_log) == 1
    assert primitives.call_log[0]["prompt"] == "foreground"
    assert primitives.bound_request_id == "foreground"
    assert primitives.get_last_inference_meta()["prompt_n"] == 99


def test_queued_shadow_freezes_request_trace_keys(enabled, monkeypatch, tmp_path):
    _set_receipt_path(monkeypatch, tmp_path)
    pending = []
    seen = []
    primitives = _FakePrimitives()
    primitives._request_trace_keys = {"x_session_id": "request-A"}

    def queue_only(fn, **kwargs):
        pending.append((fn, kwargs))
        return True

    def observe(**kwargs):
        seen.append(kwargs["primitives"]._request_trace_keys)
        return integration.ShadowArgumentProposal("abstained", None, "test")

    monkeypatch.setattr(integration, "submit_bounded_shadow_work", queue_only)
    monkeypatch.setattr(integration, "propose_shadow_typed_arguments", observe)
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    assert _make_repl(primitives, registry)._dispatch_tool(
        "deploy", mode="fast", dry_run=False
    ) == "tool-result"
    assert len(pending) == 1
    primitives._request_trace_keys = {"x_session_id": "request-B"}
    fn, kwargs = pending.pop()
    fn(**kwargs)
    assert seen == [{"x_session_id": "request-A"}]
    assert _shadow_receipt(tmp_path)["selector_status"] == "abstained"


def test_real_primitives_usage_ledger_isolated_during_shadow(enabled, monkeypatch, tmp_path):
    import threading

    from src.llm_primitives.primitives import LLMPrimitives

    primitives = LLMPrimitives(mock_mode=True)
    _set_receipt_path(monkeypatch, tmp_path)
    monkeypatch.setattr("src.llm_primitives.usage_log.write_usage_event", lambda *a, **k: None)
    schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)
    mapping = tool_schema_to_questions("deploy", schema)
    response = _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
    started, release = threading.Event(), threading.Event()
    ledgers = []

    def fake_llm_call(self, prompt, **kwargs):
        self.total_calls += 1
        ledger = self._get_usage_ledger()
        ticket = ledger.begin("worker")
        if ledger.request_id != "foreground":
            ledgers.append(ledger)
            started.set()
            assert release.wait(3)
            count = 11
        else:
            count = 99
        ledger.finish(ticket, prompt_tokens=count, completion_tokens=1,
                      provenance="estimated", source="mock:unit", finish_reason="stop")
        self._last_inference_meta_ctx.set(
            {"prompt_n": count, "cache_n": 0, "prompt_tokens": count}
        )
        return response

    monkeypatch.setattr(LLMPrimitives, "llm_call", fake_llm_call)
    primitives.bind_request_usage("foreground")
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    try:
        assert _make_repl(primitives, registry)._dispatch_tool(
            "deploy", mode="fast", dry_run=False
        ) == "tool-result"
        assert started.wait(2)
        primitives.llm_call("foreground")
    finally:
        release.set()
    receipt = _shadow_receipt(tmp_path)
    assert receipt["selector_calls"][0]["prompt_n"] == 11
    assert len(ledgers) == 1
    assert ledgers[0].summary()["prompt_tokens"] == 11
    assert primitives.get_request_usage()["prompt_tokens"] == 99
    assert primitives.total_calls == 1
    assert primitives.get_last_inference_meta()["prompt_n"] == 99
