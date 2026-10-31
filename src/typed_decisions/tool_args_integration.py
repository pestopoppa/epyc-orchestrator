"""TD-4 integration: closed-set tool arguments on the live REPL tool dispatch.

Measured basis
--------------
The TD-4 pilot (``tool-args-pilot-worker.json``, 2026-09-17: 18 deterministic
cases over 3 tool schemas) measured closed-set typed argument selection at
**18/18 exact-match with 0 failures** against **6/18 exact-match with 12
parse/schema failures** for the free-form JSON arm. This module wires the
winning arm into the tool path behind a default-off flag.

Gate and fail-open contract
---------------------------
``maybe_typed_arguments`` is called on every REPL tool dispatch but engages
ONLY when all of these hold:

* the ``typed_decisions_tool_args`` feature flag (default off) is on;
* the tool schema maps ENTIRELY to typed questions. The strict reading of
  "maps entirely" is used: ``tool_schema_to_questions`` must return at least
  one question and NO ``skipped`` entry. The parenthetical minimum ("every
  required argument mappable") is subsumed; a required-only gate was rejected
  because the assembled dict REPLACES the model's whole argument object, so
  any skipped argument would be silently dropped;
* a non-empty state string and primitives exposing an ``llm_call`` seam are
  available.

Any failure after engagement -- an unknown preferred mode, a transport error,
an unanswered question, a result whose ``failures`` tuple is non-empty (the
conservative reading: a corrective retry that recovered still leaves its first
attempt visible), or an assembly error -- returns ``None``. The caller then
proceeds with the model-provided arguments exactly as before. Nothing raises
into the tool path; every decline is logged at debug level.

Mode preference
---------------
``mode="native"`` (the default) attempts native candidate scoring first: it
is the measured speed arm (id_only cue 11.98x vs JSON at 15/16 agreement,
``bench-cue-sweep-worker.json``, n=1; the 2026-09-18 n=4 re-measurement gives
9.60x, contested pending TD-1d.0) and it fails a question closed when a label
is not natively eligible or the tokenizer is unavailable. Eligibility is a
runtime property of the tokenizer, so instead of pre-computing it this module
reruns the catalogue in ``"json"`` mode when the native pass reports a
``native_unsupported_candidates`` / ``native_tokenizer_unavailable`` failure.
An unresolvable tokenizer makes native fail with zero model calls, so that
fallback is usually cheap; a mixed catalogue can pay one native call before
the JSON rerun. Any OTHER native failure (transport, unknown candidate) makes
the pass unusable and declines. ``mode="json"`` runs the JSON arm directly;
unknown modes decline without a call.

Call-site choice
----------------
Wired in ``src/repl_environment/context.py::_dispatch_tool``, the single
chokepoint every ``CALL(...)`` / ``TOOL(...)`` invocation (and every
translated OpenAI-format tool call) passes through before
``ToolRegistry.invoke``. It is the narrowest point where the tool
declaration, the request context (``REPLEnvironment.context`` is the state
string), the primitives seam and the role all coexist, and the chosen
arguments still pass through the existing ``Tool.validate_args`` validation
after substitution. ``ToolRegistry.invoke`` itself has the declaration but no
state or primitives, and the /v1 client-tool path
(``openai_compat._run_client_tool_completion``) has a full JSON schema but no
argument validation and tools there are executed by the CLIENT, not this
process; the internal dispatch is the actual tool execution path this change
targets. The registry declares parameters in its own per-argument shape
(``Tool.parameters``), so the call site projects it through
``registry_parameters_to_schema`` first. A programmatic registration that
declares ``enum`` / ``minimum`` / ``maximum`` / ``items`` therefore engages;
the plain YAML tools (type/description/required only) decline by
construction.

Replacement semantics
---------------------
When a dict is returned it is used as the tool's arguments in place of the
model-provided kwargs. Under the strict gate above that dict carries every
declared argument, so no declared argument is lost. An unknown argument the
model invented is dropped rather than rejected -- an accepted, documented
consequence of replacement, and one more reason the flag ships off.
"""

from __future__ import annotations

import logging
import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from src.features import features
from src.typed_decisions.native import (
    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
)
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.call_recorder import record_llm_calls
from src.typed_decisions.prepared_action import append_receipt, make_receipt
from src.typed_decisions.shadow import submit_bounded_shadow_work
from src.typed_decisions.prepared_action import PreparedAction, ValidationResult
from src.typed_decisions.tool_args import (
    ToolArgumentError,
    ToolQuestionMapping,
    assemble_arguments,
    tool_schema_to_questions,
)
from src.typed_decisions.types import DecisionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowArgumentProposal:
    status: str
    proposed: dict[str, Any] | None
    reason: str | None
    calls: tuple[dict[str, Any], ...] = ()


class _ShadowInstanceOverride(Exception):
    """Instance llm_call wrapper may close over foreground state."""


_SHADOW_SCHEMA_KEYS = frozenset(
    {
        "type",
        "description",
        "title",
        "default",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "items",
        "prefixItems",
        "contains",
        "minContains",
        "maxContains",
        "properties",
        "patternProperties",
        "additionalProperties",
        "required",
        "minProperties",
        "maxProperties",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)


def _recognized_schema(spec: Any) -> bool:
    if isinstance(spec, bool):
        return True
    if not isinstance(spec, Mapping) or set(spec) - _SHADOW_SCHEMA_KEYS:
        return False
    for key in ("items", "contains", "additionalProperties", "not", "if", "then", "else"):
        if key in spec and not _recognized_schema(spec[key]):
            return False
    for key in ("properties", "patternProperties"):
        if key in spec and (
            not isinstance(spec[key], Mapping)
            or any(not _recognized_schema(child) for child in spec[key].values())
        ):
            return False
    for key in ("prefixItems", "allOf", "anyOf", "oneOf"):
        if key in spec and (
            not isinstance(spec[key], list)
            or any(not _recognized_schema(child) for child in spec[key])
        ):
            return False
    return True


def registry_parameters_to_full_schema(
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a lossless validation schema from Tool.parameters or abstain.

    The old registry projection deliberately drops constraints.  Shadow
    proposals cannot use it as a complete-action validity check.
    """
    if not isinstance(parameters, Mapping) or not parameters:
        return None
    properties = {}
    required = []
    for name, spec in parameters.items():
        if not isinstance(name, str) or not name or not isinstance(spec, Mapping):
            return None
        if not _recognized_schema(spec):
            return None
        is_required = spec.get("required", False)
        if not isinstance(is_required, bool):
            return None
        if is_required:
            required.append(name)
        properties[name] = {key: value for key, value in spec.items() if key != "required"}
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "required": required,
    }
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        return None
    return schema


def propose_shadow_typed_arguments(
    *,
    tool_name: str,
    full_schema: Mapping[str, Any] | None,
    original_kwargs: Mapping[str, Any],
    primitives: Any,
    role: str,
    state: str,
    mode: str = "json",
    tool: Any = None,
) -> ShadowArgumentProposal:
    """Observe a typed proposal; never return arguments for foreground dispatch."""
    if mode not in {"json", "native"} or not isinstance(full_schema, Mapping):
        return ShadowArgumentProposal("abstained", None, "complete_schema_unavailable")
    if not isinstance(original_kwargs, Mapping) or not isinstance(state, str) or not state.strip():
        return ShadowArgumentProposal("abstained", None, "invalid_original_or_state")
    try:
        schema = dict(full_schema)
        Draft202012Validator.check_schema(schema)
        mapping = tool_schema_to_questions(tool_name, schema)
        required = set(schema.get("required", ()))
        properties = schema.get("properties", {})
        if any(name not in properties for name in required) or any(
            skipped.startswith(f"{name}: ") for name in required for skipped in mapping.skipped
        ):
            return ShadowArgumentProposal("abstained", None, "required_argument_unmappable")
        if not mapping.questions:
            return ShadowArgumentProposal("abstained", None, "no_mappable_arguments")
        with record_llm_calls(primitives) as call_log:
            result = run_typed_decisions(
                primitives,
                state=state,
                questions=mapping.questions,
                role=role,
                mode=mode,
            )
        calls = tuple(dict(call) for call in call_log.calls)
        if result.failures:
            return ShadowArgumentProposal("abstained", None, "typed_failure", calls)
        selected = assemble_arguments(mapping.questions, result.decisions, schema)
        proposal = dict(original_kwargs)
        proposal.update(selected)
        Draft202012Validator(schema).validate(proposal)
        if tool is not None:
            errors = tool.validate_args(proposal)
            if errors:
                return ShadowArgumentProposal(
                    "abstained", None, "registry_validation_failed", calls
                )
        return ShadowArgumentProposal("proposed", proposal, None, calls)
    except (ToolArgumentError, SchemaError, ValidationError, TypeError, ValueError) as exc:
        return ShadowArgumentProposal("abstained", None, f"{type(exc).__name__}: {exc}")


def submit_tool_argument_shadow(
    *,
    primitives: Any,
    tool_name: str,
    tool: Any,
    original_kwargs: Mapping[str, Any],
    state: str,
    role: str,
    prepared: PreparedAction,
    validation: ValidationResult,
    outcome: Mapping[str, str],
    receipt_path: Path,
    mode: str = "json",
) -> bool:
    """Queue the Q4 observer; never wait for it or change executed arguments."""
    full_schema = registry_parameters_to_full_schema(getattr(tool, "parameters", None))
    tool_snapshot = copy.copy(tool)
    tool_snapshot.parameters = copy.deepcopy(getattr(tool, "parameters", {}))
    return submit_bounded_shadow_work(
        _run_tool_argument_shadow,
        primitives=primitives,
        tool_name=tool_name,
        tool=tool_snapshot,
        original_kwargs=copy.deepcopy(dict(original_kwargs)),
        full_schema=copy.deepcopy(full_schema),
        state=str(state),
        role=str(role),
        prepared=prepared,
        validation=validation,
        outcome=dict(outcome),
        request_trace_keys=copy.deepcopy(getattr(primitives, "_request_trace_keys", {})),
        receipt_path=Path(receipt_path),
        mode=mode,
    )


def _run_tool_argument_shadow(
    *,
    primitives: Any,
    tool_name: str,
    tool: Any,
    original_kwargs: Mapping[str, Any],
    full_schema: Mapping[str, Any] | None,
    state: str,
    role: str,
    prepared: PreparedAction,
    validation: ValidationResult,
    outcome: Mapping[str, str],
    request_trace_keys: Mapping[str, Any],
    receipt_path: Path,
    mode: str,
) -> None:
    """Run after foreground completion with isolated usage and one receipt."""
    started = time.perf_counter()
    proposal = ShadowArgumentProposal("abstained", None, "shadow_not_run")
    try:
        # A shallow clone retains the same backend/slot but detaches mutable
        # foreground counters, call log and bound request ledger. ContextVars
        # were copied at submission for cancellation/deadline/placement.
        shadow_primitives = copy.copy(primitives)
        # record_llm_calls and some harnesses wrap llm_call on the instance.
        # A shallow copy would retain that closure bound to the foreground.
        # Fail closed instead of logging or billing shadow work to it.
        if "llm_call" in vars(primitives):
            proposal = ShadowArgumentProposal("abstained", None, "instance_llm_call_override")
            raise _ShadowInstanceOverride
        shadow_primitives.call_log = []
        shadow_primitives._request_trace_keys = dict(request_trace_keys)
        bind_usage = getattr(shadow_primitives, "bind_request_usage", None)
        if callable(bind_usage):
            bind_usage(request_id=f"td29-shadow-{time.time_ns()}")
        if validation.status == "accepted":
            cancel_getter = getattr(shadow_primitives, "get_request_cancel_check", None)
            cancel_check = cancel_getter() if callable(cancel_getter) else None
            deadline_getter = getattr(shadow_primitives, "get_request_deadline_s", None)
            deadline_s = deadline_getter() if callable(deadline_getter) else None
            if callable(cancel_check) and cancel_check():
                proposal = ShadowArgumentProposal("abstained", None, "request_cancelled")
            elif isinstance(deadline_s, (int, float)) and time.perf_counter() >= deadline_s:
                proposal = ShadowArgumentProposal("abstained", None, "request_deadline_expired")
            else:
                proposal = propose_shadow_typed_arguments(
                    tool_name=tool_name,
                    full_schema=full_schema,
                    original_kwargs=original_kwargs,
                    primitives=shadow_primitives,
                    role=role,
                    state=state,
                    mode=mode,
                    tool=tool,
                )
        else:
            proposal = ShadowArgumentProposal("abstained", None, f"host_{validation.status}")
    except _ShadowInstanceOverride:
        pass
    except Exception as exc:
        logger.warning("tool argument shadow failed", exc_info=True)
        proposal = ShadowArgumentProposal("abstained", None, f"{type(exc).__name__}: {exc}")
    receipt = make_receipt(
        prepared,
        validation,
        authorization_result="denied" if validation.status == "unauthorized" else "allowed",
        component_timing_ms={"selection": (time.perf_counter() - started) * 1000.0},
        total_timing_ms=(time.perf_counter() - started) * 1000.0,
        observed_downstream_outcome=outcome,
        proposed_arguments=proposal.proposed,
        executed_arguments=original_kwargs,
        selector_status=proposal.status,
        selector_reason=proposal.reason,
        selector_calls=proposal.calls,
        shadow_mode=mode,
    )
    try:
        append_receipt(receipt_path, receipt)
    except Exception:
        logger.warning("tool argument shadow receipt write failed", exc_info=True)


__all__ = ["maybe_typed_arguments", "registry_parameters_to_schema"]

# Native failures that mean "this catalogue cannot use native mode", not
# "native mode produced a wrong answer"; only these trigger the JSON rerun.
_NATIVE_ELIGIBILITY_FAILURES = frozenset(
    {REASON_NATIVE_UNSUPPORTED_CANDIDATES, REASON_NATIVE_TOKENIZER_UNAVAILABLE}
)

# Registry argument-spec keys carried into the JSON-schema projection. The
# registry shape is ``{name: {"type", "required", "description", "default"}}``
# (src/registry/tool_registry.py::Tool.parameters and load_from_yaml);
# ``enum`` / ``minimum`` / ``maximum`` / ``items`` are carried through when a
# programmatic registration declares them, which is what makes an argument
# closed-set.
_SCHEMA_KEYS = ("type", "description", "enum", "minimum", "maximum", "items")


def registry_parameters_to_schema(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project a ToolRegistry parameter map into a JSON-schema object.

    The registry declares per-argument specs under a mapping
    (``{name: {"type", "required", ...}}``), not a Draft 2020-12 object. This
    adapter is the bridge that lets ``tool_schema_to_questions`` see the same
    declaration ``Tool.validate_args`` reads. Anything malformed yields ``{}``:
    the typed path then declines (fail-open) instead of guessing.

    Args:
        parameters: The tool's parameter map, or ``None``.

    Returns:
        A JSON-schema object with ``type: object`` and the projected
        ``properties`` / ``required`` lists, or ``{}``.
    """
    if not isinstance(parameters, Mapping) or not parameters:
        return {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in parameters.items():
        if not isinstance(name, str) or not isinstance(spec, Mapping):
            return {}
        prop = {key: spec[key] for key in _SCHEMA_KEYS if key in spec}
        prop.setdefault("type", "string")
        if spec.get("required", False):
            required.append(name)
        properties[name] = prop
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def maybe_typed_arguments(
    *,
    tool_name: str,
    parameters: Mapping[str, Any] | None,
    state: str | None,
    primitives: Any,
    role: str,
    mode: str = "native",
) -> dict[str, Any] | None:
    """Return a validated closed-set argument dict, or ``None`` (fail-open).

    Args:
        tool_name: Tool name shown in the generated questions.
        parameters: The tool's parameters JSON schema (at the REPL call site
            this is ``registry_parameters_to_schema(tool.parameters)``).
        state: Request/task state string injected into the decision prompt.
        primitives: The ``LLMPrimitives``-shaped seam (must expose
            ``llm_call``).
        role: Registry role the typed call is charged to.
        mode: Preferred decoding mode: ``"native"`` (default; native first with
            a JSON rerun on native-eligibility failures) or ``"json"``.

    Returns:
        The assembled, schema-validated argument dict, or ``None`` when the
        gate does not hold or any step fails. Never raises.
    """
    if not features().typed_decisions_tool_args:
        return None
    if primitives is None or not callable(getattr(primitives, "llm_call", None)):
        return None
    if not isinstance(state, str) or not state.strip():
        return None
    if not isinstance(parameters, Mapping):
        return None

    try:
        schema = dict(parameters)
        mapping = tool_schema_to_questions(tool_name, schema)
        if not mapping.questions or mapping.skipped:
            logger.debug(
                "typed tool args declined for %r: %d question(s), %d skipped",
                tool_name,
                len(mapping.questions),
                len(mapping.skipped),
            )
            return None
        result = _run_pass(primitives, state=state, mapping=mapping, role=role, mode=mode)
        if result is None or result.failures:
            return None
        return assemble_arguments(mapping.questions, result.decisions, schema)
    except Exception:
        logger.debug("typed tool args declined for %r", tool_name, exc_info=True)
        return None


def _run_pass(
    primitives: Any,
    *,
    state: str,
    mapping: ToolQuestionMapping,
    role: str,
    mode: str,
) -> DecisionResult | None:
    """Run the preferred mode, rerunning JSON on native-eligibility failures.

    ``None`` means the preferred mode is unknown (decline without a call).
    """
    if mode == "json":
        return run_typed_decisions(
            primitives,
            state=state,
            questions=mapping.questions,
            role=role,
            mode="json",
        )
    if mode != "native":
        logger.debug("typed tool args declined: unknown mode %r", mode)
        return None
    native = run_typed_decisions(
        primitives,
        state=state,
        questions=mapping.questions,
        role=role,
        mode="native",
    )
    if any(failure.reason in _NATIVE_ELIGIBILITY_FAILURES for failure in native.failures):
        logger.debug("native tool args not eligible; rerunning in json mode")
        return run_typed_decisions(
            primitives,
            state=state,
            questions=mapping.questions,
            role=role,
            mode="json",
        )
    return native
