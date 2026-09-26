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
from collections.abc import Mapping
from typing import Any

from src.features import features
from src.typed_decisions.native import (
    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
)
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.tool_args import (
    ToolQuestionMapping,
    assemble_arguments,
    tool_schema_to_questions,
)
from src.typed_decisions.types import DecisionResult

logger = logging.getLogger(__name__)

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
