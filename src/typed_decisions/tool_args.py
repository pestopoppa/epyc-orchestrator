"""Closed-set tool-argument mapping over typed decisions (TD-4).

This module is the closed-set selection path from intake-1472/1473: a tool's
JSON-schema ``parameters`` object is projected into typed questions
(``Question``), the typed-decision runner resolves them, and
``assemble_arguments`` turns the decisions back into a validated argument
dict. The model therefore SELECTS from declared candidates instead of
inventing an argument value in prose.

Mapping rules (``tool_schema_to_questions``):

* ``enum`` of integers -> one ``SCORE`` question whose ``levels`` are the
  enum values ("explicit levels").
* any other ``enum`` -> one ``CHOICE`` question whose options are the enum
  values stringified for the prompt.
* ``type: boolean`` -> one ``NOUL`` question.
* ``type: integer`` with ``minimum``/``maximum`` -> one ``SCORE`` question
  with the inclusive integer range as levels, capped at
  ``_MAX_SCORE_LEVELS`` (wider or one-sided ranges are skipped rather than
  truncated).
* ``type: array`` whose ``items`` carry an ``enum`` -> one ``NOUL`` question
  per enum value with id ``<arg>__<value>``. This is a DOCUMENTED
  SINGLE-SELECT APPROXIMATION of a multi-select argument: at most one value
  may be true, and ``assemble_arguments`` rejects more than one true.
* everything else -> recorded in the returned ``skipped`` list with a
  reason. Partial mappings never raise; only a non-object ``parameters``
  schema raises ``ToolArgumentError``.

Question ids are stable across calls (``<arg>`` for scalars,
``<arg>__<value>`` for array enums); the caller can therefore reuse a
catalogue and its decisions. Unsupported arguments are skipped, not guessed.

Return shape deviation: the TD-4 task text asked for
``tool_schema_to_questions(...) -> list[Question]`` while also requiring a
returned ``skipped: list[str]``. A bare list cannot carry ``skipped``, so
this implementation returns a ``ToolQuestionMapping`` named tuple with
``questions`` and ``skipped`` list fields — both are lists, and tuple
unpacking (``questions, skipped = ...``) still works. Raising instead was
rejected: a partial mapping is the normal case for real tool schemas.

``assemble_arguments`` fails with ``ToolArgumentError`` (carrying the full
``reasons`` list) when: a required argument has no decision; any mapped
question of a supplied argument is unanswered (partial answers are never
guessed into defaults); a value is outside the declared enum/levels; an
array-enum selects more than one value; or the assembled dict fails
``jsonschema.Draft202012Validator(parameters)`` when ``parameters`` is a
full object schema.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from src.typed_decisions.types import Decision, Question, QuestionKind

__all__ = [
    "ToolArgumentError",
    "ToolQuestionMapping",
    "assemble_arguments",
    "tool_schema_to_questions",
]

# A bounded integer range is mapped to one SCORE question per level; a wider
# range is skipped rather than silently truncated to a subset of its levels.
_MAX_SCORE_LEVELS = 16

# Array-enum ids are "<arg>__<value>" (double underscore, as specified).
_ARRAY_ID_SEPARATOR = "__"


class ToolArgumentError(ValueError):
    """A tool-argument mapping/assembly failure with machine-readable reasons.

    ``reasons`` is a list of human-readable, one-per-cause strings; the
    exception message joins them for logs. The class is local to this module
    on purpose: it is the TD-4 contract, not a generic schema error.
    """

    def __init__(self, message: str, *, reasons: Sequence[str] | None = None) -> None:
        self.reasons: list[str] = list(reasons) if reasons is not None else []
        super().__init__(message)


class ToolQuestionMapping(NamedTuple):
    """``tool_schema_to_questions`` result: mapped questions + skip reasons.

    ``skipped`` entries are ``"<arg>: <reason>"`` strings so a caller can
    surface why part of a tool schema is not selectable without parsing a
    tree. Both fields are plain lists (see module docstring for the return
    shape rationale).
    """

    questions: list[Question]
    skipped: list[str]


# ── Schema -> questions ───────────────────────────────────────────────────


def tool_schema_to_questions(tool_name: str, parameters: dict) -> ToolQuestionMapping:
    """Project a JSON-schema parameters object into typed questions.

    Args:
        tool_name: Tool name, used only in the generated question text.
        parameters: The tool's parameters JSON schema. Must describe an
            object (``type`` absent or ``"object"``); anything else raises
            ``ToolArgumentError``. Arguments whose schema cannot be mapped
            are collected in ``skipped`` with a reason.

    Returns:
        ``ToolQuestionMapping(questions, skipped)``. Arguments are visited in
        schema order; ids are ``<arg>`` (scalars) or ``<arg>__<value>``
        (array enums) and are collision-checked across the whole tool.
    """
    _require_object_schema(parameters)
    properties = parameters.get("properties") or {}
    questions: list[Question] = []
    skipped: list[str] = []
    seen_ids: set[str] = set()

    for arg_name, arg_schema in properties.items():
        if not isinstance(arg_name, str):
            skipped.append(f"{arg_name!r}: argument name is not a string")
            continue
        try:
            mapped, reason = _map_argument(tool_name, arg_name, arg_schema)
        except ValueError as exc:  # defensive: a Question invariant rejected the mapping
            mapped, reason = [], f"unusable schema: {exc}"
        if reason is not None:
            skipped.append(f"{arg_name}: {reason}")
            continue
        if not mapped:
            skipped.append(f"{arg_name}: no questions produced")
            continue
        mapped_ids = [question.id for question in mapped]
        if len(set(mapped_ids)) != len(mapped_ids) or any(
            question_id in seen_ids for question_id in mapped_ids
        ):
            skipped.append(f"{arg_name}: question ids collide with another argument")
            continue
        seen_ids.update(mapped_ids)
        questions.extend(mapped)

    return ToolQuestionMapping(questions=questions, skipped=skipped)


def _require_object_schema(parameters: Any) -> None:
    if not isinstance(parameters, Mapping):
        raise ToolArgumentError(
            "tool parameters must be a JSON-schema object",
            reasons=[f"parameters is {type(parameters).__name__}, not an object"],
        )
    schema_type = parameters.get("type")
    if schema_type is not None and schema_type != "object":
        raise ToolArgumentError(
            f"tool parameters schema must describe an object, got type={schema_type!r}",
            reasons=[f"unsupported top-level schema type: {schema_type!r}"],
        )
    properties = parameters.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        raise ToolArgumentError(
            "tool parameters 'properties' must be an object",
            reasons=[f"'properties' is {type(properties).__name__}, not an object"],
        )


def _map_argument(
    tool_name: str,
    arg_name: str,
    schema: Any,
) -> tuple[list[Question], str | None]:
    """Map one argument schema to questions, or return a skip reason."""
    if not isinstance(schema, Mapping):
        return [], f"argument schema is not an object ({type(schema).__name__})"
    schema_type = schema.get("type")
    enum_values = schema.get("enum")

    if isinstance(enum_values, list) and enum_values:
        if all(_is_int(value) for value in enum_values):
            return _score_question(
                tool_name,
                arg_name,
                schema,
                [int(value) for value in enum_values],
                "integer enum levels",
            )
        return _choice_question(tool_name, arg_name, schema, enum_values)

    if schema_type == "boolean":
        return [_noul_question(tool_name, arg_name, schema)], None

    if schema_type == "integer":
        levels = _bounded_integer_levels(schema)
        if levels is None:
            return [], (
                "integer without explicit levels and without a minimum/maximum "
                f"span of at most {_MAX_SCORE_LEVELS} values"
            )
        return _score_question(tool_name, arg_name, schema, levels, "integer bounds")

    if schema_type == "array":
        return _array_questions(tool_name, arg_name, schema)

    return [], f"unsupported JSON-schema type {schema_type!r}"


def _choice_question(
    tool_name: str,
    arg_name: str,
    schema: Mapping[str, Any],
    enum_values: Sequence[Any],
) -> tuple[list[Question], str | None]:
    labels = [str(value) for value in enum_values]
    if len(labels) < 2:
        return [], "enum has fewer than 2 values"
    if len(set(labels)) != len(labels):
        return [], "enum values collide after string conversion"
    question = Question(
        id=arg_name,
        kind=QuestionKind.CHOICE,
        text=_argument_text(tool_name, arg_name),
        options=tuple(labels),
        criteria=_criteria(schema, "choose exactly one candidate label"),
    )
    return [question], None


def _score_question(
    tool_name: str,
    arg_name: str,
    schema: Mapping[str, Any],
    levels: Sequence[int],
    source: str,
) -> tuple[list[Question], str | None]:
    if len(set(levels)) != len(levels):
        return [], f"{source} contain duplicate levels"
    if any(not _is_int(level) for level in levels):
        return [], f"{source} are not all integers"
    question = Question(
        id=arg_name,
        kind=QuestionKind.SCORE,
        text=_argument_text(tool_name, arg_name),
        levels=tuple(levels),
        criteria=_criteria(schema, "choose exactly one candidate level"),
    )
    return [question], None


def _noul_question(
    tool_name: str,
    arg_name: str,
    schema: Mapping[str, Any],
) -> Question:
    return Question(
        id=arg_name,
        kind=QuestionKind.NOUL,
        text=_argument_text(tool_name, arg_name),
        criteria=_criteria(schema, "answer true or false"),
    )


def _array_questions(
    tool_name: str,
    arg_name: str,
    schema: Mapping[str, Any],
) -> tuple[list[Question], str | None]:
    items = schema.get("items")
    if not isinstance(items, Mapping):
        return [], "array items are not a schema object"
    item_enum = items.get("enum")
    if not isinstance(item_enum, list) or not item_enum:
        return [], "array items are not an enum (a closed value set is required)"
    labels = [str(value) for value in item_enum]
    if len(set(labels)) != len(labels):
        return [], "array item enum values collide after string conversion"
    criteria = (
        "single-select approximation of a multi-select argument: answer true for at most one value",
    )
    questions = [
        Question(
            id=f"{arg_name}{_ARRAY_ID_SEPARATOR}{label}",
            kind=QuestionKind.NOUL,
            text=(
                f"Should the {arg_name!r} argument of tool {tool_name!r} "
                f"include the value {label!r}?"
            ),
            criteria=criteria,
        )
        for label in labels
    ]
    return questions, None


def _argument_text(tool_name: str, arg_name: str) -> str:
    return f"Choose the value for argument {arg_name!r} of tool {tool_name!r}."


def _criteria(schema: Mapping[str, Any], detail: str) -> tuple[str, ...]:
    criteria = [detail]
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        criteria.append(f"allowed values: {json.dumps(enum_values)}")
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if _is_int(minimum) and _is_int(maximum):
        criteria.append(f"allowed range: {minimum}..{maximum}")
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        criteria.append(description.strip())
    return tuple(criteria)


def _bounded_integer_levels(schema: Mapping[str, Any]) -> list[int] | None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if not _is_int(minimum) or not _is_int(maximum):
        return None
    low = int(minimum)
    high = int(maximum)
    if high < low or high - low + 1 > _MAX_SCORE_LEVELS:
        return None
    return list(range(low, high + 1))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ── Decisions -> arguments ────────────────────────────────────────────────


def assemble_arguments(
    questions: Sequence[Question],
    decisions: Mapping[str, Decision] | Sequence[Decision],
    parameters: dict,
) -> dict[str, Any]:
    """Build the validated argument dict from typed decisions.

    Args:
        questions: The catalogue produced by ``tool_schema_to_questions`` (or
            an equivalent list with the same ids).
        decisions: ``Decision`` records, either keyed by question id or as a
            sequence (ids are read from ``Decision.question_id``; on a
            duplicate id the last record wins).
        parameters: The tool's parameters schema. Mapped questions are
            re-derived from this schema, so the caller cannot smuggle a value
            outside the declared enum/levels.

    Returns:
        The argument dict, containing only resolved arguments (optional
        arguments with no decision are omitted). Validated with
        ``Draft202012Validator(parameters)`` when ``parameters`` is a full
        object schema (``type: object`` or an explicit ``$schema``).

    Raises:
        ToolArgumentError: with every failure in ``reasons`` — see the module
            docstring for the exact conditions. All reasons are collected
            before raising so one call reports the whole problem.
    """
    _require_object_schema(parameters)
    supplied_ids = [question.id for question in questions]
    if len(set(supplied_ids)) != len(supplied_ids):
        raise ToolArgumentError(
            "questions contain duplicate ids",
            reasons=["duplicate question id in the supplied questions"],
        )
    decision_map = _decision_map(decisions)
    properties = parameters.get("properties") or {}
    required = _required_arguments(parameters)
    assembled: dict[str, Any] = {}
    reasons: list[str] = []

    for arg_name, arg_schema in properties.items():
        if not isinstance(arg_name, str):
            reasons.append(f"{arg_name!r}: argument name is not a string")
            continue
        mapped, reason = _map_argument("", arg_name, arg_schema)
        is_required = arg_name in required
        if reason is not None:
            if is_required:
                reasons.append(f"required argument {arg_name!r} is not mappable ({reason})")
            continue
        supplied = {question.id for question in mapped} & set(supplied_ids)
        absent = sorted({question.id for question in mapped} - supplied)
        if absent and (is_required or any(question_id in decision_map for question_id in absent)):
            reasons.append(
                f"argument {arg_name!r}: questions {absent} are absent from the "
                "supplied question catalogue"
            )
            continue
        if absent:
            continue
        answered = [question for question in mapped if question.id in decision_map]
        if not answered:
            if is_required:
                reasons.append(f"required argument {arg_name!r} was not answered")
            continue
        if len(answered) != len(mapped):
            reasons.append(
                f"argument {arg_name!r} has {len(mapped) - len(answered)} of "
                f"{len(mapped)} mapped question(s) unanswered; partial answers "
                "are never guessed"
            )
            continue
        try:
            assembled[arg_name] = _argument_value(arg_name, arg_schema, mapped, decision_map)
        except ToolArgumentError as exc:
            reasons.extend(exc.reasons)

    if reasons:
        raise ToolArgumentError(
            "cannot assemble tool arguments: " + "; ".join(reasons),
            reasons=reasons,
        )

    if _is_full_schema(parameters):
        try:
            Draft202012Validator(parameters).validate(assembled)
        except ValidationError as exc:
            path = "$" + "".join(f".{part}" for part in exc.absolute_path)
            raise ToolArgumentError(
                f"assembled arguments violate the tool schema at {path}: {exc.message}",
                reasons=[f"schema validation failed at {path}: {exc.message}"],
            ) from exc
        except SchemaError as exc:
            raise ToolArgumentError(
                f"tool parameters are not a valid Draft 2020-12 schema: {exc.message}",
                reasons=[f"invalid tool schema: {exc.message}"],
            ) from exc

    return assembled


def _argument_value(
    arg_name: str,
    arg_schema: Mapping[str, Any],
    mapped: Sequence[Question],
    decision_map: Mapping[str, Decision],
) -> Any:
    schema_type = arg_schema.get("type")
    if schema_type == "array":
        return _array_value(arg_name, arg_schema, mapped, decision_map)

    question = mapped[0]
    decision = decision_map[question.id]
    if question.kind is QuestionKind.CHOICE:
        return _enum_value(arg_name, arg_schema, decision)
    if question.kind is QuestionKind.SCORE:
        value = decision.value
        if not _is_int(value) or value not in question.levels:
            raise ToolArgumentError(
                f"argument {arg_name!r} value {value!r} is outside the declared levels",
                reasons=[
                    f"argument {arg_name!r}: value {value!r} is outside the "
                    f"declared levels {list(question.levels)}"
                ],
            )
        return int(value)
    value = decision.value
    if not isinstance(value, bool):
        raise ToolArgumentError(
            f"argument {arg_name!r} value {value!r} is not a boolean",
            reasons=[f"argument {arg_name!r}: value {value!r} is not a boolean"],
        )
    return value


def _enum_value(
    arg_name: str,
    arg_schema: Mapping[str, Any],
    decision: Decision,
) -> Any:
    enum_values = arg_schema.get("enum") or []
    by_label: dict[str, Any] = {}
    for value in enum_values:
        label = str(value)
        if label in by_label:
            raise ToolArgumentError(
                f"argument {arg_name!r} enum labels are ambiguous",
                reasons=[f"argument {arg_name!r}: enum labels collide after string conversion"],
            )
        by_label[label] = value
    value = decision.value
    if not isinstance(value, str) or value not in by_label:
        raise ToolArgumentError(
            f"argument {arg_name!r} value {value!r} is outside the declared enum",
            reasons=[
                f"argument {arg_name!r}: value {value!r} is outside the declared "
                f"set {list(by_label)}"
            ],
        )
    return by_label[value]


def _array_value(
    arg_name: str,
    arg_schema: Mapping[str, Any],
    mapped: Sequence[Question],
    decision_map: Mapping[str, Decision],
) -> list[Any]:
    items = arg_schema.get("items")
    item_enum = items.get("enum") if isinstance(items, Mapping) else None
    if not isinstance(item_enum, list) or len(item_enum) != len(mapped):
        raise ToolArgumentError(
            f"argument {arg_name!r} array schema does not match the mapped questions",
            reasons=[
                f"argument {arg_name!r}: cannot map the single-select approximation "
                "back to the declared item enum"
            ],
        )
    selected: list[Any] = []
    for question, item_value in zip(mapped, item_enum):
        decision = decision_map[question.id]
        if not isinstance(decision.value, bool):
            raise ToolArgumentError(
                f"argument {arg_name!r} question {question.id!r} is not boolean",
                reasons=[
                    f"argument {arg_name!r}: {question.id} answered with "
                    f"{type(decision.value).__name__}, expected bool"
                ],
            )
        if decision.value:
            selected.append(item_value)
    if len(selected) > 1:
        raise ToolArgumentError(
            f"argument {arg_name!r} selected {len(selected)} values",
            reasons=[
                f"argument {arg_name!r}: {len(selected)} values selected; the "
                "single-select approximation allows at most one"
            ],
        )
    return selected


def _decision_map(
    decisions: Mapping[str, Decision] | Sequence[Decision],
) -> dict[str, Decision]:
    if isinstance(decisions, Mapping):
        items: list[tuple[Any, Any]] = list(decisions.items())
    else:
        items = [
            (getattr(decision, "question_id", position), decision)
            for position, decision in enumerate(decisions)
        ]
    mapping: dict[str, Decision] = {}
    for key, decision in items:
        if not isinstance(decision, Decision):
            raise ToolArgumentError(
                "decisions must be Decision records",
                reasons=[f"decision {key!r} is {type(decision).__name__}, not Decision"],
            )
        mapping[str(key)] = decision
    return mapping


def _required_arguments(parameters: Mapping[str, Any]) -> set[str]:
    required = parameters.get("required")
    if required is None:
        return set()
    if isinstance(required, str) or not isinstance(required, Sequence):
        raise ToolArgumentError(
            "tool parameters 'required' must be a list of argument names",
            reasons=[f"'required' is {type(required).__name__}, not a list"],
        )
    return {str(name) for name in required}


def _is_full_schema(parameters: Mapping[str, Any]) -> bool:
    """True when ``parameters`` can validate a result as a whole schema.

    A bare ``{"properties": ...}`` fragment is a legal parameters shape for
    mapping but is not treated as a full schema (no top-level ``type`` and no
    ``$schema``); it is still structurally enforced by the assembly checks.
    """
    return "$schema" in parameters or parameters.get("type") == "object"
