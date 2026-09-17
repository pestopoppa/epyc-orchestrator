"""Response-schema and GBNF builders for the typed decision plane (TD-1).

``build_response_schema(questions)`` returns a Draft 2020-12 JSON Schema for
ONE JSON object that answers every question in a single pass:

    {"answers": {<qid>: {"choice"|"score"|"noul": <value>,
                         "probabilities": {<label>: number},
                         "confidence": number}}}

Every object level sets ``additionalProperties: false``, so an unknown
question id or an invented candidate label is a schema violation rather than
a silent accept. ``runner.run_typed_decisions`` passes this schema to
``LLMPrimitives.llm_call(json_schema=...)`` (mode="json") and validates the
emission against it before accepting any decision.

``build_gbnf(questions)`` returns a llama.cpp GBNF grammar for the same
object, or ``None`` when a faithful grammar cannot be built. LIMITATIONS:

  * A GBNF enum alternative is a single literal token, so a candidate label
    the model's tokenizer splits into multiple tokens cannot be constrained
    by such a rule. The tokenizer is not available at build time, so this
    builder applies a conservative presumption (printable ASCII, no
    whitespace/quotes/backslash, at most ``_MAX_GBNF_LABEL_CHARS`` chars)
    and returns ``None`` rather than emit a grammar that would silently
    mangle a label.
  * GBNF enforces structure only. Candidate membership is baked in as enum
    literals, but numeric ranges, uniqueness and the "probabilities sum to
    1.0" rule are enforced by the JSON schema and the runner, never by the
    grammar.
  * The object key order is fixed to match the schema property order
    (llama.cpp can accept any order; this is a deliberate simplification).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.typed_decisions.types import Question, QuestionKind

_JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_MAX_GBNF_LABEL_CHARS = 16

_GBNF_PRIMITIVES = "\n".join(
    [
        r"ws ::= [ \t\n]*",
        r'boolean ::= "true" | "false"',
        r'number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
    ]
)


def build_response_schema(questions: Sequence[Question]) -> dict[str, Any]:
    """Build the Draft 2020-12 schema for one all-questions JSON response."""
    questions = list(questions)
    return {
        "$schema": _JSON_SCHEMA_DRAFT,
        "title": "typed_decisions_response",
        "type": "object",
        "additionalProperties": False,
        "required": ["answers"],
        "properties": {
            "answers": {
                "type": "object",
                "additionalProperties": False,
                "required": [question.id for question in questions],
                "properties": {question.id: _answer_schema(question) for question in questions},
            }
        },
    }


def build_gbnf(questions: Sequence[Question]) -> str | None:
    """Build a GBNF grammar for the response, or ``None`` when unbuildable.

    Returns ``None`` when the catalogue is empty, when any answer label is
    not a presumptive single token (see module docstring), or when any
    question id cannot be emitted as a JSON string literal in a grammar.
    """
    questions = list(questions)
    if not questions:
        return None
    if any(not _is_gbnf_literal_safe(question.id) for question in questions):
        return None

    labels_per_question = [_labels(question) for question in questions]
    if any(
        not _is_presumptive_single_token(label)
        for labels in labels_per_question
        for label in labels
    ):
        return None

    rules = ['root ::= "{" ws "\\"answers\\"" ws ":" ws answers ws "}"']
    entry_rules = []
    for index, question in enumerate(questions):
        labels = labels_per_question[index]
        answer_rule = f"answer-{index}"
        probabilities_rule = f"probabilities-{index}"
        value_key, value_rule = _value_rule(question, labels)
        entry_rules.append(
            f'entry-{index} ::= {_quoted_literal(question.id)} ws ":" ws {answer_rule}'
        )
        rules.append(
            f'{answer_rule} ::= "{{" ws {value_key} ws ":" ws {value_rule} ws "," ws '
            f'"\\"probabilities\\"" ws ":" ws {probabilities_rule} ws "," ws '
            f'"\\"confidence\\"" ws ":" ws number ws "}}"'
        )
        probability_fields = ' ws "," ws '.join(
            f'{_quoted_literal(label)} ws ":" ws number' for label in labels
        )
        rules.append(f'{probabilities_rule} ::= "{{" ws {probability_fields} ws "}}"')
    rules.append(
        'answers ::= "{" ws '
        + ' ws "," ws '.join(f"entry-{index}" for index in range(len(questions)))
        + ' ws "}"'
    )
    rules.extend(entry_rules)
    rules.append(_GBNF_PRIMITIVES)
    return "\n".join(rules) + "\n"


def _answer_schema(question: Question) -> dict[str, Any]:
    if question.kind is QuestionKind.CHOICE:
        return _object_answer_schema(
            value_key="choice",
            value_schema={"type": "string", "enum": list(question.options)},
            labels=list(question.options),
        )
    if question.kind is QuestionKind.SCORE:
        return _object_answer_schema(
            value_key="score",
            value_schema={"type": "integer", "enum": list(question.levels)},
            labels=[str(level) for level in question.levels],
        )
    return _object_answer_schema(
        value_key="noul",
        value_schema={"type": "boolean"},
        labels=["true", "false"],
    )


def _object_answer_schema(
    *,
    value_key: str,
    value_schema: dict[str, Any],
    labels: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [value_key, "probabilities", "confidence"],
        "properties": {
            value_key: value_schema,
            "probabilities": {
                "type": "object",
                "additionalProperties": False,
                "required": list(labels),
                "properties": {
                    label: {"type": "number", "minimum": 0, "maximum": 1} for label in labels
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


def _labels(question: Question) -> list[str]:
    if question.kind is QuestionKind.CHOICE:
        return list(question.options)
    if question.kind is QuestionKind.SCORE:
        return [str(level) for level in question.levels]
    return ["true", "false"]


def _value_rule(question: Question, labels: list[str]) -> tuple[str, str]:
    """Return ``(json-key-literal, value-rule)`` for one answer object."""
    if question.kind is QuestionKind.CHOICE:
        key = _quoted_literal("choice")
    elif question.kind is QuestionKind.SCORE:
        key = _quoted_literal("score")
    else:
        key = _quoted_literal("noul")
    if question.kind is QuestionKind.NOUL:
        return key, "boolean"
    if question.kind is QuestionKind.SCORE:
        alternatives = " | ".join(f'"{label}"' for label in labels)
    else:
        alternatives = " | ".join(_quoted_literal(label) for label in labels)
    return key, f"({alternatives})"


def _is_presumptive_single_token(label: str) -> bool:
    if not label or len(label) > _MAX_GBNF_LABEL_CHARS:
        return False
    if not label.isascii() or not label.isprintable():
        return False
    return not any(char in label for char in (" ", '"', "\\"))


def _is_gbnf_literal_safe(value: str) -> bool:
    return bool(value) and '"' not in value and "\\" not in value and value.isprintable()


def _quoted_literal(value: str) -> str:
    """Render a JSON string literal as a GBNF terminal (``red`` -> ``"\\"red\\""``)."""
    return '"\\"' + value + '\\""'
