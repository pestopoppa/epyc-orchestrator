"""Constrained-decoding generation + parse-failure accounting for reviewer output.

RA-7 / RA-9 of the Architect->Reviewer control plane (H2). Pure, dependency-light
(stdlib + jsonschema only) so it is unit-testable WITHOUT any model/server call:

  * ``review_decision_response_schema()`` / ``review_decision_gbnf()`` — a compact
    json_schema payload and a llama.cpp-compatible GBNF grammar constraining the
    load-bearing ReviewDecision core (decision enum + confidence + tripwire/blocking
    channel + optional advisory). The decision enum is sourced from
    ``orchestration/review_decision.schema.json`` so the grammar cannot drift.
  * ``rubric_grading_response_schema(rubric)`` / ``rubric_grading_gbnf(rubric)`` —
    a schema/grammar for grading a candidate against a ReviewRubric (per-item scores
    keyed by the rubric's item ids + an overall decision).
  * ``parse_review_decision(text)`` — extract JSON from possibly-noisy model text,
    validate it against the full review_decision schema, and return a structured
    ``(obj | None, ParseFailure | None)``. The failure object is the accounting hook
    (schema-invalid emissions are themselves a reviewer-quality signal); the counting
    integration lands later — here we only surface structured failure info.

Actually driving llama.cpp with these grammars (the GPU lane) is out of scope and
gated on the v7 grammar-sampler P0; these are pure generators + a parser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

# orchestration/ lives at the repo root: .../src/proactive_delegation/<this> -> parents[2]
_ORCH_DIR = Path(__file__).resolve().parents[2] / "orchestration"
_REVIEW_DECISION_SCHEMA = _ORCH_DIR / "review_decision.schema.json"
_REVIEW_RUBRIC_SCHEMA = _ORCH_DIR / "review_rubric.schema.json"


# ── Schema access ─────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def load_review_decision_schema() -> dict[str, Any]:
    """Load and cache the full ReviewDecision JSON schema."""
    return json.loads(_REVIEW_DECISION_SCHEMA.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _decision_enum() -> tuple[str, ...]:
    """The `decision` enum, sourced from the schema (single source of truth)."""
    schema = load_review_decision_schema()
    return tuple(schema["properties"]["decision"]["enum"])


# ── json_schema payloads (RA-7) ───────────────────────────────────────


def review_decision_response_schema() -> dict[str, Any]:
    """Compact json_schema payload for constraining a ReviewDecision emission.

    A structural subset of the full schema focused on the load-bearing fields.
    Output produced under this schema round-trips through the full validator
    (blocking.tripwire is nested to match the tripwire/advisory split).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "confidence", "blocking"],
        "properties": {
            "decision": {"type": "string", "enum": list(_decision_enum())},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "blocking": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tripwire"],
                "properties": {
                    "tripwire": {"type": "boolean"},
                    "blocking_issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["summary"],
                            "properties": {"summary": {"type": "string"}},
                        },
                    },
                },
            },
            "advisory": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "feedback": {"type": "string"},
                },
            },
        },
    }


def rubric_grading_response_schema(rubric: dict[str, Any]) -> dict[str, Any]:
    """json_schema for grading a candidate against a ReviewRubric.

    The per-item `item` field is constrained to the rubric's own item ids, and the
    overall `decision` to the ReviewDecision enum — both sourced from artifacts.
    """
    item_ids = [item["id"] for item in rubric.get("items", []) if "id" in item]
    item_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["item", "score"],
        "properties": {
            "item": ({"type": "string", "enum": item_ids} if item_ids else {"type": "string"}),
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "note": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["grades", "decision"],
        "properties": {
            "grades": {"type": "array", "minItems": 1, "items": item_schema},
            "decision": {"type": "string", "enum": list(_decision_enum())},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


# ── GBNF generation (RA-7) ────────────────────────────────────────────

# Canonical llama.cpp JSON primitive rules (string/number/boolean/ws).
_GBNF_PRIMITIVES = "\n".join(
    [
        r'ws ::= [ \t\n]*',
        r'boolean ::= "true" | "false"',
        r'number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
        r'string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\""',
    ]
)


def _quoted_literal(value: str) -> str:
    """Render a JSON string literal as a GBNF terminal (e.g. approve -> "\"approve\"")."""
    return '"\\"' + value + '\\""'


def _enum_rule(name: str, values: tuple[str, ...] | list[str]) -> str:
    alts = " | ".join(_quoted_literal(v) for v in values)
    return f"{name} ::= {alts}"


@dataclass
class _Field:
    key: str
    rule: str  # name of the GBNF rule producing this field's value
    required: bool = True


def _object_rule(name: str, fields: list[_Field]) -> str:
    """Build a GBNF rule for a JSON object with ordered, optionally-omitted keys.

    The first field must be required (true for every object we emit). Subsequent
    required fields are joined with a comma; optional fields are wrapped so they may
    be omitted, in fixed order (a deliberate simplification vs llama.cpp's
    any-order converter — documented, and adequate for constrained emission).
    """
    if not fields or not fields[0].required:
        raise ValueError("_object_rule requires a leading required field")

    parts: list[str] = ['"{" ws']
    for idx, f in enumerate(fields):
        kv = f'{_quoted_literal(f.key)} ws ":" ws {f.rule} ws'
        if idx == 0:
            parts.append(kv)
        elif f.required:
            parts.append(f'"," ws {kv}')
        else:
            parts.append(f'( "," ws {kv} )?')
    parts.append('"}"')
    return f"{name} ::= " + " ".join(parts)


def review_decision_gbnf() -> str:
    """GBNF grammar constraining a ReviewDecision emission (matches the response schema).

    Emits ``{"decision": <enum>, "confidence": <number>, "blocking": {"tripwire": <bool>},
    "advisory": {...}?}``. Confidence/score range ([0,1]) is enforced by post-hoc schema
    validation, not the grammar (GBNF cannot express numeric ranges cheaply).
    """
    rules = [
        _object_rule(
            "root",
            [
                _Field("decision", "decision"),
                _Field("confidence", "number"),
                _Field("blocking", "blocking"),
                _Field("advisory", "advisory", required=False),
            ],
        ),
        _enum_rule("decision", _decision_enum()),
        _object_rule("blocking", [_Field("tripwire", "boolean")]),
        _object_rule(
            "advisory",
            [_Field("score", "number"), _Field("feedback", "string", required=False)],
        ),
        _GBNF_PRIMITIVES,
    ]
    return "\n".join(rules) + "\n"


def rubric_grading_gbnf(rubric: dict[str, Any]) -> str:
    """GBNF grammar for a rubric-grading emission.

    Emits ``{"grades": [ {"item": <item-id-enum>, "score": <number>} , ... ],
    "decision": <enum>}``. The item enum is sourced from the rubric's item ids.
    """
    item_ids = [item["id"] for item in rubric.get("items", []) if "id" in item]
    item_value_rule = "item-id" if item_ids else "string"
    rules = [
        _object_rule(
            "root",
            [_Field("grades", "grades"), _Field("decision", "decision")],
        ),
        # one-or-more grade objects separated by commas
        'grades ::= "[" ws grade ( ws "," ws grade )* ws "]"',
        _object_rule("grade", [_Field("item", item_value_rule), _Field("score", "number")]),
        _enum_rule("decision", _decision_enum()),
    ]
    if item_ids:
        rules.append(_enum_rule("item-id", item_ids))
    rules.append(_GBNF_PRIMITIVES)
    return "\n".join(rules) + "\n"


# ── Parse + failure accounting (RA-9) ─────────────────────────────────


class ParseFailureReason(str, Enum):
    """Structured reason a reviewer emission failed to parse into a ReviewDecision."""

    NO_JSON = "no_json"  # no JSON object found in the text
    JSON_DECODE_ERROR = "json_decode_error"  # found braces but json.loads failed
    NOT_OBJECT = "not_object"  # parsed JSON is not an object
    SCHEMA_INVALID = "schema_invalid"  # parsed object violates review_decision schema
    VALIDATOR_UNAVAILABLE = "validator_unavailable"  # jsonschema not importable


@dataclass
class ParseFailure:
    """Accounting record for a failed parse (never a silent fallback)."""

    reason: ParseFailureReason
    detail: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "detail": self.detail,
            "errors": self.errors,
        }


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level ``{...}`` substring, or None.

    Brace-depth aware, string/escape aware. Tolerates fenced code blocks and prose
    around the JSON (common in chat-completion output).
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_review_decision(
    text: str,
) -> tuple[dict[str, Any] | None, ParseFailure | None]:
    """Parse + validate a reviewer emission into a ReviewDecision object.

    Returns ``(obj, None)`` on success or ``(None, ParseFailure)`` on failure. The
    ParseFailure is the accounting hook: callers increment a per-reviewer-config
    failure counter (counting integration lands later) rather than silently falling
    back to a default verdict.
    """
    candidate = _extract_json_object(text)
    if candidate is None:
        return None, ParseFailure(ParseFailureReason.NO_JSON, "no '{' found in text")

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, ParseFailure(ParseFailureReason.JSON_DECODE_ERROR, str(exc))

    if not isinstance(obj, dict):
        return None, ParseFailure(
            ParseFailureReason.NOT_OBJECT, f"top-level JSON is {type(obj).__name__}, not object"
        )

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return None, ParseFailure(
            ParseFailureReason.VALIDATOR_UNAVAILABLE, "jsonschema not installed"
        )

    validator = Draft202012Validator(load_review_decision_schema())
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    if errors:
        msgs = [f"{'$' + ''.join(f'.{p}' for p in e.absolute_path)}: {e.message}" for e in errors]
        return None, ParseFailure(
            ParseFailureReason.SCHEMA_INVALID,
            f"{len(errors)} schema violation(s)",
            errors=msgs[:20],
        )

    return obj, None
