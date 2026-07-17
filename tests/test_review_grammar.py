"""Tests for review_grammar.py — GBNF/json_schema generation + parse accounting (RA-7/RA-9).

Pure-function tests; no model/server call. llama.cpp grammar execution is out of
scope/gated, so we assert structural correctness of the generated grammar/schema and
round-trip behaviour of the parser against the full ReviewDecision schema.
"""

import pytest
from jsonschema import Draft202012Validator

from src.proactive_delegation.review_grammar import (
    ParseFailureReason,
    load_review_decision_schema,
    parse_review_decision,
    review_decision_gbnf,
    review_decision_response_schema,
    rubric_grading_gbnf,
    rubric_grading_response_schema,
)

DECISION_ENUM = [
    "approve",
    "reject",
    "reject_to_empty",
    "request_changes",
    "request_evidence",
    "escalate",
    "abstain",  # CP2/spec §9.2 — grammar auto-sources from review_decision.schema.json
]


# ── json_schema payloads ──────────────────────────────────────────────


class TestResponseSchemas:
    def test_decision_schema_enum_sourced_from_file(self):
        schema = review_decision_response_schema()
        assert set(schema["properties"]["decision"]["enum"]) == set(DECISION_ENUM)

    def test_decision_schema_is_itself_valid_metaschema(self):
        # The generated schema must be a valid JSON Schema.
        Draft202012Validator.check_schema(review_decision_response_schema())

    def test_response_shaped_output_roundtrips_full_schema(self):
        # An object produced under the compact response schema must validate against
        # the FULL review_decision schema (blocking.tripwire nesting matches).
        obj = {
            "decision": "approve",
            "confidence": 0.5,
            "blocking": {"tripwire": False},
        }
        # valid under the compact schema
        Draft202012Validator(review_decision_response_schema()).validate(obj)
        # and under the full schema
        errors = list(Draft202012Validator(load_review_decision_schema()).iter_errors(obj))
        assert errors == [], errors

    def test_rubric_grading_schema_constrains_item_ids(self):
        rubric = {"items": [{"id": "R1"}, {"id": "R2"}]}
        schema = rubric_grading_response_schema(rubric)
        item_schema = schema["properties"]["grades"]["items"]["properties"]["item"]
        assert item_schema["enum"] == ["R1", "R2"]
        assert set(schema["properties"]["decision"]["enum"]) == set(DECISION_ENUM)

    def test_rubric_grading_schema_empty_items(self):
        schema = rubric_grading_response_schema({"items": []})
        item_schema = schema["properties"]["grades"]["items"]["properties"]["item"]
        assert item_schema == {"type": "string"}


# ── GBNF grammars ─────────────────────────────────────────────────────


class TestGBNF:
    def test_decision_gbnf_has_core_rules(self):
        g = review_decision_gbnf()
        assert g.startswith("root ::=")
        for rule in ["root", "decision", "blocking", "advisory", "string", "number", "boolean", "ws"]:
            assert f"{rule} ::=" in g, f"missing rule {rule}"

    def test_decision_gbnf_enumerates_all_decisions(self):
        g = review_decision_gbnf()
        for d in DECISION_ENUM:
            assert f'"\\"{d}\\""' in g, f"decision {d} not in grammar"

    def test_decision_gbnf_blocking_requires_tripwire(self):
        g = review_decision_gbnf()
        # blocking object hard-codes the tripwire key
        assert '"\\"tripwire\\""' in g

    def test_rubric_gbnf_enumerates_item_ids(self):
        g = rubric_grading_gbnf({"items": [{"id": "R1"}, {"id": "R2"}]})
        assert "item-id ::=" in g
        assert '"\\"R1\\""' in g
        assert '"\\"R2\\""' in g

    def test_rubric_gbnf_no_item_enum_when_empty(self):
        g = rubric_grading_gbnf({"items": []})
        assert "item-id ::=" not in g
        assert "grades ::=" in g


# ── parse_review_decision + failure accounting ────────────────────────


class TestParseReviewDecision:
    def test_parse_clean_json(self):
        text = '{"decision": "approve", "confidence": 0.8, "blocking": {"tripwire": false}}'
        obj, failure = parse_review_decision(text)
        assert failure is None
        assert obj["decision"] == "approve"

    def test_parse_with_prose_and_fences(self):
        text = (
            "Sure, here is my verdict:\n```json\n"
            '{"decision": "request_changes", "confidence": 0.6, "blocking": {"tripwire": false},'
            ' "advisory": {"score": 0.4, "feedback": "fix imports"}}\n```\nThanks!'
        )
        obj, failure = parse_review_decision(text)
        assert failure is None, failure
        assert obj["decision"] == "request_changes"
        assert obj["advisory"]["feedback"] == "fix imports"

    def test_parse_no_json(self):
        obj, failure = parse_review_decision("I approve this, looks great.")
        assert obj is None
        assert failure.reason is ParseFailureReason.NO_JSON

    def test_parse_decode_error(self):
        obj, failure = parse_review_decision('{"decision": "approve", }')  # trailing comma
        assert obj is None
        assert failure.reason is ParseFailureReason.JSON_DECODE_ERROR

    def test_parse_schema_invalid_missing_blocking(self):
        obj, failure = parse_review_decision('{"decision": "approve", "confidence": 0.5}')
        assert obj is None
        assert failure.reason is ParseFailureReason.SCHEMA_INVALID
        assert failure.errors  # structured error detail present for accounting

    def test_parse_schema_invalid_bad_enum(self):
        text = '{"decision": "maybe", "confidence": 0.5, "blocking": {"tripwire": false}}'
        obj, failure = parse_review_decision(text)
        assert obj is None
        assert failure.reason is ParseFailureReason.SCHEMA_INVALID

    def test_parse_confidence_out_of_range(self):
        text = '{"decision": "approve", "confidence": 2, "blocking": {"tripwire": false}}'
        obj, failure = parse_review_decision(text)
        assert obj is None
        assert failure.reason is ParseFailureReason.SCHEMA_INVALID

    def test_failure_to_dict_is_accounting_ready(self):
        _, failure = parse_review_decision("no json here")
        d = failure.to_dict()
        assert d["reason"] == "no_json"
        assert "detail" in d and "errors" in d


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
