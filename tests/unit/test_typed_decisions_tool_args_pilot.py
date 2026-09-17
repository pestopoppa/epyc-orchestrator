"""Unit tests for the TD-4 live pilot harness (tool_args_pilot.py).

All runs use a split fake primitives object: the closed-set arm is identified
by the ``json_schema`` kwarg the typed-decision runner passes and is answered
from the case's expected argument dict (or a deliberately partial/unparseable
response in the failure tests); the free-form arm is answered with a canned
JSON arguments object. No model or server is touched.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.typed_decisions.measure import MeasurementError
from src.typed_decisions.tool_args import tool_schema_to_questions
from src.typed_decisions.tool_args_pilot import (
    PilotCase,
    build_cases,
    main,
    run_tool_args_pilot,
)
from src.typed_decisions.types import QuestionKind

ROLE = "worker"

_CASES = build_cases()
_CASE_BY_STATE = {case.state: case for case in _CASES}
_CASE_BY_ID = {case.case_id: case for case in _CASES}
_EXPECTED_ARGS = sum(len(case.expected) for case in _CASES)


def _mapping(case: PilotCase):
    return tool_schema_to_questions(case.tool, case.parameters)


def _closed_response(case: PilotCase) -> str:
    """Schema-valid runner response whose values equal the case ground truth."""
    answers: dict[str, dict] = {}
    for question in _mapping(case).questions:
        if question.kind is QuestionKind.CHOICE:
            value = case.expected[question.id]
            answers[question.id] = {
                "choice": value,
                "probabilities": {
                    label: 1.0 if label == value else 0.0 for label in question.options
                },
                "confidence": 1.0,
            }
        elif question.kind is QuestionKind.SCORE:
            value = case.expected[question.id]
            labels = [str(level) for level in question.levels]
            answers[question.id] = {
                "score": value,
                "probabilities": {label: 1.0 if label == str(value) else 0.0 for label in labels},
                "confidence": 1.0,
            }
        else:
            arg, separator, label = question.id.partition("__")
            if separator:
                selected = [str(item) for item in case.expected[arg]]
                value = label in selected
            else:
                value = bool(case.expected[question.id])
            answers[question.id] = {
                "noul": value,
                "probabilities": {
                    "true": 1.0 if value else 0.0,
                    "false": 0.0 if value else 1.0,
                },
                "confidence": 1.0,
            }
    return json.dumps({"answers": answers})


class _FakePrimitives:
    """Canned per-case responder split by which arm is calling."""

    def __init__(
        self,
        *,
        closed: Callable[[PilotCase], str] = _closed_response,
        free: Callable[[PilotCase], str] | None = None,
    ):
        self.closed = closed
        self.free = free if free is not None else (lambda case: json.dumps(case.expected))
        self.calls: list[dict] = []
        self._last_inference_meta: dict = {}

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        self._last_inference_meta = {"tokens": 9, "elapsed_ms": 1.0}
        case = self._case_for(prompt)
        if kwargs.get("json_schema") is not None:
            return self.closed(case)
        return self.free(case)

    @staticmethod
    def _case_for(prompt: str) -> PilotCase:
        for state, case in _CASE_BY_STATE.items():
            if state in prompt:
                return case
        raise AssertionError("prompt does not contain any known pilot case state")


def _case_rows(receipt: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in receipt["results"]["cases"]}


# ── 1. Case catalogue ─────────────────────────────────────────────────────


class TestBuildCases:
    def test_catalogue_is_deterministic_and_sized(self):
        first = build_cases()
        second = build_cases()

        assert first == second
        assert len(first) >= 18
        assert {case.tool for case in first} == {
            "schedule_meeting",
            "deploy_service",
            "file_ticket",
        }
        assert len({case.case_id for case in first}) == len(first)

    def test_expected_dicts_cover_the_mapping_rules(self):
        kinds: set[QuestionKind] = set()
        for case in _CASES:
            mapping = _mapping(case)
            kinds.update(question.kind for question in mapping.questions)
            Draft202012Validator(case.parameters).validate(case.expected)

            assert len(mapping.skipped) == 1, (case.case_id, mapping.skipped)
            mapped_args = {question.id.split("__", 1)[0] for question in mapping.questions}
            assert set(case.expected) == mapped_args

            for value in case.expected.values():
                if isinstance(value, bool):
                    continue
                if isinstance(value, list):
                    for item in value:
                        assert str(item) in case.state
                else:
                    assert str(value) in case.state

        assert kinds == {QuestionKind.CHOICE, QuestionKind.SCORE, QuestionKind.NOUL}
        arrays = [
            value for case in _CASES for value in case.expected.values() if isinstance(value, list)
        ]
        assert any(value == [] for value in arrays)
        assert any(value for value in arrays)
        booleans = [
            value for case in _CASES for value in case.expected.values() if isinstance(value, bool)
        ]
        assert True in booleans and False in booleans


# ── 2. Both arms scoring ──────────────────────────────────────────────────


class TestPilotArms:
    def test_both_arms_score_perfectly_and_receipt_shape(self, tmp_path: Path):
        primitives = _FakePrimitives()
        receipt_path = tmp_path / "pilot.json"

        receipt = run_tool_args_pilot(
            primitives,
            role=ROLE,
            receipt_path=receipt_path,
        )

        assert receipt["study"] == "tool_args_pilot"
        assert receipt["mode"] == "closed_set_vs_free_form"
        assert receipt["role"] == ROLE
        assert receipt["timestamp"]
        assert receipt["metric_directions"] == {
            "exact_match": "higher_better",
            "per_arg_exact_match": "higher_better",
            "agreement": "higher_better",
            "wall_ms": "lower_better",
        }
        counts = receipt["counts"]
        assert counts["cases"] == 18
        assert counts["tools"] == 3
        assert counts["closed_set_resolved"] == 18
        assert counts["closed_set_exact_match"] == 18
        assert counts["closed_set_failures"] == 0
        assert counts["free_form_resolved"] == 18
        assert counts["free_form_exact_match"] == 18
        assert counts["free_form_failures"] == 0
        assert counts["agreement_compared"] == 18
        assert counts["agreement_agreeing"] == 18

        closed = receipt["results"]["arms"]["closed_set"]
        free = receipt["results"]["arms"]["free_form"]
        assert closed["decode"] == "typed_json"
        assert free["decode"] == "free_form_text"
        assert closed["exact_match_rate"] == 1.0
        assert closed["per_arg_total"] == _EXPECTED_ARGS
        assert closed["per_arg_exact_match"] == _EXPECTED_ARGS
        assert closed["per_arg_exact_match_rate"] == 1.0
        assert closed["wall_ms"] >= 0.0
        assert len(closed["per_case_ms"]) == 18
        assert closed["tokens_generated"] == 9.0 * 18
        assert closed["calls_with_token_meta"] == 18
        assert free["exact_match_rate"] == 1.0
        assert free["tokens_generated"] == 9.0 * 18
        assert receipt["results"]["agreement"] == {"compared": 18, "agreeing": 18, "rate": 1.0}

        rows = _case_rows(receipt)
        assert set(rows) == {case.case_id for case in _CASES}
        for case in _CASES:
            row = rows[case.case_id]
            assert row["expected"] == case.expected
            assert row["closed_set"]["exact_match"] is True
            assert row["free_form"]["exact_match"] is True
        assert rows["deploy-02"]["closed_set"]["skipped"] == [
            "notes: unsupported JSON-schema type 'string'"
        ]

        assert len(primitives.calls) == 36
        assert all(call.get("json_schema") is not None for call in primitives.calls[:18])
        assert all(call.get("json_schema") is None for call in primitives.calls[18:])
        assert len(receipt["prompt_sha256"]) == 36
        assert len(set(receipt["prompt_sha256"])) == 36

        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert loaded == receipt
        assert receipt["receipt_path"] == str(receipt_path)

    def test_closed_set_partial_answers_count_as_wrong(self, tmp_path: Path):
        def partial(case: PilotCase) -> str:
            payload = json.loads(_closed_response(case))
            if case.case_id == "meeting-01":
                del payload["answers"]["priority"]
            return json.dumps(payload)

        primitives = _FakePrimitives(closed=partial)
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "pilot.json")

        counts = receipt["counts"]
        assert counts["closed_set_resolved"] == 17
        assert counts["closed_set_failures"] == 1
        assert counts["closed_set_exact_match"] == 17
        assert counts["free_form_exact_match"] == 18
        assert counts["agreement_compared"] == 17
        assert counts["agreement_agreeing"] == 17

        closed = receipt["results"]["arms"]["closed_set"]
        assert closed["per_arg_total"] == _EXPECTED_ARGS
        assert closed["per_arg_exact_match"] == _EXPECTED_ARGS - len(
            _CASE_BY_ID["meeting-01"].expected
        )
        row = _case_rows(receipt)["meeting-01"]
        assert row["closed_set"]["resolved"] is False
        assert row["closed_set"]["failure_count"] >= 1
        assert row["closed_set"]["error"]
        assert row["closed_set"]["exact_match"] is False

        closed_calls = [call for call in primitives.calls if call.get("json_schema") is not None]
        assert len(closed_calls) == 19

    def test_free_form_wrong_value_reduces_exact_match_and_agreement(self, tmp_path: Path):
        def wrong(case: PilotCase) -> str:
            payload = dict(case.expected)
            if case.case_id == "meeting-02":
                payload["priority"] = "normal"
            return json.dumps(payload)

        primitives = _FakePrimitives(free=wrong)
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "pilot.json")

        counts = receipt["counts"]
        assert counts["closed_set_exact_match"] == 18
        assert counts["free_form_resolved"] == 18
        assert counts["free_form_failures"] == 0
        assert counts["free_form_exact_match"] == 17
        assert counts["agreement_compared"] == 18
        assert counts["agreement_agreeing"] == 17

        free = receipt["results"]["arms"]["free_form"]
        assert free["per_arg_total"] == _EXPECTED_ARGS
        assert free["per_arg_exact_match"] == _EXPECTED_ARGS - 1
        row = _case_rows(receipt)["meeting-02"]
        assert row["free_form"]["resolved"] is True
        assert row["free_form"]["exact_match"] is False
        assert row["free_form"]["arguments"]["priority"] == "normal"

    def test_free_form_parse_failure_counts_as_wrong(self, tmp_path: Path):
        def unparseable(case: PilotCase) -> str:
            if case.case_id == "ticket-03":
                return "the arguments are severity p4 and nothing else"
            return json.dumps(case.expected)

        primitives = _FakePrimitives(free=unparseable)
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "pilot.json")

        counts = receipt["counts"]
        assert counts["free_form_resolved"] == 17
        assert counts["free_form_failures"] == 1
        assert counts["free_form_exact_match"] == 17
        assert counts["agreement_compared"] == 17
        row = _case_rows(receipt)["ticket-03"]
        assert row["free_form"]["resolved"] is False
        assert "no balanced JSON object" in row["free_form"]["error"]
        free = receipt["results"]["arms"]["free_form"]
        assert free["per_arg_exact_match"] == _EXPECTED_ARGS - len(row["expected"])

    def test_no_resolution_in_either_arm_reports_no_agreement(self, tmp_path: Path):
        primitives = _FakePrimitives(
            closed=lambda case: "not json",
            free=lambda case: "still not json",
        )
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "pilot.json")

        counts = receipt["counts"]
        assert counts["closed_set_resolved"] == 0
        assert counts["closed_set_failures"] == 18
        assert counts["free_form_resolved"] == 0
        assert counts["free_form_failures"] == 18
        assert receipt["results"]["agreement"] == {"compared": 0, "agreeing": 0, "rate": None}
        assert receipt["results"]["arms"]["closed_set"]["exact_match_rate"] == 0.0
        assert receipt["results"]["arms"]["closed_set"]["per_arg_exact_match"] == 0
        assert receipt["results"]["arms"]["free_form"]["per_arg_total"] == _EXPECTED_ARGS
        assert receipt["results"]["arms"]["free_form"]["tokens_generated"] == 9.0 * 18


# ── 3. Dry-run and primitives gate ────────────────────────────────────────


class TestPilotGates:
    def test_dry_run_issues_no_calls_and_writes_no_receipt(self, tmp_path: Path):
        primitives = _FakePrimitives()
        receipt_path = tmp_path / "pilot.json"

        receipt = run_tool_args_pilot(
            primitives,
            role=ROLE,
            dry_run=True,
            receipt_path=receipt_path,
        )

        assert receipt["dry_run"] is True
        plan = receipt["plan"]
        assert plan["mode"] == "closed_set_vs_free_form"
        assert plan["arms"] == ["closed_set", "free_form"]
        assert [entry["case_id"] for entry in plan["cases"]] == [case.case_id for case in _CASES]
        assert plan["cases"][0]["expected"] == _CASES[0].expected
        assert plan["cases"][0]["question_ids"]
        assert plan["cases"][0]["skipped"]
        assert plan["counts"] == {"cases": 18, "tools": 3}
        assert primitives.calls == []
        assert not receipt_path.exists()

    def test_live_primitives_are_required(self):
        with pytest.raises(MeasurementError):
            run_tool_args_pilot(None, role=ROLE)
        with pytest.raises(MeasurementError):
            run_tool_args_pilot(object(), role=ROLE)


# ── 4. CLI surface ────────────────────────────────────────────────────────


class TestPilotCli:
    def test_refuses_without_live_or_dry_run(self):
        assert main([]) == 2

    def test_dry_run_prints_the_plan(self, capsys: pytest.CaptureFixture):
        code = main(["--dry-run"])

        assert code == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["dry_run"] is True
        assert printed["plan"]["counts"]["cases"] == 18

    def test_live_runs_with_injected_primitives(self, tmp_path: Path, monkeypatch):
        primitives = _FakePrimitives()
        monkeypatch.setattr(
            "src.typed_decisions.tool_args_pilot._live_primitives",
            lambda: primitives,
        )
        receipt_path = tmp_path / "pilot.json"

        code = main(["--live", "--role", ROLE, "--receipt", str(receipt_path)])

        assert code == 0
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["counts"]["cases"] == 18
        assert receipt["counts"]["closed_set_exact_match"] == 18
