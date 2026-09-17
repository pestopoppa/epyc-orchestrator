"""Live pilot: closed-set tool arguments vs free-form JSON (TD-4).

Two arms answer the SAME deterministic cases (``build_cases()``), each a
(state text, expected argument dict) pair over one of three tool schemas:

* **closed_set** — ``tool_schema_to_questions`` projects each tool schema
  into typed questions (``Question`` records: string enum -> Choice, boolean
  -> Noul, bounded integer -> Score, array-of-enum -> one Noul per value,
  unsupported types skipped), ``run_typed_decisions`` answers them in one
  JSON-mode pass, and ``assemble_arguments`` re-validates the typed decisions
  into the argument dict. The model never invents an argument value.
* **free_form** — one ``llm_call`` asks for the raw JSON arguments object for
  the named tool and state; the emission is JSON-extracted and validated
  against the same tool schema. This is the incumbent shape the closed-set
  arm replaces.

Scoring: a case is an exact match iff the arm produced a dict equal to the
case's expected dict (type-aware, key order irrelevant). Every failure
(transport, parse, schema, assembly) counts as WRONG, never as skipped. Per
arm the receipt reports resolved cases, exact-match cases and rate, per-arg
exact-match pairs and rate (a failed case contributes zero matches but stays
in the denominator), failures, wall time, and generated tokens when
``_last_inference_meta`` exposes them. ``agreement`` is computed between the
arms on cases both resolved.

Every expected argument value is derivable from the case's state text by
construction — the state names the enum label, the integer, the boolean
directive and the array value (or "none" for an empty selection) — so
exact-match scoring is unambiguous. The three case families exercise the
mapping rules:

* ``schedule_meeting`` — ``priority`` string enum -> Choice, ``send_invite``
  -> Noul, bounded 1..8 ``duration_hours`` -> Score; ``topic`` (string) is
  the skipped unsupported argument.
* ``deploy_service`` — ``environment`` enum -> Choice, ``canary`` -> Noul,
  bounded 1..6 ``replicas`` -> Score, ``labels`` array-of-enum -> per-value
  Noul; ``notes`` (string) is skipped.
* ``file_ticket`` — ``severity`` enum -> Choice, ``escalate`` -> Noul,
  bounded 0..8 ``effort_points`` -> Score, ``categories`` array-of-enum ->
  per-value Noul; ``assignee`` (string) is skipped.

Receipts follow the ``measure`` convention (``timestamp``, ``mode``,
``role``, ``counts``, ``results`` plus explicit ``metric_directions``) and
are written to ``receipt_path`` or
``<artifacts_dir or tmp>/typed_decisions/tool_args_pilot-<utc-stamp>.json``.
Real model calls require ``--live``; the CLI refuses otherwise (exit 2). A
study with no live primitives raises ``MeasurementError`` — numbers are never
fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from src.typed_decisions.measure import (
    MeasurementError,
    _last_inference_meta,
    _live_primitives,
    _now_iso,
    _require_primitives,
    _write_receipt,
)
from src.typed_decisions.runner import _extract_json_object, run_typed_decisions
from src.typed_decisions.tool_args import (
    ToolArgumentError,
    assemble_arguments,
    tool_schema_to_questions,
)

__all__ = ["PilotCase", "build_cases", "main", "run_tool_args_pilot"]

_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# Output budget for the free-form arm; the closed arm computes its own
# budget inside the runner.
_FREE_FORM_N_TOKENS = 256

_DECODE_SEED = 0

_SCHEDULE_MEETING_SCHEMA: dict[str, Any] = {
    "$schema": _SCHEMA_DRAFT,
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
        "send_invite": {"type": "boolean"},
        "duration_hours": {"type": "integer", "minimum": 1, "maximum": 8},
        "topic": {"type": "string"},
    },
    "required": ["priority", "send_invite", "duration_hours"],
}

_DEPLOY_SERVICE_SCHEMA: dict[str, Any] = {
    "$schema": _SCHEMA_DRAFT,
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "environment": {"type": "string", "enum": ["staging", "production"]},
        "canary": {"type": "boolean"},
        "replicas": {"type": "integer", "minimum": 1, "maximum": 6},
        "labels": {
            "type": "array",
            "items": {"type": "string", "enum": ["blue", "green", "red"]},
        },
        "notes": {"type": "string"},
    },
    "required": ["environment", "canary", "replicas", "labels"],
}

_FILE_TICKET_SCHEMA: dict[str, Any] = {
    "$schema": _SCHEMA_DRAFT,
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "severity": {"type": "string", "enum": ["p1", "p2", "p3", "p4"]},
        "escalate": {"type": "boolean"},
        "effort_points": {"type": "integer", "minimum": 0, "maximum": 8},
        "categories": {
            "type": "array",
            "items": {"type": "string", "enum": ["infra", "ui", "data"]},
        },
        "assignee": {"type": "string"},
    },
    "required": ["severity", "escalate", "effort_points", "categories"],
}

# (case_id, priority, duration_hours, send_invite)
_SCHEDULE_CASES: tuple[tuple[str, str, int, bool], ...] = (
    ("meeting-01", "urgent", 2, True),
    ("meeting-02", "low", 8, False),
    ("meeting-03", "normal", 1, True),
    ("meeting-04", "high", 4, False),
    ("meeting-05", "urgent", 5, False),
    ("meeting-06", "low", 3, True),
)

# (case_id, environment, canary, replicas, label-or-None)
_DEPLOY_CASES: tuple[tuple[str, str, bool, int, str | None], ...] = (
    ("deploy-01", "production", True, 4, "green"),
    ("deploy-02", "staging", False, 1, None),
    ("deploy-03", "production", False, 6, "blue"),
    ("deploy-04", "staging", True, 2, None),
    ("deploy-05", "production", True, 5, "red"),
    ("deploy-06", "staging", False, 3, "blue"),
)

# (case_id, severity, escalate, effort_points, category-or-None)
_TICKET_CASES: tuple[tuple[str, str, bool, int, str | None], ...] = (
    ("ticket-01", "p2", False, 5, "data"),
    ("ticket-02", "p1", True, 8, None),
    ("ticket-03", "p4", False, 0, "ui"),
    ("ticket-04", "p3", True, 3, "infra"),
    ("ticket-05", "p1", False, 6, "infra"),
    ("ticket-06", "p2", True, 1, None),
)


@dataclass(frozen=True)
class PilotCase:
    """One deterministic pilot case.

    Attributes:
        case_id: Stable id (``<family>-<ordinal>``) used for receipt joins
            and for fake-primitive dispatch in tests.
        family: Schema family label (one of the three tool names).
        tool: Tool name shown to the free-form arm and passed to
            ``tool_schema_to_questions``.
        parameters: The tool's parameters JSON schema. Cases in a family
            share the constant schema object and never mutate it.
        state: Task text that fully determines every mapped argument value.
        expected: The exact argument dict derivable from ``state``; it
            contains every mapped (required) argument and no skipped one.
    """

    case_id: str
    family: str
    tool: str
    parameters: dict[str, Any]
    state: str
    expected: dict[str, Any]


def _meeting_state(priority: str, hours: int, invite: bool) -> str:
    directive = "Send the invite." if invite else "Do not send the invite."
    unit = "hour" if hours == 1 else "hours"
    return f"Meeting request: priority {priority}; duration {hours} {unit}. {directive}"


def _deploy_state(environment: str, canary: bool, replicas: int, label: str | None) -> str:
    rollout = "Enable the canary rollout." if canary else "Disable the canary rollout."
    label_line = f"Attach the {label} label." if label is not None else "Attach no label."
    return f"Deploy request: environment {environment}; replicas {replicas}. {rollout} {label_line}"


def _ticket_state(severity: str, escalate: bool, effort_points: int, category: str | None) -> str:
    escalation = "Escalate the ticket." if escalate else "Do not escalate the ticket."
    category_line = f"Category {category}." if category is not None else "Category none."
    return (
        f"Ticket request: severity {severity}; effort points {effort_points}. "
        f"{escalation} {category_line}"
    )


def build_cases() -> list[PilotCase]:
    """Build the deterministic pilot catalogue: 3 tool schemas x 6 cases.

    The catalogue is a pure function of the module constants (no randomness,
    no clock, no environment), so two invocations address the same 18 cases
    in the same order and receipts are comparable across runs. The case
    docstrings in the module header explain what each family exercises.
    """
    cases: list[PilotCase] = []
    for case_id, priority, hours, invite in _SCHEDULE_CASES:
        cases.append(
            PilotCase(
                case_id=case_id,
                family="schedule_meeting",
                tool="schedule_meeting",
                parameters=_SCHEDULE_MEETING_SCHEMA,
                state=_meeting_state(priority, hours, invite),
                expected={
                    "priority": priority,
                    "send_invite": invite,
                    "duration_hours": hours,
                },
            )
        )
    for case_id, environment, canary, replicas, label in _DEPLOY_CASES:
        cases.append(
            PilotCase(
                case_id=case_id,
                family="deploy_service",
                tool="deploy_service",
                parameters=_DEPLOY_SERVICE_SCHEMA,
                state=_deploy_state(environment, canary, replicas, label),
                expected={
                    "environment": environment,
                    "canary": canary,
                    "replicas": replicas,
                    "labels": [label] if label is not None else [],
                },
            )
        )
    for case_id, severity, escalate, effort_points, category in _TICKET_CASES:
        cases.append(
            PilotCase(
                case_id=case_id,
                family="file_ticket",
                tool="file_ticket",
                parameters=_FILE_TICKET_SCHEMA,
                state=_ticket_state(severity, escalate, effort_points, category),
                expected={
                    "severity": severity,
                    "escalate": escalate,
                    "effort_points": effort_points,
                    "categories": [category] if category is not None else [],
                },
            )
        )
    return cases


def run_tool_args_pilot(
    primitives: Any,
    *,
    role: str,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run both arms over ``build_cases()`` and return the receipt dict.

    Args:
        primitives: The ``LLMPrimitives`` seam (anything exposing
            ``llm_call``). Required unless ``dry_run``.
        role: Registry role both arms' calls are charged to.
        receipt_path: Explicit receipt path; default is
            ``<artifacts_dir or tmp>/typed_decisions/tool_args_pilot-<stamp>.json``.
        artifacts_dir: Base directory for the default receipt path.
        dry_run: Return the case plan without calling the model and without
            writing a receipt (CLI ``--dry-run``).

    Returns:
        The receipt dict (``timestamp``, ``mode``, ``role``, ``counts``,
        ``results``, ``metric_directions``, ``prompt_sha256``) including the
        resolved ``receipt_path``.

    Raises:
        MeasurementError: no live primitives were supplied for a real run.
    """
    cases = build_cases()

    if dry_run:
        return {
            "study": "tool_args_pilot",
            "dry_run": True,
            "plan": {
                "role": role,
                "mode": "closed_set_vs_free_form",
                "arms": ["closed_set", "free_form"],
                "cases": [_case_plan(case) for case in cases],
                "counts": {
                    "cases": len(cases),
                    "tools": len({case.tool for case in cases}),
                },
            },
        }

    _require_primitives(primitives, "tool-args pilot")

    closed_records, closed_wall_ms = _run_closed_arm(primitives, cases, role)
    free_records, free_wall_ms = _run_free_arm(primitives, cases, role)
    expected_by_case = {case.case_id: case.expected for case in cases}
    closed_summary = _arm_summary(
        closed_records, expected_by_case, wall_ms=closed_wall_ms, decode="typed_json"
    )
    free_summary = _arm_summary(
        free_records, expected_by_case, wall_ms=free_wall_ms, decode="free_form_text"
    )
    compared, agreeing = _agreement(closed_records, free_records)

    receipt = {
        "study": "tool_args_pilot",
        "timestamp": _now_iso(),
        "mode": "closed_set_vs_free_form",
        "role": role,
        "counts": {
            "cases": len(cases),
            "tools": len({case.tool for case in cases}),
            "closed_set_resolved": closed_summary["resolved"],
            "closed_set_exact_match": closed_summary["exact_match"],
            "closed_set_failures": closed_summary["failures"],
            "free_form_resolved": free_summary["resolved"],
            "free_form_exact_match": free_summary["exact_match"],
            "free_form_failures": free_summary["failures"],
            "agreement_compared": compared,
            "agreement_agreeing": agreeing,
        },
        "results": {
            "cases": _merged_case_records(closed_records, free_records, expected_by_case),
            "arms": {
                "closed_set": closed_summary,
                "free_form": free_summary,
            },
            "agreement": {
                "compared": compared,
                "agreeing": agreeing,
                "rate": agreeing / compared if compared else None,
            },
        },
        "metric_directions": {
            "exact_match": "higher_better",
            "per_arg_exact_match": "higher_better",
            "agreement": "higher_better",
            "wall_ms": "lower_better",
        },
        "prompt_sha256": [
            record["prompt_sha256"]
            for record in closed_records + free_records
            if record["prompt_sha256"] is not None
        ],
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


def _case_plan(case: PilotCase) -> dict[str, Any]:
    mapping = tool_schema_to_questions(case.tool, case.parameters)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "tool": case.tool,
        "question_ids": [question.id for question in mapping.questions],
        "skipped": list(mapping.skipped),
        "expected": case.expected,
    }


# ── Arm A: closed-set typed decisions -> assemble_arguments ───────────────


def _run_closed_arm(
    primitives: Any,
    cases: Sequence[PilotCase],
    role: str,
) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        records.append(_run_closed_case(primitives, case, role))
    return records, (time.perf_counter() - started) * 1000.0


def _run_closed_case(primitives: Any, case: PilotCase, role: str) -> dict[str, Any]:
    mapping = tool_schema_to_questions(case.tool, case.parameters)
    record = _case_record(case, question_ids=[q.id for q in mapping.questions])
    record["skipped"] = list(mapping.skipped)
    started = time.perf_counter()
    try:
        result = run_typed_decisions(
            primitives,
            state=case.state,
            questions=mapping.questions,
            role=role,
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    else:
        record["failure_count"] = len(result.failures)
        record["prompt_sha256"] = result.prompt_sha256
        try:
            record["arguments"] = assemble_arguments(
                mapping.questions, result.decisions, case.parameters
            )
            record["resolved"] = True
        except ToolArgumentError as exc:
            record["error"] = str(exc)
    record["wall_ms"] = (time.perf_counter() - started) * 1000.0
    _attach_meta(record, primitives)
    return record


# ── Arm B: free-form JSON generation ──────────────────────────────────────


def _run_free_arm(
    primitives: Any,
    cases: Sequence[PilotCase],
    role: str,
) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        records.append(_run_free_case(primitives, case, role))
    return records, (time.perf_counter() - started) * 1000.0


def _run_free_case(primitives: Any, case: PilotCase, role: str) -> dict[str, Any]:
    prompt = _free_form_prompt(case)
    record = _case_record(case, question_ids=None)
    record["skipped"] = []
    record["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    started = time.perf_counter()
    try:
        raw_text = str(
            primitives.llm_call(
                prompt,
                role=role,
                n_tokens=_FREE_FORM_N_TOKENS,
                temperature=0.0,
                seed=_DECODE_SEED,
            )
            or ""
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    else:
        arguments, error = _parse_free_form_arguments(raw_text, case.parameters)
        record["arguments"] = arguments
        record["error"] = error
        record["resolved"] = error is None
    record["wall_ms"] = (time.perf_counter() - started) * 1000.0
    _attach_meta(record, primitives)
    return record


def _free_form_prompt(case: PilotCase) -> str:
    schema_text = json.dumps(case.parameters, indent=2, sort_keys=True)
    return (
        "Fill in the arguments for one tool call.\n\n"
        f"TOOL: {case.tool}\n\n"
        f"ARGUMENTS JSON SCHEMA (authoritative):\n{schema_text}\n\n"
        f"STATE:\n{case.state}\n\n"
        "Return EXACTLY ONE JSON object containing the tool arguments and "
        "nothing else (no prose, no markdown fences).\n"
    )


def _parse_free_form_arguments(
    raw_text: str,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract + JSON-schema-validate one free-form emission.

    The extractor is the JSON-mode runner's, so both arms face the same
    balanced-object parse contract. Returns ``(arguments, None)`` on success
    and ``(None, error)`` otherwise; every error string is receipt-visible.
    """
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        return None, "no balanced JSON object found"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"
    if not isinstance(parsed, dict):
        return None, f"top-level JSON is {type(parsed).__name__}, not an object"
    try:
        Draft202012Validator(parameters).validate(parsed)
    except ValidationError as exc:
        path = "$" + "".join(f".{part}" for part in exc.absolute_path)
        return None, f"schema violation at {path}: {exc.message}"
    except SchemaError as exc:
        return None, f"invalid tool schema: {exc.message}"
    return parsed, None


# ── Records, metrics, agreement ───────────────────────────────────────────


def _case_record(case: PilotCase, *, question_ids: list[str] | None) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "tool": case.tool,
        "question_ids": question_ids,
        "skipped": [],
        "resolved": False,
        "arguments": None,
        "error": None,
        "failure_count": 0,
        "prompt_sha256": None,
        "wall_ms": None,
        "tokens": None,
        "meta": None,
    }


def _attach_meta(record: dict[str, Any], primitives: Any) -> None:
    meta = _last_inference_meta(primitives)
    record["meta"] = meta
    if isinstance(meta, Mapping) and isinstance(meta.get("tokens"), (int, float)):
        record["tokens"] = float(meta["tokens"])


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _value_equal(left: Any, right: Any) -> bool:
    """Type-aware equality so ``True`` can never equal ``1`` in a receipt."""
    return type(left) is type(right) and _canonical(left) == _canonical(right)


def _arm_summary(
    records: Sequence[dict[str, Any]],
    expected_by_case: Mapping[str, dict[str, Any]],
    *,
    wall_ms: float,
    decode: str,
) -> dict[str, Any]:
    cases = len(records)
    resolved = 0
    exact = 0
    per_arg_matches = 0
    per_arg_total = 0
    tokens: list[float] = []
    per_case_ms: list[float] = []
    per_case_tokens: list[float | None] = []
    for record in records:
        expected = expected_by_case[record["case_id"]]
        per_arg_total += len(expected)
        per_case_ms.append(record["wall_ms"])
        per_case_tokens.append(record["tokens"])
        if record["tokens"] is not None:
            tokens.append(record["tokens"])
        if not record["resolved"]:
            continue
        resolved += 1
        arguments = record["arguments"]
        if _value_equal(arguments, expected):
            exact += 1
        per_arg_matches += sum(
            1
            for key, value in expected.items()
            if key in arguments and _value_equal(arguments[key], value)
        )
    return {
        "decode": decode,
        "cases": cases,
        "resolved": resolved,
        "failures": cases - resolved,
        "exact_match": exact,
        "exact_match_rate": exact / cases if cases else None,
        "per_arg_exact_match": per_arg_matches,
        "per_arg_total": per_arg_total,
        "per_arg_exact_match_rate": per_arg_matches / per_arg_total if per_arg_total else None,
        "wall_ms": wall_ms,
        "per_case_ms": per_case_ms,
        "tokens_generated": sum(tokens) if tokens else None,
        "calls_with_token_meta": len(tokens),
        "per_case_tokens": per_case_tokens,
    }


def _agreement(
    closed_records: Sequence[dict[str, Any]],
    free_records: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    compared = 0
    agreeing = 0
    for closed, free in zip(closed_records, free_records, strict=True):
        if not (closed["resolved"] and free["resolved"]):
            continue
        compared += 1
        if _value_equal(closed["arguments"], free["arguments"]):
            agreeing += 1
    return compared, agreeing


def _merged_case_records(
    closed_records: Sequence[dict[str, Any]],
    free_records: Sequence[dict[str, Any]],
    expected_by_case: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for closed, free in zip(closed_records, free_records, strict=True):
        expected = expected_by_case[closed["case_id"]]
        merged.append(
            {
                "case_id": closed["case_id"],
                "family": closed["family"],
                "tool": closed["tool"],
                "expected": expected,
                "closed_set": _arm_case_view(closed, expected),
                "free_form": _arm_case_view(free, expected),
            }
        )
    return merged


def _arm_case_view(record: dict[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "resolved": record["resolved"],
        "exact_match": bool(record["resolved"] and _value_equal(record["arguments"], expected)),
        "arguments": record["arguments"],
        "error": record["error"],
        "question_ids": record["question_ids"],
        "skipped": record["skipped"],
        "failure_count": record["failure_count"],
        "wall_ms": record["wall_ms"],
        "prompt_sha256": record["prompt_sha256"],
        "tokens": record["tokens"],
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.typed_decisions.tool_args_pilot",
        description=(
            "TD-4 live pilot: closed-set tool arguments vs free-form JSON over "
            "18 deterministic cases and 3 tool schemas. Real model calls require --live."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="allow real model calls; without this (or --dry-run) the pilot refuses to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the case plan without calling the model or writing a receipt",
    )
    parser.add_argument("--role", default="worker", help="registry role the calls are charged to")
    parser.add_argument(
        "--receipt",
        default=None,
        help=(
            "receipt path (default: <artifacts-dir or tmp>/typed_decisions/"
            "tool_args_pilot-<stamp>.json)"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="directory for the default receipt path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, 1 study error, 2 gate)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.live and not args.dry_run:
        print(
            "refusing to run: pass --live for real model calls or --dry-run to print the plan",
            file=sys.stderr,
        )
        return 2

    try:
        primitives = _live_primitives() if (args.live and not args.dry_run) else None
        receipt = run_tool_args_pilot(
            primitives,
            role=args.role,
            receipt_path=args.receipt,
            artifacts_dir=args.artifacts_dir,
            dry_run=args.dry_run,
        )
    except (MeasurementError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
