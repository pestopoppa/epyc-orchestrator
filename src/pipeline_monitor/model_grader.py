"""Model-graded subjective evaluations for pipeline diagnostics.

Loads grading specs from orchestration/grading_specs/*.yaml, evaluates
trigger conditions against diagnostic records, and calls the live general
worker via call_orchestrator_forced() for CoT classification.

This module runs post-hoc during seeding analysis (not inline during
live orchestration), keeping grading decoupled from the hot path.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SPECS_DIR: Path | None = None

#: TD-21.23. Keyed by (spec_name, outcome) where outcome is "native" (the
#: schema-constrained reply parsed directly), "fished" (the schema missed but
#: the pre-existing last-line regex backstop recovered a choice), or
#: "parse_error" (neither recovered a choice -- today's silent
#: `classification=None` "ungraded" row). Mirrors the (site, status) shape of
#: `src.structured_output.repair.STRUCTURED_OUTPUT_REPAIR_COUNTS` without
#: sharing its counter: this site is a native-enum constraint, not a
#: fish-then-repair-turn conversion, so it is not a `parse_with_repair` call.
GRADER_CLASSIFICATION_OUTCOME_COUNTS: dict[tuple[str, str], int] = {}
_grader_outcome_lock = threading.Lock()


def _record_grader_outcome(spec_name: str, outcome: str) -> None:
    key = (spec_name, outcome)
    with _grader_outcome_lock:
        GRADER_CLASSIFICATION_OUTCOME_COUNTS[key] = GRADER_CLASSIFICATION_OUTCOME_COUNTS.get(key, 0) + 1
    if outcome == "parse_error":
        logger.warning(
            "Model grading reply unparseable for spec=%s; recorded as classification=parse_error "
            "instead of a silent ungraded row",
            spec_name,
        )


def reset_grader_outcome_counts_for_tests() -> None:
    """Test-only: clear the module-level counter between test cases."""
    with _grader_outcome_lock:
        GRADER_CLASSIFICATION_OUTCOME_COUNTS.clear()


def _get_specs_dir() -> Path:
    global _SPECS_DIR
    if _SPECS_DIR is None:
        _SPECS_DIR = Path(__file__).resolve().parents[2] / "orchestration" / "grading_specs"
    return _SPECS_DIR


def load_grading_specs(specs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load all grading spec YAML files from the specs directory.

    Returns:
        List of parsed spec dicts, each with an added 'spec_name' field.
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available — model grading disabled")
        return []

    directory = specs_dir or _get_specs_dir()
    if not directory.is_dir():
        logger.debug("Grading specs directory not found: %s", directory)
        return []

    specs = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            with open(path) as f:
                spec = yaml.safe_load(f)
            if spec:
                spec["spec_name"] = path.stem
                specs.append(spec)
        except Exception as e:
            logger.warning("Failed to load grading spec %s: %s", path.name, e)
    return specs


def should_trigger(spec: dict[str, Any], diagnostic: dict[str, Any]) -> bool:
    """Evaluate whether a grading spec's trigger condition is met.

    Supports simple field checks (no eval()):
    - field + equals: exact match
    - field + min_length: len(value) >= min_length
    - field + min_value: value >= min_value
    """
    trigger = spec.get("trigger", {})
    if not trigger:
        return True  # no trigger = always run

    field = trigger.get("field")
    if not field:
        return True

    value = diagnostic.get(field)

    if "equals" in trigger:
        return value == trigger["equals"]
    if "min_length" in trigger:
        try:
            return len(value or []) >= trigger["min_length"]
        except TypeError:
            return False
    if "min_value" in trigger:
        try:
            return (value or 0) >= trigger["min_value"]
        except TypeError:
            return False

    return True


def _format_prompt(spec: dict[str, Any], diagnostic: dict[str, Any]) -> str:
    """Fill the spec's prompt template with diagnostic fields."""
    template = spec.get("prompt_template", "")
    # Build substitution dict from diagnostic
    subs = {
        "question": diagnostic.get("question_id", ""),
        "expected": diagnostic.get("expected", ""),
        "answer": diagnostic.get("answer", "")[:3000],  # truncate long answers
        "scoring_method": diagnostic.get("scoring_method", ""),
        "passed": "pass" if diagnostic.get("passed") else "fail",
        "role_history": " → ".join(diagnostic.get("role_history", [])),
        "elapsed_s": f"{diagnostic.get('elapsed_s', 0):.1f}",
    }
    # Safe format — ignore missing keys
    try:
        return template.format(**subs)
    except KeyError:
        return template


def _classification_schema(choices: list[str]) -> dict[str, Any]:
    """TD-21.23: constrain the grader's reply to exactly one of the spec's
    own ``choice_strings`` — a json-schema ``enum``, not a regex, is what
    makes an off-set value structurally impossible to receive. `/chat`
    already forwards ``output_schema`` for a schema-constrained call
    (``direct_stage.py:133``; TD-21.0 widened forwarding to the `/v1` lane
    too, so this reaches the wire regardless of which lane ``judge_role``
    resolves to)."""
    return {
        "type": "object",
        "properties": {"classification": {"type": "string", "enum": list(choices)}},
        "required": ["classification"],
        "additionalProperties": False,
    }


def _native_classification(answer_text: str, choices: list[str]) -> str | None:
    """Parse the schema-constrained ``{"classification": <choice>}`` reply.

    Returns ``None`` — never a guess — when the reply is not that shape, or
    the value is not one of the spec's own closed ``choices``. The caller
    falls back to ``_extract_classification`` (the pre-existing last-line
    regex) only on a ``None`` here.
    """
    if not choices:
        return None
    try:
        value = json.loads(answer_text.strip())
    except (json.JSONDecodeError, AttributeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    classification = value.get("classification")
    return classification if classification in choices else None


def grade_answer(
    spec: dict[str, Any],
    diagnostic: dict[str, Any],
    orchestrator_url: str = "http://localhost:8000",
    timeout: int = 120,
) -> dict[str, Any] | None:
    """Grade an answer using the specified eval spec via the general worker.

    Calls call_orchestrator_forced() from the seeding pipeline context.

    TD-21.23: the call is schema-constrained to the spec's own
    ``choice_strings`` (``_classification_schema``); the reply is parsed
    natively first (``_native_classification``), falling back to the
    pre-existing last-line regex fish (``_extract_classification``) as a
    backstop. When NEITHER recovers a choice, the row is no longer silently
    "ungraded": ``classification`` is the typed sentinel ``"parse_error"``
    (``score`` stays ``None``, matching the old None-classification's score),
    and the outcome is counted in ``GRADER_CLASSIFICATION_OUTCOME_COUNTS``.

    Returns:
        Dict with keys: spec_name, classification, score, reasoning.
        None if the grading CALL itself fails (transport/import) — a
        transport failure is not a graded row at all, unlike a parse failure.
    """
    import sys
    # Add benchmark scripts to path for seeding_orchestrator import
    bench_dir = str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark")
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)

    try:
        from seeding_orchestrator import call_orchestrator_forced
    except ImportError:
        logger.warning("seeding_orchestrator not importable — model grading unavailable")
        return None

    prompt = _format_prompt(spec, diagnostic)
    judge_role = spec.get("judge_role", "worker_general")
    judge_mode = spec.get("judge_mode", "direct")
    choice_strings = spec.get("choice_strings", [])
    choice_scores = spec.get("choice_scores", {})
    spec_name = spec.get("spec_name", "unknown")

    output_schema = _classification_schema(choice_strings) if choice_strings else None

    try:
        result = call_orchestrator_forced(
            prompt=prompt,
            force_role=judge_role,
            force_mode=judge_mode,
            url=orchestrator_url,
            timeout=timeout,
            output_schema=output_schema,
        )
    except Exception as e:
        logger.warning("Model grading call failed for %s: %s", spec_name, e)
        return None

    answer_text = result.get("answer", "")

    classification = _native_classification(answer_text, choice_strings)
    if classification is not None:
        outcome = "native"
    else:
        classification = _extract_classification(answer_text, choice_strings)
        outcome = "fished" if classification is not None else "parse_error"

    _record_grader_outcome(spec_name, outcome)

    if outcome == "parse_error":
        return {
            "spec_name": spec_name,
            "classification": "parse_error",
            "score": None,
            "reasoning": answer_text[:1000],
        }

    return {
        "spec_name": spec_name,
        "classification": classification,
        "score": choice_scores.get(classification, 0.0),
        "reasoning": answer_text[:1000],
    }


def _extract_classification(text: str, choices: list[str]) -> str | None:
    """Extract classification letter from the last non-empty line of response.

    TD-21.23: kept unchanged as the backstop fish for when the native
    schema-constrained parse (``_native_classification``) misses — e.g. a
    role/backend that does not honour ``output_schema`` at all.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None

    last_line = lines[-1]
    # Match a single letter that's in our choice set
    for choice in choices:
        if re.search(rf"\b{re.escape(choice)}\b", last_line):
            return choice
    return None


def grade_diagnostic_batch(
    diagnostics: list[dict[str, Any]],
    specs: list[dict[str, Any]] | None = None,
    sample_rate: float = 0.1,
    orchestrator_url: str = "http://localhost:8000",
) -> dict[str, list[dict[str, Any]]]:
    """Grade a batch of diagnostics with all applicable specs.

    Args:
        diagnostics: List of diagnostic records.
        specs: Grading specs (loaded from YAML if None).
        sample_rate: Fraction of eligible diagnostics to actually grade.
        orchestrator_url: Orchestrator API URL.

    Returns:
        Dict mapping question_id to list of grading results.
    """
    if specs is None:
        specs = load_grading_specs()
    if not specs:
        return {}

    results: dict[str, list[dict[str, Any]]] = {}

    for diag in diagnostics:
        qid = diag.get("question_id", "unknown")

        for spec in specs:
            if not should_trigger(spec, diag):
                continue
            # Random sampling to control grading budget
            if random.random() > sample_rate:
                continue

            result = grade_answer(spec, diag, orchestrator_url=orchestrator_url)
            if result:
                results.setdefault(qid, []).append(result)

    return results
