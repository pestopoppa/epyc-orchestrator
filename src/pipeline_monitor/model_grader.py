"""Model-graded subjective evaluations for pipeline diagnostics.

Loads grading specs from orchestration/grading_specs/*.yaml, evaluates
trigger conditions against diagnostic records, and calls worker_explore
via call_orchestrator_forced() for CoT classification.

This module runs post-hoc during seeding analysis (not inline during
live orchestration), keeping grading decoupled from the hot path.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SPECS_DIR: Path | None = None


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


def grade_answer(
    spec: dict[str, Any],
    diagnostic: dict[str, Any],
    orchestrator_url: str = "http://localhost:8000",
    timeout: int = 120,
) -> dict[str, Any] | None:
    """Grade an answer using the specified eval spec via worker_explore.

    Calls call_orchestrator_forced() from the seeding pipeline context.

    Returns:
        Dict with keys: spec_name, classification, score, reasoning.
        None if grading fails.
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
    judge_role = spec.get("judge_role", "worker_explore")
    judge_mode = spec.get("judge_mode", "direct")

    try:
        result = call_orchestrator_forced(
            prompt=prompt,
            force_role=judge_role,
            force_mode=judge_mode,
            url=orchestrator_url,
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("Model grading call failed for %s: %s", spec.get("spec_name"), e)
        return None

    answer_text = result.get("answer", "")
    choice_strings = spec.get("choice_strings", [])
    choice_scores = spec.get("choice_scores", {})

    # Extract classification letter from last line
    classification = _extract_classification(answer_text, choice_strings)
    score = choice_scores.get(classification, 0.0) if classification else None

    return {
        "spec_name": spec.get("spec_name", "unknown"),
        "classification": classification,
        "score": score,
        "reasoning": answer_text[:1000],
    }


def _extract_classification(text: str, choices: list[str]) -> str | None:
    """Extract classification letter from the last non-empty line of response."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
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
