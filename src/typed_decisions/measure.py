"""Measurement harnesses for the typed decision plane (TD-2 / TD-3).

Three offline-testable studies, each driving the same ``llm_call`` seam the
TD-1 runner drives and each writing a JSON receipt:

* ``run_contamination_study`` (TD-2) — asks one question catalogue under
  several deterministic question ORDERINGS and reports per-question answer
  flips against the canonical order. Rationale: intake-1486 measured a 24.7%
  cross-question contamination rate for a batched vendor call; the number
  that matters for this stack is OUR number, measured on OUR runner.
* ``run_calibration_study`` (TD-2) — compares the local confidence statistic
  from ``Decision`` against labeled outcomes: ECE, Brier and reliability
  bins. The ECE binning and metric are reused from
  ``src.llm_primitives.stat_tests``; this module does not reimplement that
  math. (Brier is deliberately not in ``stat_tests`` — each call site owns
  its one-line Brier — so the one-line mean squared error lives here.)
* ``run_fanout_study`` (TD-3) — measures the batching multiplier directly:
  a probe catalogue answered in one batched call per state vs one singleton
  call per question, reporting serial-summed and per-call wall times, token
  counts from ``_last_inference_meta`` where the primitives object exposes
  them, and answer agreement between the two arms. Rationale: TD-3 replaces
  a vendor's advertised batching multiplier with our own measurement.
  By default the catalogue is a content-neutral GENERATED probe (a transport
  measurement); pass ``questions=`` to ask a real catalogue under every
  state, which is what makes the agreement metric interpretable. The receipt
  records the choice as ``probe_source``. Live fixture:
  ``/mnt/raid0/llm/worktrees/intake-jev-sageattn-20260917/artifacts/typed_decisions/decision_set_v1/state.json``
  and ``.../questions.json`` (referenced, never copied).

Contract:

    * Every study fails loudly with ``MeasurementError`` when a live
      primitives object is unavailable or when a run yields no usable
      decisions. A receipt NEVER carries fabricated numbers.
    * Every study writes its receipt to the caller-supplied ``receipt_path``
      (default: ``<artifacts_dir or system tmp>/typed_decisions/
      <study>-<utc-stamp>.json``) and returns the receipt dict, which
      includes the resolved ``receipt_path``.
    * ``dry_run=True`` returns a plan without calling the model and without
      writing a receipt. This is the path behind the CLI ``--dry-run``.
    * Receipts carry ``timestamp``, ``mode``, ``role``, ``counts``,
      ``results`` and the ``prompt_sha256`` values of every
      ``DecisionResult`` produced (runner.prompt_sha256, i.e. the SHA-256 of
      the canonical first-attempt prompt).

The module is import-safe offline: no network or server object is touched at
import time. The CLI (``python -m src.typed_decisions.measure``) refuses to
make real calls unless ``--live`` is passed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm_primitives.stat_tests import expected_calibration_error
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    Question,
    QuestionKind,
)

__all__ = [
    "MeasurementError",
    "main",
    "run_calibration_study",
    "run_contamination_study",
    "run_fanout_study",
]

# Equal-width reliability bins; the same count ``expected_calibration_error``
# defaults to, so the reported bins explain the reported ECE exactly.
_ECE_BINS = 10

# Fan-out probe catalogue: the study measures transport overhead, so the
# questions are deliberately generic and content-neutral. Ids are stable for
# a given ``questions_per_state`` so two invocations are comparable.
_PROBE_ID_PREFIX = "fanout"


class MeasurementError(RuntimeError):
    """A study could not produce a real measurement.

    Raised when no live primitives object is available, when the measurement
    design is undefined (e.g. fewer than two questions for contamination),
    or when a run yields no usable decisions. Studies raise this instead of
    returning zeros: a fabricated number is worse than no receipt.
    """


# ── Shared helpers ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_primitives(primitives: Any, study: str) -> None:
    """Fail loudly when the caller has no live primitives to measure."""
    if primitives is None or not callable(getattr(primitives, "llm_call", None)):
        got = "None" if primitives is None else type(primitives).__name__
        raise MeasurementError(
            f"{study} study requires a live primitives object exposing llm_call(...); "
            f"got {got}. Measurement harnesses never fabricate numbers."
        )


def _validate_unique_ids(questions: Sequence[Question]) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"duplicate question id: {question.id!r}")
        seen.add(question.id)


def _same_value(left: object, right: object) -> bool:
    """Type-aware equality (``True`` must not equal ``1`` in a receipt)."""
    return type(left) is type(right) and left == right


def _prompt_hashes(results: Sequence[DecisionResult]) -> list[str]:
    return [result.prompt_sha256 for result in results]


def _write_receipt(
    receipt: dict[str, Any],
    *,
    receipt_path: str | Path | None,
    artifacts_dir: str | Path | None,
) -> Path:
    if receipt_path is not None:
        path = Path(receipt_path)
    else:
        base = (
            Path(artifacts_dir)
            if artifacts_dir is not None
            else Path(tempfile.gettempdir()) / "typed_decisions"
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = base / f"{receipt['study']}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt["receipt_path"] = str(path)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _plan_orderings(
    questions: Sequence[Question],
    count: int,
    seed: int,
) -> list[list[Question]]:
    """Canonical order first, then distinct deterministic permutations.

    Permutations come from ``random.Random(seed).random()`` sort keys rather
    than ``shuffle`` so the sequence is reproducible across Python runtimes
    for a fixed seed. If the question count is too small to yield ``count``
    distinct orderings (e.g. two questions have only two orderings), fewer
    orderings are returned — never duplicates.
    """
    canonical = list(questions)
    orderings = [canonical]
    seen = {tuple(question.id for question in canonical)}
    rng = random.Random(seed)
    attempts = 0
    max_attempts = max(1, count) * 50
    while len(orderings) < count and attempts < max_attempts:
        attempts += 1
        indices = sorted(range(len(canonical)), key=lambda _: rng.random())
        candidate = [canonical[index] for index in indices]
        key = tuple(question.id for question in candidate)
        if key in seen:
            continue
        seen.add(key)
        orderings.append(candidate)
    return orderings


# ── TD-2: contamination ───────────────────────────────────────────────────


def run_contamination_study(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    order_permutations: int = 4,
    seed: int = 0,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Measure cross-question contamination via question-order permutations.

    The canonical order is asked first; every other ordering is a
    deterministic permutation of the same catalogue (same state, same role,
    same runner). A per-question TOP-ANSWER FLIP is a question whose resolved
    value differs between the canonical order and a permuted order.

    ``order_permutations`` is the TOTAL number of orderings including the
    canonical one, so it must be >= 2. A question that resolves in the
    canonical order but not in a permuted order counts as an
    ``unresolved_pair``, never as a flip; the flip rate denominator is the
    number of pairs resolved in both orders.

    Answers are compared with type-aware equality, so a ``true``/``false``
    answer cannot silently equal ``1``/``0``.

    Returns the receipt dict (see module docstring) and writes it to
    ``receipt_path`` / the default artifacts location.
    """
    questions = list(questions)
    if order_permutations < 2:
        raise ValueError(
            "order_permutations must be >= 2 (the canonical order plus at least one permutation)"
        )
    if len(questions) < 2:
        raise MeasurementError("contamination is undefined for fewer than 2 questions")
    _validate_unique_ids(questions)
    orderings = _plan_orderings(questions, order_permutations, seed)

    if dry_run:
        return {
            "study": "contamination",
            "dry_run": True,
            "plan": {
                "mode": "json",
                "role": role,
                "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
                "questions": [question.id for question in questions],
                "orderings": [[question.id for question in ordering] for ordering in orderings],
                "order_permutations_requested": order_permutations,
                "orderings_planned": len(orderings),
                "seed": seed,
            },
        }

    _require_primitives(primitives, "contamination")
    runs: list[dict[str, Any]] = []
    results: list[DecisionResult] = []
    for index, ordering in enumerate(orderings):
        result = run_typed_decisions(
            primitives,
            state=state,
            questions=ordering,
            role=role,
        )
        results.append(result)
        runs.append(_contamination_run_record(index, ordering, result))

    canonical = runs[0]
    canonical_values: Mapping[str, Any] = canonical["resolved"]
    per_question: list[dict[str, Any]] = []
    flips = 0
    comparable = 0
    unresolved = 0
    canonical_unresolved: list[str] = []

    for question in questions:
        question_id = question.id
        if question_id not in canonical_values:
            canonical_unresolved.append(question_id)
            per_question.append(
                {
                    "question_id": question_id,
                    "canonical_resolved": False,
                    "flips": 0,
                    "compared": 0,
                    "flip_rate": None,
                    "permuted_values": [run["resolved"].get(question_id) for run in runs[1:]],
                }
            )
            continue
        question_flips = 0
        question_compared = 0
        permuted_values: list[Any] = []
        for run in runs[1:]:
            if question_id not in run["resolved"]:
                unresolved += 1
                permuted_values.append(None)
                continue
            value = run["resolved"][question_id]
            permuted_values.append(value)
            question_compared += 1
            if not _same_value(canonical_values[question_id], value):
                question_flips += 1
        comparable += question_compared
        flips += question_flips
        per_question.append(
            {
                "question_id": question_id,
                "canonical_resolved": True,
                "canonical_value": canonical_values[question_id],
                "flips": question_flips,
                "compared": question_compared,
                "flip_rate": (question_flips / question_compared if question_compared else None),
                "permuted_values": permuted_values,
            }
        )

    if comparable == 0:
        raise MeasurementError(
            "contamination study produced no question pair resolved in both the "
            "canonical order and at least one permuted order"
        )

    receipt = {
        "study": "contamination",
        "timestamp": _now_iso(),
        "mode": "json",
        "role": role,
        "counts": {
            "questions": len(questions),
            "orderings": len(runs),
            "comparable_pairs": comparable,
            "flips": flips,
            "unresolved_pairs": unresolved,
            "canonical_unresolved": len(canonical_unresolved),
        },
        "results": {
            "canonical": canonical,
            "permutations": runs[1:],
            "per_question": per_question,
            "flip_rate": flips / comparable,
            "canonical_unresolved_ids": canonical_unresolved,
        },
        "metric_directions": {"flip_rate": "lower_better"},
        "prompt_sha256": _prompt_hashes(results),
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


def _contamination_run_record(
    order_index: int,
    questions: Sequence[Question],
    result: DecisionResult,
) -> dict[str, Any]:
    return {
        "order_index": order_index,
        "order": [question.id for question in questions],
        "is_canonical": order_index == 0,
        "mode": result.mode,
        "elapsed_ms": result.elapsed_ms,
        "prompt_sha256": result.prompt_sha256,
        "resolved": {decision.question_id: decision.value for decision in result.decisions},
        "failure_count": len(result.failures),
        "failures": [
            {"reason": failure.reason, "detail": failure.detail} for failure in result.failures
        ],
    }


# ── TD-2: calibration ─────────────────────────────────────────────────────


def run_calibration_study(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    labels: Mapping[str, object],
    role: str,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Calibrate the local confidence statistic against labeled outcomes.

    ``labels`` maps ``question_id -> correct value`` in the question's own
    typed domain: the option label for a choice question, the integer level
    for a score question, ``True``/``False`` for a noul question. (The
    alternative bool ``correct`` map was rejected because it cannot
    distinguish "wrong" from "unlabeled".) Correctness is type-aware equality
    with the resolved decision value.

    The catalogue is asked in ONE batched pass — this study calibrates the
    batched plane, not a per-question idealization; run the contamination
    study alongside it to quantify cross-question effects.

    Metrics: ``ece`` via ``stat_tests.expected_calibration_error`` (10
    equal-width bins, final bin closed), ``brier`` as mean squared error
    between confidence and the 0/1 outcome, plus per-bin reliability
    (count, mean confidence, accuracy) using the identical bin boundaries.
    """
    questions = list(questions)
    label_map = dict(labels)
    _validate_unique_ids(questions)
    question_ids = {question.id for question in questions}
    matched_labels = {qid for qid in label_map if qid in question_ids}
    unknown_labels = sorted(set(label_map) - question_ids)

    if dry_run:
        return {
            "study": "calibration",
            "dry_run": True,
            "plan": {
                "mode": "json",
                "role": role,
                "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
                "questions": [question.id for question in questions],
                "labeled_questions": sorted(matched_labels),
                "unlabeled_questions": sorted(question_ids - matched_labels),
                "unknown_labels": unknown_labels,
                "n_bins": _ECE_BINS,
            },
        }

    _require_primitives(primitives, "calibration")
    result = run_typed_decisions(
        primitives,
        state=state,
        questions=questions,
        role=role,
    )
    decisions: dict[str, Decision] = {
        decision.question_id: decision for decision in result.decisions
    }

    rows: list[dict[str, Any]] = []
    for question in questions:
        decision = decisions.get(question.id)
        if decision is None:
            continue
        correct: bool | None = None
        if question.id in label_map:
            correct = _same_value(decision.value, label_map[question.id])
        rows.append(
            {
                "question_id": question.id,
                "kind": decision.kind.value,
                "value": decision.value,
                "confidence": decision.confidence,
                "correct": correct,
            }
        )

    scored_rows = [row for row in rows if row["correct"] is not None]
    if not scored_rows:
        raise MeasurementError(
            "calibration study produced no resolved decision with a matching label; "
            "ECE/Brier would be fabricated"
        )
    confidences = [float(row["confidence"]) for row in scored_rows]
    outcomes = [1.0 if row["correct"] else 0.0 for row in scored_rows]
    ece = expected_calibration_error(confidences, outcomes, n_bins=_ECE_BINS)
    brier = sum(
        (confidence - outcome) ** 2 for confidence, outcome in zip(confidences, outcomes)
    ) / len(scored_rows)
    accuracy = sum(outcomes) / len(outcomes)
    mean_confidence = sum(confidences) / len(confidences)

    receipt = {
        "study": "calibration",
        "timestamp": _now_iso(),
        "mode": "json",
        "role": role,
        "counts": {
            "questions": len(questions),
            "resolved": len(decisions),
            "labeled": len(matched_labels),
            "scored": len(scored_rows),
            "unlabeled": len(rows) - len(scored_rows),
            "unresolved": len(questions) - len(decisions),
            "unknown_labels": len(unknown_labels),
            "failures": len(result.failures),
        },
        "results": {
            "rows": rows,
            "metrics": {
                "n": len(scored_rows),
                "ece": ece,
                "brier": brier,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "n_bins": _ECE_BINS,
                "metric_direction": {
                    "ece": "lower_better",
                    "brier": "lower_better",
                    "accuracy": "higher_better",
                },
            },
            "reliability_bins": _reliability_bins(confidences, outcomes, n_bins=_ECE_BINS),
            "unknown_label_ids": unknown_labels,
            "failures": [
                {"reason": failure.reason, "detail": failure.detail} for failure in result.failures
            ],
            "elapsed_ms": result.elapsed_ms,
        },
        "metric_directions": {
            "ece": "lower_better",
            "brier": "lower_better",
            "accuracy": "higher_better",
        },
        "prompt_sha256": _prompt_hashes([result]),
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


def _reliability_bins(
    confidences: Sequence[float],
    outcomes: Sequence[float],
    *,
    n_bins: int,
) -> list[dict[str, Any]]:
    """Per-bin reliability using ``expected_calibration_error``'s boundaries.

    Equal-width ``[i/n, (i+1)/n)`` bins, the last one closed on the right, so
    the reported bins are exactly the terms summed by the reported ECE.
    Empty bins carry ``None`` means, never ``0.0`` — absence is not accuracy.
    """
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if index < n_bins - 1:
            members = [
                position
                for position, confidence in enumerate(confidences)
                if lower <= confidence < upper
            ]
        else:
            members = [
                position
                for position, confidence in enumerate(confidences)
                if lower <= confidence <= upper
            ]
        count = len(members)
        bins.append(
            {
                "bin_index": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": (
                    sum(confidences[position] for position in members) / count if count else None
                ),
                "accuracy": (
                    sum(outcomes[position] for position in members) / count if count else None
                ),
            }
        )
    return bins


# ── TD-3: fan-out ─────────────────────────────────────────────────────────


def run_fanout_study(
    primitives: Any,
    *,
    states: Sequence[str],
    role: str,
    questions_per_state: int | None = None,
    mode: str = "json",
    questions: Sequence[Question] | None = None,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Measure batched-vs-singleton cost and answer agreement.

    Arm A (batched): one ``run_typed_decisions`` call per state answering the
    whole catalogue. Arm B (singleton): one call per (state, question) pair.

    ``questions`` selects the catalogue:

    * ``None`` (default) — the generated content-neutral probe: the study
      measures TRANSPORT/BATCHING overhead, not semantics, and its ids are
      stable for a given ``questions_per_state``. Agreement over such a probe
      is dominated by the model's position/batching sensitivity, not by the
      state, so read it as a stability statistic.
    * a provided catalogue — every state is asked that IDENTICAL catalogue
      (ids, text, options/levels and criteria verbatim). This is the
      interpretable mode: agreement answers whether batching changes the
      model's answer to a real question about a real state. The receipt
      records ``probe_source: "provided"`` (``"generated"`` on the default
      path). When ``questions`` is provided, ``questions_per_state`` may be
      omitted (it is derived from the catalogue) but if given must equal
      ``len(questions)``.

    Live catalogue fixture (reference it do not copy):
    ``/mnt/raid0/llm/worktrees/intake-jev-sageattn-20260917/artifacts/typed_decisions/decision_set_v1/questions.json``
    with the matching state fixture ``.../state.json``.

    Reports per arm: call count, wall time, serial sum of per-call
    ``DecisionResult.elapsed_ms``, the per-call list, and generated-token
    counts read from ``_last_inference_meta`` when the primitives object
    exposes them (``None``, not zero, when it does not). Agreement compares
    the resolved value of each (state, question) pair across arms with
    type-aware equality; pairs missing on either side are counted as
    unresolved, not as agreements.
    """
    states = list(states)
    if not states:
        raise MeasurementError("fanout study requires at least one state")
    if questions is not None:
        catalogue = list(questions)
        _validate_unique_ids(catalogue)
        if not catalogue:
            raise ValueError("questions must contain at least one Question")
        if questions_per_state is None:
            questions_per_state = len(catalogue)
        elif questions_per_state != len(catalogue):
            raise ValueError(
                "questions_per_state must equal the provided catalogue length "
                f"({len(catalogue)}), got {questions_per_state}"
            )
        probe_source = "provided"
    else:
        if questions_per_state is None or questions_per_state < 1:
            raise ValueError(
                "questions_per_state must be >= 1 when no questions catalogue is provided"
            )
        catalogue = _fanout_probe_questions(questions_per_state)
        probe_source = "generated"

    if dry_run:
        return {
            "study": "fanout",
            "dry_run": True,
            "probe_source": probe_source,
            "plan": {
                "mode": mode,
                "role": role,
                "probe_source": probe_source,
                "states": len(states),
                "questions": [question.id for question in catalogue],
                "questions_per_state": questions_per_state,
                "batched_calls": len(states),
                "singleton_calls": len(states) * questions_per_state,
            },
        }

    _require_primitives(primitives, "fanout")

    batched_records: list[dict[str, Any]] = []
    batched_results: list[DecisionResult] = []
    batched_started = time.perf_counter()
    for state_index, state in enumerate(states):
        result = run_typed_decisions(
            primitives,
            state=state,
            questions=catalogue,
            role=role,
            mode=mode,
        )
        batched_results.append(result)
        batched_records.append(_fanout_run_record(state_index, catalogue, result, primitives))
    batched_wall_ms = (time.perf_counter() - batched_started) * 1000.0

    singleton_records: list[dict[str, Any]] = []
    singleton_results: list[DecisionResult] = []
    singleton_started = time.perf_counter()
    for state_index, state in enumerate(states):
        for question in catalogue:
            result = run_typed_decisions(
                primitives,
                state=state,
                questions=[question],
                role=role,
                mode=mode,
            )
            singleton_results.append(result)
            singleton_records.append(
                _fanout_run_record(state_index, [question], result, primitives)
            )
    singleton_wall_ms = (time.perf_counter() - singleton_started) * 1000.0

    disagreements: list[dict[str, Any]] = []
    per_question_agreement: dict[str, dict[str, int]] = {
        question.id: {"compared": 0, "agreeing": 0} for question in catalogue
    }
    comparable = 0
    agreeing = 0
    unresolved_pairs = 0
    for state_index in range(len(states)):
        batched_values = batched_records[state_index]["resolved"]
        for question_index, question in enumerate(catalogue):
            singleton_values = singleton_records[
                state_index * questions_per_state + question_index
            ]["resolved"]
            if question.id not in batched_values or question.id not in singleton_values:
                unresolved_pairs += 1
                continue
            comparable += 1
            per_question_agreement[question.id]["compared"] += 1
            batched_value = batched_values[question.id]
            singleton_value = singleton_values[question.id]
            if _same_value(batched_value, singleton_value):
                agreeing += 1
                per_question_agreement[question.id]["agreeing"] += 1
            else:
                disagreements.append(
                    {
                        "state_index": state_index,
                        "question_id": question.id,
                        "batched_value": batched_value,
                        "singleton_value": singleton_value,
                    }
                )

    if comparable == 0:
        raise MeasurementError(
            "fanout study produced no (state, question) pair resolved in both arms"
        )

    batched_summary = _fanout_arm_summary(batched_records, batched_wall_ms)
    singleton_summary = _fanout_arm_summary(singleton_records, singleton_wall_ms)
    batched_serial = batched_summary["serial_sum_ms"]
    singleton_serial = singleton_summary["serial_sum_ms"]

    receipt = {
        "study": "fanout",
        "probe_source": probe_source,
        "timestamp": _now_iso(),
        "mode": mode,
        "role": role,
        "counts": {
            "states": len(states),
            "questions_per_state": questions_per_state,
            "batched_calls": len(batched_records),
            "singleton_calls": len(singleton_records),
            "comparable_pairs": comparable,
            "agreeing_pairs": agreeing,
            "disagreements": len(disagreements),
            "unresolved_pairs": unresolved_pairs,
        },
        "results": {
            "questions": [question.id for question in catalogue],
            "batched": batched_summary,
            "singleton": singleton_summary,
            "agreement_rate": agreeing / comparable,
            "per_question_agreement": per_question_agreement,
            "disagreements": disagreements,
            "batched_speedup_serial": (
                singleton_serial / batched_serial if batched_serial > 0.0 else None
            ),
            "batched_speedup_wall": (
                singleton_wall_ms / batched_wall_ms if batched_wall_ms > 0.0 else None
            ),
            "metric_directions": {
                "agreement_rate": "higher_better",
                "batched_speedup_serial": "higher_better",
                "batched_speedup_wall": "higher_better",
            },
        },
        "prompt_sha256": _prompt_hashes(batched_results + singleton_results),
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


def _fanout_probe_questions(count: int) -> list[Question]:
    """Deterministic, content-neutral probe catalogue for one state.

    A repeating choice/score/noul mix exercises all three answer shapes and
    roughly the per-question output volume of a real catalogue. The ids are
    ``fanout-000`` … ``fanout-<count-1>`` regardless of the state, so
    agreement is comparable across states.
    """
    criteria = ("synthetic fan-out probe; the study measures batching overhead, not semantics",)
    questions: list[Question] = []
    for index in range(count):
        slot = index % 3
        question_id = f"{_PROBE_ID_PREFIX}-{index:03d}"
        if slot == 0:
            questions.append(
                Question(
                    id=question_id,
                    kind=QuestionKind.NOUL,
                    text=f"Fan-out probe {index}: is invariant {index} satisfied?",
                    criteria=criteria,
                )
            )
        elif slot == 1:
            questions.append(
                Question(
                    id=question_id,
                    kind=QuestionKind.CHOICE,
                    text=f"Fan-out probe {index}: which branch applies to change {index}?",
                    options=("hold", "ship"),
                    criteria=criteria,
                )
            )
        else:
            questions.append(
                Question(
                    id=question_id,
                    kind=QuestionKind.SCORE,
                    text=f"Fan-out probe {index}: rate the risk of change {index}.",
                    levels=(0, 1, 2, 3),
                    criteria=criteria,
                )
            )
    return questions


def _fanout_run_record(
    state_index: int,
    questions: Sequence[Question],
    result: DecisionResult,
    primitives: Any,
) -> dict[str, Any]:
    return {
        "state_index": state_index,
        "question_ids": [question.id for question in questions],
        "mode": result.mode,
        "elapsed_ms": result.elapsed_ms,
        "prompt_sha256": result.prompt_sha256,
        "resolved": {decision.question_id: decision.value for decision in result.decisions},
        "failure_count": len(result.failures),
        "meta": _last_inference_meta(primitives),
    }


def _last_inference_meta(primitives: Any) -> dict[str, Any] | None:
    meta = getattr(primitives, "_last_inference_meta", None)
    if not isinstance(meta, Mapping):
        return None
    return {str(key): value for key, value in meta.items()}


def _fanout_arm_summary(
    records: Sequence[dict[str, Any]],
    wall_ms: float,
) -> dict[str, Any]:
    per_call = [record["elapsed_ms"] for record in records]
    tokens: list[float] = []
    for record in records:
        meta = record["meta"]
        if isinstance(meta, Mapping) and isinstance(meta.get("tokens"), (int, float)):
            tokens.append(float(meta["tokens"]))
    return {
        "calls": len(records),
        "wall_ms": wall_ms,
        "serial_sum_ms": sum(per_call),
        "per_call_ms": per_call,
        "tokens_generated": sum(tokens) if tokens else None,
        "calls_with_token_meta": len(tokens),
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="allow real model calls; without this (or --dry-run) the study refuses to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the study plan without calling the model or writing a receipt",
    )
    parser.add_argument("--role", default="worker", help="registry role the calls are charged to")
    parser.add_argument(
        "--receipt",
        default=None,
        help="receipt path (default: <artifacts-dir or tmp>/typed_decisions/<study>-<stamp>.json)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="directory for the default receipt path",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.typed_decisions.measure",
        description=(
            "Typed-decision measurement harnesses (TD-2 contamination/calibration, "
            "TD-3 fan-out). Real model calls require --live."
        ),
    )
    subparsers = parser.add_subparsers(dest="study", required=True)

    contamination = subparsers.add_parser(
        "contamination",
        help="question-order contamination study (TD-2)",
    )
    _add_common_options(contamination)
    contamination.add_argument(
        "--state-file", required=True, help="JSON file with the state string"
    )
    contamination.add_argument("--questions-file", required=True, help="JSON catalogue file")
    contamination.add_argument("--order-permutations", type=int, default=4)
    contamination.add_argument("--seed", type=int, default=0)

    calibration = subparsers.add_parser(
        "calibration",
        help="confidence calibration study (TD-2)",
    )
    _add_common_options(calibration)
    calibration.add_argument("--state-file", required=True, help="JSON file with the state string")
    calibration.add_argument("--questions-file", required=True, help="JSON catalogue file")
    calibration.add_argument(
        "--labels-file",
        required=True,
        help="JSON object mapping question_id -> correct typed value",
    )

    fanout = subparsers.add_parser(
        "fanout",
        help="batched-vs-singleton fan-out study (TD-3)",
    )
    _add_common_options(fanout)
    fanout.add_argument(
        "--states-file",
        required=True,
        help="JSON list of state strings (a single string or {'state': str} is one state)",
    )
    fanout.add_argument(
        "--questions-per-state",
        type=int,
        default=None,
        help="generated probe size; required unless --questions-file is given",
    )
    fanout.add_argument(
        "--questions-file",
        default=None,
        help=(
            "JSON catalogue used verbatim for every state (overrides the generated "
            "probe); the decision_set_v1 fixture lives at "
            ".../intake-jev-sageattn-20260917/artifacts/typed_decisions/decision_set_v1/questions.json"
        ),
    )
    fanout.add_argument("--mode", default="json", help="runner mode (json | native)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, 1 study error, 2 gate)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.study == "fanout" and args.questions_file is None and args.questions_per_state is None:
        parser.error("fanout requires --questions-per-state or --questions-file")

    if not args.live and not args.dry_run:
        print(
            "refusing to run: pass --live for real model calls or --dry-run to print the plan",
            file=sys.stderr,
        )
        return 2

    try:
        primitives = _live_primitives() if (args.live and not args.dry_run) else None
        if args.study == "contamination":
            receipt = run_contamination_study(
                primitives,
                state=_load_state(args.state_file),
                questions=_load_questions(args.questions_file),
                role=args.role,
                order_permutations=args.order_permutations,
                seed=args.seed,
                receipt_path=args.receipt,
                artifacts_dir=args.artifacts_dir,
                dry_run=args.dry_run,
            )
        elif args.study == "calibration":
            receipt = run_calibration_study(
                primitives,
                state=_load_state(args.state_file),
                questions=_load_questions(args.questions_file),
                labels=_load_labels(args.labels_file),
                role=args.role,
                receipt_path=args.receipt,
                artifacts_dir=args.artifacts_dir,
                dry_run=args.dry_run,
            )
        else:
            receipt = run_fanout_study(
                primitives,
                states=_load_states(args.states_file),
                role=args.role,
                questions_per_state=args.questions_per_state,
                mode=args.mode,
                questions=(
                    _load_questions(args.questions_file)
                    if args.questions_file is not None
                    else None
                ),
                receipt_path=args.receipt,
                artifacts_dir=args.artifacts_dir,
                dry_run=args.dry_run,
            )
    except (MeasurementError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


def _live_primitives() -> Any:
    """Build live ``LLMPrimitives`` for --live runs; import stays lazy/offline."""
    from src.config import get_config
    from src.llm_primitives import LLMPrimitives

    config = get_config()
    primitives = LLMPrimitives(
        mock_mode=False,
        server_urls=config.server_urls.as_dict(),
        num_slots=config.server.num_slots,
    )
    if not getattr(primitives, "_backends", None):
        raise MeasurementError(
            "no LLM backends available for the configured server URLs; start the stack first"
        )
    return primitives


def _load_questions(path: str | Path) -> list[Question]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("questions")
    if not isinstance(payload, list):
        raise ValueError(f"questions file {path} must contain a list or {{'questions': list}}")
    questions: list[Question] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"questions file {path} contains a non-object entry")
        questions.append(
            Question(
                id=str(item["id"]),
                kind=item["kind"],
                text=str(item["text"]),
                options=tuple(item.get("options", ())),
                levels=tuple(item.get("levels", ())),
                criteria=tuple(item.get("criteria", ())),
            )
        )
    return questions


def _load_state(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("state"), str):
        return payload["state"]
    raise ValueError(f"state file {path} must contain a string or {{'state': string}}")


def _load_states(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("states", payload.get("state"))
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return list(payload)
    raise ValueError(
        f"states file {path} must contain a string, a list of strings, "
        "or an object with 'states'/'state'"
    )


def _load_labels(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"labels file {path} must contain a JSON object")
    return {str(key): value for key, value in payload.items()}


if __name__ == "__main__":
    raise SystemExit(main())
