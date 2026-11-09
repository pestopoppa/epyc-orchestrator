"""M-19: typed memory-write record and confidence gate contract (spec as code).

NOT WIRED to any live write path. Pure value types plus a pure gate function, so
the record spec is testable now (zero inference). Rules, each from the M-19 box:

  * The record carries per-question probabilities AND a caller-computed confidence
    statistic (``computed_by == "caller"``); the model never reports its own.
  * The gate threshold comes ONLY from a ``CalibrationRecord`` for the SAME
    (catalogue id + version, model pin, statistic name). Candidate softmax is not a
    statistic: intake-1474 measured 65% of 7B wrong fields above 0.90 confidence.
  * Calibration lifts per (model pin, question catalogue), never globally, and
    enforcement needs an operator-ratified calibration record (TD-5). Without one
    the verdict is a SHADOW verdict and ``enforce`` is False.
  * An uncalibrated record is ``unscored``: it is stored and never blocked or
    passed on confidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Mapping

SCHEMA_VERSION = "typed_decisions.memory_write_record.v1"
OPS = ("ADD", "UPDATE", "DELETE", "NOOP")
# Statistic names that are candidate softmax in disguise; refused as a gate input.
FORBIDDEN_STATISTICS = frozenset({"softmax", "softmax_max", "candidate_softmax", "max_prob", "token_prob"})


class MemoryWriteRecordError(ValueError):
    pass


@dataclass(frozen=True)
class QuestionAnswer:
    question_id: str
    value: object
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.question_id:
            raise MemoryWriteRecordError("question_id required")
        total = sum(self.probabilities.values()) if self.probabilities else 0.0
        if self.probabilities and (
            any((not math.isfinite(p)) or p < 0 for p in self.probabilities.values())
            or abs(total - 1.0) > 1e-6
        ):
            raise MemoryWriteRecordError(f"{self.question_id}: probabilities must be finite, >=0 and sum to 1")


@dataclass(frozen=True)
class ConfidenceStatistic:
    name: str
    value: float
    computed_by: str = "caller"

    def __post_init__(self) -> None:
        if self.name in FORBIDDEN_STATISTICS or not self.name:
            raise MemoryWriteRecordError(f"statistic {self.name!r} is candidate softmax or empty; not a gate input")
        if self.computed_by != "caller":
            raise MemoryWriteRecordError("confidence statistic must be caller-computed")
        if not (math.isfinite(self.value) and 0.0 <= self.value <= 1.0):
            raise MemoryWriteRecordError("statistic value must be in [0, 1]")


@dataclass(frozen=True)
class MemoryWriteRecord:
    op: str
    target_key: str
    content_sha256: str
    answers: tuple[QuestionAnswer, ...]
    statistic: ConfidenceStatistic
    catalogue_id: str
    catalogue_version: str
    model_pin: Mapping[str, str]
    source_event_ids: tuple[str, ...] = ()
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise MemoryWriteRecordError(f"op must be one of {OPS}")
        if not self.target_key or not self.content_sha256:
            raise MemoryWriteRecordError("target_key and content_sha256 required")
        if not self.model_pin:
            raise MemoryWriteRecordError("model_pin required: calibration is per model pin")
        if not self.catalogue_id or not self.catalogue_version:
            raise MemoryWriteRecordError("catalogue id and version required")
        if self.op != "NOOP" and not self.source_event_ids:
            raise MemoryWriteRecordError("non-NOOP operations must cite source events")
        ids = [a.question_id for a in self.answers]
        if len(set(ids)) != len(ids):
            raise MemoryWriteRecordError("duplicate question ids")

    def digest(self) -> str:
        payload = {
            "schema": self.schema, "op": self.op, "target_key": self.target_key,
            "content_sha256": self.content_sha256, "catalogue": [self.catalogue_id, self.catalogue_version],
            "model_pin": dict(sorted(self.model_pin.items())),
            "statistic": [self.statistic.name, self.statistic.value],
            "answers": [[a.question_id, a.value, dict(sorted(a.probabilities.items()))] for a in self.answers],
            "sources": list(self.source_event_ids),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class CalibrationRecord:
    """Threshold provenance for one (catalogue, model pin, statistic)."""

    calibration_id: str
    catalogue_id: str
    catalogue_version: str
    model_pin: Mapping[str, str]
    statistic_name: str
    threshold: float
    operator_ratified: bool = False


@dataclass(frozen=True)
class GateVerdict:
    status: str  # unscored | shadow_pass | shadow_hold | pass | hold
    enforce: bool
    reason: str
    calibration_id: str | None = None


def evaluate_gate(record: MemoryWriteRecord, calibration: CalibrationRecord | None) -> GateVerdict:
    if calibration is None:
        return GateVerdict("unscored", False, "no calibration record for this catalogue and model pin")
    if (calibration.catalogue_id, calibration.catalogue_version) != (record.catalogue_id, record.catalogue_version):
        return GateVerdict("unscored", False, "calibration is for a different catalogue or version")
    if dict(calibration.model_pin) != dict(record.model_pin):
        return GateVerdict("unscored", False, "calibration is for a different model pin")
    if calibration.statistic_name != record.statistic.name:
        return GateVerdict("unscored", False, "calibration is for a different statistic")
    if not math.isfinite(calibration.threshold):
        return GateVerdict("unscored", False, "calibration threshold is not finite")
    passed = record.statistic.value >= calibration.threshold
    enforce = bool(calibration.operator_ratified)
    status = ("pass" if passed else "hold") if enforce else ("shadow_pass" if passed else "shadow_hold")
    return GateVerdict(status, enforce, "threshold from calibration record", calibration.calibration_id)
