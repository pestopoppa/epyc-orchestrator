"""RA-9 — dual-gold annotation envelope + the negative-control (false-accept) axis.

Schema: ``orchestration/gold_annotation.schema.json`` (pattern from benchmrk,
intake-948#record). This is the ONE implementation of the dual-gold schema; the
security-review skill (``security-review-skill.md``) and the reviewer corpus
(``reviewer-typed-artifacts.md`` RA-9) both use it.

What it adds that the corpus lacked: ``status: invalid`` decoys. Every gold
row we had asserted a TRUE finding, so no false-accept rate was measurable from
our own data (intake-845#record). ``false_accept_rate`` computes that rate over
decoys with an explicit denominator: decoys the reviewer never judged are
listed, never silently dropped (a shrunken denominator inflates nothing here,
but it hides coverage, so it is reported).

``annotation_errors`` = JSON Schema + the semantic rules the schema cannot say:
dual-gold agreement must be arbitrated, a machine-generated annotation's
gold-sanity record must satisfy ``gold_sanity.validate_record``, and consensus
counts must be coherent.

NO inference; pure functions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.proactive_delegation import gold_sanity

GOLD_ANNOTATION_SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "orchestration" / "gold_annotation.schema.json"

VALID = "valid"
INVALID = "invalid"


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _schema_errors(annotation: Mapping[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load_schema())
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '$'}: {err.message}"
        for err in sorted(validator.iter_errors(annotation), key=lambda e: list(e.absolute_path))
    ]


def dual_gold_agrees(annotation: Mapping[str, Any]) -> bool | None:
    """Both gold channels conclusive: True iff BOTH confirm ``status``. Otherwise None.

    The executable oracle's ``pass`` means "the witness confirms ``status``";
    the reasoning label names a status directly.
    """
    gold = annotation.get("gold") or {}
    oracle = gold.get("executable_oracle")
    label = gold.get("reasoning_label")
    if not oracle or not label or oracle.get("verdict") not in ("pass", "fail"):
        return None
    oracle_confirms = oracle["verdict"] == "pass"
    label_confirms = label.get("verdict") == annotation.get("status")
    return oracle_confirms == label_confirms and oracle_confirms


def requires_arbitration(annotation: Mapping[str, Any]) -> bool:
    """Arbitration is required when gold disagrees, contradicts status, or is inconclusive."""
    gold = annotation.get("gold") or {}
    oracle = gold.get("executable_oracle")
    label = gold.get("reasoning_label")
    if oracle and oracle.get("verdict") in ("fail", "inconclusive"):
        return True
    if label and label.get("verdict") != annotation.get("status"):
        return True
    return dual_gold_agrees(annotation) is False


def annotation_errors(annotation: Mapping[str, Any]) -> list[str]:
    """All reasons this annotation may not enter a gold corpus (empty == admissible)."""
    errs = _schema_errors(annotation)
    if errs:
        return errs
    if requires_arbitration(annotation) and not annotation["needs_arbitration"]:
        errs.append("gold is conflicting or inconclusive but needs_arbitration is false")
    consensus = annotation.get("consensus")
    if consensus:
        if consensus["n_agree"] > consensus["n_annotators"]:
            errs.append("consensus n_agree exceeds n_annotators")
        if consensus["n_annotators"] < len(annotation["annotated_by"]):
            errs.append("consensus n_annotators is smaller than annotated_by")
    subject = annotation["subject"]
    if "line_start" in subject and "line_end" in subject and subject["line_end"] < subject["line_start"]:
        errs.append("subject line_end precedes line_start")
    if annotation["origin"] == "machine_generated":
        errs.extend(f"gold_sanity: {e}" for e in gold_sanity.validate_record(annotation["gold_sanity"]))
    elif any(a["kind"] == "model" for a in annotation["annotated_by"]):
        errs.append("a model annotator requires origin machine_generated (and a gold-sanity record)")
    return errs


def expected_reviewer_action(annotation: Mapping[str, Any]) -> str:
    """``endorse`` for a true finding, ``reject`` for a decoy (the gold polarity)."""
    return "endorse" if annotation["status"] == VALID else "reject"


def corpus_gold_label(annotation: Mapping[str, Any]) -> str:
    """Map onto the corpus/ledger vocabulary: the FINDING is the candidate under review."""
    return "accept" if annotation["status"] == VALID else "reject"


@dataclass(frozen=True)
class FalseAcceptResult:
    """False-accept rate over negative controls, with its denominator stated."""

    n_decoys: int
    n_scored: int
    n_endorsed: int
    unscored: tuple[str, ...]
    excluded_for_arbitration: tuple[str, ...]

    @property
    def rate(self) -> float | None:
        return None if self.n_scored == 0 else self.n_endorsed / self.n_scored

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": "negative_control_false_accept_rate",
            "direction": "lower_is_better",
            "numerator": self.n_endorsed,
            "denominator": self.n_scored,
            "rate": self.rate,
            "n_decoys": self.n_decoys,
            "unscored": list(self.unscored),
            "excluded_for_arbitration": list(self.excluded_for_arbitration),
        }


def false_accept_rate(
    annotations: Iterable[Mapping[str, Any]],
    endorsements: Mapping[str, bool],
) -> FalseAcceptResult:
    """Share of decoys (``status: invalid``) the reviewer endorsed.

    ``endorsements`` maps ``annotation_id`` -> did the reviewer endorse the
    finding. Decoys still awaiting arbitration are excluded (their gold is not
    settled) and listed. An endorsement for an unknown id is an error: it means
    the reviewer was scored against a different corpus.
    """
    annotations = list(annotations)
    known = {a["annotation_id"] for a in annotations}
    unknown = sorted(set(endorsements) - known)
    if unknown:
        raise ValueError(f"endorsements reference annotations not in the corpus: {unknown}")
    decoys = [a for a in annotations if a["status"] == INVALID]
    excluded = tuple(sorted(a["annotation_id"] for a in decoys if a["needs_arbitration"]))
    settled = [a for a in decoys if not a["needs_arbitration"]]
    unscored = tuple(sorted(a["annotation_id"] for a in settled if a["annotation_id"] not in endorsements))
    scored = [a for a in settled if a["annotation_id"] in endorsements]
    endorsed = sum(1 for a in scored if endorsements[a["annotation_id"]])
    return FalseAcceptResult(len(decoys), len(scored), endorsed, unscored, excluded)
