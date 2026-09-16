"""SC83: persist the RA-9 negative-control false-accept rate as a durable row.

RA-9's ``gold_annotations.false_accept_rate()`` computes the rate, but nothing
stored it, so the belief kernel had nothing to read. This module is the write
side for one SCORING RUN: one reviewer configuration scored against one
gold-corpus version. It appends one line to an append-only JSONL. That line
carries the ``FalseAcceptResult`` with its denominator stated, the identity and
sha256 of every input, and at most one producer-authored ``belief_measurements``
row.

Rules, each pinned by a test:

* **Staleness is a write-side filter (RA-12).** A verdict counts only if its
  envelope passes ``review_envelope.check_binding`` against the CURRENT inputs
  for that annotation. A stale, unbound or tampered verdict is dropped before
  scoring. That decoy then shows up as ``unscored``, and the line lists it under
  ``stale`` with the reasons. A verdict is never re-bound here.
* **The endorsement comes from the signed body** (``review.endorsed`` or
  ``review.verdict`` in {endorse, reject}), never from an unsigned side field.
  A body that decides neither way counts as unscored.
* **One locator per run.** Every current verdict must share one reviewer
  configuration: model, quant, prompt bundle, pipeline and review schema. A mixed
  run is refused, never averaged.
* **The denominator is accounted for.** ``n_decoys == scored + unscored +
  excluded_for_arbitration`` is checked before writing.
* **Rates only.** Single verdicts and objections are categorical and never
  become rows. Machine objections stay ``unverified_lead`` and are not projected.
* **Observation only.** The row cites no protocol: RC-6a has not merged, so the
  rate may not gate a decision. Grading happens in the root
  ``claim_tuple.grade()``. This module never grades.
* **Absence is not zero.** With no scored decoys the rate is ``None``, so the
  line carries no belief row.

NO inference. Pure computation plus one fsync'd append.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.proactive_delegation import gold_annotations as ga
from src.proactive_delegation import review_envelope as rev

SCHEMA = "epyc.reviewer.false_accept_run.v1"
BELIEF_SCHEMA = "epyc.reviewer.false_accept_belief.v1"
WRITER_ID = "epyc-orchestrator/src/proactive_delegation/false_accept_record.py@v1"
METRIC = "reviewer.negative_control_false_accept_rate"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2] / "data" / "reviewer_eval" / "false_accept_runs.jsonl"
)
REVIEWER_CONFIG_FIELDS = (
    "reviewer_model", "reviewer_quant", "reviewer_model_hash", "prompt_bundle_hash",
    "pipeline_version", "review_schema_version",
)


class FalseAcceptRecordError(ValueError):
    """The scoring run cannot be recorded honestly."""


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row_digest(row: Mapping[str, Any]) -> str:
    body = json.loads(_canon(row))
    body.get("extra", {}).pop("row_sha256", None)
    return sha256_bytes(_canon(body).encode())


def bound_endorsement(signed_body: Mapping[str, Any]) -> bool | None:
    review = signed_body.get("review") if isinstance(signed_body, Mapping) else None
    if not isinstance(review, Mapping):
        return None
    if isinstance(review.get("endorsed"), bool):
        return review["endorsed"]
    verdict = review.get("verdict")
    if verdict == "endorse":
        return True
    if verdict == "reject":
        return False
    return None


def _reviewer_config(binding: rev.ReviewBinding) -> dict[str, Any]:
    return {name: getattr(binding, name) for name in REVIEWER_CONFIG_FIELDS}


def score_run(
    annotations: Iterable[Mapping[str, Any]],
    verdicts: Iterable[Mapping[str, Any]],
    current_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the RA-12 filter, then ``false_accept_rate``. Returns the scoring core."""
    annotations = list(annotations)
    bad = {a.get("annotation_id", "?"): errs for a in annotations
           if (errs := ga.annotation_errors(a))}
    if bad:
        raise FalseAcceptRecordError(f"corpus has inadmissible annotations: {sorted(bad)}")
    ids = [a["annotation_id"] for a in annotations]
    if len(ids) != len(set(ids)):
        raise FalseAcceptRecordError("corpus repeats an annotation_id")
    corpus_ids = sorted({a["corpus_id"] for a in annotations})
    if len(corpus_ids) != 1:
        raise FalseAcceptRecordError(f"one scoring run scores one corpus, got {corpus_ids}")

    endorsements: dict[str, bool] = {}
    stale: dict[str, list[str]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        aid = str(verdict.get("annotation_id") or "")
        if aid in endorsements or aid in stale:
            raise FalseAcceptRecordError(f"two verdicts for {aid!r} in one run")
        envelope = verdict.get("envelope") or {}
        body = verdict.get("signed_body") or {}
        current = current_bindings.get(aid)
        if current is None:
            stale[aid] = ["no current binding supplied for this annotation"]
            continue
        check = rev.check_binding(envelope, body, rev.ReviewBinding.from_schema(current))
        if not check.current:
            stale[aid] = list(check.reasons)
            continue
        endorsed = bound_endorsement(body)
        if endorsed is None:
            stale[aid] = ["signed review body decides neither endorse nor reject"]
            continue
        endorsements[aid] = endorsed
        config = _reviewer_config(rev.ReviewBinding.from_schema(envelope["binding"]))
        configs[_canon(config)] = config
    if len(configs) > 1:
        raise FalseAcceptRecordError(
            "verdicts span more than one reviewer configuration; one run = one locator")

    result = ga.false_accept_rate(annotations, endorsements)
    accounted = result.n_scored + len(result.unscored) + len(result.excluded_for_arbitration)
    if accounted != result.n_decoys:
        raise FalseAcceptRecordError(
            f"denominator dropped decoys: {result.n_decoys} decoys, {accounted} accounted for")
    decoy_ids = {a["annotation_id"] for a in annotations if a["status"] == ga.INVALID}
    return {
        "corpus_id": corpus_ids[0],
        "n_annotations": len(annotations),
        "result": result.as_dict(),
        "stale": {k: stale[k] for k in sorted(stale) if k in decoy_ids},
        "stale_non_decoy": sorted(k for k in stale if k not in decoy_ids),
        "reviewer_config": next(iter(configs.values())) if configs else None,
        "current_verdicts": len(endorsements),
    }


def belief_row(core: Mapping[str, Any], *, run_id: str, scored_at: str, out: Path,
               corpus_sha256: str, extra_ids: Mapping[str, Any]) -> dict[str, Any] | None:
    res = core["result"]
    if res["rate"] is None or not core["reviewer_config"]:
        return None
    cfg = core["reviewer_config"]
    cfg_digest = sha256_bytes(_canon(cfg).encode())
    locator_key = sha256_bytes(f"{cfg_digest}|{corpus_sha256}".encode())[:16]
    row = {
        "measurement_id": f"reviewer-fa-{locator_key}-{run_id}",
        "metric": METRIC,
        "value": res["numerator"] / res["denominator"],
        "unit": "fraction",
        "date": scored_at,
        "category": "BASELINE",
        "claim": (f"reviewer {cfg['reviewer_model']} ({cfg['reviewer_quant']}) endorsed "
                  f"{res['numerator']} of {res['denominator']} scored negative-control decoys "
                  f"in corpus {core['corpus_id']}"),
        "metric_direction": "lower_better",
        "protocol_id": "",
        "reps": res["denominator"],
        "reps_basis": ("scored: settled decoys with a current (RA-12-bound) verdict; "
                       "unscored, stale and arbitration-excluded decoys are listed, not counted"),
        "attestation_path": str(out),
        "attestation_locator": f"{out}#run={run_id}",
        "source_kind": "measurement",
        "extra": {
            "belief_schema": BELIEF_SCHEMA,
            "producer": WRITER_ID,
            "run_id": run_id,
            "numerator": res["numerator"],
            "denominator": res["denominator"],
            "n_decoys": res["n_decoys"],
            "unscored": res["unscored"],
            "excluded_for_arbitration": res["excluded_for_arbitration"],
            "stale": sorted(core["stale"]),
            "reviewer_config": cfg,
            "reviewer_config_sha256": cfg_digest,
            "corpus_id": core["corpus_id"],
            "corpus_sha256": corpus_sha256,
            "locator_basis": "scoring run = reviewer config x corpus version",
            "authority": "observation: RC-6a not merged, never decision-gating",
            "objections_projected": False,
            **dict(extra_ids),
        },
    }
    row["extra"]["row_sha256"] = row_digest(row)
    return row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise FalseAcceptRecordError(f"{path}:{n}: {exc}") from exc
    return rows


def build_run_line(
    *,
    corpus_path: Path,
    verdicts_path: Path,
    bindings_path: Path,
    run_id: str,
    out: Path,
    scored_at: str | None = None,
) -> dict[str, Any]:
    """Read the three inputs, score, and build the line to append (does not write)."""
    if not run_id.strip() or any(c in run_id for c in "#/ \n"):
        raise FalseAcceptRecordError("run_id must be a non-empty token without '#', '/' or spaces")
    blobs = {name: Path(p).read_bytes() for name, p in (
        ("corpus", corpus_path), ("verdicts", verdicts_path), ("bindings", bindings_path))}
    annotations = read_jsonl(corpus_path)
    verdicts = read_jsonl(verdicts_path)
    bindings = json.loads(blobs["bindings"])
    if not isinstance(bindings, dict):
        raise FalseAcceptRecordError("current bindings must be a JSON object keyed by annotation_id")
    core = score_run(annotations, verdicts, bindings)
    scored_at = scored_at or datetime.now(timezone.utc).isoformat()
    inputs = {
        name: {"path": str(p), "sha256": sha256_bytes(blobs[name]), "bytes": len(blobs[name])}
        for name, p in (("corpus", corpus_path), ("verdicts", verdicts_path),
                        ("bindings", bindings_path))
    }
    row = belief_row(core, run_id=run_id, scored_at=scored_at, out=Path(out),
                     corpus_sha256=inputs["corpus"]["sha256"],
                     extra_ids={"inputs": inputs})
    line = {
        "schema": SCHEMA,
        "record": "scoring_run",
        "writer": WRITER_ID,
        "run_id": run_id,
        "scored_at": scored_at,
        "gold_annotation_schema_version": ga.GOLD_ANNOTATION_SCHEMA_VERSION,
        "envelope_schema_version": rev.ENVELOPE_SCHEMA_VERSION,
        "inputs": inputs,
        **core,
        "belief_measurements": [row] if row else [],
    }
    line["line_sha256"] = sha256_bytes(_canon(line).encode())
    return line


def append_line(out: Path, line: Mapping[str, Any]) -> None:
    """fsync'd append. Refuses a run_id that is already recorded (append-only, write-once)."""
    out = Path(out)
    if out.exists():
        for raw in out.read_text(encoding="utf-8").splitlines():
            try:
                prior = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(prior, dict) and prior.get("run_id") == line["run_id"]:
                raise FalseAcceptRecordError(f"run_id {line['run_id']!r} already recorded in {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _canon(dict(line)) + "\n"
    with open(out, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


__all__ = [
    "BELIEF_SCHEMA", "DEFAULT_OUT", "FalseAcceptRecordError", "METRIC", "SCHEMA", "WRITER_ID",
    "append_line", "belief_row", "bound_endorsement", "build_run_line", "row_digest", "score_run",
]
