#!/usr/bin/env python3
"""Deterministically restamp clean multi-tier evidence across a monotone instrument fix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


SOURCE_INSTRUMENT = "resource_lanes_v9_multimodal_input_identity"
TARGET_INSTRUMENT = "resource_lanes_v10_history_scoped_quiescence"
RECODE_SCHEMA = "epyc.multitier_execution_instrument_recode.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _validate_clean(payload: dict[str, Any]) -> None:
    if payload.get("status") != "candidate_unratified":
        raise ValueError("source is not an unratified candidate")
    result = payload.get("eval_result") or {}
    details = result.get("details") or {}
    n_questions = int(result.get("n_questions") or 0)
    rows = result.get("question_results") or []
    qids = [str(row.get("qid") or "") for row in rows if isinstance(row, dict)]
    errors: list[str] = []
    if float(result.get("reliability") or 0.0) != 1.0:
        errors.append("reliability is not 1.0")
    if int(details.get("errors") or 0) != 0:
        errors.append("errors is not zero")
    if int(details.get("scoring_errors") or 0) != 0:
        errors.append("scoring_errors is not zero")
    if int(details.get("eval_backend_drain_failure_count") or 0) != 0:
        errors.append("backend drain failures are nonzero")
    if bool(details.get("eval_contaminated_by_abandoned_requests")):
        errors.append("source is marked contaminated")
    if len(rows) != n_questions:
        errors.append(f"question row count {len(rows)} != {n_questions}")
    if len(qids) != n_questions or len(set(qids)) != n_questions or not all(qids):
        errors.append("question qids are missing or duplicated")
    if details.get("eval_execution_instrument_id") != SOURCE_INSTRUMENT:
        errors.append("unexpected source execution instrument")
    profile = details.get("eval_execution_profile") or {}
    if profile.get("execution_instrument_id") != SOURCE_INSTRUMENT:
        errors.append("unexpected source execution profile")
    profile_json = json.dumps(profile, sort_keys=True, separators=(",", ":"), default=str)
    if hashlib.sha256(profile_json.encode()).hexdigest() != details.get(
        "eval_execution_profile_sha256"
    ):
        errors.append("source execution profile hash mismatch")
    if errors:
        raise ValueError("source evidence is not clean: " + "; ".join(errors))


def recode_payload(
    source: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    recode_git_head: str,
) -> dict[str, Any]:
    """Restamp only execution-instrument identity; preserve all measured fields."""
    _validate_clean(source)
    out = copy.deepcopy(source)
    details = out["eval_result"]["details"]
    profile = details["eval_execution_profile"]
    profile["execution_instrument_id"] = TARGET_INSTRUMENT
    profile_json = json.dumps(profile, sort_keys=True, separators=(",", ":"), default=str)
    details["eval_execution_instrument_id"] = TARGET_INSTRUMENT
    details["eval_execution_profile_sha256"] = hashlib.sha256(
        profile_json.encode()
    ).hexdigest()
    out["execution_instrument_recode"] = {
        "schema_version": RECODE_SCHEMA,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_execution_instrument_id": SOURCE_INSTRUMENT,
        "target_execution_instrument_id": TARGET_INSTRUMENT,
        "recode_git_head": recode_git_head,
        "reason": "v10 only accepts history-only reducer degradation when live state is intact",
        "applicability_proof": {
            "source_reliability": 1.0,
            "source_errors": 0,
            "source_scoring_errors": 0,
            "source_backend_drain_failures": 0,
            "source_contaminated": False,
            "v10_changes_previously_accepted_rows": False,
        },
        "answers_changed": False,
        "scores_changed": False,
        "timing_changed": False,
        "routing_changed": False,
        "source_collection_provenance_preserved": True,
        "allowed_changed_paths": [
            "eval_result.details.eval_execution_instrument_id",
            "eval_result.details.eval_execution_profile.execution_instrument_id",
            "eval_result.details.eval_execution_profile_sha256",
            "execution_instrument_recode",
        ],
    }
    return out


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    source = json.loads(source_path.read_text())
    payload = recode_payload(
        source,
        source_path=source_path,
        source_sha256=_sha256(source_path),
        recode_git_head=_git_head(repo_root),
    )
    _write_immutable(output_path, payload)
    print(
        json.dumps(
            {
                "status": "candidate_written",
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "tier": payload.get("tier"),
                "quality": payload["eval_result"].get("quality"),
                "reliability": payload["eval_result"].get("reliability"),
                "execution_instrument_id": TARGET_INSTRUMENT,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
