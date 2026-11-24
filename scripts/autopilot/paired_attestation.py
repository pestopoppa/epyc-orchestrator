#!/usr/bin/env python3
"""Paired-comparison attestation + ledger writer (AEI-RERUN-ATTEST).

Renders the P-PAIRED claim grammar ratified with the rerun-arm amendment
(AEI-RERUN-ARM-OD, measurement/protocols/quality-eval.md):

    verdict [P-PAIRED, n_discordant/method, rerun-flip f/n_AR, YYYY-MM-DD,
             attest <summary.json ref>, rerun <R summary.json ref>]

A verdict that downgraded to ``within_rerun_noise`` is rendered as the
no-claim form ``within rerun noise [P-AB-1, rerun-flip f/n_AR, YYYY-MM-DD]``.
A paired comparison with no R (no ``rerun`` block on the verdict, or no R
``summary.json`` ref) is refused for decision use: ``RerunArmMissingError``.
Pure formatting plus one append-only JSONL write; no inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autopilot.paired_stats import RerunArmMissingError


def _require_rerun(verdict: Mapping[str, Any], rerun_ref: str | None) -> Mapping[str, Any]:
    rerun = verdict.get("rerun")
    if not isinstance(rerun, Mapping):
        raise RerunArmMissingError(
            "refusing paired attestation for decision use: verdict carries no rerun arm R "
            "(observation only; P-AB-1 / P-PAIRED rerun arm)"
        )
    if not rerun_ref or not str(rerun_ref).strip():
        raise RerunArmMissingError(
            "refusing paired attestation for decision use: the R summary.json ref is "
            "mandatory in the attestation (P-PAIRED rerun arm)"
        )
    return rerun


def rerun_flip_string(rerun: Mapping[str, Any]) -> str:
    """``rerun-flip f/n_AR`` with f = f_AR + f_RA."""
    return f"rerun-flip {int(rerun['f_ar']) + int(rerun['f_ra'])}/{int(rerun['n_ar'])}"


def format_paired_claim(
    verdict: Mapping[str, Any],
    *,
    date: str,
    attest_ref: str,
    rerun_ref: str | None,
) -> str:
    """Render the ratified claim string, refusing a pair with no R."""
    rerun = _require_rerun(verdict, rerun_ref)
    if not attest_ref or not str(attest_ref).strip():
        raise ValueError("paired attestation needs the A-vs-B summary.json ref")
    flip = rerun_flip_string(rerun)
    if verdict.get("reason") == "within_rerun_noise":
        return f"within rerun noise [P-AB-1, {flip}, {date}]"
    return (
        f"{verdict['verdict']} [P-PAIRED, {verdict['n_discordant']}/{verdict['method']}, "
        f"{flip}, {date}, attest {attest_ref}, rerun {rerun_ref}]"
    )


def build_paired_attestation(
    verdict: Mapping[str, Any],
    *,
    date: str,
    attest_ref: str,
    rerun_ref: str | None,
    arm_a: str | None = None,
    arm_b: str | None = None,
) -> dict[str, Any]:
    """Attestation record: claim string, rerun refs, and the flip statistics."""
    claim = format_paired_claim(verdict, date=date, attest_ref=attest_ref, rerun_ref=rerun_ref)
    rerun = verdict["rerun"]
    return {
        "protocol": "P-PAIRED",
        "claim": claim,
        "verdict": verdict["verdict"],
        "reason": verdict.get("reason"),
        "rerun_flip": rerun_flip_string(rerun),
        "rerun_ref": rerun_ref,
        "attest_ref": attest_ref,
        "date": date,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_discordant": verdict["n_discordant"],
        "method": verdict["method"],
        "p_value": verdict.get("p_value"),
        "abs_delta": verdict.get("abs_delta"),
        "discordant_rate": verdict.get("discordant_rate"),
        "rerun": dict(rerun),
    }


def append_paired_ledger_row(path: Path | str, record: Mapping[str, Any]) -> Path:
    """Append one attestation as a JSONL ledger row. Refuses a record with no R."""
    if not record.get("rerun_ref") or not record.get("rerun_flip"):
        raise RerunArmMissingError(
            "refusing to ledger a paired attestation without rerun ref and rerun-flip"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return out
