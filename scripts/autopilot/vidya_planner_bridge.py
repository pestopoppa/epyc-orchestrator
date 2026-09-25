"""Read-only AutoPilot planner bridge to the Vidya settled-ground lookup.

Hypothesis generation remains unrestricted.  This block only tells the planner
whether a previously recorded resolution is sealed, provisional, or needs
review because its cited trial was later invalidated.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESOLUTIONS = ORCH_ROOT / "orchestration" / "operator_hypothesis_resolutions.jsonl"
DEFAULT_LOOKUP = Path("/workspace/scripts/vidya/autopilot_settled.py")
DEFAULT_KVQ_LOOKUP = Path("/workspace/scripts/vidya/kvq_planner_context.py")
LOOKUP_ENV = "AUTOPILOT_VIDYA_SETTLED_LOOKUP"
TIMEOUT_ENV = "AUTOPILOT_VIDYA_SETTLED_TIMEOUT_S"
KVQ_LOOKUP_ENV = "AUTOPILOT_VIDYA_KVQ_LOOKUP"


def build_settled_ground_block(
    *,
    resolutions_path: Path = DEFAULT_RESOLUTIONS,
    lookup_path: Path | None = None,
    timeout_s: float | None = None,
) -> str:
    """Return planner text; unavailable state is loud and never reads as empty."""
    try:
        if not resolutions_path.exists() or not resolutions_path.read_text().strip():
            return "  (none; no operator-hypothesis resolutions recorded)"
    except OSError as exc:
        return f"  !! VIDYA SETTLED-GROUND LOOKUP UNAVAILABLE: cannot read ledger: {exc}"

    lookup = lookup_path or Path(os.environ.get(LOOKUP_ENV, str(DEFAULT_LOOKUP)))
    if not lookup.is_file():
        return f"  !! VIDYA SETTLED-GROUND LOOKUP UNAVAILABLE: missing {lookup}"
    timeout = timeout_s
    if timeout is None:
        try:
            timeout = float(os.environ.get(TIMEOUT_ENV, "3"))
        except ValueError:
            timeout = 3.0
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(lookup),
                "--orch-root",
                str(ORCH_ROOT),
                "--resolutions",
                str(resolutions_path),
            ],
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout),
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - planner receives the explicit unknown state
        return f"  !! VIDYA SETTLED-GROUND LOOKUP UNAVAILABLE: {type(exc).__name__}: {exc}"
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = (proc.stderr or proc.stdout or "empty output").strip()[:400]
        return (
            "  !! VIDYA SETTLED-GROUND LOOKUP UNAVAILABLE: "
            f"exit={proc.returncode}; {detail}"
        )
    return proc.stdout.strip()


def build_kvq_context(
    *, lookup_path: Path | None = None, timeout_s: float | None = None
) -> dict:
    """Return one bounded text block with its source manifest for a planner turn."""
    def unavailable(reason: str) -> dict:
        return {"schema": "epyc.vidya.kvq_planner_context.v1",
                "status": "unavailable", "reason": reason,
                "run": None, "frontier": None, "state_hash": None,
                "claim_ids": [],
                "text": f"  !! VIDYA KV-QUANT EVIDENCE UNAVAILABLE: {reason}"}

    lookup = lookup_path or Path(os.environ.get(KVQ_LOOKUP_ENV, str(DEFAULT_KVQ_LOOKUP)))
    if not lookup.is_file():
        return unavailable(f"missing {lookup}")
    timeout = timeout_s
    if timeout is None:
        try:
            timeout = float(os.environ.get(TIMEOUT_ENV, "3"))
        except ValueError:
            timeout = 3.0
    try:
        proc = subprocess.run(
            [sys.executable, str(lookup), "--json"],
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout),
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - absence must stay explicit to the planner
        return unavailable(f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = (proc.stderr or proc.stdout or "empty output").strip()[:400]
        return unavailable(f"exit={proc.returncode}; {detail}")
    try:
        result = json.loads(proc.stdout)
        if not isinstance(result, dict) or result.get("schema") != "epyc.vidya.kvq_planner_context.v1":
            raise ValueError("invalid context schema")
        if result.get("status") not in {"available", "unavailable"}:
            raise ValueError("invalid evidence status")
        if not isinstance(result.get("text"), str) or not result["text"].strip():
            raise ValueError("empty evidence text")
        if result["status"] == "available" and (
            not isinstance(result.get("run"), str) or not result["run"]
            or not isinstance(result.get("frontier"), int) or result["frontier"] < 0
            or not isinstance(result.get("state_hash"), str) or not result["state_hash"]
            or not isinstance(result.get("claim_ids"), list)
            or len(result["claim_ids"]) != 12
            or any(not isinstance(cid, str) or not cid for cid in result["claim_ids"])
            or len(set(result["claim_ids"])) != 12
            or any(f"claim_id={cid}" not in result["text"] for cid in result["claim_ids"])
        ):
            raise ValueError("incomplete evidence manifest")
        if result["status"] == "unavailable":
            result["claim_ids"] = []
        return result
    except (ValueError, TypeError) as exc:
        return unavailable(f"invalid lookup output: {exc}")


def build_kvq_evidence_block(
    *, lookup_path: Path | None = None, timeout_s: float | None = None
) -> str:
    return build_kvq_context(lookup_path=lookup_path, timeout_s=timeout_s)["text"]


__all__ = ["build_settled_ground_block", "build_kvq_context", "build_kvq_evidence_block"]
