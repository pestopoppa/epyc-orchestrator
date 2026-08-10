"""Read-only AutoPilot planner bridge to the Vidya settled-ground lookup.

Hypothesis generation remains unrestricted.  This block only tells the planner
whether a previously recorded resolution is sealed, provisional, or needs
review because its cited trial was later invalidated.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESOLUTIONS = ORCH_ROOT / "orchestration" / "operator_hypothesis_resolutions.jsonl"
DEFAULT_LOOKUP = Path("/workspace/scripts/vidya/autopilot_settled.py")
LOOKUP_ENV = "AUTOPILOT_VIDYA_SETTLED_LOOKUP"
TIMEOUT_ENV = "AUTOPILOT_VIDYA_SETTLED_TIMEOUT_S"


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


__all__ = ["build_settled_ground_block"]
