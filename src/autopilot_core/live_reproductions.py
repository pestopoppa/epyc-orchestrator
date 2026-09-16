"""Live-regime reproductions of one config (operator decision (b), 2026-09-16).

While the live-epoch frontier of a tier is empty and the tier already has a baseline, a
promotion needs at least N independent reproductions of the candidate config under the live
regime. A reproduction is a journal row that:

* is in the live epoch (timestamp at or after ``exclude_before_ts``);
* is a representative-cluster member (``row_is_representative_member``: a trusted
  within-noise row, or a clean row stamped as a frontier representative);
* matches the tier and the candidate's SERVED-CONFIG identity (``row_config_identity``:
  an explicit flags/params delta, plus the AP-55 infra digest when recorded). Measurement
  actions (``seed_batch``, ``deep_eval``, …) and un-resolved mutation requests have no such
  identity and never count (gate-frontier re-review B1);
* measured every live dominance axis (its live-policy objective tuple builds);
* does NOT carry an AP-55 ``NON_COMPARABLE`` verdict (``COMPARABLE``, ``UNVERIFIED`` and
  rows written before AP-55 existed all count).

Pure and zero-inference: it reads journal rows only.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.autopilot_core.action_identity import row_config_identity
from src.autopilot_core.journal_reconstruction import (
    fold_supersession_events,
    objectives_from_journal_row,
    parse_journal_ts,
)
from src.autopilot_core.learning_exclusions import row_is_representative_member

NON_COMPARABLE = "NON_COMPARABLE"


def row_comparability_status(row: dict[str, Any]) -> str:
    """AP-55 verdict recorded on a journal row ('' when the row predates AP-55)."""
    comparability = row.get("comparability")
    if isinstance(comparability, dict) and comparability.get("status"):
        return str(comparability["status"])
    details = row.get("eval_details")
    if isinstance(details, dict) and details.get("infra_comparability"):
        return str(details["infra_comparability"])
    return ""


def live_reproductions(
    rows: Iterable[dict[str, Any]],
    *,
    tier: int,
    identity: str,
    objective_policy: str,
    exclude_before_ts: float | None,
) -> list[dict[str, Any]]:
    """Reproductions of served-config ``identity`` at ``tier`` in the live regime, by trial id."""
    if not identity:
        return []
    folded, _meta = fold_supersession_events(list(rows))
    found: dict[int, dict[str, Any]] = {}
    for row in folded:
        if "trial_id" not in row:
            continue
        try:
            trial_id = int(row["trial_id"])
            row_tier = int(row.get("tier", -1))
        except (TypeError, ValueError):
            continue
        if row_tier != int(tier) or not row_is_representative_member(row):
            continue
        if exclude_before_ts is not None:
            ts = parse_journal_ts(row.get("timestamp"))
            if ts is None or ts < exclude_before_ts:
                continue
        status = row_comparability_status(row)
        if status == NON_COMPARABLE:
            continue
        if row_config_identity(row) != identity:
            continue
        objectives = objectives_from_journal_row(row, objective_policy=objective_policy)
        if objectives is None:
            continue
        found[trial_id] = {
            "trial_id": trial_id,
            "objectives": tuple(objectives),
            "comparability": status or "UNRECORDED",
        }
    return [found[t] for t in sorted(found)]
