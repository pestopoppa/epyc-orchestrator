"""Prospective producer for CRM pair event custody and atomic terminal seals."""
from __future__ import annotations

from pathlib import Path

from .crm_pair import run_pair
from .crm_pair_seal import seal_pair, validate_provenance
from .crm_receipt import Receipt
from .crm_source_manifest import verify_manifest


def capture_pair(endpoint, services, *, seed, journal_path, seal_path, capture_kind,
                 provenance, grant_receipt_path, source_manifest_path, max_turns=8, should_stop=None):
    """Capture native facts; this does not establish grant/model authority or grade."""
    validate_provenance(capture_kind, provenance)
    for path in (grant_receipt_path, source_manifest_path):
        if not Path(path).is_file() or not Path(path).stat().st_size:
            raise ValueError(f"required custody input missing or empty: {path}")
    if capture_kind == "live":
        verify_manifest(source_manifest_path, provenance["source_commit"], provenance["source_tree_sha256"])
    receipt = Receipt(Path(journal_path), {"capture_kind": capture_kind, "provenance": provenance})
    try:
        result = run_pair(endpoint, services, seed=seed, max_turns=max_turns, receipt=receipt,
                          should_stop=should_stop)
        arms = {name: {"status": arm["status"], "delivery": arm["delivery"]}
                for name, arm in result["arms"].items()}
        for name in ("control", "attack"):
            if name not in arms:
                arms[name] = {"status": "not_attempted", "delivery": "not_observed"}
                receipt.append("arm_skipped", {"arm": name, "reason": "prior_arm_failed",
                                               "delivery": "not_observed"})
        receipt.append("terminal", {"arms": arms, "cleanup": result["cleanup"]})
    finally:
        receipt.close()
    # A provider close receipt cannot exist until this child has exited and the
    # region-lock context releases its flock. Live sealing is a parent step.
    if capture_kind == "live":
        return result, None
    seal = seal_pair(journal_path=Path(journal_path), output_path=Path(seal_path),
                     capture_kind=capture_kind, provenance=provenance,
                     grant_receipt_path=Path(grant_receipt_path),
                     source_manifest_path=Path(source_manifest_path), producer_path=Path(__file__))
    return result, seal
