"""VB-HIL-WRITE producer call at the sibling loop's finalize boundary.

The orchestrator-side finalize hook already exists: ``scripts/benchmark/hil_ab_sidecar.py``
``finalize_hil_run`` (orchestrator ``5195ad53``) computes nothing new here; it calls the root
carrier ``scripts/vidya/adapters/harness_improvement.write_belief_measurements``. This module only
projects the loop's own :class:`~src.harness_loop.interfaces.ArmResult` rows into the carrier's
observation fields, so the loop writes its belief rows at finalize rather than having them
reconstructed from journal prose later (CLAUDE.md, Belief Kernel). It invents no field: a value
the run did not capture is left out and grades down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.harness_loop.interfaces import ArmResult, ArmSpec

#: The root carrier's closed arm vocabulary is {BASELINE, CANDIDATE} and it refuses a duplicate
#: run x arm x feature x objective row. One finalize call therefore carries ONE paired comparison
#: (incumbent vs candidate); each control arm (HIL-2 items 1 and 6) is finalized in its own call
#: under its own ``run_id`` as a BASELINE row, never folded into the paired run.
CARRIER_ARM = {
    "incumbent": "BASELINE",
    "unmodified_control": "BASELINE",
    "matched_rollout_baseline": "BASELINE",
    "candidate": "CANDIDATE",
}


def observation(
    result: ArmResult,
    arm: ArmSpec,
    *,
    objective: str,
    suite_id: str,
    suite_version: str,
    split: str,
    native_verdict: str,
    prompt_param_path: str,
    harness_pin: str,
    config_sha256: str,
    identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One carrier observation (run x arm x feature x objective metric)."""
    if result.arm_id != arm.arm_id:
        raise ValueError(f"result {result.arm_id} is not arm {arm.arm_id}")
    if arm.role not in CARRIER_ARM:
        raise ValueError(f"arm role {arm.role!r} has no carrier arm label")
    if result.value is None:
        raise ValueError(f"arm {arm.arm_id}: no value; a missing measurement is not a row")
    obs: dict[str, Any] = {
        "feature_id": arm.feature_id,
        "arm": CARRIER_ARM[arm.role],
        "objective": objective,
        "metric": result.metric,
        "value": result.value,
        "unit": result.unit,
        "metric_direction": result.direction,
        "native_verdict": native_verdict,
        "suite_id": suite_id,
        "suite_version": suite_version,
        "split": split,
        "scored": result.scored,
        "attempted": result.attempted,
        "prompt_param_path": prompt_param_path,
        "prompt_param_sha256": arm.content_sha256,
        "harness_pin": harness_pin,
        "config_sha256": config_sha256,
    }
    if identities:
        obs["identities"] = dict(identities)
    if arm.pins:
        obs["arm_pins"] = dict(arm.pins)
    return obs


def _default_finalize() -> Callable[..., Path]:
    from scripts.benchmark.hil_ab_sidecar import finalize_hil_run

    return finalize_hil_run


def finalize_run(
    out_dir: Path | str,
    *,
    run_id: str,
    results_path: Path | str,
    observations: Sequence[Mapping[str, Any]],
    pins: Mapping[str, Any],
    protocol_id: str | None = None,
    epyc_root: Path | str | None = None,  # None: hil_ab_sidecar resolves $EPYC_ROOT / root-main view
    finalize_fn: Callable[..., Path] | None = None,
) -> Path:
    """Call the VB-HIL-WRITE finalize hook once, after the raw per-item result file is closed.

    ``protocol_id`` stays ``None`` unless an owning human-ratified protocol applies; rows are then
    observations (``claim_tuple.grade()`` decides, no new ladder).
    """
    if not Path(results_path).is_file():
        raise FileNotFoundError(f"raw result file {results_path} is not closed on disk")
    keys = [(o.get("arm"), o.get("feature_id"), o.get("objective")) for o in observations]
    if len(set(keys)) != len(keys):
        raise ValueError("one finalize call carries one paired comparison; a duplicate "
                         "arm x feature x objective belongs in its own run_id")
    fn = finalize_fn or _default_finalize()
    return fn(out_dir, run_id=run_id, results_path=results_path, observations=list(observations),
              pins=dict(pins), protocol_id=protocol_id, epyc_root=epyc_root)
