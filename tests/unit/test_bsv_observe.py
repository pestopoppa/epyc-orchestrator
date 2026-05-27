"""Unit tests for J11/BSV-2 observe-only payload (scripts/autopilot/bsv_observe.py).

Pure: no autopilot loop, no inference. Verifies the coarse trial-level signature, the partial-
confidence policy, and the diff severities the observe-only run would journal.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# scripts/autopilot is not a package / not on the default test path (mirror autopilot.py's runtime).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "autopilot"))

from bsv_observe import compute_bsv_observe_payload, _suite_outcomes, SUITE_PASS_QUALITY  # noqa: E402


def _er(**kw) -> SimpleNamespace:
    base = dict(routing_distribution={}, per_suite_quality={}, oracle_adequacy={},
                avg_prompt_tokens=0.0, metric_schema_version=1)
    base.update(kw)
    return SimpleNamespace(**base)


def test_suite_outcomes_proxy_uses_bsv_vocab():
    out = _suite_outcomes({"qa": 2.6, "coding": 1.0, "math": SUITE_PASS_QUALITY})
    assert out == {"qa": "pass", "coding": "fail", "math": "pass"}  # >= threshold => pass
    assert _suite_outcomes(None) == {}


def test_no_incumbent_has_no_severity():
    p = compute_bsv_observe_payload(
        _er(routing_distribution={"frontdoor": 1.0}), species_name="seeder", trial_id=1,
        incumbent_signature=None,
    )
    assert p["severity"] is None
    assert p["compared_to_incumbent"] is False
    assert p["signature_confidence"] == "partial"
    assert "route_path_hash" in p["signature"]


def test_identical_partial_signature_is_watch_not_benign():
    # Two identical trials: nothing changed, but partial-confidence cannot certify BENIGN.
    er = _er(routing_distribution={"frontdoor": 1.0}, per_suite_quality={"qa": 2.6})
    inc = compute_bsv_observe_payload(er, species_name="s", trial_id=1, incumbent_signature=None)["signature"]
    p = compute_bsv_observe_payload(er, species_name="s", trial_id=2, incumbent_signature=inc)
    assert p["severity"] == "watch"
    assert any("partial" in r for r in p["reasons"])


def test_suite_regression_is_blocking():
    inc = compute_bsv_observe_payload(
        _er(per_suite_quality={"coding": 2.6}), species_name="s", trial_id=1, incumbent_signature=None,
    )["signature"]
    p = compute_bsv_observe_payload(
        _er(per_suite_quality={"coding": 1.0}), species_name="s", trial_id=2, incumbent_signature=inc,
    )
    assert p["severity"] == "blocking"
    assert any("coding" in r for r in p["reasons"])


def test_routing_change_flags_at_least_watch():
    inc = compute_bsv_observe_payload(
        _er(routing_distribution={"frontdoor": 1.0}, per_suite_quality={"qa": 2.6}),
        species_name="s", trial_id=1, incumbent_signature=None,
    )["signature"]
    p = compute_bsv_observe_payload(
        _er(routing_distribution={"worker_coder": 1.0}, per_suite_quality={"qa": 2.6}),
        species_name="s", trial_id=2, incumbent_signature=inc,
    )
    assert p["severity"] in ("watch", "blocking")
    assert any("route_path_hash" in r for r in p["reasons"])


def test_graceful_on_empty_eval_result():
    # Missing fields must not raise — observe-only must never disrupt the trial loop.
    p = compute_bsv_observe_payload(SimpleNamespace(), species_name="", trial_id=0, incumbent_signature=None)
    assert p["severity"] is None
    assert "signature" in p and p["signature_confidence"] == "partial"
