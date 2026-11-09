"""AP-29b(b): per-role capture helper and the metric_schema_version bump."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "autopilot"))

from hle_metrics import METRIC_SCHEMA_VERSION  # noqa: E402
from role_cost_capture import role_cost_capture  # noqa: E402


def test_aggregates_by_route_and_marks_missing_baseline():
    rows = [
        {"route": "frontdoor", "tokens_generated": 100, "latency_ms": 2000},
        {"route": "Frontdoor", "tokens_generated": 50, "latency_ms": 1000},
        {"route": "architect_general", "tokens_generated": 10, "latency_ms": 500},
        {"tokens_generated": 5, "latency_ms": 100},
    ]
    out = role_cost_capture(rows, {"frontdoor": 12.7})
    assert out["per_role_tokens"] == {"frontdoor": 150, "architect_general": 10, "unknown": 5}
    assert out["per_role_elapsed_s"]["frontdoor"] == 3.0
    assert out["baseline_tps"] == {"frontdoor": 12.7, "architect_general": None, "unknown": None}


def test_empty_input_is_empty_not_error():
    assert role_cost_capture(None) == {
        "per_role_tokens": {}, "per_role_elapsed_s": {}, "baseline_tps": {},
    }


def test_schema_version_bumped_to_2():
    assert METRIC_SCHEMA_VERSION == 2


def test_safe_capture_records_failure_and_skips_v1():
    from types import SimpleNamespace

    from role_cost_capture import safe_role_cost_capture

    bad = SimpleNamespace(question_results=[{"route": "x", "tokens_generated": "abc"}])
    out = safe_role_cost_capture(bad, 2)
    assert set(out) == {"role_cost_capture_error"} and "ValueError" in out["role_cost_capture_error"]
    assert safe_role_cost_capture(bad, 1) == {}
    ok = SimpleNamespace(question_results=[{"route": "x", "tokens_generated": 3, "latency_ms": 10}])
    assert safe_role_cost_capture(ok, 2)["per_role_tokens"] == {"x": 3}
