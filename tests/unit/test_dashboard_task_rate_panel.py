"""W3b-C dual-report interim (2026-07-27, objective-task-rate-goodput.md W3c).

The Pareto panel payload must surface task_rate/goodput/tokens-per-solved (+
offered_load) telemetry per shipped point — computed via the CANONICAL helpers
in src.autopilot_core.tier_specs, joined from the same folded journal rows the
panel reconstructs from — plus a server-side divergence tripwire between the
legacy and task-rate objective policies. All of it is display-only: legacy
`median_request_tps` remains the live dominance vector.
"""
import asyncio
import json
from pathlib import Path

import src.api.routes.dashboard as dash
from src.autopilot_core.tier_specs import (
    goodput_qph_from_row,
    task_rate_qph_from_row,
)

DASHBOARD_HTML = Path(dash.__file__).with_name("dashboard.html")


def _row(tid, q, s, ts="2026-07-27T12:00:00+00:00", *, tier=1, cost=0.5, rel=1.0,
         n_questions=None, eval_wall_s=None, details=None, eval_concurrency=None):
    row = {
        "trial_id": tid, "tier": tier, "quality": q, "speed": s, "cost": cost,
        "reliability": rel, "timestamp": ts,
        "config_snapshot": {"type": "seed_batch", "n_questions": 10},
    }
    if n_questions is not None:
        row["n_questions"] = n_questions
    if eval_wall_s is not None:
        row["eval_wall_s"] = eval_wall_s
    eval_details = {}
    if details is not None:
        eval_details["details"] = details
    if eval_concurrency is not None:
        eval_details["eval_concurrency"] = eval_concurrency
    if eval_details:
        row["eval_details"] = eval_details
    return row


def _journal(tmp_path, monkeypatch, rows):
    p = tmp_path / "autopilot_journal.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(dash, "_AUTOPILOT_JOURNAL_PATH", p)
    return p


def _state(tmp_path, monkeypatch, **extra):
    p = tmp_path / "autopilot_state.json"
    data = {"trial_counter": 999}
    data.update(extra)
    p.write_text(json.dumps(data))
    monkeypatch.setattr(dash, "_AUTOPILOT_STATE_PATH", p)
    return p


def _call(scope="current", max_dominated=600):
    resp = asyncio.run(dash.pareto(max_dominated=max_dominated, scope=scope))
    return json.loads(resp.body)


def _points(payload):
    """Every shipped scatter point, keyed by trial_id."""
    pts = {}
    for lst in (
        payload["frontier"],
        payload["dominated"],
        payload["t0_audit"],
        *payload["frontiers_by_tier"].values(),
    ):
        for e in lst:
            pts[e["trial_id"]] = e
    return pts


# Fixture geometry (tier 1):
#   trial 1: legacy frontier (mid speed), SLOW eval  → task_rate 10 q/h
#   trial 2: legacy frontier (top quality), FAST eval → task_rate 100 q/h
#   trial 3: legacy frontier (top speed), SLOWEST eval → task_rate 5 q/h
# Under task_rate_3d_v1, trial 2 dominates 1 AND 3 → 2 legacy points drop.
ROW_1 = _row(1, 1.8, 50.0, n_questions=10, eval_wall_s=3600.0,
             details={"tokens_generated": 4500.0, "correct": 3},
             eval_concurrency=3)
ROW_2 = _row(2, 1.9, 40.0, n_questions=10, eval_wall_s=360.0,
             details={"tokens_generated": 1200.0, "correct": 6},
             eval_concurrency=3)
ROW_3 = _row(3, 1.7, 60.0, n_questions=10, eval_wall_s=7200.0)
# trial 4: dominated everywhere + NO eval-wall/n fields → null-safe path
ROW_4 = _row(4, 1.0, 10.0)


def test_pareto_payload_carries_task_rate_fields(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch, [ROW_1, ROW_2, ROW_3, ROW_4])
    _state(tmp_path, monkeypatch)
    pts = _points(_call())

    p1 = pts[1]
    assert p1["task_rate_qph"] == 10.0
    assert p1["goodput_qph"] == round((1.8 / 3.0) * 10.0, 2)
    assert p1["tokens_per_solved"] == 1500.0
    assert p1["offered_load"] == {"eval_concurrency": 3}

    # trial 3 has rate inputs but no token details → tokens_per_solved is None,
    # offered_load is None (no eval_concurrency recorded)
    p3 = pts[3]
    assert p3["task_rate_qph"] == 5.0
    assert p3["tokens_per_solved"] is None
    assert p3["offered_load"] is None


def test_pareto_payload_null_safe_when_row_lacks_fields(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch, [ROW_1, ROW_2, ROW_3, ROW_4])
    _state(tmp_path, monkeypatch)
    pts = _points(_call())

    p4 = pts[4]
    # Fields are PRESENT (schema-stable for the client) but explicitly null —
    # never 0.0, which would plot a fake point at 0 q/h.
    assert p4["task_rate_qph"] is None
    assert p4["goodput_qph"] is None
    assert p4["tokens_per_solved"] is None
    assert p4["offered_load"] is None


def test_toggle_data_integrity_matches_canonical_helpers(tmp_path, monkeypatch):
    """Every shipped point's task_rate/goodput equals the canonical helper output."""
    rows = [ROW_1, ROW_2, ROW_3, ROW_4]
    _journal(tmp_path, monkeypatch, rows)
    _state(tmp_path, monkeypatch)
    pts = _points(_call())
    rows_by_tid = {r["trial_id"]: r for r in rows}

    assert set(pts) == {1, 2, 3, 4}
    for tid, entry in pts.items():
        row = rows_by_tid[tid]
        expected_rate = task_rate_qph_from_row(row)
        if expected_rate > 0:
            assert entry["task_rate_qph"] == round(expected_rate, 2)
            assert entry["goodput_qph"] == round(goodput_qph_from_row(row), 2)
        else:
            assert entry["task_rate_qph"] is None
            assert entry["goodput_qph"] is None


def test_divergence_tripwire_reports_dropped_legacy_points(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch, [ROW_1, ROW_2, ROW_3])
    _state(tmp_path, monkeypatch)
    d = _call()

    div = d["task_rate_divergence"]
    assert div["error"] is None
    assert div["legacy_policy"] == "legacy_4d_v1"
    assert div["task_rate_policy"] == "task_rate_3d_v1"
    # All three points are legacy-frontier (quality/speed trade-off), but trial 2
    # dominates 1 and 3 on (quality, task_rate, reliability).
    assert div["legacy_frontier_trial_ids"] == [1, 2, 3]
    assert div["task_rate_frontier_trial_ids"] == [2]
    assert div["dropped_legacy_trial_ids"] == [1, 3]
    assert div["dropped_legacy_count"] == 2
    assert div["divergence_criterion_met"] is True

    # Dominance vector UNCHANGED: the shipped legacy frontier still holds all 3.
    assert sorted(e["trial_id"] for e in d["frontier"]) == [1, 2, 3]


def test_divergence_tripwire_quiet_when_frontiers_agree(tmp_path, monkeypatch):
    # Single admitted trial → identical frontiers under both policies.
    _journal(tmp_path, monkeypatch, [ROW_2])
    _state(tmp_path, monkeypatch)
    div = _call()["task_rate_divergence"]
    assert div["error"] is None
    assert div["legacy_frontier_trial_ids"] == [2]
    assert div["task_rate_frontier_trial_ids"] == [2]
    assert div["dropped_legacy_count"] == 0
    assert div["divergence_criterion_met"] is False


def test_divergence_summary_null_safe_without_rows():
    div = dash._task_rate_divergence_summary(None, {})
    assert div["error"] == "no journal rows available"
    assert div["dropped_legacy_count"] == 0
    assert div["divergence_criterion_met"] is False


def test_dashboard_html_ships_dual_report_ui():
    html = DASHBOARD_HTML.read_text()
    # One-line dual-report banner (W3b-C).
    assert "dominance = legacy t/s · task_rate telemetry live (W3b-C, flip armed on divergence)" in html
    # Speed-axis toggle with both labels; amber divergence badge; count note slot.
    assert 'id="pareto-speed-axis-toggle"' in html
    assert "median request t/s (end-to-end)" in html
    assert "task_rate (quality-units/h)" in html
    assert 'id="pareto-divergence-badge"' in html
    assert 'id="pareto-task-rate-note"' in html
    assert "setParetoSpeedAxis" in html
    assert "divergence criterion met — W3 flip decision live" in html
