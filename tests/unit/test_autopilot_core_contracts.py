"""Parity tests for shared autopilot_core contracts."""

from __future__ import annotations

from pathlib import Path

from scripts.autopilot.pareto_archive import ParetoArchive, ParetoEntry
from src.autopilot_core.action_identity import (
    config_fingerprint,
    config_fingerprint_from_row,
)
from src.autopilot_core.journal_reconstruction import (
    objectives_from_journal_row,
    reconstruct_archive_from_journal_rows,
)


def _row(
    trial_id: int,
    quality: float,
    speed: float,
    *,
    tier: int = 1,
    cost: float = 0.5,
    reliability: float = 1.0,
    config: dict | None = None,
    **extra,
) -> dict:
    row = {
        "trial_id": trial_id,
        "tier": tier,
        "quality": quality,
        "speed": speed,
        "cost": cost,
        "reliability": reliability,
        "timestamp": f"2026-06-04T10:{trial_id:02d}:00+00:00",
        "config_snapshot": config or {"type": "seed_batch", "n_questions": 10},
    }
    row.update(extra)
    return row


def test_action_fingerprint_ignores_narrative_keys() -> None:
    base = {"type": "seed_batch", "n_questions": 10}
    narrated = {
        "n_questions": 10,
        "type": "seed_batch",
        "description": "different story",
        "hypothesis": "h",
        "reasoning": "r",
        "expected_mechanism": "m",
    }

    assert config_fingerprint(base) == config_fingerprint(narrated)
    assert config_fingerprint(base) == config_fingerprint_from_row({
        "config_snapshot": narrated,
    })


def test_journal_reconstruction_matches_representative_archive(tmp_path: Path) -> None:
    rows = [
        _row(1, 1.90, 45.0, reliability=0.97),
        _row(10, 1.80, 50.0, bug_corrupted_by="mad_noise"),
        _row(11, 1.80, 70.0, bug_corrupted_by="mad_noise"),
        _row(12, 1.80, 60.0, eval_details={"learning_exclusion": {"by": "mad_noise"}}),
    ]

    reconstructed = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)
    assert reconstructed is not None

    archive = ParetoArchive(state_path=tmp_path / "state.json")
    for row in rows:
        objectives = objectives_from_journal_row(row)
        assert objectives is not None
        if row.get("bug_corrupted_by") == "mad_noise" or (
            row.get("eval_details") or {}
        ).get("learning_exclusion", {}).get("by") == "mad_noise":
            archive.upsert_representative(
                config_fingerprint_from_row(row),
                int(row["tier"]),
                tuple(objectives),
                trial_id=int(row["trial_id"]),
            )
        else:
            archive.update(
                ParetoEntry(
                    trial_id=int(row["trial_id"]),
                    objectives=tuple(objectives),
                    eval_tier=int(row["tier"]),
                )
            )

    reconstructed_points = sorted(
        tuple(round(float(value), 4) for value in entry["objectives"])
        for entry in reconstructed["frontier"]
    )
    archive_points = sorted(
        tuple(round(float(value), 4) for value in entry.objectives)
        for entry in archive.frontier(1)
    )
    assert reconstructed_points == archive_points


def test_benign_within_noise_rows_admit_one_median_representative() -> None:
    rows = [
        _row(1, 1.80, 50.0, eval_details={"learning_exclusion": {"by": "mad_noise"}}),
        _row(
            2,
            1.80,
            70.0,
            eval_details={"learning_exclusion": {"by": "reproduction_confirmed"}},
        ),
        _row(3, 1.80, 60.0, bug_corrupted_by="mad_noise"),
    ]

    archive = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)
    assert archive is not None

    reps = [entry for entry in archive["frontier"] if entry.get("is_representative")]
    assert len(reps) == 1
    assert reps[0]["n_reproductions"] == 3
    assert reps[0]["objectives"][1] == 60.0


def test_genuine_corruption_remains_excluded() -> None:
    rows = [
        _row(1, 1.50, 40.0),
        _row(2, 9.90, 99.0, bug_corrupted_by="exogenous_operator_reload"),
        _row(3, 9.90, 99.0, bug_corrupted_by="deadbeefcafe1234"),
    ]

    archive = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)
    assert archive is not None
    assert {entry["trial_id"] for entry in archive["all_entries"]} == {1}
    assert {entry["trial_id"] for entry in archive["frontier"]} == {1}


def test_reconstruction_reports_bug_corruption_exclusions() -> None:
    """A mass of corruption-tagged trials must be COUNTED, not silently dropped.

    This is the telemetry that lets the dashboard explain "frozen at ~700 while
    1000 trials ran" — trials 701+ tagged ``bug_corrupted_by`` vanish from the
    frontier, and the operator needs to see that they were excluded, not lost.
    """
    rows = [
        _row(1, 1.50, 40.0),
        _row(2, 1.60, 45.0, bug_corrupted_by="deadbeefcafe1234"),
        _row(3, 1.70, 50.0, bug_corrupted_by="exogenous_operator_reload"),
        _row(4, 1.80, 55.0),
    ]
    archive = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)
    assert archive is not None
    excl = archive["exclusions"]["bug_corrupted"]
    assert excl["count"] == 2
    assert excl["max_trial_id"] == 3
    assert archive["journal_max_trial_id"] == 4
    assert {e["trial_id"] for e in archive["all_entries"]} == {1, 4}


def test_reconstruction_uncapped_includes_newer_rows_and_reports_cap() -> None:
    """The hardened dashboard default (no cap) shows every journaled trial; when
    a cap IS applied it must report what it truncated so a stale state counter
    is detectable instead of silently hiding newer rows."""
    rows = [_row(i, 1.5 + i * 0.01, 40.0 + i) for i in range(1, 6)]  # trials 1..5

    uncapped = reconstruct_archive_from_journal_rows(rows, None, current_run_only=True)
    assert uncapped["journal_max_trial_id"] == 5
    assert uncapped["exclusions"]["truncated_above_cap"]["count"] == 0
    assert max(e["trial_id"] for e in uncapped["all_entries"]) == 5

    capped = reconstruct_archive_from_journal_rows(
        rows, None, current_run_only=True, max_trial_id=3
    )
    assert max(e["trial_id"] for e in capped["all_entries"]) == 3
    trunc = capped["exclusions"]["truncated_above_cap"]
    assert trunc["count"] == 2
    assert trunc["max_trial_id"] == 5
    assert capped["exclusions"]["max_trial_id_cap"] == 3
    # journal_max_trial_id tracks the FULL segment, above the cap — that is how
    # a caller spots staleness (counter 3 < journal max 5).
    assert capped["journal_max_trial_id"] == 5
