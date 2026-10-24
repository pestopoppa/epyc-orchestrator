"""Parity tests for shared autopilot_core contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.autopilot.pareto_archive import ParetoArchive, ParetoEntry
from src.autopilot_core.action_identity import (
    config_fingerprint,
    config_fingerprint_from_row,
)
from src.autopilot_core.journal_reconstruction import (
    objectives_from_journal_row,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.tier_specs import (
    LEGACY_OBJECTIVE_POLICY,
    TASK_RATE_OBJECTIVE_POLICY,
    TIER_SPECS,
    goodput_qph_from,
    goodput_qph_from_row,
    task_rate_objectives_from,
    task_rate_objectives_from_row,
    task_rate_qph_from,
    task_rate_qph_from_row,
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


def test_tier_registry_includes_expert_workflow_lane() -> None:
    assert TIER_SPECS[3].label == "T3 (expert/hard workflow eval)"


def test_task_rate_shadow_objectives_from_eval_result() -> None:
    result = SimpleNamespace(
        quality=1.5,
        reliability=0.9,
        n_questions=40,
        eval_wall_s=720.0,
        details={},
    )

    assert task_rate_qph_from(result) == 200.0
    assert goodput_qph_from(result) == 100.0
    assert task_rate_objectives_from(result) == (1.5, 200.0, 0.9)


def test_task_rate_shadow_objectives_from_journal_row() -> None:
    row = _row(
        9,
        1.5,
        50.0,
        reliability=0.9,
        eval_details={
            "eval_wall_s": 720.0,
            "details": {"total": 40},
        },
    )

    assert task_rate_qph_from_row(row) == 200.0
    assert goodput_qph_from_row(row) == 100.0
    assert task_rate_objectives_from_row(row) == (1.5, 200.0, 0.9)


def test_journal_objective_policy_can_replay_task_rate_vector() -> None:
    row = _row(
        10,
        1.5,
        999.0,
        reliability=0.9,
        eval_details={
            "eval_wall_s": 720.0,
            "details": {"total": 40},
        },
    )

    assert objectives_from_journal_row(row) == [1.5, 999.0, -0.5, 0.9]
    assert objectives_from_journal_row(
        row,
        objective_policy=TASK_RATE_OBJECTIVE_POLICY,
    ) == [1.5, 200.0, 0.9]


def test_task_rate_reconstruction_drops_bloated_legacy_winner() -> None:
    rows = [
        _row(
            1,
            1.80,
            70.0,
            reliability=1.0,
            eval_details={
                "eval_wall_s": 900.0,
                "details": {"total": 50, "tokens_generated": 12000},
            },
        ),
        _row(
            2,
            1.80,
            65.0,
            reliability=1.0,
            eval_details={
                "eval_wall_s": 600.0,
                "details": {"total": 50, "tokens_generated": 8000},
            },
        ),
    ]

    legacy = reconstruct_archive_from_journal_rows(
        rows,
        None,
        current_run_only=False,
        objective_policy=LEGACY_OBJECTIVE_POLICY,
    )
    task_rate = reconstruct_archive_from_journal_rows(
        rows,
        None,
        current_run_only=False,
        objective_policy=TASK_RATE_OBJECTIVE_POLICY,
    )

    assert legacy is not None
    assert task_rate is not None
    assert {entry["trial_id"] for entry in legacy["frontier"]} == {1}
    assert {entry["trial_id"] for entry in task_rate["frontier"]} == {2}
    assert task_rate["objective_policy"] == TASK_RATE_OBJECTIVE_POLICY


def test_task_rate_reconstruction_does_not_deinflate_rate_axis() -> None:
    row = _row(
        1,
        1.80,
        70.0,
        reliability=1.0,
        eval_details={
            "eval_wall_s": 900.0,
            "details": {"total": 50},
        },
    )

    legacy = reconstruct_archive_from_journal_rows(
        [row],
        None,
        current_run_only=False,
        deinflate_before_ts=9999999999.0,
        deinflate_factor=0.5,
        objective_policy=LEGACY_OBJECTIVE_POLICY,
    )
    task_rate = reconstruct_archive_from_journal_rows(
        [row],
        None,
        current_run_only=False,
        deinflate_before_ts=9999999999.0,
        deinflate_factor=0.5,
        objective_policy=TASK_RATE_OBJECTIVE_POLICY,
    )

    assert legacy is not None
    assert task_rate is not None
    assert legacy["frontier"][0]["objectives"][1] == 35.0
    assert task_rate["frontier"][0]["objectives"] == [1.8, 200.0, 1.0]
    assert task_rate["frontier"][0]["speed_deinflated"] is False


def test_journal_reconstruction_excludes_rows_before_epoch_boundary() -> None:
    rows = [
        _row(
            1,
            1.90,
            120.0,
            timestamp="2026-06-26T21:59:00+00:00",
        ),
        _row(
            2,
            1.80,
            50.0,
            timestamp="2026-06-26T22:08:00+00:00",
        ),
    ]

    archive = reconstruct_archive_from_journal_rows(
        rows,
        None,
        current_run_only=False,
        exclude_before_ts=1782511631.0,
    )

    assert archive is not None
    assert [entry["trial_id"] for entry in archive["all_entries"]] == [2]
    assert [entry["trial_id"] for entry in archive["frontier"]] == [2]
    assert archive["exclusions"]["before_ts"] == {"count": 1, "max_trial_id": 1}
    assert archive["exclusions"]["exclude_before_ts"] == 1782511631.0


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


def test_representative_clusters_split_by_served_config_identity() -> None:
    """B5-M2: replay pools representatives by the same served-config identity the
    promotion gate counts reproductions by, so two prompt contents or two AP-55 regimes
    behind one ``{"type": "prompt_mutation", ...}`` action are never one median."""
    from src.autopilot_core.action_identity import row_config_identity
    from src.autopilot_core.journal_snapshot_replay import representative_replay_state_from_rows

    action = {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"}

    def served(trial_id: int, sha_char: str, regime: str) -> dict:
        return _row(
            trial_id,
            1.80,
            50.0 + trial_id,
            tier=2,
            config=dict(action),
            eval_details={
                "learning_exclusion": {"by": "mad_noise"},
                "served_content": {"files": {"frontdoor.md": sha_char * 64}},
                "infra_regime_digest": regime,
            },
        )

    rows = [served(1, "a", "r1"), served(2, "b", "r1"), served(3, "a", "r2"), served(4, "a", "r1")]
    identities = {row_config_identity(row) for row in rows}
    assert len(identities) == 3 and None not in identities

    archive = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)
    assert archive is not None
    reps = [e for e in archive["all_entries"] if e.get("is_representative")]
    assert {e["config_fingerprint"] for e in reps} == identities
    assert sorted(e["n_reproductions"] for e in reps) == [1, 1, 2]

    state = representative_replay_state_from_rows(rows)
    assert {c["fingerprint"] for c in state["clusters"]} == identities

    # Identity-less measurement rows (seed_batch) still cluster on the action fingerprint.
    seed = [_row(t, 1.8, 50.0, eval_details={"learning_exclusion": {"by": "mad_noise"}}) for t in (5, 6)]
    seed_archive = reconstruct_archive_from_journal_rows(seed, None, current_run_only=False)
    (seed_rep,) = [e for e in seed_archive["all_entries"] if e.get("is_representative")]
    assert seed_rep["config_fingerprint"] == config_fingerprint_from_row(seed[0])
    assert seed_rep["n_reproductions"] == 2


def test_reconstruction_counts_rows_dropped_for_unmeasured_objectives() -> None:
    """B5-12: a row whose objective tuple cannot be built (unmeasured axis) is counted."""
    rows = [_row(1, 1.5, 40.0), _row(2, 1.6, None), _row(3, 1.7, 50.0), _row(4, 1.8, None)]
    archive = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)
    assert archive is not None
    assert archive["exclusions"]["unmeasured_objectives"] == {"count": 2, "max_trial_id": 4}
    assert {e["trial_id"] for e in archive["all_entries"]} == {1, 3}
