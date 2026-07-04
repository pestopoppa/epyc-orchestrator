from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.autopilot import w8_promotion_trajectory_report


def _row(
    trial_id: int,
    candidate: str,
    *,
    combined: float = 0.95,
    state: str = "accumulating",
    k: int = 1,
    fresh_eval: bool = False,
    confirmed: bool = False,
    finalized: bool = False,
    keep_revert_decision: str = "",
) -> dict:
    return {
        "trial_id": trial_id,
        "action_type": "numeric_trial",
        "quality": 2.1,
        "keep_revert_decision": keep_revert_decision,
        "seq": {
            "candidate": candidate,
            "state": state,
            "confirmed": confirmed,
            "baseline_promotion_combined_E": combined,
            "baseline_promotion_required_E": 100.0,
            "baseline_promotion_fresh_eval": fresh_eval,
            "baseline_promotion_finalized": finalized,
            "baseline_reference_state": "fresh",
            "E_quality": 1.1,
            "E_rate_noninf": combined,
            "k": k,
            "r_eff": 30,
        },
    }


def test_report_identifies_recent_replay_progress() -> None:
    report = w8_promotion_trajectory_report.build_w8_trajectory_report(
        [
            _row(10, "candidate-a", combined=0.91, k=1),
            _row(12, "candidate-a", combined=0.93, k=2),
            _row(13, "candidate-b", combined=0.88, k=1),
        ],
        stale_trials=3,
    )

    assert report["status"] == "progressing"
    assert report["ok"] is True
    assert report["recent_active_candidates"] == ["candidate-a"]
    trajectory = report["trajectories"][0]
    assert trajectory["candidate"] == "candidate-a"
    assert trajectory["status"] == "active_recent_replay"
    assert trajectory["attempts"] == 2
    assert trajectory["combined_E_delta"] == 0.02
    assert report["replay_concentration"]["warning"] is False


def test_report_flags_stale_accumulating_candidates() -> None:
    report = w8_promotion_trajectory_report.build_w8_trajectory_report(
        [
            _row(1, "candidate-a", combined=0.91, k=1),
            _row(20, "candidate-b", combined=0.92, state="refuted", k=12),
        ],
        stale_trials=5,
    )

    assert report["status"] == "stale_accumulating"
    assert report["ok"] is False
    assert report["stale_accumulating_candidates"] == ["candidate-a"]
    assert "stale_accumulating_candidates_present" in report["open_requirements"]


def test_report_flags_concentrated_replay_attempts() -> None:
    report = w8_promotion_trajectory_report.build_w8_trajectory_report(
        [
            _row(1, "stale-a", combined=0.91, k=1),
            _row(10, "candidate-a", combined=0.91, k=1),
            _row(12, "candidate-a", combined=0.92, k=2),
            _row(13, "candidate-a", combined=0.93, k=3),
        ],
        stale_trials=4,
    )

    concentration = report["replay_concentration"]
    assert report["status"] == "progressing"
    assert concentration["warning"] is True
    assert concentration["top_active_candidate"] == "candidate-a"
    assert concentration["top_active_attempt_share"] == 1.0
    assert concentration["stale_accumulating_count"] == 1
    assert "replay_concentration_warning" in report["open_requirements"]
    assert "Replay Concentration" in w8_promotion_trajectory_report.render_markdown(report)


def test_report_excludes_latest_reverted_candidate_from_active_replay() -> None:
    report = w8_promotion_trajectory_report.build_w8_trajectory_report(
        [
            _row(10, "candidate-a", combined=0.91, k=1),
            _row(
                12,
                "candidate-a",
                combined=0.93,
                k=2,
                keep_revert_decision="revert",
            ),
            _row(13, "candidate-b", combined=0.88, state="refuted", k=12),
        ],
        stale_trials=5,
    )

    statuses = {item["candidate"]: item["status"] for item in report["trajectories"]}
    assert statuses["candidate-a"] == "reverted"
    assert report["recent_active_candidates"] == []
    assert report["status_counts"]["reverted"] == 1
    assert "no_recent_multi_observation_accumulating_candidate" in report[
        "open_requirements"
    ]


def test_report_classifies_refuted_and_finalized_candidates() -> None:
    report = w8_promotion_trajectory_report.build_w8_trajectory_report(
        [
            _row(1, "candidate-a", state="refuted", k=12),
            _row(2, "candidate-b", state="confirmed", confirmed=True, finalized=True),
        ]
    )

    statuses = {item["candidate"]: item["status"] for item in report["trajectories"]}
    assert statuses == {
        "candidate-a": "refuted",
        "candidate-b": "finalized",
    }


def test_main_writes_json_and_markdown_outputs(tmp_path: Path, capsys) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    journal.write_text(
        "\n".join(
            [
                json.dumps(_row(10, "candidate-a", combined=0.91, k=1)),
                json.dumps(_row(11, "candidate-a", combined=0.92, k=2)),
            ]
        )
        + "\n"
    )
    out_json = tmp_path / "reports" / "w8.json"
    out_md = tmp_path / "reports" / "w8.md"

    code = w8_promotion_trajectory_report.main(
        [
            "--journal",
            str(journal),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--strict",
        ]
    )

    assert code == 0
    stdout = capsys.readouterr().out
    assert "Status: progressing" in stdout
    assert json.loads(out_json.read_text())["status"] == "progressing"
    assert "candidate-a" in out_md.read_text()


def test_direct_cli_execution_from_repo_root(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    journal.write_text(
        json.dumps(_row(1, "candidate-a", k=1))
        + "\n"
        + json.dumps(_row(2, "candidate-a", k=2))
        + "\n"
    )
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/autopilot/w8_promotion_trajectory_report.py",
            "--journal",
            str(journal),
            "--json",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(result.stdout)
    assert report["candidate_count"] == 1
    assert report["status"] == "progressing"
