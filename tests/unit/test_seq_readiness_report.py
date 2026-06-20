from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.autopilot import seq_readiness_report


def _question_results(
    correct_qids: set[int],
    *,
    n: int = 40,
) -> list[dict]:
    return [
        {"qid": f"q{i:02d}", "suite": "suite", "correct": i in correct_qids}
        for i in range(n)
    ]


def _row(
    trial_id: int,
    fingerprint: str,
    correct_qids: set[int],
    *,
    learning_exclusion: str | None = None,
    seq_state: str | None = None,
    corrupt: str | None = None,
) -> dict:
    eval_details: dict = {
        "config_fingerprint": fingerprint,
        "question_results": _question_results(correct_qids),
    }
    if learning_exclusion:
        eval_details["learning_exclusion"] = {"by": learning_exclusion, "reason": "fixture"}
    row: dict = {
        "trial_id": trial_id,
        "tier": 1,
        "quality": round(len(correct_qids) / 40 * 3, 6),
        "outcome_status": "ok",
        "eval_details": eval_details,
    }
    if seq_state:
        row["seq"] = {
            "candidate": fingerprint,
            "core_id": "core_v1",
            "z": 0.25,
            "state": seq_state,
            "policy_version": "seq-v1",
        }
    if corrupt:
        row["bug_corrupted_by"] = corrupt
    return row


def test_report_excludes_corrupted_vectors_from_trusted_history() -> None:
    rows = [
        _row(1, "fp-a", set(range(20)), corrupt="resource_contention_20260612"),
        _row(2, "fp-a", set(range(22)), learning_exclusion="mad_noise"),
    ]

    report = seq_readiness_report.build_seq_readiness_report(
        rows,
        min_trusted_vector_trials=2,
        min_seq_shadow_rows=0,
        min_shared_qids=1,
    )

    assert report["raw_vector_trials"] == 2
    assert report["trusted_vector_trials"] == 1
    assert report["untrusted_vector_trials"] == 1
    assert report["untrusted_vector_trial_ids"] == [1]
    assert any("trusted vector history too small" in b for b in report["cutover_blockers"])


def test_report_excludes_invalid_and_skipped_vectors_from_trusted_history() -> None:
    invalid = _row(1, "fp-a", set(range(20)))
    invalid["outcome_status"] = "invalid"
    skipped = _row(2, "fp-a", set(range(21)))
    skipped["outcome_status"] = "skipped"
    trusted = _row(3, "fp-b", set(range(22)))

    report = seq_readiness_report.build_seq_readiness_report(
        [invalid, skipped, trusted],
        min_trusted_vector_trials=2,
        min_seq_shadow_rows=0,
        min_shared_qids=1,
    )

    assert report["raw_vector_trials"] == 3
    assert report["trusted_vector_trials"] == 1
    assert report["untrusted_vector_trials"] == 2
    assert report["untrusted_vector_trial_ids"] == [1, 2]
    assert report["candidate_clusters"][0]["trusted_vector_trials"] == [3]


def test_report_blocks_cutover_without_seq_shadow_denominator() -> None:
    rows = [
        _row(1, "fp-a", set(range(20))),
        _row(2, "fp-a", set(range(21))),
        _row(3, "fp-b", set(range(10, 31))),
    ]

    report = seq_readiness_report.build_seq_readiness_report(
        rows,
        min_trusted_vector_trials=3,
        min_seq_shadow_rows=1,
        min_shared_qids=35,
    )

    assert not report["cutover_ready"]
    assert report["seq_shadow"]["seq_shadow_rows"] == 0
    assert "no seq-vs-legacy flip-rate denominator yet" in report["cutover_blockers"]
    assert report["pairwise_replays"][0]["shared_qids"] == 39


def test_report_can_be_cutover_ready_with_sufficient_shadow_disagreement() -> None:
    rows = [
        _row(
            1,
            "fp-a",
            set(range(20)),
            learning_exclusion="mad_noise",
            seq_state="confirmed",
        ),
        _row(
            2,
            "fp-a",
            set(range(21)),
            learning_exclusion="reproduction_confirmed",
            seq_state="confirmed",
        ),
        _row(3, "fp-b", set(range(10, 31)), seq_state="refuted"),
        _row(4, "fp-b", set(range(11, 32)), seq_state="refuted"),
    ]

    report = seq_readiness_report.build_seq_readiness_report(
        rows,
        min_trusted_vector_trials=4,
        min_seq_shadow_rows=4,
        min_flip_rate=0.30,
        min_shared_qids=35,
    )

    assert report["cutover_ready"]
    assert report["seq_shadow"]["seq_shadow_rows"] == 4
    assert report["seq_shadow"]["disagreements"] == 4
    assert report["seq_shadow"]["flip_rate"] == 1.0
    assert report["cutover_blockers"] == []


def test_main_strict_returns_nonzero_when_blocked(tmp_path: Path, capsys) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    journal.write_text(json.dumps(_row(1, "fp-a", set(range(20)))) + "\n")

    code = seq_readiness_report.main(
        [
            "--journal",
            str(journal),
            "--strict",
            "--min-trusted-vector-trials",
            "2",
            "--min-seq-shadow-rows",
            "1",
        ]
    )

    assert code == 1
    assert "Status: blocked" in capsys.readouterr().out


def test_main_writes_json_and_markdown_outputs(tmp_path: Path, capsys) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    journal.write_text(json.dumps(_row(1, "fp-a", set(range(20)))) + "\n")
    out_json = tmp_path / "reports" / "seq.json"
    out_md = tmp_path / "reports" / "seq.md"

    code = seq_readiness_report.main(
        [
            "--journal",
            str(journal),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert code == 0
    stdout = capsys.readouterr().out
    assert "Status: blocked" in stdout
    assert json.loads(out_json.read_text())["cutover_ready"] is False
    assert "Status: blocked" in out_md.read_text()


def test_direct_cli_execution_from_repo_root(tmp_path: Path) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    journal.write_text(json.dumps(_row(1, "fp-a", set(range(20)))) + "\n")
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/autopilot/seq_readiness_report.py",
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
    assert report["cutover_ready"] is False
    assert report["trusted_vector_trials"] == 1
