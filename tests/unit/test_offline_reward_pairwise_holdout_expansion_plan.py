from __future__ import annotations

import json
from pathlib import Path

from scripts.graph_router import plan_offline_reward_pairwise_holdout_expansion as mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _manifest_row(
    *,
    source_path: str,
    role_key: str,
    expected_sha256: str = "expected-hash",
    prompt_sha256: str = "prompt-hash",
) -> dict:
    return {
        "schema_version": "offline_reward_feature_input.v1",
        "item_id": f"{Path(source_path).stem}:1:{role_key}",
        "question_id": "q-live",
        "suite": "livecodebench",
        "role_key": role_key,
        "source_path": source_path,
        "source_record_offset": 0,
        "prompt_sha256": prompt_sha256,
        "expected_sha256": expected_sha256,
        "feature_context": {"source_family": "seeding_eval"},
    }


def _result_record() -> dict:
    return {
        "question_id": "q-live",
        "suite": "livecodebench",
        "prompt": "Solve task",
        "expected": "42",
        "role_results": {
            "frontdoor:direct": {
                "answer": "42",
                "passed": True,
                "elapsed_seconds": 1.0,
            },
            "coder_primary": {
                "answer": "41",
                "passed": False,
                "elapsed_seconds": 2.0,
            },
        },
    }


def test_pairwise_holdout_plan_selects_non_overlapping_cross_action_group(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    candidates = tmp_path / "candidates.jsonl"
    summary = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"

    assert (
        mod.main(
            [
                "--input",
                str(source),
                "--existing-manifest",
                str(manifest),
                "--existing-pairwise-jsonl",
                str(pairwise),
                "--candidates-jsonl",
                str(candidates),
                "--summary-json",
                str(summary),
                "--summary-md",
                str(summary_md),
                "--target-source-families",
                "seeding_eval",
                "--target-suites",
                "livecodebench",
                "--min-cross-action-candidate-groups",
                "1",
            ]
        )
        == 0
    )

    rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["canonical_action"] == "coder_escalation"
    assert rows[0]["source_family"] == "seeding_eval"
    assert rows[0]["suite"] == "livecodebench"
    assert "prompt" not in rows[0]
    assert "response" not in rows[0]
    assert payload["decision"]["status"] == "expansion_plan_ready"
    assert payload["candidate_groups"] == 1
    assert payload["selected_groups"][0]["existing_manifest_actions"] == ["frontdoor"]
    assert payload["selected_groups"][0]["candidate_actions"] == ["coder_escalation"]
    assert "# Offline Reward Pairwise Holdout Expansion Plan" in summary_md.read_text(
        encoding="utf-8"
    )


def test_pairwise_holdout_plan_excludes_existing_pairwise_groups(tmp_path: Path) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    group_key = mod._record_group_key(
        source_path=source,
        offset=0,
        question_id="q-live",
        prompt_sha256=prompt_hash,
        expected_sha256=expected_hash,
    )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [{"group_key": group_key}])

    args = mod.build_parser().parse_args(
        [
            "--input",
            str(source),
            "--existing-manifest",
            str(manifest),
            "--existing-pairwise-jsonl",
            str(pairwise),
            "--candidates-jsonl",
            str(tmp_path / "candidates.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--target-source-families",
            "seeding_eval",
            "--target-suites",
            "livecodebench",
            "--min-cross-action-candidate-groups",
            "1",
        ]
    )

    candidates, summary = mod.build_plan(args)

    assert candidates == []
    assert summary["decision"]["status"] == "insufficient_non_overlapping_cross_action_candidates"
    assert summary["skipped_pairwise_overlap_groups"] == 1


def test_pairwise_holdout_any_mode_empty_suite_does_not_match_everything(
    tmp_path: Path,
) -> None:
    source = tmp_path / "3way_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [])
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    args = mod.build_parser().parse_args(
        [
            "--input",
            str(source),
            "--existing-manifest",
            str(manifest),
            "--existing-pairwise-jsonl",
            str(pairwise),
            "--candidates-jsonl",
            str(tmp_path / "candidates.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--target-source-families",
            "seeding_eval",
            "--target-suites",
            "",
            "--target-match-mode",
            "any",
            "--min-cross-action-candidate-groups",
            "1",
        ]
    )

    candidates, summary = mod.build_plan(args)

    assert candidates == []
    assert summary["stats"]["skipped_non_target_suite_record"] == 1
