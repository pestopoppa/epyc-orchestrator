"""Tests for offline verifier-data expansion planning."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.graph_router import plan_offline_reward_verifier_expansion as mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_canonical_action_maps_historical_delegated_and_alias_roles(caplog) -> None:
    actions = {"frontdoor", "architect_general", "coder_escalation", "worker_vision"}

    caplog.set_level(logging.WARNING, logger="scripts.graph_router.action_space")
    assert mod._canonical_action("architect_general:delegated", actions) == "architect_general"
    assert mod._canonical_action("architect_coding:delegated", actions) == "architect_general"
    assert mod._canonical_action("coder_primary", actions) == "coder_escalation"
    assert mod._canonical_action("frontdoor:direct", actions) == "frontdoor"
    assert mod._canonical_action("worker_vision:direct", actions) == "worker_vision"
    assert mod._canonical_action("vision_escalation:direct", actions) is None
    assert "Unknown action" not in caplog.text


def test_build_plan_excludes_existing_rows_and_strips_private_text(tmp_path: Path) -> None:
    source = tmp_path / "results.jsonl"
    _write_jsonl(
        source,
        [
            {
                "question_id": "q1",
                "suite": "architecture",
                "prompt": "Design the thing",
                "expected": "Use a queue",
                "rewards": {
                    "architect_general:delegated": 1.0,
                    "coder_primary": 0.0,
                },
                "role_results": {
                    "architect_general:delegated": {
                        "answer": "Use a queue with backpressure",
                        "passed": True,
                    },
                    "coder_primary": {
                        "answer": "Use direct calls",
                        "passed": False,
                    },
                },
            }
        ],
    )
    existing_manifest = tmp_path / "existing.jsonl"
    _write_jsonl(
        existing_manifest,
        [
            {
                "source_path": str(source),
                "source_record_offset": 0,
                "role_key": "coder_primary",
            }
        ],
    )
    existing_summary = tmp_path / "summary.json"
    existing_summary.write_text(
        json.dumps(
            {
                "canonical_action_counts": {
                    "architect_general": 10,
                    "coder_escalation": 88,
                }
            }
        ),
        encoding="utf-8",
    )

    candidates, summary = mod.build_plan(
        argparse.Namespace(
            input=[source],
            existing_manifest=existing_manifest,
            existing_summary=existing_summary,
            target_actions="architect_general,coder_escalation",
            min_action_rows=30,
            max_candidates_per_action=20,
        )
    )

    assert len(candidates) == 1
    assert candidates[0]["canonical_action"] == "architect_general"
    assert "prompt" not in candidates[0]
    assert "expected" not in candidates[0]
    assert "response" not in candidates[0]
    assert summary["candidate_action_counts"] == {"architect_general": 1}
    assert summary["existing_rows_excluded"] == 1
    assert summary["recommended_sources"][0]["source_path"] == str(source)


def test_build_plan_deduplicates_repeated_input_paths(tmp_path: Path) -> None:
    source = tmp_path / "results.jsonl"
    _write_jsonl(
        source,
        [
            {
                "question_id": "q1",
                "suite": "architecture",
                "prompt": "Design the thing",
                "expected": "Use a queue",
                "role_results": {
                    "architect_general:delegated": {
                        "answer": "Use a queue with backpressure",
                        "passed": True,
                    },
                },
            }
        ],
    )

    candidates, summary = mod.build_plan(
        argparse.Namespace(
            input=[source, source],
            existing_manifest=None,
            existing_summary=None,
            target_actions="architect_general",
            min_action_rows=30,
            max_candidates_per_action=20,
        )
    )

    assert len(candidates) == 1
    assert summary["files_scanned"] == 2
    assert summary["candidate_action_counts"] == {"architect_general": 1}
