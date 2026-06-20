from __future__ import annotations

import datetime as dt
import json
from argparse import Namespace
from pathlib import Path

import yaml

from scripts.tasks import harvest_tasks


def _row(event_type: str, task_id: str, timestamp: str, data: dict | None = None, **extra):
    row = {
        "event_type": event_type,
        "task_id": task_id,
        "timestamp": timestamp,
        "agent_tier": None,
        "agent_role": None,
        "data": data or {},
        "memory_id": None,
        "outcome": None,
        "outcome_details": None,
    }
    row.update(extra)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _workload_model(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "task_classes": [
                    {"id": "benchmark_eval_measurement"},
                    {"id": "code_change_implementation"},
                    {"id": "governance_docs_handoff"},
                    {"id": "planning_architecture_review"},
                ]
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "progress_log_dir": str(tmp_path / "progress"),
        "workload_model": str(tmp_path / "workload_model.yaml"),
        "output": str(tmp_path / "real_tasks.jsonl"),
        "manifest": str(tmp_path / "manifest.json"),
        "start_date": None,
        "end_date": None,
        "lab_task_records": [],
        "limit": 0,
        "include_open": False,
        "exclude_synthetic_like": False,
        "dedupe_prompt": False,
        "omit_prompt_text": False,
        "compact_evidence": False,
        "training_eligible_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_harvest_progress_records_with_class_outcome_route_and_wall_time(tmp_path: Path) -> None:
    _workload_model(tmp_path / "workload_model.yaml")
    start = dt.datetime(2026, 6, 12, 1, 0, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(seconds=12.5)
    _write_jsonl(
        tmp_path / "progress" / "2026-06-12.jsonl",
        [
            _row(
                "task_started",
                "chat-1",
                start.isoformat(),
                {
                    "task_type": "chat",
                    "objective": "Run benchmark replay and produce a measurement report",
                    "priority": "interactive",
                },
            ),
            _row(
                "routing_decision",
                "chat-1",
                start.isoformat(),
                {"routing": ["benchmark_analyst"], "strategy": "rules"},
            ),
            _row(
                "task_completed",
                "chat-1",
                end.isoformat(),
                {"producer_role": "benchmark_analyst", "final_answer_role": "benchmark_analyst"},
                outcome="success",
            ),
        ],
    )

    result = harvest_tasks.run(_args(tmp_path))

    rows = _read_jsonl(Path(result["output"]))
    assert result["written"] == 1
    assert rows[0]["class"] == "benchmark_eval_measurement"
    assert rows[0]["outcome"] == "success"
    assert rows[0]["route_taken"] == ["benchmark_analyst"]
    assert rows[0]["wall_s"] == 12.5
    assert rows[0]["training_eligible"] is True
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["counts"]["by_class"] == {"benchmark_eval_measurement": 1}


def test_harvest_prefers_embedded_task_record_v1_payload(tmp_path: Path) -> None:
    _workload_model(tmp_path / "workload_model.yaml")
    _write_jsonl(
        tmp_path / "progress" / "2026-06-12.jsonl",
        [
            _row(
                "task_started",
                "chat-embedded",
                "2026-06-12T00:00:00+00:00",
                {
                    "task_type": "chat",
                    "objective": "Implement a code patch and update tests",
                },
            ),
            _row(
                "routing_decision",
                "chat-embedded",
                "2026-06-12T00:00:00+00:00",
                {"routing": ["frontdoor"], "strategy": "rules"},
            ),
            _row(
                "task_completed",
                "chat-embedded",
                "2026-06-12T00:00:08+00:00",
                {
                    "task_record_v1": {
                        "schema_version": "task_record.v1",
                        "task_id": "chat-embedded",
                        "class": "code",
                        "prompt_ref": "progress-text-sha256:abc123",
                        "route_taken": ["coder_escalation"],
                        "routing_strategy": "learned",
                        "wall_s": 4.25,
                        "tokens": 73,
                        "outcome": "success",
                        "operator_verdict": "accepted",
                        "operator_verdict_details_ref": "progress-text-sha256:def456",
                    },
                    "producer_role": "coder_escalation",
                },
                outcome="success",
            ),
        ],
    )

    result = harvest_tasks.run(_args(tmp_path))

    row = _read_jsonl(Path(result["output"]))[0]
    assert row["class"] == "code_change_implementation"
    assert row["prompt_ref"] == {
        "kind": "task_record_prompt_ref",
        "ref": "progress-text-sha256:abc123",
        "sha256": "abc123",
        "source_ref": {"path": str(tmp_path / "progress" / "2026-06-12.jsonl"), "line": 3},
    }
    assert row["route_taken"] == ["coder_escalation"]
    assert row["route_strategy"] == "learned"
    assert row["wall_s"] == 4.25
    assert row["tokens"] == 73
    assert row["outcome_source"] == "task_record_v1"
    assert row["task_record_schema_version"] == "task_record.v1"
    assert row["operator_verdict"] == "accepted"
    assert row["operator_verdict_details_ref"] == "progress-text-sha256:def456"


def test_harvest_progress_marks_uncategorized_and_synthetic_like(tmp_path: Path) -> None:
    _workload_model(tmp_path / "workload_model.yaml")
    assert harvest_tasks.synthetic_like("Write a function. Test cases: assert f(1) == 1")
    _write_jsonl(
        tmp_path / "progress" / "2026-06-12.jsonl",
        [
            _row(
                "task_started",
                "chat-2",
                "2026-06-12T00:00:00+00:00",
                {
                    "task_type": "chat",
                    "objective": "Write a casual pitch deck and use the word clearly at least 2 times.",
                },
            ),
            _row("task_completed", "chat-2", "2026-06-12T00:00:05+00:00", outcome="success"),
        ],
    )

    harvest_tasks.run(_args(tmp_path))

    rows = _read_jsonl(tmp_path / "real_tasks.jsonl")
    assert rows[0]["class"] == "uncategorized_chat"
    assert rows[0]["synthetic_like"] is True
    assert rows[0]["training_eligible"] is False
    assert set(rows[0]["eligibility_reasons"]) == {"not_taxonomy_class", "synthetic_like_prompt"}


def test_harvest_lab_task_records_normalizes_shadow_record(tmp_path: Path) -> None:
    _workload_model(tmp_path / "workload_model.yaml")
    queue = tmp_path / "queue"
    prompt_rel = "claims_grammar_check/run-1/prompt.txt"
    (queue / prompt_rel).parent.mkdir(parents=True)
    (queue / prompt_rel).write_text("Check handoff claims grammar and docs evidence.", encoding="utf-8")
    _write_jsonl(
        queue / "task_records.jsonl",
        [
            {
                "schema_version": "lab_task_record.v1",
                "record_type": "task_record",
                "run_id": "claims_grammar_check-run-1",
                "job_id": "claims_grammar_check",
                "generated_at": "2026-06-13T00:00:00+00:00",
                "stage": "shadow",
                "risk": "read_only",
                "model_role": "verifier",
                "invocation_mode": "dry_run_contract_stub",
                "validation": {"output_contract": "passed"},
                "artifacts": {"prompt": prompt_rel},
                "chat_meta": {},
            }
        ],
    )

    harvest_tasks.run(
        _args(
            tmp_path,
            progress_log_dir=str(tmp_path / "missing-progress"),
            lab_task_records=[str(queue / "task_records.jsonl")],
        )
    )

    rows = _read_jsonl(tmp_path / "real_tasks.jsonl")
    assert rows[0]["source"] == "lab_task_record_jsonl"
    assert rows[0]["route_taken"] == ["verifier"]
    assert rows[0]["class"] == "governance_docs_handoff"
    assert rows[0]["outcome"] == "success"
    assert rows[0]["training_eligible"] is False
    assert rows[0]["eligibility_reasons"] == ["dry_run"]


def test_dedupe_prompt_collapses_forced_multi_role_attempts(tmp_path: Path) -> None:
    _workload_model(tmp_path / "workload_model.yaml")
    rows = []
    for task_id, role, offset in [
        ("chat-a", "frontdoor", 1),
        ("chat-b", "worker_general", 2),
    ]:
        rows.extend(
            [
                _row(
                    "task_started",
                    task_id,
                    f"2026-06-12T00:00:0{offset}+00:00",
                    {
                        "task_type": "chat",
                        "objective": "Run benchmark replay and produce a measurement report",
                    },
                ),
                _row(
                    "routing_decision",
                    task_id,
                    f"2026-06-12T00:00:0{offset}+00:00",
                    {"routing": [role], "strategy": "forced"},
                ),
                _row("task_completed", task_id, f"2026-06-12T00:00:1{offset}+00:00"),
            ]
        )
    _write_jsonl(tmp_path / "progress" / "2026-06-12.jsonl", rows)

    result = harvest_tasks.run(_args(tmp_path, dedupe_prompt=True))

    records = _read_jsonl(Path(result["output"]))
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert len(records) == 1
    assert records[0]["duplicate_count"] == 2
    assert records[0]["duplicate_task_ids"] == ["chat-a", "chat-b"]
    assert [attempt["route_taken"] for attempt in records[0]["route_attempts"]] == [
        ["frontdoor"],
        ["worker_general"],
    ]
    assert manifest["counts"]["duplicates_collapsed"] == 1


def test_extract_tokens_accepts_runtime_token_fields() -> None:
    assert harvest_tasks._extract_tokens(
        [
            _row("task_completed", "a", "2026-06-12T00:00:00+00:00", {"tokens_generated": 42})
        ]
    ) == {"total": 42}
    assert harvest_tasks._extract_tokens(
        [
            _row(
                "task_completed",
                "b",
                "2026-06-12T00:00:00+00:00",
                {"prompt_tokens": 10, "completion_tokens": 5},
            )
        ]
    ) == {"prompt_tokens": 10, "completion_tokens": 5, "total": 15}
    assert harvest_tasks._extract_tokens(
        [
            _row(
                "task_completed",
                "c",
                "2026-06-12T00:00:00+00:00",
                {"chat_meta": {"usage": {"total_tokens": 9}}},
            )
        ]
    ) == {"total_tokens": 9}


def test_compact_evidence_keeps_gate_fields_without_bulk_attempts() -> None:
    row = {
        "schema_version": "real_task_record.v1",
        "record_type": "task_record",
        "task_id": "chat-a",
        "source": "orchestrator_progress_jsonl",
        "source_refs": [{"path": "p", "line": 1}],
        "class": "code_change_implementation",
        "class_is_taxonomy": True,
        "prompt": "raw private prompt",
        "prompt_ref": {"kind": "progress_objective_sha256", "sha256": "abc"},
        "route_taken": ["frontdoor"],
        "wall_s": 1.0,
        "outcome": "success",
        "privacy_class": "local_private",
        "synthetic_like": False,
        "training_eligible": True,
        "eligibility_reasons": [],
        "duplicate_count": 2,
        "duplicate_task_ids": ["chat-a", "chat-b"],
        "duplicate_outcomes": {"success": 2},
        "route_attempts": [
            {"task_id": "chat-a", "route_taken": ["frontdoor"], "outcome": "success"},
            {"task_id": "chat-b", "route_taken": ["worker_general"], "outcome": "success"},
        ],
    }

    compact = harvest_tasks.compact_evidence_rows([row])[0]

    assert compact["training_eligible"] is True
    assert compact["duplicate_count"] == 2
    assert compact["duplicate_outcomes"] == {"success": 2}
    assert compact["route_attempt_count"] == 2
    assert compact["route_attempt_roles"] == ["frontdoor", "worker_general"]
    assert "prompt" not in compact
    assert "prompt_ref" not in compact
    assert "source_refs" not in compact
    assert "route_attempts" not in compact
    assert "duplicate_task_ids" not in compact


def test_training_eligible_only_filters_output_and_manifest(tmp_path: Path) -> None:
    _workload_model(tmp_path / "workload_model.yaml")
    _write_jsonl(
        tmp_path / "progress" / "2026-06-12.jsonl",
        [
            _row(
                "task_started",
                "eligible",
                "2026-06-12T00:00:00+00:00",
                {
                    "task_type": "chat",
                    "objective": "Implement a code patch and update tests",
                },
            ),
            _row("task_completed", "eligible", "2026-06-12T00:00:02+00:00", outcome="success"),
            _row(
                "task_started",
                "ineligible",
                "2026-06-12T00:00:03+00:00",
                {
                    "task_type": "chat",
                    "objective": "Say hello exactly 3 times.",
                },
            ),
            _row("task_completed", "ineligible", "2026-06-12T00:00:04+00:00", outcome="success"),
        ],
    )

    result = harvest_tasks.run(
        _args(tmp_path, compact_evidence=True, omit_prompt_text=True, training_eligible_only=True)
    )

    rows = _read_jsonl(Path(result["output"]))
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert len(rows) == 1
    assert rows[0]["task_id"] == "eligible"
    assert rows[0]["training_eligible"] is True
    assert "prompt" not in rows[0]
    assert manifest["counts"]["written"] == 1
    assert manifest["counts"]["training_eligible"] == 1
    assert manifest["options"]["training_eligible_only"] is True
