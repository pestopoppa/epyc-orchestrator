"""Replay and shadow contracts for optional serving phases."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.autopilot_core.serving_phase_evidence import phase_report, plan_serving_phase
from src.backends import serving_calls
from src.backends.serving_phases import SCHEMA, phase_breakdown

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "autopilot"))

from eval_tower import EvalTower, QuestionResult  # noqa: E402
from experiment_journal import ExperimentJournal  # noqa: E402


def _record(identity, start, *, prefill=2, decode=8, fresh=20, cached=80, slot=0):
    timings = dict(
        prompt_n=fresh,
        cache_n=cached,
        predicted_n=100,
        prompt_ms=prefill * 1000,
        predicted_ms=decode * 1000,
        id_slot=slot,
    )
    record = {
        "schema": serving_calls.SCHEMA,
        "record_id": identity,
        "outcome": "ok",
        "ts_start": start,
        "ts_end": start + prefill + decode,
        "provenance": {"host": "test-host"},
        "server": {"base_url": "http://test:1"},
        "timings": timings,
        "phase_breakdown": phase_breakdown(
            timings,
            trial_id=7,
            batch_id="batch-abc",
            request_id=identity,
            stream_id=identity,
            streamed=True,
            wall_ms=(prefill + decode) * 1000,
            delivered_first_ms=None,
        ),
    }
    record["phase_breakdown"]["intervals"] = {
        name: {"source": "observed_server_phase_events", "start_epoch_s": begin, "end_epoch_s": end}
        for name, begin, end in (
            ("prefill", start, start + prefill),
            ("decode", start + prefill, start + prefill + decode),
        )
    }
    return record


def _save(path, rows):
    for row in rows:
        row["record_sha256"] = serving_calls.record_digest(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_recorder_emits_optional_versioned_phase_with_bound_identity(monkeypatch, tmp_path):
    path = tmp_path / "serving_calls.jsonl"
    monkeypatch.setenv(serving_calls.LOG_ENV, str(path))

    class Client:
        config = SimpleNamespace(base_url="http://test:1")

        @serving_calls.recorded_call("infer_stream_text")
        def infer_stream_text(self, role_config, request, on_chunk=None):
            serving_calls.note_timings({"prompt_n": 0, "predicted_ms": 500}, stream=True)
            return SimpleNamespace(success=True, first_token_ms=17)

    serving_calls.stage_caller(request_id="req", trial_id=7, batch_id="batch-abc")
    Client().infer_stream_text(SimpleNamespace(name="worker"), None)
    [written] = [json.loads(line) for line in path.read_text().splitlines()]
    phase = written["phase_breakdown"]
    assert phase["schema"] == SCHEMA
    assert phase["stream_id"] == written["record_id"]
    assert (phase["trial_id"], phase["batch_id"], phase["request_id"]) == (7, "batch-abc", "req")
    assert phase["fresh_prompt_tokens"] == 0 and phase["cached_prompt_tokens"] is None
    assert phase["delivered_first_ms"] == 17
    assert phase["draft_accepted_tokens"] is None
    assert "intervals" not in phase


def test_report_cli_distinguishes_evidence_and_unions_overlap_once(tmp_path):
    prefill = _record("fresh-prefill", 100, prefill=8, decode=2, fresh=100, cached=0)
    decode = _record("decode", 103, slot=1)
    legacy = _record("legacy", 130)
    del legacy["phase_breakdown"]
    legacy["timings_source"] = "server"
    missing = _record("missing", 145)
    missing["phase_breakdown"]["prefill_ms"] = None
    path = _save(tmp_path / "serving.jsonl", [prefill, decode, legacy, missing, prefill])
    out = subprocess.check_output(
        [
            sys.executable,
            str(REPO / "scripts/analysis/serving_calls_nonprod_report.py"),
            "--log",
            str(path),
            "--phases",
        ],
        text=True,
    )
    report = json.loads(out)
    assert report["duplicate_rows"] == 1
    findings = {row["stream_id"]: row for row in report["requests"]}
    assert findings["fresh-prefill"]["classification"] == "fresh_prefill_dominated"
    assert findings["decode"]["classification"] == "decode_dominated"
    assert findings["legacy"]["reason"] == "legacy_without_phase_breakdown"
    assert findings["legacy"]["decode_ms"] == 8000
    assert findings["legacy"]["delivered_first_ms"] is None
    assert findings["missing"]["reason"] == "missing_or_zero_required_counters"
    assert findings["missing"]["prefill_window"] is None
    # Decodes [108,110], [105,113], and [147,155] span 16 s as a union.
    [server] = report["server_aggregates"]
    assert server["observed_decode_union_s"] == 16
    assert server["observed_concurrent_decode_tps"] == pytest.approx(300 / 16)
    assert server["request_native_decode_sum_s"] == 18


@pytest.mark.parametrize(
    "loaded,expected", [(True, "admission_batch_scheduling"), (False, "kernel_recipe")]
)
def test_shadow_needs_repeated_matched_decode_loss(tmp_path, loaded, expected):
    rows = []
    for repetition in (0, 1):
        start = 100 + repetition * 100
        fast = _record(f"fast-{repetition}", start, decode=2)
        slow = _record(f"slow-{repetition}", start + 10)
        for row, arm, recipe in (
            (fast, "isolated" if loaded else "isolated_reference", "d"),
            (slow, "loaded" if loaded else "isolated", "d" if loaded else "e"),
        ):
            row["phase_breakdown"]["replay"] = {
                "pair_id": str(repetition),
                "arm": arm,
                "prompt_sha256": "a" * 64,
                "prefix_state_sha256": "b" * 64,
                "serving_family_sha256": "c" * 64,
                "recipe_sha256": recipe * 64,
                "capture_receipt_sha256": "f" * 64,
                "capture_complete": True,
            }
        rows.extend((slow, fast))
        if loaded:
            rows.append(_record(f"peer-{repetition}", start + 12, slot=1))
    path = tmp_path / "serving.jsonl"
    _save(path, rows[:3] if loaded else rows[:2])
    assert plan_serving_phase(phase_report([path]))["decision"] == "insufficient_evidence"
    _save(path, rows)
    decision = plan_serving_phase(phase_report([path]))
    assert decision["decision"] == expected
    assert len(decision["matched_decode_evidence"][expected]) == 2
    assert decision["execution_authority"] is False


def test_normal_eval_and_journal_observer_preserves_quality_goodput(tmp_path, monkeypatch):
    path = _save(
        tmp_path / "serving.jsonl",
        [
            _record("prefill", 100, prefill=8, decode=2, fresh=100, cached=0),
        ],
    )
    monkeypatch.setenv("AUTOPILOT_SERVING_PHASE_SHADOW_LOG", str(path))
    tower = EvalTower()
    tower.set_trial_context(7)
    tower.note_trial_batches(["batch-abc"], 100)
    result = tower._aggregate(
        [
            QuestionResult("q1", "unit", "a", "b", correct=True, eval_wall_s=20),
            QuestionResult("q2", "unit", "a", "b", correct=False, eval_wall_s=20),
        ],
        tier=1,
    )
    before = dict(result.details)
    event = tower.observe_serving_phases(7, result)
    assert event["decision"] == "prefix_retention"
    assert event["quality_qualified_goodput"] == {
        "quality": 1.5,
        "task_rate_qph": 360.0,
        "scored_task_rate_qph": 360.0,
        "accepted_tasks_per_hour": 180.0,
    }
    assert result.details == before and event["paid_effort"] is None
    assert tower.observe_serving_phases(8, result)["decision"] == "insufficient_evidence"
    journal_dir = tmp_path / "journal"
    journal = ExperimentJournal(journal_dir, segment_snapshots=False)
    written = journal.append_ledger_event(event)
    reopened = ExperimentJournal(journal_dir, create=False)
    assert reopened.count() == 0
    assert reopened.ledger_events("serving_phase_shadow_decision") == [written]
    monkeypatch.delenv("AUTOPILOT_SERVING_PHASE_SHADOW_LOG")
    assert tower.observe_serving_phases(7, result) is None
