from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_BENCH = _REPO / "scripts" / "benchmark"
sys.path.insert(0, str(_BENCH))
_SPEC = importlib.util.spec_from_file_location("s4_omega_ab_test", _BENCH / "s4_omega_ab.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["s4_omega_ab_test"] = _MOD
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MOD)


def test_default_suite_is_hash_pinned_and_has_three_questions():
    suite, digest = _MOD.load_suite()
    assert digest == _MOD.DEFAULT_SUITE_SHA256
    assert len(suite["questions"]) == 3
    assert len({q["id"] for q in suite["questions"]}) == 3


def test_dry_run_writes_six_self_hashed_synthetic_records(tmp_path):
    suite, digest = _MOD.load_suite()
    manifest_path, records_path, records = _MOD.run_dry_run(
        suite=suite,
        suite_sha256=digest,
        output_dir=tmp_path,
    )

    manifest = json.loads(manifest_path.read_text())
    persisted = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert len(records) == len(persisted) == 6
    assert manifest["record_count"] == 6
    assert manifest["suite_sha256"] == _MOD.DEFAULT_SUITE_SHA256
    assert manifest["schema_validation"] == "synthetic_fixture"
    assert manifest["measurement_status"] == "not_a_measurement"
    assert manifest["promotion_claim"] is False
    assert _MOD.verify_content_hash(manifest)
    assert all(_MOD.verify_content_hash(row) for row in persisted)
    assert {row["arm_id"] for row in persisted} == {"baseline", "s4_omega"}
    assert all(row["role"] == "frontdoor" and row["mode"] == "repl" for row in persisted)
    assert all(row["turns_unit"] == "orchestrator_repl_turn" for row in persisted)
    assert all(row["synthetic"] and row["schema_validation"] == "synthetic_fixture" for row in persisted)
    assert all(
        {"turns", "tokens_generated", "prompt_tokens", "passed", "elapsed_seconds"}
        <= row.keys()
        for row in persisted
    )


def test_recovery_preserves_original_bytes_and_exports_partial_view(tmp_path):
    suite, digest = _MOD.load_suite()
    question = suite["questions"][0]
    writer = _MOD.NativeRunWriter(
        tmp_path,
        run_id="recoverable-run",
        manifest={"suite_id": suite["suite_id"], "suite_sha256": digest},
    )
    initial_manifest_bytes = writer.manifest_path.read_bytes()
    record = _MOD.make_record(
        run_id="recoverable-run",
        suite_id=suite["suite_id"],
        suite_sha256=digest,
        question=question,
        arm_id="baseline",
        prompt_sha256="e" * 64,
        response=_MOD.synthetic_response(question, "baseline", 0),
        elapsed_seconds=0.01,
        synthetic=True,
    )
    writer.append(record)
    _MOD._atomic_write(writer.manifest_path, initial_manifest_bytes)
    tail = b'{"truncated":'
    with writer.records_path.open("ab") as stream:
        stream.write(tail)
    original_records = writer.records_path.read_bytes()
    original_manifest = writer.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="explicit confirmation"):
        _MOD.recover_native_run(writer.manifest_path)
    manifest, records = _MOD.recover_native_run(writer.manifest_path, source_stopped=True)
    assert manifest["run_status"] == "partial"
    assert manifest["record_count"] == 1
    assert records == [record]
    assert writer.records_path.read_bytes() == original_records
    assert writer.manifest_path.read_bytes() == original_manifest
    assert (tmp_path / "recoverable-run.recovery-tail.bin").read_bytes() == tail
    assert json.loads((tmp_path / "recoverable-run.recovered.manifest.json").read_text()) == manifest


def test_missing_prompt_tokens_remains_null_and_refuses_admission():
    suite, digest = _MOD.load_suite()
    question = suite["questions"][0]
    record = _MOD.make_record(
        run_id="live-test",
        suite_id=suite["suite_id"],
        suite_sha256=digest,
        question=question,
        arm_id="baseline",
        prompt_sha256="a" * 64,
        response={"answer": question["expected"], "turns": 1, "tokens_generated": 5},
        elapsed_seconds=0.25,
        synthetic=False,
    )
    assert record["prompt_tokens"] is None
    assert record["prompt_tokens_unavailable_reason"] == "response_missing_prompt_tokens"
    assert record["schema_validation"] == "incomplete"
    assert "response_missing_prompt_tokens" in record["metric_unavailable_reasons"]


def test_reported_zero_prompt_tokens_is_preserved_as_native_usage():
    suite, digest = _MOD.load_suite()
    question = suite["questions"][0]
    record = _MOD.make_record(
        run_id="live-zero",
        suite_id=suite["suite_id"],
        suite_sha256=digest,
        question=question,
        arm_id="baseline",
        prompt_sha256="c" * 64,
        response={"answer": question["expected"], "turns": 1, "tokens_generated": 5, "prompt_tokens": 0},
        elapsed_seconds=0.25,
        synthetic=False,
        request_sha256="d" * 64,
    )
    assert record["prompt_tokens"] == 0
    assert record["prompt_tokens_unavailable_reason"] is None
    assert record["schema_validation"] == "complete"


def test_live_preflight_refuses_before_inference_when_prompt_usage_is_not_advertised(tmp_path):
    suite, digest = _MOD.load_suite()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "components": {
                    "schemas": {
                        "ChatResponse": {
                            "properties": {"turns": {}, "tokens_generated": {}}
                        }
                    }
                }
            }

    class Client:
        closed = False

        def get(self, *_args, **_kwargs):
            return Response()

    called = False

    def call_fn(**_kwargs):
        nonlocal called
        called = True
        return {}

    with pytest.raises(_MOD.RunnerNotReady, match="prompt_tokens"):
        _MOD.run_live(
            suite=suite,
            suite_sha256=digest,
            output_dir=tmp_path,
            root_repo=tmp_path,
            override_base=tmp_path / "prompt-roots",
            client=Client(),
            call_fn=call_fn,
        )
    assert not called


def test_live_exception_keeps_reopenable_partial_manifest_and_records(tmp_path):
    suite, digest = _MOD.load_suite()
    root_repo = tmp_path / "root"
    (root_repo / _MOD.ADAPTER_RELATIVE_PATH).parent.mkdir(parents=True)
    (root_repo / _MOD.ADAPTER_RELATIVE_PATH).write_text("fixture adapter")
    source_table = root_repo / _MOD.SOURCE_TABLE_RELATIVE_PATH
    source_table.parent.mkdir(parents=True, exist_ok=True)
    source_table.write_text("S4 Omega REPL A/B runner records")
    source_task = root_repo / _MOD.SOURCE_TASK_RELATIVE_PATH
    source_task.parent.mkdir(parents=True, exist_ok=True)
    source_task.write_text("- [x] **VB-S4-OMEGA-WIRE — parent complete**")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "components": {
                    "schemas": {
                        "ChatResponse": {
                            "properties": {
                                "turns": {},
                                "tokens_generated": {},
                                "prompt_tokens": {},
                            }
                        }
                    }
                }
            }

    class Client:
        def get(self, *_args, **_kwargs):
            return Response()

    calls = 0
    captured_payloads = []

    def call_fn(**kwargs):
        nonlocal calls
        calls += 1
        manifest_paths = list((tmp_path / "out").glob("live-*.manifest.json"))
        assert len(manifest_paths) == 1
        in_progress_manifest, in_progress_records = _MOD.reopen_native_run(manifest_paths[0])
        assert in_progress_manifest["run_status"] == "running"
        assert len(in_progress_records) == calls - 1
        payload = {
            "prompt": kwargs["prompt"],
            "force_role": kwargs["force_role"],
            "force_mode": kwargs["force_mode"],
            "prompt_root": kwargs["prompt_root"],
            "batch_id": kwargs["batch_id"],
        }
        captured_payloads.append(payload)
        kwargs["client"].capture_payload(payload)
        if calls == 3:
            raise RuntimeError("injected after two completed questions")
        question = next(q for q in suite["questions"] if q["prompt"] == kwargs["prompt"])
        return {
            "answer": question["expected"],
            "turns": 2,
            "tokens_generated": 8,
            "prompt_tokens": 17,
        }

    with pytest.raises(_MOD.LiveRunInterrupted) as raised:
        _MOD.run_live(
            suite=suite,
            suite_sha256=digest,
            output_dir=tmp_path / "out",
            root_repo=root_repo,
            override_base=tmp_path / "prompt-roots",
            client=Client(),
            call_fn=call_fn,
        )

    manifest, records = _MOD.reopen_native_run(raised.value.manifest_path)
    assert manifest["run_status"] == "partial"
    assert manifest["record_count"] == 2
    assert manifest["protocol_id"] is None
    assert manifest["promotion_claim"] is False
    assert manifest["model_identity"] is None
    assert manifest["server_identity"] is None
    assert len(records) == 2
    assert all(row["request_sha256_source"] == "http_json_payload" for row in records)
    assert records[0]["request_sha256"] == _MOD.sha256_bytes(_MOD.canonical_json_bytes(captured_payloads[0]))
    assert all(row["schema_validation"] == "complete" for row in records)
    assert all(row["request_payload"] == captured_payloads[i] for i, row in enumerate(records))
    assert all(row["response"]["answer"] == row["question"]["expected"] for row in records)
    assert all(row["question"]["scoring_method"] for row in records)
    assert manifest["suite_snapshot_sha256"] == _MOD.sha256_file(
        tmp_path / "out" / manifest["suite_snapshot"]
    )
    for prompt_snapshot in manifest["prompt_snapshots"].values():
        assert _MOD.sha256_file(tmp_path / "out" / manifest["source_snapshot_dir"] / prompt_snapshot["file"]) == prompt_snapshot["sha256"]
    assert list((tmp_path / "prompt-roots").iterdir()) == []


def test_source_wiring_gate_requires_adapter_enrollment_and_completed_task(tmp_path):
    adapter = tmp_path / _MOD.ADAPTER_RELATIVE_PATH
    source_table = tmp_path / _MOD.SOURCE_TABLE_RELATIVE_PATH
    source_task = tmp_path / _MOD.SOURCE_TASK_RELATIVE_PATH
    adapter.parent.mkdir(parents=True, exist_ok=True)
    source_table.parent.mkdir(parents=True, exist_ok=True)
    source_task.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_text("fixture adapter")
    source_table.write_text("S4 Omega REPL A/B runner records")
    source_task.write_text(
        "- [ ] **VB-S4-OMEGA-WIRE — parent still open**\n"
        "  - [x] **VB-S4-OMEGA-WIRE.SRC — child complete**\n"
    )

    with pytest.raises(_MOD.RunnerNotReady, match="not complete"):
        _MOD.check_source_wiring(tmp_path)

    source_task.write_text("- [x] **VB-S4-OMEGA-WIRE — parent complete**")
    resolved_adapter, digest = _MOD.check_source_wiring(tmp_path)
    assert resolved_adapter == adapter
    assert digest == _MOD.sha256_file(adapter)


def test_request_local_prompt_roots_copy_the_two_actual_arm_texts(tmp_path):
    roots = _MOD.prepare_prompt_roots(
        run_id="fixture",
        override_base=tmp_path,
    )
    try:
        baseline = (_MOD.PROMPT_DIR / "root_lm_system.md").read_bytes()
        omega = (_MOD.PROMPT_DIR / "root_lm_system.s4_omega.md").read_bytes()
        assert (roots["baseline"] / "root_lm_system.md").read_bytes() == baseline
        assert (roots["s4_omega"] / "root_lm_system.md").read_bytes() == omega
        assert not (roots["baseline"] / "root_lm_system.s4_omega.md").exists()
        assert (roots["s4_omega"] / "root_lm_system.s4_omega.md").read_bytes() == omega
    finally:
        import shutil

        shutil.rmtree(next(iter(roots.values())).parent)
