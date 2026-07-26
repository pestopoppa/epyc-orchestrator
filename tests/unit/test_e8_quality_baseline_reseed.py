from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_reseed.py"
VALIDATOR = Path("/mnt/raid0/llm/epyc-root/artifacts/operator/prepare_e8_quality_baseline_reseed_20260726.sh")

spec = importlib.util.spec_from_file_location("e8_reseed", MODULE_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules["e8_reseed"] = runner
spec.loader.exec_module(runner)


class FakeQuestionResult:
    def __init__(
        self,
        qid: str,
        *,
        answer: str,
        error: str | None = None,
        concurrency: int = 3,
    ) -> None:
        self.qid = qid
        self.question_id = qid
        self.suite = "suite_a"
        self.answer = answer
        self.correct = error is None
        self.error = error
        self.partial = False
        self.degraded = False
        self.route_used = "frontdoor"
        self.eval_concurrency = concurrency


class FakeAggregate:
    def __init__(self, results: list[FakeQuestionResult], tier: int) -> None:
        self.quality = 3.0 if not any(row.error for row in results) else 0.0
        self.per_suite_quality = {"suite_a": self.quality}
        self.per_suite_counts = {"suite_a": len(results)}
        self.tier = tier


class FakeTower:
    calls = 0
    error_on_call: int | None = None
    wrong_vector_on_call: int | None = None
    concurrency = 3

    def __init__(self, **_kwargs: object) -> None:
        self.timeout = 1

    def _eval_batch(self, questions: list[dict], *_args: object, **_kwargs: object) -> list[FakeQuestionResult]:
        type(self).calls += 1
        error = "backend failed" if type(self).calls == type(self).error_on_call else None
        rows = [
            FakeQuestionResult(
                str(question["id"]),
                answer=str(question["expected"]),
                error=error,
                concurrency=type(self).concurrency,
            )
            for question in questions
        ]
        if type(self).calls == type(self).wrong_vector_on_call:
            rows[0].qid = "unexpected-qid"
        return rows

    def _aggregate(self, results: list[FakeQuestionResult], tier: int) -> FakeAggregate:
        return FakeAggregate(results, tier)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    state = tmp_path / "state.json"
    registry = tmp_path / "registry.yaml"
    lineup = tmp_path / "lineup.yaml"
    state.write_text(json.dumps({
        "active_instrument_eras": {"eval_quality": "E8"},
        "e8_quality_rebaseline": {"status": "hold_open"},
        "baseline_state": {"eval_quality_era": "E7-eval-instrument"},
    }))
    registry.write_text("models: []\n")
    lineup.write_text("instances: []\n")
    journal = tmp_path / "journal.jsonl"
    journal.write_text("")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    return state, registry, lineup, journal, receipt


def _args(tmp_path: Path, *extra: str):
    state, registry, lineup, journal, receipt = _paths(tmp_path)
    return runner.parse_args([
        "--execute", "--output-dir", str(tmp_path / "evidence"),
        "--state-path", str(state), "--registry-path", str(registry), "--lean-registry-path", str(registry),
        "--runtime-facts-path", str(lineup), "--stack-priors-path", str(registry), "--orchestrator-state-path", str(registry),
        "--journal-path", str(journal), "--protocol-receipt", str(receipt),
        *extra,
    ])


def _patch_clean_environment(monkeypatch, *, mutate: Path | None = None) -> None:
    FakeTower.calls = 0
    FakeTower.error_on_call = None
    FakeTower.wrong_vector_on_call = None
    FakeTower.concurrency = 3
    monkeypatch.setattr(runner, "EvalTower", FakeTower)
    monkeypatch.setattr(
        runner,
        "measurement_source_paths",
        lambda _args: [runner.RUNNER_PATH],
    )
    monkeypatch.setattr(runner, "autopilot_processes", lambda: [])
    monkeypatch.setattr(runner, "numeric_rerun_status", lambda *_args: {"completed": 16, "required": 16})
    monkeypatch.setattr(runner, "runtime_topology", lambda *_args: [{"port": 1, "roles": ["frontdoor"]}])
    monkeypatch.setattr(
        runner,
        "receipt_payload",
        lambda *_args: {
            "schema": "epyc.operator_e8_quality_baseline_protocol.v1",
            "decision": runner.PROTOCOL_DECISION,
            "era": "E8",
            "operator_attestation": "test",
            "protocol": {"protocol_id": runner.PROTOCOL_ID},
        },
    )
    monkeypatch.setattr(runner, "protocol_contract", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "runtime_binding",
        lambda _args, **_kwargs: {
            "runtime_facts_sha256": "runtime", "stack_priors_sha256": "priors",
            "orchestrator_state_sha256": "state", "stack_numa_mode": "both",
            "selected_ports": list(range(24)), "server_pids": {}, "server_binaries": {},
            "runtime_artifacts": {},
            "llama_server": "/fake/llama-server",
            **({"llama_server_sha256": "binary", "llama_server_version": "10107"} if _kwargs.get("include_binary_hash") else {}),
        },
    )
    original_repetition = runner.run_repetition
    def with_sidecar(*args, **kwargs):
        observation, detail = original_repetition(*args, **kwargs)
        sidecar = kwargs["sidecar_dir"] / f"question_results.e8-t{kwargs['tier']}-r{kwargs['repetition']}.jsonl"
        runner.write_text(sidecar, '{"row_type":"batch_complete"}\n')
        detail["sidecar_sha256"] = runner.sha256_path(sidecar)
        return observation, detail
    monkeypatch.setattr(runner, "run_repetition", with_sidecar)
    health = {"ok": True, "payload_sha256": "same", "payload": {"status": "ok"}}
    monkeypatch.setattr(runner, "api_health", lambda *_args, **_kwargs: dict(health))
    monkeypatch.setattr(
        runner,
        "question_vector",
        lambda _tower, *, tier, **_kwargs: (
            [{"id": f"t{tier}-q1", "qid": f"t{tier}-q1", "suite": "suite_a", "prompt": "p", "expected": "e", "scoring_method": "exact_match"},
             {"id": f"t{tier}-q2", "qid": f"t{tier}-q2", "suite": "suite_a", "prompt": "p2", "expected": "e2", "scoring_method": "exact_match"}],
            f"core-t{tier}",
        ),
    )
    if mutate is not None:
        original = runner.run_repetition
        def mutated(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate.write_text("models: changed\n")
            return result
        monkeypatch.setattr(runner, "run_repetition", mutated)


def test_execute_seals_and_atomically_publishes_six_observation_evidence(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    report, rc = runner.execute(_args(tmp_path))

    assert rc == 0
    assert report["decision_grade"] is True
    assert len(report["observations"][1]) == 3
    assert len(report["observations"][2]) == 3
    manifest = Path(report["evidence_manifest"])
    seal = json.loads((manifest.parent / "run_seal.json").read_text())
    assert seal["status"] == "complete"
    assert stat.S_IMODE(manifest.parent.stat().st_mode) == 0o700
    assert json.loads(manifest.read_text())["replacement"]["quality_history_by_tier"] == {"1": [3.0] * 3, "2": [3.0] * 3}


def test_execute_refuses_active_autopilot_before_any_evaluation(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    monkeypatch.setattr(runner, "autopilot_processes", lambda: ["123 autopilot.py start"])
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 75
    assert report["decision_grade"] is False
    assert "AutoPilot is active" in report["blockers"][0]
    assert FakeTower.calls == 0


def test_partial_or_error_observation_never_decision_grades(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    FakeTower.error_on_call = 4
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 2
    assert report["decision_grade"] is False
    assert report["observations"][2][0]["error_classification"] == {"request_or_scoring_error": 2}


def test_state_or_lineup_mutation_is_detected_after_execution(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    _patch_clean_environment(monkeypatch, mutate=args.registry_path)
    report, rc = runner.execute(args)
    assert rc == 2
    assert report["postconditions"]["checks"]["no_state_registry_lineup_mutation"] is False


def test_wrong_response_vector_never_decision_grades(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    FakeTower.wrong_vector_on_call = 5
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 2
    assert report["decision_grade"] is False
    assert report["observations"][2][1]["response_vector_matches_input"] is False


def test_prepare_rejects_wrong_era_or_closed_hold(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    args.state_path.write_text(json.dumps({
        "active_instrument_eras": {"eval_quality": "E7-eval-instrument"},
        "e8_quality_rebaseline": {"status": "closed"},
        "baseline_state": {"eval_quality_era": "E7-eval-instrument"},
    }))
    report = runner.prepare_report(args)
    assert report["decision_grade"] is False
    assert any("not E8" in blocker for blocker in report["blockers"])
    assert any("not open" in blocker for blocker in report["blockers"])


def test_monitor_persistence_failure_is_sticky_and_fails_closed(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    bad_path = tmp_path / "monitor-directory"
    bad_path.mkdir()
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args), bad_path)

    watcher.sample()

    assert watcher.fatal_error is not None
    assert watcher.samples[-1]["ok"] is False


def test_delayed_monitor_samples_prevent_atomic_publish(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)

    class DelayedWatcher:
        def __init__(self, _args, _binding, artifact_path):
            self._thread = type("Thread", (), {"is_alive": lambda self: False})()
            self.samples = []
            self.fatal_error = None
            self.artifact_path = artifact_path

        def start(self):
            runner.write_text(self.artifact_path, "{}\n")
            self.samples = [
                {"started_at": "2026-07-26T00:00:00Z", "finished_at": "2026-07-26T00:00:00Z", "ok": True},
                {"started_at": "2026-07-26T00:00:10Z", "finished_at": "2026-07-26T00:00:10Z", "ok": True},
            ]

        def stop(self):
            return self.samples

    monkeypatch.setattr(runner, "RuntimeWatcher", DelayedWatcher)
    report, rc = runner.execute(_args(tmp_path))

    assert rc == 2
    assert report["postconditions"]["checks"]["continuous_clean_monitor"] is False
    assert not Path(report["evidence_manifest"]).exists()


def test_repetition_artifacts_are_independent_and_vectors_are_pinned(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 0
    manifest = json.loads(Path(report["evidence_manifest"]).read_text())
    raw_paths = []
    for source in manifest["source_records"]:
        summary = json.loads(Path(source["path"]).read_text())
        raw_paths.extend(Path(item["path"]) for item in summary["observations"])
    assert len({path.name for path in raw_paths}) == 6
    assert len({runner.sha256_path(path) for path in raw_paths}) == 6
    for path in report["question_vectors"].values():
        vector = json.loads(Path(path).read_text())
        assert vector["n"] == 2


def test_tampered_sealed_bundle_is_no_longer_hash_consistent(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 0
    manifest_path = Path(report["evidence_manifest"])
    manifest = json.loads(manifest_path.read_text())
    source = manifest["source_records"][0]
    summary_path = Path(source["path"])
    summary = json.loads(summary_path.read_text())
    raw_path = Path(summary["observations"][0]["path"])
    raw = json.loads(raw_path.read_text())
    raw["n"] = 99
    raw_path.write_text(json.dumps(raw, sort_keys=True))
    seal = json.loads((manifest_path.parent / "run_seal.json").read_text())
    assert runner.sha256_path(raw_path) != seal["bundle_sha256"][str(raw_path)]


def test_terminal_stack_or_health_drift_never_publishes_acceptable_manifest(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    calls = 0
    def drifting_health(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"ok": calls < 3, "payload_sha256": "stable" if calls < 3 else "changed", "payload": {}}
    monkeypatch.setattr(runner, "api_health", drifting_health)
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 2
    assert report["decision_grade"] is False
    assert not Path(report["evidence_manifest"]).exists()


def test_ss_listener_identity_rejects_swapped_port_pid_ownership(monkeypatch) -> None:
    output = (
        'LISTEN 0 4096 127.0.0.1:8070 0.0.0.0:* users:(("llama-server",pid=222,fd=3))\n'
        'LISTEN 0 4096 127.0.0.1:8072 0.0.0.0:* users:(("llama-server",pid=111,fd=3))\n'
    )
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/ss")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    assert runner._missing_listener_identities({"8070": 111, "8072": 222}) == [
        "8070/pid=111",
        "8072/pid=222",
    ]


def test_ss_listener_identity_rejects_port_prefix_collision(monkeypatch) -> None:
    output = 'LISTEN 0 4096 127.0.0.1:18070 0.0.0.0:* users:(("llama-server",pid=111,fd=3))\n'
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/ss")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    assert runner._missing_listener_identities({"8070": 111}) == ["8070/pid=111"]


def test_ss_listener_identity_rejects_expected_pid_with_reuseport_co_listener(
    monkeypatch,
) -> None:
    output = (
        'LISTEN 0 4096 127.0.0.1:8070 0.0.0.0:* users:(("llama-server",pid=111,fd=3))\n'
        'LISTEN 0 4096 127.0.0.1:8070 0.0.0.0:* users:(("llama-server",pid=222,fd=4))\n'
    )
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/ss")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    assert runner._missing_listener_identities({"8070": 111}) == ["8070/pid=111"]


@pytest.mark.parametrize(
    "listener_inodes,pid_inodes",
    [
        ({"100", "200"}, {"100"}),
        ({"100"}, set()),
    ],
)
def test_proc_listener_identity_rejects_extra_or_unowned_inode(
    monkeypatch, listener_inodes, pid_inodes
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        runner,
        "_proc_tcp_listener_inodes",
        lambda: {8070: listener_inodes},
    )
    monkeypatch.setattr(
        runner,
        "_proc_pid_socket_inodes",
        lambda _pid: pid_inodes,
    )

    assert runner._missing_listener_identities({"8070": 111}) == ["8070/pid=111"]


def test_proc_listener_identity_rejects_recorded_pid_co_owner(monkeypatch) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        runner,
        "_proc_tcp_listener_inodes",
        lambda: {8070: {"100"}, 8072: {"200"}},
    )
    monkeypatch.setattr(
        runner,
        "_proc_pid_socket_inodes",
        lambda pid: {111: {"100"}, 222: {"100", "200"}}[pid],
    )

    assert runner._missing_listener_identities({"8070": 111, "8072": 222}) == [
        "8070/pid=111"
    ]


def test_runtime_watcher_completes_newline_only_short_write_before_fsync(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    _patch_clean_environment(monkeypatch)
    artifact = tmp_path / "watch.jsonl"
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args), artifact)
    original_write = runner.os.write
    write_payloads: list[bytes] = []
    fsync_calls: list[int] = []

    def short_write(fd: int, data) -> int:
        payload = bytes(data)
        write_payloads.append(payload)
        chunk = payload[:-1] if len(write_payloads) == 1 else payload
        return original_write(fd, chunk)

    monkeypatch.setattr(runner.os, "write", short_write)
    monkeypatch.setattr(runner.os, "fsync", lambda fd: fsync_calls.append(fd))

    watcher.sample()

    assert watcher.fatal_error is None
    assert write_payloads[-1] == b"\n"
    assert len(fsync_calls) == 1
    assert json.loads(artifact.read_text())["ok"] is True


def test_runtime_watcher_zero_write_is_fatal_and_never_fsyncs(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    _patch_clean_environment(monkeypatch)
    artifact = tmp_path / "watch.jsonl"
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args), artifact)
    fsync_calls: list[int] = []
    monkeypatch.setattr(runner.os, "write", lambda _fd, _data: 0)
    monkeypatch.setattr(runner.os, "fsync", lambda fd: fsync_calls.append(fd))

    watcher.sample()

    assert "invalid progress: 0" in str(watcher.fatal_error)
    assert watcher.samples[-1]["ok"] is False
    assert fsync_calls == []


def test_runtime_binding_pins_full_cmdline_model_flags_and_state_path(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    ports = list(range(8070, 8094))
    expected_binary = "/fake/llama-server"
    args.runtime_facts_path = tmp_path / "runtime-facts.json"
    args.orchestrator_state_path = tmp_path / "orchestrator-state.json"
    args.stack_priors_path = tmp_path / "stack-priors.yaml"
    args.registry_path = tmp_path / "registry-full.yaml"
    args.lean_registry_path = tmp_path / "registry-lean.yaml"
    args.runtime_facts_path.write_text(
        json.dumps(
            {
                "schema": "epyc.orchestrator.runtime_facts",
                "runtime_stack": {
                    "stack_numa_mode": "both",
                    "selected_ports": ports,
                    "selected_servers": [
                        {"port": port, "roles": ["test_role"]} for port in ports
                    ],
                    "paths": {"llama_server": expected_binary},
                },
            }
        )
    )
    args.orchestrator_state_path.write_text(
        json.dumps(
            {
                f"server_{port}": {
                    "pid": 100000 + port,
                    "port": port,
                    "model_path": f"/models/model-{port}.gguf",
                }
                for port in ports
            }
        )
    )
    for path in (args.stack_priors_path, args.registry_path, args.lean_registry_path):
        path.write_text("pinned: true\n")

    cmdlines = {
        100000 + port: [
            expected_binary,
            "-m",
            f"/models/model-{port}.gguf",
            "--mmproj",
            f"/models/mmproj-{port}.gguf",
            "--port",
            str(port),
            "--ctx-size",
            "4096",
        ]
        for port in ports
    }
    monkeypatch.setattr(runner.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(runner.os, "readlink", lambda _path: expected_binary)
    monkeypatch.setattr(runner, "process_cmdline", lambda pid: list(cmdlines[pid]))
    monkeypatch.setattr(runner, "_missing_listener_identities", lambda _pids: [])
    monkeypatch.setattr(
        runner,
        "runtime_artifact_identities",
        lambda paths, *, include_sha256: {
            path: {
                "path": path,
                "st_dev": 1,
                "st_ino": index,
                "st_size": 10,
                "st_mtime_ns": 20,
                **({"sha256": str(index).zfill(64)} if include_sha256 else {}),
            }
            for index, path in enumerate(dict.fromkeys(paths), 1)
        },
    )

    before = runner.runtime_binding(args)

    assert before["server_state_model_paths"]["8070"] == "/models/model-8070.gguf"
    assert before["server_model_flags"]["8070"]["mmproj"] == ["/models/mmproj-8070.gguf"]
    assert before["server_cmdlines"]["8070"][-2:] == ["--ctx-size", "4096"]

    cmdlines[108070][-1] = "8192"
    after = runner.runtime_binding(args)
    assert after != before
    assert after["server_cmdline_sha256"]["8070"] != before["server_cmdline_sha256"]["8070"]


def test_watcher_failure_aborts_before_next_repetition(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    _patch_clean_environment(monkeypatch)

    class FailingWatcher:
        latest = None

        def __init__(self, _args, _binding, artifact_path):
            type(self).latest = self
            self._thread = SimpleNamespace(is_alive=lambda: False)
            self.samples = [{"ok": True}]
            self.fatal_error = None
            self.artifact_path = artifact_path

        def start(self):
            runner.write_text(self.artifact_path, '{"ok":true}\n')

        def stop(self):
            return self.samples

    monkeypatch.setattr(runner, "RuntimeWatcher", FailingWatcher)
    original = runner.run_repetition

    def fail_after_first(*call_args, **call_kwargs):
        result = original(*call_args, **call_kwargs)
        assert FailingWatcher.latest is not None
        FailingWatcher.latest.fatal_error = "runtime monitor persistence failed"
        return result

    monkeypatch.setattr(runner, "run_repetition", fail_after_first)

    with pytest.raises(RuntimeError, match="runtime monitor persistence failed"):
        runner.execute(args)

    assert FakeTower.calls == 1
    assert not args.output_dir.exists()


def test_receipt_requires_canonical_path_and_current_runner_hash(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    canonical_receipt = tmp_path / "canonical-receipt.json"
    monkeypatch.setattr(runner, "PROTOCOL_RECEIPT", canonical_receipt)

    with pytest.raises(ValueError, match="canonical path"):
        runner.receipt_payload(args)

    args.protocol_receipt = canonical_receipt
    canonical_receipt.write_text(
        json.dumps(
            {
                "schema": "epyc.operator_e8_quality_baseline_protocol.v1",
                "decision": runner.PROTOCOL_DECISION,
                "era": "E8",
                "ratified_at": "2026-07-26T00:00:00+00:00",
                "operator_attestation": "test",
                "t2_decision": {},
                "protocol": {"protocol_id": runner.PROTOCOL_ID},
                "t1_core_file_sha256": "0" * 64,
                "expected_probe_groups": sorted(runner.EXPECTED_PROBE_GROUPS),
                "acceptance": {
                    "all_three_repetitions_clean": True,
                    "no_monitor_gap_seconds": 7,
                    "api_groups_exact": True,
                    "all_routes_frontdoor": True,
                    "sealed_atomic_publish": True,
                },
                "sha256": {"runner": "0" * 64},
                "repository_heads": {
                    "epyc_root": "a" * 40,
                    "epyc_orchestrator": "b" * 40,
                    "epyc_inference_research": "c" * 40,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="runner hash"):
        runner.receipt_payload(args)
    assert runner.RUNNER_PATH in runner.immutable_paths(args)


def test_fixed_t2_source_vector_fails_closed_on_zero_group_extract_patterns() -> None:
    tower = runner.EvalTower(url="http://127.0.0.1:8000", timeout=1)
    t1_questions, _t1_core_id = runner.question_vector(
        tower,
        tier=1,
        t1_core_id="core_v2",
        n=50,
        seed=runner.EVAL_SPEC_SEED,
    )
    questions, _core_id = runner.question_vector(
        tower,
        tier=2,
        t1_core_id="core_v2",
        n=500,
        seed=runner.EVAL_SPEC_SEED,
    )
    assert sum(question["scoring_method"] == "llm_judge" for question in t1_questions) == 4
    assert sum(question["scoring_method"] == "llm_judge" for question in questions) == 38
    invalid = {
        question["id"]: question["scoring_config"]["extract_pattern"]
        for question in questions
        if question["id"] in {"real_suite_v1_0043", "needle_039"}
    }
    assert invalid == {"real_suite_v1_0043": r"\d+", "needle_039": r"\d+"}
    with pytest.raises(ValueError, match="one capture group.*real_suite_v1_0043"):
        runner.validate_source_vector_scorer_config(questions, tier=2)


def test_protocol_proposal_rejects_invalid_source_vector_before_runtime_or_receipt_binding(
    monkeypatch,
) -> None:
    args = runner.parse_args(["--protocol-proposal", "--t2-n", "500"])

    def invalid_vector(_tower, *, tier, **_kwargs):
        return ([{
            "id": "real_suite_v1_0043" if tier == 2 else "t1-ok",
            "qid": "real_suite_v1_0043" if tier == 2 else "t1-ok",
            "suite": "suite_a",
            "prompt": "prompt",
            "expected": "256",
            "scoring_method": "exact_match",
            "scoring_config": {"extract_pattern": r"\d+"} if tier == 2 else {},
        }], "core_v2")

    monkeypatch.setattr(runner, "question_vector", invalid_vector)
    monkeypatch.setattr(
        runner,
        "runtime_binding",
        lambda *_args, **_kwargs: pytest.fail("runtime binding must not follow an invalid source vector"),
    )
    with pytest.raises(ValueError, match="real_suite_v1_0043"):
        runner.protocol_proposal(args)


def test_llm_judge_trace_is_total_for_blank_rows_and_row_identity_is_unique(
    tmp_path: Path,
) -> None:
    scorer = runner._load_orchestrator_debug_scorer()
    trace_path = tmp_path / "judge.jsonl"
    runner.write_text(trace_path, "")
    questions = [
        {"id": "judge-blank", "expected": "gold", "scoring_method": "llm_judge", "scoring_config": {}},
        {"id": "judge-fast", "expected": "gold", "scoring_method": "llm_judge", "scoring_config": {}},
    ]
    responses = [
        {"qid": "judge-blank", "answer": "", "correct": False, "error": None},
        {"qid": "judge-fast", "answer": "contains gold", "correct": True, "error": None},
    ]
    with runner.capture_llm_judge_traces(
        trace_path, default_api_url="http://127.0.0.1:8000"
    ):
        assert scorer._score_llm_judge("contains gold", "gold", {}) is True
    runner.seal_judge_trace_outcomes(
        trace_path,
        responses,
        questions,
        tier=2,
        repetition=1,
        default_api_url="http://127.0.0.1:8000",
    )
    audit = runner.validate_response_scoring(
        responses,
        questions,
        trace_path,
        default_api_url="http://127.0.0.1:8000",
        tier=2,
        repetition=1,
    )
    assert audit["judge_trace_rows"] == audit["expected_judge_trace_rows"] == 2
    traces = runner.load_jsonl(trace_path)
    assert [trace["mode"] for trace in traces] == ["blank_fast_failure", "substring_fast_path"]

    runner.write_text(trace_path, json.dumps(traces[0]) + "\n")
    with pytest.raises(ValueError, match="count does not match"):
        runner.validate_response_scoring(
            responses, questions, trace_path, default_api_url="http://127.0.0.1:8000", tier=2, repetition=1
        )


def test_llm_judge_trace_preserves_fast_and_network_scorer_behavior(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = runner._load_orchestrator_debug_scorer()
    original_scorer = scorer._score_llm_judge

    def judge_post(url, *, json, timeout):
        assert url == "http://127.0.0.1:8000/chat"
        assert json["force_role"] == runner.JUDGE_DEFAULT_ROLE
        return runner.httpx.Response(
            200,
            json={"answer": "true"},
            request=runner.httpx.Request("POST", url),
        )

    monkeypatch.setattr(runner.httpx, "post", judge_post)
    trace_path = tmp_path / "judge.jsonl"
    runner.write_text(trace_path, "")

    with runner.fixed_baseline_environment(tmp_path, "http://127.0.0.1:8000"):
        network_expected = original_scorer("final: mg/2", r"\frac{mg}{2}", {})
        with runner.capture_llm_judge_traces(
            trace_path, default_api_url="http://127.0.0.1:8000"
        ):
            assert scorer._score_llm_judge("contains gold", "gold", {}) is True
            assert (
                scorer._score_llm_judge("final: mg/2", r"\frac{mg}{2}", {})
                is network_expected
            )

    assert scorer._score_llm_judge is original_scorer
    traces = runner.load_jsonl(trace_path)
    assert [row["mode"] for row in traces] == [
        "substring_fast_path",
        "network_judge",
    ]
    assert runner.validate_llm_judge_trace(
        "contains gold",
        "gold",
        {},
        traces[0],
        default_api_url="http://127.0.0.1:8000",
    )
    assert runner.validate_llm_judge_trace(
        "final: mg/2",
        r"\frac{mg}{2}",
        {},
        traces[1],
        default_api_url="http://127.0.0.1:8000",
    )


def test_llm_judge_trace_is_thread_local_and_complete(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = runner._load_orchestrator_debug_scorer()

    def judge_post(url, *, json, timeout):
        return runner.httpx.Response(
            200,
            json={"answer": "false"},
            request=runner.httpx.Request("POST", url),
        )

    monkeypatch.setattr(runner.httpx, "post", judge_post)
    trace_path = tmp_path / "judge.jsonl"
    runner.write_text(trace_path, "")
    calls = [
        ("contains gold", "gold", {}),
        ("answer A", "answer B", {}),
        ("contains silver", "silver", {}),
        ("answer C", "answer D", {}),
    ]
    with runner.fixed_baseline_environment(tmp_path, "http://127.0.0.1:8000"):
        with runner.capture_llm_judge_traces(
            trace_path, default_api_url="http://127.0.0.1:8000"
        ):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda row: scorer._score_llm_judge(*row), calls))

    assert results == [True, False, True, False]
    traces = runner.load_jsonl(trace_path)
    assert len(traces) == len(calls)
    assert Counter(row["mode"] for row in traces) == {
        "network_judge": 2,
        "substring_fast_path": 2,
    }


def test_runtime_artifact_identity_detects_same_path_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"aaaa")
    cheap_before = runner.runtime_artifact_identities(
        [str(artifact), str(artifact)], include_sha256=False
    )
    full_before = runner.runtime_artifact_identities(
        [str(artifact)], include_sha256=True
    )

    artifact.write_bytes(b"bbbb")
    cheap_after = runner.runtime_artifact_identities(
        [str(artifact)], include_sha256=False
    )
    assert cheap_after != cheap_before

    previous_mtime = full_before[str(artifact.resolve())]["st_mtime_ns"]
    os.utime(artifact, ns=(previous_mtime, previous_mtime))
    full_after = runner.runtime_artifact_identities(
        [str(artifact)], include_sha256=True
    )
    assert full_after[str(artifact.resolve())]["sha256"] != full_before[str(artifact.resolve())]["sha256"]


def test_atomic_publish_noreplace_preserves_racing_destination(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "evidence"
    source.mkdir()
    destination.mkdir()
    (source / "source-marker").write_text("source")
    (destination / "destination-marker").write_text("destination")

    with pytest.raises(FileExistsError):
        runner.atomic_publish_noreplace(source, destination)

    assert (destination / "destination-marker").read_text() == "destination"
    assert (source / "source-marker").read_text() == "source"
