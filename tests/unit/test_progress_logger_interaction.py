"""Tests for ProgressLogger interaction/delegation compatibility."""

import hashlib
import json
import os
from pathlib import Path

import orchestration.repl_memory.progress_logger as progress_logger_module
from orchestration.repl_memory.progress_logger import EventType, ProgressEntry, ProgressLogger


# pytest loads the real root conftest before importing test modules. Capture its
# construction-time destination at collection so the test below proves that the
# override was present before application imports, without creating a logger or
# writing a row as an import side effect.
_COLLECTION_PROGRESS_LOG_DIR = os.environ.get("ORCHESTRATOR_PROGRESS_LOG_DIR")


def test_log_delegation_alias_logs_delegate_interaction(tmp_path) -> None:
    logger = ProgressLogger(log_dir=tmp_path, buffer_size=10)

    logger.log_delegation(
        task_id="t1",
        complexity="complex",
        action="architect",
        confidence=0.75,
        difficulty_score=0.44,
        difficulty_band="medium",
    )

    entry = logger._buffer[0]
    assert entry.data["interaction_type"] == "delegate"
    assert entry.data["interaction_policy_version"] == logger.INTERACTION_POLICY_VERSION
    assert entry.data["delegation_policy_version"] == logger.DELEGATION_POLICY_VERSION


def test_log_interaction_records_non_delegate_type(tmp_path) -> None:
    logger = ProgressLogger(log_dir=tmp_path, buffer_size=10)

    logger.log_interaction(
        task_id="t2",
        complexity="complex",
        action="review_before_commit",
        confidence=0.6,
        interaction_type="consult",
    )

    entry = logger._buffer[0]
    assert entry.data["interaction_type"] == "consult"
    assert entry.data["action"] == "review_before_commit"


def test_log_consult_records_roles_and_skill(tmp_path) -> None:
    logger = ProgressLogger(log_dir=tmp_path, buffer_size=10)

    logger.log_consult(
        task_id="t3",
        skill="review_before_commit",
        consultant_role="architect_general",
        requester_role="coder_escalation",
        confidence=0.8,
        outcome="denied",
        reason="contention_skip",
    )

    entry = logger._buffer[0]
    assert entry.data["interaction_type"] == "consult"
    assert entry.data["skill"] == "review_before_commit"
    assert entry.data["consultant_role"] == "architect_general"
    assert entry.data["requester_role"] == "coder_escalation"
    assert entry.data["outcome"] == "denied"
    assert entry.data["reason"] == "contention_skip"


def test_default_logger_uses_construction_time_override_and_preserves_runtime_path(
    tmp_path, monkeypatch
) -> None:
    # This value was captured while pytest imported the test module, after the
    # actual root conftest set its temporary override and before any test ran.
    assert _COLLECTION_PROGRESS_LOG_DIR
    collection_dir = Path(_COLLECTION_PROGRESS_LOG_DIR)
    assert collection_dir.is_dir()
    # Construct and write only during this test, after collection. The real
    # conftest override must still route the no-arg logger into its temp dir.
    logger = ProgressLogger(buffer_size=1)
    assert logger.log_dir == collection_dir
    logger.log(ProgressEntry(EventType.TASK_STARTED, "synthetic-conftest-logdir", data={}))
    collection_rows = [
        json.loads(line)
        for path in collection_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["task_id"] for row in collection_rows] == ["synthetic-conftest-logdir"]

    live_sentinel = tmp_path / "live-progress"
    live_sentinel.mkdir()
    sentinel_file = live_sentinel / "sentinel.jsonl"
    sentinel_file.write_text("untouched\n", encoding="utf-8")
    isolated = tmp_path / "pytest-progress"

    monkeypatch.setattr(progress_logger_module, "DEFAULT_LOG_PATH", live_sentinel)
    monkeypatch.setenv(progress_logger_module.LOG_DIR_ENV, str(isolated))
    logger = ProgressLogger(buffer_size=1)
    assert logger.log_dir == isolated
    logger.log(ProgressEntry(EventType.TASK_STARTED, "synthetic-logdir-control", data={}))

    assert sentinel_file.read_text(encoding="utf-8") == "untouched\n"
    rows = [
        json.loads(line)
        for path in isolated.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["task_id"] for row in rows] == ["synthetic-logdir-control"]

    # Copy the actual written bytes while the root conftest's global temporary
    # directory still exists. These retained files are content copies, not a
    # reconstruction of the original directories, links, ownership, or inodes.
    retained = tmp_path / "actual-progress-jsonl-copies"
    retained.mkdir()
    records = []
    for label, original_dir in (("conftest-default", collection_dir), ("override-control", isolated)):
        originals = sorted(original_dir.glob("*.jsonl"))
        assert originals
        for index, original in enumerate(originals):
            assert original.is_file() and not original.is_symlink()
            raw = original.read_bytes()
            copied = retained / f"{label}-{index}.jsonl"
            copied.write_bytes(raw)
            assert copied.read_bytes() == original.read_bytes() == raw
            records.append({
                "original_path": str(original),
                "copy_path": str(copied),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "copy_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            })
    manifest = {
        "schema": "epyc.scg.actual-jsonl-content-copies/v1",
        "scope": "byte-identical content copies made before conftest temporary-directory teardown; original filesystem topology is not preserved",
        "records": records,
    }
    (retained / "copy-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert len(records) == 2

    explicit = tmp_path / "explicit-caller-path"
    assert ProgressLogger(log_dir=explicit).log_dir == explicit
    monkeypatch.delenv(progress_logger_module.LOG_DIR_ENV)
    assert ProgressLogger().log_dir == live_sentinel
