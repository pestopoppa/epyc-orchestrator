"""Tests for ProgressLogger interaction/delegation compatibility."""

from orchestration.repl_memory.progress_logger import ProgressLogger


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
