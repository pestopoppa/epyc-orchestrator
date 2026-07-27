"""Tests for the single episodic memory-record contract.

Each test here pins one of the four defects the 2026-07-27 audit measured in the
old free-form write path.
"""
from __future__ import annotations

import json

from orchestration.repl_memory.memory_record import (
    EMBED_TEXT_MAX_CHARS,
    RECORD_VERSION,
    build_memory_record,
    context_size_bytes,
    record_from_legacy_context,
)


class TestEmbeddedIsNotStored:
    """The load-bearing invariant."""

    def test_full_objective_is_stored_untruncated(self):
        """The old path truncated to 200 chars AT WRITE TIME — text was gone."""
        objective = "x" * 5000
        rec = build_memory_record(objective=objective, task_type="chat")
        assert rec.to_context()["objective"] == objective
        assert len(rec.to_context()["objective"]) == 5000

    def test_embedding_text_is_bounded_but_storage_is_not(self):
        rec = build_memory_record(objective="y" * 9000, task_type="chat")
        assert len(rec.embedding_text()) <= EMBED_TEXT_MAX_CHARS
        assert len(rec.to_context()["objective"]) == 9000

    def test_the_work_is_stored_but_never_embedded(self):
        """Embedding the answer would make retrieval match solutions to solutions."""
        rec = build_memory_record(
            objective="sort a list",
            answer="def f(x): return sorted(x)",
            tool_calls=[{"tool": "grep", "args": {"pattern": "sort"}}],
            repl_steps=[{"step": 1, "code": "print(1)"}],
            reasoning="pick the builtin",
        )
        text = rec.embedding_text()
        assert "sort a list" in text
        for leaked in ("def f(x)", "grep", "print(1)", "pick the builtin"):
            assert leaked not in text, f"{leaked!r} leaked into the embedding text"

        work = rec.to_context()["work"]
        assert work["answer"] == "def f(x): return sorted(x)"
        assert work["tool_calls"][0]["tool"] == "grep"
        assert work["repl_steps"][0]["code"] == "print(1)"
        assert work["reasoning"] == "pick the builtin"


class TestTelemetryStaysOutOfTheIndex:
    """27,123 of 54,960 rows were number-blobs embedded as text."""

    def test_metrics_are_stored_but_not_embedded(self):
        rec = build_memory_record(
            objective="count primes",
            task_type="coder",
            source="external",
            metrics={
                "question_id": "mbpp_0207",
                "elapsed_seconds": 79.83,
                "tokens_generated": 2759,
                "predicted_tps": 18.87,
            },
        )
        text = rec.embedding_text()
        for leaked in ("mbpp_0207", "79.83", "2759", "18.87", "elapsed_seconds"):
            assert leaked not in text, f"telemetry {leaked!r} leaked into the embedding"
        assert rec.to_context()["metrics"]["tokens_generated"] == 2759

    def test_a_record_with_no_objective_is_not_task_memory(self):
        telemetry_only = build_memory_record(
            objective=None, metrics={"elapsed_seconds": 1.0}, source="external"
        )
        assert telemetry_only.is_task_memory() is False
        assert build_memory_record(objective="real task").is_task_memory() is True

    def test_whitespace_only_objective_is_not_task_memory(self):
        assert build_memory_record(objective="   \n\t ").is_task_memory() is False


class TestOneEmbeddingConvention:
    """Four conventions coexisted, so the vector encoded its writer."""

    def test_same_task_from_different_sources_embeds_identically(self):
        a = build_memory_record(objective="solve it", task_type="chat", source="progress_log")
        b = build_memory_record(
            objective="solve it",
            task_type="chat",
            source="external",
            metrics={"elapsed_seconds": 4.2},
        )
        assert a.embedding_text() == b.embedding_text()

    def test_convention_shape_is_stable(self):
        rec = build_memory_record(objective="do X", task_type="chat", priority="interactive")
        assert rec.embedding_text() == "type:chat | objective:do X | priority:interactive"

    def test_optional_fields_are_omitted_not_stringified(self):
        rec = build_memory_record(objective="do X")
        assert rec.embedding_text() == "objective:do X"
        assert "None" not in rec.embedding_text()


class TestLegacyAdaptation:
    """Both historical writer shapes must map onto the contract."""

    def test_path_a_progress_log_shape(self):
        rec = record_from_legacy_context(
            {"task_type": "chat", "objective": "build an index", "priority": "interactive"}
        )
        assert rec.objective == "build an index"
        assert rec.task_type == "chat"
        assert rec.metrics == {}

    def test_path_b_external_shape_routes_telemetry_to_metrics(self):
        rec = record_from_legacy_context(
            {
                "task_type": "coder",
                "task_description": "write a parser",
                "source": "external",
                "question_id": "mbpp_0207",
                "elapsed_seconds": 79.8,
                "tokens_generated": 2759,
            }
        )
        assert rec.objective == "write a parser"
        assert rec.source == "external"
        assert rec.metrics["question_id"] == "mbpp_0207"
        assert rec.metrics["tokens_generated"] == 2759
        # and none of it reaches the embedding
        assert "mbpp_0207" not in rec.embedding_text()

    def test_telemetry_only_legacy_row_is_recognised_as_not_task_memory(self):
        """Exactly the 27,123-row population."""
        rec = record_from_legacy_context(
            {
                "task_type": "coder",
                "source": "external",
                "question_id": "mbpp_0207",
                "elapsed_seconds": 79.8,
                "tokens_generated": 2759,
            }
        )
        assert rec.is_task_memory() is False

    def test_round_trip_through_the_contract_is_stable(self):
        original = build_memory_record(
            objective="a task",
            task_type="chat",
            answer="an answer",
            source="seed",
            metrics={"elapsed_seconds": 1.5},
            extra={"request_id": "abc"},
        )
        again = record_from_legacy_context(original.to_context())
        assert again.to_context() == original.to_context()
        assert "record_version" not in again.metrics
        assert again.embedding_text() == original.embedding_text()


class TestContractMarker:
    def test_every_context_carries_a_version(self):
        ctx = build_memory_record(objective="x").to_context()
        assert ctx["record_version"] == RECORD_VERSION

    def test_context_is_json_serializable(self):
        ctx = build_memory_record(
            objective="x", tool_calls=[{"a": 1}], metrics={"b": 2.5}
        ).to_context()
        assert json.loads(json.dumps(ctx))["objective"] == "x"

    def test_size_accounting_is_available(self):
        small = build_memory_record(objective="x").to_context()
        big = build_memory_record(objective="x", answer="y" * 10000).to_context()
        assert context_size_bytes(big) > context_size_bytes(small) + 9000
