"""Tests for StrategyStore — FAISS + SQLite strategy memory."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator")


class MockEmbedder:
    """Deterministic hash-based embedder for testing (no model needed)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec


@pytest.fixture
def store(tmp_path):
    from orchestration.repl_memory.strategy_store import StrategyStore
    s = StrategyStore(path=tmp_path / "strategies", embedding_dim=1024, embedder=MockEmbedder())
    yield s
    s.close()


class TestStrategyStore:

    def test_store_and_count(self, store):
        sid = store.store(
            description="Disable self-speculation for dense models",
            insight="HSD net-negative on hybrid",
            source_trial_id=1,
            species="config_tuner",
        )
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID
        assert store.count() == 1

    def test_store_multiple(self, store):
        for i in range(5):
            store.store(
                description=f"Strategy {i}",
                insight=f"Insight {i}",
                source_trial_id=i,
                species="explorer",
            )
        assert store.count() == 5

    def test_retrieve_returns_results(self, store):
        store.store("Enable caching for read-heavy workloads", "Cache hit rate 90%",
                     source_trial_id=1, species="perf_tuner")
        store.store("Increase batch size for throughput", "2x throughput at batch=8",
                     source_trial_id=2, species="perf_tuner")
        results = store.retrieve("caching performance", k=5)
        assert len(results) >= 1
        assert results[0].similarity_score > 0

    def test_retrieve_empty_store(self, store):
        results = store.retrieve("anything", k=5)
        assert results == []

    def test_retrieve_with_species_filter(self, store):
        store.store("Strategy A", "Insight A", source_trial_id=1, species="alpha")
        store.store("Strategy B", "Insight B", source_trial_id=2, species="beta")
        store.store("Strategy C", "Insight C", source_trial_id=3, species="alpha")

        results = store.retrieve("Strategy", k=10, species="alpha")
        assert all(r.species == "alpha" for r in results)

    def test_retrieve_excludes_source_trial_ids(self, store):
        store.store("Strategy A", "Insight A", source_trial_id=1, species="alpha")
        store.store("Strategy B", "Insight B", source_trial_id=2, species="alpha")

        results = store.retrieve("Strategy", k=10, excluded_trial_ids={2})

        assert results
        assert all(r.source_trial_id != 2 for r in results)

    def test_retrieve_excludes_any_evidence_trial_id(self, store):
        sid = store.store(
            "Strategy A",
            "Insight A",
            source_trial_id=99,
            species="alpha",
            evidence_trial_ids=[1, 2],
        )
        store.store("Strategy B", "Insight B", source_trial_id=3, species="alpha")

        results = store.retrieve("Strategy", k=10, excluded_trial_ids={2})

        assert results
        assert all(r.id != sid for r in results)
        assert any(r.source_trial_id == 3 for r in results)

    def test_retrieve_exclusion_falls_back_to_source_trial_id(self, store):
        store.store("Legacy evidence", "Insight", source_trial_id=5, species="alpha")
        store._conn.execute("UPDATE strategies SET evidence_trial_ids = '[]'")
        store._conn.commit()

        results = store.retrieve("Legacy evidence", k=10, excluded_trial_ids={5})

        assert results == []

    def test_store_frontier_journal_entry_is_idempotent(self, store):
        entry = SimpleNamespace(
            trial_id=7,
            timestamp="2026-06-19T00:00:00Z",
            species="prompt_forge",
            action_type="code_mutation",
            quality=1.2345,
            speed=42.25,
            pareto_status="frontier",
            hypothesis="repair parser",
            expected_mechanism="targeted_fix",
            outcome_status="ok",
            bug_corrupted_by="",
            eval_details={},
        )

        sid = store.store_frontier_journal_entry(entry)
        sid_again = store.store_frontier_journal_entry(entry)

        assert sid == "journal-frontier-trial-7"
        assert sid_again == sid
        assert store.count() == 1
        results = store.retrieve("repair parser", k=5)
        assert len(results) == 1
        assert results[0].id == sid
        assert results[0].source_trial_id == 7
        assert results[0].evidence_trial_ids == [7]
        assert results[0].metadata["generated_from"] == "journal_frontier"
        assert results[0].metadata["journal_trial_id"] == 7

    def test_store_frontier_journal_entry_skips_unsafe_rows(self, store):
        base = {
            "trial_id": 8,
            "timestamp": "2026-06-19T00:00:00Z",
            "species": "prompt_forge",
            "action_type": "code_mutation",
            "quality": 1.0,
            "speed": 10.0,
            "pareto_status": "frontier",
            "hypothesis": "repair parser",
            "expected_mechanism": "targeted_fix",
            "outcome_status": "ok",
            "bug_corrupted_by": "",
            "eval_details": {},
        }

        assert store.store_frontier_journal_entry(
            SimpleNamespace(**{**base, "pareto_status": "dominated"})
        ) is None
        assert store.store_frontier_journal_entry(
            SimpleNamespace(**{**base, "outcome_status": "skipped"})
        ) is None
        assert store.store_frontier_journal_entry(
            SimpleNamespace(**{**base, "bug_corrupted_by": "resource_contention"})
        ) is None
        assert store.store_frontier_journal_entry(
            SimpleNamespace(
                **{
                    **base,
                    "eval_details": {"learning_exclusion": {"by": "mad_noise"}},
                }
            )
        ) is None
        assert store.count() == 0

    def test_frontier_journal_projection_report_finds_missing_and_unexpected(self, store):
        class FakeJournal:
            def entries_with_supersessions(self):
                return [
                    SimpleNamespace(
                        trial_id=1,
                        timestamp="2026-06-19T00:00:00Z",
                        species="prompt_forge",
                        action_type="code_mutation",
                        quality=1.2,
                        speed=40.0,
                        pareto_status="frontier",
                        hypothesis="repair parser",
                        expected_mechanism="targeted_fix",
                        outcome_status="ok",
                        bug_corrupted_by="",
                        eval_details={},
                    ),
                    SimpleNamespace(
                        trial_id=2,
                        timestamp="2026-06-19T00:00:00Z",
                        species="prompt_forge",
                        action_type="code_mutation",
                        quality=1.0,
                        speed=10.0,
                        pareto_status="frontier",
                        hypothesis="unsafe row",
                        expected_mechanism="targeted_fix",
                        outcome_status="ok",
                        bug_corrupted_by="resource_contention",
                        eval_details={},
                    ),
                ]

        store.store(
            "old unsafe projection",
            "q=1.000 s=10.0 mechanism=targeted_fix",
            source_trial_id=2,
            species="prompt_forge",
            metadata={"generated_from": "journal_frontier", "journal_trial_id": 2},
            evidence_trial_ids=[2],
            entry_id="journal-frontier-trial-2",
        )

        report = store.frontier_journal_projection_report(FakeJournal())

        assert report["ok"] is False
        assert report["expected_count"] == 1
        assert report["projected_count"] == 1
        assert report["missing"] == [
            {"trial_id": 1, "strategy_id": "journal-frontier-trial-1"}
        ]
        assert report["unexpected"] == [
            {"trial_id": 2, "strategy_id": "journal-frontier-trial-2"}
        ]

    def test_sync_frontier_journal_entries_inserts_missing_only(self, store):
        entry = SimpleNamespace(
            trial_id=9,
            timestamp="2026-06-19T00:00:00Z",
            species="prompt_forge",
            action_type="code_mutation",
            quality=1.2345,
            speed=42.25,
            pareto_status="frontier",
            hypothesis="repair parser",
            expected_mechanism="targeted_fix",
            outcome_status="ok",
            bug_corrupted_by="",
            eval_details={},
        )

        class FakeJournal:
            def entries_with_supersessions(self):
                return [entry]

        dry = store.sync_frontier_journal_entries(FakeJournal(), dry_run=True)
        assert dry["ok"] is False
        assert dry["would_insert_count"] == 1
        assert dry["inserted_count"] == 0
        assert store.count() == 0

        written = store.sync_frontier_journal_entries(FakeJournal(), dry_run=False)
        assert written["ok"] is True
        assert written["would_insert_count"] == 1
        assert written["inserted_count"] == 1
        assert store.count() == 1

    def test_frontier_journal_projection_report_flags_mismatched_projection(self, store):
        entry = SimpleNamespace(
            trial_id=11,
            timestamp="2026-06-19T00:00:00Z",
            species="prompt_forge",
            action_type="code_mutation",
            quality=1.2345,
            speed=42.25,
            pareto_status="frontier",
            hypothesis="repair parser",
            expected_mechanism="targeted_fix",
            outcome_status="ok",
            bug_corrupted_by="",
            eval_details={},
        )

        class FakeJournal:
            def entries_with_supersessions(self):
                return [entry]

        store.store(
            "projection with bad evidence",
            "q=1.000 s=10.0 mechanism=targeted_fix",
            source_trial_id=11,
            species="prompt_forge",
            metadata={"generated_from": "legacy"},
            evidence_trial_ids=[99],
            entry_id="journal-frontier-trial-11",
        )

        report = store.frontier_journal_projection_report(FakeJournal())

        assert report["ok"] is False
        assert report["missing_count"] == 0
        assert report["mismatch_count"] == 1
        assert report["mismatches"][0]["trial_id"] == 11
        assert set(report["mismatches"][0]["problems"]) == {
            "evidence_trial_ids",
            "metadata.generated_from",
            "metadata.journal_trial_id",
        }

    def test_retrieve_for_journal_applies_folded_evidence_exclusions(self, store):
        sid = store.store(
            "Strategy A",
            "Insight A",
            source_trial_id=99,
            species="alpha",
            evidence_trial_ids=[1, 2],
        )
        store.store("Strategy B", "Insight B", source_trial_id=3, species="alpha")

        class FakeJournal:
            def entries_with_supersessions(self):
                return [
                    SimpleNamespace(trial_id=2, bug_corrupted_by="superseded"),
                ]

        results = store.retrieve_for_journal("Strategy", journal=FakeJournal(), k=10)

        assert results
        assert all(r.id != sid for r in results)
        assert any(r.source_trial_id == 3 for r in results)

    def test_strategy_rows_for_compression_applies_folded_evidence_exclusions(self, store):
        excluded_id = store.store(
            "Strategy A",
            "Insight A",
            source_trial_id=99,
            species="alpha",
            evidence_trial_ids=[1, 2],
        )
        kept_id = store.store(
            "Strategy B",
            "Insight B",
            source_trial_id=3,
            species="alpha",
            evidence_trial_ids=[3],
        )

        class FakeJournal:
            def entries_with_supersessions(self):
                return [
                    SimpleNamespace(trial_id=2, bug_corrupted_by="superseded"),
                ]

        rows = store.strategy_rows_for_compression(journal=FakeJournal())

        row_ids = {row["id"] for row in rows}
        assert excluded_id not in row_ids
        assert kept_id in row_ids

    def test_strategy_rows_for_compression_window_counts_eligible_rows(self, store):
        for trial_id in range(1, 5):
            store.store(
                f"Strategy {trial_id}",
                f"Insight {trial_id}",
                source_trial_id=trial_id,
                species="alpha",
                evidence_trial_ids=[trial_id],
            )

        rows = store.strategy_rows_for_compression(
            window_trials=2,
            excluded_trial_ids={4},
        )

        assert [row["source_trial_id"] for row in rows] == [3, 2]

    def test_strategy_entries_for_distillation_applies_folded_evidence_exclusions(self, store):
        excluded_id = store.store(
            "Strategy A",
            "Insight A",
            source_trial_id=99,
            species="alpha",
            evidence_trial_ids=[1, 2],
        )
        kept_id = store.store(
            "Strategy B",
            "Insight B",
            source_trial_id=3,
            species="alpha",
            evidence_trial_ids=[3],
        )

        class FakeJournal:
            def entries_with_supersessions(self):
                return [
                    SimpleNamespace(trial_id=2, bug_corrupted_by="superseded"),
                ]

        entries = store.strategy_entries_for_distillation("raw", journal=FakeJournal())

        entry_ids = {entry["id"] for entry in entries}
        assert excluded_id not in entry_ids
        assert kept_id in entry_ids

    def test_strategy_entries_for_distillation_filters_low_validity(self, store):
        kept_id = store.store("Keep", "Insight", source_trial_id=1, species="alpha")
        low_id = store.store("Drop", "Insight", source_trial_id=2, species="alpha")
        for _ in range(30):
            store.update_validity(low_id, failure=True)

        entries = store.strategy_entries_for_distillation("raw", min_validity=0.10)

        entry_ids = {entry["id"] for entry in entries}
        assert kept_id in entry_ids
        assert low_id not in entry_ids

    def test_excluded_strategy_evidence_trial_ids_prefers_folded_view(self):
        from orchestration.repl_memory.strategy_store import (
            excluded_strategy_evidence_trial_ids,
        )

        class FakeJournal:
            def all_entries(self):
                return [
                    SimpleNamespace(
                        trial_id=1,
                        bug_corrupted_by="raw_only",
                    )
                ]

            def entries_with_supersessions(self):
                return [
                    SimpleNamespace(trial_id=1, bug_corrupted_by=""),
                    SimpleNamespace(trial_id=2, bug_corrupted_by="resource_contention"),
                    SimpleNamespace(trial_id=3, outcome_status="error"),
                    SimpleNamespace(trial_id=4, keep_revert_decision="excluded"),
                    SimpleNamespace(
                        trial_id=5,
                        eval_details={"learning_exclusion": {"by": "seq_accumulating"}},
                    ),
                    SimpleNamespace(trial_id="not-an-int", bug_corrupted_by="bad-row"),
                ]

        assert excluded_strategy_evidence_trial_ids(FakeJournal()) == {2, 3, 4, 5}

    def test_excluded_strategy_evidence_trial_ids_falls_back_to_all_entries(self):
        from orchestration.repl_memory.strategy_store import (
            excluded_strategy_evidence_trial_ids,
        )

        class FakeJournal:
            def all_entries(self):
                return [
                    SimpleNamespace(trial_id=7, bug_corrupted_by="operator_scrub"),
                    SimpleNamespace(trial_id=8, bug_corrupted_by=""),
                ]

        assert excluded_strategy_evidence_trial_ids(FakeJournal()) == {7}

    def test_excluded_strategy_evidence_trial_ids_tolerates_load_errors(self):
        from orchestration.repl_memory.strategy_store import (
            excluded_strategy_evidence_trial_ids,
        )

        class BrokenJournal:
            def all_entries(self):
                raise RuntimeError("journal unavailable")

        assert excluded_strategy_evidence_trial_ids(BrokenJournal()) == set()

    def test_metadata_roundtrip(self, store):
        meta = {"key": "value", "nested": {"a": 1}}
        store.store("Test", "Test insight", source_trial_id=1, species="test",
                     metadata=meta)
        results = store.retrieve("Test", k=1)
        assert len(results) == 1
        assert results[0].metadata["key"] == "value"
        assert results[0].metadata["nested"] == {"a": 1}
        assert results[0].metadata["insight_format"]["title"] == "Test"

    def test_to_dict_serialization(self, store):
        store.store("Serialize me", "Check dict", source_trial_id=7, species="serializer")
        results = store.retrieve("Serialize me", k=1)
        d = results[0].to_dict()
        assert isinstance(d, dict)
        assert d["species"] == "serializer"
        assert d["source_trial_id"] == 7
        assert d["evidence_trial_ids"] == [7]
        assert "id" in d
        assert "created_at" in d

    def test_close_is_safe(self, store):
        store.close()
        # Double close should not raise
        store.close()


class TestAP32InsightFormat:
    """AP-32: task-agnostic insight format metadata."""

    def test_store_records_derived_insight_format_metadata(self, store):
        sid = store.store(
            "Disable brittle benchmark-specific prompt anchors.",
            "Patterns tied to one suite should stay local until cross-suite evidence exists.",
            source_trial_id=11,
            species="prompt_forge",
        )

        row = store._conn.execute(
            "SELECT metadata_json FROM strategies WHERE id=?", (sid,)
        ).fetchone()
        meta = json.loads(row["metadata_json"])
        fmt = meta["insight_format"]

        assert fmt["version"] == 1
        assert fmt["title"] == "Disable brittle benchmark-specific prompt anchors"
        assert fmt["description"] == "Disable brittle benchmark-specific prompt anchors."
        assert fmt["generalized_content"].startswith("Patterns tied to one suite")
        assert fmt["specificity_flags"] == []

        results = store.retrieve("benchmark-specific prompt anchors", k=1)
        assert results[0].title == fmt["title"]
        assert results[0].generalized_content == fmt["generalized_content"]
        assert results[0].specificity_flags == []

    def test_store_accepts_explicit_generalized_content(self, store):
        sid = store.store(
            "Prefer mutation evidence that crosses benchmark families.",
            "Changed a concrete file after one trial.",
            source_trial_id=123,
            species="structural_lab",
            title="Prefer cross-suite mechanisms before promotion",
            generalized_content=(
                "Promote a mutation pattern only after evidence spans more than one "
                "benchmark family."
            ),
        )

        row = store._conn.execute(
            "SELECT insight, metadata_json FROM strategies WHERE id=?", (sid,)
        ).fetchone()
        meta = json.loads(row["metadata_json"])
        fmt = meta["insight_format"]

        assert row["insight"] == fmt["generalized_content"]
        assert fmt["title"] == "Prefer cross-suite mechanisms before promotion"
        assert fmt["specificity_flags"] == []

    def test_audit_insight_specificity_flags_task_specific_entries(self, store):
        sid = store.store(
            "scripts/autopilot/foo.py improved trial #123",
            "Keep commit abc1234 behavior from /mnt/raid0/llm/example/path.",
            source_trial_id=123,
            species="structural_lab",
        )

        findings = store.audit_insight_specificity()

        assert len(findings) == 1
        assert findings[0]["id"] == sid
        assert findings[0]["source_trial_id"] == 123
        assert findings[0]["species"] == "structural_lab"
        assert findings[0]["specificity_flags"] == [
            "absolute_path",
            "commit_hash",
            "repo_path",
            "trial_reference",
        ]


class TestAP28HybridRetrieval:
    """AP-28: FTS5 + RRF fusion, content-hash staleness, validity weighting."""

    def test_fts5_index_present(self, store):
        # FTS5 virtual table should exist after _init_schema
        rows = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='strategies_fts'"
        ).fetchall()
        assert len(rows) == 1
        assert getattr(store, "_fts_enabled", False) is True

    def test_entry_type_default_raw(self, store):
        sid = store.store("desc", "insight", source_trial_id=1, species="alpha")
        row = store._conn.execute(
            "SELECT entry_type FROM strategies WHERE id=?", (sid,)
        ).fetchone()
        assert row["entry_type"] == "raw"

    def test_entry_type_explicit(self, store):
        sid = store.store(
            "desc", "insight", source_trial_id=1, species="alpha", entry_type="pattern"
        )
        row = store._conn.execute(
            "SELECT entry_type FROM strategies WHERE id=?", (sid,)
        ).fetchone()
        assert row["entry_type"] == "pattern"

    def test_context_hash_recorded(self, store):
        sid = store.store("desc", "insight", source_trial_id=1, species="alpha")
        row = store._conn.execute(
            "SELECT context_hash FROM strategies WHERE id=?", (sid,)
        ).fetchone()
        # context files may not exist in tests → hash of empty input is fine,
        # we just need a non-NULL string value (incl. empty).
        assert row["context_hash"] is not None

    def test_default_context_files_use_live_worker_prompt(self):
        from orchestration.repl_memory.strategy_store import DEFAULT_CONTEXT_FILES

        assert DEFAULT_CONTEXT_FILES[-1].name == "worker_general.md"
        assert all(path.name != "worker_explore.md" for path in DEFAULT_CONTEXT_FILES)

    def test_bm25_exact_term_match(self, store):
        # FAISS via MockEmbedder is hash-based and has zero semantic fidelity.
        # BM25 must surface the entry that contains the exact query term.
        store.store("Speculation tuning for Qwen3.5", "Disable HSD",
                     source_trial_id=1, species="config_tuner")
        store.store("Increase ubatch size", "Throughput +20%",
                     source_trial_id=2, species="config_tuner")
        store.store("Cache hit rate optimisation", "Use prefix cache",
                     source_trial_id=3, species="config_tuner")

        results = store.retrieve("Qwen3.5 speculation", k=1)
        assert len(results) == 1
        assert "Qwen3.5" in results[0].description

    def test_rrf_fuses_both_signals(self, store):
        # Even when FAISS+BM25 disagree, RRF should produce a deterministic
        # ordering and never error.
        for i in range(8):
            store.store(f"Strategy {i}", f"Insight {i}",
                        source_trial_id=i, species="explorer")
        results = store.retrieve("Strategy 3", k=3)
        assert 1 <= len(results) <= 3
        # All returned entries must carry the diagnostic fields.
        for r in results:
            assert r.entry_type == "raw"
            assert r.staleness == 1.0
            assert 0.0 <= r.validity_score <= 1.0

    def test_quarantined_entries_excluded(self, store):
        sid = store.store("Quarantine me", "should be hidden",
                          source_trial_id=1, species="alpha")
        # Force quarantine via repeated failures (NIB2-41 pathway)
        for _ in range(20):
            store.update_validity(sid, failure=True)
        results = store.retrieve("Quarantine", k=5)
        assert all(r.id != sid for r in results)
        # And re-includable when explicitly requested
        results_full = store.retrieve("Quarantine", k=5, include_quarantined=True)
        assert any(r.id == sid for r in results_full)

    def test_staleness_penalises_old_entries(self, store):
        # Insert with fixed hash, then pretend the world moved on by
        # rewriting compute_context_hash to return a different string.
        store.store("Old entry", "from epoch A", source_trial_id=1, species="alpha")
        # Confirm it's currently fresh
        results = store.retrieve("Old entry", k=1)
        assert results[0].staleness == 1.0
        # Force a different epoch by monkeypatching the helper
        store.compute_context_hash = lambda *a, **kw: "DIFFERENTHASH0001"
        results = store.retrieve("Old entry", k=1)
        assert results[0].staleness == 0.5

    def test_backfill_fts_idempotent(self, store):
        # Insert directly into ``strategies`` bypassing the store() FTS path,
        # then call backfill_fts to populate the index.
        store._conn.execute(
            "INSERT INTO strategies(id, description, insight, source_trial_id, species, "
            "created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("legacy-1", "legacy desc", "legacy insight", 1, "old", "now", "{}"),
        )
        store._conn.commit()
        n1 = store.backfill_fts()
        assert n1 >= 1
        # Idempotent: second call should not double-insert.
        n2 = store.backfill_fts()
        assert n2 == 0

    def test_bm25_handles_punctuation_safely(self, store):
        # FTS5 MATCH chokes on raw punctuation; sanitiser must handle it.
        store.store("Mutation type=targeted_fix", "boost q",
                    source_trial_id=1, species="prompt_forge")
        # Punctuation-heavy query should not raise.
        results = store.retrieve("type=targeted_fix?!", k=2)
        assert isinstance(results, list)
