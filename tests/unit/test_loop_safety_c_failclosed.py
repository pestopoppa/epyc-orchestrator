"""LOOP-SAFETY-C2/C3: fail-closed on silent-empty inputs; telemetry isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---- C2: journal -------------------------------------------------------------------------
def test_live_default_journal_dir_without_shards_refuses(tmp_path, monkeypatch):
    from scripts.autopilot import experiment_journal as ej

    live = tmp_path / "orchestration"
    live.mkdir()
    monkeypatch.setattr(ej, "DEFAULT_JOURNAL_DIR", live)
    with pytest.raises(ej.MissingJournalError):  # negative control: used to read empty
        ej.ExperimentJournal()
    assert list(live.iterdir()) == []  # refusal wrote nothing
    ej.ExperimentJournal(create=True)  # explicit first-time init still works


def test_journal_create_false_refuses_any_dir_and_explicit_dir_still_creates(tmp_path):
    from scripts.autopilot import experiment_journal as ej

    with pytest.raises(ej.MissingJournalError):
        ej.ExperimentJournal(tmp_path / "gone", create=False)
    assert not (tmp_path / "gone").exists()
    ej.ExperimentJournal(tmp_path / "fresh")  # explicit directory = caller's choice
    assert (tmp_path / "fresh").is_dir()


def test_live_default_journal_dir_with_a_shard_loads(tmp_path, monkeypatch):
    from scripts.autopilot import experiment_journal as ej

    live = tmp_path / "orchestration"
    live.mkdir()
    (live / "autopilot_journal.jsonl").write_text("")
    monkeypatch.setattr(ej, "DEFAULT_JOURNAL_DIR", live)
    ej.ExperimentJournal()


# ---- C2: strategy store ------------------------------------------------------------------
class _Emb:
    def embed_text(self, text, *, side):
        import numpy as np
        v = np.ones(1024, dtype="float32")
        return v / np.linalg.norm(v)


def test_live_default_strategy_store_missing_db_refuses_and_create_opts_in(tmp_path, monkeypatch):
    from orchestration.repl_memory import strategy_store as ss

    live = tmp_path / "strategies"
    monkeypatch.setattr(ss, "DEFAULT_STRATEGY_PATH", live)
    with pytest.raises(ss.MissingStrategyStoreError):  # negative control: used to recreate
        ss.StrategyStore(live, embedder=_Emb())
    assert not live.exists()
    pytest.importorskip("faiss")  # the opt-in half needs the real FAISS store
    store = ss.StrategyStore(live, embedder=_Emb(), create=True)
    store.close()
    assert (live / "strategies.db").is_file()
    ss.StrategyStore(live, embedder=_Emb()).close()  # now exists -> opens without opt-in


def test_strategy_store_create_false_refuses_explicit_missing_path(tmp_path):
    from orchestration.repl_memory import strategy_store as ss

    with pytest.raises(ss.MissingStrategyStoreError):
        ss.StrategyStore(tmp_path / "x", embedder=_Emb(), create=False)
    assert not (tmp_path / "x").exists()


# ---- C3: telemetry isolation -------------------------------------------------------------
def test_default_telemetry_path_is_under_orchestration_not_tmp():
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
    mod = importlib.import_module("seeding_telemetry")
    default = Path(mod._DEFAULT_BATCH_HISTORY_PATH)
    assert default.parent.name == "orchestration" and default.name == "seeding_batch_telemetry.jsonl"
    assert not str(default).startswith("/mnt/raid0/llm/tmp")


def test_telemetry_write_lands_in_tmp_and_default_is_untouched(tmp_path):
    """Negative control for the autouse isolation fixture in tests/conftest.py."""
    import importlib
    import os

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
    mod = importlib.import_module("seeding_telemetry")
    default = Path(mod._DEFAULT_BATCH_HISTORY_PATH)
    live_legacy = Path("/mnt/raid0/llm/tmp/seeding_batch_telemetry.jsonl")
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (default, live_legacy) if p.exists()}
    assert os.environ["SEEDING_BATCH_TELEMETRY_PATH"].startswith(str(tmp_path))
    assert Path(mod._BATCH_HISTORY_PATH).parent == tmp_path
    mod._PERSIST_BATCH_HISTORY = True
    mod.record_batch_duration(3, 30.0)
    assert (tmp_path / "seeding_batch_telemetry.jsonl").read_text().strip()
    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (default, live_legacy) if p.exists()}
    assert after == before


# ---- D-B2: journal frontier size honours the epoch fence --------------------------------
def test_journal_summary_frontier_size_respects_epoch_fence(tmp_path):
    from datetime import datetime, timezone

    from scripts.autopilot.experiment_journal import ExperimentJournal, JournalEntry

    def entry(tid, ts):
        return JournalEntry(trial_id=tid, timestamp=ts, species="seeder", action_type="seed_batch",
                            tier=1, quality=2.4, speed=18.9, cost=0.5, reliability=1.0,
                            pareto_status="frontier", failure_analysis="")

    j = ExperimentJournal(journal_dir=tmp_path, segment_snapshots=False)
    j.record(entry(1, "2026-05-01T00:00:00Z"))
    j.record(entry(2, "2026-06-01T00:00:00Z"))
    j.record(entry(3, "2026-07-01T00:00:00Z"))
    assert "(all epochs, unfenced): 3" in j.summary_text()
    j.segment_snapshot_scope = {"exclude_before_ts": datetime(
        2026, 6, 15, tzinfo=timezone.utc).timestamp()}
    assert "(current epoch): 1" in j.summary_text()  # contradicted the fenced archive before
