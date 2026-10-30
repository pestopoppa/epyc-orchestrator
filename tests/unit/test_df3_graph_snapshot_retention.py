"""D-f3a bounds observational graph history in the existing writer transaction."""

from __future__ import annotations

import json
from datetime import datetime as RealDateTime, timezone

import pytest

import src.session.sqlite_store as sqlite_store_module
from src.session import Session, SQLiteSessionStore
from src.session.lease import StaleFencingToken


def _store(tmp_path, *, runs=100, per_run=100):
    return SQLiteSessionStore(
        db_path=tmp_path / "sessions.db",
        embeddings_path=tmp_path / "embeddings.npy",
        graph_snapshot_retention_runs=runs,
        graph_snapshot_retention_per_run=per_run,
    )


def _write(store, run_id: str, data: str) -> str:
    return store.save_graph_snapshot(run_id, json.dumps(data), "state_snapshot")


def test_existing_constructor_call_shape_keeps_bounded_defaults(tmp_path):
    store = SQLiteSessionStore(
        db_path=tmp_path / "sessions.db",
        embeddings_path=tmp_path / "embeddings.npy",
    )
    try:
        assert store.graph_snapshot_retention_runs == 100
        assert store.graph_snapshot_retention_per_run == 100
    finally:
        store.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("graph_snapshot_retention_runs", 0),
        ("graph_snapshot_retention_runs", -1),
        ("graph_snapshot_retention_runs", True),
        ("graph_snapshot_retention_per_run", 0),
        ("graph_snapshot_retention_per_run", -1),
        ("graph_snapshot_retention_per_run", 1.5),
    ],
)
def test_retention_limits_must_be_positive_integers(tmp_path, field, value):
    kwargs = {
        "db_path": tmp_path / "sessions.db",
        "embeddings_path": tmp_path / "embeddings.npy",
        "graph_snapshot_retention_runs": 100,
        "graph_snapshot_retention_per_run": 100,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        SQLiteSessionStore(**kwargs)


def test_prunes_old_runs_and_old_rows_within_run(tmp_path):
    store = _store(tmp_path, runs=2, per_run=2)
    try:
        old_ids = [_write(store, "old", str(i)) for i in range(3)]
        current_ids = [_write(store, "current", str(i)) for i in range(3)]
        assert [r["id"] for r in store.get_graph_snapshots("old")] == old_ids[-2:]
        assert [r["id"] for r in store.get_graph_snapshots("current")] == current_ids[-2:]

        newest_id = _write(store, "newest", "new")
        assert store.get_graph_snapshots("old") == []
        assert [r["id"] for r in store.get_graph_snapshots("current")] == current_ids[-2:]
        assert [r["id"] for r in store.get_graph_snapshots("newest")] == [newest_id]
        with store._get_connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM graph_snapshots").fetchone()[0] <= 4
    finally:
        store.close()


def test_new_snapshot_keeps_its_active_run_recent(tmp_path):
    store = _store(tmp_path, runs=2, per_run=2)
    try:
        _write(store, "run-a", "a1")
        _write(store, "run-b", "b1")
        latest_a = _write(store, "run-a", "a2")
        assert [r["id"] for r in store.get_graph_snapshots("run-a")][-1] == latest_a
        # A new write for an existing run refreshes its recency; it does not
        # evict that run merely because another run was inserted meanwhile.
        _write(store, "run-c", "c1")
        assert store.get_graph_snapshots("run-b") == []
        assert [r["id"] for r in store.get_graph_snapshots("run-a")][-1] == latest_a
    finally:
        store.close()


def test_first_new_write_bounds_oversized_legacy_runs(tmp_path):
    store = _store(tmp_path, runs=2, per_run=2)
    try:
        with store._get_connection() as conn:
            for run_number in range(3):
                for snapshot_number in range(4):
                    conn.execute(
                        """
                        INSERT INTO graph_snapshots (
                            id, run_id, session_id, snapshot_type, created_at,
                            fencing_token, data
                        ) VALUES (?, ?, NULL, 'state_snapshot', ?, NULL, '{}')
                        """,
                        (
                            f"legacy-{run_number}-{snapshot_number}",
                            f"legacy-run-{run_number}",
                            f"2020-01-0{run_number + 1}T00:00:0{snapshot_number}+00:00",
                        ),
                    )
            conn.commit()

        # The next write upgrades old overfull history as well as applying
        # retention to its own run.
        _write(store, "new-run", "new")
        with store._get_connection() as conn:
            counts = conn.execute(
                "SELECT run_id, COUNT(*) AS n FROM graph_snapshots GROUP BY run_id"
            ).fetchall()
            total_rows = conn.execute(
                "SELECT COUNT(*) FROM graph_snapshots"
            ).fetchone()[0]
        assert len(counts) == 2
        assert all(row["n"] <= 2 for row in counts)
        assert total_rows <= 2 * 2
        assert {row["run_id"] for row in counts} == {"legacy-run-2", "new-run"}
        assert len(store.get_graph_snapshots("legacy-run-2")) == 2
        assert len(store.get_graph_snapshots("new-run")) == 1
    finally:
        store.close()


def test_run_retention_is_global_across_sessions(tmp_path):
    store = _store(tmp_path, runs=1, per_run=1)
    try:
        session_a = Session.create(name="a", working_directory=str(tmp_path))
        session_b = Session.create(name="b", working_directory=str(tmp_path))
        store.create_session(session_a)
        store.create_session(session_b)
        lease_a = store.leases.try_acquire(session_a.id, ttl_s=30)
        lease_b = store.leases.try_acquire(session_b.id, ttl_s=30)

        store.save_graph_snapshot(
            "run-a", json.dumps("a"), "state_snapshot",
            session_id=session_a.id, fencing_token=lease_a.fencing_token,
        )
        store.save_graph_snapshot(
            "run-b", json.dumps("b"), "state_snapshot",
            session_id=session_b.id, fencing_token=lease_b.fencing_token,
        )

        # The configured cap is intentionally a table-wide history policy,
        # not a separate quota per session.
        assert store.get_graph_snapshots("run-a") == []
        rows = store.get_graph_snapshots("run-b")
        assert len(rows) == 1
        assert rows[0]["session_id"] == session_b.id
        store.leases.release(lease_a)
        store.leases.release(lease_b)
    finally:
        store.close()


def test_created_at_ties_keep_latest_insertions_deterministically(tmp_path, monkeypatch):
    fixed = RealDateTime(2026, 10, 30, tzinfo=timezone.utc)

    class FixedDateTime:
        @staticmethod
        def now(tz=None):
            return fixed if tz else fixed.replace(tzinfo=None)

    monkeypatch.setattr(sqlite_store_module, "datetime", FixedDateTime)
    store = _store(tmp_path, runs=2, per_run=2)
    try:
        ids = [_write(store, "same-time", str(i)) for i in range(3)]
        rows = store.get_graph_snapshots("same-time")
        assert [r["id"] for r in rows] == ids[-2:]
        assert rows[0]["created_at"] == rows[1]["created_at"]
    finally:
        store.close()


def test_stale_fence_refusal_does_not_insert_or_prune(tmp_path):
    store = _store(tmp_path, runs=1, per_run=1)
    try:
        session = Session.create(name="retention", working_directory=str(tmp_path))
        store.create_session(session)
        old_lease = store.leases.try_acquire(session.id, ttl_s=30)
        kept_id = store.save_graph_snapshot(
            "kept-run", "{}", "state_snapshot",
            session_id=session.id, fencing_token=old_lease.fencing_token,
        )
        store.leases.release(old_lease)
        new_lease = store.leases.try_acquire(session.id, ttl_s=30)
        with pytest.raises(StaleFencingToken):
            store.save_graph_snapshot(
                "rejected-run", "{}", "state_snapshot",
                session_id=session.id, fencing_token=old_lease.fencing_token,
            )
        assert [r["id"] for r in store.get_graph_snapshots("kept-run")] == [kept_id]
        assert store.get_graph_snapshots("rejected-run") == []
        store.leases.release(new_lease)
    finally:
        store.close()


def test_failure_after_pruning_rolls_back_insert_and_deletes(tmp_path, monkeypatch):
    store = _store(tmp_path, runs=1, per_run=1)
    try:
        original_id = _write(store, "original", "keep me")
        prune = store._prune_graph_snapshots

        def prune_then_fail(conn):
            prune(conn)
            raise RuntimeError("injected after-prune failure")

        monkeypatch.setattr(store, "_prune_graph_snapshots", prune_then_fail)
        with pytest.raises(RuntimeError, match="after-prune"):
            _write(store, "replacement", "must roll back")

        assert [r["id"] for r in store.get_graph_snapshots("original")] == [original_id]
        assert store.get_graph_snapshots("replacement") == []
    finally:
        store.close()
