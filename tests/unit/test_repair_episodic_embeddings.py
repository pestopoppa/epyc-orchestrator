from __future__ import annotations

from pathlib import Path

import pytest

from scripts.maintenance import repair_episodic_embeddings as repair


def test_run_repair_refuses_stale_snapshot(monkeypatch, tmp_path: Path) -> None:
    reports = iter(
        [
            repair.HealthReport(
                n_db_routing=100,
                n_faiss_vectors=10,
                n_reembedded=10,
                overlap_live=0.1,
                faiss_coverage=0.1,
                healthy=False,
                orphan_count=90,
            ),
            repair.HealthReport(
                n_db_routing=1200,
                n_faiss_vectors=10,
                n_reembedded=100,
                overlap_live=0.1,
                faiss_coverage=0.1,
                healthy=False,
                orphan_count=1190,
            ),
        ]
    )
    rebuild_calls: list[object] = []

    monkeypatch.setattr(repair, "diagnose", lambda *_args, **_kwargs: next(reports))
    monkeypatch.setattr(repair.subprocess, "call", lambda _cmd: 0)
    monkeypatch.setattr(
        repair,
        "rebuild_faiss",
        lambda *args, **kwargs: rebuild_calls.append((args, kwargs)),
    )

    with pytest.raises(SystemExit, match="refusing to swap a stale FAISS snapshot"):
        repair.run_repair(
            db_path=tmp_path / "episodic.db",
            faiss_path=tmp_path / "embeddings.faiss",
            id_map_path=tmp_path / "id_map.npy",
            reembedded_path=tmp_path / "reembedded.npz",
            max_db_growth=1000,
        )

    assert rebuild_calls == []


def test_run_repair_allows_stale_snapshot_guard_to_be_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        repair,
        "diagnose",
        lambda *_args, **_kwargs: repair.HealthReport(
            n_db_routing=100,
            n_faiss_vectors=10,
            n_reembedded=10,
            overlap_live=0.1,
            faiss_coverage=0.1,
            healthy=False,
            orphan_count=90,
        ),
    )
    monkeypatch.setattr(repair.subprocess, "call", lambda _cmd: 0)
    monkeypatch.setattr(repair, "rebuild_faiss", lambda **_kwargs: (42, tmp_path, tmp_path))

    assert (
        repair.run_repair(
            db_path=tmp_path / "episodic.db",
            faiss_path=tmp_path / "embeddings.faiss",
            id_map_path=tmp_path / "id_map.npy",
            reembedded_path=tmp_path / "reembedded.npz",
            max_db_growth=-1,
        )
        == 42
    )


def test_run_repair_rechecks_growth_at_pre_swap(monkeypatch, tmp_path: Path) -> None:
    reports = iter(
        [
            repair.HealthReport(
                n_db_routing=100,
                n_faiss_vectors=10,
                n_reembedded=10,
                overlap_live=0.1,
                faiss_coverage=0.1,
                healthy=False,
                orphan_count=90,
            ),
            repair.HealthReport(
                n_db_routing=100,
                n_faiss_vectors=10,
                n_reembedded=100,
                overlap_live=1.0,
                faiss_coverage=0.1,
                healthy=False,
                orphan_count=90,
            ),
            repair.HealthReport(
                n_db_routing=101,
                n_faiss_vectors=10,
                n_reembedded=100,
                overlap_live=0.99,
                faiss_coverage=0.1,
                healthy=False,
                orphan_count=91,
            ),
        ]
    )

    def fake_rebuild_faiss(**kwargs):
        kwargs["pre_swap_check"]()
        return (100, tmp_path, tmp_path)

    monkeypatch.setattr(repair, "diagnose", lambda *_args, **_kwargs: next(reports))
    monkeypatch.setattr(repair.subprocess, "call", lambda _cmd: 0)
    monkeypatch.setattr(repair, "rebuild_faiss", fake_rebuild_faiss)

    with pytest.raises(SystemExit, match="phase=pre-swap"):
        repair.run_repair(
            db_path=tmp_path / "episodic.db",
            faiss_path=tmp_path / "embeddings.faiss",
            id_map_path=tmp_path / "id_map.npy",
            reembedded_path=tmp_path / "reembedded.npz",
        )
