"""Synthetic-only tests for the prospective distillation input receipt writer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.autopilot.species.distillation_pool_capture import canonical_bytes, write_capture


class _Query:
    size = 1024

    @staticmethod
    def tobytes() -> bytes:
        return b"zero-query-fixture"


def _fixture(tmp_path: Path, *, projected=True):
    source = tmp_path / "structural_lab.py"
    source.write_text("# synthetic fixture source\n")
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    db = store_dir / "episodic.db"
    db.write_bytes(b"synthetic database marker")
    index = SimpleNamespace(index_path=store_dir / "embeddings.faiss",
                            id_map_path=store_dir / "id_map.npy")
    store = SimpleNamespace(sqlite_path=db, storage_dir=store_dir,
                            _embedding_store=index, use_faiss=True)
    memory = SimpleNamespace(
        id="memory-1", action_type="escalation", q_value=0.8, outcome=None,
        model_id=None, context={"private_task_text": "must not be captured",
                                "extra": {"from_tier": "worker", "to_tier": None}},
    )
    projected_records = ([{"escalations": [{"memory_id": "memory-1"}]}]
                         if projected else [{"escalations": []}])
    legacy_pool = {"total": 1, "by_outcome": {"unknown": 1}, "min_q_value": 0.7}
    result = write_capture(
        repo_root=tmp_path,
        output_dir=tmp_path,
        source_path=source,
        loaded_function=lambda: None,
        store=store,
        memories=[memory],
        teacher_requested="claude",
        teacher_declared_model_id="declared-only",
        max_trajectories=20,
        min_q_value=0.7,
        query_vector=_Query(),
        projected_records=projected_records,
        legacy_m25_pool=legacy_pool,
        synthetic=True,
    )
    return result


def test_writer_seals_exact_metadata_and_marks_fixture_synthetic(tmp_path):
    result = _fixture(tmp_path)
    path = Path(result["path"])
    record = json.loads(path.read_bytes())
    assert result["verdict"] is True
    assert record["synthetic"] is True
    assert record["teacher"]["requested"] == "claude"
    assert record["teacher"]["declared_model_id"] == "declared-only"
    assert record["teacher"]["actual_served_model_id"] is None
    assert record["counts"]["native_escalation_count"] == 1
    assert record["counts"]["projected_escalation_count"] == 1
    assert set(record["selected_rows"][0]) == {
        "id", "action_type", "q_value", "outcome", "model_id", "from_tier", "to_tier"
    }
    assert "private_task_text" not in path.read_text()
    body = dict(record)
    claimed = body.pop("receipt_sha256")
    assert hashlib.sha256(canonical_bytes(body)).hexdigest() == claimed
    assert path.stat().st_mode & 0o777 == 0o600


def test_false_integrity_verdict_is_persisted(tmp_path):
    result = _fixture(tmp_path, projected=False)
    record = json.loads(Path(result["path"]).read_bytes())
    assert result["verdict"] is False
    assert record["integrity"]["verdict"] is False
    assert "escalation_projection_matches_native_action_type" in record["integrity"]["errors"]
