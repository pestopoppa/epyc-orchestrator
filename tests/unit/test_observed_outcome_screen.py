"""Pre-screen grouping and one-shot guards, without opening a real holdout."""

import hashlib
import json

import numpy as np
import pytest

from scripts.graph_router import screen_observed_outcomes as screen


def test_prepare_freezes_group_partition_and_refuses_reuse(tmp_path, monkeypatch):
    pool = tmp_path / "pool.jsonl"
    labels = tmp_path / "labels.jsonl"
    data = tmp_path / "data.npz"
    qids, suites, entries = [], [], []
    for i in range(540):
        suite = f"suite{i % 3}"
        prompt = " ".join(hashlib.sha256(f"{i}:{j}".encode()).hexdigest()[:12] for j in range(8))
        qids.append(hashlib.sha1(f"{suite}\0{prompt}".encode()).hexdigest()[:16])
        suites.append(suite)
        entries.append({"suite": suite, "prompt": prompt})
    pool.write_text("".join(json.dumps(row) + "\n" for row in entries))
    labels.write_text("fixture\n")
    np.savez(data, qids=qids, suites=suites, label_map=np.array(list(enumerate(screen.ROLES))))
    monkeypatch.setattr(screen, "DATA", data)
    monkeypatch.setattr(screen, "LABELS", labels)
    monkeypatch.setattr(screen, "POOL", pool)
    monkeypatch.setattr(screen, "DATA_HASH", screen.digest(data))
    monkeypatch.setattr(screen, "LABEL_HASH", screen.digest(labels))
    out = tmp_path / "frozen"
    screen.prepare(out)
    partition = json.loads((out / "partition.json").read_text())
    assert set(partition) == set(qids)
    assert {part["split"] for part in partition.values()} == {"train", "calibration", "test"}
    group_parts = {}
    for part in partition.values():
        assert group_parts.setdefault(part["group"], part["split"]) == part["split"]
    assert not (out / "TEST_OPENED").exists()
    with pytest.raises(FileExistsError):
        screen.prepare(out)
    (out / "FIT_STARTED").write_text("already used\n")
    with pytest.raises(ValueError, match="repeat is forbidden"):
        screen.run(out)
