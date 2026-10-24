"""REPL-EMB-2.10: migration-tool gaps B-1 (sessions archive-only), B-2 (strategy store as its own
unit), B-3 (GGUF identity key in the marker). B-4 (consumers refuse a marked store at the wrong
dimension) and B-5 (`side`) are in test_eg2_adoption_seam.py. Synthetic stores only; the tool is
never pointed at a live store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from tests.unit.test_reembed_stores_eg2 import (  # noqa: F401 - fixtures + helpers
    DOC,
    FakeEG2,
    Layout,
    apply_args,
    cli,
    eg2,
    layout,
    make_sessions,
    require_faiss,
    template,
    tool,
    tree_hash,
)

PIN = "68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216"


# ── B-1 ──────────────────────────────────────────────────────────────────────────────────


def test_b1_sessions_with_vectors_block_by_default_and_name_the_flag(tmp_path, capsys):
    make_sessions(tmp_path / "se", embeddings=3)
    rc, out, _ = cli(capsys, "plan", "--session-dir", str(tmp_path / "se"), "--json")
    assert rc == 3 and "--sessions-archive-only" in out


def test_b1_archive_only_plans_applies_archives_and_rolls_back(layout, capsys, eg2):
    import shutil

    shutil.rmtree(layout.se)
    make_sessions(layout.se, embeddings=3)  # replaces the template's empty sessions store
    original = layout.snapshot()
    assert cli(capsys, "plan", *layout.stores)[0] == 3  # blocked without the flag
    rc, out, _ = cli(capsys, "plan", *layout.stores, "--sessions-archive-only", "--json")
    assert rc == 0
    sess = [c for u in json.loads(out)["units"] for c in u["components"] if c["kind"] == "sessions"][0]
    assert sess["detail"]["archive_only"] is True and sess["detail"]["archived_vector_rows"] == 3
    assert any("archive-only" in w for w in sess["warnings"])

    rc, _, err = cli(capsys, *apply_args(layout, eg2, "b1", "--sessions-archive-only"))
    assert rc == 0, err
    new = np.load(layout.se / "session_embeddings.npy")
    assert new.shape[1] == 256 and not new.any()  # empty 256-d file, nothing carried forward
    pre = layout.live / "sessions.pre-eg2-b1"
    assert np.load(pre / "session_embeddings.npy").shape[1] == 1024  # archived intact
    manifest = json.loads((layout.out / "manifest-b1.json").read_text())
    unit = [u for u in manifest["units"] if u["live"].endswith("sessions")][0]
    info = unit["reembed"]["sessions"]
    assert info["mode"] == "archive-only" and info["archived_vector_rows"] == 3
    assert info["archived_sha256"] == unit["old_files"]["session_embeddings.npy"]["sha256"]
    # no session text ever reached the embedding endpoint
    assert all(t.startswith(DOC) for t in eg2.texts())

    rc, _, err = cli(capsys, "rollback", "--manifest", str(layout.out / "manifest-b1.json"),
                     "--assume-stopped")  # fmt: skip
    assert rc == 0, err
    restored = layout.snapshot()
    assert {k: v for k, v in restored.items() if k.split("/")[0] == "sessions"} == {
        k: v for k, v in original.items() if k.split("/")[0] == "sessions"
    }


# ── B-2 ──────────────────────────────────────────────────────────────────────────────────


def make_strategy_dir(d: Path, n: int = 6) -> dict[str, str]:
    faiss = pytest.importorskip("faiss")
    d.mkdir(parents=True)
    con = sqlite3.connect(d / "strategies.db")
    con.execute(
        "CREATE TABLE strategies (id TEXT PRIMARY KEY, description TEXT NOT NULL, insight TEXT NOT NULL,"
        " source_trial_id INTEGER, species TEXT NOT NULL, created_at TEXT NOT NULL,"
        " metadata_json TEXT DEFAULT '{}')"
    )
    texts = {}
    for i in range(n):
        title = f"Strategy title {i}"
        meta = json.dumps({"insight_format": {"title": title}}) if i % 2 else "{}"
        con.execute(
            "INSERT INTO strategies VALUES (?,?,?,?,?,?,?)",
            (f"s{i}", f"description {i} about caching", f"insight {i}: prefer X", i, "sp", "2026-01-01", meta),
        )
        texts[f"s{i}"] = (title if i % 2 else None, f"description {i} about caching", f"insight {i}: prefer X")
    con.commit()
    con.close()
    rng = np.random.default_rng(4)
    mat = rng.standard_normal((n, 1024)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(1024)
    index.add(mat)
    faiss.write_index(index, str(d / "strategy_embeddings.faiss"))
    np.save(d / "strategy_id_map.npy", np.array([f"s{i}" for i in range(n)], dtype=object))
    return texts


def test_b2_strategy_dir_is_its_own_unit_and_migrates_with_the_same_marker(tmp_path, capsys, eg2):
    pytest.importorskip("faiss")
    import faiss

    live = tmp_path / "strategies"
    make_strategy_dir(live)
    before = tree_hash(live)
    rc, out, _ = cli(capsys, "plan", "--strategy-dir", str(live), "--json")
    assert rc == 0, out
    comp = json.loads(out)["units"][0]["components"][0]
    assert comp["kind"] == "strategy" and comp["status"] == "READY" and comp["vectors"] == 6
    assert tree_hash(live) == before  # plan wrote nothing

    out_dir = tmp_path / "out"
    rc, _, err = cli(capsys, "run", "--apply", "--strategy-dir", str(live), "--out-dir", str(out_dir),
                     "--embed-url", eg2.url, "--assume-stopped", "--run-id", "b2",
                     "--retry-backoff", "0", "--batch-size", "4")  # fmt: skip
    assert rc == 0, err
    index = faiss.read_index(str(live / "strategy_embeddings.faiss"))
    assert (index.d, index.ntotal) == (256, 6)
    marker = tool.verify_store_marker(live, expected_dim=256)
    assert marker["dim"] == 256
    # the text embedded is title + description + insight, as StrategyStore embeds it
    sent = {t[len(DOC):] for t in eg2.texts() if "probe" not in t}
    assert "Strategy title 1 description 1 about caching insight 1: prefer X" in sent
    assert any(t.endswith("description 0 about caching insight 0: prefer X") for t in sent)  # derived title
    assert (tmp_path / "strategies.pre-eg2-b2" / "strategy_embeddings.faiss").exists()
    # the real consumer: marked 256 refuses a default (1024) StrategyStore config
    from src.embedding_dims import StoreDimensionMismatch

    from orchestration.repl_memory.strategy_store import StrategyStore

    with pytest.raises(StoreDimensionMismatch):
        StrategyStore(path=live, embedding_dim=1024)


def test_b2_a_co_located_strategy_index_still_blocks_the_other_stores(layout, capsys):
    (layout.ep / "strategy_embeddings.faiss").write_bytes(b"x")
    rc, out, _ = cli(capsys, "plan", "--episodic-dir", str(layout.ep), "--json")
    assert rc == 3 and "strategy_embeddings.faiss" in out


# ── B-3 ──────────────────────────────────────────────────────────────────────────────────


def test_b3_marker_carries_the_pool_identity_key_and_consumers_can_compare_it(layout, capsys, eg2):
    assert cli(capsys, *apply_args(layout, eg2, "b3"))[0] == 0
    marker = json.loads((layout.ep / tool.MARKER_NAME).read_text())
    assert marker["model_file"] == "embeddinggemma-2-BF16.gguf"
    assert marker["gguf_sha256"] == PIN and marker["gguf_sha256_verified"] is False
    ok = tool.verify_store_marker(
        layout.ep, expected_dim=256, expected_model_file="embeddinggemma-2-BF16.gguf",
        expected_gguf_sha256=PIN,
    )  # fmt: skip
    assert ok["dim"] == 256
    with pytest.raises(tool.StoreMarkerError, match="model_file"):
        tool.verify_store_marker(layout.ep, expected_dim=256, expected_model_file="bge.gguf")
    with pytest.raises(tool.StoreMarkerError, match="gguf_sha256"):
        tool.verify_store_marker(layout.ep, expected_dim=256, expected_gguf_sha256="0" * 64)
    # the CLI exposes the same comparison
    rc, _, err = cli(capsys, "check-marker", str(layout.ep), "--expect-gguf-sha256", "0" * 64)
    assert rc == 2 and "gguf_sha256" in err


def test_b3_a_gguf_that_is_not_the_pin_is_refused_before_anything_is_written(layout, capsys, eg2, tmp_path):
    other = tmp_path / "other.gguf"
    other.write_bytes(b"not the adopted model")
    before = layout.snapshot()
    rc, _, err = cli(capsys, *apply_args(layout, eg2, "b3x", "--gguf-path", str(other)))
    assert rc == 2 and "adopted" in err
    assert layout.snapshot() == before and not list(layout.live.glob("*.migrating-*"))


def test_b3_markers_without_the_new_keys_still_verify(tmp_path):
    """Markers written before this change (no model_file/gguf_sha256) stay readable."""
    d = tmp_path / "old"
    d.mkdir()
    ident = tool.target_identity(tool.target_spec())
    marker = tool.build_marker(ident, "r", {})
    for k in ("model_file", "gguf_sha256", "gguf_sha256_verified"):
        marker.pop(k)
    (d / tool.MARKER_NAME).write_text(json.dumps(marker))
    assert tool.verify_store_marker(d, expected_dim=256, expected_identity=ident)["dim"] == 256
    with pytest.raises(tool.StoreMarkerError):  # but a consumer that asks for the key is refused
        tool.verify_store_marker(d, expected_dim=256, expected_gguf_sha256=PIN)
