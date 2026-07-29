#!/usr/bin/env python3
"""A3 — diagnose and repair orphan embeddings in the episodic store.

Runs in two modes:

    --diagnose-only       — read-only. Prints a health report and exits 0 if healthy,
                            1 if orphans detected. Safe to call from preflight.
    --repair              — diagnose + repair. Runs reembed_episodic_store.py
                            (launches BGE servers), then rebuilds embeddings.faiss
                            and id_map.npy atomically.

Health definition:

    n_db_routing       = COUNT(*) FROM memories WHERE action_type='routing'
    n_db_indexed       = COUNT(*) FROM memories with a nonempty action_type
    n_faiss_vectors    = embeddings.faiss IndexFlatIP ntotal (read-only check)
    n_id_map           = number of IDs in id_map.npy
    n_reembedded_npz   = number of IDs in reembedded.npz (if present)
    id_map_overlap     = |id_map_ids ∩ live_indexed_ids| / n_db_indexed
    reembedded_overlap = |reembedded_ids ∩ live_indexed_ids| / n_db_indexed

A live store is "orphaned" if n_faiss_vectors / n_db_indexed < 0.99,
id_map.npy does not match embeddings.faiss, id_map lacks any live indexed IDs,
or id_map has stale/duplicate IDs. These conditions fire if FAISS was reset,
id_map.npy is stale/wrong, or an indexed action type falls outside the
headline routing count.

``reembedded.npz`` is an offline training artifact, not a retrieval backend.
Its coverage is reported separately and never changes live-store health. This
matters after a live-only reseed: retained historical embeddings deliberately
do not match the rebuilt live ``memories`` table.

Usage:
    python3 scripts/maintenance/repair_episodic_embeddings.py --diagnose-only
    python3 scripts/maintenance/repair_episodic_embeddings.py --repair [--servers 6] [--batch-size 128]

The repair step:
    1. Calls reembed_episodic_store.py to produce a fresh reembedded.npz
       for every live indexed row (uses existing parallel-BGE primitive —
       same configured-server pattern).
    2. Builds a new IndexFlatIP from the fresh embeddings.
    3. Atomically swaps embeddings.faiss + id_map.npy into place (writes to .new,
       then renames). Originals are backed up to .pre-repair-<timestamp>.
    4. Validates: re-opens FAISS and checks ntotal matches the embedding count.

Repair is idempotent — re-running on a healthy store is a no-op after the diagnose
gate.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.indexed_memory_policy import indexed_memory_predicate

try:
    from scripts.server.stack_manifest import EMBEDDER_PORTS
except Exception:  # pragma: no cover - maintenance fallback for partial checkouts
    EMBEDDER_PORTS = [8090, 8091, 8092, 8093, 8094, 8095]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repair_embeddings")

DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "orchestration/repl_memory/sessions"
DEFAULT_DB_PATH = DEFAULT_SESSIONS_DIR / "episodic.db"
DEFAULT_FAISS_PATH = DEFAULT_SESSIONS_DIR / "embeddings.faiss"
DEFAULT_ID_MAP_PATH = DEFAULT_SESSIONS_DIR / "id_map.npy"
DEFAULT_REEMBEDDED_PATH = DEFAULT_SESSIONS_DIR / "reembedded.npz"
DEFAULT_FAISS_LOCK_PATH = DEFAULT_SESSIONS_DIR / ".episodic_faiss.lock"
REEMBED_SCRIPT = PROJECT_ROOT / "scripts/graph_router/reembed_episodic_store.py"
# Retrieval is materially degraded well before catastrophic 50% coverage loss.
# Keep a small append-lag allowance for live writers, but require the live
# FAISS/id_map mirror must cover nearly all live indexed rows before health passes.
HEALTH_THRESHOLD = 0.99
MIN_ORPHANS_TO_REPAIR = 1000  # don't repair if delta is small (< this many orphans)
DEFAULT_EMBEDDER_SERVERS = len(EMBEDDER_PORTS)
DEFAULT_EMBEDDER_BASE_PORT = min(EMBEDDER_PORTS)
DEFAULT_MAX_DB_GROWTH = 0


class HealthReport(NamedTuple):
    n_db_routing: int
    n_faiss_vectors: int
    n_reembedded: int
    overlap_live: float
    faiss_coverage: float
    healthy: bool
    orphan_count: int
    n_id_map: int = 0
    id_map_overlap_live: float = 1.0
    id_map_matches_faiss: bool = True
    n_db_indexed: int = 0
    missing_id_count: int = 0
    stale_id_count: int = 0
    reembedded_missing_count: int = 0
    reembedded_stale_count: int = 0


@contextmanager
def _exclusive_file_lock(path: Path):
    """Match EpisodicStore's cross-process FAISS mutation lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _assert_db_growth_within_bound(
    db_path: Path,
    faiss_path: Path,
    reembedded_path: Path,
    *,
    start_indexed_count: int,
    max_db_growth: int,
    phase: str,
) -> None:
    if max_db_growth < 0:
        return
    del faiss_path, reembedded_path
    current_indexed_count = _live_memory_count(db_path)
    db_growth = current_indexed_count - start_indexed_count
    if db_growth > max_db_growth:
        raise SystemExit(
            "Episodic DB advanced by "
            f"{db_growth:,} indexed memory row(s) during repair "
            f"(limit {max_db_growth:,}, phase={phase}); refusing to swap a stale FAISS snapshot. "
            "Pause writers or rerun with --max-db-growth set to an accepted bound."
        )


def diagnose(
    db_path: Path = DEFAULT_DB_PATH,
    faiss_path: Path = DEFAULT_FAISS_PATH,
    reembedded_path: Path = DEFAULT_REEMBEDDED_PATH,
    id_map_path: Path = DEFAULT_ID_MAP_PATH,
    *,
    use_lock: bool = True,
    lock_path: Path | None = None,
) -> HealthReport:
    if use_lock:
        with _exclusive_file_lock(lock_path or faiss_path.parent / ".episodic_faiss.lock"):
            return diagnose(
                db_path,
                faiss_path,
                reembedded_path,
                id_map_path,
                use_lock=False,
            )

    # ── Count indexed memories in live db ──
    if not db_path.exists():
        logger.warning("No episodic db at %s — store is empty, no repair needed", db_path)
        return HealthReport(0, 0, 0, 1.0, 1.0, True, 0)
    conn = sqlite3.connect(str(db_path))
    n_db_routing = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE action_type='routing'"
    ).fetchone()[0]
    n_db_indexed = conn.execute(
        f"SELECT COUNT(*) FROM memories WHERE {indexed_memory_predicate()}",
    ).fetchone()[0]
    live_indexed_ids: set[str] = set()
    if n_db_indexed > 0:
        live_indexed_ids = {
            str(r[0])
            for r in conn.execute(
                f"SELECT id FROM memories WHERE {indexed_memory_predicate()}",
            ).fetchall()
        }
    conn.close()

    # ── Count FAISS vectors ──
    n_faiss_vectors = 0
    if faiss_path.exists():
        try:
            import faiss
            idx = faiss.read_index(str(faiss_path))
            n_faiss_vectors = idx.ntotal
        except Exception as e:
            logger.warning("Cannot read FAISS index at %s: %s", faiss_path, e)
            n_faiss_vectors = 0
    else:
        logger.warning("No FAISS index at %s", faiss_path)

    # ── Inspect id_map.npy ──
    # FAISS only stores vectors; retrieval depends on id_map.npy to translate
    # vector positions back to live SQLite memory ids. A high ntotal with a
    # stale id_map silently degrades retrieval, so diagnose it explicitly.
    n_id_map = 0
    id_map_overlap_live = 1.0 if not live_indexed_ids else 0.0
    missing_id_count = len(live_indexed_ids)
    stale_id_count = 0
    if id_map_path.exists():
        try:
            id_map_arr = np.load(id_map_path, allow_pickle=True)
            id_map_ids = {str(item) for item in id_map_arr.tolist()}
            n_id_map = len(id_map_arr)
            if live_indexed_ids:
                id_map_overlap_live = len(id_map_ids & live_indexed_ids) / max(
                    len(live_indexed_ids), 1
                )
                missing_id_count = len(live_indexed_ids - id_map_ids)
                stale_id_count = len(id_map_ids - live_indexed_ids) + (n_id_map - len(id_map_ids))
            else:
                missing_id_count = 0
                stale_id_count = len(id_map_ids) + (n_id_map - len(id_map_ids))
        except Exception as e:
            logger.warning("Cannot read id_map.npy at %s: %s", id_map_path, e)
    else:
        logger.warning("No id_map.npy at %s", id_map_path)
    id_map_matches_faiss = n_id_map == n_faiss_vectors

    # ── Inspect reembedded.npz ──
    n_reembedded = 0
    overlap_live = 0.0
    reembedded_missing_count = len(live_indexed_ids)
    reembedded_stale_count = 0
    if reembedded_path.exists():
        try:
            d = np.load(reembedded_path, allow_pickle=True)
            if "ids" in d.files:
                reembedded_list = [str(item) for item in d["ids"].tolist()]
                reembedded_ids = set(reembedded_list)
                n_reembedded = len(reembedded_list)
                if live_indexed_ids:
                    overlap = len(reembedded_ids & live_indexed_ids)
                    overlap_live = overlap / max(len(live_indexed_ids), 1)
                    reembedded_missing_count = len(live_indexed_ids - reembedded_ids)
                    reembedded_stale_count = (
                        len(reembedded_ids - live_indexed_ids)
                        + (n_reembedded - len(reembedded_ids))
                    )
                else:
                    overlap_live = 1.0 if n_reembedded == 0 else 0.0
                    reembedded_missing_count = 0
                    reembedded_stale_count = len(reembedded_ids) + (n_reembedded - len(reembedded_ids))
        except Exception as e:
            logger.warning("Cannot read reembedded.npz at %s: %s", reembedded_path, e)
    else:
        logger.warning("No reembedded.npz at %s", reembedded_path)
        reembedded_missing_count = 0

    if n_db_indexed:
        faiss_coverage = n_faiss_vectors / max(n_db_indexed, 1)
    else:
        faiss_coverage = 1.0 if n_faiss_vectors == 0 else 0.0

    healthy = (
        (faiss_coverage >= HEALTH_THRESHOLD)
        and id_map_matches_faiss
        and (id_map_overlap_live >= HEALTH_THRESHOLD)
        and missing_id_count == 0
        and stale_id_count == 0
    )
    faiss_orphan_count = max(0, n_db_indexed - n_faiss_vectors)
    id_map_orphan_count = missing_id_count if n_db_indexed else stale_id_count
    orphan_count = max(
        faiss_orphan_count,
        id_map_orphan_count,
        stale_id_count,
    )

    return HealthReport(
        n_db_routing=n_db_routing,
        n_faiss_vectors=n_faiss_vectors,
        n_reembedded=n_reembedded,
        overlap_live=overlap_live,
        faiss_coverage=faiss_coverage,
        healthy=healthy,
        orphan_count=orphan_count,
        n_id_map=n_id_map,
        id_map_overlap_live=id_map_overlap_live,
        id_map_matches_faiss=id_map_matches_faiss,
        n_db_indexed=n_db_indexed,
        missing_id_count=missing_id_count,
        stale_id_count=stale_id_count,
        reembedded_missing_count=reembedded_missing_count if n_reembedded else 0,
        reembedded_stale_count=reembedded_stale_count,
    )


def print_report(report: HealthReport) -> None:
    print("\n" + "=" * 72)
    print("Episodic Embedding Health Report")
    print("=" * 72)
    print(f"  Routing memories in db:      {report.n_db_routing:>10,}")
    print(f"  Indexed memories in db:      {report.n_db_indexed:>10,}")
    print(f"  Vectors in FAISS index:      {report.n_faiss_vectors:>10,}")
    print(f"  IDs in id_map.npy:           {report.n_id_map:>10,}")
    print(f"  IDs in reembedded.npz:       {report.n_reembedded:>10,}")
    print(f"  FAISS coverage:              {report.faiss_coverage:>10.1%}  (threshold ≥ {HEALTH_THRESHOLD:.0%})")
    print(f"  id_map matches FAISS:        {str(report.id_map_matches_faiss):>10}")
    print(f"  id_map ⋂ live db:            {report.id_map_overlap_live:>10.1%}  (threshold ≥ {HEALTH_THRESHOLD:.0%})")
    print(f"  id_map missing live IDs:     {report.missing_id_count:>10,}")
    print(f"  id_map stale/duplicate IDs:  {report.stale_id_count:>10,}")
    print(f"  Live repairable lag/stale:   {report.orphan_count:>10,}")
    print(f"  Live status:                 {'HEALTHY' if report.healthy else 'ORPHANED — repair recommended'}")
    print("  Training artifact diagnostic (non-blocking):")
    print(f"    reembedded ⋂ live db:      {report.overlap_live:>10.1%}")
    print(f"    missing live IDs:          {report.reembedded_missing_count:>10,}")
    print(f"    stale/duplicates:          {report.reembedded_stale_count:>10,}")
    print("=" * 72)


def _load_npy_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    arr = np.load(path, allow_pickle=True)
    return [str(item) for item in arr.tolist()]


def _load_reembedded_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    if "ids" not in data.files:
        return []
    return [str(item) for item in data["ids"].tolist()]


def _live_memory_ids(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    with sqlite3.connect(str(db_path)) as conn:
        return {
            str(row[0])
            for row in conn.execute(
                f"SELECT id FROM memories WHERE {indexed_memory_predicate()}",
            ).fetchall()
        }


def _live_memory_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    with sqlite3.connect(str(db_path)) as conn:
        return int(
            conn.execute(
                f"SELECT COUNT(*) FROM memories WHERE {indexed_memory_predicate()}",
            ).fetchone()[0]
        )


def _write_ids_file(path: Path, ids: Sequence[str]) -> None:
    path.write_text("\n".join(ids) + "\n")


def _invoke_reembed(
    *,
    db_path: Path,
    output_path: Path,
    servers: int,
    batch_size: int,
    base_port: int,
    only_ids_file: Path | None = None,
) -> None:
    if not REEMBED_SCRIPT.exists():
        raise SystemExit(f"reembed_episodic_store.py not found at {REEMBED_SCRIPT}")
    cmd = [
        sys.executable,
        str(REEMBED_SCRIPT),
        "--db",
        str(db_path),
        "--output",
        str(output_path),
        "--base-port",
        str(base_port),
        "--servers",
        str(servers),
        "--batch-size",
        str(batch_size),
    ]
    if only_ids_file is not None:
        cmd.extend(["--only-ids-file", str(only_ids_file)])
    logger.info("Invoking reembed_episodic_store.py: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"reembed_episodic_store.py failed (rc={rc})")
    logger.info("reembed_episodic_store.py completed OK")


def _write_faiss_and_id_map_atomic(
    *,
    index,
    ids: Sequence[str],
    faiss_path: Path,
    id_map_path: Path,
) -> tuple[Path, Path]:
    import faiss

    ts = int(time.time())
    faiss_backup = faiss_path.with_suffix(f".pre-repair-{ts}")
    id_map_backup = id_map_path.with_suffix(f".pre-repair-{ts}")
    if faiss_path.exists():
        shutil.copy2(faiss_path, faiss_backup)
        logger.info("Backed up old FAISS index → %s (%d bytes)", faiss_backup, faiss_backup.stat().st_size)
    if id_map_path.exists():
        shutil.copy2(id_map_path, id_map_backup)
        logger.info("Backed up old id_map → %s (%d bytes)", id_map_backup, id_map_backup.stat().st_size)

    faiss_new = faiss_path.with_name(faiss_path.name + ".new")
    id_map_new = id_map_path.with_name(id_map_path.name + ".new")
    faiss.write_index(index, str(faiss_new))
    with open(id_map_new, "wb") as f:
        np.save(f, np.array(list(ids), dtype=object), allow_pickle=True)
    if not faiss_new.exists():
        raise SystemExit(f"FAISS .new file did not materialize at {faiss_new}")
    if not id_map_new.exists():
        raise SystemExit(f"id_map .new file did not materialize at {id_map_new}")
    faiss_new.rename(faiss_path)
    id_map_new.rename(id_map_path)
    return faiss_backup, id_map_backup


def append_missing_faiss_vectors(
    *,
    incremental_path: Path,
    ids_to_append: set[str],
    faiss_path: Path,
    id_map_path: Path,
    lock_path: Path = DEFAULT_FAISS_LOCK_PATH,
    pre_swap_check: Callable[[], None] | None = None,
) -> int:
    """Append missing vectors to a structurally consistent FAISS/id_map mirror."""
    if not ids_to_append:
        logger.info("No FAISS/id_map IDs missing; append step is a no-op")
        return 0

    import faiss

    data = np.load(incremental_path, allow_pickle=True)
    incremental_ids = [str(item) for item in data["ids"].tolist()]
    positions = [idx for idx, memory_id in enumerate(incremental_ids) if memory_id in ids_to_append]
    found_ids = {incremental_ids[idx] for idx in positions}
    missing = ids_to_append - found_ids
    if missing:
        raise SystemExit(
            f"Incremental embedding output is missing {len(missing)} FAISS append ID(s); "
            f"first missing: {sorted(missing)[:5]}"
        )

    embs = data["embeddings"][positions].astype(np.float32)
    if embs.ndim == 3:
        embs = embs.squeeze(axis=1)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    append_ids = [incremental_ids[idx] for idx in positions]

    with _exclusive_file_lock(lock_path):
        if pre_swap_check is not None:
            pre_swap_check()
        current_ids = _load_npy_ids(id_map_path)
        current_id_set = set(current_ids)
        filtered = [
            (memory_id, emb)
            for memory_id, emb in zip(append_ids, embs / norms, strict=True)
            if memory_id not in current_id_set
        ]
        if not filtered:
            logger.info("All requested FAISS append IDs are already present")
            return 0

        index = faiss.read_index(str(faiss_path))
        if index.ntotal != len(current_ids):
            raise SystemExit(
                f"Cannot incrementally repair inconsistent mirror: FAISS ntotal={index.ntotal}, "
                f"id_map ids={len(current_ids)}"
            )
        final_append_ids = [memory_id for memory_id, _ in filtered]
        final_embs = np.stack([emb for _, emb in filtered]).astype(np.float32)
        index.add(final_embs)
        final_ids = [*current_ids, *final_append_ids]
        faiss_backup, id_map_backup = _write_faiss_and_id_map_atomic(
            index=index,
            ids=final_ids,
            faiss_path=faiss_path,
            id_map_path=id_map_path,
        )
        verify = faiss.read_index(str(faiss_path))
        if verify.ntotal != len(final_ids):
            raise SystemExit(
                f"FAISS append validation failed: id_map={len(final_ids)}, ntotal={verify.ntotal}"
            )
        logger.info(
            "Incremental FAISS repair appended %d vectors. Backups: %s, %s",
            len(final_append_ids),
            faiss_backup.name,
            id_map_backup.name,
        )
        return len(final_append_ids)


def merge_reembedded_npz(
    *,
    reembedded_path: Path,
    incremental_path: Path,
) -> int:
    """Merge incremental embeddings into reembedded.npz for future rebuilds."""
    inc = np.load(incremental_path, allow_pickle=True)
    inc_ids = [str(item) for item in inc["ids"].tolist()]
    if not reembedded_path.exists():
        tmp = reembedded_path.with_name(reembedded_path.name + ".new")
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                ids=inc["ids"],
                embeddings=inc["embeddings"],
                actions=inc["actions"],
                q_values=inc["q_values"],
                contexts=inc["contexts"],
            )
        tmp.rename(reembedded_path)
        return len(inc_ids)

    old = np.load(reembedded_path, allow_pickle=True)
    old_ids = [str(item) for item in old["ids"].tolist()]
    old_id_set = set(old_ids)
    append_positions = [idx for idx, memory_id in enumerate(inc_ids) if memory_id not in old_id_set]
    if not append_positions:
        logger.info("reembedded.npz already contains all incremental IDs")
        return 0

    tmp = reembedded_path.with_name(reembedded_path.name + ".new")
    backup = reembedded_path.with_suffix(f".pre-repair-{int(time.time())}")
    shutil.copy2(reembedded_path, backup)
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            ids=np.concatenate([old["ids"], inc["ids"][append_positions]]),
            embeddings=np.concatenate([old["embeddings"], inc["embeddings"][append_positions]]),
            actions=np.concatenate([old["actions"], inc["actions"][append_positions]]),
            q_values=np.concatenate([old["q_values"], inc["q_values"][append_positions]]),
            contexts=np.concatenate([old["contexts"], inc["contexts"][append_positions]]),
        )
    tmp.rename(reembedded_path)
    logger.info(
        "Merged %d embeddings into reembedded.npz (backup: %s)",
        len(append_positions),
        backup.name,
    )
    return len(append_positions)


def rebuild_faiss(
    reembedded_path: Path,
    faiss_path: Path,
    id_map_path: Path,
    dim: int = 1024,
    *,
    lock_path: Path = DEFAULT_FAISS_LOCK_PATH,
    pre_swap_check: Callable[[], None] | None = None,
) -> tuple[int, Path, Path]:
    """Build a fresh FAISS IndexFlatIP from reembedded.npz and write atomically.

    Returns (n_vectors_written, faiss_backup_path, id_map_backup_path).
    """
    import faiss

    logger.info("Loading reembedded NPZ from %s", reembedded_path)
    d = np.load(reembedded_path, allow_pickle=True)
    ids = d["ids"]
    embs = d["embeddings"].astype(np.float32)
    if embs.ndim == 3:
        embs = embs.squeeze(axis=1)  # (N, 1, D) -> (N, D)
    n_vec, emb_dim = embs.shape
    if emb_dim != dim:
        raise SystemExit(f"Embedding dim mismatch: NPZ has {emb_dim}, expected {dim}")

    logger.info("Building IndexFlatIP for %d vectors (dim=%d)", n_vec, dim)
    new_index = faiss.IndexFlatIP(dim)
    # L2-normalize for cosine-similarity-via-IP (matches FAISSEmbeddingStore.add)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_index.add(embs / norms)

    with _exclusive_file_lock(lock_path):
        if pre_swap_check is not None:
            pre_swap_check()

        # Backup originals (if they exist)
        ts = int(time.time())
        faiss_backup = faiss_path.with_suffix(f".pre-repair-{ts}")
        id_map_backup = id_map_path.with_suffix(f".pre-repair-{ts}")
        if faiss_path.exists():
            shutil.copy2(faiss_path, faiss_backup)
            logger.info("Backed up old FAISS index → %s (%d bytes)",
                        faiss_backup, faiss_backup.stat().st_size)
        if id_map_path.exists():
            shutil.copy2(id_map_path, id_map_backup)
            logger.info("Backed up old id_map → %s (%d bytes)",
                        id_map_backup, id_map_backup.stat().st_size)

        # Write new files (atomic: write to .new then rename).
        # NOTE: np.save() auto-appends .npy if the path doesn't already end in .npy.
        # We therefore use an explicit ".new.npy" temp name and avoid the prior
        # bug where Path.with_suffix(".new") produced "id_map.new" but np.save
        # actually wrote to "id_map.new.npy", causing the subsequent rename to
        # fail silently (and leaving id_map.npy stale despite a successful FAISS
        # rebuild). See learned-routing-controller.md Phase 6 A3 follow-up.
        faiss_new = faiss_path.with_name(faiss_path.name + ".new")
        id_map_new = id_map_path.with_name(id_map_path.name + ".new")
        faiss.write_index(new_index, str(faiss_new))
        np.save(str(id_map_new), np.array(ids, dtype=object), allow_pickle=True)
        # Validate both .new files actually landed before renaming, so we fail
        # loudly if np.save changed extension behavior in a future numpy version.
        if not faiss_new.exists():
            raise SystemExit(f"FAISS .new file did not materialize at {faiss_new}")
        if not id_map_new.exists():
            # np.save may have appended .npy despite our explicit name — find it.
            alt = id_map_new.with_name(id_map_new.name + ".npy")
            if alt.exists():
                alt.rename(id_map_new)
            else:
                raise SystemExit(f"id_map .new file did not materialize at {id_map_new} or {alt}")
        faiss_new.rename(faiss_path)
        id_map_new.rename(id_map_path)
        logger.info("Wrote fresh FAISS to %s (%d vectors)", faiss_path, new_index.ntotal)
        logger.info("Wrote fresh id_map to %s (%d ids)", id_map_path, len(ids))

        # Re-validate while still holding the mutation lock so live writers cannot
        # advance id_map between the swap and the consistency check.
        verify = faiss.read_index(str(faiss_path))
        if verify.ntotal != n_vec:
            raise SystemExit(
                f"FAISS re-validation failed: wrote {n_vec}, read back {verify.ntotal}"
            )
        logger.info("FAISS re-validation OK (ntotal=%d)", verify.ntotal)

    return n_vec, faiss_backup, id_map_backup


def run_repair(
    db_path: Path,
    faiss_path: Path,
    id_map_path: Path,
    reembedded_path: Path,
    servers: int = DEFAULT_EMBEDDER_SERVERS,
    batch_size: int = 128,
    base_port: int = DEFAULT_EMBEDDER_BASE_PORT,
    skip_reembed: bool = False,
    max_db_growth: int = DEFAULT_MAX_DB_GROWTH,
    incremental: bool = True,
) -> int:
    start_report = diagnose(db_path, faiss_path, reembedded_path)
    start_indexed_count = start_report.n_db_indexed or start_report.n_db_routing
    if incremental and not skip_reembed and faiss_path.exists() and id_map_path.exists():
        if start_report.id_map_matches_faiss:
            live_indexed_ids = _live_memory_ids(db_path)
            id_map_ids = set(_load_npy_ids(id_map_path))
            reembedded_ids = set(_load_reembedded_ids(reembedded_path))
            ids_missing_faiss = live_indexed_ids - id_map_ids
            ids_missing_reembedded = live_indexed_ids - reembedded_ids
            ids_to_embed = sorted(ids_missing_faiss | ids_missing_reembedded)
            if ids_to_embed:
                ts = int(time.time())
                ids_file = reembedded_path.with_name(f"reembedded.incremental-{ts}.ids")
                incremental_path = reembedded_path.with_name(f"reembedded.incremental-{ts}.npz")
                _write_ids_file(ids_file, ids_to_embed)
                logger.info(
                    "Incremental repair selected %d ID(s): %d missing FAISS/id_map, "
                    "%d missing reembedded.npz",
                    len(ids_to_embed),
                    len(ids_missing_faiss),
                    len(ids_missing_reembedded),
                )
                _invoke_reembed(
                    db_path=db_path,
                    output_path=incremental_path,
                    servers=servers,
                    batch_size=batch_size,
                    base_port=base_port,
                    only_ids_file=ids_file,
                )
                _assert_db_growth_within_bound(
                    db_path,
                    faiss_path,
                    reembedded_path,
                    start_indexed_count=start_indexed_count,
                    max_db_growth=max_db_growth,
                    phase="post-incremental-reembed",
                )
                appended = append_missing_faiss_vectors(
                    incremental_path=incremental_path,
                    ids_to_append=ids_missing_faiss,
                    faiss_path=faiss_path,
                    id_map_path=id_map_path,
                    pre_swap_check=lambda: _assert_db_growth_within_bound(
                        db_path,
                        faiss_path,
                        reembedded_path,
                        start_indexed_count=start_indexed_count,
                        max_db_growth=max_db_growth,
                        phase="pre-incremental-swap",
                    ),
                )
                merged = merge_reembedded_npz(
                    reembedded_path=reembedded_path,
                    incremental_path=incremental_path,
                )
                logger.info(
                    "Incremental repair complete: appended %d FAISS vector(s), "
                    "merged %d reembedded row(s)",
                    appended,
                    merged,
                )
                return appended
            logger.info("Incremental repair found no missing IDs; falling through to verification")
            return 0
        logger.warning(
            "Cannot use incremental repair because id_map does not match FAISS; "
            "falling back to full rebuild."
        )

    if not skip_reembed:
        _invoke_reembed(
            db_path=db_path,
            output_path=reembedded_path,
            servers=servers,
            batch_size=batch_size,
            base_port=base_port,
        )

    _assert_db_growth_within_bound(
        db_path,
        faiss_path,
        reembedded_path,
        start_indexed_count=start_indexed_count,
        max_db_growth=max_db_growth,
        phase="post-reembed",
    )

    n_written, faiss_bk, id_map_bk = rebuild_faiss(
        reembedded_path=reembedded_path,
        faiss_path=faiss_path,
        id_map_path=id_map_path,
        pre_swap_check=lambda: _assert_db_growth_within_bound(
            db_path,
            faiss_path,
            reembedded_path,
            start_indexed_count=start_indexed_count,
            max_db_growth=max_db_growth,
            phase="pre-swap",
        ),
    )
    logger.info(
        "Repair complete. Wrote %d FAISS vectors. Backups: %s, %s",
        n_written, faiss_bk.name, id_map_bk.name,
    )
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose / repair orphan episodic embeddings (A3)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--diagnose-only", action="store_true",
        help="Print health report and exit. Exit 0 if healthy, 1 if orphans detected.",
    )
    group.add_argument(
        "--repair", action="store_true",
        help="Diagnose + repair (re-embed live db, rebuild FAISS).",
    )
    parser.add_argument(
        "--skip-reembed", action="store_true",
        help="With --repair: skip BGE re-embed, only rebuild FAISS from existing reembedded.npz",
    )
    parser.add_argument(
        "--full-rebuild", action="store_true",
        help="With --repair: rebuild the full FAISS/id_map mirror instead of appending missing IDs.",
    )
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--faiss", type=str, default=str(DEFAULT_FAISS_PATH))
    parser.add_argument("--id-map", type=str, default=str(DEFAULT_ID_MAP_PATH))
    parser.add_argument("--reembedded", type=str, default=str(DEFAULT_REEMBEDDED_PATH))
    parser.add_argument("--servers", type=int, default=DEFAULT_EMBEDDER_SERVERS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-port", type=int, default=DEFAULT_EMBEDDER_BASE_PORT)
    parser.add_argument(
        "--max-db-growth",
        type=int,
        default=DEFAULT_MAX_DB_GROWTH,
        help=(
            "Refuse the final FAISS swap if live indexed-memory rows grow by more "
            "than this during re-embed; negative disables the guard "
            "(default %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-orphans", type=int, default=MIN_ORPHANS_TO_REPAIR,
        help="Skip repair if orphan count is below this threshold (default %(default)s)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    faiss_path = Path(args.faiss)
    id_map_path = Path(args.id_map)
    reembedded_path = Path(args.reembedded)

    report = diagnose(db_path, faiss_path, reembedded_path, id_map_path)
    print_report(report)

    if args.diagnose_only:
        return 0 if report.healthy else 1

    # --repair path
    if report.healthy and report.orphan_count == 0:
        print("\nStore is healthy — no repair needed. Exiting.")
        return 0
    if report.orphan_count < args.min_orphans:
        print(
            f"\nOrphan count {report.orphan_count:,} below threshold {args.min_orphans:,} — "
            "skipping repair. Run with --min-orphans 0 to force.",
        )
        return 0

    print(f"\nProceeding with repair ({report.orphan_count:,} orphans).")
    print(f"This will use {args.servers} BGE servers and may take several minutes.\n")
    run_repair(
        db_path=db_path,
        faiss_path=faiss_path,
        id_map_path=id_map_path,
        reembedded_path=reembedded_path,
        servers=args.servers,
        batch_size=args.batch_size,
        base_port=args.base_port,
        skip_reembed=args.skip_reembed,
        max_db_growth=args.max_db_growth,
        incremental=not args.full_rebuild,
    )
    print("\nRepair complete. Re-running diagnostic to verify:")
    report2 = diagnose(db_path, faiss_path, reembedded_path, id_map_path)
    print_report(report2)
    return 0 if report2.healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
