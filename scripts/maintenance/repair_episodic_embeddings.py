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
    n_faiss_vectors    = embeddings.faiss IndexFlatIP ntotal (read-only check)
    n_id_map           = number of IDs in id_map.npy
    n_reembedded_npz   = number of IDs in reembedded.npz (if present)
    id_map_overlap     = |id_map_ids ∩ live_db_routing_ids| / n_db_routing
    overlap_live       = |reembedded_ids ∩ live_db_routing_ids| / n_db_routing

A store is "orphaned" if n_faiss_vectors / n_db_routing < 0.5, id_map.npy
does not match embeddings.faiss, id_map_overlap < 0.5, OR overlap_live < 0.5.
These conditions fire if FAISS was reset, id_map.npy is stale/wrong, or
reembedded.npz is stale relative to live db.

Usage:
    python3 scripts/maintenance/repair_episodic_embeddings.py --diagnose-only
    python3 scripts/maintenance/repair_episodic_embeddings.py --repair [--servers 6] [--batch-size 128]

The repair step:
    1. Calls reembed_episodic_store.py to produce a fresh reembedded.npz
       (uses existing parallel-BGE primitive — same configured-server pattern).
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
from typing import Callable, NamedTuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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

HEALTH_THRESHOLD = 0.5  # n_faiss / n_db must be ≥ this to be considered healthy
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
    start_routing_count: int,
    max_db_growth: int,
    phase: str,
) -> None:
    if max_db_growth < 0:
        return
    current_routing_count = diagnose(
        db_path,
        faiss_path,
        reembedded_path,
        use_lock=False,
    ).n_db_routing
    db_growth = current_routing_count - start_routing_count
    if db_growth > max_db_growth:
        raise SystemExit(
            "Episodic DB advanced by "
            f"{db_growth:,} routing row(s) during repair "
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

    # ── Count routing memories in live db ──
    if not db_path.exists():
        logger.warning("No episodic db at %s — store is empty, no repair needed", db_path)
        return HealthReport(0, 0, 0, 1.0, 1.0, True, 0)
    conn = sqlite3.connect(str(db_path))
    n_db_routing = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE action_type='routing'"
    ).fetchone()[0]
    live_routing_ids: set[str] = set()
    if n_db_routing > 0:
        live_routing_ids = {
            r[0] for r in conn.execute(
                "SELECT id FROM memories WHERE action_type='routing'"
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
    id_map_overlap_live = 1.0 if not live_routing_ids else 0.0
    id_map_live_count = 0
    if id_map_path.exists():
        try:
            id_map_arr = np.load(id_map_path, allow_pickle=True)
            id_map_ids = {str(item) for item in id_map_arr.tolist()}
            n_id_map = len(id_map_arr)
            if live_routing_ids:
                id_map_live_count = len(id_map_ids & live_routing_ids)
                id_map_overlap_live = id_map_live_count / max(len(live_routing_ids), 1)
        except Exception as e:
            logger.warning("Cannot read id_map.npy at %s: %s", id_map_path, e)
    else:
        logger.warning("No id_map.npy at %s", id_map_path)
    id_map_matches_faiss = n_id_map == n_faiss_vectors

    # ── Inspect reembedded.npz ──
    n_reembedded = 0
    overlap_live = 0.0
    if reembedded_path.exists():
        try:
            d = np.load(reembedded_path, allow_pickle=True)
            if "ids" in d.files:
                reembedded_ids = set(d["ids"].tolist())
                n_reembedded = len(reembedded_ids)
                if live_routing_ids:
                    overlap = len(reembedded_ids & live_routing_ids)
                    overlap_live = overlap / max(len(live_routing_ids), 1)
        except Exception as e:
            logger.warning("Cannot read reembedded.npz at %s: %s", reembedded_path, e)
    else:
        logger.warning("No reembedded.npz at %s", reembedded_path)

    faiss_coverage = n_faiss_vectors / max(n_db_routing, 1) if n_db_routing else 1.0

    healthy = (
        (faiss_coverage >= HEALTH_THRESHOLD)
        and id_map_matches_faiss
        and (id_map_overlap_live >= HEALTH_THRESHOLD)
        and (n_reembedded == 0 or overlap_live >= HEALTH_THRESHOLD)
    )
    faiss_orphan_count = max(0, n_db_routing - n_faiss_vectors)
    id_map_orphan_count = max(0, n_db_routing - id_map_live_count) if n_db_routing else 0
    orphan_count = max(faiss_orphan_count, id_map_orphan_count)

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
    )


def print_report(report: HealthReport) -> None:
    print("\n" + "=" * 72)
    print("Episodic Embedding Health Report")
    print("=" * 72)
    print(f"  Routing memories in db:      {report.n_db_routing:>10,}")
    print(f"  Vectors in FAISS index:      {report.n_faiss_vectors:>10,}")
    print(f"  IDs in id_map.npy:           {report.n_id_map:>10,}")
    print(f"  IDs in reembedded.npz:       {report.n_reembedded:>10,}")
    print(f"  FAISS coverage:              {report.faiss_coverage:>10.1%}  (threshold ≥ {HEALTH_THRESHOLD:.0%})")
    print(f"  id_map matches FAISS:        {str(report.id_map_matches_faiss):>10}")
    print(f"  id_map ⋂ live db:            {report.id_map_overlap_live:>10.1%}  (threshold ≥ {HEALTH_THRESHOLD:.0%})")
    print(f"  reembedded ⋂ live db:        {report.overlap_live:>10.1%}  (threshold ≥ {HEALTH_THRESHOLD:.0%})")
    print(f"  Orphan count (db − FAISS):   {report.orphan_count:>10,}")
    print(f"  Status:                      {'HEALTHY' if report.healthy else 'ORPHANED — repair recommended'}")
    print("=" * 72)


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
) -> int:
    start_routing_count = diagnose(db_path, faiss_path, reembedded_path).n_db_routing
    if not skip_reembed:
        if not REEMBED_SCRIPT.exists():
            raise SystemExit(f"reembed_episodic_store.py not found at {REEMBED_SCRIPT}")
        cmd = [
            sys.executable, str(REEMBED_SCRIPT),
            "--db", str(db_path),  # reembed_episodic_store expects the .db file path
            "--output", str(reembedded_path),
            "--base-port", str(base_port),
            "--servers", str(servers),
            "--batch-size", str(batch_size),
        ]
        logger.info("Invoking reembed_episodic_store.py: %s", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(f"reembed_episodic_store.py failed (rc={rc})")
        logger.info("reembed_episodic_store.py completed OK")

    _assert_db_growth_within_bound(
        db_path,
        faiss_path,
        reembedded_path,
        start_routing_count=start_routing_count,
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
            start_routing_count=start_routing_count,
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
            "Refuse the final FAISS swap if live routing-memory rows grow by more "
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

    report = diagnose(db_path, faiss_path, reembedded_path)
    print_report(report)

    if args.diagnose_only:
        return 0 if report.healthy else 1

    # --repair path
    if report.healthy:
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
    )
    print("\nRepair complete. Re-running diagnostic to verify:")
    report2 = diagnose(db_path, faiss_path, reembedded_path)
    print_report(report2)
    return 0 if report2.healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
