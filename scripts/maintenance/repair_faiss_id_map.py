#!/usr/bin/env python3
"""repair_faiss_id_map.py — repair the 2026-07-05 FAISS/id_map desync.

WHAT BROKE
----------
``faiss_store.save()`` published ``embeddings.faiss`` and ``id_map.npy`` with two
separate renames. A crash between them left the index ahead of the id_map.
``_load()``'s "truncate id_map to match index" was a silent no-op in that
direction, and ``add()`` then returned ``index.ntotal`` as the position while
appending the id at ``len(id_map)`` — so every subsequent write inherited the
offset and persisted it into ``memories.embedding_idx``. Permanent, cumulative,
+1 per interrupted publish. Onset 2026-07-05T15:01:12; the live store reached a
drift of 42 and mis-resolved ~30,238 of 54,960 live rows.

The write-path defects are fixed in ``faiss_store.py`` (publish id_map first,
derive positions from ``len(id_map)``, fail closed on desync). This script
repairs the data those defects already corrupted.

WHY THIS IS EXACT, NOT INFERRED
-------------------------------
``id_map`` is a list of memory ids. The true position of a memory is therefore
simply its index in that list — a reverse lookup, not an offset model. No
re-embedding, no inference, no heuristic: the vectors were always correct, only
the pointers were wrong.

WHAT IT DOES
------------
1. Takes the same cross-process lock the writer uses, so it cannot race the API.
2. Backs up ``embeddings.faiss``, ``id_map.npy`` and the DB rows it will touch.
3. Truncates the index to ``len(id_map)`` if the index is ahead (the trailing
   vectors have no id and are unreachable).
4. Rewrites ``memories.embedding_idx`` from the id_map reverse lookup.
5. Verifies: every repaired row must satisfy ``id_map[embedding_idx] == id``.

Usage:
    python scripts/maintenance/repair_faiss_id_map.py --dry-run     # report only
    python scripts/maintenance/repair_faiss_id_map.py --apply
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

SESSIONS = _REPO_ROOT / "orchestration/repl_memory/sessions"


def log(msg: str) -> None:
    print(msg, flush=True)


@contextmanager
def writer_lock(path: Path):
    """The same lock episodic_store.py takes around FAISS mutations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def diagnose(sessions: Path) -> dict:
    import faiss

    index = faiss.read_index(str(sessions / "embeddings.faiss"))
    id_map = np.load(sessions / "id_map.npy", allow_pickle=True).tolist()
    desync = index.ntotal - len(id_map)

    pos = {}
    dupes = 0
    for i, mid in enumerate(id_map):
        key = str(mid)
        if key in pos:
            dupes += 1
        pos[key] = i

    con = sqlite3.connect(f"file:{sessions / 'episodic.db'}?mode=ro", uri=True)
    rows = con.execute("SELECT id, embedding_idx FROM memories").fetchall()
    con.close()

    correct = missing = wrong = 0
    for mid, ei in rows:
        true_pos = pos.get(str(mid))
        if true_pos is None:
            missing += 1
        elif true_pos == ei:
            correct += 1
        else:
            wrong += 1

    return {
        "ntotal": index.ntotal,
        "id_map_len": len(id_map),
        "desync": desync,
        "id_map_duplicate_ids": dupes,
        "db_rows": len(rows),
        "embedding_idx_correct": correct,
        "embedding_idx_wrong": wrong,
        "rows_missing_from_id_map": missing,
    }


def repair(sessions: Path, apply: bool) -> int:
    import faiss

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = diagnose(sessions)
    log("=== DIAGNOSIS ===")
    for k, v in d.items():
        log(f"  {k:<28} {v}")

    if d["rows_missing_from_id_map"]:
        log(
            f"\n  WARNING: {d['rows_missing_from_id_map']} DB rows have no id in id_map. "
            "Their embedding_idx cannot be repaired and will be set NULL so readers "
            "fail closed instead of resolving a wrong vector."
        )

    if d["desync"] == 0 and d["embedding_idx_wrong"] == 0:
        log("\n  Nothing to repair.")
        return 0

    if not apply:
        log("\n  DRY RUN — re-run with --apply to repair.")
        return 0

    with writer_lock(sessions / ".episodic_faiss.lock"):
        # Re-read under the lock; the live API may have written since diagnose().
        index = faiss.read_index(str(sessions / "embeddings.faiss"))
        id_map = np.load(sessions / "id_map.npy", allow_pickle=True).tolist()
        log(f"\n[lock] re-read under lock: ntotal={index.ntotal} id_map={len(id_map)}")

        for name in ("embeddings.faiss", "id_map.npy", "episodic.db"):
            src = sessions / name
            dst = sessions / f"{name}.pre-repair-{stamp}"
            shutil.copy2(src, dst)
            log(f"[backup] {name} -> {dst.name}")

        # 1. Reconcile the pair. id_map is authoritative for identity; trailing
        #    vectors with no id are unreachable and must go.
        keep = min(index.ntotal, len(id_map))
        if index.ntotal > keep:
            log(f"[index] truncating {index.ntotal} -> {keep} vectors (dropping unreachable tail)")
            vecs = index.reconstruct_n(0, keep)
            new_index = faiss.IndexFlatIP(index.d)
            new_index.add(vecs)
            index = new_index
        if len(id_map) > keep:
            log(f"[id_map] truncating {len(id_map)} -> {keep} ids (no vector)")
            id_map = id_map[:keep]

        # 2. Publish id_map FIRST, then the index — the corrected order, so an
        #    interruption here lands in the recoverable direction.
        im_tmp = sessions / f".id_map.npy.repair-{stamp}.tmp"
        ix_tmp = sessions / f".embeddings.faiss.repair-{stamp}.tmp"
        np.save(str(im_tmp), np.array(id_map, dtype=object), allow_pickle=True)
        if not im_tmp.exists() and im_tmp.with_suffix(".tmp.npy").exists():
            im_tmp.with_suffix(".tmp.npy").rename(im_tmp)
        faiss.write_index(index, str(ix_tmp))
        im_tmp.replace(sessions / "id_map.npy")
        ix_tmp.replace(sessions / "embeddings.faiss")
        log(f"[publish] id_map then index, both at {keep}")

        # 3. Rewrite embedding_idx from the reverse lookup — exact.
        pos = {str(mid): i for i, mid in enumerate(id_map)}
        con = sqlite3.connect(sessions / "episodic.db")
        rows = con.execute("SELECT id, embedding_idx FROM memories").fetchall()
        fixed = nulled = unchanged = 0
        for mid, ei in rows:
            true_pos = pos.get(str(mid))
            if true_pos is None:
                if ei is not None:
                    con.execute("UPDATE memories SET embedding_idx = NULL WHERE id = ?", (mid,))
                    nulled += 1
            elif true_pos != ei:
                con.execute(
                    "UPDATE memories SET embedding_idx = ? WHERE id = ?", (true_pos, mid)
                )
                fixed += 1
            else:
                unchanged += 1
        con.commit()
        log(f"[db] embedding_idx: {fixed} fixed, {nulled} nulled, {unchanged} already correct")

        # 4. Verify.
        bad = 0
        for mid, ei in con.execute(
            "SELECT id, embedding_idx FROM memories WHERE embedding_idx IS NOT NULL"
        ):
            if ei >= len(id_map) or str(id_map[ei]) != str(mid):
                bad += 1
        con.close()
        log(f"[verify] rows whose embedding_idx does not resolve to themselves: {bad}")
        if bad:
            log("  REPAIR INCOMPLETE — backups retained.")
            return 1

    log("\n=== REPAIRED ===")
    post = diagnose(sessions)
    for k, v in post.items():
        log(f"  {k:<28} {v}")
    (sessions / f"repair_faiss_id_map_{stamp}.json").write_text(
        json.dumps({"before": d, "after": post, "stamp": stamp}, indent=2)
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-dir", default=str(SESSIONS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    return repair(Path(args.sessions_dir), apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
