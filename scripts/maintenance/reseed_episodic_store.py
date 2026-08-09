#!/usr/bin/env python3
"""reseed_episodic_store.py — rebuild the episodic store on the fixed contract.

STAGED, NOT FIRED. ``--apply`` requires ``--i-understand-this-re-embeds``, and
re-embedding is INFERENCE (BGE), so it needs operator approval and a window.

WHY A RESEED IS NECESSARY
-------------------------
The 2026-07-27 audit chain established, in order:

1. ``faiss_store.save()`` published the index before the id_map with two separate
   renames. A crash left the index ahead; ``_load()``'s truncation was a silent
   no-op in that direction; ``add()`` then returned ``index.ntotal`` while
   appending at ``len(id_map)``. Permanent, cumulative drift. FIXED in code.
2. ``memories.embedding_idx`` was wrong for 57,721 of 57,960 rows. REPAIRED
   exactly by ``repair_faiss_id_map.py`` (id_map is a list of ids, so a row's
   true position is a reverse lookup, not an offset model).
3. But the mapping ``id_map position -> vector`` is ALSO misaligned, by a
   region-dependent offset, and that is NOT repairable by pointer arithmetic.
   Measured: re-embedding a row's own text and comparing to its stored vector
   gives mean cosine **0.5505** (0/12 above 0.9) — random-pair territory. The
   stored vectors do not belong to their rows.

The BGE server itself is FINE: 16 concurrent embeddings of one text agree at
pairwise cosine 1.0000. (An earlier "non-determinism" claim was an artifact of
hashing float32 bytes — sub-ULP jitter breaks a byte hash but not a cosine.
Compare embeddings with cosine, never with a hash.)

So the vectors must be recomputed from text. Since the schema is changing
anyway — full untruncated objective, the work, telemetry out of the semantic
index, one embedding convention — the reseed is the natural place to do it.

WHAT THIS DOES AND DOES NOT RECOVER
-----------------------------------
RECOVERS: correct vectors, correct alignment, one field name for the task text
(the store currently splits 30,571 ``objective`` / 27,562 ``task_description``),
telemetry moved out of the record body into ``metrics``, ``update_count``
seeded, contract-stamped contexts.

DOES NOT RECOVER: the work. Answers, tool calls, REPL steps and reasoning were
never written, so they cannot be reseeded — and the historical objectives were
truncated to 200 chars AT WRITE TIME, so the full text is gone too. A reseed
therefore produces a correct index of 200-char stubs. Trajectories only arrive
via new writes through the fixed path. Do not approve this expecting it to
deliver the trajectory store; it delivers the clean baseline underneath it.

Usage:
    python scripts/maintenance/reseed_episodic_store.py --dry-run
    python scripts/maintenance/reseed_episodic_store.py --apply \
        --i-understand-this-re-embeds
"""

from __future__ import annotations

import asyncio
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
# Six four-slot BGE servers complete 24-way balanced fan-out in ~1.3s on this
# host; larger waves overload their request queues.  Use the certified fleet
# width so the full rebuild does not hold its SQLite transaction for hours.
BATCH = 24


class ReseedVerificationError(RuntimeError):
    """The published artifacts do not satisfy the reseed contract."""


def log(msg: str) -> None:
    print(msg, flush=True)


@contextmanager
def writer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def survey(sessions: Path) -> dict:
    """Classify what is reseedable, without touching anything."""
    from orchestration.repl_memory.memory_record import record_from_legacy_context

    con = sqlite3.connect(f"file:{sessions / 'episodic.db'}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, action, action_type, context, outcome, q_value, update_count FROM memories"
    ).fetchall()
    con.close()

    task = telemetry = unparseable = 0
    already_contract = 0
    objective_chars = []
    for _mid, _a, _at, ctx, _o, _q, _uc in rows:
        try:
            raw = json.loads(ctx)
        except Exception:
            unparseable += 1
            continue
        if isinstance(raw, dict) and "record_version" in raw:
            already_contract += 1
        rec = record_from_legacy_context(raw if isinstance(raw, dict) else {})
        if rec.is_task_memory():
            task += 1
            objective_chars.append(len(rec.objective))
        else:
            telemetry += 1

    at_cap = sum(1 for n in objective_chars if n >= 199)
    return {
        "total_rows": len(rows),
        "task_memories_reseedable": task,
        "telemetry_rows_excluded_from_index": telemetry,
        "unparseable_contexts": unparseable,
        "already_on_contract": already_contract,
        "objective_mean_chars": round(float(np.mean(objective_chars)), 1) if objective_chars else 0,
        "objectives_at_200_char_cap": at_cap,
        "embeddings_required": task,
    }


def _fsync(path: Path) -> None:
    with path.open("rb") as fh:
        import os

        os.fsync(fh.fileno())


def _write_receipt(path: Path, payload: dict) -> None:
    """Atomically publish a durable recovery receipt."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _fsync(tmp)
    tmp.replace(path)


def _backup_sqlite(source: Path, target: Path) -> None:
    """Use SQLite's backup API; copying a WAL database file is not a backup."""
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
        dst.commit()
    finally:
        dst.close()
        src.close()
    _fsync(target)


def _strict_embedder():
    """Return a BGE-server-only embedder; semantic fallback is forbidden here."""
    from orchestration.repl_memory.embedder import EmbeddingConfig, TaskEmbedder
    from orchestration.repl_memory.parallel_embedder import (
        EmbedderPoolConfig,
        ParallelEmbedderClient,
    )

    embedder = TaskEmbedder(
        EmbeddingConfig(
            use_server=True, use_parallel=True, use_fallback=False, allow_subprocess=False
        )
    )
    # TaskEmbedder's normal pool defaults to a hash fallback.  Override it
    # explicitly: a partial reseed must fail, never synthesize vectors.
    embedder._parallel_client = ParallelEmbedderClient(EmbedderPoolConfig(use_fallback=False))
    return embedder


def _checked_batch(embedder, texts: list[str]) -> np.ndarray:
    # TaskEmbedder.embed_batch is a serial Python loop even when its parallel
    # six-server client is configured.  A full 60k-row rebuild would therefore
    # hold the SQLite write transaction for hours.  Use the already-supported
    # async batch fan-out directly; retain the generic method for test doubles.
    parallel = getattr(embedder, "_parallel_client", None)
    if parallel is not None and hasattr(parallel, "_embed_single_server"):
        raw = _balanced_parallel_batch(parallel, texts)
    elif parallel is not None and hasattr(parallel, "embed_batch_sync"):
        raw = parallel.embed_batch_sync(texts)
    else:
        raw = embedder.embed_batch(texts)
    vecs = np.asarray(raw, dtype=np.float32)
    if vecs.shape != (len(texts), 1024):
        raise ReseedVerificationError(
            f"embedding batch shape {vecs.shape}, expected {(len(texts), 1024)}"
        )
    if not np.isfinite(vecs).all():
        raise ReseedVerificationError("embedding batch contains non-finite values")
    norms = np.linalg.norm(vecs, axis=1)
    if np.any(norms <= 0):
        raise ReseedVerificationError("embedding batch contains a zero vector")
    return vecs / norms[:, None]


def _balanced_parallel_batch(parallel, texts: list[str]) -> np.ndarray:
    """Fan a maintenance batch evenly across every configured BGE server."""

    async def run() -> np.ndarray:
        urls = list(parallel.config.server_urls)
        if not urls:
            raise RuntimeError("no embedding servers configured")
        client = await parallel._get_client()

        async def embed_one(index: int, text: str) -> np.ndarray:
            for offset in range(len(urls)):
                url = urls[(index + offset) % len(urls)]
                embedding = await parallel._embed_single_server(client, url, text)
                if embedding is not None:
                    return embedding
            raise RuntimeError(f"all embedding servers failed for batch item {index}")

        try:
            return np.asarray(
                await asyncio.gather(
                    *(embed_one(index, text) for index, text in enumerate(texts))
                ),
                dtype=np.float32,
            )
        finally:
            await parallel.close()

    return asyncio.run(run())


def verify_persisted(sessions: Path, expected_task_ids: set[str]) -> dict:
    """Re-open disk artifacts and prove the complete DB/index bijection."""
    import faiss

    index = faiss.read_index(str(sessions / "embeddings.faiss"))
    id_map = [str(mid) for mid in np.load(sessions / "id_map.npy", allow_pickle=True).tolist()]
    if index.ntotal != len(id_map):
        raise ReseedVerificationError(f"desync={index.ntotal - len(id_map)}")
    if len(id_map) != len(set(id_map)):
        raise ReseedVerificationError("id_map contains duplicate IDs")
    if set(id_map) != expected_task_ids:
        raise ReseedVerificationError("id_map membership differs from reseedable task rows")

    con = sqlite3.connect(f"file:{sessions / 'episodic.db'}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, embedding_idx FROM memories").fetchall()
    finally:
        con.close()
    bad = 0
    for mid, ei in rows:
        if str(mid) in expected_task_ids:
            if (
                ei is None
                or not isinstance(ei, int)
                or ei < 0
                or ei >= len(id_map)
                or id_map[ei] != str(mid)
            ):
                bad += 1
        elif ei is not None:
            bad += 1
    if bad:
        raise ReseedVerificationError(
            f"{bad} task rows do not resolve to themselves or non-task rows are indexed"
        )
    return {"ntotal": index.ntotal, "id_map_len": len(id_map), "desync": 0, "bad": 0}


def reseed(sessions: Path, apply: bool, limit: int | None) -> int:
    from orchestration.repl_memory.memory_record import record_from_legacy_context

    if apply and limit is not None:
        raise ValueError("--limit is unsafe for a live reseed and is disabled")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    s = survey(sessions)
    log("=== SURVEY ===")
    for k, v in s.items():
        log(f"  {k:<38} {v}")
    log(
        f"\n  Rows with no task text under EITHER key are not embedded "
        f"({s['telemetry_rows_excluded_from_index']} found); they are kept in the DB with "
        f"metrics intact. Note the store's two historical key names (objective / "
        f"task_description) both carry real task text — the contract unifies them."
    )
    log(
        f"  {s['objectives_at_200_char_cap']} objectives sit at the historical 200-char cap. "
        "That text is GONE — the reseed cannot restore it."
    )

    if not apply:
        log("\n  DRY RUN — nothing written.")
        log(f"  Would embed {s['embeddings_required']} task memories (INFERENCE).")
        return 0

    import faiss

    with writer_lock(sessions / ".episodic_faiss.lock"):
        receipt = sessions / f"reseed_episodic_store_{stamp}.in-progress.json"
        backups = {
            name: sessions / f"{name}.pre-reseed-{stamp}"
            for name in ("embeddings.faiss", "id_map.npy", "episodic.db")
        }
        _write_receipt(
            receipt,
            {
                "stamp": stamp,
                "state": "started",
                "backups": {k: str(v) for k, v in backups.items()},
            },
        )
        for name in ("embeddings.faiss", "id_map.npy", "episodic.db"):
            src = sessions / name
            if src.exists():
                if name == "episodic.db":
                    _backup_sqlite(src, backups[name])
                else:
                    shutil.copy2(src, backups[name])
                    _fsync(backups[name])
                log(f"[backup] {name} -> {name}.pre-reseed-{stamp}")
        _write_receipt(
            receipt,
            {
                "stamp": stamp,
                "state": "backups_complete",
                "backups": {k: str(v) for k, v in backups.items()},
            },
        )

        con = sqlite3.connect(sessions / "episodic.db")
        con.execute("BEGIN IMMEDIATE")
        rows = con.execute("SELECT id, context FROM memories ORDER BY created_at").fetchall()
        embedder = _strict_embedder()
        index = faiss.IndexFlatIP(1024)
        id_map: list[str] = []
        task_ids: set[str] = set()

        pending: list[tuple[str, str, dict]] = []
        embedded = skipped = 0

        def flush() -> None:
            nonlocal embedded
            if not pending:
                return
            arr = _checked_batch(embedder, [t for _, t, _ in pending])
            index.add(arr)
            for mid, _t, ctx in pending:
                id_map.append(mid)
                con.execute(
                    "UPDATE memories SET embedding_idx = ?, context = ?, "
                    "update_count = COALESCE(update_count, 0) WHERE id = ?",
                    (len(id_map) - 1, json.dumps(ctx), mid),
                )
            embedded += len(pending)
            pending.clear()

        for mid, ctx_raw in rows:
            try:
                raw = json.loads(ctx_raw)
            except Exception:
                raw = {}
            rec = record_from_legacy_context(raw if isinstance(raw, dict) else {})
            new_ctx = rec.to_context()
            if not rec.is_task_memory():
                # Keep the row and its metrics; take it out of the index.
                con.execute(
                    "UPDATE memories SET embedding_idx = NULL, context = ?, "
                    "update_count = COALESCE(update_count, 0) WHERE id = ?",
                    (json.dumps(new_ctx), mid),
                )
                skipped += 1
                continue
            pending.append((mid, rec.embedding_text(), new_ctx))
            task_ids.add(str(mid))
            if len(pending) >= BATCH:
                flush()
                log(f"  embedded {embedded} / {s['embeddings_required']}")
        flush()
        im_tmp = sessions / f".id_map.npy.reseed-{stamp}.tmp"
        ix_tmp = sessions / f".embeddings.faiss.reseed-{stamp}.tmp"
        np.save(str(im_tmp), np.array(id_map, dtype=object), allow_pickle=True)
        if not im_tmp.exists() and im_tmp.with_suffix(".tmp.npy").exists():
            im_tmp.with_suffix(".tmp.npy").rename(im_tmp)
        faiss.write_index(index, str(ix_tmp))
        _fsync(im_tmp)
        _fsync(ix_tmp)
        im_tmp.replace(sessions / "id_map.npy")  # id_map FIRST — recoverable direction
        ix_tmp.replace(sessions / "embeddings.faiss")
        _write_receipt(
            receipt,
            {
                "stamp": stamp,
                "state": "faiss_published_db_uncommitted",
                "backups": {k: str(v) for k, v in backups.items()},
            },
        )
        con.commit()
        con.close()
        _write_receipt(
            receipt,
            {
                "stamp": stamp,
                "state": "published",
                "backups": {k: str(v) for k, v in backups.items()},
            },
        )

        post = verify_persisted(sessions, task_ids)
        receipt.replace(sessions / f"reseed_episodic_store_{stamp}.receipt.json")

    log("\n=== RESEEDED ===")
    log(f"  embedded (in index): {embedded}")
    log(f"  telemetry rows kept, de-indexed: {skipped}")
    log(f"  index/id_map: {post['ntotal']} / {post['id_map_len']}  desync={post['desync']}")
    log(f"  rows whose embedding_idx does not resolve to themselves: {post['bad']}")
    (sessions / f"reseed_episodic_store_{stamp}.json").write_text(
        json.dumps(
            {"survey": s, "embedded": embedded, "deindexed": skipped, "verification": post},
            indent=2,
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions-dir", default=str(SESSIONS))
    ap.add_argument(
        "--limit", type=int, default=None, help="reseed only the first N rows (testing)"
    )
    ap.add_argument(
        "--i-understand-this-re-embeds",
        action="store_true",
        help="required with --apply: re-embedding is inference and needs a window",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.apply and not args.i_understand_this_re_embeds:
        log(
            "REFUSING: --apply re-embeds every task memory through the BGE servers, "
            "which is inference. Re-run with --i-understand-this-re-embeds once a "
            "window is available."
        )
        return 2
    return reseed(Path(args.sessions_dir), apply=args.apply, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
