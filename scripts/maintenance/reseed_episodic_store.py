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

SESSIONS = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions")
BATCH = 256


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


def reseed(sessions: Path, apply: bool, limit: int | None) -> int:
    from orchestration.repl_memory.embedder import TaskEmbedder
    from orchestration.repl_memory.memory_record import record_from_legacy_context

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
        for name in ("embeddings.faiss", "id_map.npy", "episodic.db"):
            src = sessions / name
            if src.exists():
                shutil.copy2(src, sessions / f"{name}.pre-reseed-{stamp}")
                log(f"[backup] {name} -> {name}.pre-reseed-{stamp}")

        con = sqlite3.connect(sessions / "episodic.db")
        rows = con.execute(
            "SELECT id, context FROM memories ORDER BY created_at"
        ).fetchall()
        if limit:
            rows = rows[:limit]

        embedder = TaskEmbedder()
        index = faiss.IndexFlatIP(1024)
        id_map: list[str] = []

        pending: list[tuple[str, str, dict]] = []
        embedded = skipped = 0

        def flush() -> None:
            nonlocal embedded
            if not pending:
                return
            vecs = embedder.embed_batch([t for _, t, _ in pending])
            arr = np.asarray(vecs, dtype=np.float32)
            faiss.normalize_L2(arr)
            index.add(arr)
            for (mid, _t, ctx) in pending:
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
            if len(pending) >= BATCH:
                flush()
                log(f"  embedded {embedded} / {s['embeddings_required']}")
        flush()
        con.commit()

        im_tmp = sessions / f".id_map.npy.reseed-{stamp}.tmp"
        ix_tmp = sessions / f".embeddings.faiss.reseed-{stamp}.tmp"
        np.save(str(im_tmp), np.array(id_map, dtype=object), allow_pickle=True)
        if not im_tmp.exists() and im_tmp.with_suffix(".tmp.npy").exists():
            im_tmp.with_suffix(".tmp.npy").rename(im_tmp)
        faiss.write_index(index, str(ix_tmp))
        im_tmp.replace(sessions / "id_map.npy")   # id_map FIRST — recoverable direction
        ix_tmp.replace(sessions / "embeddings.faiss")

        bad = 0
        for mid, ei in con.execute(
            "SELECT id, embedding_idx FROM memories WHERE embedding_idx IS NOT NULL"
        ):
            if ei >= len(id_map) or str(id_map[ei]) != str(mid):
                bad += 1
        con.close()

    log(f"\n=== RESEEDED ===")
    log(f"  embedded (in index): {embedded}")
    log(f"  telemetry rows kept, de-indexed: {skipped}")
    log(f"  index/id_map: {index.ntotal} / {len(id_map)}  desync={index.ntotal - len(id_map)}")
    log(f"  rows whose embedding_idx does not resolve to themselves: {bad}")
    (sessions / f"reseed_episodic_store_{stamp}.json").write_text(
        json.dumps({"survey": s, "embedded": embedded, "deindexed": skipped, "bad": bad}, indent=2)
    )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions-dir", default=str(SESSIONS))
    ap.add_argument("--limit", type=int, default=None, help="reseed only the first N rows (testing)")
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
