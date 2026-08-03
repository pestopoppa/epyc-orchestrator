#!/usr/bin/env python3
"""backfill_skill_embeddings.py — make the distilled SkillBank actually retrievable.

WHAT IS WRONG
-------------
``SkillBank.store(skill, embedding=None)`` takes the embedding as OPTIONAL and only
assigns ``embedding_idx`` when one is supplied (skill_bank.py:251-255). The
distillation pipeline never supplied one, so all 57 stored skills carry
``embedding_idx = NULL`` and no ``skill_embeddings.faiss`` index exists on disk at
all.

Level-2 retrieval is a FAISS similarity search of the task embedding against skill
embeddings (skill_retriever.py:~118, ``search_by_embedding``). With no embeddings
it can never match, which is exactly what the store shows: **retrieval_count = 0
across all 57 skills**, despite the whole path being wired and live —
``state.hybrid_router`` is replaced by ``SkillAugmentedRouter`` at
services/memrl.py:481 whenever the ``skillbank`` flag is on, and it is on.

So the distillation output has never once reached a routing decision. Not because
the consumer is missing — it is present and running — but because the search key
was never computed.

WHAT THIS DOES
--------------
Embeds each skill by WHEN IT APPLIES, not by what it says. Retrieval matches a
skill against the embedding of the incoming *task*, so the skill's own vector has
to live in task space: ``title`` + ``when_to_apply`` + ``task_types``. Embedding
the ``principle`` instead would place skills in advice space and match poorly.

Uses SkillBank's own FAISS store (``skill_embeddings.faiss`` /
``skill_id_map.npy``), which is SEPARATE from the episodic index — so this is
independent of the episodic reseed and safe to run before it.

Cost: one BGE embedding per skill (57 today). Negligible, but it IS inference, so
``--apply`` is explicit.

Usage:
    python scripts/maintenance/backfill_skill_embeddings.py --dry-run
    python scripts/maintenance/backfill_skill_embeddings.py --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

SESSIONS = _REPO_ROOT / "orchestration/repl_memory/sessions"


def log(msg: str) -> None:
    print(msg, flush=True)


# Canonical convention lives with the store so every writer shares it.
from orchestration.repl_memory.skill_bank import skill_embedding_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions-dir", default=str(SESSIONS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    sessions = Path(args.sessions_dir)
    db = sessions / "skills.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, title, skill_type, when_to_apply, task_types, embedding_idx, "
        "retrieval_count FROM skills"
    ).fetchall()
    con.close()

    missing = [r for r in rows if r[5] is None]
    log("=== SKILLBANK ===")
    log(f"  skills:                {len(rows)}")
    log(f"  missing embedding_idx: {len(missing)}")
    log(f"  total retrieval_count: {sum(r[6] or 0 for r in rows)}")
    by_type: dict[str, int] = {}
    for r in missing:
        by_type[r[2]] = by_type.get(r[2], 0) + 1
    for k, v in sorted(by_type.items()):
        log(f"    {k:<18} {v}")

    if not missing:
        log("\n  Nothing to backfill.")
        return 0

    log("\n  Embedding text is built from title + when_to_apply + task_types")
    log("  (task space, because retrieval matches skills against the TASK embedding).")
    log(f"\n  Example: {skill_embedding_text(missing[0][1], missing[0][3], missing[0][4])[:160]}")

    if not args.dry_run:
        from orchestration.repl_memory.embedder import TaskEmbedder
        from orchestration.repl_memory.skill_bank import SkillBank

        bank = SkillBank(db_path=db, faiss_path=sessions, embedding_dim=1024)
        store = bank._get_embedding_store()
        embedder = TaskEmbedder()

        con = sqlite3.connect(db)
        done = 0
        for sid, title, _st, when, types, _ei, _rc in missing:
            text = skill_embedding_text(title, when, types)
            vec = embedder.embed_text(text)
            idx = store.add(sid, vec)
            con.execute("UPDATE skills SET embedding_idx = ? WHERE id = ?", (idx, sid))
            done += 1
        store.save()
        con.commit()

        remaining = con.execute(
            "SELECT COUNT(*) FROM skills WHERE embedding_idx IS NULL"
        ).fetchone()[0]
        con.close()
        log(f"\n=== BACKFILLED ===")
        log(f"  embedded:                  {done}")
        log(f"  still missing:             {remaining}")
        log(f"  skill index ntotal/id_map: {store.index.ntotal} / {len(store.id_map)}")
        return 1 if remaining else 0

    log(f"\n  DRY RUN — would embed {len(missing)} skills (INFERENCE, ~{len(missing)} BGE calls).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
