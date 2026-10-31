#!/usr/bin/env python3
"""Re-embed one memory without shifting any FAISS positions.

Default is an offline dry run. Apply only with API/readers stopped (their cached
index cannot observe an atomic disk replacement). Embedding servers may remain
up. The shared writer lock excludes cooperating writers; a SQLite write lock
also freezes the source context until publication. No models are launched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestration.repl_memory.memory_record import record_from_legacy_context
from scripts.maintenance.repair_episodic_embeddings import _exclusive_file_lock


def strict_embed(text: str) -> np.ndarray:
    from orchestration.repl_memory.embedder import EmbeddingConfig, TaskEmbedder

    return TaskEmbedder(
        EmbeddingConfig(use_fallback=False, use_parallel=False, allow_subprocess=False)
    ).embed_text(text, side="document")


def repair_vector(
    sessions: Path,
    memory_id: str,
    *,
    apply: bool = False,
    embed: Callable[[str], np.ndarray] = strict_embed,
) -> dict:
    import faiss

    index_path = sessions / "embeddings.faiss"
    with _exclusive_file_lock(sessions / ".episodic_faiss.lock"):
        with sqlite3.connect(f"file:{sessions / 'episodic.db'}?mode=rw", uri=True) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT embedding_idx, context FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown memory: {memory_id}")
            index = faiss.read_index(str(index_path))
            if not isinstance(index, faiss.IndexFlatIP):
                raise ValueError("repair requires IndexFlatIP; refusing to change index semantics")
            ids = np.load(sessions / "id_map.npy", allow_pickle=True).tolist()
            if len(ids) != index.ntotal or len(set(ids)) != len(ids):
                raise ValueError("id_map is desynchronized or contains duplicate IDs")
            positions = dict(conn.execute("SELECT id, embedding_idx FROM memories"))
            if any(positions.get(str(mid)) != i for i, mid in enumerate(ids)):
                raise ValueError("cross-store embedding_idx binding mismatch")
            # NULL is the established unindexed-row marker; negative indices
            # have no unset sentinel contract in EpisodicStore and are refused.
            if any(
                idx is not None
                and (
                    not isinstance(idx, int) or not 0 <= idx < len(ids) or str(ids[idx]) != str(mid)
                )
                for mid, idx in positions.items()
            ):
                raise ValueError("reverse cross-store embedding_idx binding mismatch")
            position, raw_context = row
            if (
                not isinstance(position, int)
                or not 0 <= position < len(ids)
                or str(ids[position]) != memory_id
            ):
                raise ValueError("memory embedding_idx does not match id_map")
            context = json.loads(raw_context)
            if not isinstance(context, dict):
                raise ValueError("memory context is not an object")
            text = record_from_legacy_context(context).embedding_text()
            if not text.strip():
                raise ValueError("memory has no embedding text")
            report = {
                "memory_id": memory_id,
                "position": position,
                "ntotal": index.ntotal,
                "applied": False,
            }
            if not apply:
                return report
            vector = np.asarray(embed(text), dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if (
                vector.shape != (index.d,)
                or not np.isfinite(vector).all()
                or not np.isfinite(norm)
                or norm <= 0
            ):
                raise ValueError("embedder returned invalid vector")
            vector = (vector / norm).astype(np.float32)
            if not np.isfinite(vector).all() or not np.isclose(
                np.linalg.norm(vector.astype(np.float64)), 1.0, rtol=1e-6, atol=1e-6
            ):
                raise ValueError("normalized embedding is not a finite unit vector")
            vectors = index.reconstruct_n(0, index.ntotal)
            vectors[position] = vector
            replacement = faiss.IndexFlatIP(index.d)
            replacement.add(vectors)
            backup = index_path.with_name(f"embeddings.faiss.pre-vector-repair-{uuid.uuid4().hex}")
            shutil.copy2(index_path, backup)
            # Publish only the index; id_map and SQLite positions stay byte-identical.
            with tempfile.NamedTemporaryFile(
                dir=sessions, prefix=".vector-repair-", delete=False
            ) as f:
                temporary = Path(f.name)
            try:
                faiss.write_index(replacement, str(temporary))
                verified = faiss.read_index(str(temporary))
                if verified.ntotal != index.ntotal or not np.array_equal(
                    verified.reconstruct(position), vector
                ):
                    raise RuntimeError("replacement index verification failed")
                temporary.replace(index_path)
            finally:
                temporary.unlink(missing_ok=True)
            return {**report, "applied": True, "backup": str(backup)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True, type=Path)
    parser.add_argument("--memory-id", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--readers-stopped",
        action="store_true",
        help="confirm API and other cached-index readers are stopped",
    )
    args = parser.parse_args()
    if args.apply and not args.readers_stopped:
        parser.error("--apply requires --readers-stopped")
    print(
        json.dumps(repair_vector(args.sessions, args.memory_id, apply=args.apply), sort_keys=True)
    )


if __name__ == "__main__":
    main()
