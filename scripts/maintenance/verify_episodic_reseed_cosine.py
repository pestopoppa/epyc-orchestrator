#!/usr/bin/env python3
"""Read-only cosine acceptance probe for a completed episodic-store reseed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

DEFAULT_SESSIONS = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions")


def _embedding(url: str, text: str) -> np.ndarray:
    import httpx

    response = httpx.post(f"{url}/embedding", json={"content": text}, timeout=15.0)
    response.raise_for_status()
    body = response.json()
    if isinstance(body, list):
        body = body[0]
    values = body.get("embedding") or body["data"][0]["embedding"]
    if values and isinstance(values[0], list):
        values = values[0]
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (1024,) or not np.isfinite(vector).all():
        raise RuntimeError(f"invalid server embedding shape={vector.shape}")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise RuntimeError("server returned zero embedding")
    return vector / norm


def _task_rows(rows: list[tuple]) -> list[tuple]:
    """Return only rows that are expected to have an index entry."""
    from orchestration.repl_memory.memory_record import record_from_legacy_context

    return [
        row
        for row in rows
        if record_from_legacy_context(json.loads(row[2])).is_task_memory()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS)
    ap.add_argument("--server-url", default="http://127.0.0.1:8090")
    ap.add_argument("--sample-size", type=int, default=12)
    args = ap.parse_args()
    if args.sample_size < 1:
        ap.error("--sample-size must be positive")

    import faiss
    from orchestration.repl_memory.memory_record import record_from_legacy_context
    from scripts.maintenance.reseed_episodic_store import verify_persisted

    con = sqlite3.connect(f"file:{args.sessions_dir / 'episodic.db'}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, embedding_idx, context FROM memories ORDER BY id").fetchall()
    finally:
        con.close()
    task_rows = _task_rows(rows)
    expected_ids = {str(mid) for mid, _idx, _ctx in task_rows}
    structural = verify_persisted(args.sessions_dir, expected_ids)
    ranked = sorted(
        task_rows, key=lambda row: (hashlib.sha256(str(row[0]).encode()).digest(), str(row[0]))
    )
    sample = ranked[: min(args.sample_size, len(ranked))]
    index = faiss.read_index(str(args.sessions_dir / "embeddings.faiss"))

    cosines: list[float] = []
    details = []
    for mid, idx, raw_context in sample:
        record = record_from_legacy_context(json.loads(raw_context))
        own = _embedding(args.server_url, record.embedding_text())
        stored = index.reconstruct(int(idx))
        stored = stored / np.linalg.norm(stored)
        cosine = float(np.dot(own, stored))
        cosines.append(cosine)
        details.append({"id": str(mid), "embedding_idx": idx, "cosine": round(cosine, 6)})

    mean = float(np.mean(cosines)) if cosines else 0.0
    report = {
        "structural": structural,
        "sample_size": len(sample),
        "mean_cosine": mean,
        "above_0_9": sum(x > 0.9 for x in cosines),
        "samples": details,
    }
    print(json.dumps(report, indent=2))
    return 0 if len(sample) == args.sample_size and mean > 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
