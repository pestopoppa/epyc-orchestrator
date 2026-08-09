#!/usr/bin/env python3
"""Repair misnamespaced actions and legacy parallel-fallback vectors.

The repair is intentionally narrow and idempotent. It never deletes memories:

* non-serving actions are moved out of the ``routing`` namespace;
* vectors exactly matching the retired parallel embedder fallback are
  re-embedded by a live BGE server and replaced at the same FAISS position;
* a complete preimage backup and an immutable JSON receipt are written before
  and after mutation.

Run only while the orchestrator API is stopped; model and embedding servers may
remain online.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from orchestration.repl_memory.memory_record import (  # noqa: E402
    record_from_legacy_context,
)
from orchestration.repl_memory.parallel_embedder import (  # noqa: E402
    is_parallel_hash_fallback_embedding,
)
from src.registry.stack_priors import live_stack_role_ids  # noqa: E402
from src.roles import Role  # noqa: E402

DEFAULT_SESSIONS = REPO_ROOT / "orchestration/repl_memory/sessions"
EMBEDDER_URLS = tuple(f"http://127.0.0.1:{port}" for port in range(8090, 8096))
SCHEMA = "epyc.episodic_routing_poison_repair.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def classify_namespace(
    action: str | None,
    action_type: str | None,
    context: dict[str, Any] | None,
    live_roles: set[str],
) -> str | None:
    """Return the corrected namespace, or ``None`` when no change is needed."""
    if action_type != "routing":
        return None
    action_text = str(action or "")
    if action_text.startswith("plan_review:"):
        return "plan_review"

    metrics = (context or {}).get("metrics")
    if (
        action_text.startswith("escalate:")
        and isinstance(metrics, dict)
        and metrics.get("action_type") == "escalation"
    ):
        return "escalation"

    if not action_text:
        return "quarantined_invalid_route"
    for component in action_text.split(","):
        role_text = component.strip().split(":", 1)[0]
        role = Role.from_string(role_text)
        if role is None or role.value not in live_roles:
            return "quarantined_invalid_route"
    return None


def _parse_context(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _embedding_text(context: dict[str, Any]) -> str:
    return record_from_legacy_context(context).embedding_text()


def _parse_embedding(payload: Any) -> np.ndarray:
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected embedding payload type: {type(payload).__name__}")
    if "embedding" in payload:
        value = payload["embedding"]
        if value and isinstance(value[0], list):
            value = value[0]
    elif payload.get("data"):
        value = payload["data"][0]["embedding"]
    else:
        raise ValueError("embedding payload has no vector")
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (1024,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"invalid embedding vector shape/content: {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if not norm:
        raise ValueError("embedding vector has zero norm")
    return vector / norm


def _strict_embed(text: str) -> tuple[np.ndarray, str]:
    errors: list[str] = []
    with httpx.Client(timeout=30, headers={"Connection": "close"}) as client:
        for url in EMBEDDER_URLS:
            try:
                response = client.post(f"{url}/embedding", json={"content": text})
                response.raise_for_status()
                return _parse_embedding(response.json()), url
            except Exception as exc:  # noqa: BLE001 - receipt needs each endpoint failure
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"no BGE embedder returned a valid vector: {errors}")


def audit(sessions: Path) -> dict[str, Any]:
    db_path = sessions / "episodic.db"
    index_path = sessions / "embeddings.faiss"
    id_map_path = sessions / "id_map.npy"
    index = faiss.read_index(str(index_path))
    id_map = np.load(id_map_path, allow_pickle=True).tolist()
    if index.ntotal != len(id_map):
        raise RuntimeError(
            f"FAISS/id_map desync: ntotal={index.ntotal}, ids={len(id_map)}"
        )

    live_roles = set(live_stack_role_ids())
    if not live_roles:
        raise RuntimeError("realized-fleet role set is empty")

    namespace_updates: list[dict[str, Any]] = []
    fallback_vectors: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT id, embedding_idx, action, action_type, context "
            "FROM memories ORDER BY embedding_idx"
        ).fetchall()

    for memory_id, embedding_idx, action, action_type, context_raw in rows:
        context = _parse_context(context_raw)
        target = classify_namespace(action, action_type, context, live_roles)
        if target is not None:
            namespace_updates.append(
                {
                    "id": memory_id,
                    "action": action,
                    "from": action_type,
                    "to": target,
                }
            )
        if embedding_idx is None:
            continue
        text = _embedding_text(context)
        vector = index.reconstruct(int(embedding_idx))
        if text and is_parallel_hash_fallback_embedding(text, vector):
            fallback_vectors.append(
                {
                    "id": memory_id,
                    "embedding_idx": int(embedding_idx),
                    "embedding_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )

    return {
        "row_count": len(rows),
        "faiss_ntotal": int(index.ntotal),
        "id_map_count": len(id_map),
        "namespace_updates": namespace_updates,
        "fallback_vectors": fallback_vectors,
        "audit_sha256": _canonical_json_sha(
            {
                "namespace_updates": namespace_updates,
                "fallback_vectors": fallback_vectors,
            }
        ),
    }


def _backup_preimage(sessions: Path, token: str) -> Path:
    backup = sessions / "backups" / token
    backup.mkdir(parents=True, exist_ok=False)
    for name in ("episodic.db", "embeddings.faiss", "id_map.npy"):
        shutil.copy2(sessions / name, backup / name)
    return backup


def _replace_vectors(
    sessions: Path, fallback_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not fallback_rows:
        return []
    index_path = sessions / "embeddings.faiss"
    index = faiss.read_index(str(index_path))
    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)

    repaired: list[dict[str, Any]] = []
    db_path = sessions / "episodic.db"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        for item in fallback_rows:
            memory_id = item["id"]
            row = connection.execute(
                "SELECT embedding_idx, context FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if row is None or int(row[0]) != int(item["embedding_idx"]):
                raise RuntimeError(f"fallback row preimage changed: {memory_id}")
            text = _embedding_text(_parse_context(row[1]))
            old_vector = vectors[int(row[0])].copy()
            if not is_parallel_hash_fallback_embedding(text, old_vector):
                raise RuntimeError(f"fallback vector preimage changed: {memory_id}")
            new_vector, source_url = _strict_embed(text)
            if is_parallel_hash_fallback_embedding(text, new_vector):
                raise RuntimeError(f"embedder returned fallback for {memory_id}")
            vectors[int(row[0])] = new_vector
            repaired.append(
                {
                    **item,
                    "source_url": source_url,
                    "old_new_cosine": float(np.dot(old_vector, new_vector)),
                    "new_norm": float(np.linalg.norm(new_vector)),
                }
            )

    faiss.normalize_L2(vectors)
    replacement = faiss.IndexFlatIP(index.d)
    replacement.add(vectors)
    if replacement.ntotal != index.ntotal:
        raise RuntimeError("replacement index row count changed")

    tmp = index_path.with_name(f".{index_path.name}.{uuid.uuid4().hex}.repair.tmp")
    try:
        faiss.write_index(replacement, str(tmp))
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        tmp.replace(index_path)
    finally:
        tmp.unlink(missing_ok=True)
    return repaired


def _update_namespaces(db_path: Path, updates: list[dict[str, Any]]) -> int:
    if not updates:
        return 0
    with sqlite3.connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        changed = 0
        for item in updates:
            cursor = connection.execute(
                "UPDATE memories SET action_type = ? "
                "WHERE id = ? AND action_type = ? AND action IS ?",
                (item["to"], item["id"], item["from"], item["action"]),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"namespace row preimage changed: {item['id']}")
            changed += 1
        connection.commit()
    return changed


def apply_repair(sessions: Path, receipt_path: Path) -> dict[str, Any]:
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {receipt_path}")
    lock_path = sessions / ".episodic_faiss.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        before = audit(sessions)
        token = (
            "episodic_routing_poison_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + before["audit_sha256"][:12]
        )
        before_hashes = {
            name: _sha256(sessions / name)
            for name in ("episodic.db", "embeddings.faiss", "id_map.npy")
        }
        backup = _backup_preimage(sessions, token)
        repaired_vectors = _replace_vectors(sessions, before["fallback_vectors"])
        namespace_count = _update_namespaces(
            sessions / "episodic.db", before["namespace_updates"]
        )
        after = audit(sessions)
        if after["namespace_updates"] or after["fallback_vectors"]:
            raise RuntimeError(f"repair postcondition failed: {after}")
        after_hashes = {
            name: _sha256(sessions / name)
            for name in ("episodic.db", "embeddings.faiss", "id_map.npy")
        }
        receipt = {
            "schema_version": SCHEMA,
            "status": "applied_for_validation_unratified",
            "applied_at": _utc_now(),
            "sessions": str(sessions),
            "backup": str(backup),
            "before_hashes": before_hashes,
            "after_hashes": after_hashes,
            "before_audit": before,
            "after_audit": after,
            "namespace_rows_updated": namespace_count,
            "vectors_repaired": repaired_vectors,
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt_path.with_name(f".{receipt_path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        tmp.replace(receipt_path)
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    sessions = args.sessions.expanduser().resolve()
    if args.apply:
        if args.receipt is None:
            raise SystemExit("--apply requires --receipt")
        result = apply_repair(sessions, args.receipt.expanduser().resolve())
    else:
        result = audit(sessions)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
