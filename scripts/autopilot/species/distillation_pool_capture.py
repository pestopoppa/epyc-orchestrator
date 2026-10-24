"""Write an immutable, prospective receipt for one selected distillation input pool.

The receipt covers the exact metadata rows returned to one invocation. It does not
claim an atomic or complete snapshot of the mutable episodic store.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import marshal
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any

SCHEMA = "epyc.distillation_pool_integrity.v1"
CAPTURE_DIR = "distillation_pool_captures"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _counts(rows: list[dict[str, Any]], key: str, *, missing: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        has_value = value is not None and not (key == "outcome" and not value)
        label = str(value) if has_value else missing
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        return {"path": str(path.absolute()), "available": False, "error": type(exc).__name__}
    return {
        "path": str(resolved),
        "available": True,
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def _code_provenance(repo_root: Path, source_path: Path, loaded_function: Any) -> dict[str, Any]:
    source_bytes = source_path.read_bytes()
    helper_path = Path(__file__).resolve()
    helper_bytes = helper_path.read_bytes()
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        dirty_result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True, timeout=2,
        )
        tracked_dirty: bool | None = bool(dirty_result.stdout.strip())
        git_error = None
    except (OSError, subprocess.SubprocessError) as exc:
        head = None
        tracked_dirty = None
        git_error = type(exc).__name__
    loaded_callable = getattr(loaded_function, "__func__", loaded_function)
    code = getattr(loaded_callable, "__code__", None)
    loaded_fingerprint = sha256_bytes(marshal.dumps(code)) if code is not None else None
    helper_code = getattr(write_capture, "__code__", None)
    loaded_helper_fingerprint = (
        sha256_bytes(marshal.dumps(helper_code)) if helper_code is not None else None
    )
    return {
        "repo_root": str(repo_root.resolve()),
        "git_head": head,
        "tracked_dirty": tracked_dirty,
        "git_probe_error": git_error,
        "source_file": str(source_path.resolve()),
        "source_file_sha256": sha256_bytes(source_bytes),
        "capture_helper_file": str(helper_path),
        "capture_helper_sha256": sha256_bytes(helper_bytes),
        "loaded_capture_helper_sha256": loaded_helper_fingerprint,
        "loaded_capture_helper_identity_known": loaded_helper_fingerprint is not None,
        "loaded_function_sha256": loaded_fingerprint,
        "loaded_function_identity_known": loaded_fingerprint is not None,
        "python_version": sys.version.split()[0],
    }


def _selected_row(mem: Any) -> dict[str, Any]:
    context = mem.context if isinstance(mem.context, dict) else {}
    extra = context.get("extra") if isinstance(context.get("extra"), dict) else {}
    return {
        "id": mem.id,
        "action_type": mem.action_type,
        "q_value": mem.q_value,
        "outcome": mem.outcome,
        "model_id": mem.model_id,
        "from_tier": extra.get("from_tier"),
        "to_tier": extra.get("to_tier"),
    }


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(fd)
        # Publish a complete file atomically without ever replacing an existing receipt.
        os.link(temp_path, path, follow_symlinks=False)
    finally:
        os.close(fd)
        temp_path.unlink(missing_ok=True)
    dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_capture(
    *,
    repo_root: Path,
    output_dir: Path,
    source_path: Path,
    loaded_function: Any,
    store: Any,
    memories: list[Any],
    teacher_requested: str,
    teacher_declared_model_id: str | None,
    max_trajectories: int,
    min_q_value: float,
    query_vector: Any,
    projected_records: list[dict[str, Any]],
    legacy_m25_pool: dict[str, Any],
    synthetic: bool = False,
) -> dict[str, Any]:
    """Persist rows and the count-integrity verdict; return path/digest/verdict."""
    invocation_id = str(uuid.uuid4())
    rows = [_selected_row(mem) for mem in memories]
    projection_marker_shape_valid = True
    projected_escalation_ids: list[str | None] = []
    for row in projected_records:
        markers = row.get("escalations")
        if not isinstance(markers, list):
            projection_marker_shape_valid = False
            continue
        for marker in markers:
            if not isinstance(marker, dict):
                projection_marker_shape_valid = False
                projected_escalation_ids.append(None)
                continue
            marker_id = marker.get("memory_id")
            if not isinstance(marker_id, str) or not marker_id:
                projection_marker_shape_valid = False
                projected_escalation_ids.append(None)
            else:
                projected_escalation_ids.append(marker_id)
    projected_escalation_ids.sort(key=lambda value: "" if value is None else value)
    native_escalation_ids = sorted(
        row["id"] for row in rows if row.get("action_type") == "escalation"
    )
    native_escalations = sum(1 for row in rows if row.get("action_type") == "escalation")
    derived = {
        "selected_total": len(rows),
        "projected_record_count": len(projected_records),
        "by_action_type": _counts(rows, "action_type", missing="<null>"),
        "by_outcome": _counts(rows, "outcome", missing="unknown"),
        "by_model_id": _counts(rows, "model_id", missing="<null>"),
        "legacy_m25_by_outcome": _counts(rows, "outcome", missing="unknown"),
        "native_escalation_count": native_escalations,
        "projected_escalation_count": len(projected_escalation_ids),
    }
    checks = {
        "selected_total_matches_native_selection": derived["selected_total"] == len(memories),
        "selected_total_matches_projected_records": (
            derived["selected_total"] == derived["projected_record_count"]
        ),
        "selected_total_within_requested_top_k": len(rows) <= max_trajectories,
        "selected_q_values_meet_filter": all(row["q_value"] >= min_q_value for row in rows),
        "group_counts_rederive_from_rows": (
            sum(derived["by_action_type"].values()) == len(rows)
            and sum(derived["by_outcome"].values()) == len(rows)
            and sum(derived["by_model_id"].values()) == len(rows)
        ),
        "legacy_m25_outcomes_match_native_rows": (
            legacy_m25_pool.get("total") == len(rows)
            and legacy_m25_pool.get("by_outcome") == derived["by_outcome"]
        ),
        "legacy_m25_filter_matches_selector": legacy_m25_pool.get("min_q_value") == min_q_value,
        "projection_marker_shape_valid": projection_marker_shape_valid,
        "escalation_projection_matches_native_action_type": (
            projected_escalation_ids == native_escalation_ids
            and derived["projected_escalation_count"] == len(projected_escalation_ids)
        ),
        "native_escalation_count_matches_rows": (
            derived["native_escalation_count"] == native_escalations
        ),
        "selected_ids_unique": len({row["id"] for row in rows}) == len(rows),
    }
    errors = [name for name, passed in checks.items() if not passed]
    verdict = not errors
    result_word = "passed" if verdict else "failed"
    detail = (
        "The returned rows meet the recorded top-k and Q filters; their IDs and metadata groups "
        "are internally consistent; legacy M-25 outcome counts rederive; and projected "
        "escalation marker IDs match rows whose native action_type is escalation."
    )
    proposition = (
        f"Selected-memory integrity checks for invocation {invocation_id} {result_word}: {detail}"
        if verdict else
        f"Selected-memory integrity checks for invocation {invocation_id} failed; "
        f"false checks: {', '.join(errors)}."
    )
    query_bytes = getattr(query_vector, "tobytes", lambda: b"")()
    db_path = Path(store.sqlite_path)
    embedding_store = getattr(store, "_embedding_store", None)
    record = {
        "schema": SCHEMA,
        "invocation_id": invocation_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "capture_stage": "selected_pool_before_teacher",
        "synthetic": synthetic or teacher_requested == "mock",
        "unit": "episodic_memory_pattern",
        "selector": {
            "method": "EpisodicStore.retrieve_by_similarity",
            "query_method": "zero_vector",
            "query_dimension": int(getattr(query_vector, "size", 0)),
            "query_vector_sha256": sha256_bytes(query_bytes),
            "k": max_trajectories,
            "candidate_overfetch_multiplier": 2,
            "action_type_filter": None,
            "min_q_value": min_q_value,
            "indexed_row_predicate": "TRIM(COALESCE(action_type, '')) != ''",
            "return_order": "similarity_score_descending_then_top_k",
            "backend": type(embedding_store).__name__ if embedding_store is not None else None,
            "use_faiss": getattr(store, "use_faiss", None),
        },
        "teacher": {
            "requested": teacher_requested,
            "declared_model_id": teacher_declared_model_id,
            "actual_served_model_id": None,
            "actual_served_identity_basis": "unobserved",
        },
        "code_provenance": _code_provenance(repo_root, source_path, loaded_function),
        "source_store": {
            "sqlite": _file_identity(db_path),
            "storage_dir": _file_identity(Path(store.storage_dir)),
            "embedding_index": _file_identity(Path(getattr(
                embedding_store, "index_path", db_path.parent / "embeddings.faiss"
            ))),
            "id_map": _file_identity(Path(getattr(
                embedding_store, "id_map_path", db_path.parent / "id_map.npy"
            ))),
            "snapshot_claim": (
                "none; exact returned metadata rows only, no whole-store hash or atomic snapshot"
            ),
        },
        "selected_rows": rows,
        "selected_rows_sha256": sha256_bytes(canonical_bytes(rows)),
        "projection_markers": {"escalation_memory_ids": projected_escalation_ids},
        "counts": derived,
        "legacy_m25_pool": legacy_m25_pool,
        "integrity": {
            "decided_proposition": proposition,
            "verdict": verdict,
            "checks": checks,
            "errors": errors,
        },
    }
    record["receipt_sha256"] = sha256_bytes(canonical_bytes(record))
    path = output_dir / CAPTURE_DIR / f"{invocation_id}.json"
    _write_exclusive(path, canonical_bytes(record))
    return {
        "path": str(path),
        "sha256": record["receipt_sha256"],
        "verdict": verdict,
        "errors": errors,
    }
