#!/usr/bin/env python3
"""Build verifier-compatible NPZ data from offline reward feature manifests.

This consumes the prompt-free A9 feature manifest, re-reads source benchmark
prompts locally for embedding, and emits the same core verifier NPZ contract as
extract_verifier_training_data.py:

    Z, correct, sample_weights, actions, q_weights, feature_dim, n_actions

The output is an offline preparation artifact. It is not a live routing weight
file and it does not replace the existing outcome-backed verifier data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.action_space import (
    action_index_for_raw_label,
    load_live_canonical_actions,
)
from scripts.graph_router.build_offline_reward_feature_manifest import (
    FEATURE_ROW_SCHEMA_VERSION,
    _read_source_records,
)

logger = logging.getLogger("offline_reward_verifier_npz")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NPZ_SUMMARY_SCHEMA_VERSION = "offline_reward_verifier_npz_summary.v1"
ROLE_ALIASES = {
    "coder_primary": "coder_escalation",
}
PRIVATE_FIELDS = {"answer", "expected", "prompt", "reference", "response"}


class OfflineRewardVerifierNpzError(ValueError):
    """Raised when manifest rows cannot be converted into verifier data."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise OfflineRewardVerifierNpzError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _source_prompt(row: dict[str, Any], source_cache: dict[Path, list[dict[str, Any]]]) -> str:
    source_path = Path(str(row.get("source_path") or ""))
    offset = row.get("source_record_offset")
    if not isinstance(offset, int):
        raise OfflineRewardVerifierNpzError(f"{row.get('item_id')}: source_record_offset must be int")
    if source_path not in source_cache:
        source_cache[source_path] = _read_source_records(source_path)
    records = source_cache[source_path]
    if offset < 0 or offset >= len(records):
        raise OfflineRewardVerifierNpzError(
            f"{row.get('item_id')}: source_record_offset={offset} outside 0..{len(records) - 1}"
        )
    prompt = records[offset].get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise OfflineRewardVerifierNpzError(f"{row.get('item_id')}: source prompt is empty")
    return prompt


def _engineered_features(row: dict[str, Any]) -> np.ndarray:
    context = row.get("feature_context")
    if not isinstance(context, dict):
        raise OfflineRewardVerifierNpzError(f"{row.get('item_id')}: feature_context must be object")
    task_vec = context.get("task_type_onehot")
    if not isinstance(task_vec, list) or len(task_vec) != 5:
        raise OfflineRewardVerifierNpzError(
            f"{row.get('item_id')}: task_type_onehot must have length 5"
        )
    context_length = context.get("context_length_chars")
    if not isinstance(context_length, int) or context_length < 0:
        raise OfflineRewardVerifierNpzError(
            f"{row.get('item_id')}: context_length_chars must be non-negative int"
        )
    has_images = bool(context.get("has_images", False))
    return np.array(
        [*map(float, task_vec), float(np.log1p(context_length) / 12.0), 1.0 if has_images else 0.0],
        dtype=np.float32,
    )


def _embedding_from_response(data: Any) -> np.ndarray:
    if isinstance(data, dict) and "embedding" in data:
        embedding = data["embedding"]
        if embedding and isinstance(embedding[0], list):
            embedding = embedding[0]
        return np.asarray(embedding, dtype=np.float32)
    if isinstance(data, dict) and "data" in data:
        return np.asarray(data["data"][0]["embedding"], dtype=np.float32)
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "embedding" in data[0]:
            embedding = data[0]["embedding"]
            if embedding and isinstance(embedding[0], list):
                embedding = embedding[0]
            return np.asarray(embedding, dtype=np.float32)
        if data and isinstance(data[0], list):
            return np.asarray(data[0], dtype=np.float32)
    raise OfflineRewardVerifierNpzError(f"unexpected embedding response shape: {type(data).__name__}")


def embed_text_http(text: str, ports: list[int], timeout: float) -> np.ndarray:
    last_error: Exception | None = None
    for port in ports:
        try:
            response = requests.post(
                f"http://127.0.0.1:{port}/embedding",
                json={"content": text},
                timeout=timeout,
            )
            response.raise_for_status()
            embedding = _embedding_from_response(response.json()).reshape(-1)
            if embedding.shape[0] != 1024:
                raise OfflineRewardVerifierNpzError(
                    f"embedding server {port} returned dim={embedding.shape[0]}, expected 1024"
                )
            return embedding.astype(np.float32)
        except Exception as exc:
            last_error = exc
            logger.warning("Embedding failed on port %d: %s", port, exc)
    raise OfflineRewardVerifierNpzError("all embedding ports failed") from last_error


def _sample_weights(correct: np.ndarray) -> np.ndarray:
    n = int(correct.shape[0])
    n_pos = int(correct.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise OfflineRewardVerifierNpzError("degenerate oracle labels; need both classes")
    pos_weight = n / (2.0 * n_pos)
    neg_weight = n / (2.0 * n_neg)
    return np.where(correct == 1.0, pos_weight, neg_weight).astype(np.float32)


def _assert_prompt_free_metadata(metadata_rows: Iterable[dict[str, Any]]) -> None:
    for index, row in enumerate(metadata_rows, start=1):
        present = sorted(PRIVATE_FIELDS & set(row))
        if present:
            raise OfflineRewardVerifierNpzError(
                f"metadata row {index}: private fields present: {', '.join(present)}"
            )


def build_verifier_npz(
    manifest_jsonl: Path,
    out_npz: Path,
    *,
    summary_json: Path | None = None,
    summary_md: Path | None = None,
    embedding_ports: list[int] | None = None,
    embedding_timeout: float = 120.0,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    drop_unmapped_actions: bool = False,
) -> dict[str, Any]:
    rows = load_jsonl(manifest_jsonl)
    if not rows:
        raise OfflineRewardVerifierNpzError("manifest is empty")

    canonical_actions = load_live_canonical_actions()
    n_actions = len(canonical_actions)
    if n_actions == 0:
        raise OfflineRewardVerifierNpzError("no canonical actions available")

    ports = embedding_ports or [8090, 8091, 8092, 8093, 8094, 8095]
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    embedding_cache: dict[tuple[str, int], np.ndarray] = {}
    X_rows: list[np.ndarray] = []
    correct_labels: list[float] = []
    action_labels: list[int] = []
    q_weights: list[float] = []
    metadata_rows: list[dict[str, Any]] = []
    role_counts = Counter()
    canonical_role_counts = Counter()
    dropped_unmapped = Counter()

    for row_number, row in enumerate(rows, start=1):
        if row.get("schema_version") != FEATURE_ROW_SCHEMA_VERSION:
            raise OfflineRewardVerifierNpzError(
                f"{manifest_jsonl}:{row_number}: expected schema_version={FEATURE_ROW_SCHEMA_VERSION!r}"
            )
        role_key = str(row.get("role_key") or "")
        canonical_role = ROLE_ALIASES.get(role_key, role_key)
        action_idx = action_index_for_raw_label(
            canonical_role,
            canonical_actions,
            include_seeded_frontdoor=True,
        )
        if action_idx is None:
            if drop_unmapped_actions:
                dropped_unmapped[role_key] += 1
                continue
            raise OfflineRewardVerifierNpzError(
                f"{manifest_jsonl}:{row_number}: cannot map role_key={role_key!r}"
            )

        source_path = Path(str(row.get("source_path") or ""))
        offset = row.get("source_record_offset")
        if not isinstance(offset, int):
            raise OfflineRewardVerifierNpzError(
                f"{manifest_jsonl}:{row_number}: source_record_offset must be int"
            )
        cache_key = (str(source_path), offset)
        if cache_key not in embedding_cache:
            prompt = _source_prompt(row, source_cache)
            embedding = embed_fn(prompt) if embed_fn is not None else embed_text_http(prompt, ports, embedding_timeout)
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if embedding.shape[0] != 1024:
                raise OfflineRewardVerifierNpzError(
                    f"{manifest_jsonl}:{row_number}: embedding dim={embedding.shape[0]}, expected 1024"
                )
            embedding_cache[cache_key] = embedding

        engineered = _engineered_features(row)
        features = np.concatenate([embedding_cache[cache_key], engineered]).astype(np.float32)
        if features.shape[0] != 1031:
            raise OfflineRewardVerifierNpzError(
                f"{manifest_jsonl}:{row_number}: feature dim={features.shape[0]}, expected 1031"
            )

        oracle_label = row.get("oracle_binary_label")
        oracle_score = row.get("oracle_score")
        if oracle_label not in (0, 1):
            raise OfflineRewardVerifierNpzError(
                f"{manifest_jsonl}:{row_number}: oracle_binary_label must be 0/1"
            )
        if not isinstance(oracle_score, (int, float)):
            raise OfflineRewardVerifierNpzError(
                f"{manifest_jsonl}:{row_number}: oracle_score must be numeric"
            )

        X_rows.append(features)
        correct_labels.append(float(oracle_label))
        action_labels.append(action_idx)
        q_weights.append(max(0.01, float(oracle_score)))
        role_counts[role_key] += 1
        canonical_role_counts[canonical_actions[action_idx]] += 1
        metadata_rows.append(
            {
                "item_id": row.get("item_id"),
                "join_key": row.get("join_key"),
                "question_id": row.get("question_id"),
                "suite": row.get("suite"),
                "role_key": role_key,
                "canonical_action": canonical_actions[action_idx],
                "source_path": str(source_path),
                "source_record_offset": offset,
                "source_record_index": row.get("source_record_index"),
                "source_record_index_base": row.get("source_record_index_base"),
                "prompt_sha256": row.get("prompt_sha256"),
                "expected_sha256": row.get("expected_sha256"),
                "answer_sha256": row.get("answer_sha256"),
                "oracle_score": float(oracle_score),
                "oracle_threshold": row.get("oracle_threshold"),
                "target_binary_label": row.get("target_binary_label"),
            }
        )

    if not X_rows:
        raise OfflineRewardVerifierNpzError("no rows converted into verifier data")

    _assert_prompt_free_metadata(metadata_rows)
    X = np.stack(X_rows).astype(np.float32)
    correct = np.asarray(correct_labels, dtype=np.float32)
    actions = np.asarray(action_labels, dtype=np.int64)
    one_hot = np.zeros((X.shape[0], n_actions), dtype=np.float32)
    one_hot[np.arange(X.shape[0]), actions] = 1.0
    Z = np.concatenate([X, one_hot], axis=1).astype(np.float32)
    q_arr = np.asarray(q_weights, dtype=np.float32)
    sample_weights = _sample_weights(correct)

    summary = {
        "schema_version": NPZ_SUMMARY_SCHEMA_VERSION,
        "manifest_jsonl": str(manifest_jsonl),
        "out_npz": str(out_npz),
        "rows": int(Z.shape[0]),
        "unique_source_records_embedded": len(embedding_cache),
        "feature_dim": int(X.shape[1]),
        "n_actions": n_actions,
        "z_dim": int(Z.shape[1]),
        "n_pos": int(correct.sum()),
        "n_neg": int(correct.shape[0] - int(correct.sum())),
        "role_counts": dict(sorted(role_counts.items())),
        "canonical_action_counts": dict(sorted(canonical_role_counts.items())),
        "role_aliases": ROLE_ALIASES,
        "dropped_unmapped_actions": dict(sorted(dropped_unmapped.items())),
        "label_source": "offline_reward_feature_manifest/reference_token_coverage@0.86",
        "privacy": {
            "private_fields_excluded": sorted(PRIVATE_FIELDS),
            "text_represented_by_sha256_and_lengths": True,
            "npz_contains_embeddings_not_prompt_text": True,
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        Z=Z,
        correct=correct,
        sample_weights=sample_weights,
        actions=actions,
        q_weights=q_arr,
        feature_dim=np.int64(X.shape[1]),
        n_actions=np.int64(n_actions),
        label_map=np.array(list(enumerate(canonical_actions)), dtype=object),
        canonical_actions=np.array(canonical_actions, dtype=object),
        label_source=np.array(summary["label_source"], dtype=object),
        metadata=np.array(metadata_rows, dtype=object),
        manifest_jsonl=np.array(str(manifest_jsonl), dtype=object),
    )
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary_md is not None:
        summary_md.parent.mkdir(parents=True, exist_ok=True)
        summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Offline Reward Verifier NPZ",
            "",
            f"- Manifest: `{summary['manifest_jsonl']}`",
            f"- Output: `{summary['out_npz']}`",
            f"- Rows: `{summary['rows']}`",
            f"- Unique source records embedded: `{summary['unique_source_records_embedded']}`",
            f"- Feature dimension: `{summary['feature_dim']}`",
            f"- Action count: `{summary['n_actions']}`",
            f"- Z dimension: `{summary['z_dim']}`",
            f"- Positives / negatives: `{summary['n_pos']}` / `{summary['n_neg']}`",
            "",
            "This artifact is offline verifier-training preparation. It is not a live",
            "routing weight file and does not enable the frontdoor verifier gate.",
            "",
        ]
    )


def _parse_ports(value: str) -> list[int]:
    ports = [int(part) for part in value.split(",") if part.strip()]
    if not ports:
        raise argparse.ArgumentTypeError("must provide at least one port")
    return ports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build verifier-compatible NPZ from offline reward feature manifests",
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--embedding-ports", type=_parse_ports, default=[8090, 8091, 8092, 8093, 8094, 8095])
    parser.add_argument("--embedding-timeout", type=float, default=120.0)
    parser.add_argument("--drop-unmapped-actions", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = build_verifier_npz(
            args.manifest_jsonl,
            args.out_npz,
            summary_json=args.summary_json,
            summary_md=args.summary_md,
            embedding_ports=args.embedding_ports,
            embedding_timeout=args.embedding_timeout,
            drop_unmapped_actions=args.drop_unmapped_actions,
        )
    except (OfflineRewardVerifierNpzError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
