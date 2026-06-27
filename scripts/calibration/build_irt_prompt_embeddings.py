#!/usr/bin/env python3
"""Build keyed prompt embeddings and optional IRT scores for cold-start audits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibration.irt_cold_start_ab import BaselineRecord, load_baseline_records
from scripts.graph_router.irt_scorer import predict_irt_from_embeddings

DEFAULT_EMBEDDER_URLS = tuple(f"http://127.0.0.1:{port}" for port in range(8090, 8096))


def _parse_embedder_urls(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_EMBEDDER_URLS)
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


def _embedding_from_payload(data: Any) -> np.ndarray:
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"unexpected embedding response type: {type(data).__name__}")
    if "embedding" in data:
        value = data["embedding"]
        if value and isinstance(value[0], list):
            value = value[0]
        embedding = np.asarray(value, dtype=np.float32)
    elif data.get("data"):
        embedding = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
    else:
        raise ValueError(f"unexpected embedding response keys: {sorted(data)}")
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def load_prompt_embedding_artifact(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data.files:
        raise ValueError(f"{path} does not contain an embeddings array")
    if "prompt_hashes" not in data.files and "question_ids" not in data.files:
        raise ValueError(f"{path} does not contain prompt_hashes or question_ids")
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    prompt_hashes = (
        np.array([str(item) for item in data["prompt_hashes"].tolist()], dtype=object)
        if "prompt_hashes" in data.files
        else None
    )
    question_ids = (
        np.array([str(item) for item in data["question_ids"].tolist()], dtype=object)
        if "question_ids" in data.files
        else None
    )
    if (prompt_hashes is not None and len(embeddings) != len(prompt_hashes)) or (
        question_ids is not None and len(embeddings) != len(question_ids)
    ):
        raise ValueError("keyed embedding artifact key/embedding lengths are inconsistent")
    return {
        "embeddings": embeddings,
        "prompt_hashes": prompt_hashes,
        "question_ids": question_ids,
    }


def _select_cached_embeddings(
    artifact: dict[str, Any],
    records: list[BaselineRecord],
    *,
    max_records: int | None,
) -> dict[str, Any]:
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    selected = records[:max_records] if max_records is not None else records
    embeddings = artifact["embeddings"]
    by_prompt_hash = {
        str(key): idx
        for idx, key in enumerate(artifact["prompt_hashes"] if artifact["prompt_hashes"] is not None else [])
    }
    by_question_id = {
        str(key): idx
        for idx, key in enumerate(artifact["question_ids"] if artifact["question_ids"] is not None else [])
    }
    matched_embeddings: list[np.ndarray] = []
    prompt_hashes: list[str] = []
    question_ids: list[str] = []
    suites: list[str] = []
    question_keys: list[str] = []
    missing: list[str] = []
    for record in selected:
        idx = by_prompt_hash.get(record.prompt_hash)
        if idx is None:
            idx = by_question_id.get(record.question_id)
        if idx is None:
            missing.append(record.prompt_hash)
            continue
        matched_embeddings.append(embeddings[idx])
        prompt_hashes.append(record.prompt_hash)
        question_ids.append(record.question_id)
        suites.append(record.suite)
        question_keys.append(record.question_key)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"cached embeddings missing {len(missing)} baseline prompt(s): {preview}")
    if not matched_embeddings:
        raise ValueError("no cached embeddings matched baseline records")
    return {
        "embeddings": np.stack(matched_embeddings).astype(np.float32),
        "prompt_hashes": np.array(prompt_hashes, dtype=object),
        "question_ids": np.array(question_ids, dtype=object),
        "suites": np.array(suites, dtype=object),
        "question_keys": np.array(question_keys, dtype=object),
    }


def make_http_embedder(
    urls: list[str],
    *,
    timeout_s: float = 10.0,
) -> Callable[[str], np.ndarray]:
    """Return an embedding callable that fails over across live embedder URLs."""

    import httpx

    client = httpx.Client(timeout=timeout_s)

    def embed(text: str) -> np.ndarray:
        last_error: Exception | None = None
        for url in urls:
            try:
                response = client.post(f"{url}/embedding", json={"content": text})
                response.raise_for_status()
                return _embedding_from_payload(response.json())
            except Exception as exc:  # pragma: no cover - exercised with live smoke, not unit tests
                last_error = exc
                continue
        raise RuntimeError(f"all embedder URLs failed: {last_error}")

    return embed


def embed_records(
    records: list[BaselineRecord],
    embed_text: Callable[[str], np.ndarray],
    *,
    max_records: int | None = None,
) -> dict[str, Any]:
    selected = records[:max_records] if max_records is not None else records
    embeddings: list[np.ndarray] = []
    prompt_hashes: list[str] = []
    question_ids: list[str] = []
    suites: list[str] = []
    question_keys: list[str] = []
    for record in selected:
        embeddings.append(embed_text(record.prompt))
        prompt_hashes.append(record.prompt_hash)
        question_ids.append(record.question_id)
        suites.append(record.suite)
        question_keys.append(record.question_key)
    if not embeddings:
        raise ValueError("no baseline records to embed")
    return {
        "embeddings": np.stack(embeddings).astype(np.float32),
        "prompt_hashes": np.array(prompt_hashes, dtype=object),
        "question_ids": np.array(question_ids, dtype=object),
        "suites": np.array(suites, dtype=object),
        "question_keys": np.array(question_keys, dtype=object),
    }


def write_embeddings_artifact(
    output_path: Path,
    *,
    baseline_path: Path,
    records: list[BaselineRecord],
    embedded: dict[str, Any],
    embedder_urls: list[str],
    elapsed_s: float,
) -> dict[str, Any]:
    metadata = {
        "schema": "epyc.graph_router.irt_prompt_embeddings.v1",
        "baseline_path": str(baseline_path),
        "records": int(embedded["embeddings"].shape[0]),
        "embedding_dim": int(embedded["embeddings"].shape[1]),
        "source_records": len(records),
        "embedder_urls": embedder_urls,
        "elapsed_s": round(elapsed_s, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **embedded, metadata=np.array(metadata, dtype=object))
    return metadata


def write_keyed_irt_scores(
    output_path: Path,
    *,
    embeddings_artifact: dict[str, Any],
    irt_scores_path: Path,
) -> dict[str, Any]:
    irt = np.load(irt_scores_path, allow_pickle=True)
    required = {
        "embedding_dim",
        "projector_feature_mean",
        "projector_feature_scale",
        "projector_weights",
        "projector_target_mean",
        "projector_target_scale",
    }
    if not required.issubset(set(irt.files)):
        raise ValueError(f"{irt_scores_path} does not contain a persisted embedding projector")

    difficulty, discrimination = predict_irt_from_embeddings(
        embeddings_artifact["embeddings"],
        embedding_dim=int(irt["embedding_dim"]),
        feature_mean=irt["projector_feature_mean"],
        feature_scale=irt["projector_feature_scale"],
        weights=irt["projector_weights"],
        target_mean=irt["projector_target_mean"],
        target_scale=irt["projector_target_scale"],
    )
    metadata = {
        "schema": "epyc.graph_router.keyed_irt_scores.v1",
        "irt_scores_path": str(irt_scores_path),
        "records": int(difficulty.shape[0]),
        "response_source": (irt["metadata"].item().get("response_source") if "metadata" in irt.files else None),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prompt_hashes=embeddings_artifact["prompt_hashes"],
        question_ids=embeddings_artifact["question_ids"],
        suites=embeddings_artifact["suites"],
        question_keys=embeddings_artifact["question_keys"],
        latent_difficulty=difficulty,
        latent_discrimination=discrimination,
        metadata=np.array(metadata, dtype=object),
    )
    return metadata


def build_artifacts(
    baseline_path: Path,
    output_embeddings: Path,
    *,
    embed_text: Callable[[str], np.ndarray] | None = None,
    embedder_urls: list[str] | None = None,
    max_records: int | None = None,
    irt_scores_path: Path | None = None,
    output_irt_scores: Path | None = None,
    prompt_embedding_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _, records = load_baseline_records(baseline_path)
    if embed_text is None and prompt_embedding_artifact is None:
        raise ValueError("either embed_text or prompt_embedding_artifact must be provided")
    if embed_text is not None and prompt_embedding_artifact is not None:
        raise ValueError("provide either embed_text or prompt_embedding_artifact, not both")

    if prompt_embedding_artifact is None:
        t0 = time.time()
        embedded = embed_records(records, embed_text, max_records=max_records)  # type: ignore[arg-type]
        elapsed_s = time.time() - t0
    else:
        embedded = _select_cached_embeddings(prompt_embedding_artifact, records, max_records=max_records)
        if embedded["embeddings"].ndim != 2:
            raise ValueError("prompt_embedding_artifact.embeddings must be 2D")
        elapsed_s = 0.0

    embedding_metadata = write_embeddings_artifact(
        output_embeddings,
        baseline_path=baseline_path,
        records=records,
        embedded=embedded,
        embedder_urls=embedder_urls or [],
        elapsed_s=elapsed_s,
    )
    report = {"embeddings": embedding_metadata}
    if irt_scores_path is not None and output_irt_scores is not None:
        report["irt_scores"] = write_keyed_irt_scores(
            output_irt_scores,
            embeddings_artifact=embedded,
            irt_scores_path=irt_scores_path,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build keyed prompt embeddings for IRT cold-start audits")
    parser.add_argument("--baseline", type=Path, required=True, help="Full baseline JSON artifact")
    parser.add_argument("--output-embeddings", type=Path, required=True, help="Output keyed embeddings NPZ")
    parser.add_argument("--embedder-urls", type=str, default=None, help="Comma-separated embedder base URLs")
    parser.add_argument(
        "--input-embeddings",
        type=Path,
        default=None,
        help="Optional keyed prompt embeddings artifact to reuse without live inference",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Optional row cap for smoke runs")
    parser.add_argument("--irt-scores", type=Path, default=None, help="Optional P5.1 IRT scorer artifact")
    parser.add_argument("--output-irt-scores", type=Path, default=None, help="Optional output keyed IRT scores NPZ")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Per-request timeout")
    args = parser.parse_args()

    if (args.irt_scores is None) != (args.output_irt_scores is None):
        parser.error("--irt-scores and --output-irt-scores must be provided together")
    if args.embedder_urls is not None and args.input_embeddings is not None:
        parser.error("--embedder-urls cannot be used with --input-embeddings")

    cached_embeddings = None
    embed_text = None
    embedder_urls: list[str]
    if args.input_embeddings is not None:
        cached_embeddings = load_prompt_embedding_artifact(args.input_embeddings)
        embedder_urls = []
    else:
        urls = _parse_embedder_urls(args.embedder_urls)
        embed_text = make_http_embedder(urls, timeout_s=args.timeout_s)
        embedder_urls = urls
    report = build_artifacts(
        args.baseline,
        args.output_embeddings,
        embed_text=embed_text,
        embedder_urls=embedder_urls,
        max_records=args.max_records,
        prompt_embedding_artifact=cached_embeddings,
        irt_scores_path=args.irt_scores,
        output_irt_scores=args.output_irt_scores,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
