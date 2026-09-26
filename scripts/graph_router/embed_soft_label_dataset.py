#!/usr/bin/env python3
"""P4.5 Phase B step 1: embed soft-label question texts into a training NPZ.

Reads soft_labels.jsonl (per-qid soft label distributions from
extract_journal_soft_labels.py), recovers each question's prompt text from the
benchmark question pool (qid = sha1(f"{suite}\\x00{prompt}")[:16], matching
eval_tower._stable_question_qid), embeds via the BGE /embedding endpoint, and
builds a 1031-dim feature matrix matching the production RoutingClassifier:

    [ 1024 BGE(prompt) | 5 task_type one-hot | 1 norm_ctx_len | 1 has_images ]

Output NPZ fields:
    X            (N, 1031) float32 feature matrix
    soft_labels  (N, 6)    float32 per-role soft target (softmax of correctness)
    hard_labels  (N,)      int64   argmax(soft_labels) — for the CE baseline arm
    qids         (N,)      str
    suites       (N,)      str
    label_map    (6, 2)    object  [[idx, role], ...] over CANONICAL_ROLES

Usage:
    python3 scripts/graph_router/embed_soft_label_dataset.py \
        [--soft-labels PATH] [--pool PATH] [--output PATH] \
        [--base-port 8090] [--servers 4] [--batch-size 64] [--slot-ctx-tokens N]

Every prompt is fitted to the embedder's PER-SLOT context (read from the live
servers' /props n_ctx, default 256) using the server's own /tokenize, and any
prompt that still cannot be embedded aborts the run. A prompt is never replaced
by a zero vector.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("embed_soft_label_dataset")

DEFAULT_SOFT_LABELS = PROJECT_ROOT / "orchestration/reports/p45_soft_labels/soft_labels.jsonl"
DEFAULT_POOL = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl")
DEFAULT_OUTPUT = PROJECT_ROOT / "orchestration/reports/p45_soft_labels/soft_labels_embedded.npz"

# Must match extract_journal_soft_labels.CANONICAL_ROLES and
# routing_classifier label ordering.
CANONICAL_ROLES = [
    "frontdoor",
    "coder_escalation",
    "worker_general",
    "architect_general",
    "ingest_long_context",
    "worker_vision",
]

# Mirrors extract_training_data.TASK_TYPES (production feature ordering).
TASK_TYPES = ["code", "chat", "architecture", "ingest", "general"]

# Suite → production task_type. Code-execution/code-gen suites map to "code";
# multi-doc retrieval maps to "ingest"; everything else is "general". Chat /
# architecture have no eval-suite analog. This only needs to be *consistent*
# across the hard-label and soft-label arms (same features, different target),
# which it is.
SUITE_TO_TASK_TYPE = {
    "cruxeval": "code",
    "bigcodebench": "code",
    "livecodebench": "code",
    "coder": "code",
    "debugbench": "code",
    "long_context": "ingest",
    "hotpotqa": "ingest",
}

VISION_SUITES = {"vl"}

# Token budget per prompt. BGE-large accepts at most 512 tokens, but the
# llama-server pool splits -c across -np slots and rejects (HTTP 400,
# exceed_context_size_error) any input longer than ONE slot. The live pool
# (8090-8095) ran -c 512 -np 4 => n_ctx 256 per slot (verified 2026-09-26 via
# /props); UFH-12 Phase 0 restores -c 2048 -np 4 => 512 per slot. The old fixed cap of 1400 chars was sized for -c 2048 -np 4 (512
# tokens/slot) and could reach ~518 tokens, so dense prompts overflowed the
# 256-token slot and the per-item fallback silently substituted ZERO VECTORS.
#
# Now: the slot size is read from /props at runtime (DEFAULT_SLOT_CTX_TOKENS if
# no server answers), each prompt is pre-truncated by characters and then fitted
# EXACTLY with the server's /tokenize, and an input that still fails raises.
BGE_MAX_TOKENS = 512
DEFAULT_SLOT_CTX_TOKENS = 256
# [CLS] + [SEP] are added at embed time but not counted by /tokenize (live: 400
# words -> /tokenize 400, /embedding n_prompt_tokens 402).
_SPECIAL_TOKEN_RESERVE = 2
# Coarse pre-truncation only (keeps /tokenize requests small); the exact fit is
# done with /tokenize. ~2.5 chars/token is dense (code/tables) BGE text.
_PRETRUNC_CHARS_PER_TOKEN = 2.5
_FIT_MAX_ROUNDS = 8


class EmbeddingInputTooLong(RuntimeError):
    """A prompt exceeds the embedder slot and could not be fitted to it."""


class EmbeddingFailed(RuntimeError):
    """A prompt could not be embedded on any server. Never zero-filled."""


def token_budget(slot_ctx_tokens: int) -> int:
    """Max /tokenize count that fits one slot once [CLS]/[SEP] are added."""
    budget = min(int(slot_ctx_tokens), BGE_MAX_TOKENS) - _SPECIAL_TOKEN_RESERVE
    if budget <= 0:
        raise ValueError(f"slot_ctx_tokens={slot_ctx_tokens} leaves no token budget")
    return budget


def max_prompt_chars(slot_ctx_tokens: int) -> int:
    """Character pre-truncation cap for a slot (coarse; exact fit follows)."""
    return int(token_budget(slot_ctx_tokens) * _PRETRUNC_CHARS_PER_TOKEN)


def probe_slot_ctx_tokens(
    ports: list[int], default: int = DEFAULT_SLOT_CTX_TOKENS, timeout: float = 5.0
) -> int:
    """Smallest per-slot n_ctx across the pool (from /props), capped at 512.

    Falls back to ``default`` when no server answers, so the budget is never
    larger than the live slot on a reachable pool.
    """
    seen: list[int] = []
    for port in ports:
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/props", timeout=timeout)
            resp.raise_for_status()
            n_ctx = resp.json()["default_generation_settings"]["n_ctx"]
            seen.append(int(n_ctx))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read /props n_ctx from port %d: %s", port, exc)
    if not seen:
        logger.warning("No embedder /props answered; assuming %d tokens/slot", default)
        return min(int(default), BGE_MAX_TOKENS)
    if len(set(seen)) > 1:
        logger.warning("Embedder pool has mixed slot sizes %s; using the smallest", seen)
    return min(min(seen), BGE_MAX_TOKENS)


def _count_tokens(text: str, port: int) -> int:
    resp = requests.post(
        f"http://127.0.0.1:{port}/tokenize", json={"content": text}, timeout=60
    )
    resp.raise_for_status()
    return len(resp.json()["tokens"])


def fit_to_token_budget(text: str, port: int, budget: int) -> str:
    """Truncate ``text`` until the server's /tokenize count is <= ``budget``.

    Raises EmbeddingInputTooLong if it cannot be fitted in _FIT_MAX_ROUNDS.
    """
    n_tokens = _count_tokens(text, port)
    rounds = 0
    while n_tokens > budget and rounds < _FIT_MAX_ROUNDS:
        keep = max(1, int(len(text) * budget / n_tokens * 0.95))
        if keep >= len(text):
            keep = len(text) - 1
        text = text[:keep]
        n_tokens = _count_tokens(text, port)
        rounds += 1
    if n_tokens > budget:
        raise EmbeddingInputTooLong(
            f"prompt still {n_tokens} tokens (> budget {budget}) after {rounds} "
            f"truncation rounds ({len(text)} chars)"
        )
    return text


def _stable_qid(suite: str, prompt: str) -> str:
    """Replicate eval_tower._stable_question_qid (sha1, null-byte separator)."""
    payload = f"{suite}\x00{prompt}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _task_type_onehot(task_type: str) -> np.ndarray:
    vec = np.zeros(len(TASK_TYPES), dtype=np.float32)
    tt = (task_type or "general").lower()
    for i, name in enumerate(TASK_TYPES):
        if name in tt:
            vec[i] = 1.0
            return vec
    vec[TASK_TYPES.index("general")] = 1.0
    return vec


def _build_pool_index(pool_path: Path) -> dict[str, dict]:
    """Map qid -> {suite, prompt} for every scoreable pool entry."""
    index: dict[str, dict] = {}
    with open(pool_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            suite = obj.get("suite")
            prompt = obj.get("prompt")
            if not suite or not prompt:
                continue
            index[_stable_qid(suite, prompt)] = {
                "suite": suite,
                "prompt": prompt,
                "context": obj.get("context", ""),
            }
    return index


def _embed_batch(texts: list[str], port: int) -> np.ndarray:
    """Embed a batch via the llama-server /embedding endpoint (CLS pooling)."""
    url = f"http://127.0.0.1:{port}/embedding"
    resp = requests.post(url, json={"content": texts}, timeout=120)
    if resp.status_code == 400 and "exceed_context_size" in (resp.text or ""):
        # Same slot size on every port — retrying elsewhere cannot help.
        raise EmbeddingInputTooLong(f"port {port}: {resp.text[:300]}")
    resp.raise_for_status()
    data = resp.json()
    out = []
    for item in data:
        if isinstance(item, dict) and "embedding" in item:
            emb = item["embedding"]
        elif isinstance(item, list):
            emb = item
        else:
            raise ValueError(f"Unexpected embedding format: {type(item)}")
        # llama-server returns embedding as a nested [[...]] (1 x dim) after
        # pooling; flatten a single-row nesting to a flat vector.
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 2:
            # Multiple token rows returned without pooling — mean-pool defensively.
            arr = arr.mean(axis=0)
        out.append(arr)
    return np.vstack(out).astype(np.float32)


def _embed_indexed(batch_idx: int, texts: list[str], ports: list[int]) -> tuple[int, np.ndarray]:
    """Embed a batch, round-robin retrying across ports.

    On whole-batch failure, fall back to per-item embedding so a transient
    batch error does not drop the rest. A single item that still fails on every
    port RAISES (EmbeddingInputTooLong / EmbeddingFailed): a zero vector is a
    structurally valid feature row that silently poisons training data.
    """
    primary = ports[batch_idx % len(ports)]
    order = [ports[(ports.index(primary) + o) % len(ports)] for o in range(len(ports))]
    for cand in order:
        try:
            return batch_idx, _embed_batch(texts, cand)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch %d failed on port %d: %s", batch_idx, cand, exc)

    # Per-item fallback.
    logger.warning("Batch %d: falling back to per-item embedding", batch_idx)
    rows: list[np.ndarray] = []
    for item_idx, text in enumerate(texts):
        emb = None
        last_exc: Exception | None = None
        for cand in order:
            try:
                emb = _embed_batch([text], cand)[0]
                break
            except EmbeddingInputTooLong:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        if emb is None:
            raise EmbeddingFailed(
                f"batch {batch_idx} item {item_idx} ({len(text)} chars) could not be "
                f"embedded on any of ports {order}: {last_exc}"
            )
        rows.append(emb)
    return batch_idx, np.vstack(rows).astype(np.float32)


def _assert_embeddings_valid(emb_matrix: np.ndarray) -> None:
    """Refuse zero / non-finite rows: they are silent corruption, not data."""
    if not np.all(np.isfinite(emb_matrix)):
        raise EmbeddingFailed("non-finite values in embedding matrix")
    zero_rows = np.flatnonzero(~np.any(emb_matrix != 0.0, axis=1))
    if zero_rows.size:
        raise EmbeddingFailed(
            f"{zero_rows.size} all-zero embedding row(s), first at index {int(zero_rows[0])}"
        )


def fit_prompts(prompts: list[str], ports: list[int], slot_ctx_tokens: int) -> list[str]:
    """Pre-truncate by chars, then fit each prompt exactly with /tokenize."""
    budget = token_budget(slot_ctx_tokens)
    cap = max_prompt_chars(slot_ctx_tokens)
    pre = [p[:cap] for p in prompts]

    def _fit(i: int) -> str:
        return fit_to_token_budget(pre[i], ports[i % len(ports)], budget)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as ex:
        return list(ex.map(_fit, range(len(pre))))


def build_dataset(
    soft_labels_path: Path,
    pool_path: Path,
    output_path: Path,
    base_port: int,
    servers: int,
    batch_size: int,
    slot_ctx_tokens: int | None = None,
) -> dict:
    logger.info("Loading soft labels from %s", soft_labels_path)
    records = []
    with open(soft_labels_path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d soft-label records", len(records))

    logger.info("Building pool index from %s", pool_path)
    pool = _build_pool_index(pool_path)
    logger.info("Pool index: %d qids", len(pool))

    # Resolve text; drop records whose qid is not recoverable.
    resolved, unresolved = [], 0
    for rec in records:
        hit = pool.get(rec["qid"])
        if hit is None:
            unresolved += 1
            continue
        resolved.append((rec, hit))
    logger.info("Resolved %d/%d records (%d unresolved)", len(resolved), len(records), unresolved)
    if not resolved:
        raise SystemExit("No records resolved to pool text — aborting.")

    # Embed prompts in batches across the BGE server pool. Each prompt is fitted
    # to ONE server slot (see DEFAULT_SLOT_CTX_TOKENS); the CLS embedding
    # captures the leading-prompt gist, which is what routing needs.
    ports = list(range(base_port, base_port + servers))
    if slot_ctx_tokens is None:
        slot_ctx_tokens = probe_slot_ctx_tokens(ports)
    logger.info(
        "Embedder slot: %d tokens (budget %d, pre-truncation %d chars)",
        slot_ctx_tokens, token_budget(slot_ctx_tokens), max_prompt_chars(slot_ctx_tokens),
    )
    prompts = fit_prompts([hit["prompt"] for _, hit in resolved], ports, slot_ctx_tokens)
    logger.info("Embedding %d prompts across ports %s", len(prompts), ports)

    # Batches keyed by ordinal position so reassembly is order-stable.
    batches = [
        (pos, prompts[i:i + batch_size])
        for pos, i in enumerate(range(0, len(prompts), batch_size))
    ]
    results: dict[int, np.ndarray] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as ex:
        futs = [ex.submit(_embed_indexed, pos, texts, ports) for pos, texts in batches]
        for fut in concurrent.futures.as_completed(futs):
            pos, embs = fut.result()
            results[pos] = embs

    emb_matrix = np.vstack([results[pos] for pos in range(len(batches))])
    if emb_matrix.shape[0] != len(resolved):
        raise SystemExit(
            f"Embedding count mismatch: {emb_matrix.shape[0]} != {len(resolved)}"
        )
    if emb_matrix.shape[1] != 1024:
        raise SystemExit(f"Expected 1024-d BGE, got {emb_matrix.shape[1]}-d")
    _assert_embeddings_valid(emb_matrix)

    # Build the 1031-d feature matrix + targets.
    X_rows, soft_rows, hard_rows, corr_rows, qids, suites = [], [], [], [], [], []
    for (rec, hit), emb in zip(resolved, emb_matrix):
        suite = rec["suite"]
        tt = SUITE_TO_TASK_TYPE.get(suite, "general")
        tt_vec = _task_type_onehot(tt)
        prompt_len = len(hit["prompt"]) + len(hit.get("context", "") or "")
        norm_ctx_len = np.float32(np.log1p(prompt_len) / 12.0)
        has_images = np.float32(1.0 if suite in VISION_SUITES else 0.0)
        feat = np.concatenate([emb, tt_vec, [norm_ctx_len], [has_images]]).astype(np.float32)

        soft = np.asarray(rec["soft_labels"], dtype=np.float32)
        # correctness_vector is the raw per-role mean correctness over CANONICAL_ROLES
        corr = np.asarray(rec["correctness_vector"], dtype=np.float32)
        X_rows.append(feat)
        soft_rows.append(soft)
        corr_rows.append(corr)
        hard_rows.append(int(np.argmax(soft)))
        qids.append(rec["qid"])
        suites.append(suite)

    X = np.vstack(X_rows).astype(np.float32)
    soft_labels = np.vstack(soft_rows).astype(np.float32)
    correctness = np.vstack(corr_rows).astype(np.float32)
    hard_labels = np.asarray(hard_rows, dtype=np.int64)
    label_map = np.array(
        [[i, r] for i, r in enumerate(CANONICAL_ROLES)], dtype=object
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        soft_labels=soft_labels,
        correctness=correctness,
        hard_labels=hard_labels,
        qids=np.array(qids, dtype=object),
        suites=np.array(suites, dtype=object),
        label_map=label_map,
    )
    logger.info("Saved embedded dataset to %s", output_path)

    summary = {
        "records_in": len(records),
        "resolved": len(resolved),
        "unresolved": unresolved,
        "feature_dim": int(X.shape[1]),
        "n_roles": len(CANONICAL_ROLES),
        "output": str(output_path),
        "hard_label_distribution": {
            CANONICAL_ROLES[i]: int((hard_labels == i).sum())
            for i in range(len(CANONICAL_ROLES))
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft-labels", type=Path, default=DEFAULT_SOFT_LABELS)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-port", type=int, default=8090)
    parser.add_argument("--servers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--slot-ctx-tokens",
        type=int,
        default=None,
        help="Per-slot embedder context; default: read /props n_ctx from the pool "
        f"(fallback {DEFAULT_SLOT_CTX_TOKENS}).",
    )
    args = parser.parse_args()

    summary = build_dataset(
        soft_labels_path=args.soft_labels,
        pool_path=args.pool,
        output_path=args.output,
        base_port=args.base_port,
        servers=args.servers,
        batch_size=args.batch_size,
        slot_ctx_tokens=args.slot_ctx_tokens,
    )

    print("\n=== Embed Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
