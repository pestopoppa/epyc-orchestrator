#!/usr/bin/env python3
"""Re-embed episodic memories that lack FAISS embeddings.

Launches multiple temporary llama-server instances with BGE-large in parallel,
embeds all memories with non-empty objectives using concurrent requests,
and saves the result as a numpy array keyed by memory ID.

One-time operation to recover embeddings lost when the FAISS index was rebuilt.
The output is consumed by extract_training_data.py.

Usage:
    python3 scripts/graph_router/reembed_episodic_store.py [--servers 8] [--batch-size 128]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reembed")

DEFAULT_DB = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions/episodic.db"
)
DEFAULT_MODEL = Path("/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf")
DEFAULT_OUTPUT = PROJECT_ROOT / "orchestration/repl_memory/sessions/reembedded.npz"
LLAMA_SERVER = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-server")

from scripts.graph_router.extract_training_data import normalize_action


def probe_existing_server(port: int, timeout: float = 1.0) -> bool:
    """Return True if a healthy embedding server is already running on `port`.

    Used to skip-and-reuse stack-owned embedders rather than colliding with them.
    """
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def start_embedding_server(model_path: Path, port: int, threads: int) -> subprocess.Popen:
    """Launch temporary llama-server for embedding."""
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-np", "4",
        "-t", str(threads),
        "-c", "512",
        "--embeddings",
        "--pooling", "cls",
        "--flash-attn", "on",
        "--log-disable",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    url = f"http://127.0.0.1:{port}/health"
    for i in range(60):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                logger.info("BGE server ready on port %d (took %ds)", port, i)
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)

    proc.kill()
    raise RuntimeError(f"BGE server on port {port} failed to start within 60s")


def embed_batch(texts: list[str], port: int) -> np.ndarray:
    """Embed a batch of texts via llama-server /embedding endpoint."""
    url = f"http://127.0.0.1:{port}/embedding"
    resp = requests.post(url, json={"content": texts}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    embeddings = []
    for item in data:
        if isinstance(item, dict) and "embedding" in item:
            embeddings.append(item["embedding"])
        elif isinstance(item, list):
            embeddings.append(item)
        else:
            raise ValueError(f"Unexpected embedding format: {type(item)}")

    return np.array(embeddings, dtype=np.float32)


def retry_port_order(primary_port: int, ports: list[int]) -> list[int]:
    """Return primary port, then the rest of the pool in round-robin order."""
    try:
        start_idx = ports.index(primary_port)
    except ValueError:
        return [primary_port] + ports
    return [ports[(start_idx + offset) % len(ports)] for offset in range(len(ports))]


def embed_batch_indexed(batch_idx: int, texts: list[str], port: int, ports: list[int]):
    """Embed a batch and return (batch_idx, embeddings) for ordered reassembly."""
    retry_ports = retry_port_order(port, ports)
    last_error: Exception | None = None
    for attempt, candidate_port in enumerate(retry_ports, start=1):
        try:
            embs = embed_batch(texts, candidate_port)
            if attempt > 1:
                logger.info(
                    "Batch %d recovered on port %d after %d attempt(s)",
                    batch_idx, candidate_port, attempt,
                )
            return batch_idx, embs
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Batch %d failed on port %d attempt %d/%d: %s",
                batch_idx, candidate_port, attempt, len(retry_ports), exc,
            )
    raise RuntimeError(f"batch {batch_idx} failed on all embedding ports") from last_error


def main():
    parser = argparse.ArgumentParser(description="Re-embed episodic memories (parallel)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--base-port", type=int, default=8090)
    parser.add_argument("--servers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads-per-server", type=int, default=4)
    args = parser.parse_args()

    db_path = Path(args.db)
    output_path = Path(args.output)
    ports = list(range(args.base_port, args.base_port + args.servers))

    # ── Extract memories from SQLite ──
    logger.info("Reading memories from %s", db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT id, action, action_type, context, outcome, q_value
        FROM memories
        WHERE (action_type = 'routing' OR action_type = 'escalation')
        ORDER BY created_at
    """).fetchall()
    conn.close()
    logger.info("Loaded %d memories", len(rows))

    # Filter to normalizable actions with non-empty objectives
    valid_rows = []
    skipped = {"no_action": 0, "no_objective": 0}
    for row in rows:
        mem_id, action, action_type, ctx_json, outcome, q_value = row
        canonical = normalize_action(action)
        if canonical is None:
            skipped["no_action"] += 1
            continue

        ctx = json.loads(ctx_json) if ctx_json else {}
        objective = ctx.get("objective", "")
        if not objective or not objective.strip():
            skipped["no_objective"] += 1
            continue

        valid_rows.append((mem_id, canonical, ctx, objective.strip()[:450], q_value))

    total = len(valid_rows)
    logger.info(
        "Valid for embedding: %d (skipped: action=%d, no_objective=%d)",
        total, skipped["no_action"], skipped["no_objective"],
    )
    if not valid_rows:
        logger.error("No valid rows to embed")
        sys.exit(1)

    # ── Probe existing servers; launch only the missing ones ──
    # Stack-owned embedders (orchestrator_stack.py runs 6 on 8090-8095) are
    # reused in place. We only own & later kill the ones we spawned ourselves.
    reused_ports: list[int] = []
    spawned_ports: list[int] = []
    server_procs: list[subprocess.Popen] = []
    for port in ports:
        if probe_existing_server(port):
            reused_ports.append(port)
            logger.info("Reusing existing healthy embedding server on port %d", port)
            continue
        spawned_ports.append(port)
    if reused_ports:
        logger.info("Reusing %d stack-owned server(s): %s", len(reused_ports), reused_ports)
    if spawned_ports:
        logger.info("Launching %d new BGE server(s) on ports %s", len(spawned_ports), spawned_ports)
        # Launch sequentially to avoid concurrent mlock
        for port in spawned_ports:
            proc = start_embedding_server(Path(args.model), port, args.threads_per_server)
            server_procs.append(proc)

    try:
        # ── Build batches ──
        batches = []
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            texts = [valid_rows[i][3] for i in range(start, end)]
            batches.append((start, end, texts))

        logger.info(
            "Processing %d batches of size %d across %d servers",
            len(batches), args.batch_size, args.servers,
        )

        # ── Embed in parallel with ThreadPoolExecutor ──
        results_map: dict[int, np.ndarray] = {}
        completed = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.servers) as executor:
            futures = {}
            for batch_idx, (start, end, texts) in enumerate(batches):
                port = ports[batch_idx % args.servers]
                fut = executor.submit(embed_batch_indexed, batch_idx, texts, port, ports)
                futures[fut] = (batch_idx, start, end)

            for fut in as_completed(futures):
                batch_idx, start, end = futures[fut]
                try:
                    idx, embs = fut.result()
                    results_map[idx] = embs
                    completed += end - start
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    if len(results_map) % 50 == 0 or completed == total:
                        eta = (total - completed) / rate if rate > 0 else 0
                        logger.info(
                            "Progress: %d / %d (%.1f%%) — %.0f emb/s — ETA %.0fs",
                            completed, total, 100.0 * completed / total, rate, eta,
                        )
                except Exception as e:
                    logger.error("Batch %d-%d failed: %s", start, end, e)
                    raise

        # ── Reassemble in original order ──
        if len(results_map) != len(batches):
            missing = sorted(set(range(len(batches))) - set(results_map))
            raise RuntimeError(
                f"Missing embeddings for {len(missing)} batch(es); first missing: {missing[:10]}"
            )
        all_ids = []
        all_embeddings = []
        all_actions = []
        all_q_values = []
        all_contexts = []

        for batch_idx in sorted(results_map.keys()):
            start, end, _ = batches[batch_idx]
            embs = results_map[batch_idx]
            for i, row_idx in enumerate(range(start, end)):
                mem_id, canonical, ctx, _, q_value = valid_rows[row_idx]
                all_ids.append(mem_id)
                all_embeddings.append(embs[i])
                all_actions.append(canonical)
                all_q_values.append(q_value)
                all_contexts.append(json.dumps(ctx))

        logger.info("Embedded %d / %d memories in %.1fs", len(all_ids), total, time.time() - t0)

    finally:
        # Kill all servers
        for proc in server_procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        logger.info("All BGE servers stopped")

    # ── Save ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.array(all_ids, dtype=object),
        embeddings=np.stack(all_embeddings),
        actions=np.array(all_actions, dtype=object),
        q_values=np.array(all_q_values, dtype=np.float32),
        contexts=np.array(all_contexts, dtype=object),
    )
    logger.info(
        "Saved %d embeddings to %s (%.1f MB)",
        len(all_ids), output_path,
        output_path.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
