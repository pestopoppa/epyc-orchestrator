#!/usr/bin/env python3
"""
Seed the episodic memory with canonical REPL tool usage examples.

Usage:
    python orchestration/repl_memory/seed_loader.py [--init] [--force]

Options:
    --init      Load any missing canonical seeds without clearing memories
    --force     Clear existing memories and reload all seeds
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.repl_memory.embedder import TaskEmbedder
from orchestration.repl_memory.episodic_store import EpisodicStore
from orchestration.repl_memory.memory_record import record_from_legacy_context

logger = logging.getLogger(__name__)

SEED_FILE = Path(__file__).parent / "seed_examples.json"


def load_seeds() -> list[dict]:
    """Load seed examples from JSON file."""
    with open(SEED_FILE) as f:
        return json.load(f)


def _seed_key(action_type: str, action: str, task_description: str) -> tuple[str, str, str]:
    """Stable identity used to avoid duplicate canonical seeds on startup."""
    return (action_type, action, task_description)


def _get_persona_seeds() -> list[dict]:
    """Return persona-selection seed examples.

    These teach MemRL which persona overlay works best for which task
    types, bootstrapping persona selection before enough real task
    outcomes accumulate.

    Returns:
        List of persona seed dictionaries.
    """
    return [
        # Engineering personas
        {
            "task": "Review this code for SQL injection vulnerabilities",
            "action": "persona:security_auditor",
            "outcome": "success",
        },
        {
            "task": "Check the authentication flow for CSRF and session fixation",
            "action": "persona:security_auditor",
            "outcome": "success",
        },
        {
            "task": "Write documentation explaining the orchestration architecture",
            "action": "persona:technical_writer",
            "outcome": "success",
        },
        {
            "task": "Profile the database query and reduce latency",
            "action": "persona:performance_optimizer",
            "outcome": "success",
        },
        {
            "task": "Optimize memory allocation in the batch processing pipeline",
            "action": "persona:performance_optimizer",
            "outcome": "success",
        },
        {
            "task": "Write unit tests for the authentication module",
            "action": "persona:test_designer",
            "outcome": "success",
        },
        {
            "task": "Generate edge case tests for the rate limiter",
            "action": "persona:test_designer",
            "outcome": "success",
        },
        {
            "task": "Review the refactored module for code quality issues",
            "action": "persona:code_reviewer",
            "outcome": "success",
        },
        {
            "task": "Analyze the CSV export data for statistical anomalies",
            "action": "persona:data_analyst",
            "outcome": "success",
        },
        {
            "task": "Tune the KV cache settings for the inference server",
            "action": "persona:inference_specialist",
            "outcome": "success",
        },
        {
            "task": "Compare benchmark scores across model configurations",
            "action": "persona:benchmark_analyst",
            "outcome": "success",
        },
        {
            "task": "Implement the finite element solver for the heat equation",
            "action": "persona:computational_physicist",
            "outcome": "success",
        },
        {
            "task": "Design the training pipeline for the reward model",
            "action": "persona:ai_engineer",
            "outcome": "success",
        },
        # Research & academic personas
        {
            "task": "Design an experiment to measure speculative decoding quality impact",
            "action": "persona:research_architect",
            "outcome": "success",
        },
        {
            "task": "Write up the findings from the MoE expert reduction study",
            "action": "persona:research_writer",
            "outcome": "success",
        },
        {
            "task": "Summarize the meeting notes and extract action items",
            "action": "persona:secretary",
            "outcome": "success",
        },
        {
            "task": "Review the literature on transformer attention mechanisms",
            "action": "persona:research_analyst",
            "outcome": "success",
        },
        {
            "task": "Derive the partition function for the Ising model",
            "action": "persona:theoretical_physicist",
            "outcome": "success",
        },
        {
            "task": "Analyze the epistemological implications of AI alignment",
            "action": "persona:philosopher",
            "outcome": "success",
        },
        {
            "task": "Contextualize the development of computing within Cold War history",
            "action": "persona:academic_historian",
            "outcome": "success",
        },
        # Practical persona
        {
            "task": "Help me configure the RAID array and UPS monitoring",
            "action": "persona:hardware_specialist",
            "outcome": "success",
        },
    ]


def _get_routing_seeds() -> list[dict]:
    """Return mode-annotated routing seed examples.

    These teach MemRL which execution mode (direct/repl) works best for
    which task types, bootstrapping the mode selection before enough
    real task outcomes accumulate.

    2026-05-25 — react mode was unified into repl (REPL is a superset of
    direct+react with structured_mode for one-tool-per-turn execution).
    Five legacy "frontdoor:react" seeds were rewritten to "frontdoor:repl"
    so tool-needing queries land on the unified REPL surface. Without
    this fix the learned router replayed react-tagged seeds and emitted
    `chosen_action="frontdoor:react"` in routing decisions even though
    no live code path actually serves react anymore.

    Returns:
        List of routing seed dictionaries.
    """
    return [
        # Direct mode seeds — instruction following, reasoning, formatting
        {
            "task": "Solve this logic puzzle about truth-tellers and liars",
            "task_type": "reasoning",
            "action": "frontdoor:direct",
            "mode": "direct",
            "outcome": "success",
        },
        {
            "task": "Reformat this text as a numbered list with exactly 5 items",
            "task_type": "formatting",
            "action": "frontdoor:direct",
            "mode": "direct",
            "outcome": "success",
        },
        {
            "task": "Write a haiku about autumn",
            "task_type": "creative",
            "action": "frontdoor:direct",
            "mode": "direct",
            "outcome": "success",
        },
        {
            "task": "Explain the difference between TCP and UDP",
            "task_type": "knowledge",
            "action": "frontdoor:direct",
            "mode": "direct",
            "outcome": "success",
        },
        {
            "task": "Prove that the square root of 2 is irrational",
            "task_type": "math",
            "action": "frontdoor:direct",
            "mode": "direct",
            "outcome": "success",
        },
        {
            "task": "Generate JSON output with user name and age fields",
            "task_type": "formatting",
            "action": "frontdoor:direct",
            "mode": "direct",
            "outcome": "success",
        },
        # Tool-needing queries — REPL with structured_mode handles these
        # (the legacy "react" mode was a subset of REPL; unified 2026-05-25).
        {
            "task": "Search for recent papers on transformer architectures",
            "task_type": "research",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "What is today's date?",
            "task_type": "factual",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "Calculate the compound interest on $10000 at 5% for 10 years",
            "task_type": "math",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "Look up the Wikipedia article about quantum entanglement",
            "task_type": "research",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "Search arXiv for papers about reinforcement learning from human feedback",
            "task_type": "research",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        # REPL mode seeds — file exploration, large context, code execution
        {
            "task": "Read the configuration file and summarize its settings",
            "task_type": "file_exploration",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "List all Python files in the source directory",
            "task_type": "file_exploration",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "Summarize this 50-page document about climate change",
            "task_type": "ingest",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "Find all functions that handle error cases in the codebase",
            "task_type": "code_exploration",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "Execute the benchmark script and report the results",
            "task_type": "execution",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
        {
            "task": "Grep for all TODO comments in the project",
            "task_type": "code_exploration",
            "action": "frontdoor:repl",
            "mode": "repl",
            "outcome": "success",
        },
    ]


def _build_seed_records() -> list[dict]:
    """Build canonical seed records from REPL, routing, and persona sources."""
    records: list[dict] = []

    for seed in load_seeds():
        task = seed["task"]
        category = seed.get("category", "unknown")
        records.append(
            {
                "task": task,
                "action": seed["code"],
                "action_type": "exploration",
                "context": {
                    "task_description": task,
                    "category": category,
                    "tools_used": seed.get("tools_used", []),
                    "is_seed": True,
                },
                "category": category,
                "initial_q": 0.9,
            }
        )

    for rseed in _get_routing_seeds():
        records.append(
            {
                "task": rseed["task"],
                "action": rseed["action"],
                "action_type": "routing",
                "context": {
                    "task_description": rseed["task"],
                    "task_type": rseed["task_type"],
                    "category": "routing",
                    "mode": rseed["mode"],
                    "is_seed": True,
                },
                "category": "routing",
                "outcome": rseed.get("outcome", "success"),
                "initial_q": 0.85,
            }
        )

    for pseed in _get_persona_seeds():
        records.append(
            {
                "task": pseed["task"],
                "action": pseed["action"],
                "action_type": "persona",
                "context": {
                    "task_description": pseed["task"],
                    "category": "persona",
                    "is_seed": True,
                },
                "category": "persona",
                "outcome": pseed.get("outcome", "success"),
                "initial_q": 0.85,
            }
        )

    return records


def _existing_seed_keys(store: EpisodicStore) -> set[tuple[str, str, str]]:
    """Read existing canonical seed identities from SQLite metadata."""
    if not store.sqlite_path.exists():
        return set()

    keys: set[tuple[str, str, str]] = set()
    with sqlite3.connect(store.sqlite_path) as conn:
        rows = conn.execute(
            "SELECT action, action_type, context FROM memories WHERE context LIKE ?",
            ('%"is_seed"%',),
        ).fetchall()

    for action, action_type, context_json in rows:
        try:
            context = json.loads(context_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if context.get("is_seed") is not True:
            continue
        task_description = context.get("task_description")
        if isinstance(task_description, str):
            keys.add(_seed_key(action_type, action, task_description))

    return keys


def seed_memory(force: bool = False, init: bool = False) -> dict:
    """
    Load seed examples into episodic memory.

    Args:
        force: If True, clear existing memories first
        init: If True, add missing canonical seeds without clearing existing memories

    Returns:
        Stats dict with counts
    """
    store = EpisodicStore()
    embedder = TaskEmbedder()

    # Check current state
    current_stats = store.get_stats()
    current_count = current_stats.get("total_memories", 0)

    if current_count > 0 and not force and not init:
        print(f"Memory already has {current_count} entries.")
        print("Use --init to add missing seeds or --force to clear and reload all seeds.")
        return {"skipped": True, "existing": current_count}

    if force and current_count > 0:
        print(f"Clearing {current_count} existing memories...")
        # Clear by removing the SQLite database and FAISS files
        # FAISS backend uses storage_dir/episodic.db and storage_dir/embeddings.faiss
        sqlite_path = store.sqlite_path
        faiss_path = store.storage_dir / "embeddings.faiss"
        id_map_path = store.storage_dir / "id_map.npy"

        store.close()  # Close before deleting

        sqlite_path.unlink(missing_ok=True)
        faiss_path.unlink(missing_ok=True)
        id_map_path.unlink(missing_ok=True)

        store = EpisodicStore()  # Reinitialize

    seed_records = _build_seed_records()
    existing_seed_keys = _existing_seed_keys(store) if init and not force else set()
    if init and existing_seed_keys:
        print(
            f"Found {len(existing_seed_keys)} existing canonical seeds; "
            "loading only missing seeds..."
        )

    print(f"Loading {len(seed_records)} canonical seed examples...")

    stats = {
        "loaded": 0,
        "failed": 0,
        "skipped": 0,
        "by_category": {},
    }

    for i, seed in enumerate(seed_records):
        task = seed["task"]
        action = seed["action"]
        action_type = seed["action_type"]
        category = seed["category"]
        context = seed["context"]
        key = _seed_key(action_type, action, task)

        if key in existing_seed_keys:
            stats["skipped"] += 1
            continue

        try:
            # Generate embedding for the task description
            embedding = embedder.embed_text(task)

            # Store in episodic memory
            # Seeds go through the same record contract as live writes, so a
            # reseeded store is shaped identically to an organically grown one.
            record = record_from_legacy_context(context)
            record.source = "seed"
            if not record.objective:
                record.objective = task
            store.store(
                embedding=embedding,
                action=action,
                action_type=action_type,
                context=record.to_context(),
                outcome=seed.get("outcome", "success"),
                initial_q=seed["initial_q"],
            )

            stats["loaded"] += 1
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            existing_seed_keys.add(key)

            if stats["loaded"] > 0 and stats["loaded"] % 10 == 0:
                print(f"  Loaded {stats['loaded']}/{len(seed_records)} examples...")

        except Exception as e:
            print(f"  Failed to load '{task[:50]}...': {e}")
            stats["failed"] += 1

    # Flush FAISS index to disk before reporting stats
    if stats["loaded"] > 0:
        store.flush()
        store._embedding_store.save()

    print("\nSeeding complete!")
    print(f"  Loaded: {stats['loaded']}")
    print(f"  Skipped existing: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  By category: {stats['by_category']}")

    # Show final stats
    final_stats = store.get_stats()
    print("\nMemory stats:")
    print(f"  Total memories: {final_stats['total_memories']}")
    print(f"  FAISS embeddings: {store._embedding_store.count}")
    print(f"  Average Q-value: {final_stats['overall_avg_q']:.2f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Seed episodic memory with REPL examples")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Load missing canonical seeds without clearing existing memories",
    )
    parser.add_argument("--force", action="store_true", help="Clear existing memories first")
    args = parser.parse_args()

    seed_memory(force=args.force, init=args.init)


if __name__ == "__main__":
    main()
