#!/usr/bin/env python3
"""Wipe the episodic memory store and reseed from canonical seed_loader.

Used when legacy pollution slips into stored memories — e.g. 2026-05-25
when the routing-decision telemetry was emitting `chosen_action="frontdoor:react"`
because 5 seed entries hardcoded a now-unified mode. Rewriting the seed
file alone is not enough: existing memories carrying the legacy mode
keep replaying via nearest-neighbor retrieval, so the store must be
cleared and reloaded from the corrected seeds.

Safety:
  - Refuses to run while the orchestrator API is up on port 8000
    (the orchestrator holds the SQLite + FAISS files open via a long-
    lived EpisodicStore singleton; unlink would race or be silently
    ignored). Stop the orchestrator first:
        python3 scripts/server/orchestrator_stack.py stop orchestrator
  - Backs up the SQLite + FAISS + id_map files before deleting.
  - Prints the seeded counts after reload.

Usage:
    python3 scripts/maintenance/wipe_episodic_memory.py [--force]

Without --force this just prints what *would* be cleared/reseeded.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import time
from pathlib import Path


def _orchestrator_listening(port: int = 8000) -> bool:
    """Return True if a process is listening on the orchestrator port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Wipe + reseed the episodic memory store.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually wipe + reseed. Without this flag, runs a dry-check.",
    )
    parser.add_argument(
        "--skip-listen-check",
        action="store_true",
        help="Skip the orchestrator-up safety check. Only use if you know "
             "the listening process is not the orchestrator.",
    )
    args = parser.parse_args()

    # Make the orchestration module importable
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from orchestration.repl_memory import EpisodicStore
    from orchestration.repl_memory.seed_loader import seed_memory

    if _orchestrator_listening() and not args.skip_listen_check:
        print(
            "ERROR: orchestrator API is listening on :8000 — it has the "
            "episodic store SQLite + FAISS files held open. Stop it first:\n"
            "    python3 scripts/server/orchestrator_stack.py stop orchestrator\n"
            "Then re-run this script. (Or pass --skip-listen-check to override.)"
        )
        return 2

    store = EpisodicStore()
    stats = store.get_stats()
    current = int(stats.get("total_memories", 0))
    sqlite_path = store.sqlite_path
    storage_dir = store.storage_dir
    print(f"Current store: {current} memories at {storage_dir}")
    print(f"  sqlite: {sqlite_path}")
    for f in ("embeddings.faiss", "id_map.npy"):
        p = storage_dir / f
        if p.exists():
            print(f"  {f}: {p.stat().st_size:,} bytes")

    if not args.force:
        print()
        print("DRY RUN — pass --force to actually wipe + reseed.")
        store.close()
        return 0

    # Back up the existing files so the operator can recover if reseed fails.
    backup_stamp = time.strftime("%Y%m%d_%H%M%S")
    backed_up: list[Path] = []
    for fname in ("episodic.db", "embeddings.faiss", "id_map.npy"):
        src = storage_dir / fname
        if src.exists():
            dst = src.with_name(f"{fname}.wipe-bak-{backup_stamp}")
            shutil.copy2(src, dst)
            backed_up.append(dst)
            print(f"  backed up → {dst.name}")

    store.close()
    print("Closed open store handle.")

    print("Running seed_memory(force=True) — this clears + reseeds...")
    result = seed_memory(force=True)
    print()
    print(f"Reseed complete: {result}")

    # Verify with a fresh handle
    fresh = EpisodicStore()
    fresh_stats = fresh.get_stats()
    fresh_count = int(fresh_stats.get("total_memories", 0))
    fresh.close()
    print(f"Post-reseed: {fresh_count} memories ({fresh_count - current:+d} delta).")

    if backed_up:
        print()
        print("Backups left in place; remove with:")
        for p in backed_up:
            print(f"    rm {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
