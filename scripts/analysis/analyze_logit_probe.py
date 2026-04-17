#!/usr/bin/env python3
"""Analyze logit probe data for Learned Routing Controller P1.5.

Reads first-token log-probabilities captured from frontdoor inference
(when ORCHESTRATOR_LOGIT_PROBE=1) and trains a linear probe to predict
routing decisions from token probabilities.

Usage:
    python3 scripts/analysis/analyze_logit_probe.py [--min-samples 100]

Data file: data/logit_probe.jsonl (written by llama_server.py)
Schema: {timestamp, prompt_hash, prompt_len, first_token, top_k_probs: [{tok, prob}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "logit_probe.jsonl"


def load_probe_data(path: Path = DATA_PATH) -> list[dict]:
    """Load logit probe entries from JSONL file."""
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def summarize(entries: list[dict]) -> dict:
    """Compute summary statistics for collected probe data."""
    if not entries:
        return {"count": 0, "status": "no data"}

    prompt_lens = [e.get("prompt_len", 0) for e in entries]
    n_probs = [len(e.get("top_k_probs", [])) for e in entries]
    unique_first_tokens = set(e.get("first_token", "") for e in entries)

    return {
        "count": len(entries),
        "prompt_len_mean": sum(prompt_lens) / len(prompt_lens),
        "prompt_len_range": (min(prompt_lens), max(prompt_lens)),
        "avg_probs_per_entry": sum(n_probs) / len(n_probs),
        "unique_first_tokens": len(unique_first_tokens),
        "top_first_tokens": _top_n(
            [e.get("first_token", "") for e in entries], 10
        ),
    }


def _top_n(items: list[str], n: int) -> list[tuple[str, int]]:
    """Return top-n most common items with counts."""
    from collections import Counter
    return Counter(items).most_common(n)


def main():
    parser = argparse.ArgumentParser(description="Analyze logit probe data")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="Minimum samples needed for probe training (default 100)")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH),
                        help="Path to logit_probe.jsonl")
    args = parser.parse_args()

    path = Path(args.data_path)
    entries = load_probe_data(path)

    print(f"=== Logit Probe Analysis ===")
    print(f"Data file: {path}")
    print(f"Entries: {len(entries)}")

    if not entries:
        print("\nNo data collected yet.")
        print("Enable with: ORCHESTRATOR_LOGIT_PROBE=1")
        print("Data accumulates from frontdoor inference requests.")
        return 1

    stats = summarize(entries)
    print(f"\nPrompt length: mean={stats['prompt_len_mean']:.0f}, "
          f"range={stats['prompt_len_range']}")
    print(f"Avg probs per entry: {stats['avg_probs_per_entry']:.0f}")
    print(f"Unique first tokens: {stats['unique_first_tokens']}")
    print(f"\nTop first tokens:")
    for tok, count in stats["top_first_tokens"]:
        print(f"  {tok!r}: {count}")

    if len(entries) < args.min_samples:
        print(f"\nNeed {args.min_samples - len(entries)} more samples before probe training.")
        print("P1.5 decision gate: >= 80% accuracy → proceed to Phase 2.")
        return 0

    print(f"\n{len(entries)} samples available — ready for P1.5 linear probe training.")
    print("Next step: train_routing_probe.py (not yet implemented)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
