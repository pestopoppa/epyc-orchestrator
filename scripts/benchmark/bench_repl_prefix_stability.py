#!/usr/bin/env python3
from __future__ import annotations

"""Measure llama-server KV-cache prefix reuse across REPL turns.

RTE-Prefix A/B harness: drives the REAL frontdoor REPL prompt sequence
(build_root_lm_prompt, the same builder graph/helpers.py uses per turn)
through llama-server /completion with cache_prompt=true and a pinned slot,
then reports the true cache-hit count per turn (timings.cache_n /
n_prompt_tokens_cache — NOT tokens_cached, which is total slot occupancy).

Usage:
    python scripts/benchmark/bench_repl_prefix_stability.py \
        --server http://localhost:8080 --turns 6 --output /tmp/repl_ab.json

    # Default: runs BOTH orders (legacy and prefix-stable) and prints a
    # comparison table. The flag is toggled programmatically per run, so no
    # env vars are needed:
    python scripts/benchmark/bench_repl_prefix_stability.py --server http://localhost:8080

    # Single-order run (for a dedicated compute window):
    python scripts/benchmark/bench_repl_prefix_stability.py \
        --server http://localhost:8080 --order legacy --output /tmp/legacy.json
    python scripts/benchmark/bench_repl_prefix_stability.py \
        --server http://localhost:8080 --order stable --output /tmp/stable.json

Requirements:
    - llama-server (production-consolidated-v9) with -np 1 --n-cpu-moe 0
      (single slot so the KV cache is NOT evicted between turns)
    - Orchestrator repo importable from cwd (src/ on path)

Output per turn: total prompt tokens, cache-hit tokens, hit ratio, and
prompt eval ms. Hit ratio uses n_prompt_tokens_cache from the /completion
timings (cache_n) — the count of tokens served from the KV cache.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_turn_prompt(
    turn: int,
    last_output: str,
    last_error: str,
    corpus_context: str,
) -> str:
    """Build the exact per-turn prompt the frontdoor REPL produces.

    Mirrors graph/helpers.py:733-741 (state=Turn N, corpus only on turn 0,
    last output/error spillover) through the same builder.
    """
    from src.prompt_builders.builder import build_root_lm_prompt

    return build_root_lm_prompt(
        state="ready",
        original_prompt="Implement a prefix-cache-friendly REPL loop in Python.",
        last_output=last_output,
        last_error=last_error,
        turn=turn,
        corpus_context=corpus_context,
    )


def run_sequence(
    server_url: str,
    turns: int,
    n_predict: int,
    corpus_context: str,
) -> list[dict[str, Any]]:
    """Send a REPL turn sequence to a live llama-server, sampling cache reuse.

    Uses raw /completion (the same endpoint the frontdoor REPL uses — the
    REPL bypasses PrefixRouter by default) with cache_prompt=true and a pinned
    id_slot=0 so the KV cache persists across turns.
    """
    import httpx

    url = server_url.rstrip("/")
    rows: list[dict[str, Any]] = []
    last_output = ""
    last_error = ""

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0, read=120.0, write=120.0)) as client:
        for turn in range(turns):
            prompt = _build_turn_prompt(turn, last_output, last_error, corpus_context)
            payload = {
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0.0,
                "cache_prompt": True,
                "id_slot": 0,
                "stop": ["\n```\n"],
            }
            t0 = time.time()
            resp = client.post(f"{url}/completion", json=payload)
            elapsed_ms = (time.time() - t0) * 1000.0
            resp.raise_for_status()
            data = resp.json()

            timings = data.get("timings", {}) or {}
            # total = full prompt size (top-level tokens_evaluated =
            # n_prompt_tokens, server-task.cpp:379); processed = timings.prompt_n
            # (tokens actually evaluated, server-context.cpp:513); cached =
            # timings.cache_n = n_prompt_tokens_cache (server-context.cpp:511).
            total = int(data.get("tokens_evaluated", 0) or 0)
            cached = int(timings.get("cache_n", 0))
            processed = int(timings.get("prompt_n", 0))
            rows.append(
                {
                    "turn": turn,
                    "prompt_chars": len(prompt),
                    "prompt_tokens_total": total,
                    "prompt_tokens_cache_hit": cached,
                    "prompt_tokens_processed": processed,
                    "hit_ratio": (cached / total) if total > 0 else 0.0,
                    "prompt_eval_ms": float(timings.get("prompt_ms", 0.0) or 0.0),
                    "http_ms": round(elapsed_ms, 1),
                }
            )

            # Feed this turn's generated code into the next turn's last_output
            # (mirrors the REPL spillover).
            last_output = (data.get("content", "") or "")[:2000]
            if last_output:
                last_output = "```python\n" + last_output + "\n```"

    return rows


def _order_from_flags() -> str:
    from src.features import features

    return "stable" if features().prefix_stable_order else "legacy"


def run_with_order(server_url: str, turns: int, n_predict: int, corpus_context: str, order: str):
    """Run a sequence with a specific prompt order, returning rows + summary."""
    from src.features import Features, reset_features, set_features

    set_features(Features(prefix_stable_order=(order == "stable")))
    try:
        rows = run_sequence(server_url, turns, n_predict, corpus_context)
    finally:
        reset_features()

    total_tokens = sum(r["prompt_tokens_total"] for r in rows)
    total_cached = sum(r["prompt_tokens_cache_hit"] for r in rows)
    evals = [r for r in rows if r["prompt_tokens_processed"] > 0]
    return {
        "order": order,
        "turns": rows,
        "total_prompt_tokens": total_tokens,
        "total_cache_hits": total_cached,
        "overall_hit_ratio": (total_cached / total_tokens) if total_tokens else 0.0,
        "turn2_hit_ratio": rows[1]["hit_ratio"] if len(rows) > 1 else 0.0,
        "avg_prompt_eval_ms": (
            sum(r["prompt_eval_ms"] for r in evals) / len(evals) if evals else 0.0
        ),
    }


def _print_summary(result: dict[str, Any]) -> None:
    print(f"\nOrder: {result['order']}")
    print(f"  {'Turn':<5}{'Total':>8}{'Cached':>8}{'Hit%':>8}{'EvalMs':>10}")
    for r in result["turns"]:
        print(
            f"  {r['turn']:<5}{r['prompt_tokens_total']:>8}"
            f"{r['prompt_tokens_cache_hit']:>8}{r['hit_ratio'] * 100:>7.1f}%"
            f"{r['prompt_eval_ms']:>10.1f}"
        )
    print(
        f"  Overall hit ratio: {result['overall_hit_ratio'] * 100:.1f}% "
        f"(turn-2: {result['turn2_hit_ratio'] * 100:.1f}%)"
    )
    print(f"  Avg prompt eval (non-cached turns): {result['avg_prompt_eval_ms']:.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure KV-cache prefix reuse across REPL turns (RTE-Prefix A/B)"
    )
    parser.add_argument("--server", default="http://localhost:8080", help="llama-server URL")
    parser.add_argument("--turns", type=int, default=6, help="REPL turns to send")
    parser.add_argument("--n-predict", type=int, default=8, help="Tokens to generate per turn")
    parser.add_argument(
        "--corpus",
        default="",
        help="Corpus-context text (fed only on turn 0, mirroring the REPL)",
    )
    parser.add_argument(
        "--order",
        choices=["both", "legacy", "stable"],
        default="both",
        help="Which prompt order(s) to measure (default: both)",
    )
    parser.add_argument("--output", type=str, help="Write JSON result to this path")
    args = parser.parse_args()

    orders = ["legacy", "stable"] if args.order == "both" else [args.order]
    results = []
    for order in orders:
        print(f"\n=== Running order: {order} ===")
        try:
            result = run_with_order(
                args.server, args.turns, args.n_predict, args.corpus, order
            )
        except Exception as e:  # noqa: BLE001 — report and continue the other order
            print(f"ERROR running order {order}: {e}", file=sys.stderr)
            results.append({"order": order, "error": str(e)})
            continue
        _print_summary(result)
        results.append(result)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
