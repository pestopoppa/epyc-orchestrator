#!/usr/bin/env python3
"""Standalone dual-half scaling probe (bypasses the matrix topology gate).

The dual-half experiment adds a complementary half instance (e.g. frontdoor Half1), which
changes the NUMA_CONFIG topology_hash → the closed-world `bench-within-role` correctly refuses
(matrix stale). This probe measures the specific Half0∥Half1 pair directly via the same
_http_bench primitive, so we can decide whether dual-half disjoint concurrency is worth wiring
into production before re-certifying the whole matrix.

ratio = parallel_aggregate_tps / sequential_aggregate_tps  (same semantics as the matrix).
~2.0 = perfect disjoint scaling; >=~1.5 = worth it; ~1.0 = no benefit (revert).

Usage: python3 scripts/benchmark/dualhalf_probe.py --pa 8070 --pb 8073 --label frontdoor --samples 3
ALONE on a quiet host (feedback_no_concurrent_inference).
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.server.contention_matrix import _http_bench  # noqa: E402


def _tps(port: int) -> float:
    return _http_bench(port, safe_sampling=True)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", type=int, required=True, help="Half0 port")
    ap.add_argument("--pb", type=int, required=True, help="Half1 port")
    ap.add_argument("--label", default="role")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    ratios: list[float] = []
    for s in range(args.samples):
        a = _tps(args.pa)
        b = _tps(args.pb)
        seq = a + b  # sequential aggregate (each solo)
        with ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(_http_bench, args.pa, safe_sampling=True)
            fb = ex.submit(_http_bench, args.pb, safe_sampling=True)
            pa, pb = fa.result()[0], fb.result()[0]
        par = pa + pb  # parallel aggregate (concurrent)
        ratio = par / seq if seq else 0.0
        ratios.append(ratio)
        print(f"  sample {s}: solo({a:.1f}+{b:.1f}={seq:.1f}) par({pa:.1f}+{pb:.1f}={par:.1f}) ratio={ratio:.3f}")

    mean = st.mean(ratios)
    cv = (st.pstdev(ratios) / mean) if mean else 0.0
    verdict = "allow" if mean >= 1.0 else ("borderline" if mean >= 0.85 else "block")
    worth = "WORTH WIRING" if mean >= 1.5 else ("marginal" if mean >= 1.0 else "NOT WORTH (revert)")
    print(f"\n  {args.label} Half0(:{args.pa}) ∥ Half1(:{args.pb}): mean ratio={mean:.3f} cv={cv:.3f} "
          f"verdict={verdict} → {worth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
