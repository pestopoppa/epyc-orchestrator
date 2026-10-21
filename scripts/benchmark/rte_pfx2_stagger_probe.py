#!/usr/bin/env python3
"""RTE-PFX-2: hold-first prefix stagger probe for same-prefix fan-out.

Arms, per trial, on ONE served role (one llama-server, cache_prompt=true, no id_slot pin):
  simultaneous   N same-prefix requests released together (what parallel_dispatch does)
  hold_first_tok request 0 first; siblings 1..N-1 released when request 0's FIRST token arrives
  hold_first_end control: siblings released when request 0 COMPLETES

Every trial uses a fresh random salt at the very front of the prefix, so no earlier trial or
arm can have warmed it. Arms are interleaved (A,B,C,A,B,C,...) and the order is fixed by --seed.
Per request it records, from the server's own timings: prompt_n (uncached tokens evaluated),
cache_n (tokens served from cache), prompt_ms, and the client TTFT; and the slot-selection
evidence: response id_slot when the server reports it, plus a /slots snapshot taken WHILE the
siblings are in flight (a sample after the run proves nothing).

Slot-selection check (the point of the probe): a sibling lands on the cached prefix only if the
server picks the slot that holds it. A busy slot cannot be picked, so under hold_first_tok the
siblings may land on EMPTY slots and re-evaluate the whole prefix. The verdict per sibling is
cache_n >= expected_prefix_tokens * HIT_FRACTION, with expected_prefix_tokens taken from request
0's own n_prompt - tokens of the unique tail (measured via /tokenize), never assumed.

Refuses unless WINDOW_CONFIRMED=1 and run under `region-lock` (a TIMING run). Needs -np >= N.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time

HIT_FRACTION = 0.9


def summarise(rows: list[dict], expected_prefix_tokens: int) -> dict:
    """Pure analysis of one trial's request rows (unit-tested without a server)."""
    sib = [r for r in rows if r["idx"] > 0]
    landed = [r for r in sib if r["cache_n"] >= HIT_FRACTION * expected_prefix_tokens]
    return {"n_siblings": len(sib), "siblings_on_cached_prefix": len(landed),
            "sibling_uncached_prompt_tokens": [r["prompt_n"] for r in sib],
            "sibling_ttft_ms": [r["ttft_ms"] for r in sib],
            "first_ttft_ms": rows[0]["ttft_ms"] if rows else None,
            "total_uncached_prompt_tokens": sum(r["prompt_n"] for r in rows),
            "slot_ids": [r.get("id_slot") for r in rows]}


def _one(client, url, idx, prompt, n_predict, first_tok: threading.Event | None, out, t_ref):
    t0 = time.monotonic()
    ttft = None
    final = {}
    with client.stream("POST", f"{url}/completion", json={
            "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
            "cache_prompt": True, "stream": True}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ttft is None and ev.get("content"):
                ttft = (time.monotonic() - t0) * 1000.0
                if first_tok is not None:
                    first_tok.set()
            if ev.get("stop"):
                final = ev
    tm = final.get("timings") or {}
    out[idx] = {"idx": idx, "ttft_ms": ttft, "prompt_n": int(tm.get("prompt_n", -1)),
                "cache_n": int(tm.get("cache_n", -1)), "prompt_ms": tm.get("prompt_ms"),
                "id_slot": final.get("id_slot"), "start_offset_ms": (t0 - t_ref) * 1000.0}


def run_trial(client, url, arm, n, prefix_text, n_predict, rng):
    salt = f"[trial-{rng.getrandbits(48):012x}]\n"
    prompts = [salt + prefix_text + f"\n\nQuestion {i}: say the number {i}." for i in range(n)]
    out: dict[int, dict] = {}
    first = threading.Event()
    t_ref = time.monotonic()
    th = [threading.Thread(target=_one, args=(client, url, i, prompts[i], n_predict,
                                              first if i == 0 else None, out, t_ref))
          for i in range(n)]
    snapshot: dict = {}
    if arm == "simultaneous":
        for t in th: t.start()
    else:
        th[0].start()
        if arm == "hold_first_tok":
            first.wait(timeout=300)
        else:
            th[0].join()
        for t in th[1:]: t.start()
    time.sleep(0.05)
    try:   # sampled DURING the siblings' flight
        snapshot = {"slots": client.get(f"{url}/slots", timeout=5).json()}
    except Exception as exc:  # noqa: BLE001 - evidence, not control flow
        snapshot = {"error": str(exc)}
    for t in th: t.join()
    return [out[i] for i in range(n)], snapshot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--trials", type=int, default=5, help="trials PER ARM")
    ap.add_argument("--prefix-file", required=True, help="shared prefix text (declare its source)")
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        print("REFUSED: timing probe; needs a window under region-lock (set WINDOW_CONFIRMED=1)",
              file=sys.stderr)
        return 3
    import httpx
    rng = random.Random(a.seed)
    prefix = open(a.prefix_file).read()
    arms = ["simultaneous", "hold_first_tok", "hold_first_end"]
    results = []
    with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
        props = client.get(f"{a.server}/props").json()
        n_slots = props.get("total_slots")
        if n_slots is not None and n_slots < a.n:
            print(f"REFUSED: server has {n_slots} slots < N={a.n}", file=sys.stderr)
            return 4
        expected = len(client.post(f"{a.server}/tokenize", json={"content": prefix}).json()["tokens"])
        for trial in range(a.trials):
            for arm in arms:               # interleaved, fixed order
                rows, snap = run_trial(client, a.server, arm, a.n, prefix, a.n_predict, rng)
                results.append({"trial": trial, "arm": arm, "rows": rows, "slots_during": snap,
                                "summary": summarise(rows, expected)})
    json.dump({"schema": "epyc.rte_pfx2.v1", "server": a.server, "n": a.n,
               "props": {k: props.get(k) for k in ("total_slots", "model_path", "build_info")},
               "expected_prefix_tokens": expected, "hit_fraction": HIT_FRACTION,
               "results": results}, open(a.output, "w"), indent=1)
    for arm in arms:
        s = [r["summary"]["total_uncached_prompt_tokens"] for r in results if r["arm"] == arm]
        print(arm, "median total uncached prompt tokens", statistics.median(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
