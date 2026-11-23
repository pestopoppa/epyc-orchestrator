"""20-cut synthetic smoke for the KV evict-cost probe (box KV-EVICT-COST-1).

READ-ONLY: the only server traffic is one GET /slots. Run it under a row
against a live server; reports k/20 and exits 0 iff k >= PASS_MIN (19).

Each synthetic cut (deterministic, seeded) is checked against invariants the
probe must satisfy on real server state:
  - p* <= cut and p* is a member of the server checkpoint set (or None)
  - re-prefilled == T' - p* (T' - 0 when no checkpoint)
  - seconds == a + b * cells, finite, >= 0
  - alignment class consistent with the gap
  - hybrid servers: every cut classifies cannot_shift with no prediction
Cuts are spread so that exact, aligned, misaligned and pre-first-checkpoint
cases are all drawn when the checkpoint set allows it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import Any, Sequence

from scripts.autopilot.kv_evict_cost import (
    ALIGN_ALIGNED, ALIGN_CANNOT_SHIFT, ALIGN_EXACT, ALIGN_MISALIGNED,
    ALIGN_NO_CHECKPOINT, Calibration, fetch_slot_state, is_hybrid_arch,
    predict_evict_cost,
)

N_CUTS = 20
PASS_MIN = 19


def synth_cuts(t_prime: int, checkpoints: Sequence[int], n: int = N_CUTS, seed: int = 1123,
               align_tol: int = 64) -> list[int]:
    rng = random.Random(seed)
    cps = sorted(c for c in set(checkpoints) if 0 <= c <= t_prime)
    cuts: list[int] = []
    for c in cps:  # exact, aligned, misaligned per checkpoint (round-robin)
        cuts.append(c)
        cuts.append(min(t_prime, c + max(1, align_tol // 2)))
        cuts.append(min(t_prime, c + align_tol * 4 + 1))
    if cps and cps[0] > 0:
        cuts.append(rng.randrange(0, cps[0]))  # pre-first-checkpoint
    cuts = cuts[:n]
    while len(cuts) < n:
        cuts.append(rng.randrange(0, t_prime + 1))
    rng.shuffle(cuts)
    return cuts[:n]


def check_cut(t_prime: int, cut: int, cps: Sequence[int], cal: Calibration, hybrid: bool,
              align_tol: int) -> tuple[bool, str]:
    r = predict_evict_cost(t_prime, cut, cps, cal, hybrid=hybrid, align_tol=align_tol)
    if hybrid:
        ok = r.alignment == ALIGN_CANNOT_SHIFT and r.reprefilled_tokens is None
        return ok, r.alignment
    cpset = set(cps)
    if r.p_star is not None and (r.p_star > cut or r.p_star not in cpset):
        return False, "p*_invalid"
    if any(c <= cut and (r.p_star is None or c > r.p_star) for c in cpset):
        return False, "p*_not_nearest"
    if r.reprefilled_tokens != t_prime - (r.p_star or 0):
        return False, "reprefill_mismatch"
    if r.seconds is None or not math.isfinite(r.seconds) or r.seconds < 0:
        return False, "seconds_invalid"
    if abs(r.seconds - cal.seconds(r.reprefilled_tokens)) > 1e-9:
        return False, "seconds_mismatch"
    gap = r.cut_to_checkpoint_gap
    want = (ALIGN_NO_CHECKPOINT if r.p_star is None else ALIGN_EXACT if gap == 0
            else ALIGN_ALIGNED if gap <= align_tol else ALIGN_MISALIGNED)
    return r.alignment == want, r.alignment


def run_smoke(state: dict[str, Any], cal: Calibration, align_tol: int = 64,
              hybrid: bool | None = None) -> dict[str, Any]:
    t = int(state["t_prime"])
    cps = list(state.get("checkpoints") or [])
    hyb = is_hybrid_arch(state.get("arch")) if hybrid is None else hybrid
    cuts = synth_cuts(t, cps, align_tol=align_tol)
    rows, k = [], 0
    for cut in cuts:
        ok, cls = check_cut(t, cut, cps, cal, hyb, align_tol)
        k += ok
        rows.append({"cut": cut, "ok": ok, "class": cls})
    return {"k": k, "n": len(cuts), "pass": k >= PASS_MIN and len(cuts) == N_CUTS,
            "t_prime": t, "n_checkpoints": len(cps), "hybrid": hyb, "rows": rows}


class HttpClient:
    """Minimal llama-server client for --measure (POST /tokenize, /completion; GET /slots)."""

    def __init__(self, port: int, slot: int = 0, timeout: float = 600.0):
        self.base, self.slot, self.timeout = f"http://localhost:{port}", slot, timeout

    def tokenize(self, text: str) -> list[int]:
        import httpx
        r = httpx.post(f"{self.base}/tokenize", json={"content": text}, timeout=self.timeout)
        r.raise_for_status()
        return list(r.json()["tokens"])

    def completion(self, tokens: list[int]) -> int:
        """Run a 1-token completion on the slot; return timings.prompt_n."""
        import httpx
        r = httpx.post(f"{self.base}/completion", timeout=self.timeout, json={
            "prompt": tokens, "n_predict": 1, "cache_prompt": True,
            "id_slot": self.slot, "temperature": 0.0})
        r.raise_for_status()
        return int(r.json()["timings"]["prompt_n"])

    def slot_state(self) -> dict[str, Any]:
        return fetch_slot_state(int(self.base.rsplit(":", 1)[1]), self.slot)


def run_measured(client: Any, base_tokens: list[int], cal: Calibration, n_vocab: int,
                 cuts: Sequence[int], align_tol: int = 64) -> dict[str, Any]:
    """Compare predicted vs MEASURED re-prefilled tokens (timings.prompt_n).

    Per cut: refill the slot with base_tokens (creates checkpoints), read the
    checkpoint set, then send base[:cut] + a diverging tail of equal total
    length; the server must re-prefill T' - p*. Exact equality is the pass
    criterion. This DOES run inference: only call it under a row.
    """
    t = len(base_tokens)
    rows, k = [], 0
    for cut in cuts:
        client.completion(base_tokens)
        st = client.slot_state()
        hyb = is_hybrid_arch(st.get("arch"))
        pred = predict_evict_cost(t, cut, st["checkpoints"], cal, hybrid=hyb, align_tol=align_tol)
        tail = [(x + 1) % n_vocab for x in base_tokens[cut:]]
        measured = client.completion(list(base_tokens[:cut]) + tail)
        if hyb:
            ok = pred.alignment == ALIGN_CANNOT_SHIFT
        else:
            ok = measured == pred.reprefilled_tokens
        k += ok
        rows.append({"cut": cut, "ok": ok, "predicted": pred.reprefilled_tokens,
                     "measured": measured, "class": pred.alignment})
    return {"k": k, "n": len(rows), "pass": k >= PASS_MIN and len(rows) == N_CUTS,
            "t_prime": t, "rows": rows}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", type=int, default=0, metavar="T_PRIME",
                    help="INFERENCE: build a T_PRIME-token base prompt and compare predicted vs "
                         "measured timings.prompt_n over the 20 cuts (default: read-only invariants)")
    ap.add_argument("--n-vocab", type=int, default=32000)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--b", type=float, required=True)
    ap.add_argument("--checkpoints", default="", help="override positions, comma-separated")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--align-tol", type=int, default=64)
    ap.add_argument("--out", default=None, help="write JSON result here")
    args = ap.parse_args(argv)

    cal = Calibration(args.a, args.b)
    if args.measure:
        client = HttpClient(args.port, args.slot)
        words = " ".join(f"w{i}" for i in range(args.measure))
        base = client.tokenize(words)[: args.measure]
        client.completion(base)
        st = client.slot_state()
        res = run_measured(client, base, cal, args.n_vocab,
                           synth_cuts(len(base), st["checkpoints"], align_tol=args.align_tol),
                           args.align_tol)
    else:
        state = fetch_slot_state(args.port, args.slot)
        if args.checkpoints.strip():
            state["checkpoints"] = [int(x) for x in args.checkpoints.split(",") if x.strip()]
        if args.arch:
            state["arch"] = args.arch
        res = run_smoke(state, cal, args.align_tol)
    text = json.dumps(res)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(f"KV-EVICT-COST-1 smoke: {res['k']}/{res['n']} {'PASS' if res['pass'] else 'FAIL'}")
    print(text)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
