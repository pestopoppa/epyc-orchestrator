"""Read-only llama-server KV evict-cost probe (box KV-EVICT-COST-1).

Given a slot's token count T', a cut position and the server's context
checkpoint positions, predict what a mid-context eviction would cost:

    p*              = nearest checkpoint position <= cut   (0 if none)
    re-prefilled    = T' - p*
    seconds         = a + b * cells          (cells = re-prefilled tokens)

Alignment classes:
    exact       a checkpoint sits exactly at the cut (p* == cut)
    aligned     p* within ``align_tol`` tokens below the cut
    misaligned  a checkpoint exists below the cut but further than align_tol
    no_checkpoint  nothing at or before the cut: full re-prefill from 0

Hybrid GDN / recurrent models cannot position-shift their state
(llama-memory-recurrent.cpp seq_add/seq_rm refuse a partial range; see
``llama_memory_recurrent`` in the frozen tree, src/llama-memory-recurrent.cpp
around line 832 for the state I/O that only round-trips whole sequences).
They are classified ``cannot_shift`` and carry no prediction.

Nothing here writes to a server: the only I/O helper is ``fetch_slot_state``
(GET /slots), and every function accepts plain values so tests use fakes.
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

ALIGN_EXACT = "exact"
ALIGN_ALIGNED = "aligned"
ALIGN_MISALIGNED = "misaligned"
ALIGN_NO_CHECKPOINT = "no_checkpoint"
ALIGN_CANNOT_SHIFT = "cannot_shift"

CLASSES = (ALIGN_EXACT, ALIGN_ALIGNED, ALIGN_MISALIGNED, ALIGN_NO_CHECKPOINT, ALIGN_CANNOT_SHIFT)

DEFAULT_ALIGN_TOL = 64

# Architectures whose memory is recurrent/hybrid (cannot shift a partial range).
HYBRID_ARCH_MARKERS = (
    "qwen3next", "qwen35", "qwen3.5", "qwen3_5", "qwen38", "qwen3.8",
    "gdn", "gated_delta", "mamba", "rwkv", "jamba", "recurrent", "hybrid",
)


@dataclass(frozen=True)
class Calibration:
    """seconds = a + b * cells (cells = re-prefilled tokens)."""

    a: float
    b: float
    device: str = ""

    def seconds(self, cells: int) -> float:
        return self.a + self.b * max(0, int(cells))


@dataclass(frozen=True)
class EvictCost:
    t_prime: int
    cut: int
    p_star: int | None
    reprefilled_tokens: int | None
    seconds: float | None
    alignment: str
    cut_to_checkpoint_gap: int | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_hybrid_arch(arch: str | None) -> bool:
    if not arch:
        return False
    low = str(arch).strip().lower()
    return any(m in low for m in HYBRID_ARCH_MARKERS)


def nearest_checkpoint_at_or_before(cut: int, checkpoints: Iterable[int]) -> int | None:
    pts = sorted({int(c) for c in checkpoints if int(c) >= 0})
    i = bisect.bisect_right(pts, cut)
    return pts[i - 1] if i else None


def predict_evict_cost(
    t_prime: int,
    cut: int,
    checkpoints: Sequence[int],
    calibration: Calibration | None,
    *,
    hybrid: bool = False,
    align_tol: int = DEFAULT_ALIGN_TOL,
) -> EvictCost:
    if t_prime < 0 or cut < 0:
        raise ValueError("t_prime and cut must be >= 0")
    if cut > t_prime:
        raise ValueError(f"cut {cut} beyond slot token count {t_prime}")
    if hybrid:
        return EvictCost(t_prime, cut, None, None, None, ALIGN_CANNOT_SHIFT, None,
                         "hybrid/recurrent memory cannot shift a partial range")
    p = nearest_checkpoint_at_or_before(cut, checkpoints)
    p_eff = 0 if p is None else p
    cells = t_prime - p_eff
    secs = calibration.seconds(cells) if calibration is not None else None
    if p is None:
        align, gap = ALIGN_NO_CHECKPOINT, None
    else:
        gap = cut - p
        if gap == 0:
            align = ALIGN_EXACT
        elif gap <= align_tol:
            align = ALIGN_ALIGNED
        else:
            align = ALIGN_MISALIGNED
    return EvictCost(t_prime, cut, p, cells, secs, align, gap)


def load_calibration(spec: dict[str, Any]) -> Calibration:
    """Build a Calibration from a dict with numeric keys a and b."""
    a, b = float(spec["a"]), float(spec["b"])
    if b < 0 or a < 0:
        raise ValueError("calibration a and b must be >= 0")
    return Calibration(a=a, b=b, device=str(spec.get("device", "")))


def _to_int_list(raw: Any) -> list[int]:
    out: list[int] = []
    for item in raw or []:
        if isinstance(item, dict):
            for key in ("pos_max", "n_tokens", "pos", "position"):
                if key in item:
                    out.append(int(item[key]))
                    break
        else:
            out.append(int(item))
    return out


def parse_slot_state(slots_json: Any, slot_id: int = 0) -> dict[str, Any]:
    """Extract t_prime / checkpoints / arch from a /slots payload (pure)."""
    for slot in slots_json or []:
        if slot.get("id") == slot_id:
            t = slot.get("n_past", slot.get("n_tokens"))
            if t is None:
                nt = slot.get("next_token") or [{}]
                t = nt[0].get("n_decoded") if isinstance(nt, list) else None
            return {
                "t_prime": None if t is None else int(t),
                "checkpoints": _to_int_list(slot.get("checkpoints") or slot.get("ctx_checkpoints")),
                "arch": slot.get("model_arch") or slot.get("arch"),
            }
    raise KeyError(f"slot {slot_id} not in /slots payload")


def fetch_slot_state(port: int, slot_id: int = 0, timeout: float = 5.0,
                     http_get: Callable[..., Any] | None = None) -> dict[str, Any]:
    """GET /slots only (read-only)."""
    if http_get is None:
        import httpx
        http_get = httpx.get
    resp = http_get(f"http://localhost:{port}/slots", timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"/slots returned {resp.status_code}")
    return parse_slot_state(resp.json(), slot_id)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--t-prime", type=int, help="slot token count (else read from --port)")
    ap.add_argument("--cut", type=int, required=True)
    ap.add_argument("--checkpoints", default="", help="comma-separated positions")
    ap.add_argument("--a", type=float, help="calibration intercept (s)")
    ap.add_argument("--b", type=float, help="calibration slope (s per cell)")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--port", type=int)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--align-tol", type=int, default=DEFAULT_ALIGN_TOL)
    args = ap.parse_args(argv)

    t_prime, cps, arch = args.t_prime, _to_int_list(
        [x for x in args.checkpoints.split(",") if x.strip()]), args.arch
    if args.port is not None:
        st = fetch_slot_state(args.port, args.slot)
        t_prime = t_prime if t_prime is not None else st["t_prime"]
        cps = cps or st["checkpoints"]
        arch = arch or st["arch"]
    if t_prime is None:
        ap.error("need --t-prime or --port")
    cal = Calibration(args.a, args.b) if args.a is not None and args.b is not None else None
    res = predict_evict_cost(t_prime, args.cut, cps, cal,
                             hybrid=args.hybrid or is_hybrid_arch(arch), align_tol=args.align_tol)
    print(json.dumps(res.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
