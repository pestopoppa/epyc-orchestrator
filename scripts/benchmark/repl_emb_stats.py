#!/usr/bin/env python3
"""REPL-EMB-2.2 statistics: recall legs, paired bootstrap, kill rule, selection rule, strata.

Pure functions over arguments (numpy only, no I/O). Everything here implements the REPL-EMB-2.1
rule block as REGISTERED on 2026-09-27 (recall@5, M = 0.10, X = 0.05, n >= 120, paired-bootstrap
95% lower bound > 0). Changing k, M, X, n or the CI condition needs a new dated, operator-approved
registration; nothing in this file is a tunable.

PREDECLARED INTERPRETATIONS (frozen before any arm runs; echoed into every run's config):

@@INTERPRETATIONS@@

Conventions:

* A pointer is ``(section, start_line, end_line)``; gold is a list of ``(section, start, end)``.
  A pointer hits a query iff it has the same section and overlaps a gold span by >= 1 line
  (inclusive on both ends).
* ``ranked_hits_per_query[i]`` is the ranked pointer list of query i, or ``None`` when that query
  FAILED (dense failure, invalid or degenerate vector, timeout). ``None`` is a MISS, counted in
  ``n_failed``. An empty list is a legitimate "returned nothing" miss, not a failure.
* Bootstrap: by-query paired resampling of the per-query difference a - b, ``resamples`` draws
  with ``numpy.random.default_rng(seed)`` (fixed chunking, so the result is deterministic),
  percentile 95% interval (linear interpolation) of the resampled mean difference.
* Per-stratum proportions carry a Wilson score 95% interval (z = 1.959964).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_SEED = 20261020
DEFAULT_RESAMPLES = 10000
CI_LEVEL = 0.95
KS = (1, 3, 5, 10)
PRIMARY_K = 5
M_DEFAULT = 0.10
X_DEFAULT = 0.05
N_MIN = 120
#: Float slack for the ``>=`` comparisons (a margin of exactly 12/120 must pass M = 0.10).
EPS = 1e-9
BOOTSTRAP_CHUNK = 1000
WILSON_Z = 1.959963984540054

PREDECLARED_INTERPRETATIONS: dict[str, str] = {
    "a_decision_set": (
        "(a) The decision set is probe45 (45) plus the scored trace-derived queries (strata agent_nav, "
        "synthetic_spill and, if present, spill_follow), pooled; trivial_head is outside it. It is "
        "reported as TWO decision sets, pooled WITH and pooled WITHOUT synthetic_spill, each with its "
        "own n floor: the kill rule requires n >= 120 for that set, with fewer its verdict is "
        "UNDERPOWERED/KILL per REPL-EMB-2.1 ('small n kills by design'). (Amended 2026-10-20, before "
        "any arm ran: the single spill_follow source yielded n = 0 and the trace stratum became "
        "agent_nav + synthetic_spill.)"
    ),
    "b_registered_hybrids": (
        "(b) The kill rule is evaluated for EACH registered hybrid (hybrid-BGE, hybrid-Granite) "
        "against L; it PASSES iff at least one registered hybrid passes both conditions "
        "(mean diff >= M AND bootstrap CI lower bound > 0). Every individual result is reported. "
        "Hybrids over extension arms (EG2, potion) are evaluated afterwards; any changed verdict is "
        "flagged extension_only and never replaces the registered verdict."
    ),
    "c_selection": (
        "(c) The selection rule applies only if the kill rule passed, and uses cpu_s_per_query "
        "supplied from the speed block (a JSON file argument). Without it the harness refuses to "
        "select and reports 'selection: pending speed block'."
    ),
    "d_selection_candidates": (
        "(d) Selection candidates are the hybrid arms (registered and extension): the shipped "
        "context.search primitive is hybrid by construction, so dense-only, lexical, weighted and "
        "ColGREP rows are reported but not eligible. An arm with no cost row keeps selection pending."
    ),
    "e_incomplete": (
        "(e) A verdict is computed only from arms that scored EVERY deciding query. If L's legs or "
        "every registered hybrid are incomplete the verdict is INCOMPLETE, never a pass or fail from "
        "partial data."
    ),
}

__doc__ = (__doc__ or "").replace(
    "@@INTERPRETATIONS@@",
    "\n".join("  " + text for text in PREDECLARED_INTERPRETATIONS.values()),
)

Pointer = tuple[str, int, int]


# ───────────────────────────────────────────────────────────────────────────── hit accounting


def overlaps(chunk_lines: tuple[int, int], gold_ranges: Sequence[tuple[int, int]]) -> bool:
    """True iff the inclusive line range overlaps any inclusive gold range by >= 1 line."""
    cs, ce = int(chunk_lines[0]), int(chunk_lines[1])
    if ce < cs:
        raise ValueError(f"inverted chunk line range {chunk_lines!r}")
    for gs, ge in gold_ranges:
        gs, ge = int(gs), int(ge)
        if ge < gs:
            raise ValueError(f"inverted gold range {(gs, ge)!r}")
        if max(cs, gs) <= min(ce, ge):
            return True
    return False


def pointer_hits(pointer: Pointer, gold: Sequence[Pointer]) -> bool:
    """Same section AND >= 1 line of overlap with some gold span."""
    section, start, end = pointer
    same = [(gs, ge) for gsec, gs, ge in gold if gsec == section]
    return bool(same) and overlaps((start, end), same)


def recall_at_k(
    ranked_hits_per_query: Sequence[Sequence[Pointer] | None],
    gold: Sequence[Sequence[Pointer]],
    k: int,
) -> np.ndarray:
    """Per-query 0/1 hit vector at ``k``. A failed query (``None``) is a miss."""
    if len(ranked_hits_per_query) != len(gold):
        raise ValueError(f"{len(ranked_hits_per_query)} result rows for {len(gold)} gold rows")
    if k < 1:
        raise ValueError("k must be >= 1")
    out = np.zeros(len(gold), dtype=float)
    for i, (ptrs, g) in enumerate(zip(ranked_hits_per_query, gold)):
        if ptrs is None:
            continue
        out[i] = 1.0 if any(pointer_hits(tuple(p), g) for p in list(ptrs)[:k]) else 0.0  # type: ignore[arg-type]
    return out


def count_failed(ranked_hits_per_query: Sequence[Any]) -> int:
    """Queries with failure/invalid/timeout (``None``): misses, tracked apart from empty results."""
    return sum(1 for r in ranked_hits_per_query if r is None)


def recall_all_k(
    ranked_hits_per_query: Sequence[Sequence[Pointer] | None],
    gold: Sequence[Sequence[Pointer]],
    ks: Sequence[int] = KS,
) -> dict[int, np.ndarray]:
    return {int(k): recall_at_k(ranked_hits_per_query, gold, int(k)) for k in ks}


# ───────────────────────────────────────────────────────────────────────────── intervals


def wilson_interval(hits: float, n: int, z: float = WILSON_Z) -> tuple[float | None, float | None]:
    """Wilson score interval for a proportion ``hits / n`` (None, None when n == 0)."""
    if n <= 0:
        return None, None
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _as_vec(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-d, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    return a


def paired_bootstrap(
    a: Any,
    b: Any,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """By-query paired bootstrap of the mean of ``a - b``; deterministic under ``seed``."""
    av, bv = _as_vec(a, "a"), _as_vec(b, "b")
    if av.shape != bv.shape:
        raise ValueError(f"paired vectors differ in length: {av.shape} vs {bv.shape}")
    n = len(av)
    if n == 0:
        raise ValueError("paired_bootstrap needs at least one query")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    d = av - bv
    rng = np.random.default_rng(seed)
    means = np.empty(int(resamples), dtype=float)
    done = 0
    while done < resamples:
        m = min(BOOTSTRAP_CHUNK, resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done : done + m] = d[idx].mean(axis=1)
        done += m
    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    lo, hi = np.percentile(means, [tail, 100.0 - tail])
    return {
        "n": int(n),
        "mean_a": float(av.mean()),
        "mean_b": float(bv.mean()),
        "mean_diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_level": CI_LEVEL,
        "method": "paired by-query percentile bootstrap (linear interpolation)",
        "resamples": int(resamples),
        "seed": int(seed),
    }


# ───────────────────────────────────────────────────────────────────────────── rules


def kill_rule(
    hybrid_recall5: Any,
    lexical_recall5: Any,
    M: float = M_DEFAULT,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """REPL-EMB-2.1 kill rule for ONE hybrid vs L: PASS iff mean diff >= M AND CI lower > 0."""
    bs = paired_bootstrap(hybrid_recall5, lexical_recall5, resamples, seed)
    margin_ok = bs["mean_diff"] >= M - EPS
    ci_ok = bs["ci_lo"] > 0.0
    return {
        "rule": "hybrid recall@5 - L recall@5 >= M AND paired-bootstrap 95% CI lower bound > 0",
        "M": float(M),
        "n": bs["n"],
        "hybrid_recall5": bs["mean_a"],
        "lexical_recall5": bs["mean_b"],
        "mean_diff": bs["mean_diff"],
        "ci_lo": bs["ci_lo"],
        "ci_hi": bs["ci_hi"],
        "ci_level": bs["ci_level"],
        "method": bs["method"],
        "resamples": bs["resamples"],
        "seed": bs["seed"],
        "margin_ok": bool(margin_ok),
        "ci_ok": bool(ci_ok),
        "pass": bool(margin_ok and ci_ok),
    }


def kill_decision(
    registered: Mapping[str, Any],
    extension: Mapping[str, Any],
    lexical_recall5: Any,
    lexical_name: str,
    *,
    M: float = M_DEFAULT,
    n_min: int = N_MIN,
    missing_registered: Sequence[str] = (),
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Interpretations (a), (b), (e): the verdict over registered hybrids, extension flagged.

    ``registered`` / ``extension`` map hybrid-arm name -> per-query recall@5 vector (complete on
    the deciding set). ``missing_registered`` names registered hybrids that could not be scored.
    Verdicts: ``PASS`` | ``FAIL`` | ``UNDERPOWERED/KILL`` (n < n_min) | ``INCOMPLETE``.
    """
    lex = _as_vec(lexical_recall5, "lexical_recall5")
    n = len(lex)

    def evaluate(group: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            name: kill_rule(vec, lex, M, resamples=resamples, seed=seed)
            for name, vec in group.items()
        }

    reg = evaluate(registered)
    ext = evaluate(extension)
    reg_pass = sorted(k for k, v in reg.items() if v["pass"])
    ext_pass = sorted(k for k, v in ext.items() if v["pass"])
    underpowered = n < n_min

    def verdict(passers: Sequence[str], can_fail: bool) -> str:
        if underpowered:
            return "UNDERPOWERED/KILL"
        if passers:
            return "PASS"
        return "FAIL" if can_fail else "INCOMPLETE"

    v_reg = verdict(reg_pass, can_fail=bool(reg) and not missing_registered)
    v_all = verdict(
        sorted(set(reg_pass) | set(ext_pass)), can_fail=bool(reg) and not missing_registered
    )
    return {
        "M": float(M),
        "n": int(n),
        "n_min": int(n_min),
        "underpowered": bool(underpowered),
        "lexical_baseline": lexical_name,
        "lexical_recall5": float(lex.mean()) if n else None,
        "registered": reg,
        "extension": ext,
        "missing_registered": list(missing_registered),
        "registered_passers": reg_pass,
        "extension_passers": ext_pass,
        "verdict": v_reg,
        "verdict_with_extension": v_all,
        "changed_by_extension": v_all != v_reg,
        "extension_only": (
            sorted(ext_pass) if (v_all != v_reg and not reg_pass and not underpowered) else []
        ),
    }


def lexical_baseline(grep_r5: Any, bm25_r5: Any) -> dict[str, Any]:
    """L = the better lexical arm by mean recall@5 on the deciding set; ties go to BM25."""
    g, b = _as_vec(grep_r5, "grep_r5"), _as_vec(bm25_r5, "bm25_r5")
    if g.shape != b.shape:
        raise ValueError("grep and bm25 vectors differ in length")
    gm = float(g.mean()) if len(g) else 0.0
    bm = float(b.mean()) if len(b) else 0.0
    tie = abs(gm - bm) <= EPS
    which = "grep" if (gm > bm + EPS) else "bm25"
    return {
        "L": which,
        "which": which,
        "grep_recall5": gm,
        "bm25_recall5": bm,
        "tie": bool(tie),
        "n": int(len(g)),
        "rule": "better of grep / BM25 by mean recall@5 on the deciding set; ties -> bm25",
    }


def selection_rule(arms: Mapping[str, Mapping[str, Any]], X: float = X_DEFAULT) -> dict[str, Any]:
    """Cheapest arm (CPU-s/query, then p95 latency) with recall@5 >= best - X.

    ``arms[name]`` = ``{recall5, cpu_s_per_query, p95_ms}``. A candidate without a cost keeps the
    selection pending (never guessed)."""
    if not arms:
        return {"status": "no arms", "selected": None, "table": [], "X": float(X)}
    best = max(float(a["recall5"]) for a in arms.values())
    threshold = best - X
    inf = float("inf")
    table = []
    for name in sorted(arms, key=lambda n: (-float(arms[n]["recall5"]), n)):
        a = arms[name]
        r5 = float(a["recall5"])
        cpu = a.get("cpu_s_per_query")
        p95 = a.get("p95_ms")
        table.append(
            {
                "arm": name,
                "recall5": r5,
                "cpu_s_per_query": None if cpu is None else float(cpu),
                "p95_ms": None if p95 is None else float(p95),
                "candidate": bool(r5 >= threshold - EPS),
            }
        )
    cands = [r for r in table if r["candidate"]]
    missing = [r["arm"] for r in cands if r["cpu_s_per_query"] is None]
    out: dict[str, Any] = {
        "X": float(X),
        "best_recall5": best,
        "best_arm": table[0]["arm"],
        "threshold": threshold,
        "table": table,
        "candidates": [r["arm"] for r in cands],
        "rule": "cheapest by cpu_s_per_query (tie: p95_ms) among arms with recall@5 >= best - X",
    }
    if missing:
        out.update({"status": "pending speed block", "selected": None, "missing_cost": missing})
        return out
    chosen = min(
        cands,
        key=lambda r: (r["cpu_s_per_query"], inf if r["p95_ms"] is None else r["p95_ms"], r["arm"]),
    )
    for r in table:
        r["selected"] = r["arm"] == chosen["arm"]
    out.update({"status": "selected", "selected": chosen["arm"], "missing_cost": []})
    return out


def stratum_report(
    per_query_vectors: Mapping[str, Mapping[int, Any]],
    strata: Sequence[str],
    ks: Sequence[int] = KS,
    *,
    pooled_label: str = "pooled",
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """recall@k per arm per stratum (plus pooled) with Wilson 95% intervals.

    ``per_query_vectors[arm][k]`` is the 0/1 vector over the same queries as ``strata``."""
    labels = list(strata)
    out: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    groups = sorted(set(labels)) + [pooled_label]
    for arm, by_k in per_query_vectors.items():
        out[arm] = {}
        for g in groups:
            sel = (
                np.ones(len(labels), dtype=bool)
                if g == pooled_label
                else np.array([lab == g for lab in labels])
            )
            out[arm][g] = {}
            for k in ks:
                vec = np.asarray(by_k[int(k)], dtype=float)
                if len(vec) != len(labels):
                    raise ValueError(
                        f"{arm}@{k}: {len(vec)} values for {len(labels)} strata labels"
                    )
                sub = vec[sel]
                n = int(sel.sum())
                hits = float(sub.sum())
                lo, hi = wilson_interval(hits, n)
                out[arm][g][int(k)] = {
                    "n": n,
                    "hits": hits,
                    "recall": (hits / n) if n else None,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci": "wilson-95",
                }
    return out


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([x for x in latencies_ms if x is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    return {
        "n": int(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }


__all__ = [
    "CI_LEVEL",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "KS",
    "M_DEFAULT",
    "N_MIN",
    "PREDECLARED_INTERPRETATIONS",
    "PRIMARY_K",
    "X_DEFAULT",
    "count_failed",
    "kill_decision",
    "kill_rule",
    "latency_summary",
    "lexical_baseline",
    "overlaps",
    "paired_bootstrap",
    "pointer_hits",
    "recall_all_k",
    "recall_at_k",
    "selection_rule",
    "stratum_report",
    "wilson_interval",
]
