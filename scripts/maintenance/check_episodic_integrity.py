#!/usr/bin/env python3
"""check_episodic_integrity.py — fail loudly the day the episodic store rots.

WHY THIS EXISTS
---------------
The store's vector resolution was silently wrong from 2026-07-05T15:01:12 until
2026-07-27 — **22 days** — and nothing noticed, because nothing looked. Every
component reported healthy the whole time: the index loaded, queries returned
neighbours, `id_map[embedding_idx] == id` held internally, and retrieval produced
plausible-looking results that were semantically random.

The lesson is not "that bug"; it is that an internally-consistent store can be
completely wrong. These assertions check the properties that were actually
violated, each of which would have fired on day one.

CHECKS
------
1. **index/id_map sync** — `ntotal == len(id_map)`. Non-zero desync in the
   index-ahead direction is unrecoverable and poisons every later write.
2. **embedding_idx round-trip** — a sampled row's `embedding_idx` must resolve
   through `id_map` back to that same row. During the incident this held for
   239 of 54,960 rows (0.4%).
3. **vector diversity floor** — distinct vectors per **distinct objective**, not
   per row. Benchmark traffic replays the same objectives constantly (500 recent
   rows carried only 57 distinct objectives) and identical text *should* share a
   vector, so a row denominator flags healthy replay as collapse. During the
   incident 47 unrelated objectives shared one vector — that is what this ratio
   catches.
4. **degenerate vectors** — all-zero or non-finite. These are what an embedder
   outage writes: `use_fallback` defaults to True everywhere, so a BGE outage
   silently substitutes a SHA-256 pseudo-vector, which measures 89.0% all-zero
   and 2.8% NaN over real task text. Crucially this check needs **no embedder**,
   so it still works in exactly the condition that produces the corruption —
   closing the blind spot where the only detector required the thing that was
   down.
5. **semantic self-match (the decisive one)** — re-embed a row's own objective
   and cosine it against its stored vector. This is the check that no amount of
   internal consistency can fake. During the incident: mean **0.5505**. After
   repair: **1.0**. Requires the BGE servers, so it is opt-in via
   ``--semantic``; the other three are pure metadata and always run.

Exit 0 = healthy, 1 = a check failed. Intended for `health_check.sh` and any
pre-Autopilot gate.

Usage:
    python scripts/maintenance/check_episodic_integrity.py
    python scripts/maintenance/check_episodic_integrity.py --semantic --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

SESSIONS = _REPO_ROOT / "orchestration/repl_memory/sessions"

# Thresholds. Chosen against measured healthy state (2026-07-28: diversity 1.00,
# self-match 1.0) with margin for legitimate duplicate objectives.
MIN_DIVERSITY = 0.50
MIN_SELF_MATCH = 0.90
ROUNDTRIP_SAMPLE = 500

# 2026-08-04: this was a MEAN over 8 rows, which is both too fragile and too blind.
#
# Too fragile: ONE bad vector in a 60,340-row store failed the whole gate and refused to
# start AutoPilot. Measured that day — 59 of 60 sampled rows self-matched at ~0.996 and a
# single row scored 0.0212, giving (7 x 0.996 + 0.021) / 8 = 0.874 against a 0.90 floor.
# The reported failure was exactly 0.8718.
#
# Too blind: a mean hides the corruption that matters. Ten rows at 0.55 — vectors that
# belong to OTHER rows — among ninety perfect ones averages 0.955 and PASSES.
#
# A per-row threshold with a tolerated fraction fixes both, and a cross-row hit is now a
# hard fail on its own because it is unambiguous: that vector resolves to a different
# memory, no averaging required.
SEMANTIC_SAMPLE = 60
# Below this a vector is not a degraded self-match, it belongs to a DIFFERENT row.
CROSS_ROW_CEILING = 0.55
# Tolerated share of sampled rows under MIN_SELF_MATCH before the store is DEGRADED.
MAX_BELOW_FLOOR_FRACTION = 0.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-dir", default=str(SESSIONS))
    ap.add_argument("--semantic", action="store_true", help="also run the BGE self-match check")
    ap.add_argument(
        "--require-semantic",
        action="store_true",
        help="fail when the requested semantic self-match check cannot complete",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--embedder-url", default="http://127.0.0.1:8090/embedding")
    args = ap.parse_args()
    if args.require_semantic and not args.semantic:
        ap.error("--require-semantic requires --semantic")

    import faiss
    import numpy as np

    d = Path(args.sessions_dir)
    checks: list[dict] = []
    ok = True

    def record(name: str, passed: bool, detail: str, **extra) -> None:
        nonlocal ok
        ok = ok and passed
        checks.append({"check": name, "pass": passed, "detail": detail, **extra})

    def skip(name: str, detail: str) -> None:
        """A check that could not run. Reported LOUDLY and never silently.

        A skip does not fail the gate — BGE may simply still be booting — but it
        must be visible, because "nothing looked" is precisely how the store
        rotted for 22 days.
        """
        checks.append({"check": name, "pass": None, "skipped": True, "detail": detail})

    index = faiss.read_index(str(d / "embeddings.faiss"))
    id_map = np.load(d / "id_map.npy", allow_pickle=True).tolist()

    # 1. index / id_map sync
    desync = index.ntotal - len(id_map)
    record(
        "index_id_map_sync",
        desync == 0,
        f"ntotal={index.ntotal} id_map={len(id_map)} desync={desync}"
        + ("" if desync == 0 else "  <- index-ahead is UNRECOVERABLE; run repair_faiss_id_map.py"),
        desync=desync,
    )

    con = sqlite3.connect(f"file:{d / 'episodic.db'}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, embedding_idx, context FROM memories "
        "WHERE embedding_idx IS NOT NULL ORDER BY created_at DESC LIMIT ?",
        (ROUNDTRIP_SAMPLE,),
    ).fetchall()

    # 2. embedding_idx round-trip
    bad = [r for r in rows if r[1] >= len(id_map) or str(id_map[r[1]]) != str(r[0])]
    record(
        "embedding_idx_roundtrip",
        not bad,
        f"{len(rows) - len(bad)}/{len(rows)} sampled rows resolve to themselves"
        + ("" if not bad else f"  <- {len(bad)} DO NOT; the DB->vector mapping is wrong"),
        checked=len(rows),
        failed=len(bad),
    )

    # 3. vector diversity floor — measured against DISTINCT OBJECTIVES, not rows.
    #
    # The denominator matters and I got it wrong first: benchmark traffic replays
    # the same objectives constantly (500 recent rows carried only 57 distinct
    # objectives), and identical text SHOULD produce an identical vector. Dividing
    # by row count therefore flags healthy replay as collapse. The real property
    # is one-vector-per-distinct-objective; during the incident 47 unrelated
    # objectives shared a single vector, which this ratio catches and the
    # row-denominator version buried.
    if rows:
        vecs, objs = set(), set()
        for _mid, idx, ctx in rows:
            vecs.add(hashlib.blake2b(index.reconstruct(int(idx)).tobytes(), digest_size=8).hexdigest())
            try:
                objs.add(((json.loads(ctx) or {}).get("objective") or "").strip())
            except Exception:
                objs.add("")
        diversity = len(vecs) / max(1, len(objs))
        record(
            "vector_diversity",
            diversity >= MIN_DIVERSITY,
            f"{len(vecs)} distinct vectors for {len(objs)} distinct objectives "
            f"across {len(rows)} rows (ratio {diversity:.3f}, floor {MIN_DIVERSITY})"
            + ("" if diversity >= MIN_DIVERSITY else "  <- distinct objectives are sharing vectors"),
            distinct_vectors=len(vecs),
            distinct_objectives=len(objs),
            diversity=round(diversity, 4),
        )

    # 4. degenerate vectors — the embedder-outage signature, detectable WITHOUT
    # an embedder. This is the check that works when semantic_self_match cannot.
    if rows:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
        try:
            from orchestration.repl_memory.embedder import is_degenerate_embedding
        except Exception:
            is_degenerate_embedding = None
        if is_degenerate_embedding is not None:
            bad_vecs: dict[str, int] = {}
            for _mid, idx, _ctx in rows:
                reason = is_degenerate_embedding(index.reconstruct(int(idx)))
                if reason:
                    bad_vecs[reason] = bad_vecs.get(reason, 0) + 1
            total_bad = sum(bad_vecs.values())
            record(
                "degenerate_vectors",
                total_bad == 0,
                f"{total_bad}/{len(rows)} sampled vectors are degenerate"
                + (f" {bad_vecs}  <- an embedder outage wrote hash pseudo-vectors; "
                   "these rows are not semantically retrievable" if total_bad else ""),
                counts=bad_vecs,
            )

    # 5. semantic self-match — the check internal consistency cannot fake
    if args.semantic:
        import httpx

        sims = []
        try:
            with httpx.Client(timeout=30) as client:
                for mid, idx, ctx in rows:
                    try:
                        obj = (json.loads(ctx) or {}).get("objective") or ""
                    except Exception:
                        obj = ""
                    if not obj.strip():
                        continue
                    task_type = (json.loads(ctx) or {}).get("task_type")
                    text = (f"type:{task_type} | " if task_type else "") + f"objective:{obj.strip()}"
                    r = client.post(args.embedder_url, json={"content": text[:2000]}).json()
                    if isinstance(r, list):
                        r = r[0]
                    e = r["embedding"]
                    e = np.asarray(e[0] if isinstance(e[0], list) else e, dtype=np.float32)
                    n = np.linalg.norm(e)
                    if n:
                        e = e / n
                        sims.append(float(np.dot(e, index.reconstruct(int(idx)))))
                        if len(sims) >= SEMANTIC_SAMPLE:
                            break
        except Exception as exc:  # BGE unreachable / still booting
            sims = []
            detail = (
                f"BGE embedder unreachable ({type(exc).__name__}: {exc}) "
                "— the DECISIVE check did not run; re-run once the embedders are up"
            )
            if args.require_semantic:
                record("semantic_self_match", False, detail)
            else:
                skip("semantic_self_match", detail)
        if sims:
            arr = np.asarray(sims, dtype=np.float64)
            mean = float(arr.mean())
            median = float(np.median(arr))
            cross_row = int((arr < CROSS_ROW_CEILING).sum())
            below_floor = int((arr < MIN_SELF_MATCH).sum())
            below_fraction = below_floor / len(arr)
            # Per-row verdict, not an average. A cross-row hit is unambiguous corruption
            # and fails on its own; otherwise a small share of degraded rows is tolerated.
            passed = cross_row == 0 and below_fraction <= MAX_BELOW_FLOOR_FRACTION
            record(
                "semantic_self_match",
                passed,
                (
                    f"{below_floor}/{len(arr)} rows below floor {MIN_SELF_MATCH} "
                    f"({below_fraction:.1%}, tolerated {MAX_BELOW_FLOOR_FRACTION:.0%}); "
                    f"{cross_row} row(s) below {CROSS_ROW_CEILING} i.e. the vector belongs "
                    f"to ANOTHER row (0 tolerated); mean {mean:.4f} median {median:.4f}"
                ),
                mean_cosine=round(mean, 4),
                median_cosine=round(median, 4),
                sampled=len(arr),
                below_floor=below_floor,
                below_floor_fraction=round(below_fraction, 4),
                cross_row_hits=cross_row,
                min_cosine=round(float(arr.min()), 4),
            )
        elif args.require_semantic and not any(
            check["check"] == "semantic_self_match" for check in checks
        ):
            record(
                "semantic_self_match",
                False,
                "semantic self-match produced no usable objective samples",
            )
    con.close()

    if args.json:
        print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    else:
        print("=== EPISODIC STORE INTEGRITY ===")
        for c in checks:
            tag = "SKIP" if c.get("skipped") else ("PASS" if c["pass"] else "FAIL")
            print(f"  [{tag}] {c['check']}: {c['detail']}")
        print(f"\n  {'HEALTHY' if ok else 'DEGRADED — do not trust memory-derived results'}")
        if not args.semantic:
            print("  (run with --semantic for the decisive re-embed check)")
        elif any(c.get("skipped") for c in checks):
            print("  ⚠ the decisive semantic check was SKIPPED — health is only partly established")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
