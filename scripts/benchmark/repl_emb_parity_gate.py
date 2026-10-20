#!/usr/bin/env python3
from __future__ import annotations

"""REPL-EMB-2.5 / 2.6 reference-parity gate for the EmbeddingGemma 2 lanes.

Vector parity ONLY (finite vectors, per-item norms, per-item cosine against the sentence-
transformers FP32 reference). It makes no recall or speed claim: retrieval quality is REPL-EMB-2.7.

Predeclared gate (REPL-EMB-2.5 / 2.6, operator-confirmed 2026-10-20), fixed in ``GATES`` below
before any run:

  * zero non-finite vectors (NaN or Inf anywhere in any returned vector), AND
  * per-item cosine to the ST FP32 reference >= 0.99 for ``st-bf16`` and ``gguf-bf16``,
    >= 0.98 for ``gguf-q8_0``; the MINIMUM over every item decides (not the mean),
  * on >= 200 chunks (real project text through the SHIPPED chunker) plus the query set.

An arm that fails is recorded as failed, never dropped.

Subcommands (each lane that loads a model runs in its own process so RSS is clean):

  encode   one lane -> ``<out>/vec-<lane>.npz`` + ``<out>/lane-<lane>.json`` (incl. peak RSS)
  compare  every ``vec-*.npz`` against ``vec-st-fp32-full.npz`` -> ``<out>/parity.json`` + table
  all      build the item set, run the ST lanes as subprocesses, encode GGUF lanes against
           ``--gguf NAME=URL`` servers (started and owned by the caller), then compare

The text-only versus full-model equivalence check (card: ``config_kwargs={"vision_config": None,
"audio_config": None}``) is the lane pair ``st-fp32-full`` / ``st-fp32-textonly``; its RSS and its
vector equivalence are reported next to the gate.
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repl_emb_arms as arms  # noqa: E402

# lane -> (threshold, kind). Predeclared; do not edit after a run.
GATES: dict[str, float] = {"st-bf16": 0.99, "gguf-bf16": 0.99, "gguf-q8_0": 0.98}
REFERENCE_LANE = "st-fp32-full"
MIN_CHUNKS = 200
UFH07_QUESTIONS = 45


# ───────────────────────────────────────────────────────────────────────────── item set


class _Section:
    def __init__(self, name: str, text: str) -> None:
        self.name, self.text = name, text


def _kind(path: str) -> str:
    return (
        "code"
        if path.endswith(".py")
        else ("json" if path.endswith((".json", ".yaml", ".yml")) else "prose")
    )


def build_items(
    corpus_files: list[Path], n_chunks: int, n_queries: int, queries_file: Path | None
) -> dict[str, Any]:
    """Deterministic item set: real files -> shipped chunker -> stride-selected chunks (all three
    kinds represented) plus the query set. Everything is hashed into the returned record."""
    cs = arms.load_context_search()
    sections, file_rows = [], []
    for p in sorted(corpus_files):
        text = p.read_text(encoding="utf-8", errors="replace")
        sections.append(_Section(str(p), text))
        file_rows.append(
            {
                "path": str(p),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "kind": _kind(str(p)),
                "chars": len(text),
            }
        )
    chunks = cs.chunk_sections(sections, cs.CHUNK_MAX_CHARS)
    chunks = [c for c in chunks if len(c.text.strip()) >= 40]
    by_kind: dict[str, list[Any]] = {}
    for c in chunks:
        by_kind.setdefault(_kind(c.section), []).append(c)
    picked: list[Any] = []
    kinds = sorted(by_kind)
    per = max(1, -(-n_chunks // len(kinds)))
    for k in kinds:
        rows = by_kind[k]
        step = max(1, len(rows) // per)
        picked.extend(rows[::step][:per])
    picked = sorted(
        picked, key=lambda c: hashlib.sha1(f"{c.section}:{c.start}".encode()).hexdigest()
    )
    doc_texts = [c.text[: cs.DENSE_INPUT_MAX_CHARS] for c in picked]
    if queries_file is not None:
        qrows = [json.loads(line) for line in queries_file.read_text().splitlines() if line.strip()]
        queries = [
            {
                "id": str(r["id"]),
                "text": r["text"],
                "code_intent": bool(r.get("code_intent", False)),
            }
            for r in qrows
        ]
        qkind = "file:" + str(queries_file)
    else:
        # PROXY: first non-empty line of every k-th picked chunk. These are NOT the intake-1357#03 probe
        # questions (that set is not frozen yet); the verdict is labelled accordingly.
        queries = []
        step = max(1, len(picked) // n_queries)
        for c in picked[::step][:n_queries]:
            first = next(
                (ln.strip() for ln in c.text.splitlines() if len(ln.strip()) >= 12),
                c.text.strip()[:80],
            )
            queries.append(
                {
                    "id": f"proxy-{len(queries):03d}",
                    "text": first[:200],
                    "code_intent": _kind(c.section) == "code",
                }
            )
        qkind = "proxy-first-line"
    return {
        "files": file_rows,
        "chunk_max_chars": cs.CHUNK_MAX_CHARS,
        "dense_input_max_chars": cs.DENSE_INPUT_MAX_CHARS,
        "n_chunks": len(doc_texts),
        "chunk_kinds": {k: sum(1 for c in picked if _kind(c.section) == k) for k in kinds},
        "chunks": [
            {
                "section": c.section,
                "start": c.start,
                "end": c.end,
                "text_sha256": hashlib.sha256(t.encode()).hexdigest(),
            }
            for c, t in zip(picked, doc_texts)
        ],
        "doc_texts": doc_texts,
        "queries": queries,
        "query_set_kind": qkind,
        "query_set_is_ufh07": queries_file is not None
        and ("probe45" in queries_file.name.lower() or "ufh07" in queries_file.name.lower()),
    }


# ───────────────────────────────────────────────────────────────────────────── lanes


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    return -1.0


def _spec(lane: str) -> arms.ArmSpec:
    st_rt = "sentence-transformers/transformers/torch-cpu"
    by = {s.arm_id: s for s in arms.arm_specs(st_rt, "llama-server-experimental", "model2vec")}
    return {
        "st-fp32-full": by["eg2-st-fp32-768"],
        "st-fp32-textonly": by["eg2-st-fp32-768"],
        "st-bf16": by["eg2-st-bf16-768"],
        "gguf-bf16": by["eg2-gguf-bf16-768"],
        "gguf-q8_0": by["eg2-gguf-q8_0-768"],
    }[lane]


def encode_lane(
    lane: str,
    items: dict[str, Any],
    out: Path,
    *,
    model_path: str | None,
    url: str | None,
    threads: int,
) -> dict[str, Any]:
    import numpy as np

    spec = _spec(lane)
    t0 = time.perf_counter()
    if lane.startswith("st-"):
        assert model_path, "--model-path required for ST lanes"
        ad = arms.SentenceTransformersAdapter(
            spec, model_path, text_only=(lane != "st-fp32-full"), threads=threads
        )
    else:
        assert url, "--url required for GGUF lanes"
        ad = arms.LlamaServerAdapter(spec, url)
    load_s = round(time.perf_counter() - t0, 2)
    rss_loaded = _rss_mb()
    # Encode RAW (pre-truncation) so non-finite values are counted, not renormalised away.
    t1 = time.perf_counter()
    docs_raw = np.asarray(
        ad._encode(ad._prefixed(items["doc_texts"], "document")), dtype=np.float32
    )
    q_raw = []
    for q in items["queries"]:
        with ad.query_intent(q["code_intent"]):
            q_raw.append(ad._encode(ad._prefixed([q["text"]], "query"))[0])
    q_raw = np.asarray(q_raw, dtype=np.float32)
    enc_s = round(time.perf_counter() - t1, 2)
    np.savez(out / f"vec-{lane}.npz", docs=docs_raw, queries=q_raw)
    rec = {
        "lane": lane,
        "arm_id": spec.arm_id,
        "index_identity": spec.index_identity(),
        "load_s": load_s,
        "encode_s": enc_s,
        "rss_after_load_mb": rss_loaded,
        "rss_after_encode_mb": _rss_mb(),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "threads": threads,
        "max_seq_length": getattr(ad, "max_seq_length", None),
        "n_docs": int(docs_raw.shape[0]),
        "n_queries": int(q_raw.shape[0]),
        "raw_dim": int(docs_raw.shape[1]),
    }
    (out / f"lane-{lane}.json").write_text(json.dumps(rec, indent=2))
    return rec


# ───────────────────────────────────────────────────────────────────────────── compare


def _stats(ref: Any, cand: Any) -> dict[str, Any]:
    import numpy as np

    finite_rows = np.isfinite(cand).all(axis=1)
    nonfinite = int((~finite_rows).sum())
    nan_vals = int(np.isnan(cand).sum())
    inf_vals = int(np.isinf(cand).sum())
    norms = np.linalg.norm(np.where(np.isfinite(cand), cand, 0.0), axis=1)
    c = arms.l2_normalise(np.where(np.isfinite(cand), cand, 0.0))
    r = arms.l2_normalise(ref)
    cos = np.where(finite_rows, (c * r).sum(axis=1), -1.0)  # a non-finite row can never pass
    return {
        "n": int(cand.shape[0]),
        "nonfinite_vectors": nonfinite,
        "nan_values": nan_vals,
        "inf_values": inf_vals,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "cos_min": float(cos.min()),
        "cos_p01": float(np.quantile(cos, 0.01)),
        "cos_mean": float(cos.mean()),
        "cos_max": float(cos.max()),
        "argmin": int(cos.argmin()),
        "per_item_cos": [round(float(x), 6) for x in cos],
        "per_item_norm": [round(float(x), 6) for x in norms],
    }


def compare(out: Path, items_meta: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    ref_path = out / f"vec-{REFERENCE_LANE}.npz"
    ref = np.load(ref_path)
    ref_stats = {
        "nonfinite_vectors": int(
            (~np.isfinite(ref["docs"]).all(axis=1)).sum()
            + (~np.isfinite(ref["queries"]).all(axis=1)).sum()
        )
    }
    lanes: dict[str, Any] = {}
    for p in sorted(out.glob("vec-*.npz")):
        lane = p.stem[4:]
        if lane == REFERENCE_LANE:
            continue
        cand = np.load(p)
        lane_rec = json.loads((out / f"lane-{lane}.json").read_text())
        d, q = _stats(ref["docs"], cand["docs"]), _stats(ref["queries"], cand["queries"])
        both_min = min(d["cos_min"], q["cos_min"])
        nonfinite = d["nonfinite_vectors"] + q["nonfinite_vectors"]
        thr = GATES.get(lane)
        verdict = (
            None if thr is None else ("PASS" if (nonfinite == 0 and both_min >= thr) else "FAIL")
        )
        # 256-d view for information only (MRL slice + re-normalise), not gated.
        c256 = arms.truncate_renormalise(
            np.where(np.isfinite(cand["docs"]), cand["docs"], 0.0), 256
        )
        r256 = arms.truncate_renormalise(ref["docs"], 256)
        cos256 = (c256 * r256).sum(axis=1)
        lanes[lane] = {
            "threshold": thr,
            "verdict": verdict,
            "nonfinite_vectors_total": nonfinite,
            "cos_min_all_items": both_min,
            "docs": d,
            "queries": q,
            "info_256d_docs_cos_min": float(cos256.min()),
            "info_256d_docs_cos_mean": float(cos256.mean()),
            "lane": lane_rec,
        }
    ref_lane = json.loads((out / f"lane-{REFERENCE_LANE}.json").read_text())
    res = {
        "gate": {
            "thresholds": GATES,
            "rule": "zero non-finite vectors AND min per-item cosine >= threshold vs ST FP32 reference",
            "min_chunks": MIN_CHUNKS,
            "reference_lane": REFERENCE_LANE,
        },
        "reference": {"lane": ref_lane, **ref_stats},
        "lanes": lanes,
        "items": items_meta,
    }
    n_chunks = items_meta["n_chunks"]
    res["scope_complete"] = bool(
        n_chunks >= MIN_CHUNKS
        and items_meta["query_set_is_ufh07"]
        and len(items_meta["queries"]) >= UFH07_QUESTIONS
    )
    res["scope_note"] = (
        "complete: >=200 chunks and the 45 behavioural questions (intake-1357#03)"
        if res["scope_complete"]
        else f"PARTIAL: {n_chunks} chunks (>= {MIN_CHUNKS}: {n_chunks >= MIN_CHUNKS}); query set = {items_meta['query_set_kind']} "
        f"({len(items_meta['queries'])} queries), NOT the 45 behavioural questions (intake-1357#03)"
    )
    return res


def render_table(res: dict[str, Any]) -> str:
    rows = [
        "| lane | threshold | non-finite | cos min (docs) | cos min (queries) | cos mean (docs) | norm range | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lane, r in res["lanes"].items():
        rows.append(
            f"| {lane} | {r['threshold']} | {r['nonfinite_vectors_total']} | {r['docs']['cos_min']:.6f} | "
            f"{r['queries']['cos_min']:.6f} | {r['docs']['cos_mean']:.6f} | "
            f"{min(r['docs']['norm_min'], r['queries']['norm_min']):.4f}-{max(r['docs']['norm_max'], r['queries']['norm_max']):.4f} | {r['verdict']} |"
        )
    return "\n".join(rows)


# ───────────────────────────────────────────────────────────────────────────── CLI


def _load_items(out: Path) -> dict[str, Any]:
    return json.loads((out / "items.json").read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "encode", "compare", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name in ("prepare", "all"):
            p.add_argument("--corpus-file", type=Path, action="append", default=[])
            p.add_argument("--n-chunks", type=int, default=240)
            p.add_argument("--n-queries", type=int, default=UFH07_QUESTIONS)
            p.add_argument(
                "--queries",
                type=Path,
                default=None,
                help="JSONL {id,text,code_intent}; the 45 behavioural questions of intake-1357#03 (file name containing probe45)",
            )
            p.add_argument(
                "--reuse-items",
                action="store_true",
                help="(all) use the existing <out>/items.json from `prepare`",
            )
        if name in ("encode", "all"):
            p.add_argument("--model-path", default=None)
            p.add_argument("--threads", type=int, default=4)
        if name == "encode":
            p.add_argument(
                "--lane",
                required=True,
                choices=["st-fp32-full", "st-fp32-textonly", "st-bf16", "gguf-bf16", "gguf-q8_0"],
            )
            p.add_argument("--url", default=None)
        if name == "all":
            p.add_argument(
                "--gguf",
                action="append",
                default=[],
                help="LANE=URL for a caller-owned llama-server, e.g. gguf-bf16=http://127.0.0.1:18741",
            )
            p.add_argument("--st-lanes", default="st-fp32-full,st-fp32-textonly,st-bf16")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.cmd in ("prepare", "all") and not (a.cmd == "all" and a.reuse_items):
        if not a.corpus_file:
            ap.error("--corpus-file required")
        items = build_items(a.corpus_file, a.n_chunks, a.n_queries, a.queries)
        (a.out / "items.json").write_text(json.dumps(items, indent=1))
        print(
            f"items: {items['n_chunks']} chunks {items['chunk_kinds']}, {len(items['queries'])} queries ({items['query_set_kind']})"
        )
        if a.cmd == "prepare":
            return 0
    if a.cmd == "encode":
        rec = encode_lane(
            a.lane, _load_items(a.out), a.out, model_path=a.model_path, url=a.url, threads=a.threads
        )
        print(json.dumps(rec))
        return 0
    if a.cmd == "all":
        for lane in [x for x in a.st_lanes.split(",") if x]:
            cmd = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "encode",
                "--out",
                str(a.out),
                "--lane",
                lane,
                "--model-path",
                a.model_path,
                "--threads",
                str(a.threads),
            ]
            subprocess.run(
                cmd,
                check=True,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent)},
            )
        for spec in a.gguf:
            lane, url = spec.split("=", 1)
            cmd = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "encode",
                "--out",
                str(a.out),
                "--lane",
                lane,
                "--url",
                url,
            ]
            subprocess.run(cmd, check=True)
    items = _load_items(a.out)
    res = compare(a.out, {k: v for k, v in items.items() if k not in ("doc_texts",)})
    (a.out / "parity.json").write_text(json.dumps(res, indent=1))
    table = render_table(res)
    (a.out / "parity.md").write_text(table + "\n\n" + res["scope_note"] + "\n")
    print(table)
    print(res["scope_note"])
    return 0 if all(r["verdict"] in ("PASS", None) for r in res["lanes"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
