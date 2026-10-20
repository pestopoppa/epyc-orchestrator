#!/usr/bin/env python3
from __future__ import annotations

"""REPL-EMB-2.5: write the frozen arm manifest for the REPL-EMB-2.2 extension arms.

Deterministic infrastructure: no recall or speed claim. The manifest pins (a) the compatibility
matrix and the lock gap, (b) exact revisions and file digests, (c) per-arm encoding config,
(d) one isolated index identity per arm, (e) the count of code-intent queries with blind
code-span gold in the frozen 2.1 query set (``null`` + reason while that set is not frozen),
and (f) the adapter names and the parity-gate script digest. Its sha256 (over the canonical JSON
without the ``manifest_sha256`` field) goes into ``FROZEN-AT-LAUNCH.sha256`` at the 2.2 launch.

Run in the ISOLATED venv (never the orchestrator's locked environment):

    <iso-venv>/bin/python scripts/benchmark/repl_emb_arm_manifest.py \
        --hf-home <dir with hub/models--*/snapshots/*> --gguf-dir <dir> --bge-gguf <file> --out <manifest.json>
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repl_emb_arms as arms  # noqa: E402

REPO_ROOT = arms.REPO_ROOT
ST_MIN, TRANSFORMERS_MIN = "6.1.0", "5.19.0"
MAINTAINER_RUN = {
    "sentence-transformers": "6.1.0",
    "transformers": "5.20.1",
}  # MTEB results, per the 2.5 task text


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def lock_pins(lock: Path) -> dict[str, str]:
    """Versions pinned in the orchestrator's uv.lock (read only; the lock is not modified)."""
    text = lock.read_text()
    out = {}
    for pkg in ("sentence-transformers", "transformers", "torch", "model2vec", "numpy"):
        m = re.search(rf'\[\[package\]\]\nname = "{re.escape(pkg)}"\nversion = "([^"]+)"', text)
        out[pkg] = m.group(1) if m else None
    return out


def snapshot_dir(hf_home: Path, repo: str, revision: str) -> Path:
    d = hf_home / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not d.is_dir():
        raise SystemExit(f"snapshot for {repo}@{revision} not found under {hf_home}")
    return d


def model_digests(snap: Path, names: list[str]) -> dict[str, Any]:
    return {
        n: {"sha256": sha256_file(snap / n), "bytes": (snap / n).stat().st_size}
        for n in names
        if (snap / n).exists()
    }


def count_code_intent(queries: Path | None) -> dict[str, Any]:
    if queries is None or not queries.exists():
        return {
            "status": "query_set_not_frozen",
            "code_intent_with_blind_code_span_gold": None,
            "reason": "REPL-EMB-2.2 has not launched; no frozen 2.1 query set or gold labels exist yet "
            "(data/repl_retrieval_eval/<run-id>/ is absent). Recount at the 2.2 freeze.",
        }
    rows = [json.loads(line) for line in queries.read_text().splitlines() if line.strip()]
    n = sum(
        1
        for r in rows
        if r.get("code_intent") and r.get("gold_spans") and r.get("gold_blind") is True
    )
    return {
        "status": "counted",
        "queries_file": str(queries),
        "queries_file_sha256": sha256_file(queries),
        "n_queries": len(rows),
        "code_intent_with_blind_code_span_gold": n,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    ver = lambda p: md.version(p)  # noqa: E731
    iso = {
        p: ver(p)
        for p in (
            "sentence-transformers",
            "transformers",
            "torch",
            "model2vec",
            "numpy",
            "safetensors",
            "tokenizers",
        )
    }
    lock = lock_pins(REPO_ROOT / "uv.lock")
    st_rt = f"sentence-transformers=={iso['sentence-transformers']}/transformers=={iso['transformers']}/torch=={iso['torch']}"
    ll_rt = args.llama_runtime
    m2v_rt = f"model2vec=={iso['model2vec']}/numpy=={iso['numpy']}"
    specs = arms.arm_specs(st_rt, ll_rt, m2v_rt)

    eg2 = snapshot_dir(args.hf_home, arms.EG2_REPO, arms.EG2_REV)
    pc = snapshot_dir(args.hf_home, arms.POTION_CODE_REPO, arms.POTION_CODE_REV)
    pb = snapshot_dir(args.hf_home, arms.POTION_BASE_REPO, arms.POTION_BASE_REV)
    eg2_cfg = json.loads((eg2 / "config.json").read_text())
    txt = eg2_cfg["text_config"]
    gg = {
        n: {"sha256": sha256_file(args.gguf_dir / n), "bytes": (args.gguf_dir / n).stat().st_size}
        for n in ("embeddinggemma-2-BF16.gguf", "embeddinggemma-2-Q8_0.gguf")
        if (args.gguf_dir / n).exists()
    }
    bge = {
        "path": str(args.bge_gguf),
        "sha256": sha256_file(args.bge_gguf),
        "bytes": args.bge_gguf.stat().st_size,
    }

    gate_py = Path(__file__).resolve().parent / "repl_emb_parity_gate.py"
    man: dict[str, Any] = {
        "schema": "repl-emb-arm-manifest/1",
        "task": "REPL-EMB-2.5",
        "scope": "deterministic infrastructure; no recall or speed claim",
        "orchestrator_commit": args.orchestrator_commit,
        "compat_matrix": {
            "required": {
                "sentence-transformers": f">={ST_MIN}",
                "transformers": f">={TRANSFORMERS_MIN}",
            },
            "isolated_venv": iso,
            "production_lock_uv_lock": lock,
            "lock_gap": {
                "statement": "The production uv.lock pins sentence-transformers "
                f"{lock['sentence-transformers']}, transformers {lock['transformers']}, torch {lock['torch']}; "
                f"EmbeddingGemma 2 needs sentence-transformers >= {ST_MIN} and transformers >= {TRANSFORMERS_MIN}. "
                "The lock is NOT touched; every EmbeddingGemma 2 ST lane runs in the isolated venv.",
                "maintainer_run_versions": MAINTAINER_RUN,
                "isolated_matches_maintainer_run": all(
                    iso[k] == v for k, v in MAINTAINER_RUN.items()
                ),
            },
            "hf_card_requirement": "config_sentence_transformers.json requires sentence-transformers >= 6.1.0",
        },
        "revisions": {
            arms.EG2_REPO: {
                "revision": arms.EG2_REV,
                "files": model_digests(
                    eg2,
                    [
                        "model.safetensors",
                        "config.json",
                        "config_sentence_transformers.json",
                        "tokenizer.json",
                        "tokenizer.model",
                        "modules.json",
                    ],
                ),
            },
            arms.POTION_CODE_REPO: {
                "revision": arms.POTION_CODE_REV,
                "files": model_digests(pc, ["model.safetensors", "config.json", "tokenizer.json"]),
            },
            arms.POTION_BASE_REPO: {
                "revision": arms.POTION_BASE_REV,
                "resolved": "HF API main HEAD on 2026-10-20 (lastModified 2026-03-27)",
                "files": model_digests(pb, ["model.safetensors", "config.json", "tokenizer.json"]),
            },
            arms.EG2_GGUF_REPO: {"revision": arms.EG2_GGUF_REV, "files": gg},
            "incumbent:bge-large-en-v1.5-f16.gguf": bge,
        },
        "eg2_model_facts": {
            "layers": txt["num_hidden_layers"],
            "hidden_size": txt["hidden_size"],
            "intermediate_size": txt["intermediate_size"],
            "heads": txt["num_attention_heads"],
            "kv_heads": txt["num_key_value_heads"],
            "head_dim": txt["head_dim"],
            "vocab_size": txt["vocab_size"],
            "embedding_dim": txt["embedding_dim"],
            "sliding_window": txt["sliding_window"],
            "max_position_embeddings_config_json": txt["max_position_embeddings"],
            "context_window_model_card": 8192,
            "source": "config.json of the pinned revision (read from the snapshot, not recalled)",
        },
        "encoding": {
            "eg2": {
                "query": arms.EG2_QUERY_PROMPT + "{q}",
                "query_code_intent": arms.EG2_CODE_QUERY_PROMPT + "{q}",
                "document": arms.EG2_DOC_PROMPT + "{chunk}",
                "code_intent_rule": "CodeRetrieval query form only for queries labelled code-intent in the frozen 2.1 query set; "
                "documents always use the Document form (the card's code-search row uses `title: {title or filename} | text: {code}`; "
                "the task text fixes `title: none` and the harness follows the task text).",
                "pooling": "mean (1_Pooling/config.json include_prompt=true)",
                "dtype": "FP32 on CPU; BF16 reference lane; FP16 is never used (card: NaN or silent degradation)",
                "mrl": {
                    "dims": [768, 256],
                    "method": "keep leading dims then L2 re-normalise (applied by the adapter, not by ST truncate_dim)",
                },
                "effective_input_cap": {
                    "chunk_chars": arms.load_context_search().DENSE_INPUT_MAX_CHARS,
                    "note": "the shipped index clips every embedded text to 800 chars (~200-400 tokens), far below the card's 8,192 tokens "
                    "and config.json's 262,144 positions; the cap actually in force is the character clip. "
                    "ST max_seq_length and the llama-server n_ctx/n_ubatch actually used are recorded per run by the parity gate.",
                },
                "text_only_loading": {
                    "config_kwargs": {"vision_config": None, "audio_config": None},
                    "card_param_count": "270M",
                },
            },
            "static": {
                "potion-code-16m-v2": "256-d lookup-and-mean, L2",
                "potion-base-8m": "256-d lookup-and-mean, L2",
                "prefixes": "none",
                "runtime": m2v_rt,
            },
        },
        "prompt_side_mechanism": {
            "problem": "ContextSearchIndex only prefixes queries (query_text + BGE-only _QUERY_PREFIXES) and sends documents unprefixed.",
            "adapter_rule": "the adapter prefixes both sides; it tells them apart by the admission_wait_s the shipped index passes "
            "(INDEX_ADMISSION_WAIT_S for chunk batches, QUERY_ADMISSION_WAIT_S for the query); repl_emb_arms.side_for asserts the constants differ.",
            "constants": {
                "INDEX_ADMISSION_WAIT_S": arms.load_context_search().INDEX_ADMISSION_WAIT_S,
                "QUERY_ADMISSION_WAIT_S": arms.load_context_search().QUERY_ADMISSION_WAIT_S,
            },
            "shipped_code_modified": False,
        },
        "arms": [
            {
                **{k: v for k, v in s.__dict__.items()},
                "index_identity": s.index_identity(),
                "dense_embedder_model_id": s.model_id(),
                "adapter": {
                    "st": "SentenceTransformersAdapter",
                    "gguf": "LlamaServerAdapter",
                    "static": "Model2VecAdapter",
                }[s.family],
            }
            for s in specs
        ],
        "query_set": count_code_intent(args.queries),
        "parity_gate": {
            "script": str(gate_py.relative_to(REPO_ROOT)),
            "script_sha256": sha256_file(gate_py),
            "arms_module_sha256": sha256_file(Path(arms.__file__)),
            "predeclared": {
                "zero_nonfinite": True,
                "st_bf16_vs_st_fp32_min_cos": 0.99,
                "gguf_bf16_min_cos": 0.99,
                "gguf_q8_0_min_cos": 0.98,
                "min_chunks": 200,
                "question_set": "the 45 UFH-07 questions",
            },
        },
        "frozen_context_search": {
            "path": "src/repl_environment/context_search.py",
            "sha256": sha256_file(REPO_ROOT / "src/repl_environment/context_search.py"),
            "constants": {
                k: getattr(arms.load_context_search(), k)
                for k in (
                    "CHUNK_MAX_CHARS",
                    "DENSE_INPUT_MAX_CHARS",
                    "MAX_DENSE_CHUNKS",
                    "DENSE_BATCH_CHUNKS",
                    "INDEX_ASSUMED_TEXTS_PER_S",
                    "RRF_K",
                    "CANDIDATES_PER_RANKER",
                )
            },
        },
    }
    canon = json.dumps(man, sort_keys=True, separators=(",", ":"))
    man["manifest_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return man


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hf-home", type=Path, required=True)
    ap.add_argument("--gguf-dir", type=Path, required=True)
    ap.add_argument("--bge-gguf", type=Path, required=True)
    ap.add_argument(
        "--queries", type=Path, default=None, help="frozen 2.1 query set JSONL, if one exists"
    )
    ap.add_argument(
        "--llama-runtime",
        default="llama-server experimental/gemma-embedding2-on-v10 (CPU, build-cpu)",
    )
    ap.add_argument("--orchestrator-commit", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    man = build(a)
    a.out.write_text(json.dumps(man, indent=1, sort_keys=True))
    print(f"wrote {a.out} sha256={man['manifest_sha256']}")
    return 0


def verify(path: Path) -> bool:
    """True when the saved manifest's digest matches its content."""
    man = json.loads(path.read_text())
    want = man.pop("manifest_sha256")
    return (
        hashlib.sha256(json.dumps(man, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        == want
    )


if __name__ == "__main__":
    raise SystemExit(main())
