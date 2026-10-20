#!/usr/bin/env python3
"""Project the finished REPL-EMB-2.2 offline retrieval run through the belief-kernel write side.

Task VB-UFH12-RETR / REPL-EMB-2.4. The 2.2 harness (``repl_emb_eval.py``) ran before the write side
was wired into it, so this script is the post-hoc projector: it reads the run's ``analysis.json``
(schema ``repl-emb-analysis/3``, produced by ``repl_emb_eval.py analyze``) and writes ONE immutable
``<run_id>.record.json`` plus its ``<run_id>.belief_measurements.jsonl`` sidecar through
:class:`repl_retrieval_capture.RetrievalRun`. It computes nothing: every hit count, Wilson interval
and paired-bootstrap number is the analysis's own, and the writer re-derives value == hits/n and
diff == recall(A) - recall(B) or refuses.

WHAT IS WRITTEN (recall only; the speed block was dropped by operator ruling and every timing in the
run is contended, so NO cost row is written)::

    recall_at_{1,3,5,10}   every arm/variant x stratum ``all`` (the PRIMARY deciding set, n=346) and
                           the per-stratum rows probe45 / agent_nav / synthetic_spill (the
                           unflagged members, as the analysis scored them)
    recall_at_5_paired_diff  each dense / hybrid / weighted-hybrid variant minus L (grep) on ``all``
                           (the analysis's bootstrap: diff, ci95, n, resamples, seed), plus ColGREP
                           minus L on its own stratum probe45 (ColGREP is code-only, so it exists
                           only there and never enters ``all``)

The kill/selection verdict the analysis computed is NOT projected (the writer never computes one).
The rule is :func:`registered_rule` unchanged: the writer's rule check admits no extra key (the
``trace_min=75`` floor and the pre-run amendment (q) primary-set definition are carried in the
config echo instead) and refuses an ``amendment`` whose values equal the registration, so no
amendment is, or can be, written.

IDENTITIES come from the frozen harness itself: ``repl_emb_eval.eval_specs`` rebuilds every dense
arm's :class:`ArmSpec` from the frozen arm manifest and each rebuilt ``index_identity`` digest must
equal the digest the run recorded (``config-echo.json`` and ``armmeta-*.json``) or this refuses. The
lexical arms carry explicit nulls. ColGREP records no model identity at run time, so its identity is
resolved post-hoc and says so (``COLGREP_IDENTITY_BASIS``, echoed on every ColGREP row).

Run once per run id; the record is immutable. ``python3 repl_emb_write_belief.py --help``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
import socket
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import repl_emb_eval as ev  # noqa: E402
import repl_retrieval_capture as cap  # noqa: E402

REPO_ROOT = HERE.parents[1]
RUN_ID = "repl-emb-22-20261020"
DEFAULT_RUN_DIR = Path("/mnt/raid0/llm/tmp/repl-emb-22-20261020/run")
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "repl_retrieval_eval" / RUN_ID
DEFAULT_ARM_MANIFEST = ev.DEFAULT_ARM_MANIFEST

ANALYSIS_SCHEMA = "repl-emb-analysis/3"
POOLED = cap.POOLED_CATEGORY  # "all": the PRIMARY deciding set
POOLED_IN_ANALYSIS = "pooled"  # the analysis's own key for the pooled set
STRATA = ("probe45", "agent_nav", "synthetic_spill")  # the analysis's group names, report order
KS = cap.RECALL_KS
PRIMARY_K = 5
BASELINE_ARMS = frozenset(
    {"grep", "bm25", "bge-large"}
)  # operator ruling; every other arm CANDIDATE
LEXICAL_ARMS = tuple(ev.LEXICAL_ARMS)
COLGREP = "colgrep"
WEIGHTED = "whybrid"

#: small evidence files copied beside the record (NEVER the per-query rows or any query/document text)
EVIDENCE_FILES = (
    "analysis.json",
    "analysis.md",
    "config-echo.json",
    "floor-check.json",
    "predeclared-interpretations.json",
    "FREEZE-RULES.md",
    "DEVIATIONS.md",
    "FROZEN-AT-LAUNCH.sha256",
    "bundles-summary.json",
)
EVIDENCE_GLOBS = ("armmeta-*.json",)

PIPELINES = {
    "grep": "harness grep_rank: distinct stopword-filtered tokens (len>=3) as substrings of the "
    "lowercased 800-char chunks, ranked by term count, top-10",
    "bm25": "shipped context_search BM25 (k1=1.2, b=0.75) over the same chunks, top-10",
    "dense": "dense-only: cosine over the one query embedding the shipped search embedded, top-10",
    "hybrid": "shipped ContextSearchIndex.search: BM25 + dense, RRF k=60, 64 candidates per "
    "ranker, top-10",
    WEIGHTED: "weighted RRF 0.75 dense / 0.25 BM25, k=60, over the same two 64-candidate rankings "
    "(static arms only; reported, non-deciding)",
    COLGREP: "colgrep-1.2.0 search -k 10 --alpha 0.95 (ColBERT MaxSim + FTS5 keyword fusion), "
    "CPU-forced, probe45 only (code-only: tree-sitter)",
}

DEVIATIONS_SUMMARY = (
    "BGE recipe changed mid-run: queries 1-57 on the pool recipe (-np 1 -c 512, co-located on q0 "
    "CPUs 0-23, nice 19, no region claim), the rest on CPUs 80-87,176-183 under another session's "
    "claim with -np 4 -c 2048 -b 2048 -ub 2048 -t 16 (same GGUF; batch composition can change the "
    "last bits). Other arms on q2 CPUs 48-71 (tag repl-emb-22-recall); ColGREP on SMT siblings "
    "144-167 beside the EG2 arms. All timings are CONTAMINATED; none is claimed."
)
HOST_LEGS = {
    "bge-large queries 1-57": "0-23 (q0, nice 19, no region claim)",
    "bge-large resumed": "80-87,176-183 (q3 claim by agreement)",
    "grep/bm25/EG2/Granite/potion": "48-71 (q2, region-lock tag repl-emb-22-recall)",
    "colgrep": "144-167 (q2 SMT siblings)",
}
HOST_CPU_LIST = "0-23,48-71,80-87,144-167,176-183"

#: ColGREP records no model identity (no model/revision/dtype in armmeta or the invocation log), so
#: the identity the writer requires is resolved POST HOC. Every fact below was read from disk on
#: 2026-10-20 after the run; the two things that were not recorded are named as unrecorded.
COLGREP_MODEL = "lightonai/LateOn-Code-edge"
COLGREP_DIM = 48
COLGREP_REVISION = "unrecorded-at-run"
COLGREP_DTYPE = "plaid-nbits4;onnx-precision-unrecorded"
COLGREP_PROMPTS = {"query_prefix": "[Q] ", "document_prefix": "[D] "}
COLGREP_IDENTITY_BASIS = {
    "status": "RESOLVED POST HOC; model, revision and ONNX precision were NOT captured at run time",
    "dim_and_nbits": "embedding_dim 48 and PLAID nbits 4 read from the run's own three indices "
    "(~/.local/share/colgrep/indices/{mwaskom__seaborn-e1c8a8c1,pylint-dev__pylint-e60d8fe0,"
    "pytest-dev__pytest-4c00993c}/index/metadata.json, cli_version 1.2.0, 15 searches each)",
    "model": "lightonai/LateOn-Code-edge: the only ColGREP model in either HF cache and the only "
    "one whose onnx_config.json embedding_dim is 48 (LateOn-Code 130M is 128-d)",
    "model_files_sha256": {
        "onnx_config.json": "fa4fef89820dcdc33c5504c62c1d5efc19603cfbfebf02368a70d51a4dbe6651",
        "model.onnx": "ac5a92a685512b163c3c591438f518379309d2a98c4818a9c6e2986f789dc8ef",
        "model_int8.onnx": "eac35bdaa862e2762e6455337f7a3e704b05dbc4259f00929fcc8e10207f11c7",
    },
    "hf_snapshots_byte_identical_blobs": [
        "4bcdf5ed93f791259eb130b577a240f753d68dd8 (~/.cache/huggingface)",
        "07ef20f406c86badca122464808f4cac2f6e4b25 (/mnt/raid0/llm/hf-home)",
    ],
    "unrecorded": ["HF revision actually loaded", "ONNX precision actually used (fp32 or int8)"],
    "prompt_hash": "sha256 of the canonical JSON of the model's own query/document prefixes",
}


class WriteError(RuntimeError):
    """The projector refuses; nothing is written."""


# ------------------------------------------------------------------ small helpers


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def split_variant(variant: str) -> tuple[str, str, str]:
    """(writer arm_id, mode, pipeline key) for one analysis variant name."""
    if variant in LEXICAL_ARMS:
        return variant, "lexical", variant
    if variant == COLGREP:
        return COLGREP, "hybrid", COLGREP
    base, _, kind = variant.rpartition(".")
    if not base or kind not in ("dense", "hybrid", WEIGHTED):
        raise WriteError(f"unrecognised analysis variant {variant!r}")
    if kind == "dense":
        return base, "dense", "dense"
    # a weighted-RRF pipeline is a different claim from the shipped RRF one but the writer has one
    # "hybrid" mode: it gets its own arm_id so the two never collide on a locator
    return (f"{base}.{WEIGHTED}" if kind == WEIGHTED else base), "hybrid", kind


def role_of(base_arm: str) -> str:
    return "BASELINE" if base_arm in BASELINE_ARMS else "CANDIDATE"


def whole_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise WriteError(f"{what} is not a whole number: {value!r}")
    return int(value)


# ------------------------------------------------------------------ arm identities


def dense_identity(
    spec: Any, armmeta: Mapping[str, Any], config_echo: Mapping[str, Any]
) -> dict[str, Any]:
    """An arm identity rebuilt from the frozen spec; refuses unless the digest matches the run's."""
    arm_id = spec.arm_id
    dim = whole_int(armmeta["dim"], f"{arm_id} armmeta dim")
    if dim != spec.dim:  # Granite reads its true width from the first response (autodim)
        spec = dataclasses.replace(spec, dim=dim, native_dim=dim)
    ident = spec.index_identity()
    recorded = {
        "armmeta": armmeta["identity_digest"],
        "config-echo": config_echo["stable"]["arm_identities"][arm_id]["digest"],
    }
    wrong = {src: d for src, d in recorded.items() if d != ident["digest"]}
    if wrong:
        raise WriteError(
            f"{arm_id}: rebuilt index identity {ident['digest']} differs from the run's {wrong}"
        )
    return {
        "arm_id": arm_id,
        "model": spec.model_repo,
        "revision": spec.revision,
        "dim": spec.dim,
        "dtype_or_quant": spec.dtype,
        "runtime": spec.runtime,
        "prompt_hash": spec.prompt_hash(),
        "index_identity_digest": ident["digest"],
        "role": role_of(arm_id),
    }


def lexical_identity(arm_id: str, config_echo: Mapping[str, Any]) -> dict[str, Any]:
    stable = config_echo["stable"]
    pre = stable["predeclared"]
    consts = pre["context_search_constants"]
    harness_sha = stable["harness_sha256"]["repl_emb_eval.py"]
    cs_sha = stable["context_search_sha256"]
    if arm_id == "grep":
        runtime = f"repl_emb_eval.py@{harness_sha[:12]}/grep_rank over context_search chunks"
        basis: dict[str, Any] = {
            "arm_id": arm_id,
            "harness_sha256": harness_sha,
            "context_search_sha256": cs_sha,
            "chunk_max_chars": consts["CHUNK_MAX_CHARS"],
            "grep_stopwords_sha256": pre["grep_stopwords_sha256"],
            "top_k": pre["top_k"],
        }
    elif arm_id == "bm25":
        runtime = (
            f"context_search.py@{cs_sha[:12]}/bm25 (k1={consts['BM25_K1']}, b={consts['BM25_B']})"
        )
        basis = {
            "arm_id": arm_id,
            "context_search_sha256": cs_sha,
            "chunk_max_chars": consts["CHUNK_MAX_CHARS"],
            "bm25_k1": consts["BM25_K1"],
            "bm25_b": consts["BM25_B"],
            "top_k": pre["top_k"],
        }
    else:
        raise WriteError(f"no lexical identity for {arm_id!r}")
    return {
        "arm_id": arm_id,
        "model": None,
        "revision": None,
        "dim": None,
        "dtype_or_quant": None,
        "runtime": runtime,
        "prompt_hash": None,
        "index_identity_digest": cap.content_hash(basis),
        "role": role_of(arm_id),
    }


def colgrep_identity(config_echo: Mapping[str, Any]) -> dict[str, Any]:
    colgrep = config_echo["stable"]["predeclared"]["colgrep"]
    runtime = (
        f"colgrep-1.2.0 sha256:{colgrep['expected_sha256'][:12]} "
        f"NEXT_PLAID_FORCE_CPU=1 alpha={colgrep['alpha']}"
    )
    prompt_hash = cap.content_hash(COLGREP_PROMPTS)
    digest = cap.content_hash(
        {
            "arm_id": COLGREP,
            "binary_sha256": colgrep["expected_sha256"],
            "alpha": colgrep["alpha"],
            "env": colgrep["env"],
            "model": COLGREP_MODEL,
            "dim": COLGREP_DIM,
            "dtype_or_quant": COLGREP_DTYPE,
            "prompt_hash": prompt_hash,
        }
    )
    return {
        "arm_id": COLGREP,
        "model": COLGREP_MODEL,
        "revision": COLGREP_REVISION,
        "dim": COLGREP_DIM,
        "dtype_or_quant": COLGREP_DTYPE,
        "runtime": runtime,
        "prompt_hash": prompt_hash,
        "index_identity_digest": digest,
        "role": role_of(COLGREP),
    }


def resolve_identities(
    variants: Sequence[str],
    run_dir: Path,
    manifest: Mapping[str, Any],
    config_echo: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """variant -> writer arm identity (the weighted pipeline of an arm gets its own arm_id)."""
    llama_runtime, m2v_runtime = ev.runtimes_from_manifest(manifest)
    specs = ev.eval_specs(llama_runtime, m2v_runtime)
    out: dict[str, dict[str, Any]] = {}
    for variant in variants:
        arm_id, _, kind = split_variant(variant)
        base = variant.split(".")[0]
        if base in LEXICAL_ARMS:
            out[variant] = lexical_identity(base, config_echo)
        elif base == COLGREP:
            out[variant] = colgrep_identity(config_echo)
        elif base in specs:
            armmeta = read_json(run_dir / f"armmeta-{base}.json")
            ident = dense_identity(specs[base], armmeta, config_echo)
            out[variant] = {**ident, "arm_id": arm_id}
        else:
            raise WriteError(f"variant {variant!r} has no arm spec")
    return out


# ------------------------------------------------------------------ inputs, digests, echo


def map_query_files(inputs: Mapping[str, Any], labelled: Sequence[Mapping[str, Any]]) -> dict:
    """stratum -> the frozen query file whose id set IS that stratum's id set (verified, not assumed)."""
    groups: dict[str, set[str]] = {}
    for q in labelled:
        groups.setdefault(q["group"], set()).add(q["id"])
    out: dict[str, Path] = {}
    for path in [inputs["probe45"], *inputs["trace_queries"]]:
        ids = {row["id"] for row in ev.read_jsonl(path)}
        match = [g for g in STRATA if groups.get(g) == ids]
        if len(match) != 1 or match[0] in out:
            raise WriteError(f"query file {path} does not match exactly one unclaimed stratum")
        out[match[0]] = Path(path)
    missing = [g for g in STRATA if g not in out]
    if missing:
        raise WriteError(f"no frozen query file for strata {missing}")
    return out


def map_corpus_files(
    run_dir: Path, inputs: Mapping[str, Any], labelled: Sequence[Mapping[str, Any]]
) -> dict:
    """stratum -> the corpus artefact: the probe45 listing, or the bundles file holding its bundles."""
    out: dict[str, Path] = {"probe45": run_dir / "corpus-digest-probe45.txt"}
    for group in ("agent_nav", "synthetic_spill"):
        need = {q["bundle"] for q in labelled if q["group"] == group}
        found = [Path(p) for p in inputs["bundles_json"] if need <= set(read_json(p))]
        if len(found) != 1:
            raise WriteError(f"{group}: {len(found)} bundles files hold its bundles, expected 1")
        out[group] = found[0]
    return out


def invocation_window(run_dir: Path) -> tuple[str, dict[str, list[str]]]:
    """(earliest invocation timestamp as ISO UTC, arm -> served endpoints) from invocations.jsonl."""
    stamps: list[str] = []
    endpoints: dict[str, list[str]] = {}
    for line in (run_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stamps.append(row["ts"])
        for arm, url in sorted(row.get("endpoints", {}).items()):
            if url not in endpoints.setdefault(arm, []):
                endpoints[arm].append(url)
    if not stamps:
        raise WriteError("invocations.jsonl has no invocation")
    started = min(datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps)
    return started.astimezone(UTC).isoformat(), endpoints


def build_config_echo(
    analysis: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    run_dir: Path,
    primary: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Sequence[str]],
    writer_sha256: str,
) -> dict[str, Any]:
    """The compact run-level echo carried on every recall row."""
    stable = cfg["stable"]
    pre = stable["predeclared"]
    stats = pre["stats"]
    consts = pre["context_search_constants"]
    primary_sets = analysis["decision_sets"]
    failed_primary = analysis["completeness"]["bge-large.dense"]["n_failed"]
    failed_all = primary_sets["pooled_flagged_included"]["completeness"]["bge-large.dense"][
        "n_failed"
    ]
    by_group: dict[str, list[str]] = {}
    for q in primary:
        by_group.setdefault(q["group"], []).append(q["id"])
    return {
        "recall_only": "REPL-EMB-2.2 post-hoc projection of analysis.json; no cost row (speed "
        "block dropped by operator ruling, timings contended)",
        "harness": {
            "git_sha": cfg["git_sha"],
            "dirty": cfg["harness_files_dirty"],
            "repl_emb_eval_sha256": stable["harness_sha256"]["repl_emb_eval.py"],
            "context_search_sha256": stable["context_search_sha256"],
            "writer_sha256": writer_sha256,
        },
        "constants": {
            "context_search": {k: consts[k] for k in sorted(consts)},
            "top_k": pre["top_k"],
            "weighted_hybrid": pre["weighted_hybrid"],
            "colgrep_alpha": pre["colgrep"]["alpha"],
        },
        "stats": {
            "k_primary": stats["k_primary"],
            "M": stats["M"],
            "X": stats["X"],
            "n_min": stats["n_min"],
            "trace_min": stats["trace_min"],
            "ci": "recall: wilson-95; paired diff: by-query percentile bootstrap, 95%",
            "resamples": stats["resamples"],
            "seed": stats["seed"],
        },
        "decision_set": {
            "id": "primary",
            "definition": "pre-run amendment (q) 2026-10-20: scored non-trivial rows minus every "
            "answer_named_verbatim row (probe45 unflagged, kept whole); 'all' is this set, the "
            "stratum rows its parts; trivial_head is outside; the registered rule is written "
            "unchanged and trace_min/(q) travel here",
            "n": analysis["n_deciding"],
            "n_by_stratum": {g: len(ids) for g, ids in sorted(by_group.items())},
            "flagged_excluded": analysis["n_flagged_excluded"],
            "trace_rows": analysis["floor_check"]["n_trace"],
            "trace_min": analysis["floor_check"]["trace_min"],
            "primary_ids_sha256": ids_digest([q["id"] for q in primary]),
        },
        "failures_count_as_misses": analysis["failures_count_as_misses"],
        "bge_failures": {
            "failed_queries": failed_all,
            "of_scored_nontrivial": primary_sets["pooled_flagged_included"]["n_deciding"],
            "failed_in_primary_set": failed_primary,
            "cause": "index_build:HTTPError on the affected bundles; a miss for BOTH "
            "bge-large.dense and bge-large.hybrid",
        },
        "synthetic_stratum": analysis["bias_note"],
        "endpoints": {arm: list(urls) for arm, urls in sorted(endpoints.items())},
        "deviations": DEVIATIONS_SUMMARY,
        "deviations_md_sha256": cap.file_sha256(run_dir / "DEVIATIONS.md"),
    }


# ------------------------------------------------------------------ analysis -> writer calls


def check_analysis(analysis: Mapping[str, Any]) -> None:
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise WriteError(f"analysis schema {analysis.get('schema')!r} is not {ANALYSIS_SCHEMA}")
    primary = analysis["decision_sets"]["primary"]
    for key in ("recall", "bootstrap_recall5"):
        if analysis[key] != primary[key]:
            raise WriteError(f"top-level {key} is not the PRIMARY set's {key}")
    if not analysis["floor_check"]["ok"]:
        raise WriteError("the analysis's own floor check failed")
    if analysis["lexical_baseline"]["L"] != "grep":
        raise WriteError("L is not grep: the paired baseline would have to be rewired")


def recall_cell(analysis: Mapping[str, Any], variant: str, stratum: str, k: int) -> dict:
    """One analysis cell with its hits as an int; refuses a cell that does not re-derive."""
    cell = analysis["recall"][variant][stratum][str(k)]
    hits, n = whole_int(cell["hits"], "hits"), whole_int(cell["n"], "n")
    if not math.isclose(cell["recall"], hits / n, rel_tol=1e-9, abs_tol=1e-12):
        raise WriteError(f"{variant}/{stratum}/k{k}: recall does not re-derive from hits/n")
    if cell.get("ci") != "wilson-95":
        raise WriteError(f"{variant}/{stratum}/k{k}: interval is not wilson-95")
    return {"hits": hits, "n": n, "ci95": [cell["ci_lo"], cell["ci_hi"]]}


def check_partition(analysis: Mapping[str, Any], variant: str) -> None:
    """The strata partition the pooled set (hits and n), so ``all`` and the strata cannot disagree."""
    for k in KS:
        pooled = recall_cell(analysis, variant, POOLED_IN_ANALYSIS, k)
        cells = [recall_cell(analysis, variant, s, k) for s in STRATA]
        if (sum(c["hits"] for c in cells), sum(c["n"] for c in cells)) != (
            pooled["hits"],
            pooled["n"],
        ):
            raise WriteError(f"{variant} k{k}: the strata do not sum to the pooled set")


def add_recall_rows(
    run: cap.RetrievalRun,
    analysis: Mapping[str, Any],
    variants: Sequence[str],
    identities: Mapping[str, Mapping[str, Any]],
    echo: Mapping[str, Any],
) -> int:
    count = 0
    for variant in variants:
        _, mode, pipeline = split_variant(variant)
        if variant == COLGREP:
            strata = [("probe45", "probe45")]  # code-only: it exists on probe45 and nowhere else
        else:
            check_partition(analysis, variant)
            strata = [(POOLED, POOLED_IN_ANALYSIS)] + [(s, s) for s in STRATA]
        row_echo = {
            **echo,
            "variant": variant,
            "pipeline": PIPELINES[pipeline],
            "set": "primary (unflagged)",
        }
        if variant == COLGREP:
            row_echo["identity_basis"] = COLGREP_IDENTITY_BASIS
            row_echo["scope"] = "probe45 only: never in 'all' (its n is 45, not 346)"
        for writer_stratum, analysis_stratum in strata:
            for k in KS:
                cell = recall_cell(analysis, variant, analysis_stratum, k)
                run.add_recall(
                    identities[variant],
                    writer_stratum,
                    mode,
                    k,
                    "recall",
                    cell["hits"] / cell["n"],
                    cell["n"],
                    cell["hits"],
                    cell["ci95"],
                    row_echo,
                )
                count += 1
    return count


def add_paired_rows(
    run: cap.RetrievalRun,
    analysis: Mapping[str, Any],
    variants: Sequence[str],
    identities: Mapping[str, Mapping[str, Any]],
) -> int:
    """Each dense / hybrid / weighted variant minus L on ``all``; ColGREP minus L on probe45."""
    count = 0
    lexical = analysis["lexical_baseline"]["L"]
    for variant in variants:
        _, mode, _ = split_variant(variant)
        if mode == "lexical":
            continue
        boot = analysis["bootstrap_recall5"][variant]["vs_L"]
        if boot["L"] != lexical:
            raise WriteError(f"{variant}: bootstrap baseline {boot['L']!r} is not {lexical!r}")
        writer_stratum, analysis_stratum = (
            ("probe45", "probe45") if variant == COLGREP else (POOLED, POOLED_IN_ANALYSIS)
        )
        a = recall_cell(analysis, variant, analysis_stratum, PRIMARY_K)
        b = recall_cell(analysis, lexical, analysis_stratum, PRIMARY_K)
        if not (a["n"] == b["n"] == boot["n"]):
            raise WriteError(f"{variant}: paired n differs from its recall rows")
        # derive the diff from the hits so it is exactly recall(A) - recall(B), then require the
        # analysis's own mean to agree to float rounding
        diff = (a["hits"] - b["hits"]) / a["n"]
        if not math.isclose(diff, boot["mean_diff"], rel_tol=0.0, abs_tol=1e-9):
            raise WriteError(f"{variant}: bootstrap mean_diff {boot['mean_diff']} != {diff}")
        run.add_paired_diff(
            identities[variant],
            identities[lexical],
            writer_stratum,
            PRIMARY_K,
            diff,
            [boot["ci_lo"], boot["ci_hi"]],
            a["n"],
            boot["resamples"],
            boot["seed"],
            mode_a=mode,
            mode_b="lexical",
        )
        count += 1
    return count


# ------------------------------------------------------------------ the write


def copy_evidence(run_dir: Path, out_dir: Path) -> list[str]:
    names = list(EVIDENCE_FILES)
    for pattern in EVIDENCE_GLOBS:
        names += sorted(p.name for p in run_dir.glob(pattern))
    for name in names:
        src, dst = run_dir / name, out_dir / name
        if dst.exists() and cap.file_sha256(dst) != cap.file_sha256(src):
            raise WriteError(f"{dst} exists and differs from {src}: evidence is not overwritten")
        shutil.copyfile(src, dst)
    return names


def write_run(
    run_dir: Path = DEFAULT_RUN_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    arm_manifest: Path = DEFAULT_ARM_MANIFEST,
    run_id: str = RUN_ID,
    harness_path: Path | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Project one finished run. Returns a summary; raises WriteError/CaptureError, writing no
    record on refusal (the evidence copy may remain and is idempotent)."""
    run_dir, out_dir, arm_manifest = Path(run_dir), Path(out_dir), Path(arm_manifest)
    harness = Path(harness_path) if harness_path else Path(ev.__file__).resolve()
    analysis = read_json(run_dir / "analysis.json")
    check_analysis(analysis)
    cfg = read_json(run_dir / "config-echo.json")
    if cap.file_sha256(harness) != cfg["stable"]["harness_sha256"]["repl_emb_eval.py"]:
        raise WriteError(f"{harness} is not the harness the run froze: its sha256 differs")
    inputs = read_json(run_dir / "inputs.json")
    manifest = read_json(arm_manifest)
    if Path(inputs["arm_manifest"]).resolve() != arm_manifest.resolve():
        raise WriteError("the run used a different arm manifest than the one offered")
    record_path = cap.record_path_for(out_dir, run_id)
    for target in (record_path, cap.sidecar_path(record_path)):
        if target.exists():
            raise WriteError(f"{target} already exists: records are immutable")

    labelled = ev.read_jsonl(run_dir / "queries-labelled.jsonl")
    primary = ev.primary_rows(labelled)
    if len(primary) != analysis["n_deciding"]:
        raise WriteError(
            f"queries-labelled.jsonl yields {len(primary)} primary rows, analysis {analysis['n_deciding']}"
        )
    started_at, endpoints = invocation_window(run_dir)
    query_files = map_query_files(inputs, labelled)
    corpus_files = map_corpus_files(run_dir, inputs, labelled)
    labelled_sha = cap.file_sha256(run_dir / "queries-labelled.jsonl")
    label_digests = {"queries-labelled.jsonl": labelled_sha}
    if (run_dir / "gold-check.json").is_file():
        label_digests["gold-check.json"] = cap.file_sha256(run_dir / "gold-check.json")
    query_digests = {s: cap.file_sha256(query_files[s]) for s in STRATA}
    query_digests[POOLED] = labelled_sha  # the pooled set is cut from the frozen labelled file
    corpus_digests = {s: cap.file_sha256(corpus_files[s]) for s in STRATA}

    variants = list(analysis["variants"])
    identities = resolve_identities(variants, run_dir, manifest, cfg)
    echo = build_config_echo(
        analysis,
        cfg,
        run_dir=run_dir,
        primary=primary,
        endpoints=endpoints,
        writer_sha256=cap.file_sha256(Path(__file__).resolve()),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_evidence(run_dir, out_dir)
    ticks = iter([started_at])
    run = cap.RetrievalRun(
        out_dir,
        run_id,
        cap.registered_rule(),
        corpus_digests,
        query_digests,
        label_digests,
        cap.file_sha256(arm_manifest),
        {
            "hostname": hostname or socket.gethostname(),
            "cpu_list": HOST_CPU_LIST,
            "legs": HOST_LEGS,
            "contended": True,
            "note": "recall-only run: no timing leg, no region-lock claim is asserted for the run "
            "as a whole",
        },
        frozen_at_launch_path=out_dir / "FROZEN-AT-LAUNCH.sha256",
        harness_path=harness,
        producer="repl_emb_eval.analyze projected by repl_emb_write_belief",
        # first tick = the run's real start; the finalize tick = the moment of writing
        clock=lambda: next(ticks, None) or cap.utc_now(),
    )
    n_recall = add_recall_rows(run, analysis, variants, identities, echo)
    n_paired = add_paired_rows(run, analysis, variants, identities)
    sidecar = run.finalize()
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    problems = cap.validate_rows(rows)
    if problems:
        raise WriteError("the written sidecar fails its own validator: " + "; ".join(problems))
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    return {
        "run_id": run_id,
        "record": str(record_path),
        "record_sha256": cap.file_sha256(record_path),
        "sidecar": str(sidecar),
        "sidecar_sha256": cap.file_sha256(sidecar),
        "rows": len(rows),
        "rows_by_kind": kinds,
        "recall_rows": n_recall,
        "paired_rows": n_paired,
        "amendment": rows[0]["extra"]["provenance"]["rule"].get("amendment"),
        "started_at": rows[0]["extra"]["provenance"]["run"]["started_at"],
        "finished_at": rows[0]["extra"]["provenance"]["run"]["finished_at"],
        "evidence_copied": copied,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--arm-manifest", type=Path, default=DEFAULT_ARM_MANIFEST)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args(argv)
    try:
        summary = write_run(
            args.run_dir, args.out_dir, arm_manifest=args.arm_manifest, run_id=args.run_id
        )
    except (WriteError, cap.CaptureError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
