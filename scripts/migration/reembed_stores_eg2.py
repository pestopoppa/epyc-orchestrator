#!/usr/bin/env python3
"""reembed_stores_eg2.py -- REPL-EMB-2.2 adoption: re-embed the BGE-large stores into EmbeddingGemma 2.

STATUS: written and tested on synthetic stores ONLY. It has never been pointed at a live store, and
the task that produced it forbade doing so. ``run --apply`` is the operator's step, in a window.

WHAT IT DOES
------------
Re-embeds the three vector-bearing stores from their retained source TEXT into the REPL-EMB-2.2
candidate ``eg2-gguf-bf16-256`` (identity fields come from ``scripts/benchmark/repl_emb_arms.py``
``ArmSpec`` unchanged -- model, revision, dtype, dim, prompt hash):

    EmbeddingGemma 2 BF16 GGUF -> 768-d model output -> keep the leading 256 columns -> L2
    re-normalise (``truncate_renormalise``); STORED ITEMS get the document prefix
    ``title: none | text: `` (queries get ``task: search result | query: `` -- see "Consumers").

    store      files (in the store directory)                       text it re-embeds from
    ---------  ---------------------------------------------------  ---------------------------------
    episodic   episodic.db, embeddings.faiss, id_map.npy            memories.context JSON ->
                                                                    memory_record.record_from_legacy_
                                                                    context(ctx).embedding_text()
                                                                    (objective|task_description +
                                                                    task_type + priority; the same
                                                                    convention reseed_episodic_store
                                                                    wrote)
    skills     skills.db, skill_embeddings.faiss, skill_id_map.npy  skills.title + when_to_apply +
                                                                    task_types via
                                                                    skill_bank.skill_embedding_text
    sessions   sessions.db, session_embeddings.npy                  NONE RETAINED: the ``embeddings``
                                                                    table keeps (id, session_id,
                                                                    content_type) only and no
                                                                    production code calls
                                                                    store_embedding.

The conventions are the SHIPPED functions, loaded by file path (not re-spelled here). A store whose
vectors have no retained source text is reported BLOCKED with the reason; text is never invented.
The migration keeps every position: new vector i is the embedding of the text of ``id_map[i]``, and
``id_map.npy`` / the sqlite files are copied byte-for-byte, so ``memories.embedding_idx`` and every
id mapping stay valid. A BLOCKED store blocks the run (all-or-nothing): the dimension is global, a
fleet half at 1024-d and half at 256-d is the failure this tool exists to prevent.

The unit of migration and swap is the STORE DIRECTORY. The real layout puts all three stores in one
directory (``orchestration/repl_memory/sessions``: 22 GB of backups, a pickle HMAC key, reseed
receipts around them), so a directory is a unit of components and ``--episodic-dir/--skill-dir/
--session-dir`` that name the same directory describe one unit. Only the MANAGED files above are
copied and rewritten. Everything else in the directory is carried into the new directory by hard
link (``os.link``; recursive for sub-directories, symlinks preserved, copy fallback if linking is
refused), so no 22 GB duplicate and no second copy of a secret key. The old directory is only ever
RENAMED, never rewritten or deleted.

RUNBOOK (order matters)
-----------------------
 0. STOP THE WRITERS. Everything that opens these files must be down: the orchestrator API
    (uvicorn :8000 workers), autopilot, the q-scorer / Q-update loops, the skill-distillation and
    seeding jobs, any REPL session that opened a SessionStore, and any maintenance script. Use
    ``orchestrator_stack.py`` for the lifecycle; this tool never starts or stops a process.
    The tool cannot prove that nobody is writing, so apply requires ``--assume-stopped`` AND
    independently refuses when it sees a live indicator even with the flag: a non-empty/present
    sqlite ``-wal``/``-shm``/``-journal``, a held ``flock`` on a ``*.lock`` file, a process holding
    a store file open or mapped (``/proc/*/fd`` and ``/proc/*/maps``; only processes you may read).
    The directory's stat signature is re-checked right before the swap and the run aborts if it
    moved while it was copying.
 1. ``plan``       read-only: counts, dims, byte sizes, source text field, BLOCKED reasons, live
                   indicators. Fix every BLOCKED store first (repair_faiss_id_map.py,
                   reseed_episodic_store.py ...). Opens sqlite ``immutable=1``: it creates no
                   ``-wal``/``-shm`` files in a live directory.
 2. ``run`` (dry)  ``run --out-dir O`` without ``--apply`` is the default: it computes the plan and
                   the preflight and prints the exact files it would create and rename. No
                   embedding request is made; nothing is written anywhere.
 3. ``run --apply --embed-url URL --out-dir O --assume-stopped``  against a harness-owned EG2
                   llama-server (BF16 GGUF, embedding mode, mean pooling, ``/v1/embeddings``). It
                   probes the endpoint identity, copies each unit to ``<dir>.migrating-<run_id>/``,
                   re-embeds into the copy, validates the copy, writes the MANIFEST and the
                   rollback script, then swaps. Nothing live is touched before the swap; on any
                   failure the live directory is exactly as it was (the tool deletes only its own
                   ``.migrating-<run_id>`` copy; ``--keep-failed`` keeps it for forensics).
 4. verify         the summary re-hashes the swapped-in files against the manifest.
                   ``check-marker DIR`` is the consumer-side load check.
 5. RESTART CONSUMERS WITH THE EG2 POOL. Not before: a consumer still embedding queries with BGE
    against a 256-d store would be refused by the dimension marker, or (with the check disabled)
    return garbage. Configure ``embedding_dim=256`` everywhere in the list below at the same time.
 6. KEEP ``<dir>.pre-eg2-<run_id>/`` UNTIL THE OPERATOR CONFIRMS DELETION. The tool never deletes
    it, refuses to start while any ``<dir>.pre-eg2-*`` exists, and ``rollback`` needs it.
 7. Rollback: ``rollback --manifest M --assume-stopped`` (or the recorded
    ``rollback-<run_id>.sh --assume-stopped``, which needs only bash + sha256sum + mv). It verifies
    every old-file hash first, renames the EG2 directory to ``<dir>.eg2-rolledback-<run_id>/``
    (kept) and renames the pre-EG2 directory back. Writes made to the EG2 store after the swap
    are NOT in the restored store; the tool refuses if the EG2 files diverged from the manifest
    unless ``--allow-diverged``.

MANIFEST (``manifest-<run_id>.json``, sealed with its own sha256): the target identity (model, revision,
dtype, dim, prompt hash, runtime, digest), the endpoint probe, and per unit the sha256 + size of every
MANAGED file old and new (the sqlite files, id maps, lock file, vector files, marker), the
validation results, the old directory's stat signature, and which unmanaged entries were linked or
copied. Unmanaged entries (backups, keys) are recorded by name and stat signature, NOT by content
hash: hashing 22 GB of immutable backups buys nothing the signature plus the hard link does not
already give, and rollback restores that directory by renaming it, not by copying anything.

Writes nothing outside the given store directories (their ``.migrating-*`` / ``.pre-eg2-*`` /
``.eg2-rolledback-*`` siblings, named from the given path) and the explicit ``--out-dir``
(manifest, run log, rollback script, tool lock).

DIMENSION MARKER
----------------
Each migrated directory gets ``embedding_model.json``::

    {"model", "revision", "dim", "dtype", "prompt_hash", "arm_id", "native_dim", "normalise",
     "doc_prompt", "runtime", "digest", "run_id", "migrated_utc", "migrated_from": {...}}

``verify_store_marker(dir, expected_dim=..., expected_identity=...)`` is the consumer-side check:
a directory WITHOUT the file is a legacy BGE-1024 store, so a consumer configured for 256 refuses
it and a consumer configured for 1024 refuses a marked 256-d directory. The tool runs this check in
its own validation and ``check-marker`` exposes it. REPL-EMB-2.10 (2026-10-24): the marker also
carries ``model_file`` + ``gguf_sha256`` (B-3, the pool's own identity key; ``--expect-model-file`` /
``--expect-gguf-sha256`` compare them), and ``FAISSEmbeddingStore`` / ``NumpyEmbeddingStore`` /
``SQLiteSessionStore`` now refuse a marked directory whose ``dim`` differs from their config, naming
this tool (B-4, ``src/embedding_dims.py::check_store_marker``). ``--sessions-archive-only`` (B-1)
archives sessions' source-less vectors instead of blocking; ``--strategy-dir`` (B-2) migrates
autopilot's StrategyStore as its own unit. The CONSUMERS list below is the original survey; the ones
not named above still have to follow the marker's dim.

EXIT CODES: 0 ok | 2 refused (precondition, usage) | 3 BLOCKED store | 4 validation failed (live
untouched) | 5 swap failed (reverted) | 6 rollback verification failed.

CONSUMERS THAT MUST READ ``embedding_model.json`` / SWITCH TO 256 / USE THE DOC PREFIX
--------------------------------------------------------------------------------------
(reported, not edited; line numbers are of the lane worktree this was written in)
 * orchestration/repl_memory/faiss_store.py  FAISSEmbeddingStore.__init__(dim=1024) / _load():
   reads ``index.d`` but never compares it to ``self.dim`` -- opening a 256-d index with dim=1024
   loads fine and fails only at the first add()/search(); the right place for verify_store_marker.
   NumpyEmbeddingStore(dim=1024) the same.
 * orchestration/repl_memory/episodic_store.py  EpisodicStore(embedding_dim=1024) -> _init_embedding_store.
 * orchestration/repl_memory/skill_bank.py  SkillBank(embedding_dim=1024) -> _get_embedding_store.
 * src/api/services/memrl.py  ``SkillBank(..., embedding_dim=1024)`` hard-coded (and the
   EpisodicStore construction above it).
 * src/session/sqlite_store.py  SQLiteSessionStore(embedding_dim=1024) / _load_embeddings (already
   refuses a mismatched npy; should also read the marker so a 256-d file under a 1024-d config
   and vice versa both name the migration).
 * orchestration/repl_memory/strategy_store.py  StrategyStore(embedding_dim=1024): NOT migrated by
   this tool (its own ``strategy_embeddings.faiss`` in ``orchestration/repl_memory/strategies``);
   it has to be migrated or rebuilt on its own, and this tool BLOCKS a unit that has a strategy
   index co-located with the stores it swaps.
 * orchestration/repl_memory/embedder.py  EmbeddingConfig.embedding_dim=1024 and
   hash_fallback_embedding(embedding_dim=1024); orchestration/repl_memory/parallel_embedder.py
   EmbedderPoolConfig.embedding_dim=1024; src/embedding_pool/policy.py ClientPolicy.embedding_dim
   =1024 (the pool client's own dimension check). The pool must return 256-d vectors AND apply
   ``task: search result | query: `` to queries (``title: none | text: `` to stored items): this
   tool prefixes the DOCUMENT side only, because a query embedded without its prefix against
   prefixed documents is a different (worse) retrieval, not an error anyone will see.

RETRAINING / RE-CALIBRATION FLAGS (anything trained or calibrated on BGE 1024-d vectors)
----------------------------------------------------------------------------------------
(flagged, NOT retrained; the tool changes only stored vectors)
From a read-only survey of the worktree and of /mnt/raid0/llm/epyc-orchestrator (2026-10-20; the
file facts marked [v] were re-checked by hand, the rest are the survey's and unverified).

 A. Trained on / shaped by BGE vectors -- every one takes 1031 = 1024 BGE + 7 engineered features
    (hybrid_router.py:357-386), so each fails or silently misroutes on 256-d input:
    * orchestration/repl_memory/routing_classifier_weights.npz [v 2026-06-12] W1 (1031,128), 8 classes,
      per-class thresholds -> retrain scripts/graph_router/train_routing_classifier.py, then
      re-run calibrate_thresholds (target precision 0.9).
    * routing_classifier_weights_kl.npz (1031,128, 6 classes) -> train_routing_classifier_kl.py.
    * verifier_head_weights.npz (1031,64,32,1) -> retrain (frontdoor verifier).
    * orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/
      offline_reward_{frontdoor_verifier,multi_action_verifier,..._calibrated,..._histogram}_weights.npz
      (1031 / 1041 inputs; Platt, isotonic and histogram calibrations) -> retrain AND recalibrate.
    * Training data, all BGE, must be re-embedded and rebuilt first: orchestration/repl_memory/
      training_data.npz (526,661 x 1031), reports/p45_soft_labels/soft_labels_embedded.npz,
      sessions/reembedded.npz (+ reembedded.incremental-*.npz), the 12 offline_reward_verifier_data_*.npz.
    * lightweight_gat.py (input_dim=1024; load() never checks the stored dim) and the Kuzu
      routing_graph (DOUBLE[1024] node tables; CREATE ... IF NOT EXISTS will not alter them): no GAT
      weights exist on disk, so it needs a FIRST training; drop/recreate the three node tables and
      repopulate with scripts/graph_router/train_graph_router.py. Whether Kuzu can ALTER an array
      type was not checked.
    * Feature flags: graph_router, routing_classifier and skillbank default OFF in production
      (src/features.py:121-130), so the three models above are inert until enabled; memrl and
      specialist_routing are ON and are what the migrated stores serve.
    NOT affected: q-values / rewards (they stay on the rows and survive re-embedding),
    bilinear_scorer.py (8 heuristic features), factual_risk calibration (prompt text),
    classifier_config.yaml, kb_rag/ColBERT, src/vision (CLIP). The offline embedding IRT
    projector IS affected: its fitted ridge uses the BGE prompt embeddings (REPL-EMB-R1).
 B. Cosine-distribution thresholds tuned on BGE (no documented sweep found; EG2@256's cosine
    distribution has NOT been measured -- measure it before trusting any of these):
    min_similarity 0.3 / semantic_k 20 (src/config/models.py:1376, env ORCHESTRATOR_MEMRL_RETRIEVAL_
    MIN_SIMILARITY; retriever.py:278,:750) feeding the Q-median confidence threshold 0.6;
    q_scorer.py:2141 same-item filter 0.85 (query-prefixed vs document-prefixed identical text scores
    below 1.0); retriever.py:284 combined = 0.7*Q + 0.3*sim and classification_retriever.py:32,:252
    (0.4 / 0.6); skill_bank.py:455 (0.3), :495 find_duplicates 0.85, distillation/pipeline.py:329
    (0.85), skill_retriever.py:43 (0.4), replay/skill_replay.py:51 (0.4), replay/meta_agent.py:420
    (0.1-0.9); knowledge_distiller.py:42-43 (0.75 / 0.70); session sqlite_store 0.3;
    research_context.py:85 (0.7); hybrid_router classifier_confidence_threshold 0.8 and verifier 0.5
    (recalibrate with the retrain in A).
 C. Derived stores this tool does NOT migrate: orchestration/repl_memory/strategies/
    strategy_embeddings.faiss (1,428 vectors; StrategyStore.rebuild_search_indexes re-embeds from
    its sqlite rows) and the Kuzu graph above. The live skills.db holds 0 skills [v], so the skills
    store has nothing to migrate today; session_embeddings.npy is (1000,1024) and empty [v by size].
 D. Rollback copies that re-introduce 1024-d vectors beside 256-d code: orchestration/
    autopilot_checkpoints/*/{embeddings.faiss,id_map.npy,routing_classifier_weights.npz,
    strategy_store/} (restored by scripts/autopilot/species/structural_lab.py and
    ratify_and_apply_multitier_baseline_v10.py), sessions/*.pre-repair-* / *.pre-reseed-*,
    orchestration/archived_backups/. Do not restore any of them after the swap.
 E. Code that must change with the dimension (crash or silently wrong, not edited here):
    literal 1024 in episodic_store.py:254, faiss_store.py:98,:553, skill_bank.py:187,
    strategy_store.py:296, replay/{engine.py:83,skill_replay.py:121}, src/api/services/memrl.py:641,
    src/session/sqlite_store.py:158 (built with defaults at src/api/dependencies.py:193,
    src/cli_sessions.py:103, src/cli_orch.py:208), embedder.py:77,:146, parallel_embedder.py:45,:80,
    src/embedding_pool/policy.py:29 + orchestration/embedding_pool_policy.yaml:41 (client.py:93
    rejects any other size) and scripts/{maintenance,memory,graph_router,calibration,autopilot,
    skillbank,smoke}/* (full list in the survey; extract_training_data.py skips bad-dim rows, which
    yields a silently EMPTY training set). Nothing in TaskEmbedder / ParallelEmbedderClient /
    the pool does 768->256 truncation or query/document prefixing (only scripts/benchmark/
    repl_emb_arms.py:74 truncates); TaskEmbedder has one entry point for both sides, and q_scorer.py
    :1796 and :2112 use ONE vector for a lookup and then store it, so those callers need a side
    argument. Scripts that POST /embedding directly get raw 768-d unprefixed vectors:
    maintenance/{verify_episodic_reseed_cosine,repair_episodic_routing_poison,
    check_episodic_integrity}.py, memory/{backfill_faiss_embeddings,seed_remaining_phase_b,
    seed_success_patterns}.py, graph_router/{reembed_episodic_store,embed_soft_label_dataset,
    build_offline_reward_verifier_npz}.py, calibration/build_irt_prompt_embeddings.py,
    benchmark/seeding_injection.py.
 F. Server config: orchestration/launch_manifest.yaml:326,:368,:380 and scripts/server/
    orchestrator_stack.py:994 run the embedder with ``pooling: cls`` for BGE; EG2 needs ``mean``
    (the harness arm says so) and the repo has no EG2 GGUF path yet (/mnt/raid0/llm/models has none),
    so the production recipe is still to be written. orchestration/model_registry.yaml:170 pins
    ``dim: 1024``.

Owned follow-ups (epyc-root ``handoffs/active/repl-embedding-retrieval.md``; none run here):
 REPL-EMB-R1: rebuild prompt embeddings and refit the embedding IRT projector; preserve its
              held-out rank correlation baseline and refuse a mismatched fitted dimension.
 REPL-EMB-R2: regenerate soft-label / training / offline-reward datasets, retrain routing
              classifier and verifier weights, then recalibrate against frozen BGE accuracy.
 REPL-EMB-R3: derive lightweight_gat input_dim from the adopted seam, check stored weight
              dimensions, and perform its first EG2 training.
 REPL-EMB-R4: migrate routing_graph fixed-size Kuzu/DuckDB node arrays and repopulate them;
              CREATE IF NOT EXISTS does not change an existing array dimension.
 REPL-EMB-R5: sweep and recalibrate similarity thresholds, recording each old BGE value.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import re
import secrets
import shlex
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "reembed-stores-eg2/v1"
MARKER_NAME = "embedding_model.json"
TARGET_ARM_ID = "eg2-gguf-bf16-256"
SOURCE_DIM = 1024  # BGE-large-en-v1.5
TARGET_DIM = 256
NATIVE_DIM = 768
#: Same runtime string as repl_emb_eval's default, so the index digest equals the benchmarked arm's.
DEFAULT_RUNTIME = "llama-server experimental/gemma-embedding2-on-v10 (CPU, build-cpu)"
#: The fields that decide whether two vector sets may be scored against each other.
CORE_KEYS = ("model", "revision", "dim", "dtype", "prompt_hash")
#: A production pool / harness server reports the GGUF basename; see context_search markers.
MODEL_NAME_MARKERS = ("embeddinggemma-2", "egemma2")
#: REPL-EMB-2.10 B-3: the adopted GGUF's identity key (package section 1). A marker carries the
#: file name and sha256 the pool reports/serves, so consumers compare ONE key, not ArmSpec fields.
EG2_MODEL_FILE = "embeddinggemma-2-BF16.gguf"
EG2_GGUF_SHA256 = "68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216"
MIN_NORM = 1e-6
SEMANTIC_MIN_COSINE = 0.99999  # REPL-EMB-2.10 owner acceptance, not retrieval quality
SQL_CHUNK = 500
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

EXIT_OK, EXIT_REFUSED, EXIT_BLOCKED, EXIT_INVALID, EXIT_SWAP, EXIT_ROLLBACK = 0, 2, 3, 4, 5, 6

# Component status vocabulary.
READY, EMPTY, NO_VECTORS, BLOCKED = "READY", "EMPTY", "NO_VECTORS", "BLOCKED"
ALREADY_MIGRATED, NOTHING_TO_DO = "ALREADY_MIGRATED", "NOTHING_TO_DO"

#: Vector files that belong to stores this tool does not manage; co-location blocks the unit.
FOREIGN_VECTOR_FILES = {
    "strategy_embeddings.faiss": "StrategyStore index (embedding_dim=1024)",
    "strategy_id_map.npy": "StrategyStore id map",
}


class ToolError(RuntimeError):
    """Base class for refusals the CLI turns into an exit code."""


class Refused(ToolError):
    def __init__(self, reasons: Sequence[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


class BlockedError(ToolError):
    pass


class ValidationError(ToolError):
    def __init__(self, failures: Sequence[str]):
        self.failures = list(failures)
        super().__init__("; ".join(self.failures))


class SwapError(ToolError):
    pass


class RollbackError(ToolError):
    pass


class EmbedError(RuntimeError):
    """The embedding endpoint produced nothing usable."""


class WrongDimensionError(EmbedError):
    """Deterministic: retrying cannot change the vector length."""


class StoreMarkerError(ToolError):
    """A store directory's dimension marker contradicts what the consumer expects."""


# ───────────────────────────────────────────────────────────────────────────── shipped code


def _load_by_path(name: str, relpath: str) -> Any:
    """Load a shipped module by file path under a private name (no package __init__ side effects)."""
    if name in sys.modules:
        return sys.modules[name]
    if str(REPO_ROOT) not in sys.path:
        # shipped modules import `src.embedding_dims` (REPL-EMB-2.9 dimension constant)
        sys.path.insert(0, str(REPO_ROOT))
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ToolError(f"cannot load shipped module {relpath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _arms() -> Any:
    return _load_by_path("epyc_reembed_arms", "scripts/benchmark/repl_emb_arms.py")


def _memory_record() -> Any:
    return _load_by_path("epyc_reembed_memory_record", "orchestration/repl_memory/memory_record.py")


def _skill_bank() -> Any:
    return _load_by_path("epyc_reembed_skill_bank", "orchestration/repl_memory/skill_bank.py")


def _strategy_store_helpers() -> dict[str, Any]:
    mod = _load_by_path(
        "epyc_reembed_strategy_store", "orchestration/repl_memory/strategy_store.py"
    )
    return {"compact": mod._compact_text, "derive_title": mod._derive_title}


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - exercised only where faiss is absent
        raise ToolError(f"faiss is not importable ({exc}); FAISS stores cannot be read") from exc
    return faiss


def target_spec(runtime: str = DEFAULT_RUNTIME) -> Any:
    """The ArmSpec of the migration target, straight from the benchmark harness."""
    arms = _arms()
    by_id = {s.arm_id: s for s in arms.arm_specs("n/a", runtime, "n/a")}
    spec = by_id[TARGET_ARM_ID]
    if (spec.dim, spec.native_dim, spec.dtype) != (TARGET_DIM, NATIVE_DIM, "bf16"):
        raise ToolError(f"ArmSpec {TARGET_ARM_ID} no longer matches this tool: {spec}")
    return spec


def target_identity(spec: Any) -> dict[str, Any]:
    ident = spec.index_identity()  # model, revision, dim, dtype, prompt_hash, runtime, digest
    return {
        "model": ident["model"],
        "revision": ident["revision"],
        "dim": ident["dim"],
        "dtype": ident["dtype"],
        "prompt_hash": ident["prompt_hash"],
        "arm_id": spec.arm_id,
        "native_dim": spec.native_dim,
        "normalise": spec.normalise,
        "doc_prompt": spec.doc_prompt,
        "runtime": ident["runtime"],
        "digest": ident["digest"],
    }


# ───────────────────────────────────────────────────────────────────────────── small helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"  # pragma: no cover


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_hashed(src: Path, dst: Path, chunk: int = 1 << 22) -> tuple[str, int]:
    """Stream-copy ``src`` -> ``dst`` (new file), return (sha256 of what was read, bytes)."""
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as fin, open(dst, "xb") as fout:
        while block := fin.read(chunk):
            h.update(block)
            fout.write(block)
            n += len(block)
        fout.flush()
        os.fsync(fout.fileno())
    shutil.copystat(src, dst)
    return h.hexdigest(), n


def tree_signature(root: Path) -> dict[str, Any]:
    """Cheap change detector for a whole directory tree: names, sizes, mtimes (no content)."""
    entries: list[tuple[str, int, int]] = []

    def walk(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as it:
            for entry in sorted(it, key=lambda e: e.name):
                rel = prefix + entry.name
                st = entry.stat(follow_symlinks=False)
                if entry.is_dir(follow_symlinks=False):
                    entries.append((rel + "/", 0, st.st_mtime_ns))
                    walk(Path(entry.path), rel + "/")
                else:
                    entries.append((rel, st.st_size, st.st_mtime_ns))

    st_root = root.stat()
    entries.append(("./", 0, st_root.st_mtime_ns))
    walk(root, "")
    return {"digest": sha256_bytes(canonical_json(entries).encode()), "entries": len(entries)}


def dir_size(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _ro_connect(path: Path) -> sqlite3.Connection:
    """Read-only, ``immutable=1``: never creates ``-wal``/``-shm`` next to a live database."""
    uri = "file:" + urllib.parse.quote(str(path)) + "?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _chunks(seq: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class RunLog:
    """stderr + (once an out-dir exists) a JSON-lines file. Events before attach are buffered."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.path: Path | None = None
        self._buffer: list[str] = []

    def attach(self, path: Path) -> None:
        self.path = path
        with open(path, "a", encoding="utf-8") as fh:
            for line in self._buffer:
                fh.write(line + "\n")
        self._buffer.clear()

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": utc_now(), "event": kind, **fields}
        line = json.dumps(record, sort_keys=True, default=str)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        else:
            self._buffer.append(line)
        if not self.quiet:
            extras = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[{record['ts']}] {kind} {extras}".rstrip(), file=sys.stderr, flush=True)


# ───────────────────────────────────────────────────────────────────────────── dimension marker


def read_marker(store_dir: Path) -> dict[str, Any] | None:
    path = Path(store_dir) / MARKER_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StoreMarkerError(f"{path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise StoreMarkerError(f"{path} is not a JSON object")
    return data


def marker_matches(marker: dict[str, Any], identity: dict[str, Any]) -> bool:
    if not all(marker.get(k) == identity.get(k) for k in CORE_KEYS):
        return False
    # Pre-B-3 markers remain readable. A key that IS present must name the adopted GGUF.
    return all(
        key not in marker or marker[key] == want
        for key, want in (("model_file", EG2_MODEL_FILE), ("gguf_sha256", EG2_GGUF_SHA256))
    )


def verify_store_marker(
    store_dir: Path,
    *,
    expected_dim: int,
    expected_identity: dict[str, Any] | None = None,
    legacy_dim: int = SOURCE_DIM,
    expected_model_file: str | None = None,
    expected_gguf_sha256: str | None = None,
) -> dict[str, Any]:
    """Consumer-side load check. Raises ``StoreMarkerError`` on any mixed-store condition.

    No marker means a legacy (pre-migration, BGE) store of ``legacy_dim``: a consumer expecting
    any other dimension must refuse it. A marker pins model, revision, dim, dtype and prompt hash;
    ``expected_identity`` (optional) must agree on all of them.
    """
    marker = read_marker(store_dir)
    if marker is None:
        if expected_dim != legacy_dim:
            raise StoreMarkerError(
                f"{store_dir} has no {MARKER_NAME}: it is a legacy {legacy_dim}-d store, but the "
                f"consumer expects dim={expected_dim}. Migrate it or fix the consumer config."
            )
        return {"dim": legacy_dim, "legacy": True}
    if marker.get("dim") != expected_dim:
        raise StoreMarkerError(
            f"{store_dir} is marked dim={marker.get('dim')} ({marker.get('model')}), the consumer "
            f"expects dim={expected_dim}: refusing a mixed store."
        )
    if expected_identity is not None:
        diff = [k for k in CORE_KEYS if marker.get(k) != expected_identity.get(k)]
        if diff:
            raise StoreMarkerError(
                f"{store_dir} marker disagrees with the expected embedding identity on {diff}"
            )
    # B-3: compare the pool's own key (GGUF basename + sha256) when the consumer supplies it.
    for key, want in (("model_file", expected_model_file), ("gguf_sha256", expected_gguf_sha256)):
        if want is not None and marker.get(key) != want:
            raise StoreMarkerError(
                f"{store_dir} marker {key}={marker.get(key)!r} but the serving pool reports "
                f"{want!r}: vectors from a different model file"
            )
    return marker


def build_marker(
    identity: dict[str, Any],
    run_id: str,
    migrated_from: dict[str, Any],
    *,
    model_file: str = EG2_MODEL_FILE,
    gguf_sha256: str = EG2_GGUF_SHA256,
    gguf_sha256_verified: bool = False,
) -> dict[str, Any]:
    marker = {k: identity[k] for k in CORE_KEYS}
    marker.update(
        {
            # B-3: the pool's identity key. `verified` is True only when this run hashed the GGUF.
            "model_file": model_file,
            "gguf_sha256": gguf_sha256,
            "gguf_sha256_verified": gguf_sha256_verified,
            "arm_id": identity["arm_id"],
            "native_dim": identity["native_dim"],
            "normalise": identity["normalise"],
            "doc_prompt": identity["doc_prompt"],
            "runtime": identity["runtime"],
            "digest": identity["digest"],
            "run_id": run_id,
            "migrated_utc": utc_now(),
            "migrated_from": migrated_from,
        }
    )
    return marker


# ───────────────────────────────────────────────────────────────────────────── embedding client


class EmbedClient:
    """EG2 llama-server ``/v1/embeddings`` client: batching, retries, strict vector validation.

    ``embed_documents`` returns unit-norm float32 ``(N, 256)``: prefix each text with the document
    prompt, take the server's 768-d vector, refuse anything non-finite / wrong-dim / ~zero-norm,
    keep the leading 256 columns and re-normalise (``truncate_renormalise``).
    """

    def __init__(
        self,
        base_url: str,
        spec: Any,
        *,
        batch_size: int = 16,
        retries: int = 4,
        timeout_s: float = 120.0,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if urllib.parse.urlsplit(base_url).scheme not in ("http", "https"):
            raise Refused([f"--embed-url {base_url!r} must be an http(s) URL"])
        self.base_url = base_url.rstrip("/")
        self.spec = spec
        self.batch_size = max(1, int(batch_size))
        self.retries = max(0, int(retries))
        self.timeout_s = timeout_s
        self.backoff_s = backoff_s
        self._sleep = sleep
        self.requests = 0
        self.retried = 0
        self.texts_embedded = 0

    # -- transport
    def _post(self, texts: list[str]) -> dict[str, Any]:
        body = json.dumps({"input": texts, "encoding_format": "float"}).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - operator URL
            return json.loads(resp.read())

    def _get_json(self, path: str) -> Any:
        req = urllib.request.Request(self.base_url + path)
        with urllib.request.urlopen(req, timeout=min(self.timeout_s, 30.0)) as resp:  # noqa: S310
            return json.loads(resp.read())

    # -- validation
    def _raw_matrix(self, payload: dict[str, Any], n: int) -> np.ndarray:
        try:
            rows = payload["data"]
            if not isinstance(rows, list):
                raise TypeError("data is not a list")
            if len(rows) != n:
                raise EmbedError(f"server returned {len(rows)} rows for {n} texts")
            if any(isinstance(r, dict) and "index" in r for r in rows):
                indices = [r.get("index") if isinstance(r, dict) else None for r in rows]
                if any(type(i) is not int for i in indices) or sorted(indices) != list(range(n)):
                    raise EmbedError(
                        "server response indices must be a complete, unique permutation "
                        f"of 0..{n - 1}"
                    )
                rows = sorted(rows, key=lambda r: r["index"])
            raw = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        except EmbedError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise WrongDimensionError(f"unparseable /v1/embeddings payload: {exc!r}") from exc
        if raw.ndim != 2 or raw.shape[1] != self.spec.native_dim:
            raise WrongDimensionError(
                f"server vectors have shape {tuple(raw.shape)}, expected (*, {self.spec.native_dim});"
                " is this the EmbeddingGemma 2 server (mean pooling), not BGE or pooling=none?"
            )
        if not np.isfinite(raw).all():
            bad = int((~np.isfinite(raw).all(axis=1)).sum())
            raise EmbedError(f"{bad} non-finite vector(s) from the server")
        norms = np.linalg.norm(raw, axis=1)
        if (norms <= MIN_NORM).any():
            raise EmbedError(
                f"{int((norms <= MIN_NORM).sum())} zero-norm vector(s) from the server"
            )
        head = raw[:, : self.spec.dim]
        if (np.linalg.norm(head, axis=1) <= MIN_NORM).any():
            raise EmbedError("a vector has a ~zero leading-256 slice; cannot re-normalise")
        return raw

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        arms = _arms()
        prefixed = [self.spec.doc_prompt + t for t in texts]
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self.requests += 1
                raw = self._raw_matrix(self._post(prefixed), len(prefixed))
                out = arms.truncate_renormalise(raw, self.spec.dim)
                if out.shape != (len(texts), self.spec.dim) or not np.isfinite(out).all():
                    raise EmbedError(f"bad truncated shape/values {out.shape}")
                if np.abs(np.linalg.norm(out, axis=1) - 1.0).max() > 1e-3:
                    raise EmbedError("re-normalised vectors are not unit norm")
                return out.astype(np.float32)
            except WrongDimensionError:
                raise
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read(300).decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    pass
                last = EmbedError(f"HTTP {exc.code} from /v1/embeddings: {detail}")
                if exc.code < 500 and exc.code not in (408, 429):
                    raise last from exc  # a 4xx (e.g. text too long) will not fix itself
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
                ValueError,
            ) as exc:
                last = EmbedError(f"{type(exc).__name__}: {exc}")
            except EmbedError as exc:
                last = exc
            if attempt < self.retries:
                self.retried += 1
                self._sleep(self.backoff_s * (2**attempt))
        raise EmbedError(f"giving up after {self.retries + 1} attempts: {last}") from last

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        parts = []
        for chunk in _chunks(texts, self.batch_size):
            parts.append(self._embed_batch(list(chunk)))
            self.texts_embedded += len(chunk)
        return np.vstack(parts)

    def probe(self, *, check_model: bool = True) -> dict[str, Any]:
        """Endpoint identity + determinism. Raises ``Refused`` when it is not an EG2 server."""
        info: dict[str, Any] = {"url": self.base_url}
        try:
            models = self._get_json("/v1/models")
            ids = [str(m.get("id", "")) for m in models.get("data", []) if isinstance(m, dict)]
            info["models"] = ids
        except Exception as exc:  # noqa: BLE001 - /v1/models is best effort on llama-server
            info["models"] = None
            info["models_error"] = f"{type(exc).__name__}: {exc}"
        if not check_model:
            info["model_check"] = "skipped by --skip-model-check"
        elif info["models"]:
            joined = " ".join(info["models"]).lower()
            if not any(m in joined for m in MODEL_NAME_MARKERS):
                raise Refused(
                    [
                        f"endpoint {self.base_url} reports models {info['models']}, none looks like "
                        f"EmbeddingGemma 2 ({'/'.join(MODEL_NAME_MARKERS)}); refusing to write "
                        "another model's vectors under the EG2 identity (--skip-model-check to override)"
                    ]
                )
        try:
            a = self.embed_documents(["reembed-stores-eg2 probe"])
            b = self.embed_documents(["reembed-stores-eg2 probe"])
        except EmbedError as exc:
            raise Refused([f"embedding endpoint probe failed: {exc}"]) from exc
        cos = float(np.dot(a[0], b[0]))
        info["probe_cosine_repeat"] = round(cos, 6)
        info["native_dim"] = self.spec.native_dim
        if cos < 0.9999:
            raise Refused([f"endpoint is not deterministic on a repeated text (cosine {cos:.6f})"])
        return info


# ───────────────────────────────────────────────────────────────────────────── components


@dataclass
class ComponentReport:
    kind: str
    status: str = NO_VECTORS
    files: dict[str, str] = field(default_factory=dict)  # role -> relative path
    sizes: dict[str, int] = field(default_factory=dict)
    dim: int | None = None
    vectors: int = 0
    id_map_len: int | None = None
    rows: int | None = None
    text_source: str = ""
    blocked: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "files": self.files,
            "bytes": self.sizes,
            "dim": self.dim,
            "vectors": self.vectors,
            "id_map_len": self.id_map_len,
            "rows": self.rows,
            "text_source": self.text_source,
            "blocked": self.blocked,
            "warnings": self.warnings,
            "detail": self.detail,
        }


class Component:
    kind = ""
    #: files copied byte-for-byte into the new directory
    copy_files: tuple[str, ...] = ()
    #: vector files rewritten (never copied)
    vector_files: tuple[str, ...] = ()
    #: sqlite databases whose sidecars (-wal/-shm/-journal) signal a live writer
    databases: tuple[str, ...] = ()

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def exists(self) -> bool:
        return any((self.dir / n).exists() for n in self.copy_files + self.vector_files)

    def managed(self) -> set[str]:
        return set(self.copy_files) | set(self.vector_files)

    def _file_facts(self, report: ComponentReport, roles: dict[str, str]) -> None:
        for role, name in roles.items():
            p = self.dir / name
            if p.exists():
                report.files[role] = name
                report.sizes[name] = p.stat().st_size


class FaissComponent(Component):
    """A sqlite table + FAISS index + id_map triple (episodic and skills)."""

    db_name = ""
    index_name = ""
    idmap_name = ""
    table = ""
    text_source = ""

    @property
    def copy_files(self) -> tuple[str, ...]:  # type: ignore[override]
        return (self.db_name, self.idmap_name) + self._extra_copy

    _extra_copy: tuple[str, ...] = ()

    @property
    def vector_files(self) -> tuple[str, ...]:  # type: ignore[override]
        return (self.index_name,)

    @property
    def databases(self) -> tuple[str, ...]:  # type: ignore[override]
        return (self.db_name,)

    def exists(self) -> bool:
        return any(
            (self.dir / n).exists() for n in (self.db_name, self.index_name, self.idmap_name)
        )

    # -- text
    def resolve_rows(
        self, con: sqlite3.Connection, ids: Sequence[str]
    ) -> dict[str, tuple[str | None, str | None]]:
        raise NotImplementedError

    def _load_id_map(self) -> list[str]:
        arr = np.load(self.dir / self.idmap_name, allow_pickle=True)  # store's own trusted file
        return [str(x) for x in arr.tolist()]

    def _read_index_header(self) -> tuple[int, int, str]:
        index = _faiss().read_index(str(self.dir / self.index_name))
        return int(index.d), int(index.ntotal), type(index).__name__

    def inspect(self) -> ComponentReport:
        r = ComponentReport(self.kind, text_source=self.text_source)
        self._file_facts(
            r,
            {"db": self.db_name, "index": self.index_name, "id_map": self.idmap_name}
            | {f"extra{i}": n for i, n in enumerate(self._extra_copy)},
        )
        has_index = (self.dir / self.index_name).exists()
        has_map = (self.dir / self.idmap_name).exists()
        has_db = (self.dir / self.db_name).exists()
        self._inspect_legacy(r)
        if r.blocked:
            r.status = BLOCKED
            return r
        if not has_index and not has_map:
            r.status = NO_VECTORS
            if has_db:
                r.rows = self._count_rows()
            return r
        if has_index != has_map:
            r.blocked.append(
                f"index/id_map pair incomplete ({self.index_name}: {has_index}, "
                f"{self.idmap_name}: {has_map})"
            )
            r.status = BLOCKED
            return r
        try:
            r.dim, r.vectors, index_type = self._read_index_header()
            r.detail["index_type"] = index_type
            id_map = self._load_id_map()
        except Exception as exc:  # noqa: BLE001 - report, never crash a plan
            r.blocked.append(f"cannot read vector files: {type(exc).__name__}: {exc}")
            r.status = BLOCKED
            return r
        r.id_map_len = len(id_map)
        if r.vectors != len(id_map):
            r.blocked.append(
                f"index has {r.vectors} vectors but id_map has {len(id_map)} ids (desync; "
                "run scripts/maintenance/repair_faiss_id_map.py first)"
            )
        if r.dim not in (SOURCE_DIM, TARGET_DIM):
            r.blocked.append(
                f"index dim {r.dim} is neither the BGE source {SOURCE_DIM} nor {TARGET_DIM}"
            )
        if len(set(id_map)) != len(id_map):
            r.warnings.append(
                f"{len(id_map) - len(set(id_map))} duplicate id(s) in id_map (positions kept)"
            )
        if not has_db:
            r.blocked.append(
                f"{self.db_name} is missing: vectors only, the original text is not retained"
            )
        elif id_map:
            self._inspect_text(r, id_map)
        if r.blocked:
            r.status = BLOCKED
        elif r.vectors == 0:
            r.status = EMPTY
        else:
            r.status = READY
        return r

    def _inspect_legacy(self, r: ComponentReport) -> None:
        return None

    def _count_rows(self) -> int | None:
        try:
            with _ro_connect(self.dir / self.db_name) as con:
                return int(con.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])
        except sqlite3.Error:
            return None

    def _inspect_text(self, r: ComponentReport, id_map: list[str]) -> None:
        unresolved: dict[str, list[str]] = {}
        resolved = 0
        try:
            with _ro_connect(self.dir / self.db_name) as con:
                r.rows = int(con.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])
                for chunk in _chunks(id_map, SQL_CHUNK):
                    got = self.resolve_rows(con, chunk)
                    for mid in chunk:
                        text, reason = got.get(mid, (None, "no_db_row"))
                        if text is None:
                            unresolved.setdefault(reason or "unresolved", []).append(mid)
                        else:
                            resolved += 1
                self._extra_checks(con, r, id_map)
        except sqlite3.Error as exc:
            r.blocked.append(f"cannot read {self.db_name}: {type(exc).__name__}: {exc}")
            return
        r.detail["resolved_text"] = resolved
        if unresolved:
            counts = {k: len(v) for k, v in sorted(unresolved.items())}
            r.detail["unresolved"] = counts
            r.detail["unresolved_samples"] = {k: v[:5] for k, v in sorted(unresolved.items())}
            total = sum(counts.values())
            r.blocked.append(
                f"{total} of {len(id_map)} vectors have no retained source text {counts}; "
                "the text is not invented -- repair/reseed the store (or drop those rows) first"
            )

    def _extra_checks(self, con: sqlite3.Connection, r: ComponentReport, id_map: list[str]) -> None:
        return None

    # -- apply-time streaming
    def iter_texts(self) -> Iterator[tuple[list[str], list[str]]]:
        """Yield (ids, texts) for id_map positions in order, ``SQL_CHUNK`` at a time."""
        id_map = self._load_id_map()
        with _ro_connect(self.dir / self.db_name) as con:
            for chunk in _chunks(id_map, SQL_CHUNK):
                got = self.resolve_rows(con, chunk)
                texts = []
                for mid in chunk:
                    text, reason = got.get(mid, (None, "no_db_row"))
                    if text is None:
                        raise BlockedError(f"{self.kind}: id {mid} has no source text ({reason})")
                    texts.append(text)
                yield list(chunk), texts

    def texts_at(self, positions: Sequence[int]) -> list[str]:
        id_map = self._load_id_map()
        ids = [id_map[p] for p in positions]
        with _ro_connect(self.dir / self.db_name) as con:
            got = self.resolve_rows(con, sorted(set(ids)))
        out = []
        for mid in ids:
            text, reason = got.get(mid, (None, "no_db_row"))
            if text is None:
                raise BlockedError(f"{self.kind}: id {mid} has no source text ({reason})")
            out.append(text)
        return out


class EpisodicComponent(FaissComponent):
    kind = "episodic"
    db_name = "episodic.db"
    index_name = "embeddings.faiss"
    idmap_name = "id_map.npy"
    table = "memories"
    _extra_copy = (".episodic_faiss.lock",)
    text_source = (
        "memories.context (JSON) -> record_from_legacy_context(ctx).embedding_text() "
        "[objective|task_description + task_type + priority; orchestration/repl_memory/"
        "memory_record.py, the convention reseed_episodic_store.py wrote]"
    )

    def _inspect_legacy(self, r: ComponentReport) -> None:
        if (self.dir / "embeddings.npy").exists() and not (self.dir / self.index_name).exists():
            r.blocked.append(
                "legacy NumpyEmbeddingStore layout (embeddings.npy, no embeddings.faiss) is not "
                "supported; run scripts/memory/migrate_to_faiss.py first"
            )

    def resolve_rows(
        self, con: sqlite3.Connection, ids: Sequence[str]
    ) -> dict[str, tuple[str | None, str | None]]:
        mr = _memory_record()
        marks = ",".join("?" * len(ids))
        out: dict[str, tuple[str | None, str | None]] = {}
        for mid, ctx_json in con.execute(
            f"SELECT id, context FROM memories WHERE id IN ({marks})", list(ids)
        ):
            try:
                ctx = json.loads(ctx_json)
            except (TypeError, ValueError):
                out[mid] = (None, "context_not_json")
                continue
            if not isinstance(ctx, dict):
                out[mid] = (None, "context_not_object")
                continue
            try:
                rec = mr.record_from_legacy_context(ctx)
                if not rec.is_task_memory():
                    out[mid] = (None, "no_task_text")
                    continue
                text = rec.embedding_text()
            except Exception:  # noqa: BLE001 - odd field types: report, do not guess
                out[mid] = (None, "bad_task_fields")
                continue
            out[mid] = (text, None) if text else (None, "no_task_text")
        return out

    def _extra_checks(self, con: sqlite3.Connection, r: ComponentReport, id_map: list[str]) -> None:
        pos = {mid: i for i, mid in enumerate(id_map)}
        bad = 0
        no_vector = 0
        for mid, idx in con.execute("SELECT id, embedding_idx FROM memories"):
            if mid not in pos:
                no_vector += 1
            elif idx is None or pos[mid] != idx:
                bad += 1
        if bad:
            r.warnings.append(
                f"{bad} row(s) whose memories.embedding_idx does not equal their id_map position "
                "(left as is; retrieval resolves by id; see repair_faiss_id_map.py)"
            )
        r.detail["rows_without_vector"] = no_vector


class SkillsComponent(FaissComponent):
    kind = "skills"
    db_name = "skills.db"
    index_name = "skill_embeddings.faiss"
    idmap_name = "skill_id_map.npy"
    table = "skills"
    text_source = (
        "skills.title + when_to_apply + task_types -> skill_bank.skill_embedding_text() "
        "[orchestration/repl_memory/skill_bank.py]"
    )

    def resolve_rows(
        self, con: sqlite3.Connection, ids: Sequence[str]
    ) -> dict[str, tuple[str | None, str | None]]:
        sb = _skill_bank()
        marks = ",".join("?" * len(ids))
        out: dict[str, tuple[str | None, str | None]] = {}
        for sid, title, when, types in con.execute(
            f"SELECT id, title, when_to_apply, task_types FROM skills WHERE id IN ({marks})",
            list(ids),
        ):
            if not (title or "").strip():
                out[sid] = (None, "no_title")
                continue
            out[sid] = (sb.skill_embedding_text(title, when, types), None)
        return out


class SessionsComponent(Component):
    kind = "sessions"
    copy_files = ("sessions.db",)
    vector_files = ("session_embeddings.npy",)
    databases = ("sessions.db",)
    text_source = (
        "NONE RETAINED: the embeddings table keeps (id, session_id, content_type) only and no "
        "production writer defines the embedded text"
    )
    #: B-1: explicit, manifest-recorded archive-only mode (``--sessions-archive-only``). The old
    #: vectors are NOT carried forward: they travel into ``<dir>.pre-eg2-<run_id>/`` (hashed in
    #: the manifest, restored by rollback) and the migrated store gets a zeroed 256-d file.
    archive_only = False

    def inspect(self) -> ComponentReport:
        r = ComponentReport(self.kind, text_source=self.text_source)
        self._file_facts(r, {"db": "sessions.db", "npy": "session_embeddings.npy"})
        npy = self.dir / "session_embeddings.npy"
        if not npy.exists() or npy.stat().st_size == 0:
            r.status = NO_VECTORS
            return r
        try:
            arr = np.load(npy, mmap_mode="r")
        except Exception as exc:  # noqa: BLE001
            r.blocked.append(f"cannot read session_embeddings.npy: {type(exc).__name__}: {exc}")
            r.status = BLOCKED
            return r
        if arr.ndim != 2 or arr.dtype != np.float32:
            r.blocked.append(f"unexpected array layout shape={arr.shape} dtype={arr.dtype}")
            r.status = BLOCKED
            return r
        r.dim, rows = int(arr.shape[1]), int(arr.shape[0])
        r.detail["capacity_rows"] = rows
        if r.dim not in (SOURCE_DIM, TARGET_DIM):
            r.blocked.append(
                f"npy dim {r.dim} is neither the BGE source {SOURCE_DIM} nor {TARGET_DIM}"
            )
        used = 0
        by_type: dict[str, int] = {}
        db = self.dir / "sessions.db"
        if db.exists():
            try:
                with _ro_connect(db) as con:
                    for ctype, n in con.execute(
                        "SELECT content_type, COUNT(*) FROM embeddings GROUP BY content_type"
                    ):
                        by_type[str(ctype)] = int(n)
                        used += int(n)
            except sqlite3.Error as exc:
                r.warnings.append(f"cannot read embeddings table: {type(exc).__name__}: {exc}")
        nonzero = 0
        for start in range(0, rows, 4096):
            nonzero += int(
                np.count_nonzero(np.any(np.asarray(arr[start : start + 4096]) != 0, axis=1))
            )
        r.vectors = max(used, nonzero)
        r.rows = used
        r.detail.update({"metadata_rows": by_type, "nonzero_vector_rows": nonzero})
        if r.vectors and self.archive_only:
            r.detail["archive_only"] = True
            r.detail["archived_vector_rows"] = nonzero
            r.warnings.append(
                f"archive-only: {nonzero} non-zero 1024-d session vector(s) are NOT migrated (no "
                "source text exists); they stay in the .pre-eg2 directory and the migrated store "
                "starts with an empty 256-d file"
            )
            r.status = BLOCKED if r.blocked else READY
        elif r.vectors:
            r.blocked.append(
                f"{used} embedding metadata row(s) and {nonzero} non-zero vector row(s) but NO "
                "retained source text (the embedded text is not stored and no writer in src/ "
                "defines it); cannot re-embed -- pass --sessions-archive-only to archive the "
                "vectors deliberately, or keep this store at its current dimension outside this "
                "migration"
            )
            r.status = BLOCKED
        elif r.blocked:
            r.status = BLOCKED
        else:
            r.status = EMPTY
        return r


class StrategyComponent(FaissComponent):
    """B-2: autopilot's StrategyStore (``strategies.db`` + ``strategy_embeddings.faiss`` +
    ``strategy_id_map.npy``), a 1024-d store in its own directory. Same copy-then-swap, same
    marker. Embedded text = ``f"{title} {description} {insight}"``, exactly what
    ``StrategyStore._publish_search_indexes_from_rows`` embeds (title from
    ``metadata_json.insight_format.title`` else derived from the description)."""

    kind = "strategy"
    db_name = "strategies.db"
    index_name = "strategy_embeddings.faiss"
    idmap_name = "strategy_id_map.npy"
    table = "strategies"
    text_source = (
        "strategies: title + description + insight "
        "[orchestration/repl_memory/strategy_store.py _publish_search_indexes_from_rows]"
    )

    def resolve_rows(
        self, con: sqlite3.Connection, ids: Sequence[str]
    ) -> dict[str, tuple[str | None, str | None]]:
        ss = _strategy_store_helpers()
        marks = ",".join("?" * len(ids))
        out: dict[str, tuple[str | None, str | None]] = {}
        for sid, description, insight, meta_json in con.execute(
            f"SELECT id, description, insight, metadata_json FROM strategies WHERE id IN ({marks})",
            list(ids),
        ):
            if not (description or "").strip():
                out[sid] = (None, "no_description")
                continue
            try:
                meta = json.loads(meta_json or "{}")
            except (TypeError, ValueError):
                meta = {}
            fmt = meta.get("insight_format") if isinstance(meta, dict) else {}
            if not isinstance(fmt, dict):
                fmt = {}
            title = ss["compact"](fmt.get("title")) or ss["derive_title"](description)
            out[sid] = (f"{title} {description} {insight}", None)
        return out


COMPONENT_CLASSES: tuple[type[Component], ...] = (
    EpisodicComponent,
    SkillsComponent,
    SessionsComponent,
)
KIND_OF_FLAG = {
    "episodic_dir": "episodic",
    "skill_dir": "skills",
    "session_dir": "sessions",
    "strategy_dir": "strategy",
}


# ───────────────────────────────────────────────────────────────────────────── units


@dataclass
class Unit:
    path: Path  # resolved real path
    given: list[str]
    declared: set[str]
    components: list[Component] = field(default_factory=list)
    reports: list[ComponentReport] = field(default_factory=list)
    marker: dict[str, Any] | None = None
    status: str = NOTHING_TO_DO
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unmanaged: list[str] = field(default_factory=list)
    total_bytes: int = 0

    def managed_names(self) -> set[str]:
        names: set[str] = set()
        for c in self.components:
            names |= c.managed()
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "given": self.given,
            "status": self.status,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "marker": self.marker,
            "components": [r.to_dict() for r in self.reports],
            "unmanaged_entries": len(self.unmanaged),
            "directory_bytes": self.total_bytes,
        }


def has_component_files(directory: Path) -> bool:
    return any(cls(directory).exists() for cls in COMPONENT_CLASSES)


def inspect_unit(
    path: Path,
    given: Sequence[str],
    declared: set[str],
    identity: dict[str, Any],
    *,
    sessions_archive_only: bool = False,
) -> Unit:
    real = Path(os.path.realpath(path))
    unit = Unit(path=real, given=list(given), declared=set(declared))
    if not real.is_dir():
        unit.status = BLOCKED
        unit.reasons.append(f"{path} is not a directory")
        return unit
    # B-2: the strategy store is its own directory/unit, so it is only inspected when declared
    # (--strategy-dir); a strategy index co-located with the other stores still blocks that unit.
    classes = COMPONENT_CLASSES + ((StrategyComponent,) if "strategy" in unit.declared else ())
    for cls in classes:
        comp = cls(real)
        if isinstance(comp, SessionsComponent):
            comp.archive_only = sessions_archive_only
        if comp.exists():
            unit.components.append(comp)
            unit.reports.append(comp.inspect())
    found = {c.kind for c in unit.components}
    for kind in sorted(unit.declared - found):
        unit.reasons.append(
            f"declared as the {kind} store but no {kind} files were found in {real}"
        )
    try:
        unit.marker = read_marker(real)
    except StoreMarkerError as exc:
        unit.reasons.append(str(exc))
    managed = unit.managed_names() | {MARKER_NAME}
    for entry in sorted(os.listdir(real)):
        if entry in managed:
            continue
        if entry in FOREIGN_VECTOR_FILES:
            unit.reasons.append(
                f"{entry} ({FOREIGN_VECTOR_FILES[entry]}) is co-located: swapping this directory "
                "would leave a 1024-d index inside a store marked 256-d; migrate it separately first"
            )
        elif entry.endswith((".faiss", ".npy")):
            unit.warnings.append(f"unmanaged vector file {entry} is carried over untouched")
        unit.unmanaged.append(entry)
    unit.total_bytes = dir_size(real)

    for rep in unit.reports:
        for reason in rep.blocked:
            unit.reasons.append(f"{rep.kind}: {reason}")
        for warn in rep.warnings:
            unit.warnings.append(f"{rep.kind}: {warn}")
    vector_reports = [r for r in unit.reports if r.dim is not None]
    if unit.reasons:
        unit.status = BLOCKED
    elif unit.marker is not None:
        if not marker_matches(unit.marker, identity):
            unit.status = BLOCKED
            unit.reasons.append(
                f"{MARKER_NAME} names a different embedding identity "
                f"({unit.marker.get('model')} dim={unit.marker.get('dim')}): refusing to re-embed"
            )
        elif any(r.dim != TARGET_DIM for r in vector_reports):
            unit.status = BLOCKED
            unit.reasons.append(
                f"mixed store: marked dim={TARGET_DIM} but vector files have dims "
                f"{sorted({r.dim for r in vector_reports})}"
            )
        else:
            unit.status = ALREADY_MIGRATED
    elif any(r.dim == TARGET_DIM for r in vector_reports):
        unit.status = BLOCKED
        unit.reasons.append(f"{TARGET_DIM}-d vectors without {MARKER_NAME}: unknown provenance")
    elif not vector_reports:
        unit.status = NOTHING_TO_DO
    else:
        unit.status = READY
    return unit


def discover_dirs(args: argparse.Namespace) -> dict[Path, tuple[list[str], set[str]]]:
    """Map resolved dir -> (paths as given, declared component kinds)."""
    found: dict[Path, tuple[list[str], set[str]]] = {}

    def add(p: str | Path, kind: str | None) -> None:
        real = Path(os.path.realpath(p))
        given, declared = found.setdefault(real, ([], set()))
        if str(p) not in given:
            given.append(str(p))
        if kind:
            declared.add(kind)

    for flag, kind in KIND_OF_FLAG.items():
        value = getattr(args, flag, None)
        if value:
            add(value, kind)
    if getattr(args, "stores_dir", None):
        root = Path(args.stores_dir)
        if not root.is_dir():
            raise Refused([f"--stores-dir {root} is not a directory"])
        candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
        for cand in candidates:
            if has_component_files(cand):
                add(cand, None)
    if not found:
        raise Refused(
            ["no store given: pass --stores-dir and/or --episodic-dir/--skill-dir/--session-dir"]
        )
    return found


def build_plan(args: argparse.Namespace, identity: dict[str, Any]) -> dict[str, Any]:
    dirs = discover_dirs(args)
    units = [
        inspect_unit(
            p,
            given,
            declared,
            identity,
            sessions_archive_only=bool(getattr(args, "sessions_archive_only", False)),
        )
        for p, (given, declared) in sorted(dirs.items())
    ]
    reasons = []
    paths = [u.path for u in units]
    for i, a in enumerate(paths):
        for b in paths[i + 1 :]:
            if a in b.parents or b in a.parents:
                reasons.append(f"store directories are nested: {a} and {b}")
    return {
        "schema": SCHEMA,
        "target": identity,
        "units": units,
        "plan_refusals": reasons,
        "blocked": any(u.status == BLOCKED for u in units),
    }


def plan_to_json(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": plan["schema"],
        "target": plan["target"],
        "blocked": plan["blocked"],
        "plan_refusals": plan["plan_refusals"],
        "units": [u.to_dict() for u in plan["units"]],
    }


def render_plan(plan: dict[str, Any]) -> str:
    t = plan["target"]
    lines = [
        f"target  {t['arm_id']}  model={t['model']}",
        f"        revision={t['revision']} dtype={t['dtype']} dim={t['dim']} (native {t['native_dim']})",
        f"        prompt_hash={t['prompt_hash'][:16]}.. digest={t['digest'][:16]}..",
        f"        document prefix={t['doc_prompt']!r}",
    ]
    for unit in plan["units"]:
        lines.append("")
        lines.append(
            f"store directory {unit.path}  [{unit.status}]  total {human(unit.total_bytes)}"
        )
        if unit.marker:
            lines.append(f"  marker: dim={unit.marker.get('dim')} model={unit.marker.get('model')}")
        for rep in unit.reports:
            lines.append(
                f"  {rep.kind:<9} {rep.status:<11} vectors={rep.vectors} dim={rep.dim} "
                f"id_map={rep.id_map_len} rows={rep.rows}"
            )
            for name, size in rep.sizes.items():
                lines.append(f"      {name:<28} {human(size):>10}")
            lines.append(f"      text source: {rep.text_source}")
            for reason in rep.blocked:
                lines.append(f"      BLOCKED: {reason}")
        for reason in unit.reasons:
            if not any(reason.endswith(b) for r in unit.reports for b in r.blocked):
                lines.append(f"  BLOCKED: {reason}")
        for warn in unit.warnings:
            lines.append(f"  warning: {warn}")
        lines.append(f"  unmanaged entries carried by hard link: {len(unit.unmanaged)}")
    for reason in plan["plan_refusals"]:
        lines.append(f"REFUSED: {reason}")
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────────────── preflight


def _proc_holders(prefix: str, proc_root: str, own_pid: int) -> list[str]:
    holders: list[str] = []
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit() and int(p) != own_pid]
    except OSError:
        return holders
    for pid in pids:
        hit = None
        fd_dir = os.path.join(proc_root, pid, "fd")
        try:
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                if target.startswith(prefix):
                    hit = f"fd -> {target}"
                    break
        except OSError:
            pass
        if hit is None:
            try:
                with open(
                    os.path.join(proc_root, pid, "maps"), encoding="utf-8", errors="replace"
                ) as fh:
                    for line in fh:
                        idx = line.find(prefix)
                        if idx >= 0:
                            hit = f"mmap {line[idx:].strip()}"
                            break
            except OSError:
                pass
        if hit:
            holders.append(f"pid {pid} holds {hit}")
    return holders


def live_indicators(unit: Unit, proc_root: str = "/proc") -> list[str]:
    """Evidence of a live writer in ``unit.path``. Empty list = none seen (not a proof)."""
    found: list[str] = []
    root = unit.path
    for comp in unit.components:
        for db in comp.databases:
            for suffix in ("-wal", "-shm", "-journal"):
                side = root / (db + suffix)
                if side.exists():
                    found.append(
                        f"sqlite sidecar {side.name} present ({side.stat().st_size} bytes): an open "
                        "connection or an unclean shutdown"
                    )
    for entry in sorted(os.listdir(root)):
        if not entry.endswith(".lock"):
            continue
        path = root / entry
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            found.append(f"flock held on {entry}")
        except OSError:
            pass
        finally:
            os.close(fd)
    found.extend(_proc_holders(str(root) + "/", proc_root, os.getpid()))
    return found


def sibling(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.name}.{tag}")


def preflight(
    plan: dict[str, Any],
    *,
    out_dir: Path | None,
    assume_stopped: bool,
    require_attestation: bool,
    proc_root: str = "/proc",
) -> list[str]:
    """Reasons apply must refuse. ``require_attestation`` False = dry-run (reports, not gates)."""
    reasons = list(plan["plan_refusals"])
    units: list[Unit] = plan["units"]
    if require_attestation and not assume_stopped:
        reasons.append(
            "--assume-stopped not given: stop every writer (API workers, autopilot, q-scorer, "
            "distillation/seeding jobs, REPL sessions) and pass it to attest that"
        )
    for unit in units:
        if unit.status == BLOCKED:
            reasons.extend(f"BLOCKED {unit.path}: {r}" for r in (unit.reasons or ["blocked"]))
            continue
        if unit.status in (ALREADY_MIGRATED, NOTHING_TO_DO):
            continue
        for old in sorted(unit.path.parent.glob(glob_escape(unit.path.name) + ".pre-eg2-*")):
            reasons.append(
                f"{old} exists: a previous migration's old store is still retained "
                "(confirm and remove it yourself; this tool never deletes it)"
            )
        for stale in sorted(unit.path.parent.glob(glob_escape(unit.path.name) + ".migrating-*")):
            reasons.append(
                f"{stale} exists: a stale migrating copy from an earlier run; remove it first"
            )
        for ind in live_indicators(unit, proc_root):
            reasons.append(f"live indicator in {unit.path}: {ind}")
        need = sum(r.sizes.get(n, 0) for r in unit.reports for n in r.sizes)
        need = int(need * 1.25) + (64 << 20)
        free = shutil.disk_usage(unit.path.parent).free
        if free < need:
            reasons.append(
                f"not enough free space beside {unit.path}: need ~{human(need)}, have {human(free)}"
            )
    if out_dir is not None:
        od = Path(os.path.realpath(out_dir))
        for unit in units:
            if od == unit.path or unit.path in od.parents or od in unit.path.parents:
                reasons.append(f"--out-dir {od} overlaps the store directory {unit.path}")
    return reasons


def glob_escape(name: str) -> str:
    import glob as _glob

    return _glob.escape(name)


# ───────────────────────────────────────────────────────────────────────────── copy + re-embed


def _link_or_copy(src: Path, dst: Path, mode_log: dict[str, str], rel: str) -> None:
    if os.path.islink(src):
        os.symlink(os.readlink(src), dst)
        mode_log[rel] = "symlink"
    elif os.path.isdir(src):
        os.mkdir(dst)
        for name in sorted(os.listdir(src)):
            _link_or_copy(src / name, dst / name, mode_log, f"{rel}/{name}")
        shutil.copystat(src, dst, follow_symlinks=False)
        mode_log[rel] = "dir"
    else:
        try:
            os.link(src, dst, follow_symlinks=False)
            mode_log[rel] = "hardlink"
        except OSError:
            shutil.copy2(src, dst, follow_symlinks=False)
            mode_log[rel] = "copy"


def copy_unit(unit: Unit, migrating: Path, log: RunLog) -> dict[str, Any]:
    """Create ``migrating`` beside the live dir: managed files copied (hashed), the rest linked."""
    migrating.mkdir()
    fsync_path(migrating.parent)
    managed_copy: dict[str, dict[str, Any]] = {}
    old_vectors: dict[str, dict[str, Any]] = {}
    for comp in unit.components:
        for name in comp.copy_files:
            src = unit.path / name
            if src.exists():
                digest, size = copy_hashed(src, migrating / name)
                managed_copy[name] = {"sha256": digest, "bytes": size}
        for name in comp.vector_files:
            src = unit.path / name
            if src.exists():
                old_vectors[name] = {"sha256": sha256_file(src), "bytes": src.stat().st_size}
    carried: dict[str, str] = {}
    managed = unit.managed_names()
    for entry in sorted(os.listdir(unit.path)):
        if entry in managed:
            continue
        _link_or_copy(unit.path / entry, migrating / entry, carried, entry)
    log.event(
        "copied",
        unit=str(unit.path),
        managed_files=len(managed_copy),
        carried=len([m for m in carried.values() if m != "dir"]),
        copied_instead_of_linked=len([m for m in carried.values() if m == "copy"]),
    )
    return {"managed_copy": managed_copy, "old_vectors": old_vectors, "carried": carried}


def reembed_faiss_component(
    comp: FaissComponent, migrating: Path, client: EmbedClient, log: RunLog
) -> dict[str, Any]:
    faiss = _faiss()
    id_map = comp._load_id_map()
    n = len(id_map)
    mat = np.empty((n, TARGET_DIM), dtype=np.float32)
    pos = 0
    next_report = 0.1
    t0 = time.monotonic()
    for ids, texts in comp.iter_texts():
        vecs = client.embed_documents(texts)
        if vecs.shape != (len(ids), TARGET_DIM):
            raise EmbedError(f"{comp.kind}: embed returned {vecs.shape} for {len(ids)} texts")
        mat[pos : pos + len(ids)] = vecs
        pos += len(ids)
        if n and pos / n >= next_report:
            log.event(
                "embedding",
                store=comp.kind,
                done=pos,
                total=n,
                elapsed_s=round(time.monotonic() - t0, 1),
            )
            next_report += 0.1
    if pos != n:
        raise EmbedError(f"{comp.kind}: embedded {pos} of {n} positions")
    index = faiss.IndexFlatIP(TARGET_DIM)
    if n:
        index.add(mat)
    out = migrating / comp.index_name
    faiss.write_index(index, str(out))
    fsync_path(out)
    return {"vectors": n, "dim": TARGET_DIM, "file": comp.index_name}


def rewrite_sessions(
    comp: SessionsComponent, migrating: Path, rep: ComponentReport
) -> dict[str, Any]:
    if rep.status == NO_VECTORS:
        # nothing to rewrite: the file is absent/zero-byte and the store creates it fresh
        src = comp.dir / "session_embeddings.npy"
        if src.exists():
            shutil.copy2(src, migrating / src.name)
        return {"vectors": 0, "dim": None, "file": None}
    rows = int(rep.detail["capacity_rows"])
    out = migrating / "session_embeddings.npy"
    np.save(out, np.zeros((rows, TARGET_DIM), dtype=np.float32))
    fsync_path(out)
    info: dict[str, Any] = {
        "vectors": 0,
        "dim": TARGET_DIM,
        "file": out.name,
        "capacity_rows": rows,
    }
    if rep.detail.get("archive_only"):
        src = comp.dir / "session_embeddings.npy"
        info.update(
            {
                "mode": "archive-only",
                "archived_vector_rows": rep.detail.get("archived_vector_rows"),
                "archived_sha256": sha256_file(src),
                "archived_in": "the .pre-eg2-<run_id> directory (rollback restores it)",
            }
        )
    return info


# ───────────────────────────────────────────────────────────────────────────── validation


def _nn_selfcheck(index: Any, positions: Sequence[int]) -> list[int]:
    bad: list[int] = []
    for p in positions:
        vec = index.reconstruct(int(p)).astype(np.float32)
        scores, labels = index.search(vec.reshape(1, -1), 1)
        label, score = int(labels[0][0]), float(scores[0][0])
        if label == p:
            continue
        if label >= 0 and score >= 1 - 1e-4:
            other = index.reconstruct(label).astype(np.float32)
            if np.allclose(other, vec, atol=1e-5):  # duplicate text -> identical vector: a tie
                continue
        bad.append(int(p))
    return bad


def validate_unit(
    unit: Unit,
    migrating: Path,
    copy_info: dict[str, Any],
    client: EmbedClient,
    identity: dict[str, Any],
    *,
    run_id: str,
    nn_sample: int,
    semantic_sample: int,
) -> dict[str, Any]:
    failures: list[str] = []
    result: dict[str, Any] = {"components": {}}

    # marker: written, loadable, agrees with the real vector dims
    try:
        marker = verify_store_marker(migrating, expected_dim=TARGET_DIM, expected_identity=identity)
        if marker.get("legacy"):
            failures.append("marker missing from the migrated copy")
    except StoreMarkerError as exc:
        failures.append(f"marker: {exc}")

    # managed copies byte-identical to their sources
    for name, rec in copy_info["managed_copy"].items():
        got = sha256_file(migrating / name)
        if got != rec["sha256"]:
            failures.append(f"{name}: copy hash {got[:12]} != source {rec['sha256'][:12]}")

    # directory contents: nothing lost, nothing extra
    expect = set(os.listdir(unit.path)) | {MARKER_NAME}
    have = set(os.listdir(migrating))
    for gone in sorted(expect - have):
        failures.append(f"entry missing from the migrated copy: {gone}")
    for extra in sorted(have - expect):
        failures.append(f"unexpected entry in the migrated copy: {extra}")

    rng = random.Random(f"reembed-eg2:{run_id}")
    for comp, rep in zip(unit.components, unit.reports):
        comp_res: dict[str, Any] = {"status": rep.status}
        result["components"][comp.kind] = comp_res
        if isinstance(comp, SessionsComponent):
            new = migrating / "session_embeddings.npy"
            if rep.status != NO_VECTORS:
                arr = np.load(new, mmap_mode="r")
                comp_res.update({"shape": list(arr.shape)})
                if arr.shape != (rep.detail["capacity_rows"], TARGET_DIM):
                    failures.append(f"sessions: npy shape {arr.shape}")
                if arr.dtype != np.float32 or np.any(np.asarray(arr) != 0):
                    failures.append("sessions: migrated npy is not an all-zero float32 array")
            continue
        if rep.status == NO_VECTORS:
            continue
        assert isinstance(comp, FaissComponent)
        faiss = _faiss()
        try:
            index = faiss.read_index(str(migrating / comp.index_name))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{comp.kind}: cannot read the new index: {exc}")
            continue
        n_old = rep.vectors
        comp_res.update({"ntotal": int(index.ntotal), "d": int(index.d)})
        if index.d != TARGET_DIM:
            failures.append(f"{comp.kind}: new index dim {index.d} != {TARGET_DIM}")
        if index.ntotal != n_old:
            failures.append(f"{comp.kind}: row count {index.ntotal} != {n_old}")
        new_map = FaissComponent._load_id_map(type(comp)(migrating))  # same ids, same order
        old_map = comp._load_id_map()
        if new_map != old_map:
            failures.append(f"{comp.kind}: id_map differs from the source (ids must map 1:1)")
        try:
            with _ro_connect(migrating / comp.db_name) as con:
                rows_new = int(con.execute(f"SELECT COUNT(*) FROM {comp.table}").fetchone()[0])
            if rep.rows is not None and rows_new != rep.rows:
                failures.append(f"{comp.kind}: sqlite rows {rows_new} != {rep.rows}")
        except sqlite3.Error as exc:
            failures.append(f"{comp.kind}: cannot read the copied db: {exc}")
        n = int(index.ntotal)
        if n == 0:
            continue
        # every vector finite and unit-norm
        nonfinite = 0
        offnorm = 0
        for start in range(0, n, 8192):
            block = index.reconstruct_n(start, min(8192, n - start))
            nonfinite += int((~np.isfinite(block)).any(axis=1).sum())
            norms = np.linalg.norm(np.nan_to_num(block), axis=1)
            offnorm += int((np.abs(norms - 1.0) > 1e-3).sum())
        comp_res.update({"nonfinite_rows": nonfinite, "off_unit_norm_rows": offnorm})
        if nonfinite:
            failures.append(f"{comp.kind}: {nonfinite} non-finite vector(s)")
        if offnorm:
            failures.append(f"{comp.kind}: {offnorm} vector(s) not unit-norm")
        # nearest-neighbour self-check (index consistency)
        positions = sorted(rng.sample(range(n), min(nn_sample, n)))
        bad_nn = _nn_selfcheck(index, positions)
        comp_res["nn_self_check"] = {"sampled": len(positions), "failed": len(bad_nn)}
        if bad_nn:
            failures.append(
                f"{comp.kind}: nearest-neighbour self-check failed at positions {bad_nn[:5]}"
            )
        # semantic self-check: re-embed a sample (different batching) and compare to the stored vector
        sem_pos = sorted(rng.sample(range(n), min(semantic_sample, n)))
        if sem_pos:
            texts = comp.texts_at(sem_pos)
            fresh = client.embed_documents(list(reversed(texts)))[::-1]
            cos = [
                float(np.dot(fresh[i], index.reconstruct(int(p)))) for i, p in enumerate(sem_pos)
            ]
            comp_res["semantic_self_check"] = {
                "sampled": len(sem_pos),
                "min_cosine": round(min(cos), 6),
            }
            if min(cos) < SEMANTIC_MIN_COSINE:
                failures.append(
                    f"{comp.kind}: semantic self-check failed (min cosine {min(cos):.8f} "
                    f"< {SEMANTIC_MIN_COSINE}): "
                    "stored vectors do not belong to their rows"
                )
    if failures:
        raise ValidationError(failures)
    return result


# ───────────────────────────────────────────────────────────────────────────── swap


def _rename(src: Path, dst: Path) -> None:
    os.rename(src, dst)


def _try_exchange(a: Path, b: Path) -> bool:
    """``renameat2(RENAME_EXCHANGE)``: swap two directories atomically. False = unsupported here."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = libc.renameat2
    except (OSError, AttributeError):
        return False
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    at_fdcwd, rename_exchange = -100, 2
    if fn(at_fdcwd, os.fsencode(a), at_fdcwd, os.fsencode(b), rename_exchange) == 0:
        return True
    err = ctypes.get_errno()
    if err in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EPERM):
        return False
    raise OSError(err, os.strerror(err), str(a))


def atomic_swap(live: Path, new: Path, pre: Path) -> str:
    """Afterwards ``live`` holds the new content and ``pre`` the old; never deletes anything."""
    if pre.exists() or os.path.lexists(pre):
        raise FileExistsError(f"{pre} already exists")
    if _try_exchange(live, new):  # live path was never absent
        _rename(new, pre)
        return "exchange"
    _rename(live, pre)
    try:
        _rename(new, live)
    except BaseException:
        _rename(pre, live)
        raise
    return "rename"


def revert_swap(live: Path, new: Path, pre: Path) -> None:
    """Undo ``atomic_swap``: live (new content) -> ``new`` name, ``pre`` -> live."""
    _rename(live, new)
    _rename(pre, live)


def swap_all(pairs: Sequence[tuple[Path, Path, Path]], log: RunLog) -> list[str]:
    done: list[tuple[Path, Path, Path]] = []
    methods: list[str] = []
    try:
        for live, new, pre in pairs:
            methods.append(atomic_swap(live, new, pre))
            done.append((live, new, pre))
            log.event("swapped", live=str(live), pre=str(pre), method=methods[-1])
    except BaseException as exc:
        for live, new, pre in reversed(done):
            try:
                revert_swap(live, new, pre)
                log.event("swap_reverted", live=str(live))
            except BaseException as rexc:  # pragma: no cover - needs two simultaneous faults
                log.event(
                    "CRITICAL_swap_revert_failed", live=str(live), pre=str(pre), error=repr(rexc)
                )
                raise SwapError(
                    f"swap failed ({exc!r}) AND reverting {live} failed ({rexc!r}); "
                    f"the old store is intact at {pre}; restore it by hand: "
                    f"mv -T {shlex.quote(str(live))} {shlex.quote(str(new))}; "
                    f"mv -T {shlex.quote(str(pre))} {shlex.quote(str(live))}"
                ) from rexc
        raise SwapError(f"swap failed and was reverted: {exc!r}") from exc
    return methods


# ───────────────────────────────────────────────────────────────────────────── manifest + rollback


def hash_tree_files(root: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for name in sorted(names):
        p = root / name
        if p.exists():
            out[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def seal(manifest: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(body).encode())
    return manifest


def verify_seal(manifest: dict[str, Any]) -> bool:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return manifest.get("manifest_sha256") == sha256_bytes(canonical_json(body).encode())


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def rollback_script_text(manifest: dict[str, Any]) -> str:
    run_id = manifest["run_id"]
    q = shlex.quote
    lines = [
        "#!/bin/bash",
        f"# Generated by reembed_stores_eg2.py for run {run_id} at {manifest['created_utc']}.",
        "# Restores the pre-EG2 store directories. Needs only bash, sha256sum and mv.",
        "set -euo pipefail",
        'if [[ "${1:-}" != "--assume-stopped" ]]; then',
        '  echo "refusing: stop every consumer of the stores first, then pass --assume-stopped" >&2',
        "  exit 2",
        "fi",
        "",
        "# 1. every old file must still hash to what the manifest recorded",
    ]
    for unit in manifest["units"]:
        pre = q(unit["pre"])
        lines.append(f"[[ -d {pre} ]] || {{ echo 'missing old store' >&2; exit 2; }}")
        lines.append(f"(cd {pre} && sha256sum --check --quiet --strict -) <<'SUMS'")
        for name, rec in sorted(unit["old_files"].items()):
            lines.append(f"{rec['sha256']}  {name}")
        lines.append("SUMS")
    lines += ["", "# 2. nothing may block the renames"]
    for unit in manifest["units"]:
        lines.append(
            f"[[ ! -e {q(unit['rolledback'])} ]] || {{ echo 'target exists' >&2; exit 2; }}"
        )
    lines += [
        "",
        "# 3. swap back; the EG2 directory is kept (renamed), never deleted",
        "done_live=(); done_rb=(); done_pre=()",
        "undo() {",
        "  local i",
        "  for ((i=${#done_live[@]}-1; i>=0; i--)); do",
        '    mv -T "${done_live[i]}" "${done_pre[i]}" && mv -T "${done_rb[i]}" "${done_live[i]}"',
        "  done",
        "}",
        "swap_back() {  # live rolledback pre",
        '  mv -T "$1" "$2" || { undo; echo "rename failed for $1" >&2; exit 5; }',
        '  mv -T "$3" "$1" || { mv -T "$2" "$1"; undo; echo "rename failed for $1" >&2; exit 5; }',
        '  done_live+=("$1"); done_rb+=("$2"); done_pre+=("$3")',
        "}",
    ]
    for unit in manifest["units"]:
        lines.append(f"swap_back {q(unit['live'])} {q(unit['rolledback'])} {q(unit['pre'])}")
    lines.append(f'echo "rolled back run {run_id}; EG2 directories kept as *.eg2-rolledback-*"')
    lines.append("")
    return "\n".join(lines)


def write_rollback_script(out_dir: Path, manifest: dict[str, Any]) -> Path:
    path = out_dir / f"rollback-{manifest['run_id']}.sh"
    path.write_text(rollback_script_text(manifest), encoding="utf-8")
    os.chmod(path, 0o755)
    return path


def run_rollback(
    manifest_path: Path,
    *,
    assume_stopped: bool,
    allow_diverged: bool,
    out_dir: Path | None,
    log: RunLog,
    proc_root: str = "/proc",
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or not verify_seal(manifest):
        raise RollbackError(f"{manifest_path} is not an intact {SCHEMA} manifest (seal mismatch)")
    reasons: list[str] = []
    if not assume_stopped:
        reasons.append("--assume-stopped not given: stop every consumer of the stores first")
    pairs = []
    diverged: list[str] = []
    for u in manifest["units"]:
        live, pre, rb = Path(u["live"]), Path(u["pre"]), Path(u["rolledback"])
        if not pre.is_dir():
            reasons.append(f"{pre} (the old store) is missing")
            continue
        if not live.is_dir():
            reasons.append(f"{live} is missing")
            continue
        if os.path.lexists(rb):
            reasons.append(f"{rb} already exists")
        for name, rec in u["old_files"].items():
            p = pre / name
            if not p.exists() or sha256_file(p) != rec["sha256"]:
                reasons.append(f"old file {pre / name} does not match the manifest hash")
        for name, rec in u["new_files"].items():
            p = live / name
            if not p.exists() or sha256_file(p) != rec["sha256"]:
                diverged.append(str(p))
        probe = Unit(path=live, given=[str(live)], declared=set())
        probe.components = [cls(live) for cls in COMPONENT_CLASSES if cls(live).exists()]
        reasons.extend(f"live indicator in {live}: {i}" for i in live_indicators(probe, proc_root))
        pairs.append((live, rb, pre))
    if diverged and not allow_diverged:
        reasons.append(
            f"{len(diverged)} EG2 file(s) changed since the swap (writes landed in the EG2 store): "
            f"{diverged[:3]}; --allow-diverged restores anyway (EG2 dir is kept)"
        )
    if reasons:
        raise RollbackError("; ".join(reasons))
    done: list[tuple[Path, Path, Path]] = []
    try:
        for live, rb, pre in pairs:
            _rename(live, rb)
            _rename(pre, live)
            done.append((live, rb, pre))
            log.event("rolled_back", live=str(live), eg2_kept=str(rb))
    except BaseException as exc:
        for live, rb, pre in reversed(done):
            _rename(live, pre)
            _rename(rb, live)
        raise RollbackError(f"rollback failed and was undone: {exc!r}") from exc
    receipt = {
        "schema": SCHEMA,
        "kind": "rollback-receipt",
        "run_id": manifest["run_id"],
        "rolled_back_utc": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "diverged_files": diverged,
        "units": [{"live": str(a), "eg2_kept": str(b)} for a, b, _ in pairs],
    }
    target = Path(out_dir) if out_dir else Path(manifest["out_dir"])
    if target.is_dir():
        write_json(target / f"rollback-receipt-{manifest['run_id']}.json", receipt)
    return receipt


# ───────────────────────────────────────────────────────────────────────────── run


@contextmanager
def tool_lock(out_dir: Path) -> Iterator[None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / ".reembed-eg2.lock", "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused(
                [f"another reembed_stores_eg2 run holds {out_dir}/.reembed-eg2.lock"]
            ) from exc
        yield
    finally:
        fh.close()


def planned_operations(
    plan: dict[str, Any], run_id: str, out_dir: Path | None
) -> list[dict[str, Any]]:
    """The exact files an apply would create or rename (the dry-run output)."""
    ops: list[dict[str, Any]] = []
    for unit in plan["units"]:
        if unit.status != READY:
            ops.append({"unit": str(unit.path), "action": "skip", "why": unit.status})
            continue
        mig = sibling(unit.path, f"migrating-{run_id}")
        pre = sibling(unit.path, f"pre-eg2-{run_id}")
        ops.append({"unit": str(unit.path), "action": "create_dir", "path": str(mig)})
        for comp, rep in zip(unit.components, unit.reports):
            for name in comp.copy_files:
                src = unit.path / name
                if src.exists():
                    ops.append(
                        {"action": "copy", "path": str(mig / name), "bytes": src.stat().st_size}
                    )
            if rep.status == NO_VECTORS:
                continue
            for name in comp.vector_files:
                rows = (
                    rep.detail.get("capacity_rows", 0) if comp.kind == "sessions" else rep.vectors
                )
                ops.append(
                    {
                        "action": "write_new",
                        "path": str(mig / name),
                        "approx_bytes": int(rows * TARGET_DIM * 4),
                        "source_dim": rep.dim,
                        "target_dim": TARGET_DIM,
                        "replaces": str(unit.path / name),
                    }
                )
        ops.append({"action": "hardlink_unmanaged", "count": len(unit.unmanaged), "into": str(mig)})
        ops.append({"action": "write", "path": str(mig / MARKER_NAME)})
        ops.append({"action": "validate", "path": str(mig)})
        ops.append(
            {
                "action": "swap",
                "rename_old_to": str(pre),
                "rename_new_to": str(unit.path),
                "old_kept_intact": True,
            }
        )
    if out_dir is not None:
        for name in (f"manifest-{run_id}.json", f"rollback-{run_id}.sh", f"run-{run_id}.log.jsonl"):
            ops.append({"action": "write", "path": str(out_dir / name)})
        ops.append({"action": "write", "path": str(out_dir / f"swap-receipt-{run_id}.json")})
        ops.append({"action": "lock", "path": str(out_dir / ".reembed-eg2.lock")})
    return ops


def safe_remove_own_copy(path: Path, live: Path, run_id: str, log: RunLog) -> None:
    """Delete ONLY this run's own ``.migrating-<run_id>`` copy (never the live or pre dir)."""
    if (
        path.name == f"{live.name}.migrating-{run_id}"
        and path.parent == live.parent
        and path != live
        and not path.is_symlink()
        and path.is_dir()
    ):
        shutil.rmtree(path)
        log.event("removed_own_migrating_copy", path=str(path))


def run_apply(
    args: argparse.Namespace, plan: dict[str, Any], identity: dict[str, Any], spec: Any, log: RunLog
) -> dict[str, Any]:
    run_id = args.run_id
    out_dir = Path(args.out_dir)
    units: list[Unit] = [u for u in plan["units"] if u.status == READY]
    if not units:
        log.event("nothing_to_migrate", statuses=[u.status for u in plan["units"]])
        return {"run_id": run_id, "migrated": [], "statuses": [u.status for u in plan["units"]]}
    client = EmbedClient(
        args.embed_url,
        spec,
        batch_size=args.batch_size,
        retries=args.retries,
        timeout_s=args.timeout,
        backoff_s=args.retry_backoff,
    )
    # Probe first: a wrong endpoint is refused before anything (even --out-dir) is created.
    client_probe = client.probe(check_model=not args.skip_model_check)
    log.event(
        "endpoint_ok", **{k: v for k, v in client_probe.items() if k not in ("url", "models")}
    )
    with tool_lock(out_dir):
        log.attach(out_dir / f"run-{run_id}.log.jsonl")
        log.event("start", run_id=run_id, units=[str(u.path) for u in units])
        gguf = None
        if args.gguf_path:
            gp = Path(args.gguf_path)
            gguf = {"path": str(gp), "sha256": sha256_file(gp), "bytes": gp.stat().st_size}
            if gguf["sha256"] != EG2_GGUF_SHA256:
                raise Refused(
                    [
                        f"--gguf-path {gp} sha256 {gguf['sha256'][:16]}.. is not the adopted "
                        f"EmbeddingGemma 2 BF16 pin {EG2_GGUF_SHA256[:16]}..; refusing to mark "
                        "stores with another model file's identity"
                    ]
                )

        before = {u.path: tree_signature(u.path) for u in units}
        copies: dict[Path, dict[str, Any]] = {}
        created: list[tuple[Path, Path]] = []
        unit_results: list[dict[str, Any]] = []
        try:
            for unit in units:
                mig = sibling(unit.path, f"migrating-{run_id}")
                copy_info = copy_unit(unit, mig, log)
                created.append((unit.path, mig))
                copies[unit.path] = copy_info
                reembed_info: dict[str, Any] = {}
                for comp, rep in zip(unit.components, unit.reports):
                    if isinstance(comp, FaissComponent):
                        if rep.status in (READY, EMPTY):
                            reembed_info[comp.kind] = reembed_faiss_component(
                                comp, mig, client, log
                            )
                    elif isinstance(comp, SessionsComponent):
                        reembed_info[comp.kind] = rewrite_sessions(comp, mig, rep)
                from_info = {
                    "dims": sorted({r.dim for r in unit.reports if r.dim is not None}),
                    "assumed_source": "BGE-large-en-v1.5 (unmarked legacy store)",
                }
                marker = build_marker(
                    identity,
                    run_id,
                    from_info,
                    model_file=(Path(gguf["path"]).name if gguf else EG2_MODEL_FILE),
                    gguf_sha256=(gguf["sha256"] if gguf else EG2_GGUF_SHA256),
                    gguf_sha256_verified=bool(gguf),
                )
                (mig / MARKER_NAME).write_text(
                    json.dumps(marker, indent=1, sort_keys=True) + "\n", encoding="utf-8"
                )
                fsync_path(mig / MARKER_NAME)
                fsync_path(mig)
                validation = validate_unit(
                    unit,
                    mig,
                    copy_info,
                    client,
                    identity,
                    run_id=run_id,
                    nn_sample=args.nn_sample,
                    semantic_sample=args.selfcheck_sample,
                )
                log.event("validated", unit=str(unit.path))
                unit_results.append(
                    {
                        "unit": unit,
                        "mig": mig,
                        "reembed": reembed_info,
                        "validation": validation,
                        "marker": marker,
                    }
                )
            # writers must still be stopped: nothing under the live dirs may have moved since the copy
            for unit in units:
                if tree_signature(unit.path) != before[unit.path]:
                    raise ValidationError(
                        [
                            f"{unit.path} changed while it was being migrated (a writer is alive?); aborted before the swap"
                        ]
                    )
                inds = live_indicators(unit, args.proc_root)
                if inds:
                    raise ValidationError([f"live indicator appeared in {unit.path}: {inds}"])
        except BaseException:
            if not args.keep_failed:
                for live, mig in created:
                    safe_remove_own_copy(mig, live, run_id, log)
            else:
                log.event("kept_failed_copies", paths=[str(m) for _l, m in created])
            raise

        manifest = {
            "schema": SCHEMA,
            "kind": "manifest",
            "run_id": run_id,
            "created_utc": utc_now(),
            "out_dir": str(out_dir),
            "tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "args": {k: v for k, v in sorted(vars(args).items()) if k != "cmd"},
            "target": identity,
            "gguf": gguf,
            "endpoint": client_probe,
            "embedding": {
                "batch_size": client.batch_size,
                "requests": client.requests,
                "retried": client.retried,
                "texts": client.texts_embedded,
            },
            "units": [],
        }
        pairs = []
        for res in unit_results:
            unit: Unit = res["unit"]
            mig: Path = res["mig"]
            info = copies[unit.path]
            pre = sibling(unit.path, f"pre-eg2-{run_id}")
            rb = sibling(unit.path, f"eg2-rolledback-{run_id}")
            managed = sorted(unit.managed_names())
            old_files = {**info["managed_copy"], **info["old_vectors"]}
            new_files = hash_tree_files(mig, managed + [MARKER_NAME])
            manifest["units"].append(
                {
                    "live": str(unit.path),
                    "given": unit.given,
                    "migrating": str(mig),
                    "pre": str(pre),
                    "rolledback": str(rb),
                    "components": [r.to_dict() for r in unit.reports],
                    "reembed": res["reembed"],
                    "validation": res["validation"],
                    "marker": res["marker"],
                    "old_files": old_files,
                    "new_files": new_files,
                    "old_tree": before[unit.path],
                    "unmanaged_carried": {
                        k: v for k, v in sorted(info["carried"].items()) if "/" not in k
                    },
                }
            )
            pairs.append((unit.path, mig, pre))
        seal(manifest)
        manifest_path = out_dir / f"manifest-{run_id}.json"
        write_json(manifest_path, manifest)
        script = write_rollback_script(out_dir, manifest)
        log.event(
            "manifest_written",
            path=str(manifest_path),
            sha256=manifest["manifest_sha256"],
            rollback_script=str(script),
        )

        try:
            methods = swap_all(pairs, log)
        except BaseException:
            for stale in (manifest_path, script):  # describe a swap that did not happen
                try:
                    os.replace(stale, stale.with_name(stale.name + ".swap-failed"))
                except OSError:
                    pass
            if not args.keep_failed:
                for live, mig, _pre in pairs:
                    safe_remove_own_copy(mig, live, run_id, log)
            raise

        # post-swap verification against the manifest
        problems: list[str] = []
        for u in manifest["units"]:
            live = Path(u["live"])
            for name, rec in u["new_files"].items():
                if sha256_file(live / name) != rec["sha256"]:
                    problems.append(f"{live / name} does not match the manifest after the swap")
            try:
                verify_store_marker(live, expected_dim=TARGET_DIM, expected_identity=identity)
            except StoreMarkerError as exc:
                problems.append(str(exc))
        receipt = {
            "schema": SCHEMA,
            "kind": "swap-receipt",
            "run_id": run_id,
            "swapped_utc": utc_now(),
            "manifest_sha256": manifest["manifest_sha256"],
            "swap_methods": methods,
            "post_swap_problems": problems,
        }
        write_json(out_dir / f"swap-receipt-{run_id}.json", receipt)
        log.event("done", problems=len(problems))
        if problems:
            raise ValidationError(problems)
    return {
        "run_id": run_id,
        "manifest": str(manifest_path),
        "rollback_script": str(script),
        "migrated": [u["live"] for u in manifest["units"]],
        "retained_old": [u["pre"] for u in manifest["units"]],
        "swap_methods": methods,
    }


# ───────────────────────────────────────────────────────────────────────────── CLI


def add_store_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("stores (explicit paths only; there are no built-in live paths)")
    g.add_argument(
        "--stores-dir", help="discover store directories in this dir and its immediate subdirs"
    )
    g.add_argument(
        "--episodic-dir", help="directory holding episodic.db + embeddings.faiss + id_map.npy"
    )
    g.add_argument(
        "--skill-dir",
        help="directory holding skills.db + skill_embeddings.faiss + skill_id_map.npy",
    )
    g.add_argument("--session-dir", help="directory holding sessions.db + session_embeddings.npy")
    g.add_argument(
        "--strategy-dir",
        help="directory holding strategies.db + strategy_embeddings.faiss + strategy_id_map.npy "
        "(autopilot's StrategyStore; its own unit, same marker)",
    )
    g.add_argument(
        "--sessions-archive-only",
        action="store_true",
        help="B-1: sessions keep no source text; archive their 1024-d vectors in the .pre-eg2 "
        "directory instead of blocking, and start the migrated store with an empty 256-d file",
    )
    p.add_argument(
        "--runtime", default=DEFAULT_RUNTIME, help="runtime string recorded in the identity"
    )
    p.add_argument("--json", action="store_true", help="machine-readable output on stdout")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="reembed_stores_eg2.py",
        description=__doc__.split("\n\n")[0],
        epilog="See the module docstring for the runbook, exit codes and the retraining flags.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="read-only survey of the stores (writes nothing)")
    add_store_args(p_plan)

    p_run = sub.add_parser("run", help="dry-run (default) or --apply the copy-then-swap migration")
    add_store_args(p_run)
    mode = p_run.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="the default: write nothing, embed nothing"
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="actually migrate (needs --embed-url, --out-dir, --assume-stopped)",
    )
    p_run.add_argument(
        "--out-dir", help="manifests, run log, rollback script, tool lock (required for --apply)"
    )
    p_run.add_argument("--embed-url", help="base URL of the EG2 llama-server (/v1/embeddings)")
    p_run.add_argument(
        "--assume-stopped", action="store_true", help="attest that every writer is stopped"
    )
    p_run.add_argument("--run-id", default=None, help="default: UTC timestamp + random suffix")
    p_run.add_argument("--batch-size", type=int, default=16)
    p_run.add_argument("--retries", type=int, default=4)
    p_run.add_argument("--retry-backoff", type=float, default=1.0)
    p_run.add_argument("--timeout", type=float, default=120.0)
    p_run.add_argument(
        "--nn-sample", type=int, default=64, help="nearest-neighbour self-check sample per store"
    )
    p_run.add_argument(
        "--selfcheck-sample",
        type=int,
        default=16,
        help="re-embed this many rows per store and compare",
    )
    p_run.add_argument("--gguf-path", help="optional: hash this GGUF into the manifest")
    p_run.add_argument(
        "--skip-model-check", action="store_true", help="do not require /v1/models to look like EG2"
    )
    p_run.add_argument(
        "--keep-failed", action="store_true", help="keep this run's .migrating-* copy on failure"
    )
    p_run.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)

    p_rb = sub.add_parser("rollback", help="restore the pre-EG2 directories recorded in a manifest")
    p_rb.add_argument("--manifest", required=True)
    p_rb.add_argument("--assume-stopped", action="store_true")
    p_rb.add_argument("--allow-diverged", action="store_true")
    p_rb.add_argument(
        "--out-dir", help="where to write the receipt (default: the manifest's out_dir)"
    )
    p_rb.add_argument("--json", action="store_true")
    p_rb.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)

    p_ck = sub.add_parser("check-marker", help="consumer-side load check of one store directory")
    p_ck.add_argument("dir")
    p_ck.add_argument("--expect-dim", type=int, default=TARGET_DIM)
    p_ck.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p_ck.add_argument("--expect-model-file", default=None, help="B-3: the pool's GGUF basename")
    p_ck.add_argument("--expect-gguf-sha256", default=None, help="B-3: the pool's GGUF sha256")
    return ap


def emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=1, sort_keys=True, default=str))
    else:
        print(text)


def cmd_plan(args: argparse.Namespace) -> int:
    identity = target_identity(target_spec(args.runtime))
    plan = build_plan(args, identity)
    emit(args, plan_to_json(plan), render_plan(plan))
    return EXIT_BLOCKED if plan["blocked"] else EXIT_OK


def cmd_run(args: argparse.Namespace, log: RunLog) -> int:
    spec = target_spec(args.runtime)
    identity = target_identity(spec)
    args.run_id = args.run_id or new_run_id()
    if not RUN_ID_RE.match(args.run_id):
        raise Refused([f"--run-id {args.run_id!r} must match {RUN_ID_RE.pattern}"])
    plan = build_plan(args, identity)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if plan["blocked"]:
        emit(args, plan_to_json(plan), render_plan(plan))
        log.event("blocked", run_id=args.run_id)
        return EXIT_BLOCKED
    if args.apply:
        missing = [
            f for f, v in (("--embed-url", args.embed_url), ("--out-dir", args.out_dir)) if not v
        ]
        if missing:
            raise Refused([f"--apply needs {' and '.join(missing)}"])
    reasons = preflight(
        plan,
        out_dir=out_dir,
        assume_stopped=args.assume_stopped,
        require_attestation=args.apply,
        proc_root=args.proc_root,
    )
    ops = planned_operations(plan, args.run_id, out_dir)
    if not args.apply:
        payload = {
            **plan_to_json(plan),
            "mode": "dry-run",
            "run_id": args.run_id,
            "would_refuse": reasons,
            "operations": ops,
        }
        text = (
            render_plan(plan)
            + "\n\nDRY RUN -- nothing was written or embedded. Operations an apply would perform:\n"
        )
        text += "\n".join("  " + json.dumps(op, sort_keys=True) for op in ops)
        if reasons:
            text += "\n\nan --apply would REFUSE:\n" + "\n".join("  - " + r for r in reasons)
        emit(args, payload, text)
        return EXIT_REFUSED if reasons else EXIT_OK
    if reasons:
        raise Refused(reasons)
    result = run_apply(args, plan, identity, spec, log)
    text = (
        f"run {result['run_id']} done.\n"
        + "\n".join(
            f"  migrated {m}\n  old store retained at {r}"
            for m, r in zip(result.get("migrated", []), result.get("retained_old", []))
        )
        + (
            f"\n  manifest: {result['manifest']}\n  rollback: {result['rollback_script']} --assume-stopped"
            "\nNEXT: restart consumers with the EG2 pool (embedding_dim=256, query prefix on); keep the"
            "\n.pre-eg2-* directories until the operator confirms their deletion."
            if result.get("migrated")
            else "\nnothing to migrate."
        )
    )
    emit(args, result, text)
    return EXIT_OK


def cmd_rollback(args: argparse.Namespace, log: RunLog) -> int:
    receipt = run_rollback(
        Path(args.manifest),
        assume_stopped=args.assume_stopped,
        allow_diverged=args.allow_diverged,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        log=log,
        proc_root=args.proc_root,
    )
    emit(
        args,
        receipt,
        f"rolled back run {receipt['run_id']}; EG2 directories kept: "
        + ", ".join(u["eg2_kept"] for u in receipt["units"]),
    )
    return EXIT_OK


def cmd_check_marker(args: argparse.Namespace) -> int:
    identity = target_identity(target_spec(args.runtime))
    marker = verify_store_marker(
        Path(args.dir),
        expected_dim=args.expect_dim,
        expected_identity=identity if args.expect_dim == TARGET_DIM else None,
        expected_model_file=args.expect_model_file,
        expected_gguf_sha256=args.expect_gguf_sha256,
    )
    print(json.dumps(marker, indent=1, sort_keys=True, default=str))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = RunLog(quiet=getattr(args, "json", False))
    try:
        if args.cmd == "plan":
            return cmd_plan(args)
        if args.cmd == "run":
            return cmd_run(args, log)
        if args.cmd == "rollback":
            return cmd_rollback(args, log)
        return cmd_check_marker(args)
    except Refused as exc:
        print("REFUSED:\n" + "\n".join(f"  - {r}" for r in exc.reasons), file=sys.stderr)
        return EXIT_REFUSED
    except StoreMarkerError as exc:
        print(f"MARKER: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except BlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except ValidationError as exc:
        print(
            "VALIDATION FAILED (the live stores were not swapped / are unchanged):\n"
            + "\n".join(f"  - {f}" for f in exc.failures),
            file=sys.stderr,
        )
        return EXIT_INVALID
    except EmbedError as exc:
        print(f"EMBEDDING FAILED (the live stores are unchanged): {exc}", file=sys.stderr)
        return EXIT_INVALID
    except SwapError as exc:
        print(f"SWAP FAILED: {exc}", file=sys.stderr)
        return EXIT_SWAP
    except RollbackError as exc:
        print(f"ROLLBACK REFUSED: {exc}", file=sys.stderr)
        return EXIT_ROLLBACK
    except OSError as exc:
        print(
            f"I/O ERROR (live stores unchanged unless a swap was reported): {exc!r}",
            file=sys.stderr,
        )
        return EXIT_INVALID


__all__ = [
    "EmbedClient",
    "EmbedError",
    "MARKER_NAME",
    "StoreMarkerError",
    "main",
    "read_marker",
    "verify_store_marker",
]

if __name__ == "__main__":
    sys.exit(main())
