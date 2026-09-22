"""TD-10: model-based counterfactual evaluation of the TD-9 routing policy.

TD-9's 93.5% is **agreement with the incumbent under a frozen label that
describes the incumbent's outcome**. Enforcement needs an estimate of what
would happen under the typed policy's chosen actions, not agreement. This
module estimates ``P(success | e(x), action=a)`` for the typed policy's chosen
action and the recorded incumbent action, and reports the value difference
(regret) with the probe's cross-validated, embedding-grouped estimator.

Method (and its limits — read before believing any number)
----------------------------------------------------------
* **Corpus** is the admissible frozen snapshot (``episodic.db.backup-20260415``
  by default; the live DB is refused by ``resolve_snapshot`` because the
  2026-09-17 leak purge changed it). Labels are ``outcome`` on
  ``action_type='routing'`` rows.
* **Join**: the TD-9 receipt rows carry ``position`` / ``state_sha256`` /
  ``incumbent`` / ``label`` but no ``embedding_idx``. The module reconstructs
  the deterministic sample (``load_rows`` frame -> ``sample_rows(n, seed)``)
  from the receipt's own config and verifies every row on all four fields. A
  single mismatch refuses the join: no guessed joins, ever. A receipt that
  carries ``embedding_idx`` per row is joined directly. When neither key is
  present the module says so and names the fix (``--live-rerun``). The
  reconstruction never needs a model: it is offline and exact.
* **Vectors**: the model-based estimator needs ``e(x)``. The frozen snapshot's
  DB is a direct file with no paired ``embeddings.faiss``, and the surviving
  store generations do not align with this corpus (their id maps have zero
  overlap with the snapshot's memory ids), so stored vectors are *not*
  recoverable offline. ``--vectors npz`` takes memory_id-keyed vectors;
  ``--vectors faiss`` uses a paired index and verifies ``id_map[idx] == id``
  (or refuses unverified positions); ``--vectors embed`` computes the canonical
  embedding text live with ``use_fallback=False`` and refuses degenerate or
  hash-fallback vectors. Without any of these the receipt is written with
  ``status='vectors-unavailable'`` and an explicit live-run requirement —
  never a fabricated estimate.
* **Estimator**: ``scripts/analysis/escalation_prediction_probe.py`` owns the
  statistics. This module adds nothing to them: it calls the probe's
  ``cross_fit_predict`` (same ``LogisticRegression``, same
  ``GroupShuffleSplit`` grouped by embedding identity, same failure-label
  sign). Actions that are single-class or below the probe's support floors
  (``MIN_ROWS_PER_ROLE`` / ``MIN_POSITIVES``) are flagged **unevaluable** and
  their rows are excluded from the value, never scored.
* **Caveat**: every numeric block carries ``COUNTERFACTUAL_CAVEAT``. The
  frozen label is the incumbent's realized outcome; for a disagreeing row it
  is not the chosen action's outcome, and the probe's own criticisms
  (EPD-1..3: label confounds, embedding reuse, suite/task_type identity) apply.

Receipt: JSON with ``metric_directions``, ``support``, ``policy_value``,
``estimator``, ``join``, ``vector_source`` and ``live_run_requirements``.

CLI::

    python -m src.typed_decisions.routing_counterfactual --dry-run \\
        --snapshot <frozen.db> --receipt <td9-receipt.json>
    python -m src.typed_decisions.routing_counterfactual \\
        --snapshot <frozen.db> --receipt <td9-receipt.json> \\
        --vectors npz --npz <memory-id-keyed.npz> --receipt-out <out.json>
    python -m src.typed_decisions.routing_counterfactual --live-rerun --live \\
        --snapshot <frozen.db> --server-url http://127.0.0.1:8199 --n 200 \\
        --vectors embed --embed-server http://127.0.0.1:8090 --receipt-out <out.json>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.typed_decisions.routing_replay import (
    DEFAULT_N as REPLAY_DEFAULT_N,
)
from src.typed_decisions.routing_replay import (
    DEFAULT_ROLE,
    DEFAULT_SERVER_URL,
    DEFAULT_STATE_BUDGET_CHARS,
    ReplayError,
    RoutingRow,
    Snapshot,
    prepare_state,
    resolve_snapshot,
    sample_rows,
)

__all__ = [
    "COUNTERFACTUAL_CAVEAT",
    "DEFAULT_CV_SEED",
    "DEFAULT_CV_SPLITS",
    "DEFAULT_CV_TEST_SIZE",
    "PROBE_CRITICISMS",
    "RECEIPT_KIND",
    "CounterfactualError",
    "EmbeddedVectorSource",
    "FaissVectorSource",
    "JoinedRow",
    "JoinUnavailable",
    "NpzVectorSource",
    "ProbeAdapter",
    "SnapshotCall",
    "SupportRecord",
    "UnavailableVectorSource",
    "VectorsUnavailable",
    "build_probe_adapter",
    "build_vector_source",
    "classify_support",
    "decision_breakdown",
    "embedding_text_for_context",
    "estimate_policy_value",
    "join_receipt_to_snapshot",
    "live_run_requirements",
    "load_probe_module",
    "load_snapshot_calls",
    "main",
    "run_counterfactual",
    "summarize_actions",
]

# ── constants ─────────────────────────────────────────────────────────────

RECEIPT_KIND = "td10-routing-counterfactual"
TD9_RECEIPT_KIND = "td7-routing-replay"

DEFAULT_CV_SPLITS = 5
DEFAULT_CV_TEST_SIZE = 0.25
DEFAULT_CV_SEED = 42

PROBE_RELPATH = Path("scripts/analysis/escalation_prediction_probe.py")

KNOWN_REEMBEDDED = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions/reembedded.npz"
)

SNAPSHOT_CALLS_SQL = """
SELECT id, embedding_idx, action, outcome, context
FROM memories
WHERE action_type='routing' AND outcome IN ('success','failure')
"""

COUNTERFACTUAL_CAVEAT = (
    "Model-based counterfactual estimate, NOT an observation. The frozen label "
    "describes the INCUMBENT action's outcome; for a disagreeing row it is not "
    "the chosen action's outcome. P(success | e(x), action=a) is an assumption "
    "of the probe's estimator fit on historical incumbent outcomes, and carries "
    "the probe's own confounds (EPD-1..3: label written once at INSERT, "
    "embedding/convention reuse, suite/task_type identity). Actions without two "
    "outcome classes in the frozen logs are unevaluable and are excluded, not "
    "scored."
)

PROBE_CRITICISMS = (
    "escalation_prediction_probe is REFUTED as a task-difficulty decoder: a "
    "13-feature length/format model and suite/task_type identity beat the "
    "1024-d embedding on 4/5 roles, and the headline lived in objectives seen "
    "exactly once (EPD-1..3, learned-routing-controller.md). This module "
    "reuses its estimator for counterfactual VALUE, not as evidence that the "
    "embedding decodes difficulty.",
    "The frozen outcome label is written once at INSERT and never updated "
    "(EPD-1), so it records the incumbent policy's realized result, not a "
    "counterfactual outcome.",
    "Cross-fit groups are the embedding vectors themselves (probe._vector_key), "
    "so vector reuse cannot leak across the train/test boundary.",
    "Support floors are the probe's (MIN_ROWS_PER_ROLE / MIN_POSITIVES); "
    "single-outcome-class and rare actions are flagged unevaluable.",
    "If --vectors embed ran, e(x) is a fresh canonical re-embedding "
    "(embedding_text_for), not the decision-time vector; that removes the "
    "EPD-3 writer-path leak but is not byte-identical to the stored vector.",
)

_METRIC_DIRECTIONS: dict[str, str] = {
    "policy_value.typed.success_mean": "higher_is_better",
    "policy_value.incumbent.success_mean": "higher_is_better",
    "policy_value.delta_typed_minus_incumbent.mean": "higher_is_better",
    "policy_value.regret_incumbent_minus_typed.mean": "lower_is_better",
    "estimator.coverage": "higher_is_better",
    "join.n_joined": "higher_is_better",
}

ENV_SERVER = "TD_ROUTING_REPLAY_SERVER"
ENV_ROLE = "TD_ROUTING_REPLAY_ROLE"


class CounterfactualError(RuntimeError):
    """The counterfactual evaluation cannot produce an admissible measurement."""


class JoinUnavailable(CounterfactualError):
    """The receipt cannot be joined to the snapshot without guessing."""


class VectorsUnavailable(CounterfactualError):
    """No admissible e(x) source covers the joined rows."""


# ── snapshot calls and join ───────────────────────────────────────────────


@dataclass(frozen=True)
class SnapshotCall:
    """One frozen routing row with the identity the join needs."""

    memory_id: str
    embedding_idx: int | None
    action: str
    outcome: str
    context: str

    @property
    def label(self) -> bool:
        return self.outcome == "success"

    def routing_row(self) -> RoutingRow:
        return RoutingRow(
            embedding_idx=self.embedding_idx,
            action=self.action,
            outcome=self.outcome,
            context=self.context,
        )


def load_snapshot_calls(db_path: str | Path) -> list[SnapshotCall]:
    """Read the frozen routing corpus (with memory id), read-only."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        raw = con.execute(SNAPSHOT_CALLS_SQL).fetchall()
    except sqlite3.Error as exc:
        raise CounterfactualError(f"cannot read routing rows from {db_path}: {exc}") from exc
    finally:
        con.close()
    return [
        SnapshotCall(
            memory_id=str(row[0] or ""),
            embedding_idx=row[1] if row[1] is None else int(row[1]),
            action=str(row[2] or ""),
            outcome=str(row[3] or ""),
            context=str(row[4] or ""),
        )
        for row in raw
    ]


@dataclass(frozen=True)
class JoinedRow:
    """A TD-9 decision joined to its frozen snapshot row and label."""

    position: int
    memory_id: str
    embedding_idx: int | None
    context: str
    incumbent: str
    label: bool
    chosen: str | None
    chosen_code: str | None
    confidence: float | None
    state_sha256: str
    join_mode: str

    def to_record(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "memory_id": self.memory_id,
            "embedding_idx": self.embedding_idx,
            "incumbent": self.incumbent,
            "label": self.label,
            "chosen": self.chosen,
            "chosen_code": self.chosen_code,
            "confidence": self.confidence,
            "state_sha256": self.state_sha256,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _receipt_rows(receipt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = receipt.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise JoinUnavailable(
            "receipt carries no non-empty 'rows' list; nothing to join. Re-run "
            "routing_replay with --live-rerun to obtain a usable receipt."
        )
    return list(rows)


def _verify_row(
    *,
    position: int,
    receipt_row: Mapping[str, Any],
    call: SnapshotCall,
    expected_state_sha256: str | None,
) -> list[str]:
    """Return the list of failed verification fields (empty == verified)."""
    failures: list[str] = []
    if int(receipt_row.get("position", position)) != position:
        failures.append("position")
    if (
        receipt_row.get("state_sha256")
        and expected_state_sha256 is not None
        and receipt_row.get("state_sha256") != expected_state_sha256
    ):
        failures.append("state_sha256")
    if receipt_row.get("incumbent") != call.action:
        failures.append("incumbent")
    if bool(receipt_row.get("label")) != call.label:
        failures.append("label")
    return failures


def join_receipt_to_snapshot(
    receipt: Mapping[str, Any],
    calls: Sequence[SnapshotCall],
) -> tuple[list[JoinedRow], dict[str, Any]]:
    """Join a TD-9 receipt to the frozen snapshot rows, or refuse.

    Two admissible modes, both verified field-by-field against the frozen
    corpus (never a fuzzy or guessed match):

    * ``receipt_embedding_idx`` — every receipt row carries ``embedding_idx``.
    * ``reconstructed_sample_state_sha256`` — the receipt's own ``config``
      (n/seed/state_budget_chars) reconstructs the deterministic sample via
      ``sample_rows`` and each row is verified on
      position + state_sha256 + incumbent + label.

    Raises:
        JoinUnavailable: no usable join key, or any row fails verification.
    """
    rows = _receipt_rows(receipt)
    config = receipt.get("config") if isinstance(receipt.get("config"), Mapping) else {}
    receipt_sha = (
        receipt.get("snapshot", {}).get("db_sha256")
        if isinstance(receipt.get("snapshot"), Mapping)
        else None
    )

    with_embedding_idx = all(row.get("embedding_idx") is not None for row in rows)
    with_state = all(row.get("state_sha256") for row in rows)

    if with_embedding_idx:
        by_idx = {call.embedding_idx: call for call in calls if call.embedding_idx is not None}
        joined: list[JoinedRow] = []
        mismatches: list[dict[str, Any]] = []
        for position, row in enumerate(rows):
            call = by_idx.get(int(row["embedding_idx"]))
            if call is None:
                mismatches.append({"position": position, "reason": "embedding_idx_not_in_snapshot"})
                continue
            expected = _sha256_text(
                prepare_state(
                    call.context,
                    int(config.get("state_budget_chars") or DEFAULT_STATE_BUDGET_CHARS),
                )[0]
            )
            failures = _verify_row(
                position=position, receipt_row=row, call=call, expected_state_sha256=expected
            )
            if failures:
                mismatches.append({"position": position, "reason": ",".join(failures)})
                continue
            joined.append(_joined_row(row, call, position, "receipt_embedding_idx"))
        if mismatches:
            raise JoinUnavailable(
                f"receipt embedding_idx join failed verification for {len(mismatches)}/{len(rows)} "
                f"rows (first: {mismatches[0]}); refusing a guessed join"
            )
        report = _join_report("receipt_embedding_idx", rows, joined, config, receipt_sha)
        return joined, report

    if with_state:
        n = config.get("n")
        seed = config.get("seed")
        budget = int(config.get("state_budget_chars") or DEFAULT_STATE_BUDGET_CHARS)
        if n is None or seed is None:
            raise JoinUnavailable(
                "receipt has state_sha256 but its config lacks n/seed, so the deterministic "
                "sample cannot be reconstructed; re-run routing_replay with --live-rerun"
            )
        frame = [call for call in calls if call.action.strip()]
        routing_rows = [call.routing_row() for call in frame]
        sampled = sample_rows(routing_rows, int(n), int(seed))
        by_object = {id(routing_row): call for routing_row, call in zip(routing_rows, frame)}
        if len(sampled) != len(rows):
            raise JoinUnavailable(
                f"receipt has {len(rows)} rows but the reconstructed sample has {len(sampled)}; "
                "the receipt's config does not describe this snapshot"
            )
        joined = []
        mismatches = []
        for position, (row, routing_row) in enumerate(zip(rows, sampled)):
            call = by_object[id(routing_row)]
            expected = _sha256_text(prepare_state(call.context, budget)[0])
            failures = _verify_row(
                position=position, receipt_row=row, call=call, expected_state_sha256=expected
            )
            if failures:
                mismatches.append({"position": position, "reason": ",".join(failures)})
                continue
            joined.append(_joined_row(row, call, position, "reconstructed_sample_state_sha256"))
        if mismatches:
            raise JoinUnavailable(
                f"receipt verification failed for {len(mismatches)}/{len(rows)} rows "
                f"(first: {mismatches[0]}); refusing a guessed join"
            )
        report = _join_report(
            "reconstructed_sample_state_sha256", rows, joined, config, receipt_sha
        )
        return joined, report

    raise JoinUnavailable(
        "receipt rows carry neither embedding_idx nor a state_sha256/position identity, and "
        "the config cannot reconstruct the sample; refusing to guess a join. Re-run "
        "routing_replay (CLI: --live-rerun) to obtain a receipt with per-row identity."
    )


def _joined_row(
    receipt_row: Mapping[str, Any], call: SnapshotCall, position: int, mode: str
) -> JoinedRow:
    chosen = receipt_row.get("action")
    confidence = receipt_row.get("confidence")
    return JoinedRow(
        position=position,
        memory_id=call.memory_id,
        embedding_idx=call.embedding_idx,
        context=call.context,
        incumbent=call.action,
        label=call.label,
        chosen=str(chosen) if chosen is not None else None,
        chosen_code=(
            str(receipt_row["chosen_code"]) if receipt_row.get("chosen_code") is not None else None
        ),
        confidence=float(confidence) if confidence is not None else None,
        state_sha256=str(receipt_row.get("state_sha256") or ""),
        join_mode=mode,
    )


def _join_report(
    mode: str,
    receipt_rows: Sequence[Mapping[str, Any]],
    joined: Sequence[JoinedRow],
    config: Mapping[str, Any],
    receipt_sha: str | None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "n_receipt_rows": len(receipt_rows),
        "n_joined": len(joined),
        "verified_fields": ["position", "state_sha256", "incumbent", "label"],
        "mismatches": 0,
        "receipt_config": {
            "n": config.get("n"),
            "seed": config.get("seed"),
            "state_budget_chars": config.get("state_budget_chars"),
        },
        "receipt_snapshot_db_sha256": receipt_sha,
        "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
    }


# ── vector sources ────────────────────────────────────────────────────────


class VectorSource(Protocol):
    """Resolve ``e(x)`` for snapshot rows keyed by memory id."""

    name: str

    def resolve(
        self, calls: Sequence[SnapshotCall]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...


@dataclass
class DictVectorSource:
    """In-memory memory_id -> vector source (tests and ``--vectors embed``)."""

    vectors: Mapping[str, np.ndarray]
    name: str = "dict"
    detail: Mapping[str, Any] | None = None

    def resolve(
        self, calls: Sequence[SnapshotCall]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        resolved = {
            call.memory_id: np.asarray(self.vectors[call.memory_id], dtype=np.float32).reshape(-1)
            for call in calls
            if call.memory_id in self.vectors
        }
        report = {
            "name": self.name,
            "n_requested": len(calls),
            "n_resolved": len(resolved),
            "coverage": (len(resolved) / len(calls)) if calls else None,
            "detail": dict(self.detail or {}),
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        }
        return resolved, report


@dataclass
class NpzVectorSource:
    """Vectors from an ``ids``/``embeddings`` npz (memory-id keyed)."""

    path: Path
    id_field: str = "ids"
    vector_field: str = "embeddings"
    name: str = "npz"

    def resolve(
        self, calls: Sequence[SnapshotCall]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if not self.path.exists():
            raise VectorsUnavailable(f"npz vector source does not exist: {self.path}")
        data = np.load(self.path, allow_pickle=True)
        if self.id_field not in data or self.vector_field not in data:
            raise VectorsUnavailable(
                f"{self.path} must carry '{self.id_field}' and '{self.vector_field}' arrays; "
                f"found {sorted(data.files)}"
            )
        ids = data[self.id_field].tolist()
        embeddings = data[self.vector_field]
        if len(ids) != len(embeddings):
            raise VectorsUnavailable(
                f"{self.path} has {len(ids)} ids but {len(embeddings)} vectors"
            )
        by_id = {
            str(memory_id): np.asarray(embeddings[index]).reshape(-1).astype(np.float32)
            for index, memory_id in enumerate(ids)
        }
        resolved = {
            call.memory_id: by_id[call.memory_id] for call in calls if call.memory_id in by_id
        }
        if not resolved:
            raise VectorsUnavailable(
                f"{self.path} covers 0/{len(calls)} snapshot memory ids; the vectors do not "
                "belong to this corpus generation"
            )
        report = {
            "name": self.name,
            "path": str(self.path),
            "n_stored": len(ids),
            "n_requested": len(calls),
            "n_resolved": len(resolved),
            "coverage": len(resolved) / len(calls) if calls else None,
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        }
        return resolved, report


@dataclass
class FaissVectorSource:
    """Vectors from a FAISS index, keyed by ``embedding_idx``.

    A positional index is only admissible with an ``id_map`` that maps the
    same position back to the snapshot's memory id (``id_map[idx] == id``); the
    zero-alignment generation mismatch this corpus exhibits is refused rather
    than silently scored. ``allow_positional`` is the explicit override.
    """

    faiss_path: Path
    id_map_path: Path | None = None
    allow_positional: bool = False
    name: str = "faiss"

    def resolve(
        self, calls: Sequence[SnapshotCall]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if not self.faiss_path.exists():
            raise VectorsUnavailable(f"faiss vector source does not exist: {self.faiss_path}")
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - faiss is a project dependency
            raise CounterfactualError("faiss-cpu is not importable") from exc
        index = faiss.read_index(str(self.faiss_path))
        id_map = None
        if self.id_map_path is not None:
            if not self.id_map_path.exists():
                raise VectorsUnavailable(f"id_map does not exist: {self.id_map_path}")
            id_map = np.load(self.id_map_path, allow_pickle=True).tolist()

        resolved: dict[str, np.ndarray] = {}
        out_of_range = 0
        aligned = 0
        mismatched = 0
        unverified = 0
        for call in calls:
            if call.embedding_idx is None:
                continue
            if not 0 <= call.embedding_idx < index.ntotal:
                out_of_range += 1
                continue
            if id_map is not None:
                if str(id_map[call.embedding_idx]) == call.memory_id:
                    aligned += 1
                    resolved[call.memory_id] = index.reconstruct(int(call.embedding_idx)).astype(
                        np.float32
                    )
                else:
                    mismatched += 1
            elif self.allow_positional:
                unverified += 1
                resolved[call.memory_id] = index.reconstruct(int(call.embedding_idx)).astype(
                    np.float32
                )
            else:
                unverified += 1

        if id_map is not None and aligned == 0 and mismatched > 0:
            raise VectorsUnavailable(
                f"id_map alignment is 0/{mismatched} for {self.faiss_path}: the index does not "
                "belong to this snapshot's generation; refusing positional vectors"
            )
        if id_map is None and not self.allow_positional and unverified:
            raise VectorsUnavailable(
                f"{self.faiss_path} has no id_map, so embedding_idx positions cannot be verified "
                "against the snapshot's memory ids; pass --allow-positional-faiss only if the "
                "index is known to pair with this corpus"
            )
        if not resolved:
            raise VectorsUnavailable(
                f"{self.faiss_path} resolves 0/{len(calls)} snapshot ids "
                f"(out_of_range={out_of_range}, aligned={aligned}, mismatched={mismatched}, "
                f"unverified={unverified})"
            )
        report = {
            "name": self.name,
            "faiss_path": str(self.faiss_path),
            "faiss_ntotal": int(index.ntotal),
            "id_map_path": str(self.id_map_path) if self.id_map_path else None,
            "n_requested": len(calls),
            "n_resolved": len(resolved),
            "id_map_aligned": aligned,
            "id_map_mismatched": mismatched,
            "out_of_range": out_of_range,
            "unverified_positions": unverified,
            "coverage": len(resolved) / len(calls) if calls else None,
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        }
        return resolved, report


@dataclass
class EmbeddedVectorSource:
    """Live canonical re-embedding of the snapshot contexts (``--vectors embed``).

    Uses ``orchestration.repl_memory``'s ``TaskEmbedder`` over the canonical
    ``embedding_text_for`` text, with ``use_fallback=False``: a hash
    pseudo-embedding is refused, never silently scored. Because identical
    texts map to identical vectors, only distinct texts are embedded.
    """

    server_url: str = "http://127.0.0.1:8090"
    use_parallel: bool = False
    embed_out: Path | None = None
    name: str = "embedded"

    def resolve(
        self, calls: Sequence[SnapshotCall]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        from orchestration.repl_memory.embedder import (
            EmbeddingConfig,
            TaskEmbedder,
            is_degenerate_embedding,
            is_hash_fallback_embedding,
        )

        id_to_text: dict[str, str] = {}
        unique_texts: dict[str, np.ndarray | None] = {}
        for call in calls:
            text = embedding_text_for_context(call.context)
            id_to_text[call.memory_id] = text
            unique_texts.setdefault(text, None)

        embedder = TaskEmbedder(
            EmbeddingConfig(
                use_server=True,
                use_parallel=self.use_parallel,
                use_fallback=False,
                server_url=self.server_url,
            )
        )
        started = time.perf_counter()
        for text in unique_texts:
            try:
                vector = embedder.embed_text(text)
            except Exception as exc:  # noqa: BLE001 - any failure is no embedding
                raise VectorsUnavailable(
                    f"live embedding failed at {self.server_url} (fallback disabled): {exc}"
                ) from exc
            reason = is_degenerate_embedding(vector)
            if reason:
                raise VectorsUnavailable(
                    f"refusing degenerate live embedding ({reason}); the embedder is not "
                    "producing usable e(x)"
                )
            if is_hash_fallback_embedding(text, vector):
                raise VectorsUnavailable(
                    "refusing hash-fallback pseudo-embedding for the live e(x) source"
                )
            unique_texts[text] = np.asarray(vector, dtype=np.float32).reshape(-1)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        resolved = {call.memory_id: unique_texts[id_to_text[call.memory_id]] for call in calls}
        if self.embed_out is not None:
            ids = list(resolved)
            np.savez(
                self.embed_out,
                ids=np.array(ids, dtype=object),
                embeddings=np.vstack([resolved[memory_id] for memory_id in ids]),
                texts=np.array([id_to_text[memory_id] for memory_id in ids], dtype=object),
            )
        texts, embeddings = [], []
        for text, vector in unique_texts.items():
            texts.append(text)
            embeddings.append(vector)
        report = {
            "name": self.name,
            "server_url": self.server_url,
            "use_parallel": self.use_parallel,
            "n_requested": len(calls),
            "n_resolved": len(resolved),
            "n_unique_texts": len(unique_texts),
            "embedding_dim": int(len(embeddings[0])) if embeddings else None,
            "embed_out": str(self.embed_out) if self.embed_out else None,
            "wall_ms": elapsed_ms,
            "convention": "embedding_text_for (canonical task convention)",
            "coverage": len(resolved) / len(calls) if calls else None,
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        }
        return resolved, report


@dataclass
class UnavailableVectorSource:
    """No vector source: the receipt records why and what must run live."""

    reason: str
    name: str = "unavailable"

    def resolve(
        self, calls: Sequence[SnapshotCall]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        raise VectorsUnavailable(self.reason)


def embedding_text_for_context(context: str) -> str:
    """Canonical embedding text for a stored context JSON (no inference)."""
    from orchestration.repl_memory.memory_record import record_from_legacy_context

    try:
        payload = json.loads(context) if context else {}
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return record_from_legacy_context(payload).embedding_text()


# ── probe adapter (the estimator) ─────────────────────────────────────────


def load_probe_module(probe_path: str | Path | None = None) -> Any:
    """Import ``escalation_prediction_probe`` by path so its estimator is reused."""
    path = Path(probe_path) if probe_path else Path(__file__).resolve().parents[2] / PROBE_RELPATH
    if not path.exists():
        raise CounterfactualError(
            f"estimator probe not found at {path}; the probe owns the statistics and is "
            "required (--probe-path may point at it)"
        )
    spec = importlib.util.spec_from_file_location("escalation_prediction_probe", path)
    if spec is None or spec.loader is None:
        raise CounterfactualError(f"cannot load the estimator probe at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "cross_fit_predict"):
        raise CounterfactualError(
            f"{path} does not expose cross_fit_predict; the probe must provide the estimator"
        )
    return module


class ProbeAdapter:
    """Thin seam over the probe module so tests can inject a tiny fake model."""

    def __init__(self, module: Any):
        self.module = module

    def group_key(self, vector: np.ndarray) -> str:
        return str(self.module._vector_key(np.asarray(vector, dtype=np.float32)))

    def support_floors(self) -> tuple[int, int]:
        return int(self.module.MIN_ROWS_PER_ROLE), int(self.module.MIN_POSITIVES)

    def cross_fit_predict(
        self,
        *,
        X_fit: np.ndarray,
        y_failure_fit: np.ndarray,
        groups_fit: np.ndarray,
        X_eval: np.ndarray,
        groups_eval: np.ndarray,
        seed: int,
        n_splits: int,
        test_size: float,
    ) -> dict[str, Any]:
        return self.module.cross_fit_predict(
            X_fit=X_fit,
            y_failure_fit=y_failure_fit,
            groups_fit=groups_fit,
            X_eval=X_eval,
            groups_eval=groups_eval,
            seed=seed,
            n_splits=n_splits,
            test_size=test_size,
        )


def build_probe_adapter(probe_path: str | Path | None = None) -> ProbeAdapter:
    return ProbeAdapter(load_probe_module(probe_path))


# ── support classification and value estimation ───────────────────────────


@dataclass(frozen=True)
class SupportRecord:
    action: str
    n_rows: int
    n_success: int
    n_failure: int
    evaluable: bool
    reason: str | None = None

    @property
    def base_rate(self) -> float | None:
        return (self.n_success / self.n_rows) if self.n_rows else None

    def to_record(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_success": self.n_success,
            "n_failure": self.n_failure,
            "base_rate": self.base_rate,
            "evaluable": self.evaluable,
            "reason": self.reason,
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        }


def summarize_actions(calls: Sequence[SnapshotCall]) -> dict[str, dict[str, Any]]:
    """Raw per-action counts over the frozen corpus (no vectors, no model)."""
    grouped: dict[str, list[SnapshotCall]] = {}
    for call in calls:
        grouped.setdefault(call.action, []).append(call)
    summary: dict[str, dict[str, Any]] = {}
    for action, members in sorted(grouped.items()):
        successes = sum(1 for call in members if call.label)
        summary[action] = {
            "n_rows": len(members),
            "n_success": successes,
            "n_failure": len(members) - successes,
            "base_rate": successes / len(members) if members else None,
        }
    return summary


def classify_support(
    calls: Sequence[SnapshotCall],
    vectors: Mapping[str, np.ndarray],
    *,
    min_rows: int,
    min_positives: int,
) -> dict[str, SupportRecord]:
    """Apply the probe's per-action support floors to rows with a vector."""
    grouped: dict[str, list[SnapshotCall]] = {}
    for call in calls:
        grouped.setdefault(call.action, []).append(call)
    support: dict[str, SupportRecord] = {}
    for action, members in sorted(grouped.items()):
        usable = [call for call in members if call.memory_id in vectors]
        successes = sum(1 for call in usable if call.label)
        failures = len(usable) - successes
        if len(usable) < min_rows:
            evaluable, reason = False, "insufficient_rows"
        elif failures == 0 or successes == 0:
            evaluable, reason = False, "single_outcome_class"
        elif successes < min_positives or failures < min_positives:
            evaluable, reason = False, "insufficient_outcome_class"
        else:
            evaluable, reason = True, None
        support[action] = SupportRecord(
            action=action,
            n_rows=len(usable),
            n_success=successes,
            n_failure=failures,
            evaluable=evaluable,
            reason=reason,
        )
    return support


def _expected_evaluability(record: Mapping[str, Any], *, min_rows: int, min_positives: int) -> str:
    """Evaluability from raw counts, assuming every row has a vector."""
    if record["n_rows"] < min_rows:
        return "insufficient_rows"
    if record["n_failure"] == 0 or record["n_success"] == 0:
        return "single_outcome_class"
    if record["n_success"] < min_positives or record["n_failure"] < min_positives:
        return "insufficient_outcome_class"
    return "evaluable"


def decision_breakdown(
    joined: Sequence[JoinedRow],
    calls: Sequence[SnapshotCall],
    *,
    min_rows: int,
    min_positives: int,
) -> dict[str, Any]:
    """Offline picture of which decisions the value estimate can even touch.

    Computed from raw snapshot counts (no vectors), i.e. it assumes full e(x)
    coverage, which a fresh canonical re-embedding provides. It exists so the
    operator sees how thin the disagreement support is BEFORE spending an
    embedding run: agreeing rows contribute exactly zero to the delta, because
    both sides of the difference use the same action model.
    """
    raw = summarize_actions(calls)
    verdicts = {
        action: _expected_evaluability(record, min_rows=min_rows, min_positives=min_positives)
        for action, record in raw.items()
    }
    unevaluable = {action for action, verdict in verdicts.items() if verdict != "evaluable"}
    disagreements: Counter[str] = Counter()
    for row in joined:
        if row.chosen is not None and row.chosen != row.incumbent:
            disagreements[f"{row.incumbent}->{row.chosen}"] += 1
    return {
        "n_rows": len(joined),
        "n_agreement": sum(1 for row in joined if row.chosen == row.incumbent),
        "n_disagreement": sum(
            1 for row in joined if row.chosen is not None and row.chosen != row.incumbent
        ),
        "n_unresolved": sum(1 for row in joined if row.chosen is None),
        "disagreements_by_incumbent_to_chosen": dict(sorted(disagreements.items())),
        "n_rows_touching_expected_unevaluable_action": sum(
            1 for row in joined if row.incumbent in unevaluable or (row.chosen or "") in unevaluable
        ),
        "expected_unevaluable_actions": sorted(unevaluable),
        "expected_evaluability_by_action": verdicts,
        "assumption": (
            "computed from raw counts without vectors; assumes e(x) covers every row "
            "(true for a fresh canonical re-embedding)"
        ),
        "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
    }


def _support_floors(
    adapter: Any, *, min_rows: int | None, min_positives: int | None
) -> tuple[int, int]:
    """Resolve support floors from explicit overrides or the probe's constants."""
    if min_rows is not None and min_positives is not None:
        return int(min_rows), int(min_positives)
    if adapter is None:
        adapter = build_probe_adapter()
    default_rows, default_positives = adapter.support_floors()
    return (
        int(min_rows) if min_rows is not None else int(default_rows),
        int(min_positives) if min_positives is not None else int(default_positives),
    )


def _summarize_folds(values: Sequence[float]) -> dict[str, Any]:
    """Probe-shaped fold summary (mean/std/min/max/folds) plus a 95% fold CI."""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "folds": 0, "ci95": None}
    mean = float(np.mean(values))
    std = float(np.std(values))
    folds = len(values)
    ci95 = (
        [mean - 1.96 * std / math.sqrt(folds), mean + 1.96 * std / math.sqrt(folds)]
        if folds >= 2
        else None
    )
    return {
        "mean": mean,
        "std": std,
        "min": float(min(values)),
        "max": float(max(values)),
        "folds": folds,
        "ci95": ci95,
        "ci95_method": "normal_approx_over_cross_fit_folds (split variance only)",
    }


def estimate_policy_value(
    joined: Sequence[JoinedRow],
    calls: Sequence[SnapshotCall],
    vectors: Mapping[str, np.ndarray],
    adapter: Any,
    *,
    seed: int = DEFAULT_CV_SEED,
    n_splits: int = DEFAULT_CV_SPLITS,
    test_size: float = DEFAULT_CV_TEST_SIZE,
    min_rows: int | None = None,
    min_positives: int | None = None,
) -> dict[str, Any]:
    """Cross-fitted value of the typed and incumbent actions, and the regret.

    Per-action conditional success models are the probe's estimator
    (``adapter.cross_fit_predict``), fit on every frozen row of that action
    with a vector. Evaluation rows need BOTH their chosen and incumbent action
    evaluable and vector-covered; anything else is excluded and counted, never
    scored with a fallback. The frozen label remains the incumbent's outcome.
    """
    floors = _support_floors(adapter, min_rows=min_rows, min_positives=min_positives)
    min_rows, min_positives = floors

    support = classify_support(
        calls, vectors, min_rows=int(min_rows), min_positives=int(min_positives)
    )
    evaluable = {action for action, record in support.items() if record.evaluable}
    excluded_by_reason: Counter[str] = Counter()
    excluded_rows: list[dict[str, Any]] = []
    eval_rows: list[JoinedRow] = []
    for row in joined:
        reasons: list[str] = []
        if row.chosen is None:
            reasons.append("unresolved_chosen")
        if row.memory_id not in vectors:
            reasons.append("missing_vector")
        elif row.chosen is not None and row.chosen not in evaluable:
            reasons.append("chosen_action_unevaluable")
        if row.incumbent not in evaluable:
            reasons.append("incumbent_action_unevaluable")
        if reasons:
            excluded_by_reason.update(reasons)
            excluded_rows.append({"position": row.position, "reasons": reasons})
        else:
            eval_rows.append(row)

    if not eval_rows:
        return {
            "status": "no_evaluable_rows",
            "support": {action: record.to_record() for action, record in sorted(support.items())},
            "excluded": {
                "n_input_rows": len(joined),
                "n_evaluable": 0,
                "by_reason": dict(sorted(excluded_by_reason.items())),
                "rows": excluded_rows,
            },
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        }

    eval_X = np.vstack([vectors[row.memory_id] for row in eval_rows]).astype(np.float32)
    eval_groups = np.array([adapter.group_key(vector) for vector in eval_X], dtype=object)

    model_runs: dict[str, dict[str, Any]] = {}
    for action in sorted(evaluable):
        members = [call for call in calls if call.action == action and call.memory_id in vectors]
        X_fit = np.vstack([vectors[call.memory_id] for call in members]).astype(np.float32)
        y_failure = np.array([0 if call.label else 1 for call in members], dtype=np.int64)
        groups_fit = np.array([adapter.group_key(vector) for vector in X_fit], dtype=object)
        model_runs[action] = adapter.cross_fit_predict(
            X_fit=X_fit,
            y_failure_fit=y_failure,
            groups_fit=groups_fit,
            X_eval=eval_X,
            groups_eval=eval_groups,
            seed=seed,
            n_splits=n_splits,
            test_size=test_size,
        )

    p_typed: list[float | None] = []
    p_incumbent: list[float | None] = []
    for index, row in enumerate(eval_rows):
        typed_run, incumbent_run = model_runs[row.chosen], model_runs[row.incumbent]
        typed_values, incumbent_values = [], []
        for split_index in range(min(len(typed_run["per_split"]), len(incumbent_run["per_split"]))):
            typed_split = typed_run["per_split"][split_index]
            incumbent_split = incumbent_run["per_split"][split_index]
            if typed_split["oof"][index] and incumbent_split["oof"][index]:
                typed_values.append(float(typed_split["p_success"][index]))
                incumbent_values.append(float(incumbent_split["p_success"][index]))
        p_typed.append(float(np.mean(typed_values)) if typed_values else None)
        p_incumbent.append(float(np.mean(incumbent_values)) if incumbent_values else None)

    covered = [
        index
        for index in range(len(eval_rows))
        if p_typed[index] is not None and p_incumbent[index] is not None
    ]
    coverage = len(covered) / len(eval_rows)

    typed_fold, incumbent_fold, delta_fold = [], [], []
    max_folds = max((run["folds"] for run in model_runs.values()), default=0)
    for split_index in range(max_folds):
        typed_split_values, incumbent_split_values = [], []
        for index, row in enumerate(eval_rows):
            typed_run, incumbent_run = model_runs[row.chosen], model_runs[row.incumbent]
            if split_index >= len(typed_run["per_split"]) or split_index >= len(
                incumbent_run["per_split"]
            ):
                continue
            typed_split = typed_run["per_split"][split_index]
            incumbent_split = incumbent_run["per_split"][split_index]
            if typed_split["oof"][index] and incumbent_split["oof"][index]:
                typed_split_values.append(float(typed_split["p_success"][index]))
                incumbent_split_values.append(float(incumbent_split["p_success"][index]))
        if typed_split_values:
            typed_fold.append(float(np.mean(typed_split_values)))
            incumbent_fold.append(float(np.mean(incumbent_split_values)))
            delta_fold.append(typed_fold[-1] - incumbent_fold[-1])

    typed_point = float(np.mean([p_typed[index] for index in covered]))
    incumbent_point = float(np.mean([p_incumbent[index] for index in covered]))
    delta_point = typed_point - incumbent_point

    per_action: dict[str, dict[str, Any]] = {}
    unevaluable: dict[str, dict[str, Any]] = {}
    for action, record in sorted(support.items()):
        if not record.evaluable:
            unevaluable[action] = record.to_record()
            continue
        as_chosen = [
            index
            for index, row in enumerate(eval_rows)
            if row.chosen == action and p_typed[index] is not None
        ]
        as_incumbent = [
            index
            for index, row in enumerate(eval_rows)
            if row.incumbent == action and p_incumbent[index] is not None
        ]
        per_action[action] = {
            "support": record.to_record(),
            "n_eval_as_chosen": sum(1 for row in eval_rows if row.chosen == action),
            "n_eval_as_chosen_scored": len(as_chosen),
            "mean_p_success_as_chosen": (
                float(np.mean([p_typed[index] for index in as_chosen])) if as_chosen else None
            ),
            "n_eval_as_incumbent": sum(1 for row in eval_rows if row.incumbent == action),
            "n_eval_as_incumbent_scored": len(as_incumbent),
            "mean_p_success_as_incumbent": (
                float(np.mean([p_incumbent[index] for index in as_incumbent]))
                if as_incumbent
                else None
            ),
        }

    agreements = sum(1 for row in eval_rows if row.chosen == row.incumbent)
    return {
        "status": "estimated",
        "n_input_rows": len(joined),
        "n_evaluable_rows": len(eval_rows),
        "n_covered_rows": len(covered),
        "n_agreement_rows": agreements,
        "n_disagreement_rows": len(eval_rows) - agreements,
        "base_rate_labels_evaluable": float(np.mean([float(row.label) for row in eval_rows])),
        "typed": {
            "success_mean": typed_point,
            "fold_summary": _summarize_folds(typed_fold),
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        },
        "incumbent": {
            "success_mean": incumbent_point,
            "fold_summary": _summarize_folds(incumbent_fold),
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        },
        "delta_typed_minus_incumbent": {
            "mean": delta_point,
            "fold_summary": _summarize_folds(delta_fold),
            "definition": "typed_value - incumbent_value (positive: typed is better)",
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        },
        "regret_incumbent_minus_typed": {
            "mean": -delta_point,
            "fold_summary": _summarize_folds([-value for value in delta_fold]),
            "definition": "incumbent_value - typed_value (positive: typed is worse)",
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        },
        "per_action": per_action,
        "unevaluable_actions": unevaluable,
        "excluded": {
            "n_input_rows": len(joined),
            "n_evaluable": len(eval_rows),
            "by_reason": dict(sorted(excluded_by_reason.items())),
            "rows": excluded_rows,
        },
        "estimator": {
            "name": "escalation_prediction_probe.cross_fit_predict",
            "classifier": "LogisticRegression(max_iter=2000, C=1.0, solver=lbfgs)",
            "splitter": (
                f"GroupShuffleSplit(n_splits={int(n_splits)}, "
                f"test_size={float(test_size)}, random_state=seed)"
            ),
            "group_key": "vector content sha1 (probe._vector_key)",
            "label": "success = 1 - P(failure); failure = outcome=='failure'",
            "seed": int(seed),
            "n_splits": int(n_splits),
            "test_size": float(test_size),
            "folds_per_action": {
                action: run["folds"] for action, run in sorted(model_runs.items())
            },
            "coverage": coverage,
            "per_action_oof_coverage": {
                action: run["coverage"] for action, run in sorted(model_runs.items())
            },
            "support_floor_rows": int(min_rows),
            "support_floor_positives": int(min_positives),
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        },
        "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
    }


# ── receipt assembly ──────────────────────────────────────────────────────


def live_run_requirements(
    *,
    snapshot: Snapshot,
    join_status: str,
    vector_status: str,
) -> list[dict[str, Any]]:
    """The precise commands/conditions that would close what is still missing."""
    requirements: list[dict[str, Any]] = []
    if join_status != "ok":
        requirements.append(
            {
                "item": "decisions",
                "status": "needed",
                "why": "the receipt could not be joined to the snapshot without guessing",
                "command": (
                    "python -m src.typed_decisions.routing_replay --live "
                    f"--snapshot {snapshot.path} --n 200 --role frontdoor "
                    "--server-url http://127.0.0.1:8199 --artifacts-dir <artifacts>"
                ),
            }
        )
    else:
        requirements.append(
            {
                "item": "decisions",
                "status": "present",
                "why": "receipt rows verified offline via reconstructed sample + state_sha256",
                "command": None,
            }
        )
    if vector_status == "ok":
        requirements.append(
            {
                "item": "embeddings",
                "status": "present",
                "why": "e(x) resolved from the supplied vector source",
                "command": None,
            }
        )
    elif vector_status == "unknown":
        requirements.append(
            {
                "item": "embeddings",
                "status": "unknown",
                "why": "not checked because the join failed; re-run decisions first",
                "command": None,
            }
        )
    else:
        requirements.append(
            {
                "item": "embeddings",
                "status": "needed",
                "why": (
                    "the frozen snapshot has no paired embeddings.faiss and no stored "
                    "vector generation aligns with this corpus, so e(x) must be produced live"
                ),
                "command": (
                    "python -m src.typed_decisions.routing_counterfactual "
                    f"--snapshot {snapshot.path} --receipt <td9-receipt.json> "
                    "--vectors embed --embed-server http://127.0.0.1:8090 "
                    "--receipt-out <artifacts>/routing-counterfactual-n200-20260918.json"
                ),
                "alternative": (
                    "produce a memory_id-keyed npz (ids/embeddings) and pass "
                    "--vectors npz --npz <path>"
                ),
            }
        )
    return requirements


def run_counterfactual(
    *,
    snapshot: Snapshot,
    calls: Sequence[SnapshotCall],
    receipt: Mapping[str, Any],
    vector_source: VectorSource,
    adapter: Any = None,
    receipt_sha256: str | None = None,
    cv_seed: int = DEFAULT_CV_SEED,
    cv_splits: int = DEFAULT_CV_SPLITS,
    cv_test_size: float = DEFAULT_CV_TEST_SIZE,
    min_rows: int | None = None,
    min_positives: int | None = None,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Join, resolve vectors, estimate and (optionally) write the receipt.

    Never fabricates: a join failure or an unavailable vector source produces a
    receipt with the failure status, the computable diagnostics, and explicit
    ``live_run_requirements`` instead of numbers.
    """
    started_at = timestamp or _utc_now()
    base: dict[str, Any] = {
        "receipt": RECEIPT_KIND,
        "timestamp": started_at,
        "snapshot": snapshot.provenance(),
        "td9_receipt_sha256": receipt_sha256,
        "metric_directions": dict(_METRIC_DIRECTIONS),
        "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        "probe_criticisms": list(PROBE_CRITICISMS),
        "label_provenance": {
            "status": "frozen-snapshot",
            "label_definition": "outcome == 'success' from the frozen snapshot above",
            "snapshot_db_sha256": snapshot.db_sha256,
            "admissibility_reason": snapshot.admissibility_reason,
            "warning": (
                "The LIVE episodic.db is not admissible ground truth: the 2026-09-17 leak "
                "purge changed it. Every label here was read from the frozen snapshot."
            ),
            "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        },
    }

    try:
        joined, join_report = join_receipt_to_snapshot(receipt, calls)
    except JoinUnavailable as exc:
        base.update(
            {
                "status": "join-unavailable",
                "join": {
                    "mode": "unavailable",
                    "reason": str(exc),
                    "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
                },
                "raw_support": summarize_actions(calls),
                "live_run_requirements": live_run_requirements(
                    snapshot=snapshot, join_status="unavailable", vector_status="unknown"
                ),
            }
        )
        return _maybe_write(base, receipt_path, artifacts_dir, started_at)

    base["join"] = join_report
    base["raw_support"] = summarize_actions(calls)
    base["rows"] = [row.to_record() for row in joined]
    floors = _support_floors(adapter, min_rows=min_rows, min_positives=min_positives)
    base["decision_breakdown"] = decision_breakdown(
        joined, calls, min_rows=floors[0], min_positives=floors[1]
    )

    try:
        vectors, vector_report = vector_source.resolve(calls)
    except VectorsUnavailable as exc:
        base.update(
            {
                "status": "vectors-unavailable",
                "vector_source": {
                    "name": vector_source.name,
                    "reason": str(exc),
                    "snapshot_faiss_path": (
                        str(snapshot.faiss_path) if snapshot.faiss_path else None
                    ),
                    "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
                },
                "live_run_requirements": live_run_requirements(
                    snapshot=snapshot, join_status="ok", vector_status="unavailable"
                ),
            }
        )
        return _maybe_write(base, receipt_path, artifacts_dir, started_at)

    base["vector_source"] = vector_report
    if adapter is None:
        adapter = build_probe_adapter()

    policy = estimate_policy_value(
        joined,
        calls,
        vectors,
        adapter,
        seed=cv_seed,
        n_splits=cv_splits,
        test_size=cv_test_size,
        min_rows=min_rows,
        min_positives=min_positives,
    )
    base["status"] = "estimated" if policy.get("status") == "estimated" else policy.get("status")
    base["policy_value"] = policy
    base["live_run_requirements"] = live_run_requirements(
        snapshot=snapshot, join_status="ok", vector_status="ok"
    )
    return _maybe_write(base, receipt_path, artifacts_dir, started_at)


def _maybe_write(
    receipt: dict[str, Any],
    receipt_path: str | Path | None,
    artifacts_dir: str | Path | None,
    stamp: str,
) -> dict[str, Any]:
    resolved = _resolve_receipt_path(receipt_path, artifacts_dir, stamp)
    if resolved is not None:
        receipt["receipt_path"] = str(resolved)
        _write_receipt(resolved, receipt)
    return receipt


def _resolve_receipt_path(
    receipt_path: str | Path | None,
    artifacts_dir: str | Path | None,
    stamp: str,
) -> Path | None:
    if receipt_path is not None:
        return Path(receipt_path)
    if artifacts_dir is None:
        return None
    safe_stamp = "".join(char for char in stamp if char.isalnum())
    return Path(artifacts_dir) / "typed_decisions" / f"routing_counterfactual-{safe_stamp}.json"


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── live rerun (decisions) ────────────────────────────────────────────────


def run_live_rerun(
    snapshot: Snapshot,
    calls: Sequence[SnapshotCall],
    *,
    n: int,
    seed: int,
    role: str,
    state_budget: int,
    json_n_tokens: int,
    server_url: str,
    live: bool,
) -> dict[str, Any]:
    """Re-run routing_replay to obtain decisions when the receipt is unusable.

    Imports the TD-9 runner lazily so the offline path never touches a model.
    Requires ``live=True`` for real calls (the TD-9 CLI's own gate).
    """
    from src.typed_decisions.routing_replay import (
        default_tokenize_fn,
        live_primitives,
        run_replay,
    )

    primitives = None
    tokenize_fn = None
    if live:
        primitives = live_primitives(server_url=server_url, role=role)
        tokenize_fn = default_tokenize_fn(primitives, role)
    try:
        return run_replay(
            [call.routing_row() for call in calls],
            snapshot_provenance=snapshot.provenance(),
            primitives=primitives,
            n=n,
            seed=seed,
            role=role,
            state_budget=state_budget,
            json_n_tokens=json_n_tokens,
            tokenize_fn=tokenize_fn,
            dry_run=not live,
        )
    finally:
        close = getattr(tokenize_fn, "close", None)
        if callable(close):
            close()


def _load_receipt(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CounterfactualError(f"receipt is not JSON: {path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise CounterfactualError(f"receipt is not a JSON object: {path}")
    return receipt, sha256


def build_vector_source(args: argparse.Namespace, snapshot: Snapshot) -> VectorSource:
    """Resolve the ``--vectors`` mode into a source (never a silent fallback)."""
    mode = args.vectors
    if mode == "none":
        return UnavailableVectorSource(
            reason="--vectors none: no e(x) source was supplied; the estimate is not computed"
        )
    if mode == "npz":
        path = Path(args.npz or args.reembedded or KNOWN_REEMBEDDED)
        return NpzVectorSource(path=path)
    if mode == "faiss":
        path = Path(args.faiss) if args.faiss else snapshot.faiss_path
        if path is None:
            raise VectorsUnavailable(
                "no faiss path given and the snapshot has no paired embeddings.faiss"
            )
        return FaissVectorSource(
            faiss_path=Path(path),
            id_map_path=Path(args.faiss_id_map) if args.faiss_id_map else None,
            allow_positional=bool(args.allow_positional_faiss),
        )
    if mode == "embed":
        return EmbeddedVectorSource(
            server_url=args.embed_server,
            use_parallel=bool(args.embed_parallel),
            embed_out=Path(args.embed_out) if args.embed_out else None,
        )
    # auto: only a snapshot-paired index counts as stored; anything else must be explicit.
    if snapshot.faiss_path is not None:
        id_map_path = Path(args.faiss_id_map) if args.faiss_id_map else None
        return FaissVectorSource(
            faiss_path=snapshot.faiss_path,
            id_map_path=id_map_path,
            allow_positional=bool(args.allow_positional_faiss),
        )
    return UnavailableVectorSource(
        reason=(
            "auto: the frozen snapshot is a direct episodic.db file with no paired "
            "embeddings.faiss, and no stored vector generation aligns with this corpus; "
            "supply --vectors npz (memory-id keyed) or --vectors embed (live canonical "
            "re-embedding)"
        )
    )


# ── dry-run plan ──────────────────────────────────────────────────────────


def build_plan(
    *,
    snapshot: Snapshot,
    calls: Sequence[SnapshotCall],
    receipt: Mapping[str, Any] | None,
    vector_source: VectorSource,
    cv_seed: int,
    cv_splits: int,
    cv_test_size: float,
    min_rows: int | None = None,
    min_positives: int | None = None,
) -> dict[str, Any]:
    """Dry-run plan: corpus, join, vector source and estimator, nothing fitted."""
    join_block: dict[str, Any] = {"mode": "not-attempted"}
    breakdown: dict[str, Any] | None = None
    if receipt is not None:
        try:
            joined, join_report = join_receipt_to_snapshot(receipt, calls)
            join_block = dict(join_report)
            join_block["n_joined_now"] = len(joined)
            floors = _support_floors(None, min_rows=min_rows, min_positives=min_positives)
            breakdown = decision_breakdown(
                joined, calls, min_rows=floors[0], min_positives=floors[1]
            )
        except JoinUnavailable as exc:
            join_block = {"mode": "unavailable", "reason": str(exc)}
    return {
        "receipt": f"{RECEIPT_KIND}-plan",
        "dry_run": True,
        "snapshot": snapshot.provenance(),
        "corpus": {
            "routing_rows": len(calls),
            "distinct_actions": len({call.action for call in calls}),
        },
        "raw_support": summarize_actions(calls),
        "join": join_block,
        "decision_breakdown": breakdown,
        "vector_source": {
            "name": vector_source.name,
            "requested": getattr(vector_source, "faiss_path", None)
            or getattr(vector_source, "path", None)
            or getattr(vector_source, "server_url", None),
        },
        "estimator": {
            "name": "escalation_prediction_probe.cross_fit_predict",
            "seed": cv_seed,
            "n_splits": cv_splits,
            "test_size": cv_test_size,
        },
        "metric_directions": dict(_METRIC_DIRECTIONS),
        "counterfactual_caveat": COUNTERFACTUAL_CAVEAT,
        "probe_criticisms": list(PROBE_CRITICISMS),
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.typed_decisions.routing_counterfactual",
        description=(
            "TD-10: counterfactual value/regret of the TD-9 routing policy on the frozen "
            "routing snapshot. Estimates are model-based and carry the counterfactual caveat."
        ),
    )
    parser.add_argument("--snapshot", required=True, help="frozen episodic.db or snapshot dir")
    parser.add_argument("--receipt", default=None, help="TD-9 routing-replay receipt JSON")
    parser.add_argument(
        "--live-rerun",
        action="store_true",
        help="re-run routing_replay to obtain decisions when the receipt is unusable",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; no fitting, no receipt"
    )
    parser.add_argument(
        "--live", action="store_true", help="allow real model calls for --live-rerun"
    )
    parser.add_argument("--n", type=int, default=REPLAY_DEFAULT_N, help="live-rerun sample size")
    parser.add_argument("--seed", type=int, default=20260918, help="live-rerun sampling seed")
    parser.add_argument(
        "--role", default=os.environ.get(ENV_ROLE, DEFAULT_ROLE), help="role charged for live calls"
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get(ENV_SERVER, DEFAULT_SERVER_URL),
        help="llama-server base URL for --live-rerun",
    )
    parser.add_argument(
        "--state-budget-chars", type=int, default=DEFAULT_STATE_BUDGET_CHARS, help="STATE budget"
    )
    parser.add_argument("--json-n-tokens", type=int, default=768, help="JSON fallback budget")
    parser.add_argument(
        "--vectors",
        choices=("auto", "none", "npz", "faiss", "embed"),
        default="auto",
        help="e(x) source (default auto: snapshot-paired faiss only, else unavailable)",
    )
    parser.add_argument("--npz", default=None, help="memory-id-keyed npz (ids/embeddings)")
    parser.add_argument("--reembedded", default=None, help="alias for --npz")
    parser.add_argument("--faiss", default=None, help="FAISS index path")
    parser.add_argument("--faiss-id-map", default=None, help="id_map.npy for --faiss alignment")
    parser.add_argument(
        "--allow-positional-faiss",
        action="store_true",
        help="accept unverified positional vectors (only for a known-paired index)",
    )
    parser.add_argument("--embed-server", default="http://127.0.0.1:8090", help="embedding server")
    parser.add_argument(
        "--embed-parallel", action="store_true", help="use the parallel embedder probe"
    )
    parser.add_argument("--embed-out", default=None, help="optional npz path for fresh embeddings")
    parser.add_argument("--cv-seed", type=int, default=DEFAULT_CV_SEED, help="cross-fit seed")
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS, help="cross-fit draws")
    parser.add_argument(
        "--cv-test-size", type=float, default=DEFAULT_CV_TEST_SIZE, help="cross-fit test fraction"
    )
    parser.add_argument("--min-support-rows", type=int, default=None, help="override probe floor")
    parser.add_argument(
        "--min-support-positives", type=int, default=None, help="override probe floor"
    )
    parser.add_argument("--probe-path", default=None, help="override the estimator probe path")
    parser.add_argument("--receipt-out", default=None, help="exact output receipt path")
    parser.add_argument("--artifacts-dir", default=None, help="directory for the default receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Returns 0 estimated, 1 error/refusal, 2 gate violation, 3 diagnostic-only."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        snapshot = resolve_snapshot(args.snapshot)
        calls = load_snapshot_calls(snapshot.db_path)
    except (ReplayError, CounterfactualError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        vector_source = build_vector_source(args, snapshot)
    except VectorsUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        receipt = None
        if args.receipt:
            try:
                receipt, _ = _load_receipt(args.receipt)
            except CounterfactualError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        plan = build_plan(
            snapshot=snapshot,
            calls=calls,
            receipt=receipt,
            vector_source=vector_source,
            cv_seed=args.cv_seed,
            cv_splits=args.cv_splits,
            cv_test_size=args.cv_test_size,
            min_rows=args.min_support_rows,
            min_positives=args.min_support_positives,
        )
        print(json.dumps(plan, indent=2, sort_keys=True, default=_json_default))
        return 0

    if not args.receipt and not args.live_rerun:
        parser.error("--receipt or --live-rerun is required (or --dry-run to print the plan)")

    receipt_sha256: str | None = None
    if args.live_rerun:
        if not args.live:
            print(
                "refusing to run: --live-rerun needs --live for real model calls "
                "(or --dry-run to print the plan)",
                file=sys.stderr,
            )
            return 2
        try:
            receipt = run_live_rerun(
                snapshot,
                calls,
                n=args.n,
                seed=args.seed,
                role=args.role,
                state_budget=args.state_budget_chars,
                json_n_tokens=args.json_n_tokens,
                server_url=args.server_url,
                live=True,
            )
        except (ReplayError, CounterfactualError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        receipt_sha256 = _sha256_text(json.dumps(receipt, sort_keys=True, default=str))
    else:
        try:
            receipt, receipt_sha256 = _load_receipt(args.receipt)
        except (CounterfactualError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    try:
        adapter = build_probe_adapter(args.probe_path)
    except CounterfactualError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        result = run_counterfactual(
            snapshot=snapshot,
            calls=calls,
            receipt=receipt,
            vector_source=vector_source,
            adapter=adapter,
            receipt_sha256=receipt_sha256,
            cv_seed=args.cv_seed,
            cv_splits=args.cv_splits,
            cv_test_size=args.cv_test_size,
            min_rows=args.min_support_rows,
            min_positives=args.min_support_positives,
            receipt_path=args.receipt_out,
            artifacts_dir=args.artifacts_dir,
        )
    except CounterfactualError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    if result.get("status") == "estimated":
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
