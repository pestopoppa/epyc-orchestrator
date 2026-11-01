"""Isolated paired StrategyStore experiment for AP-29, without live wiring."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

from .strategy_store import StrategyStore


@dataclass(frozen=True)
class PairedArms:
    """Native retrieval results; adoption requires a separately authorized rule."""

    control: list[Any]
    candidate: list[Any]
    candidate_result: Any
    deleted_ids: frozenset[str]
    adoption: str = "pending_comparison_rule"


def run_paired_arms(
    episodes: Sequence[Mapping[str, Any]],
    *,
    delete_ids: set[str],
    embedder: Any,
    embedding_dim: int,
    query_text: str,
    abstract: Callable[[StrategyStore], Any],
    retrieve_options: Mapping[str, Any] | None = None,
) -> PairedArms:
    """Run matched episodic and abstracting arms in new, isolated stores.

    Input records are StrategyStore.store arguments with ``id`` in place of
    ``entry_id``. Only raw entries are accepted. Deletions are applied before
    either store is populated. Both stores receive identical surviving inputs,
    embedder, dimensions and retrieval options. Only the candidate store is
    passed to ``abstract``. Temporary stores are always closed and removed;
    neither caller-supplied paths nor an existing live store can be selected.
    This runs no model itself, scores no outcomes, and adopts no winner.
    """
    originals = [deepcopy(dict(episode)) for episode in episodes]
    ids = [episode.get("id") for episode in originals]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("episodes require unique non-empty string IDs")
    if any(episode.get("entry_type", "raw") != "raw" for episode in originals):
        raise ValueError("only original raw episodes are allowed")
    deleted = frozenset(delete_ids)
    if not deleted.issubset(ids):
        raise ValueError("deletion IDs must identify input episodes")
    if embedder is None:
        raise ValueError("an explicit experiment embedder is required")
    retained = [episode for episode in originals if episode["id"] not in deleted]
    options = dict(retrieve_options or {})
    with TemporaryDirectory(prefix="ap29-paired-") as directory:
        control = StrategyStore(Path(directory) / "control", embedding_dim, embedder)
        try:
            candidate = StrategyStore(Path(directory) / "candidate", embedding_dim, embedder)
            try:
                for episode in retained:
                    entry = deepcopy(episode)
                    entry["entry_id"] = entry.pop("id")
                    control.store(**deepcopy(entry))
                    candidate.store(**deepcopy(entry))
                candidate_result = abstract(candidate)
                control_results = control.retrieve(query_text, **options)
                candidate_results = candidate.retrieve(query_text, **options)
                return PairedArms(control_results, candidate_results, candidate_result, deleted)
            finally:
                candidate.close()
        finally:
            control.close()
