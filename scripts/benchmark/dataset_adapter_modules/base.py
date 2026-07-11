"""BaseAdapter dataset adapter

Extracted from scripts/benchmark/dataset_adapters.py during the 2026-05-22
Task-A refactor. Re-exported from dataset_adapters.py for backwards
compatibility — existing imports keep working.
"""

from __future__ import annotations

import random



class BaseAdapter:
    """Base class for dataset adapters."""

    suite_name: str = ""
    _dataset = None

    # Adapters with real difficulty data should set this True
    has_real_tiers: bool = False

    def _ensure_loaded(self):
        raise NotImplementedError

    @property
    def total_available(self) -> int:
        self._ensure_loaded()
        return len(self._dataset) if self._dataset is not None else 0

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        raise NotImplementedError

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        """Sample n questions. If stratify=True AND adapter has real tiers,
        draw equal counts per tier for balanced difficulty distribution."""
        self._ensure_loaded()
        if not self._dataset:
            return []
        if stratify and self.has_real_tiers:
            return self._stratified_sample(n, seed)
        rng = random.Random(seed)
        indices = rng.sample(range(len(self._dataset)), min(n, len(self._dataset)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]

    def _stratified_sample(self, n: int, seed: int) -> list[dict]:
        """Draw equal questions per tier. Requires _get_tier_for_index()."""
        rng = random.Random(seed)
        # Bucket indices by tier
        tier_buckets: dict[int, list[int]] = {}
        for i in range(len(self._dataset)):
            t = self._get_tier_for_index(i)
            tier_buckets.setdefault(t, []).append(i)

        tiers = sorted(tier_buckets.keys())
        if not tiers:
            return []

        # Equal share per tier, remainder distributed round-robin
        per_tier = n // len(tiers)
        remainder = n % len(tiers)

        results = []
        for i, t in enumerate(tiers):
            bucket = tier_buckets[t]
            count = per_tier + (1 if i < remainder else 0)
            count = min(count, len(bucket))
            indices = rng.sample(bucket, count)
            results.extend(self._row_to_prompt(idx, self._dataset[idx]) for idx in indices)

        rng.shuffle(results)
        return results

    def extract_all(self) -> list[dict]:
        """Extract ALL questions from this adapter as prompt dicts.

        Calls _ensure_loaded() then iterates the full dataset through
        _row_to_prompt(). Used by question_pool.py to pre-extract the
        complete question corpus into a JSONL file.
        """
        self._ensure_loaded()
        if not self._dataset:
            return []
        results = []
        for i in range(len(self._dataset)):
            try:
                row = self._dataset[i] if not isinstance(self._dataset[i], int) else {}
                prompt = self._row_to_prompt(i, row)
                if prompt:
                    results.append(prompt)
            except Exception:
                continue
        return results

    def _get_tier_for_index(self, idx: int) -> int:
        """Return tier for a given dataset index. Override in adapters with real tiers."""
        return 1


# ── MMLU (General Knowledge) ─────────────────────────────────────────────


