"""VLAdapter, GaiaAdapter dataset adapters

Extracted from scripts/benchmark/dataset_adapters.py during the 2026-05-22
Task-A refactor. Re-exported from dataset_adapters.py for backwards
compatibility — existing imports keep working.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAdapter


class VLAdapter(BaseAdapter):
    """VL: delegates to extract_vl_debug_suite.VLDatasetAdapter (3,500 questions)."""

    suite_name = "vl"
    _vl_adapter = None

    def _ensure_loaded(self):
        if self._vl_adapter is not None:
            return
        try:
            from extract_vl_debug_suite import VLDatasetAdapter
            self._vl_adapter = VLDatasetAdapter()
            self._dataset = list(range(self._vl_adapter.total_available))
        except ImportError:
            print("  [adapter] VL adapter not available (extract_vl_debug_suite.py)")
            self._dataset = []

    @property
    def total_available(self) -> int:
        self._ensure_loaded()
        return self._vl_adapter.total_available if self._vl_adapter else 0

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if self._vl_adapter:
            return self._vl_adapter.sample(n=n, seed=seed, extract_images=True)
        return []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        return {}  # Not used — sample() delegates directly

    def extract_all(self) -> list[dict]:
        """VL adapter delegates to VLDatasetAdapter — sample everything."""
        self._ensure_loaded()
        if self._vl_adapter:
            try:
                return self._vl_adapter.sample(
                    n=self._vl_adapter.total_available, seed=0, extract_images=True,
                )
            except Exception:
                return []
        return []


# ── GAIA (Multi-step tool use) ───────────────────────────────────────────




class GaiaAdapter(BaseAdapter):
    """GAIA: 165 dev questions requiring multi-step reasoning and tool use.

    Source: gaia-benchmark/GAIA on HuggingFace (CC-BY-4.0).
    Questions have exact-match answers (number, name, or short string).
    Levels 1-3 map to tiers T1-T3.

    File attachments are staged to /mnt/raid0/llm/tmp/gaia/{question_id}/
    so REPL mode can access them.
    """

    suite_name = "gaia"
    has_real_tiers = True
    _STAGING_DIR = Path("/mnt/raid0/llm/tmp/gaia")
    # Skip questions requiring audio/video processing
    _SKIP_EXTENSIONS = {".mp3", ".wav", ".mp4", ".avi", ".mov", ".flac", ".ogg"}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            ds = hf.load_dataset(
                "gaia-benchmark/GAIA", "2023_all", split="validation",
            )
            # Filter out questions with unsupported file types
            filtered = []
            for i, row in enumerate(ds):
                file_name = row.get("file_name", "") or ""
                if file_name:
                    ext = Path(file_name).suffix.lower()
                    if ext in self._SKIP_EXTENSIONS:
                        continue
                filtered.append(row)
            self._dataset = filtered
        except Exception as e:
            print(f"  [adapter] GAIA load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        level = self._dataset[idx].get("Level", 1)
        return min(max(int(level), 1), 3)

    def _stage_file(self, question_id: str, row: dict) -> str:
        """Stage attached file to temp dir. Returns path hint or empty string."""
        file_name = row.get("file_name", "") or ""
        file_bytes = row.get("file_path", "") or ""
        if not file_name:
            return ""

        staging = self._STAGING_DIR / question_id
        staging.mkdir(parents=True, exist_ok=True)
        dest = staging / file_name

        if not dest.exists():
            # file_path in GAIA dataset is the actual path to the file
            # In HF datasets, this may be a local cache path
            try:
                if isinstance(file_bytes, (str, Path)) and Path(file_bytes).exists():
                    import shutil
                    shutil.copy2(file_bytes, dest)
                elif isinstance(file_bytes, bytes):
                    dest.write_bytes(file_bytes)
            except Exception as e:
                return ""

        return f"\nThe file is available at: {dest}"

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("Question", "")
        answer = row.get("Final answer", "") or row.get("answer", "")
        level = row.get("Level", 1)
        task_id = row.get("task_id", f"gaia_{idx:04d}")

        # Clean question ID for filesystem
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(task_id))

        # Stage any attached files
        file_hint = self._stage_file(clean_id, row)

        prompt = question.strip()
        if file_hint:
            prompt += file_hint

        prompt += (
            "\n\nGive a short, precise answer. "
            "If the answer is a number, give just the number. "
            "Put your final answer inside <answer></answer> tags."
        )

        return {
            "id": f"gaia_{clean_id}",
            "suite": "gaia",
            "prompt": prompt,
            "context": "",
            "expected": str(answer).strip(),
            "scoring": [],
            "image_path": "",
            "tier": min(max(int(level), 1), 3),
            "scoring_method": "exact_match",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "normalize": True,
            },
        }


# ── CRUXEval (Code output/input prediction) ─────────────────────────────


