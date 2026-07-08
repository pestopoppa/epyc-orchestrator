"""MathAdapter dataset adapter

Extracted from scripts/benchmark/dataset_adapters.py during the 2026-05-22
Task-A refactor. Re-exported from dataset_adapters.py for backwards
compatibility — existing imports keep working.
"""

from __future__ import annotations

import random
import re

from .base import BaseAdapter


class MathAdapter(BaseAdapter):
    """GSM8K (1,319) + MATH-500 (500) = 1,819 math problems."""

    suite_name = "math"
    has_real_tiers = True  # GSM8K=T1, MATH-500 level 1-3=T2, level 4-5=T3
    _gsm8k = None
    _math500 = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._gsm8k = hf.load_dataset("openai/gsm8k", "main", split="test")
            try:
                self._math500 = hf.load_dataset("HuggingFaceH4/MATH-500", split="test")
            except Exception:
                self._math500 = []
            # Combine into unified list
            self._dataset = list(range(len(self._gsm8k) + len(self._math500)))
        except Exception as e:
            print(f"  [adapter] Math datasets load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        # idx is from unified list; row is ignored, we index directly
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0

        if idx < gsm8k_len:
            return self._gsm8k_prompt(idx, self._gsm8k[idx])
        else:
            math_idx = idx - gsm8k_len
            return self._math500_prompt(math_idx, self._math500[math_idx])

    def _get_tier_for_index(self, idx: int) -> int:
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0
        if idx < gsm8k_len:
            return 1  # GSM8K = grade-school
        math_idx = idx - gsm8k_len
        if self._math500 and math_idx < len(self._math500):
            level = self._math500[math_idx].get("level", 3)
            return 2 if level <= 3 else 3
        return 1

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        if stratify:
            return self._stratified_sample(n, seed)
        rng = random.Random(seed)
        # Split: ~60% GSM8K, ~40% MATH-500
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0
        math_len = len(self._math500) if self._math500 else 0

        n_gsm = min(int(n * 0.6), gsm8k_len)
        n_math = min(n - n_gsm, math_len)
        if n_math < n - n_gsm:
            n_gsm = min(n - n_math, gsm8k_len)

        results = []
        if n_gsm > 0:
            gsm_indices = rng.sample(range(gsm8k_len), n_gsm)
            results.extend(self._gsm8k_prompt(i, self._gsm8k[i]) for i in gsm_indices)
        if n_math > 0:
            math_indices = rng.sample(range(math_len), n_math)
            results.extend(self._math500_prompt(i, self._math500[i]) for i in math_indices)

        rng.shuffle(results)
        return results

    @staticmethod
    def _extract_gsm8k_answer(answer_text: str) -> str:
        """Extract numeric answer from GSM8K solution (after ####)."""
        # Try <answer> tag format first, fall back to legacy ####
        match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)
        if not match:
            match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return answer_text.strip()

    def _gsm8k_prompt(self, idx: int, row: dict) -> dict:
        question = row["question"]
        answer_text = row["answer"]
        expected = self._extract_gsm8k_answer(answer_text)

        return {
            "id": f"gsm8k_{idx:05d}",
            "suite": "math",
            "prompt": question + "\n\nSolve step by step. Put your final numeric answer inside <answer></answer> tags.",
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": 1,  # GSM8K is grade-school level
            "scoring_method": "exact_match",
            "scoring_config": {"extract_pattern": r"<answer>(.*?)</answer>"},
        }

    def _math500_prompt(self, idx: int, row: dict) -> dict:
        problem = row["problem"]
        answer = row.get("answer", "")
        level = row.get("level", 3)
        subject = row.get("subject", "")

        # Map MATH difficulty level to tier
        tier = 2 if level <= 3 else 3

        return {
            "id": f"math500_{subject}_{idx:05d}",
            "suite": "math",
            "prompt": problem + "\n\nPut your final answer in \\boxed{}.",
            "context": "",
            "expected": answer,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": False},
        }


# ── HumanEval + MBPP (Coder) ─────────────────────────────────────────────

