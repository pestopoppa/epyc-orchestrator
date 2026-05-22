"""MMLUAdapter, GPQAAdapter, SimpleQAAdapter, HotpotQAAdapter dataset adapters

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


class MMLUAdapter(BaseAdapter):
    """MMLU: 14,042 multiple-choice questions across 57 subjects."""

    suite_name = "general"
    has_real_tiers = True  # Subject-based difficulty mapping
    CHOICE_LABELS = ["A", "B", "C", "D"]

    HARD_SUBJECTS = {
        "abstract_algebra", "college_mathematics", "formal_logic",
        "college_physics", "electrical_engineering", "machine_learning",
        "conceptual_physics", "college_chemistry", "anatomy",
    }
    EASY_SUBJECTS = {
        "high_school_geography", "high_school_us_history",
        "miscellaneous", "us_foreign_policy",
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset("cais/mmlu", "all", split="test")
        except Exception as e:
            print(f"  [adapter] MMLU load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        subject = row.get("subject", "general")

        # Build multiple-choice prompt
        prompt_lines = [question, ""]
        for i, choice in enumerate(choices):
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {choice}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A, B, C, or D).")

        expected = self.CHOICE_LABELS[answer_idx]

        # Tier based on subject difficulty
        if subject in self.HARD_SUBJECTS:
            tier = 3
        elif subject in self.EASY_SUBJECTS:
            tier = 1
        else:
            tier = 2

        return {
            "id": f"mmlu_{subject}_{idx:05d}",
            "suite": "general",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }

    def _get_tier_for_index(self, idx: int) -> int:
        subject = self._dataset[idx].get("subject", "general")
        if subject in self.HARD_SUBJECTS:
            return 3
        elif subject in self.EASY_SUBJECTS:
            return 1
        return 2


# ── GSM8K + MATH-500 (Math) ──────────────────────────────────────────────




class GPQAAdapter(BaseAdapter):
    """GPQA Diamond: 448 graduate-level science questions.

    Source: Idavidrein/gpqa on HuggingFace.
    Questions designed to be "Google-proof" — experts score 65%, GPT-4 = 39%.
    Perfect for mode-advantage: frontdoor fails, tools/specialists help.

    Scoring: multiple_choice (A/B/C/D).
    Tiers: Based on subdomain difficulty.
    """

    suite_name = "gpqa"
    has_real_tiers = True
    CHOICE_LABELS = ["A", "B", "C", "D"]

    # Subdomains with higher difficulty (based on benchmark papers)
    HARD_SUBDOMAINS = {"physics", "chemistry"}
    EASY_SUBDOMAINS = {"biology"}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            # Use ankner/gpqa (ungated mirror of the original)
            # Original Idavidrein/gpqa requires access approval
            self._dataset = hf.load_dataset(
                "ankner/gpqa", split="train",
            )
        except Exception as e:
            print(f"  [adapter] GPQA load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        subdomain = row.get("Subdomain", "").lower()
        # Map subdomain to tier
        if any(hard in subdomain for hard in self.HARD_SUBDOMAINS):
            return 3
        elif any(easy in subdomain for easy in self.EASY_SUBDOMAINS):
            return 1
        return 2

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("Question", "")
        # GPQA has Correct Answer and Incorrect Answer 1-3 fields
        correct_answer = row.get("Correct Answer", "")
        incorrect_1 = row.get("Incorrect Answer 1", "")
        incorrect_2 = row.get("Incorrect Answer 2", "")
        incorrect_3 = row.get("Incorrect Answer 3", "")

        # Collect all non-empty choices
        choices = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
        choices = [c for c in choices if c]

        # Randomize choice order deterministically based on question hash
        import hashlib
        seed = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(choices)

        # Find correct answer index after shuffle
        correct_idx = choices.index(correct_answer) if correct_answer in choices else 0
        expected_letter = self.CHOICE_LABELS[correct_idx]

        # Build prompt
        prompt_lines = [question, ""]
        for i, choice in enumerate(choices[:4]):  # Max 4 choices
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {choice}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A, B, C, or D).")

        subdomain = row.get("Subdomain", "general")
        tier = self._get_tier_for_index(idx)

        return {
            "id": f"gpqa_{subdomain}_{idx:04d}",
            "suite": "gpqa",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": expected_letter,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }


# ── SimpleQA (Factual Accuracy) ───────────────────────────────────────────




class SimpleQAAdapter(BaseAdapter):
    """SimpleQA: 4,326 short factual questions.

    Source: MAISAAI/openai_simple_qa_test_set on HuggingFace.
    Questions have unambiguous, short factual answers.
    GPT-4 scores <40% — ideal for mode-advantage with search tools.

    Scoring: exact_match (normalized).
    Tiers: Based on question complexity heuristics.
    """

    suite_name = "simpleqa"
    has_real_tiers = True

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset(
                "MAISAAI/openai_simple_qa_test_set", split="train",
            )
        except Exception as e:
            print(f"  [adapter] SimpleQA load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        question = row.get("problem", "")
        answer = row.get("answer", "")

        # Tier heuristics:
        # T3: Long answers (likely multi-part)
        # T2: Medium answers or questions with dates/numbers
        # T1: Short answers
        answer_words = len(answer.split())
        if answer_words > 10:
            return 3
        elif answer_words > 3 or re.search(r"\d{4}", question):
            return 2
        return 1

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("problem", "")
        answer = row.get("answer", "")
        metadata = row.get("metadata", {}) or {}
        topic = metadata.get("topic", "general") if isinstance(metadata, dict) else "general"

        prompt = (
            f"{question}\n\n"
            "Give a short, precise answer. "
            "Put your final answer inside <answer></answer> tags."
        )

        tier = self._get_tier_for_index(idx)
        clean_topic = re.sub(r"[^a-zA-Z0-9_]", "_", str(topic))[:20]

        return {
            "id": f"simpleqa_{clean_topic}_{idx:05d}",
            "suite": "simpleqa",
            "prompt": prompt,
            "context": "",
            "expected": answer.strip(),
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "f1",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "threshold": 0.5,
                "normalize": True,
            },
        }


# ── HotpotQA (Multi-hop Reasoning) ────────────────────────────────────────




class HotpotQAAdapter(BaseAdapter):
    """HotpotQA: Multi-hop reasoning questions requiring 2+ facts.

    Source: hotpotqa/hotpot_qa on HuggingFace.
    Questions require combining information from multiple documents.
    30B fails at ~40%, search tools can push to ~80%.

    Scoring: f1 (token-level F1 score).
    Tiers: Based on question type (bridge vs comparison).
    """

    suite_name = "hotpotqa"
    has_real_tiers = True

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            # Load the distractor setting (harder than fullwiki)
            ds = hf.load_dataset(
                "hotpotqa/hotpot_qa", "distractor", split="validation",
            )
            # Filter to "hard" questions only
            self._dataset = ds.filter(lambda x: x.get("level", "") == "hard")
        except Exception as e:
            print(f"  [adapter] HotpotQA load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        q_type = row.get("type", "")
        # Comparison questions are generally harder than bridge questions
        if q_type == "comparison":
            return 3
        return 2  # All "hard" level questions are at least T2

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("question", "")
        answer = row.get("answer", "")
        q_id = row.get("id", f"hotpot_{idx:05d}")
        q_type = row.get("type", "bridge")
        supporting_facts = row.get("supporting_facts", {})
        context = row.get("context", {})

        # Build context from supporting paragraphs
        context_text = ""
        if context:
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                context_text += f"### {title}\n"
                context_text += " ".join(sentences) + "\n\n"

        prompt = question
        if context_text:
            prompt = f"Context:\n{context_text.strip()}\n\nQuestion: {question}"

        prompt += (
            "\n\nGive a short, precise answer based on the context. "
            "Put your final answer inside <answer></answer> tags."
        )

        tier = self._get_tier_for_index(idx)

        return {
            "id": f"hotpot_{q_type}_{q_id}",
            "suite": "hotpotqa",
            "prompt": prompt,
            "context": context_text,
            "expected": answer.strip(),
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "f1",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "threshold": 0.5,  # Minimum F1 to count as correct
            },
        }


# ── LiveCodeBench (Competition Programming) ───────────────────────────────


