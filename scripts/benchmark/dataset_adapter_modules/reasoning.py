"""ThinkingAdapter, IFEvalAdapter dataset adapters

Extracted from scripts/benchmark/dataset_adapters.py during the 2026-05-22
Task-A refactor. Re-exported from dataset_adapters.py for backwards
compatibility — existing imports keep working.
"""

from __future__ import annotations

import random

from .base import BaseAdapter


class ThinkingAdapter(BaseAdapter):
    """ARC-Challenge (1,172) + HellaSwag (10,042) = 11,214 reasoning questions."""

    suite_name = "thinking"
    _arc = None
    _hellaswag = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._arc = hf.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            self._hellaswag = hf.load_dataset("Rowan/hellaswag", split="validation")
            self._dataset = list(range(len(self._arc) + len(self._hellaswag)))
        except Exception as e:
            print(f"  [adapter] Thinking datasets load failed: {e}")
            self._dataset = []

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        rng = random.Random(seed)
        arc_len = len(self._arc) if self._arc else 0
        hs_len = len(self._hellaswag) if self._hellaswag else 0

        n_arc = min(int(n * 0.5), arc_len)
        n_hs = min(n - n_arc, hs_len)
        if n_hs < n - n_arc:
            n_arc = min(n - n_hs, arc_len)

        results = []
        if n_arc > 0:
            arc_indices = rng.sample(range(arc_len), n_arc)
            results.extend(self._arc_prompt(i) for i in arc_indices)
        if n_hs > 0:
            hs_indices = rng.sample(range(hs_len), n_hs)
            results.extend(self._hellaswag_prompt(i) for i in hs_indices)

        rng.shuffle(results)
        return results

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        arc_len = len(self._arc) if self._arc else 0
        if idx < arc_len:
            return self._arc_prompt(idx)
        return self._hellaswag_prompt(idx - arc_len)

    CHOICE_LABELS = ["A", "B", "C", "D", "E"]

    def _arc_prompt(self, idx: int) -> dict:
        row = self._arc[idx]
        question = row["question"]
        choices_data = row["choices"]
        answer_key = row["answerKey"]
        qid = row["id"]

        # ARC choices format: {"text": [...], "label": [...]}
        labels = choices_data["label"]
        texts = choices_data["text"]

        prompt_lines = [question, ""]
        for label, text in zip(labels, texts):
            prompt_lines.append(f"{label}) {text}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only.")

        return {
            "id": f"arc_{qid}",
            "suite": "thinking",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": answer_key,
            "scoring": [],
            "image_path": "",
            "tier": 2,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }

    def _hellaswag_prompt(self, idx: int) -> dict:
        row = self._hellaswag[idx]
        context = row["ctx"]
        endings = row["endings"]
        label = row["label"]
        ind = row["ind"]

        prompt_lines = [
            "Choose the most plausible continuation:",
            "",
            f"Context: {context}",
            "",
        ]
        for i, ending in enumerate(endings):
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {ending}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A, B, C, or D).")

        expected = self.CHOICE_LABELS[int(label)] if isinstance(label, (int, str)) else "A"

        return {
            "id": f"hellaswag_{ind:05d}",
            "suite": "thinking",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": 1,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }


# ── IFEval (Instruction Precision) ───────────────────────────────────────




class IFEvalAdapter(BaseAdapter):
    """IFEval: 541 instruction-following prompts with verifiable constraints."""

    suite_name = "instruction_precision"
    has_real_tiers = True  # Tier from constraint count: 1→T1, 2-3→T2, 4+→T3

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset("google/IFEval", split="train")
        except Exception as e:
            print(f"  [adapter] IFEval load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        prompt = row["prompt"]
        key = row["key"]
        instruction_ids = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])

        # IFEval doesn't have simple expected answers — it has constraint verifiers.
        # We extract the first instruction as the primary constraint to check.
        primary_constraint = instruction_ids[0] if instruction_ids else "unknown"

        # Build scoring config from IFEval's constraint types
        scoring_method, scoring_config = self._constraint_to_scoring(
            primary_constraint, kwargs_list[0] if kwargs_list else {}
        )

        # Determine tier from constraint complexity
        n_constraints = len(instruction_ids)
        tier = 1 if n_constraints <= 1 else (2 if n_constraints <= 3 else 3)

        return {
            "id": f"ifeval_{key}",
            "suite": "instruction_precision",
            "prompt": prompt,
            "context": "",
            "expected": "",  # IFEval uses programmatic verification
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": scoring_method,
            "scoring_config": scoring_config,
            "ifeval_instructions": instruction_ids,
            "ifeval_kwargs": kwargs_list,
        }

    @staticmethod
    def _constraint_to_scoring(constraint_id: str, kwargs: dict) -> tuple[str, dict]:
        """Map IFEval constraint type to our scoring system."""
        # IFEval constraints: https://github.com/google-research/google-research/tree/master/instruction_following_eval
        if "no_comma" in constraint_id:
            return "programmatic", {"verifier": "no_comma"}
        elif "number_highlighted_sections" in constraint_id:
            n = kwargs.get("num_highlights", 1)
            return "programmatic", {"verifier": "highlighted_sections", "count": n}
        elif "number_paragraphs" in constraint_id:
            n = kwargs.get("num_paragraphs", 1)
            return "programmatic", {"verifier": "paragraph_count", "count": n}
        elif "number_words" in constraint_id or "length" in constraint_id:
            n = kwargs.get("num_words")
            rel = kwargs.get("relation", "at_least")
            return "programmatic", {"verifier": "word_count", "count": n, "relation": rel}
        elif "number_sentences" in constraint_id:
            n = kwargs.get("num_sentences")
            rel = kwargs.get("relation", "at_least")
            return "programmatic", {"verifier": "sentence_count", "count": n, "relation": rel}
        elif "postscript" in constraint_id:
            return "substring", {"case_sensitive": False, "substring": "P.S."}
        elif "title" in constraint_id:
            return "programmatic", {"verifier": "has_title"}
        elif "json_format" in constraint_id or "json" in constraint_id:
            return "programmatic", {"verifier": "json_valid"}
        elif "number_placeholders" in constraint_id:
            n = kwargs.get("num_placeholders", 1)
            return "programmatic", {"verifier": "placeholder_count", "count": n}
        elif "bullet_list" in constraint_id or "number_bullet" in constraint_id:
            n = kwargs.get("num_bullets")
            return "programmatic", {"verifier": "bullet_count", "count": n}
        elif "keywords" in constraint_id:
            kw = kwargs.get("keywords", [])
            return "programmatic", {"verifier": "contains_keywords", "keywords": kw}
        elif "forbidden" in constraint_id:
            fw = kwargs.get("forbidden_words", [])
            return "programmatic", {"verifier": "no_forbidden_words", "forbidden": fw}
        elif "language" in constraint_id:
            lang = kwargs.get("language", "en")
            return "programmatic", {"verifier": "language", "language": lang}
        else:
            # Generic fallback — just check response is non-empty
            return "programmatic", {"verifier": "non_empty", "constraint": constraint_id}

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        n_constraints = len(row.get("instruction_id_list", []))
        return 1 if n_constraints <= 1 else (2 if n_constraints <= 3 else 3)


# ── VL (Vision-Language) ──────────────────────────────────────────────────


