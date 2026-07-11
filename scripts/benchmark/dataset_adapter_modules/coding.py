"""CoderAdapter, CRUXEvalAdapter, BigCodeBenchAdapter, LiveCodeBenchAdapter, DebugBenchAdapter, USACOAdapter dataset adapters

Extracted from scripts/benchmark/dataset_adapters.py during the 2026-05-22
Task-A refactor. Re-exported from dataset_adapters.py for backwards
compatibility — existing imports keep working.
"""

from __future__ import annotations

import random
import re

from .base import BaseAdapter


class CoderAdapter(BaseAdapter):
    """HumanEval (164) + MBPP (500) = 664 coding problems."""

    suite_name = "coder"
    _humaneval = None
    _mbpp = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._humaneval = hf.load_dataset("openai/openai_humaneval", split="test")
            self._mbpp = hf.load_dataset("google-research-datasets/mbpp", split="test")
            self._dataset = list(range(len(self._humaneval) + len(self._mbpp)))
        except Exception as e:
            print(f"  [adapter] Coder datasets load failed: {e}")
            self._dataset = []

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        rng = random.Random(seed)
        he_len = len(self._humaneval) if self._humaneval else 0
        mbpp_len = len(self._mbpp) if self._mbpp else 0

        n_he = min(int(n * 0.4), he_len)
        n_mbpp = min(n - n_he, mbpp_len)
        if n_mbpp < n - n_he:
            n_he = min(n - n_mbpp, he_len)

        results = []
        if n_he > 0:
            he_indices = rng.sample(range(he_len), n_he)
            results.extend(self._humaneval_prompt(i) for i in he_indices)
        if n_mbpp > 0:
            mbpp_indices = rng.sample(range(mbpp_len), n_mbpp)
            results.extend(self._mbpp_prompt(i) for i in mbpp_indices)

        rng.shuffle(results)
        return results

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        he_len = len(self._humaneval) if self._humaneval else 0
        if idx < he_len:
            return self._humaneval_prompt(idx)
        return self._mbpp_prompt(idx - he_len)

    def _humaneval_prompt(self, idx: int) -> dict:
        row = self._humaneval[idx]
        prompt_text = row["prompt"]
        _ = row["canonical_solution"]  # kept for schema completeness
        _ = row["test"]  # kept for schema completeness
        entry_point = row["entry_point"]
        task_id = row["task_id"]

        # Build a prompt that asks to complete the function
        full_prompt = (
            f"Complete the following Python function:\n\n"
            f"```python\n{prompt_text}```\n\n"
            f"Write only the function body (the part after the signature)."
        )

        return {
            "id": f"humaneval_{task_id.replace('/', '_')}",
            "suite": "coder",
            "prompt": full_prompt,
            "context": "",
            "expected": entry_point,
            "scoring": [],
            "image_path": "",
            "tier": 2,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": True, "substring": entry_point},
        }

    def _mbpp_prompt(self, idx: int) -> dict:
        row = self._mbpp[idx]
        task_id = row["task_id"]
        text = row["text"]
        test_list = row.get("test_list", [])

        # Include test cases as hints
        test_hint = ""
        if test_list:
            test_hint = "\n\nTest cases:\n" + "\n".join(f"  {t}" for t in test_list[:3])

        full_prompt = (
            f"{text}{test_hint}\n\n"
            f"Write a Python function to solve this."
        )

        # Extract expected function name from test cases
        func_name = ""
        if test_list:
            match = re.search(r"assert\s+(\w+)\(", test_list[0])
            if match:
                func_name = match.group(1)

        return {
            "id": f"mbpp_{task_id:04d}",
            "suite": "coder",
            "prompt": full_prompt,
            "context": "",
            "expected": func_name or "def",
            "scoring": [],
            "image_path": "",
            "tier": 1,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": True, "substring": func_name or "def"},
        }


# ── ARC-Challenge + HellaSwag (Thinking) ──────────────────────────────────




class CRUXEvalAdapter(BaseAdapter):
    """CRUXEval: 800 functions × 2 tasks (output + input prediction).

    Source: cruxeval-org/cruxeval on HuggingFace.
    Output prediction is the pure REPL-advantage case: "just run the code."
    Input prediction tests reasoning: "what input gives this output?"

    Scoring: code_execution (assertion-based).
    """

    suite_name = "cruxeval"
    _raw_dataset = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._raw_dataset = hf.load_dataset(
                "cruxeval-org/cruxeval", split="test",
            )
            # Each row becomes 2 questions (output pred + input pred)
            self._dataset = list(range(len(self._raw_dataset) * 2))
        except Exception as e:
            print(f"  [adapter] CRUXEval load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        # idx 0..N-1 = output prediction, N..2N-1 = input prediction
        raw_len = len(self._raw_dataset) if self._raw_dataset else 0
        if idx < raw_len:
            return self._output_prompt(idx)
        return self._input_prompt(idx - raw_len)

    def _output_prompt(self, idx: int) -> dict:
        row = self._raw_dataset[idx]
        code = row.get("code", "")
        input_val = row.get("input", "")
        output_val = row.get("output", "")

        prompt = (
            f"What does the following Python code print when called with "
            f"the given input?\n\n"
            f"```python\n{code}\n```\n\n"
            f"Input: `{input_val}`\n\n"
            f"Give the exact output inside <answer></answer> tags."
        )

        return {
            "id": f"cruxeval_output_{idx:04d}",
            "suite": "cruxeval",
            "prompt": prompt,
            "context": "",
            "expected": str(output_val).strip(),
            "scoring": [],
            "image_path": "",
            "tier": 1,  # Output prediction = just run it
            "scoring_method": "exact_match",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "normalize": True,
            },
        }

    def _input_prompt(self, idx: int) -> dict:
        row = self._raw_dataset[idx]
        code = row.get("code", "")
        input_val = row.get("input", "")
        output_val = row.get("output", "")

        prompt = (
            f"Given the following Python code and its output, determine what "
            f"input was provided.\n\n"
            f"```python\n{code}\n```\n\n"
            f"Output: `{output_val}`\n\n"
            f"Give the exact input value inside <answer></answer> tags."
        )

        return {
            "id": f"cruxeval_input_{idx:04d}",
            "suite": "cruxeval",
            "prompt": prompt,
            "context": "",
            "expected": str(input_val).strip(),
            "scoring": [],
            "image_path": "",
            "tier": 2,  # Input prediction = harder reasoning
            "scoring_method": "exact_match",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "normalize": True,
            },
        }


# ── BigCodeBench (Multi-library coding) ──────────────────────────────────




class BigCodeBenchAdapter(BaseAdapter):
    """BigCodeBench: 1,140 coding tasks requiring 139 Python libraries.

    Source: bigcode/bigcodebench on HuggingFace (Apache 2.0).
    Scoring: code_execution (5.6 test cases per task, 99% branch coverage).

    Multi-library composition (pandas + matplotlib + scipy in one task) is
    where REPL + specialized coder >> direct frontdoor.
    """

    suite_name = "bigcodebench"

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset(
                "bigcode/bigcodebench", split="v0.1.2",
            )
        except Exception:
            try:
                import datasets as hf
                # Fallback to default split
                self._dataset = hf.load_dataset(
                    "bigcode/bigcodebench", split="default",
                )
            except Exception as e:
                print(f"  [adapter] BigCodeBench load failed: {e}")
                self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        task_id = row.get("task_id", f"bcb_{idx:04d}")
        instruct_prompt = row.get("instruct_prompt", "")
        complete_prompt = row.get("complete_prompt", "")
        test_code = row.get("test", "")
        _canonical = row.get("canonical_solution", "")
        entry_point = row.get("entry_point", "")
        libs = row.get("libs", [])

        # Use instruct prompt if available, else complete_prompt
        prompt_text = instruct_prompt or complete_prompt
        if not prompt_text:
            prompt_text = f"Implement the function `{entry_point}`."

        # Determine tier based on library complexity
        lib_count = len(libs) if isinstance(libs, list) else 0
        if lib_count >= 3:
            tier = 3  # Multi-library = hard
        elif lib_count >= 2:
            tier = 2
        else:
            tier = 1

        # Build test assertions from test field
        scoring_config: dict = {
            "language": "python",
            "timeout": 30,  # BigCodeBench tasks can be complex
        }
        if test_code:
            scoring_config["test_code"] = test_code
        elif entry_point:
            scoring_config["entry_point"] = entry_point

        return {
            "id": f"bcb_{task_id}",
            "suite": "bigcodebench",
            "prompt": prompt_text.strip(),
            "context": "",
            "expected": entry_point,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "code_execution",
            "scoring_config": scoring_config,
        }


# ── GPQA (Graduate-level Science) ─────────────────────────────────────────




class LiveCodeBenchAdapter(BaseAdapter):
    """LiveCodeBench: Competition programming problems from LeetCode.

    Source: greengerong/leetcode on HuggingFace (2,360 problems).
    Alternative to livecodebench/code_generation (deprecated loading script).

    Problems include difficulty tags and reference solutions.
    Ideal for REPL mode-advantage: iterative testing beats direct inference.

    Scoring: code_execution (against extracted test cases) or substring.
    Tiers: Based on LeetCode difficulty (Easy/Medium/Hard).
    """

    suite_name = "livecodebench"
    has_real_tiers = True

    # LeetCode difficulty mapping
    DIFFICULTY_MAP = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset(
                "greengerong/leetcode", split="train",
            )
        except Exception as e:
            print(f"  [adapter] LiveCodeBench (LeetCode) load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        difficulty = row.get("difficulty", "")
        if difficulty:
            difficulty_lower = str(difficulty).lower()
            return self.DIFFICULTY_MAP.get(difficulty_lower, 2)
        # Fallback: estimate from problem ID (later problems tend to be harder)
        problem_id = row.get("id", idx)
        if isinstance(problem_id, int) and problem_id > 1500:
            return 3
        elif isinstance(problem_id, int) and problem_id > 800:
            return 2
        return 1

    def _extract_test_cases(self, content: str) -> list[tuple[str, str]]:
        """Extract example test cases from problem content."""
        tests = []
        # Pattern: Input: X Output: Y (or similar variations)
        pattern = re.compile(
            r"(?:Input|Example[^:]*Input)[:\s]*`?([^`\n]+)`?\s*"
            r"(?:Output)[:\s]*`?([^`\n]+)`?",
            re.IGNORECASE | re.MULTILINE
        )
        for match in pattern.finditer(content):
            inp = match.group(1).strip()
            out = match.group(2).strip()
            if inp and out:
                tests.append((inp, out))
        return tests[:3]  # Limit to 3 examples

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        title = row.get("title", f"Problem {idx}")
        content = row.get("content", "")
        _difficulty = row.get("difficulty", "Medium")
        slug = row.get("slug", f"problem-{idx}")
        python_solution = row.get("python", "")

        # Clean HTML from content
        content_clean = re.sub(r"<[^>]+>", " ", content)
        content_clean = re.sub(r"\s+", " ", content_clean).strip()

        # Extract test cases
        test_cases = self._extract_test_cases(content)

        # Build prompt
        prompt_lines = [
            f"# {title}",
            "",
            content_clean,
            "",
        ]

        if test_cases:
            prompt_lines.append("### Examples:")
            for i, (inp, out) in enumerate(test_cases, 1):
                prompt_lines.append(f"Example {i}:")
                prompt_lines.append(f"  Input: {inp}")
                prompt_lines.append(f"  Output: {out}")
                prompt_lines.append("")

        prompt_lines.append(
            "Write a Python function to solve this problem. "
            "Include proper type hints and handle edge cases."
        )

        tier = self._get_tier_for_index(idx)

        # Build test code from extracted cases
        test_code = ""
        if test_cases and python_solution:
            # Try to extract function name from solution
            fn_match = re.search(r"def\s+(\w+)\s*\(", python_solution)
            if fn_match:
                fn_name = fn_match.group(1)
                test_code = f"# Test cases for {fn_name}\n"
                for inp, out in test_cases:
                    test_code += f"# assert {fn_name}({inp}) == {out}\n"

        # Determine scoring method based on content
        scoring_method = "code_execution" if test_code else "substring"
        scoring_config = {
            "language": "python",
            "timeout": 30,
        }
        if test_code:
            scoring_config["test_code"] = test_code
        else:
            # Fallback: check for function definition
            scoring_config["case_sensitive"] = True
            scoring_config["substring"] = "def "

        return {
            "id": f"leetcode_{slug}",
            "suite": "livecodebench",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": "def ",  # At minimum, expect a function
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": scoring_method,
            "scoring_config": scoring_config,
        }

    def sample(
        self, n: int = 10, seed: int = 42, stratify: bool = False,
        filter_difficulty: str | None = None,
    ) -> list[dict]:
        """Sample with optional difficulty filter.

        Args:
            n: Number of samples.
            seed: Random seed.
            stratify: Whether to stratify by tier.
            filter_difficulty: "easy", "medium", "hard", or None for all.
        """
        self._ensure_loaded()
        if not self._dataset:
            return []

        if filter_difficulty:
            # Filter by difficulty string
            target_tier = self.DIFFICULTY_MAP.get(filter_difficulty.lower(), 2)
            filtered_indices = [
                i for i in range(len(self._dataset))
                if self._get_tier_for_index(i) == target_tier
            ]
            rng = random.Random(seed)
            indices = rng.sample(filtered_indices, min(n, len(filtered_indices)))
            return [self._row_to_prompt(i, self._dataset[i]) for i in indices]

        # Default sampling
        if stratify and self.has_real_tiers:
            return self._stratified_sample(n, seed)

        rng = random.Random(seed)
        indices = rng.sample(range(len(self._dataset)), min(n, len(self._dataset)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]


# ── DebugBench (Bug Finding/Fixing) ───────────────────────────────────────




class DebugBenchAdapter(BaseAdapter):
    """DebugBench: 4,253 buggy code instances across 3 languages.

    Source: Rtian/DebugBench on HuggingFace.
    Contains buggy code with explanations, solutions, and bug categories.
    Perfect for REPL mode-advantage: iterative debugging >> direct inference.

    Scoring: code_execution (run fixed code against test cases).
    Tiers: easy=T1, medium=T2, hard=T3 (from LeetCode difficulty).
    """

    suite_name = "debugbench"
    has_real_tiers = True

    LEVEL_MAP = {"easy": 1, "medium": 2, "hard": 3}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset("Rtian/DebugBench", split="test")
        except Exception as e:
            print(f"  [adapter] DebugBench load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        level = row.get("level", "medium").lower()
        return self.LEVEL_MAP.get(level, 2)

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("question", "")
        buggy_code = row.get("buggy_code", "")
        solution = row.get("solution", "")
        bug_explanation = row.get("bug_explanation", "")
        examples = row.get("examples", [])
        constraints = row.get("constraints", "")
        language = row.get("language", "python3")
        _level = row.get("level", "medium")
        slug = row.get("slug", f"debug_{idx:04d}")
        category = row.get("category", "")

        # Map language to standard names
        lang_map = {"python3": "python", "cpp": "cpp", "java": "java"}
        lang = lang_map.get(language, language)

        # Build prompt
        prompt_lines = [
            f"# Bug Fixing Task ({lang.upper()})",
            "",
            "## Problem Description",
            question[:500] if len(question) > 500 else question,
            "",
        ]

        if examples:
            prompt_lines.append("## Examples")
            for i, ex in enumerate(examples[:2]):
                prompt_lines.append("```")
                prompt_lines.append(str(ex)[:200])
                prompt_lines.append("```")
            prompt_lines.append("")

        if constraints:
            prompt_lines.append("## Constraints")
            prompt_lines.append(constraints[:200])
            prompt_lines.append("")

        prompt_lines.extend([
            "## Buggy Code",
            f"```{lang}",
            buggy_code[:1000] if len(buggy_code) > 1000 else buggy_code,
            "```",
            "",
            "Find and fix the bug(s) in the code above. "
            "Provide the corrected code. "
            "Fix ONLY the bug — do NOT rewrite, rename variables, "
            "change data structures, or optimize. Keep the original code structure.",
        ])

        tier = self._get_tier_for_index(idx)

        # For Python, we can do code_execution scoring
        scoring_method = "code_execution" if lang == "python" else "substring"
        scoring_config = {"language": lang, "timeout": 30}

        if lang != "python":
            # For non-Python, check that key parts of solution appear
            scoring_config = {"case_sensitive": True}

        return {
            "id": f"debugbench_{slug}_{lang}",
            "suite": "debugbench",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": solution[:100] if solution else "def ",
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": scoring_method,
            "scoring_config": scoring_config,
            "metadata": {
                "language": lang,
                "category": category,
                "bug_explanation": bug_explanation[:200],
            },
        }

    def sample(
        self,
        n: int = 10,
        seed: int = 42,
        stratify: bool = False,
        filter_language: str | None = None,
        filter_category: str | None = None,
    ) -> list[dict]:
        """Sample with optional language/category filter.

        Args:
            n: Number of samples.
            seed: Random seed.
            stratify: Whether to stratify by tier.
            filter_language: "python3", "cpp", "java", or None for all.
            filter_category: Bug category filter or None for all.
        """
        self._ensure_loaded()
        if not self._dataset:
            return []

        # Apply filters
        filtered_indices = list(range(len(self._dataset)))

        if filter_language:
            filtered_indices = [
                i for i in filtered_indices
                if self._dataset[i].get("language", "") == filter_language
            ]

        if filter_category:
            filtered_indices = [
                i for i in filtered_indices
                if filter_category.lower() in self._dataset[i].get("category", "").lower()
            ]

        if not filtered_indices:
            return []

        rng = random.Random(seed)
        indices = rng.sample(filtered_indices, min(n, len(filtered_indices)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]


# ── USACO (Olympiad Programming) ──────────────────────────────────────────




class USACOAdapter(BaseAdapter):
    """USACO: Olympiad-level competitive programming problems.

    Source: codegenning/usacobench_formatted on HuggingFace.
    307 problems across Bronze/Silver/Gold/Platinum divisions.
    GPT-4 scores 8.7% zero-shot — ideal for REPL + specialist escalation.

    Scoring: code_execution (against test cases).
    Tiers: Bronze=T1, Silver=T2, Gold/Platinum=T3.
    """

    suite_name = "usaco"
    has_real_tiers = True

    DIVISION_MAP = {
        "bronze": 1,
        "silver": 2,
        "gold": 3,
        "platinum": 3,
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            # Use streaming to avoid timeout on large dataset
            self._dataset = hf.load_dataset(
                "codegenning/usacobench_formatted",
                split="test",
                streaming=False,
            )
        except Exception as e:
            print(f"  [adapter] USACO load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        division = row.get("division", row.get("level", "silver")).lower()
        return self.DIVISION_MAP.get(division, 2)

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        # Schema varies — try multiple field names
        problem = row.get("problem", row.get("question", row.get("prompt", "")))
        problem_id = row.get("problem_id", row.get("id", f"usaco_{idx:04d}"))
        division = row.get("division", row.get("level", "silver"))
        _solution = row.get("solution", row.get("code", ""))
        test_cases = row.get("test_cases", row.get("tests", []))

        # Build prompt
        prompt_lines = [
            f"# USACO Problem ({division.title()} Division)",
            "",
            problem[:2000] if len(problem) > 2000 else problem,
            "",
            "Write a Python solution that reads from stdin and writes to stdout.",
            "Your solution should handle all test cases within time limits.",
        ]

        tier = self._get_tier_for_index(idx)

        # Build test code if available
        test_code = ""
        if test_cases and isinstance(test_cases, list):
            test_cases_str = []
            for tc in test_cases[:5]:
                if isinstance(tc, dict):
                    inp = repr(tc.get("input", ""))
                    out = repr(tc.get("output", tc.get("expected", "")))
                    test_cases_str.append(f"({inp}, {out})")
            if test_cases_str:
                test_code = f"TEST_CASES = [{', '.join(test_cases_str)}]"

        return {
            "id": f"usaco_{division}_{problem_id}",
            "suite": "usaco",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": "",  # Code execution determines correctness
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "code_execution",
            "scoring_config": {
                "language": "python",
                "timeout": 120,  # USACO problems need more time
                "test_code": test_code,
            },
        }

    def sample(
        self,
        n: int = 10,
        seed: int = 42,
        stratify: bool = False,
        filter_division: str | None = None,
    ) -> list[dict]:
        """Sample with optional division filter.

        Args:
            n: Number of samples.
            seed: Random seed.
            stratify: Whether to stratify by tier.
            filter_division: "bronze", "silver", "gold", "platinum", or None.
        """
        self._ensure_loaded()
        if not self._dataset:
            return []

        if filter_division:
            filtered_indices = [
                i for i in range(len(self._dataset))
                if self._dataset[i].get("division", "").lower() == filter_division.lower()
            ]
            if not filtered_indices:
                return []
            rng = random.Random(seed)
            indices = rng.sample(filtered_indices, min(n, len(filtered_indices)))
            return [self._row_to_prompt(i, self._dataset[i]) for i in indices]

        if stratify and self.has_real_tiers:
            return self._stratified_sample(n, seed)

        rng = random.Random(seed)
        indices = rng.sample(range(len(self._dataset)), min(n, len(self._dataset)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]
