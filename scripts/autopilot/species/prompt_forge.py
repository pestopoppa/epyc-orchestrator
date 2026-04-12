"""Species 2 — PromptForge: LLM-guided prompt optimization.

Uses Claude CLI (Popen + session persistence) to analyze failure cases
and propose targeted prompt mutations on hot-swappable .md files.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("autopilot.prompt_forge")

import ast
import importlib
import subprocess as _subprocess  # avoid shadowing

ORCH_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ORCH_ROOT / "orchestration" / "prompts"
PROJECT_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")

# Meta-Harness Tier 2: Python files that code mutations may touch.
# This is the eval trust boundary — files NOT on this list are immutable.
CODE_MUTATION_ALLOWLIST = [
    "src/prompt_builders/resolver.py",      # Prompt resolution logic
    "src/escalation.py",                     # Escalation policy & retry logic
    "src/graph/escalation_helpers.py",       # Role cycle detection
    "src/tool_policy.py",                    # Tool access control rules
]

MUTATION_TYPES = [
    "targeted_fix",  # Fix specific failure patterns
    "compress",  # Reduce token count while maintaining behavior
    "few_shot_evolution",  # Add/remove/modify examples
    "crossover",  # Merge sections from two prompts
    "style_transfer",  # Apply patterns from one prompt to another
    "gepa",  # AP-19: GEPA evolutionary optimization (runs internal eval loop)
]


@dataclass
class PromptMutation:
    file: str  # e.g., "frontdoor.md"
    mutation_type: str
    description: str
    original_content: str = ""
    mutated_content: str = ""
    git_diff: str = ""
    accepted: bool = False


@dataclass
class CodeMutation:
    file: str  # Relative path, e.g. "src/escalation.py"
    mutation_type: str
    description: str
    original_content: str = ""
    mutated_content: str = ""
    git_diff: str = ""
    accepted: bool = False
    syntax_valid: bool = False


class PromptForge:
    """Species 2: LLM-guided prompt mutation and optimization."""

    def __init__(
        self,
        prompts_dir: Path | None = None,
        timeout: int = 300,
        auto_commit: bool = True,
    ):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.timeout = timeout
        self.auto_commit = auto_commit
        self._session_id: str | None = None

    def list_prompts(self) -> list[str]:
        """List all hot-swappable prompt files."""
        if not self.prompts_dir.exists():
            return []
        return sorted(f.name for f in self.prompts_dir.glob("*.md"))

    def read_prompt(self, filename: str) -> str:
        """Read a prompt file."""
        path = self.prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        return path.read_text()

    def write_prompt(self, filename: str, content: str) -> None:
        """Write a prompt file (picked up on next request)."""
        path = self.prompts_dir / filename
        path.write_text(content)
        log.info("Wrote prompt: %s (%d chars)", filename, len(content))

    def propose_mutation(
        self,
        target_file: str,
        mutation_type: str = "targeted_fix",
        failure_context: str = "",
        per_suite_quality: dict[str, float] | None = None,
        description: str = "",
        eval_tower=None,
        gepa_max_evals: int = 50,
    ) -> PromptMutation:
        """Propose a prompt mutation via Claude CLI or GEPA.

        When mutation_type="gepa", delegates to GEPA evolutionary optimization
        (AP-19). Requires eval_tower to be passed for orchestrator-based eval.

        Returns PromptMutation with the proposed changes.
        """
        if mutation_type not in MUTATION_TYPES:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        # AP-19: GEPA evolutionary optimization
        if mutation_type == "gepa":
            return self._propose_via_gepa(
                target_file=target_file,
                eval_tower=eval_tower,
                max_evals=gepa_max_evals,
                description=description,
            )

        original = self.read_prompt(target_file)

        prompt = self._build_mutation_prompt(
            target_file=target_file,
            mutation_type=mutation_type,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        result = self._invoke_claude(prompt)
        mutated_content = self._extract_mutation(result, original)

        mutation = PromptMutation(
            file=target_file,
            mutation_type=mutation_type,
            description=description or f"{mutation_type} on {target_file}",
            original_content=original,
            mutated_content=mutated_content,
        )
        return mutation

    def _propose_via_gepa(
        self,
        target_file: str,
        eval_tower=None,
        max_evals: int = 50,
        description: str = "",
    ) -> PromptMutation:
        """AP-19: Use GEPA evolutionary optimization to propose a mutation.

        Runs GEPA's reflective-mutation + Pareto-selection loop through the
        full orchestrator pipeline (eval_tower), returning the best candidate
        as a PromptMutation.
        """
        from .gepa_optimizer import GEPAPromptOptimizer

        if eval_tower is None:
            raise ValueError("gepa mutation requires eval_tower to be passed")

        optimizer = GEPAPromptOptimizer(
            eval_tower=eval_tower,
            prompt_forge=self,
        )
        result = optimizer.run(
            target_file=target_file,
            max_evals=max_evals,
        )

        if result is None:
            # GEPA failed — return a no-op mutation
            original = self.read_prompt(target_file)
            return PromptMutation(
                file=target_file,
                mutation_type="gepa",
                description="GEPA optimization failed — no mutation proposed",
                original_content=original,
                mutated_content=original,
            )

        return result.to_prompt_mutation()

    def apply_mutation(self, mutation: PromptMutation) -> dict[str, Any]:
        """Apply a mutation (write file + optional git commit)."""
        # Git snapshot before
        git_before = self._capture_git_state()

        # Write the mutated prompt
        self.write_prompt(mutation.file, mutation.mutated_content)

        # Git snapshot after
        git_after = self._capture_git_state()
        mutation.git_diff = self._diff_states(git_before, git_after)
        mutation.accepted = True

        if self.auto_commit and mutation.git_diff:
            self._git_commit(
                f"autopilot: {mutation.mutation_type} on {mutation.file}\n\n"
                f"{mutation.description}"
            )

        return {
            "status": "applied",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "diff_lines": len(mutation.git_diff.splitlines()),
        }

    def revert_mutation(self, mutation: PromptMutation) -> None:
        """Revert a mutation to original content and commit the revert."""
        self.write_prompt(mutation.file, mutation.original_content)
        mutation.accepted = False
        # Commit the revert so corrupted state is never the HEAD
        if self.auto_commit:
            self._git_commit(
                f"autopilot: revert prompt mutation on {mutation.file}\n\n"
                f"Reverted: {mutation.description}"
            )
        log.info("Reverted prompt mutation on %s (committed)", mutation.file)

    # ── Claude CLI invocation ────────────────────────────────────

    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude CLI following the claude_debugger pattern."""
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--allowedTools", "Read,Grep,Glob",
        ]
        if self._session_id:
            cmd.extend(["--resume", self._session_id])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)

            if proc.returncode != 0:
                log.error("Claude CLI failed (rc=%d): %s", proc.returncode, stderr[:500])
                return ""

            # Parse JSON response
            try:
                response = json.loads(stdout)
                # Capture session ID for resume
                if "session_id" in response:
                    self._session_id = response["session_id"]
                return response.get("result", stdout)
            except json.JSONDecodeError:
                return stdout

        except subprocess.TimeoutExpired:
            proc.kill()
            log.error("Claude CLI timed out after %ds", self.timeout)
            return ""
        except FileNotFoundError:
            log.error("Claude CLI not found. Ensure 'claude' is on PATH.")
            return ""

    # ── prompt building ──────────────────────────────────────────

    def _build_mutation_prompt(
        self,
        target_file: str,
        mutation_type: str,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> str:
        """Build the prompt for Claude CLI to propose a mutation."""
        lines = [
            f"You are an expert prompt engineer optimizing an LLM orchestration system.",
            f"",
            f"## Task: {mutation_type} mutation on `{target_file}`",
            f"",
        ]

        if description:
            lines.append(f"Goal: {description}\n")

        # Mutation type instructions
        type_instructions = {
            "targeted_fix": (
                "Analyze the failure cases below and make targeted edits to fix "
                "the specific failure patterns. Keep changes minimal and focused."
            ),
            "compress": (
                "Reduce the token count of this prompt while preserving its behavior. "
                "Remove redundant instructions, merge similar sections, use concise language."
            ),
            "few_shot_evolution": (
                "Improve the examples/few-shot demonstrations in this prompt. "
                "Add examples for underperforming suites, remove unhelpful ones."
            ),
            "crossover": (
                "Identify the strongest sections of this prompt and strengthen "
                "weaker sections by applying similar patterns."
            ),
            "style_transfer": (
                "Apply successful structural patterns (section organization, "
                "instruction phrasing, constraint framing) from high-performing "
                "prompts to this one."
            ),
        }
        lines.append(type_instructions.get(mutation_type, "Improve this prompt."))
        lines.append("")

        # Current prompt
        lines.append(f"## Current prompt ({target_file}):\n```markdown")
        lines.append(original_content)
        lines.append("```\n")

        # Failure context
        if failure_context:
            lines.append(f"## Recent failure cases:\n{failure_context}\n")

        # Per-suite quality
        if per_suite_quality:
            lines.append("## Per-suite quality (0-3 scale):")
            for suite, quality in sorted(per_suite_quality.items()):
                bar = "█" * int(quality) + "░" * (3 - int(quality))
                lines.append(f"  {suite}: {quality:.2f} {bar}")
            lines.append("")

        # Output format
        lines.append(
            "## Output format:\n"
            "Return the complete mutated prompt inside a ```markdown fenced block. "
            "Also include a brief explanation of your changes in a "
            "```json:autopilot_actions block:\n"
            '```json:autopilot_actions\n'
            '{"changes": ["change1", "change2"], "rationale": "..."}\n'
            "```"
        )

        return "\n".join(lines)

    def _extract_mutation(self, result: str, original: str) -> str:
        """Extract mutated prompt from Claude's response."""
        # Look for markdown fenced block
        if "```markdown" in result:
            start = result.index("```markdown") + len("```markdown")
            end = result.index("```", start)
            return result[start:end].strip()

        # Fallback: look for any fenced block that looks like a prompt
        if "```" in result:
            blocks = result.split("```")
            for i in range(1, len(blocks), 2):
                block = blocks[i]
                # Skip json blocks
                if block.strip().startswith(("json", "{")):
                    continue
                # Skip short blocks
                if len(block.strip()) > 100:
                    # Remove language tag if present
                    lines = block.strip().split("\n")
                    if lines[0].strip() in ("md", "markdown", "text"):
                        return "\n".join(lines[1:]).strip()
                    return block.strip()

        log.warning("Could not extract mutation from response, returning original")
        return original

    # ── git operations ───────────────────────────────────────────

    def _capture_git_state(self) -> dict[str, str]:
        """Capture git diff state of prompts directory."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", str(self.prompts_dir)],
                capture_output=True, text=True, timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            return {"diff_stat": result.stdout}
        except Exception:
            return {}

    def _diff_states(
        self, before: dict[str, str], after: dict[str, str]
    ) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", str(self.prompts_dir)],
                capture_output=True, text=True, timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            return result.stdout
        except Exception:
            return ""

    def _git_commit(self, message: str) -> None:
        try:
            subprocess.run(
                ["git", "add", str(self.prompts_dir)],
                timeout=10, check=True,
                cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                timeout=10, check=True,
                cwd=str(PROJECT_ROOT),
            )
            log.info("Committed prompt mutation")
        except Exception as e:
            log.warning("Git commit failed: %s", e)

    # ── Worktree-isolated mutations (AP-11) ────────────────────────

    def apply_mutation_isolated(
        self,
        mutation: PromptMutation,
        trial_name: str,
    ) -> "ExperimentContext":
        """Apply a prompt mutation in an isolated worktree.

        Returns an ExperimentContext. The caller must call ctx.accept() or
        ctx.reject() after evaluation. If neither is called, the context
        manager auto-rejects on cleanup.

        Usage:
            from scripts.autopilot.worktree_manager import WorktreeManager
            wt = WorktreeManager()
            with wt.experiment(trial_name) as ctx:
                forge.apply_mutation_in_context(ctx, mutation)
                result = tower.hybrid_eval()
                if result.quality > baseline:
                    ctx.accept(f"autopilot: {mutation.mutation_type} on {mutation.file}")
                else:
                    ctx.reject()
        """
        from scripts.autopilot.worktree_manager import WorktreeManager
        wt = WorktreeManager(PROJECT_ROOT)
        return wt.experiment(trial_name)

    def apply_mutation_in_context(
        self,
        ctx: Any,
        mutation: "PromptMutation",
    ) -> dict[str, Any]:
        """Apply a prompt mutation within an experiment context.

        The context handles file backup, worktree versioning, and
        copying the mutated file to the main repo for live eval.
        """
        rel_path = f"orchestration/prompts/{mutation.file}"
        ctx.apply_file(rel_path, mutation.mutated_content)
        mutation.accepted = True
        return {
            "status": "applied_isolated",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "worktree": str(ctx.worktree_path),
        }

    def apply_code_mutation_in_context(
        self,
        ctx: Any,
        mutation: "CodeMutation",
    ) -> dict[str, Any]:
        """Apply a code mutation within an experiment context."""
        if not mutation.syntax_valid:
            return {"status": "rejected", "reason": "syntax_invalid"}
        ctx.apply_file(mutation.file, mutation.mutated_content)
        mutation.accepted = True
        return {
            "status": "applied_isolated",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "worktree": str(ctx.worktree_path),
        }

    def summary(self) -> dict[str, Any]:
        """Summary for controller."""
        prompts = self.list_prompts()
        return {
            "available_prompts": prompts,
            "n_prompts": len(prompts),
            "session_active": self._session_id is not None,
            "mutation_types": MUTATION_TYPES,
            "code_mutation_targets": CODE_MUTATION_ALLOWLIST,
        }

    # ── Meta-Harness Tier 2: Code mutations ──────────────────────

    def propose_code_mutation(
        self,
        target_file: str,
        mutation_type: str = "targeted_fix",
        failure_context: str = "",
        per_suite_quality: dict[str, float] | None = None,
        description: str = "",
    ) -> CodeMutation:
        """Propose a mutation to a Python code file (Tier 2 search space).

        Only files in CODE_MUTATION_ALLOWLIST may be mutated.
        """
        if target_file not in CODE_MUTATION_ALLOWLIST:
            raise ValueError(
                f"Code mutation blocked: {target_file} not in allowlist. "
                f"Allowed: {CODE_MUTATION_ALLOWLIST}"
            )

        abs_path = PROJECT_ROOT / target_file
        if not abs_path.exists():
            raise FileNotFoundError(f"Target file not found: {abs_path}")

        original = abs_path.read_text()

        prompt = self._build_code_mutation_prompt(
            target_file=target_file,
            mutation_type=mutation_type,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        result = self._invoke_claude(prompt)
        mutated_content = self._extract_code_mutation(result, original)

        mutation = CodeMutation(
            file=target_file,
            mutation_type=mutation_type,
            description=description or f"{mutation_type} on {target_file}",
            original_content=original,
            mutated_content=mutated_content,
        )

        # Deep validation: syntax + shrinkage + public names + import test
        valid, reason = self._validate_code_mutation(original, mutated_content, target_file)
        mutation.syntax_valid = valid
        if not valid:
            log.warning("Code mutation rejected (%s): %s", target_file, reason)
            mutation.mutated_content = original

        return mutation

    def apply_code_mutation(self, mutation: CodeMutation) -> dict[str, Any]:
        """Apply a code mutation with syntax validation + git safety."""
        if not mutation.syntax_valid:
            return {"status": "rejected", "reason": "syntax_invalid"}

        abs_path = PROJECT_ROOT / mutation.file

        # Git commit current state before mutation (safety net)
        try:
            subprocess.run(
                ["git", "add", str(abs_path)],
                timeout=10, cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                ["git", "commit", "-m",
                 f"autopilot: pre-code-mutation checkpoint ({mutation.file})"],
                timeout=10, cwd=str(PROJECT_ROOT),
                capture_output=True,
            )
        except Exception:
            pass  # Commit may fail if no changes — that's OK

        # Write the mutated code
        abs_path.write_text(mutation.mutated_content)
        mutation.accepted = True

        # Capture diff
        try:
            result = subprocess.run(
                ["git", "diff", str(abs_path)],
                capture_output=True, text=True, timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            mutation.git_diff = result.stdout
        except Exception:
            mutation.git_diff = ""

        if self.auto_commit and mutation.git_diff:
            self._git_commit_file(
                abs_path,
                f"autopilot: code {mutation.mutation_type} on {mutation.file}\n\n"
                f"{mutation.description}",
            )

        return {
            "status": "applied",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "diff_lines": len(mutation.git_diff.splitlines()),
        }

    def revert_code_mutation(self, mutation: CodeMutation) -> None:
        """Revert a code mutation to original content and commit the revert."""
        abs_path = PROJECT_ROOT / mutation.file
        abs_path.write_text(mutation.original_content)
        mutation.accepted = False
        # Commit the revert so corrupted state is never the HEAD
        if self.auto_commit:
            self._git_commit_file(
                abs_path,
                f"autopilot: revert code mutation on {mutation.file}\n\n"
                f"Reverted: {mutation.description}",
            )
        log.info("Reverted code mutation on %s (committed)", mutation.file)

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax via ast.parse."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            log.warning("Syntax error in mutated code: %s", e)
            return False

    def _validate_code_mutation(
        self, original: str, mutated: str, target_file: str
    ) -> tuple[bool, str]:
        """Deep validation of a code mutation beyond syntax.

        Returns (valid, reason). Checks:
        1. Syntax (ast.parse)
        2. No catastrophic size reduction (>60% shrinkage)
        3. Public names preserved (classes, functions defined at module level)
        4. Import test (actually importable, no circular imports)
        """
        # 1. Syntax
        try:
            mutated_tree = ast.parse(mutated)
        except SyntaxError as e:
            return False, f"syntax error: {e}"

        # 2. Catastrophic shrinkage — reject if >60% of lines removed
        orig_lines = len(original.splitlines())
        new_lines = len(mutated.splitlines())
        if orig_lines > 10 and new_lines < orig_lines * 0.4:
            return False, (
                f"catastrophic shrinkage: {orig_lines}→{new_lines} lines "
                f"({100 * (1 - new_lines / orig_lines):.0f}% removed)"
            )

        # 3. Public names preserved — every class/function at module level
        #    in the original must still exist in the mutated version
        def _top_level_names(tree: ast.AST) -> set[str]:
            names = set()
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
            return names

        orig_tree = ast.parse(original)
        orig_names = _top_level_names(orig_tree)
        new_names = _top_level_names(mutated_tree)
        missing = orig_names - new_names
        if missing:
            return False, f"missing public names: {missing}"

        # 4. Import test — write to temp, try importing
        abs_path = PROJECT_ROOT / target_file
        try:
            # Temporarily write mutated code
            backup = abs_path.read_text()
            abs_path.write_text(mutated)
            try:
                module_name = target_file.replace("/", ".").removesuffix(".py")
                # Clear any cached version
                import sys
                if module_name in sys.modules:
                    del sys.modules[module_name]
                importlib.import_module(module_name)
            except Exception as e:
                abs_path.write_text(backup)
                return False, f"import failed: {e}"
            finally:
                # Always restore original before returning
                abs_path.write_text(backup)
        except Exception as e:
            return False, f"validation IO error: {e}"

        return True, "ok"

    def _build_code_mutation_prompt(
        self,
        target_file: str,
        mutation_type: str,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> str:
        """Build prompt for code mutation."""
        lines = [
            "You are an expert Python engineer optimizing an LLM orchestration system.",
            "",
            f"## Task: {mutation_type} mutation on `{target_file}`",
            "",
        ]

        if description:
            lines.append(f"Goal: {description}\n")

        type_instructions = {
            "targeted_fix": (
                "Analyze the failure cases below and make targeted edits to fix "
                "specific failure patterns. Keep changes minimal and focused. "
                "Do NOT refactor or add features beyond what's needed to fix the issue."
            ),
            "compress": (
                "Reduce complexity while preserving behavior. Remove dead code, "
                "simplify conditionals, merge redundant branches."
            ),
        }
        lines.append(type_instructions.get(mutation_type, "Improve this code with minimal changes."))
        lines.append("")

        lines.append(f"## Current code (`{target_file}`):\n```python")
        lines.append(original_content)
        lines.append("```\n")

        if failure_context:
            lines.append(f"## Context (failures, traces, insights):\n{failure_context}\n")

        if per_suite_quality:
            lines.append("## Per-suite quality (0-3 scale):")
            for suite, quality in sorted(per_suite_quality.items()):
                bar = "█" * int(quality) + "░" * (3 - int(quality))
                lines.append(f"  {suite}: {quality:.2f} {bar}")
            lines.append("")

        lines.append(
            "## IMPORTANT CONSTRAINTS:\n"
            "1. Return the COMPLETE modified file in a ```python fenced block\n"
            "2. Do NOT change function signatures or class names\n"
            "3. Do NOT add new dependencies\n"
            "4. Keep changes minimal — one logical change only\n"
            "5. The code must pass ast.parse() (valid Python syntax)\n"
        )

        lines.append(
            "## Output format:\n"
            "Return the complete modified file inside a ```python fenced block."
        )

        return "\n".join(lines)

    def _extract_code_mutation(self, result: str, original: str) -> str:
        """Extract mutated Python code from Claude's response."""
        if "```python" in result:
            start = result.index("```python") + len("```python")
            end = result.index("```", start)
            return result[start:end].strip()

        if "```" in result:
            blocks = result.split("```")
            for i in range(1, len(blocks), 2):
                block = blocks[i]
                if block.strip().startswith(("json", "{")):
                    continue
                if len(block.strip()) > 100:
                    lines = block.strip().split("\n")
                    if lines[0].strip() in ("python", "py"):
                        return "\n".join(lines[1:]).strip()
                    return block.strip()

        log.warning("Could not extract code mutation from response, returning original")
        return original

    def _git_commit_file(self, path: Path, message: str) -> None:
        """Git add + commit a specific file."""
        try:
            subprocess.run(
                ["git", "add", str(path)],
                timeout=10, check=True,
                cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                timeout=10, check=True,
                cwd=str(PROJECT_ROOT),
            )
            log.info("Committed code mutation: %s", path.name)
        except Exception as e:
            log.warning("Git commit failed: %s", e)
