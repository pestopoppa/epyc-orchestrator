"""Species 2 — PromptForge: LLM-guided prompt optimization.

Uses Claude CLI (Popen + session persistence) to analyze failure cases
and propose targeted prompt mutations on hot-swappable .md files.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger("autopilot.prompt_forge")

if TYPE_CHECKING:
    from scripts.autopilot.worktree_manager import ExperimentContext

import ast
import importlib
import math

ORCH_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ORCH_ROOT / "orchestration" / "prompts"
PROJECT_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")

# Meta-Harness Tier 2: Python files that code mutations may touch.
# This is the eval trust boundary — files NOT on this list are immutable.
CODE_MUTATION_ALLOWLIST = [
    "src/prompt_builders/resolver.py",  # Prompt resolution logic
    "src/escalation.py",  # Escalation policy & retry logic
    "src/graph/escalation_helpers.py",  # Role cycle detection
    "src/tool_policy.py",  # Tool access control rules
    "src/api/routes/chat.py",  # Chat pipeline (cheap-first, routing, response)
]

# New-file code mutations are more permissive than the existing-file allowlist,
# but they stay directory-scoped. ``src/`` is for ordinary code scaffolds;
# ``schema_evolution/`` is the AutoMem/MH-9 lane for default-inert memory
# schema/scaffold proposals.
NEW_FILE_MUTATION_ROOT = PROJECT_ROOT / "src"
MEMORY_SCHEMA_MUTATION_ROOT = (
    PROJECT_ROOT / "orchestration" / "repl_memory" / "schema_evolution"
)

MUTATION_TYPES = [
    "targeted_fix",  # Fix specific failure patterns
    "compress",  # Reduce token count while maintaining behavior
    "few_shot_evolution",  # Add/remove/modify examples
    "crossover",  # Merge sections from two prompts
    "style_transfer",  # Apply patterns from one prompt to another
    "gepa",  # AP-19: GEPA evolutionary optimization (runs internal eval loop)
]

_MIN_VALIDATION_TRIALS = 5
_SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "aime": ("aime",),
    "coder": ("coder", "humaneval", "mbpp"),
    "cruxeval": ("cruxeval", "crux eval"),
    "debugbench": ("debugbench", "debug bench"),
    "gpqa": ("gpqa",),
    "gsm8k": ("gsm8k",),
    "hotpotqa": ("hotpotqa", "hotpot qa"),
    "livecodebench": ("livecodebench", "live code bench", "lcb"),
    "math": ("math",),
    "skill_transfer": ("skill_transfer", "skill transfer"),
    "thinking": ("thinking",),
    "usaco": ("usaco",),
}
_SUITE_TERM_TO_CANONICAL = {
    term: canonical for canonical, aliases in _SUITE_ALIASES.items() for term in aliases
}
_SUITE_TERM_RE = re.compile(
    r"(?<![\w-])("
    + "|".join(re.escape(term) for term in sorted(_SUITE_TERM_TO_CANONICAL, key=len, reverse=True))
    + r")(?![\w-])",
    re.IGNORECASE,
)
_TRIAL_REF_RE = re.compile(r"(?:\btrial\s*#?\s*|\[?t)(\d+)\]?", re.IGNORECASE)
_UNIVERSAL_TRANSFER_RE = re.compile(
    r"\b(always|never|universally|global(?:ly)?|all\s+(?:tasks|prompts|suites|benchmarks)|"
    r"every\s+(?:task|prompt|suite|benchmark))\b",
    re.IGNORECASE,
)
_FRONTDOOR_REQUIRED_MARKERS = (
    "# Front Door Orchestrator",
    "TaskIR mode",
    "Direct-answer mode",
    "Answer tags (scoped)",
)
_FRONTDOOR_CORRUPTION_MARKERS = (
    "fenced block from my response",
    "i should **not** edit the file directly",
    "one note worth flagging",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _coverage_retrieve(
    strategy_store: Any,
    query_text: str,
    *,
    journal: Any | None,
    k: int,
    species: str | None,
) -> list[Any]:
    if journal is not None and hasattr(strategy_store, "retrieve_for_journal"):
        try:
            return list(
                strategy_store.retrieve_for_journal(
                    query_text,
                    journal=journal,
                    k=k,
                    species=species,
                )
            )
        except TypeError:
            return list(
                strategy_store.retrieve_for_journal(
                    query_text,
                    journal=journal,
                    k=k,
                )
            )

    if hasattr(strategy_store, "retrieve"):
        try:
            return list(strategy_store.retrieve(query_text, k=k, species=species))
        except TypeError:
            return list(strategy_store.retrieve(query_text, k=k))

    raise AttributeError("strategy_store has neither retrieve_for_journal() nor retrieve()")


def diversity_coverage_penalty(
    query_text: str,
    strategy_store: Any | None,
    *,
    journal: Any | None = None,
    k: int = 8,
    species: str | None = "prompt_forge",
    min_density: float = 1e-6,
) -> dict[str, Any]:
    """Estimate mutation-neighborhood density from StrategyStore retrieval.

    AP-35 uses the existing strategy-memory index as an observe-only density
    proxy. ``negative_log_density`` is high in sparse neighborhoods and low
    near already-covered strategy clusters; callers decide how to present it.
    """
    query = str(query_text or "").strip()
    if strategy_store is None:
        return {
            "status": "unavailable",
            "reason": "missing_strategy_store",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }
    if not query:
        return {
            "status": "unavailable",
            "reason": "empty_query",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }

    k = max(1, int(k))
    floor = max(_safe_float(min_density, 1e-6), 1e-12)
    try:
        entries = _coverage_retrieve(
            strategy_store,
            query,
            journal=journal,
            k=k,
            species=species,
        )
    except Exception as exc:  # noqa: BLE001 - density hints must not block mutation dispatch
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }

    top_matches: list[dict[str, Any]] = []
    scores: list[float] = []
    for entry in entries:
        score = max(0.0, _safe_float(getattr(entry, "similarity_score", 0.0)))
        scores.append(score)
        top_matches.append(
            {
                "id": str(getattr(entry, "id", "") or ""),
                "source_trial_id": getattr(entry, "source_trial_id", None),
                "species": str(getattr(entry, "species", "") or ""),
                "description": str(getattr(entry, "description", "") or ""),
                "insight": str(
                    getattr(entry, "generalized_content", "") or getattr(entry, "insight", "") or ""
                ),
                "similarity_score": score,
            }
        )

    if scores:
        density = sum(scores) / len(scores)
        status = "ok"
    else:
        density = 0.0
        status = "sparse"

    negative_log_density = -math.log(max(density, floor))
    return {
        "status": status,
        "reason": "ok" if scores else "no_nearby_strategy_entries",
        "query_text": query,
        "density": density,
        "negative_log_density": negative_log_density,
        "penalty": negative_log_density,
        "similar_count": len(scores),
        "top_matches": top_matches[:k],
        "interpretation": (
            "Higher negative_log_density means the mutation target is less covered "
            "by strategy memory; use as exploration pressure, not as an acceptance gate."
        ),
    }


@dataclass(frozen=True)
class TransferSafetyVerdict:
    valid: bool
    reason: str = "ok"
    warnings: tuple[str, ...] = ()
    source_suites: tuple[str, ...] = ()
    introduced_suites: tuple[str, ...] = ()
    evidence_trial_count: int = 0


@dataclass
class PromptMutation:
    file: str  # e.g., "frontdoor.md"
    mutation_type: str
    description: str
    original_content: str = ""
    mutated_content: str = ""
    git_diff: str = ""
    accepted: bool = False
    safety_valid: bool = True
    safety_reason: str = "ok"
    safety_warnings: list[str] = field(default_factory=list)


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
    safety_valid: bool = True
    safety_reason: str = "ok"
    safety_warnings: list[str] = field(default_factory=list)


def _resolve_code_mutation_target(target_file: str) -> Path:
    """Resolve a code-mutation target while rejecting traversal and escapes."""
    requested = Path(target_file)
    if requested.is_absolute() or ".." in requested.parts:
        raise FileNotFoundError(f"Target file not found: {PROJECT_ROOT / target_file}")
    resolved = (PROJECT_ROOT / requested).resolve(strict=False)
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise FileNotFoundError(f"Target file not found: {PROJECT_ROOT / target_file}")
    return resolved


def new_file_mutation_roots() -> tuple[Path, ...]:
    """Directory roots where MH-9 may create brand-new Python modules."""
    return (NEW_FILE_MUTATION_ROOT, MEMORY_SCHEMA_MUTATION_ROOT)


def new_file_mutation_root_labels() -> tuple[str, ...]:
    """Planner-facing labels for sanctioned new-file mutation roots."""
    labels: list[str] = []
    for root in new_file_mutation_roots():
        try:
            labels.append(str(root.relative_to(PROJECT_ROOT)))
        except ValueError:
            labels.append(str(root))
    return tuple(labels)


def _is_under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _is_memory_schema_evolution_target(path: Path) -> bool:
    return path.is_relative_to(MEMORY_SCHEMA_MUTATION_ROOT)


def _suite_mentions(text: str) -> set[str]:
    return {
        _SUITE_TERM_TO_CANONICAL[match.group(1).lower()]
        for match in _SUITE_TERM_RE.finditer(text or "")
    }


def _added_text(original: str, mutated: str) -> str:
    original_lines = {line.strip() for line in original.splitlines() if line.strip()}
    return "\n".join(
        line.strip()
        for line in mutated.splitlines()
        if line.strip() and line.strip() not in original_lines
    )


def _trial_reference_count(text: str) -> int:
    return len({int(match) for match in _TRIAL_REF_RE.findall(text or "")})


def _prompt_integrity_reason(filename: str, content: str) -> str | None:
    """Return a rejection reason for prompt text known to be structurally corrupt."""
    if filename != "frontdoor.md":
        return None
    lowered = content.lower()
    for marker in _FRONTDOOR_CORRUPTION_MARKERS:
        if marker in lowered:
            return f"frontdoor_corruption_marker:{marker}"
    missing = [marker for marker in _FRONTDOOR_REQUIRED_MARKERS if marker not in content]
    if missing:
        return "frontdoor_missing_required_markers:" + ",".join(missing)
    return None


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
        """List all hot-swappable prompt files (flat + roles/ subdirectory)."""
        if not self.prompts_dir.exists():
            return []
        return sorted(f.name for f in self.prompts_dir.rglob("*.md"))

    def _resolve_prompt_path(self, filename: str) -> Path:
        """Resolve prompt file, searching multiple locations."""
        root = self.prompts_dir.resolve()
        requested = Path(filename)
        path = self.prompts_dir / filename
        if requested.is_absolute() or ".." in requested.parts:
            raise FileNotFoundError(f"Prompt not found: {path}")

        def safe_existing(candidate: Path) -> Path | None:
            if not candidate.exists():
                return None
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                return None
            return resolved

        # Try exact path first (handles roles/worker_explore.md from controller)
        resolved_path = safe_existing(path)
        if resolved_path is not None:
            return resolved_path
        # Try roles/ subdirectory (flat filename like worker_explore.md)
        roles_path = self.prompts_dir / "roles" / filename
        resolved_roles_path = safe_existing(roles_path)
        if resolved_roles_path is not None:
            return resolved_roles_path
        # Try stripping roles/ prefix if controller included it redundantly
        basename = requested.name
        if basename != filename:
            for candidate in [self.prompts_dir / basename, self.prompts_dir / "roles" / basename]:
                resolved_candidate = safe_existing(candidate)
                if resolved_candidate is not None:
                    return resolved_candidate
        raise FileNotFoundError(f"Prompt not found: {path}")

    def read_prompt(self, filename: str) -> str:
        """Read a prompt file."""
        return self._resolve_prompt_path(filename).read_text()

    def write_prompt(self, filename: str, content: str) -> None:
        """Write a prompt file (picked up on next request)."""
        path = self._resolve_prompt_path(filename)
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
        integrity_reason = _prompt_integrity_reason(target_file, mutated_content)
        if integrity_reason:
            mutation.safety_valid = False
            mutation.safety_reason = "prompt_integrity:" + integrity_reason
            mutation.mutated_content = original
            log.warning(
                "Prompt mutation rejected by integrity guard (%s): %s",
                target_file,
                integrity_reason,
            )
            return mutation
        self._attach_transfer_safety(
            mutation,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )
        if not mutation.safety_valid:
            log.warning(
                "Prompt mutation rejected by transfer safety (%s): %s",
                target_file,
                mutation.safety_reason,
            )
            mutation.mutated_content = original
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

        mutation = result.to_prompt_mutation()
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            mutation.safety_valid = False
            mutation.safety_reason = "prompt_integrity:" + integrity_reason
            mutation.mutated_content = mutation.original_content
            log.warning(
                "GEPA prompt mutation rejected by integrity guard (%s): %s",
                mutation.file,
                integrity_reason,
            )
        return mutation

    def apply_mutation(self, mutation: PromptMutation) -> dict[str, Any]:
        """Apply a mutation (write file + optional git commit)."""
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected mutation: {integrity_reason}")

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
                f"autopilot: {mutation.mutation_type} on {mutation.file}\n\n{mutation.description}"
            )

        return {
            "status": "applied",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "diff_lines": len(mutation.git_diff.splitlines()),
        }

    def revert_mutation(self, mutation: PromptMutation) -> None:
        """Revert a mutation to original content and commit the revert."""
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.original_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected revert: {integrity_reason}")

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
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--allowedTools",
            "Read,Grep,Glob",
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
            "You are an expert prompt engineer optimizing an LLM orchestration system.",
            "",
            f"## Task: {mutation_type} mutation on `{target_file}`",
            "",
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

        lines.append(self._negative_transfer_safety_block())
        lines.append("")

        # Output format
        lines.append(
            "## Output format:\n"
            "Return the complete mutated prompt inside a ```markdown fenced block. "
            "Also include a brief explanation of your changes in a "
            "```json:autopilot_actions block:\n"
            "```json:autopilot_actions\n"
            '{"changes": ["change1", "change2"], "rationale": "..."}\n'
            "```"
        )

        return "\n".join(lines)

    def _extract_mutation(self, result: str, original: str) -> str:
        """Extract mutated prompt from Claude's response.

        Only fences whose opening backticks begin a line are treated as block
        delimiters. This prevents an inline fenced-code mention inside the
        reply's *prose* (e.g. a sentence quoting ``result.index(...)``) from
        being mis-captured as the payload — a bug that overwrote a prompt file
        with the model's prose and committed it.
        """
        # Line-anchored fenced blocks: the opening backticks (with optional
        # language tag) must start a line; the body runs to the next line that
        # starts with a fence, or end-of-string.
        fence = re.compile(
            r"^[ \t]*`{3,}[ \t]*([\w:.\-]*)[ \t]*\r?\n(.*?)(?:\r?\n[ \t]*`{3,}|\Z)",
            re.DOTALL | re.MULTILINE,
        )
        blocks = [(tag.strip().lower(), body) for tag, body in fence.findall(result)]

        # Prefer an explicitly prose/markdown-tagged block.
        for tag, body in blocks:
            if tag in ("markdown", "md", "text"):
                return body.strip()

        # Fallback: the largest block that is not a json/actions or object block.
        candidates = [
            body
            for tag, body in blocks
            if not tag.startswith("json")
            and not body.lstrip().startswith("{")
            and len(body.strip()) > 100
        ]
        if candidates:
            return max(candidates, key=lambda b: len(b.strip())).strip()

        log.warning("Could not extract mutation from response, returning original")
        return original

    # ── git operations ───────────────────────────────────────────

    def _capture_git_state(self) -> dict[str, str]:
        """Capture git diff state of prompts directory."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", str(self.prompts_dir)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            return {"diff_stat": result.stdout}
        except Exception:
            return {}

    def _diff_states(self, before: dict[str, str], after: dict[str, str]) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", str(self.prompts_dir)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            return result.stdout
        except Exception:
            return ""

    def _git_commit(self, message: str) -> None:
        try:
            subprocess.run(
                ["git", "add", str(self.prompts_dir)],
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                timeout=10,
                check=True,
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
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected isolated mutation: {integrity_reason}")

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
        if mutation_type not in {"targeted_fix", "compress", "new_file"}:
            raise ValueError(f"Unknown code mutation type: {mutation_type}")

        abs_path = _resolve_code_mutation_target(target_file)
        if mutation_type == "new_file":
            roots = new_file_mutation_roots()
            if not _is_under_any(abs_path.parent, roots):
                raise ValueError(
                    f"New-file mutation blocked: {target_file} must stay under "
                    f"one of {', '.join(new_file_mutation_root_labels())}"
                )
            if not abs_path.parent.exists():
                raise FileNotFoundError(f"New-file parent directory not found: {abs_path.parent}")
            if abs_path.exists():
                raise FileExistsError(f"New-file mutation blocked: {abs_path} already exists")
            original = ""
        else:
            if target_file not in CODE_MUTATION_ALLOWLIST:
                raise ValueError(
                    f"Code mutation blocked: {target_file} not in allowlist. "
                    f"Allowed: {CODE_MUTATION_ALLOWLIST}"
                )
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
        self._attach_transfer_safety(
            mutation,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        # Deep validation: syntax + shrinkage + public names + import test
        valid, reason = self._validate_code_mutation(
            original,
            mutated_content,
            target_file,
            is_new_file=(mutation_type == "new_file"),
        )
        mutation.syntax_valid = valid
        if not valid:
            log.warning("Code mutation rejected (%s): %s", target_file, reason)
            mutation.mutated_content = original
        if not mutation.safety_valid:
            log.warning(
                "Code mutation rejected by transfer safety (%s): %s",
                target_file,
                mutation.safety_reason,
            )
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
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"autopilot: pre-code-mutation checkpoint ({mutation.file})",
                ],
                timeout=10,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
            )
        except Exception:
            pass  # Commit may fail if no changes — that's OK

        # Write the mutated code
        abs_path.write_text(mutation.mutated_content)
        mutation.accepted = True

        # Capture diff
        try:
            if mutation.mutation_type == "new_file" and not mutation.original_content:
                result = subprocess.run(
                    ["git", "diff", "--no-index", "--", "/dev/null", str(abs_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(PROJECT_ROOT),
                )
            else:
                result = subprocess.run(
                    ["git", "diff", str(abs_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
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
        if mutation.mutation_type == "new_file" and not mutation.original_content:
            abs_path.unlink(missing_ok=True)
        else:
            abs_path.write_text(mutation.original_content)
        mutation.accepted = False
        # Commit the revert so corrupted state is never the HEAD
        if self.auto_commit:
            if mutation.mutation_type == "new_file" and not mutation.original_content:
                try:
                    subprocess.run(
                        ["git", "add", "-A", str(abs_path)],
                        timeout=10,
                        check=True,
                        cwd=str(PROJECT_ROOT),
                    )
                    subprocess.run(
                        [
                            "git",
                            "commit",
                            "-m",
                            f"autopilot: revert code mutation on {mutation.file}\n\n"
                            f"Reverted: {mutation.description}",
                        ],
                        timeout=10,
                        check=True,
                        cwd=str(PROJECT_ROOT),
                    )
                except Exception as e:
                    log.warning("Git commit failed: %s", e)
            else:
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
        self,
        original: str,
        mutated: str,
        target_file: str,
        *,
        is_new_file: bool = False,
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
        if not is_new_file and orig_lines > 10 and new_lines < orig_lines * 0.4:
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

        if not is_new_file:
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
            backup_exists = abs_path.exists()
            backup = abs_path.read_text() if backup_exists else ""
            abs_path.write_text(mutated)
            try:
                module_name = target_file.replace("/", ".").removesuffix(".py")
                # Clear any cached version
                import sys

                if module_name in sys.modules:
                    del sys.modules[module_name]
                importlib.import_module(module_name)
            except Exception as e:
                if backup_exists:
                    abs_path.write_text(backup)
                else:
                    abs_path.unlink(missing_ok=True)
                return False, f"import failed: {e}"
            finally:
                # Always restore original before returning
                if backup_exists:
                    abs_path.write_text(backup)
                else:
                    abs_path.unlink(missing_ok=True)
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
            "new_file": (
                "Create a new Python module at the requested path. Keep it "
                "directory-scoped, minimal, and self-contained. Do not alter "
                "existing files. Allowed roots: "
                f"{', '.join(new_file_mutation_root_labels())}."
            ),
        }
        lines.append(
            type_instructions.get(mutation_type, "Improve this code with minimal changes.")
        )
        lines.append("")

        if mutation_type == "new_file" and not original_content.strip():
            lines.append(
                f"## Current code (`{target_file}`):\n"
                "(This file does not exist yet. Create it from scratch.)\n```python"
            )
        else:
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
            "## Proposer-prior contract (MH-6):\n"
            "Read inputs in this order and do not skip ahead:\n"
            "1. Failed traces and recent regressions in the context above.\n"
            "2. Current frontier or accepted behavior implied by the existing code.\n"
            "3. Strategy-store or prior-mutation notes present in the context.\n"
            "4. The operator request / mutation goal.\n"
            "\n"
            "Before proposing code, estimate:\n"
            "- expected_quality_delta: signed expected quality change on the cited "
            "failure surface; use a small numeric value and say when evidence is weak.\n"
            "- expected_cost_delta: signed expected runtime/token/complexity change; "
            "use 0.0 when the change should be behavior-only.\n"
            "\n"
            "no-task-specific-hints: do not hard-code benchmark IDs, exact prompts, "
            "known answers, or dataset-specific shortcuts. Generalize only from "
            "observable failure mechanisms."
        )
        lines.append("")

        try:
            target_abs = _resolve_code_mutation_target(target_file)
        except FileNotFoundError:
            target_abs = PROJECT_ROOT / target_file
        if mutation_type == "new_file" and _is_memory_schema_evolution_target(target_abs):
            lines.append(
                "## AutoMem memory schema-evolution contract (MH-9/P2):\n"
                "- Create a default-inert schema/scaffold module for "
                "`MemoryAction` / `MemoryActionStore`; importing it must not "
                "write files, start subprocesses, call inference, or touch the "
                "trace store.\n"
                "- Express schema-evolution moves as prompt-free helpers, "
                "contracts, constants, or pure validators over "
                "APPEND/CREATE/UPSERT and the status/inventory/strategy/plan/log "
                "channels.\n"
                "- Do not change SafetyGate, Pareto admission, eval scoring, "
                "blacklists, thresholds, planner spend-breaker flags, or live "
                "runtime behavior.\n"
                "- Keep exports narrow and include explicit blockers when "
                "calibration, process, or validation evidence is missing."
            )
            lines.append("")

        lines.append(self._negative_transfer_safety_block())
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
            "Return the complete modified file inside a ```python fenced block first. "
            "Then include a ```json:autopilot_actions block with keys "
            "`expected_quality_delta`, `expected_cost_delta`, `read_order_used`, "
            "`no_task_specific_hints`, and `rationale`."
        )

        return "\n".join(lines)

    def _negative_transfer_safety_block(self) -> str:
        return (
            "## Negative-transfer safety (AP-33):\n"
            "- Do not import tactics anchored to a benchmark suite or domain that is "
            "not present in the failure context or per-suite quality list.\n"
            f"- If fewer than {_MIN_VALIDATION_TRIALS} trial IDs are cited, phrase "
            "changes as exploratory and do not claim validation.\n"
            "- Do not turn suite-specific fixes into universal always/never/all-tasks "
            "best practices."
        )

    def _attach_transfer_safety(
        self,
        mutation: PromptMutation | CodeMutation,
        *,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> TransferSafetyVerdict:
        verdict = self._transfer_safety_verdict(
            original_content=original_content,
            mutated_content=mutation.mutated_content,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description or mutation.description,
        )
        mutation.safety_valid = verdict.valid
        mutation.safety_reason = verdict.reason
        mutation.safety_warnings = list(verdict.warnings)
        return verdict

    def _transfer_safety_verdict(
        self,
        *,
        original_content: str,
        mutated_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> TransferSafetyVerdict:
        source_text = " ".join(str(suite) for suite in (per_suite_quality or {}))
        source_suites = _suite_mentions(source_text)
        if not source_suites:
            source_suites = _suite_mentions(failure_context)

        introduced_text = f"{description}\n{_added_text(original_content, mutated_content)}"
        introduced_suites = _suite_mentions(introduced_text)
        evidence_count = _trial_reference_count(failure_context)

        warnings: list[str] = []
        if failure_context.strip() and evidence_count < _MIN_VALIDATION_TRIALS:
            warnings.append(f"low_evidence_trial_count:{evidence_count}")

        mismatched = introduced_suites - source_suites
        if source_suites and mismatched:
            return TransferSafetyVerdict(
                valid=False,
                reason=(
                    "domain_mismatched_anchoring:"
                    f" introduced_suites={sorted(mismatched)}"
                    f" source_suites={sorted(source_suites)}"
                ),
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        if introduced_suites and _UNIVERSAL_TRANSFER_RE.search(introduced_text):
            return TransferSafetyVerdict(
                valid=False,
                reason=(f"misapplied_best_practice: introduced_suites={sorted(introduced_suites)}"),
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        return TransferSafetyVerdict(
            valid=True,
            warnings=tuple(warnings),
            source_suites=tuple(sorted(source_suites)),
            introduced_suites=tuple(sorted(introduced_suites)),
            evidence_trial_count=evidence_count,
        )

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
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            log.info("Committed code mutation: %s", path.name)
        except Exception as e:
            log.warning("Git commit failed: %s", e)
