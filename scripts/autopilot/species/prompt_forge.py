"""Species 2 — PromptForge: LLM-guided prompt optimization.

Uses Claude CLI (Popen + session persistence) to analyze failure cases
and propose targeted prompt mutations on hot-swappable .md files.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time  # noqa: F401 - kept as ``prompt_forge.time`` (tests patch time.monotonic through it)
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger("autopilot.prompt_forge")

if TYPE_CHECKING:
    from scripts.autopilot.worktree_manager import ExperimentContext

import ast
import math

# HIL-6 S1 (2026-10-22): the pure MHS-1/2/4/15 guards moved byte-for-byte to the
# import-safe ``src.autopilot_core.mutation_guards``; every name is re-exported here so
# ``prompt_forge.X is mutation_guards.X`` and no call site changes.
from src.autopilot_core.mutation_guards import (  # noqa: E402,F401 - re-exported API
    MutationEffect,
    _EFFECT_TOKEN_LIMIT,
    _EFFECT_REASON_LIMIT,
    _EFFECT_ALIASES,
    _truncate_reason,
    SAFETY_BANNED_CALLS,
    SAFETY_RESTRICTED_CALLS,
    SAFETY_BANNED_MODULES,
    SAFETY_IMPORT_ALLOWLIST,
    SAFETY_FIRST_PARTY_ROOTS,
    SAFETY_DUNDER_ATTR_ALLOWLIST,
    SAFETY_TOPLEVEL_ALLOWED,
    SAFETY_STRICT_NODE_DENYLIST,
    StaticSafetyReport,
    _dotted_name,
    _collect_dotted_calls,
    _collect_imported_roots,
    _top_level_definition_names,
    screen_static_safety,
    classify_mutation_effect,
    MUTATION_EFFECT_RISK,
    MUTATION_RISK_GATE_ENV,
    DEFAULT_MUTATION_RISK_GATE,
    mutation_effect_risk,
    mutation_risk_gate,
    mutation_risk_gate_reason,
    rank_mutations_by_risk,
    classify_prompt_effect,
    INTERFERENCE_WARNING_PREFIX,
    INTERFERENCE_EXAMPLE_ADD_THRESHOLD,
    _INTERFERENCE_MIN_DUP_CHARS,
    _INTERFERENCE_POS_RE,
    _INTERFERENCE_NEG_RE,
    _INTERFERENCE_DIRECTIVE_SPLIT_RE,
    _INTERFERENCE_WORD_RE,
    _INTERFERENCE_STOPWORDS,
    _INTERFERENCE_PRIORITY_RE,
    _INTERFERENCE_EXAMPLE_RE,
    _interference_lines,
    _interference_dup_keys,
    _interference_scope_nouns,
    _interference_conflicts,
    _interference_priority_lines,
    _interference_example_count,
    prompt_interference_lint,
)

# HIL-6 S2 (2026-10-22): the MHS-3 eval-leakage guards (id vocabulary, content index,
# suite special-casing), the frontdoor integrity check and the harness corpus moved to
# ``src.autopilot_core.mutation_guards``. Every name is re-exported so ``prompt_forge.X is
# mutation_guards.X``. Their module globals (caches, ``_CONTENT_*`` tunables, the leakage
# observer, ``_iter_eval_rows``) now live THERE: rebinding one on this module changes nothing,
# so patch ``mutation_guards`` instead. ``harness_content`` below is a wrapper.
from src.autopilot_core.mutation_guards import (  # noqa: E402,F401 - re-exported API
    CODE_MUTATION_ALLOWLIST,
    _SUITE_ALIASES,
    _SUITE_TERM_TO_CANONICAL,
    _SUITE_TERM_RE,
    _FRONTDOOR_REQUIRED_MARKERS,
    _FRONTDOOR_CORRUPTION_MARKERS,
    _suite_mentions,
    _added_text,
    _prompt_integrity_reason,
    EVAL_ID_VOCAB_SOURCES_ENV,
    _EVAL_ID_KEYS,
    _LEAKAGE_MIN_ID_LEN,
    _LEAKAGE_MIN_NUMERIC_ID_LEN,
    _LEAKAGE_MIN_FAMILY_PREFIX,
    _LEAKAGE_MIN_FAMILY_MEMBERS,
    _LEAKAGE_MIN_STEM,
    _LEAKAGE_MAX_REPORTED,
    _LEAKAGE_GENERIC_RE,
    _ID_FAMILY_TAIL_RE,
    _ID_NATIVE_FAMILY_RE,
    _ID_STEM_RE,
    _default_eval_id_sources,
    _iter_eval_rows,
    stable_question_qid,
    _is_identifier_shaped,
    _is_word_char,
    _CONTENT_NGRAM,
    _CONTENT_TEMPLATE_DF,
    _CONTENT_LONG_ROW_BYTES,
    _CONTENT_LONG_EDGE_SHINGLES,
    _CONTENT_WINNOW,
    _CONTENT_CHUNK_BYTES,
    _CONTENT_SKETCH_BITS,
    _CONTENT_SKETCH_FLUSH,
    _CONTENT_RARE_TF,
    _CONTENT_MAX_ROWS,
    _ANSWER_MAX_TOKENS,
    _ANSWER_MIN_BYTES,
    _ANCHOR_MIN_BYTES,
    _ANSWER_UNSHAPED_MAX_TF,
    _SOURCE_MIN_SHINGLE_HITS,
    _CONTENT_MIN_UNANCHORED_RUN,
    _SOURCE_MIN_SHARED_ANCHORS,
    _SOURCE_MAX_QIDS,
    _ANSWER_PLACEHOLDER_RE,
    _HASH_PRIME,
    _HASH_MASK,
    _HASH_M1,
    _HASH_M2,
    _HASH_LEN_SALT,
    _HASH_SEQ_SALT,
    _HASH_CONSTS,
    _np,
    _hash_consts,
    _mix64,
    _hash_token_buffer,
    _content_tokens,
    _window_hashes,
    _sequence_hashes,
    _packed_lookup,
    _row_label_id,
    _anchor_mask,
    EvalContentIndex,
    _EvalContentIndexBuilder,
    EvalIdVocabulary,
    _EVAL_ID_VOCAB_CACHE,
    EVAL_ID_VOCAB_NEG_TTL_ENV,
    DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S,
    _EVAL_ID_VOCAB_NEG_CACHE,
    _eval_id_vocab_neg_ttl_s,
    clear_eval_id_vocabulary_cache,
    describe_eval_id_sources,
    set_eval_leakage_observer,
    _notify_eval_leakage_observer,
    load_eval_id_vocabulary,
    _vocabulary_error,
    eval_leakage_reason,
    _TRACE_SECTION_RE,
    _CONTEXT_SECTION_END_RE,
    _CONTRASTIVE_ENTRY_RE,
    _TRACE_IR_JSON_RE,
    _TRACE_QID_FIELD_RE,
    _TRACE_ID_FIELDS,
    _trace_ir_blocks,
    _trace_blocks,
    _HarnessContent,
    _HARNESS_CONTENT_CACHE,
    _text_shingles_tokens,
    _in_sorted,
    _span_text,
    _resolve_trace_qids,
    eval_content_leakage_reason,
    _SUITE_MENTION_MAX_TF,
    _SUITE_PHRASE_MAX_TF,
    _SUITE_MIN_POOL_TOKENS,
    _SUITE_HARNESS_VOCAB_FILES,
    _SUITE_SHAPED_RE,
    _SUITE_RULES_CACHE,
    _suite_term_pattern,
    _suite_rules,
    suite_special_casing_reason,
    harness_content as _guards_harness_content,
)

# HIL-6j (2026-10-22): repo-relative targets a sibling measurement asked PromptForge to leave
# alone (default: none; read on every call). See ``src.autopilot_core.mutation_exclusions``.
from src.autopilot_core import mutation_exclusions as _mutation_exclusions  # noqa: E402

ORCH_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ORCH_ROOT / "orchestration" / "prompts"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Meta-Harness Tier 2 ``CODE_MUTATION_ALLOWLIST`` (the eval trust boundary) moved with the
# MHS-3 harness corpus to ``src.autopilot_core.mutation_guards`` (HIL-6 S2); re-exported below.

# New-file code mutations are more permissive than the existing-file allowlist,
# but they stay directory-scoped. ``src/`` is for ordinary code scaffolds;
# ``schema_evolution/`` is the AutoMem/MH-9 lane for default-inert memory
# schema/scaffold proposals.
NEW_FILE_MUTATION_ROOT = PROJECT_ROOT / "src"
MEMORY_SCHEMA_MUTATION_ROOT = PROJECT_ROOT / "orchestration" / "repl_memory" / "schema_evolution"

# RTG-55: the shape the MH-9 schema-evolution prompt asks for. It is shipped in
# the prompt verbatim and MUST itself pass ``screen_static_safety(strict=True)``
# — a test asserts that, so prompt/denylist drift cannot recur silently.
MEMORY_SCHEMA_SHAPE_EXAMPLE = '''"""Proposed plan-channel memory schema (inert)."""

SCHEMA_VERSION = 1
ACTIONS = ("APPEND", "CREATE", "UPSERT")
CHANNELS = ("status", "inventory", "strategy", "plan", "log")
SCHEMA = {
    "fields": (
        {"name": "channel", "required": True, "kind": "str"},
        {"name": "content", "required": True, "kind": "str"},
    ),
    "blockers": ("no calibration evidence for plan-channel upserts",),
}


def required_fields():
    return tuple(spec["name"] for spec in SCHEMA["fields"] if spec["required"])


def validate_action(action):
    if not isinstance(action, dict):
        return (False, "action must be a dict")
    for name in required_fields():
        if not action.get(name):
            return (False, f"missing field: {name}")
    if action.get("channel") not in CHANNELS:
        return (False, "channel must be one of CHANNELS")
    return (True, "ok")
'''

MUTATION_TYPES = [
    "targeted_fix",  # Fix specific failure patterns
    "compress",  # Reduce token count while maintaining behavior
    "few_shot_evolution",  # Add/remove/modify examples
    "crossover",  # Merge sections from two prompts
    "style_transfer",  # Apply patterns from one prompt to another
    "gepa",  # AP-19: GEPA evolutionary optimization (runs internal eval loop)
]

_MIN_VALIDATION_TRIALS = 5
_TRIAL_REF_RE = re.compile(r"(?:\btrial\s*#?\s*|\[?t)(\d+)\]?", re.IGNORECASE)
_UNIVERSAL_TRANSFER_RE = re.compile(
    r"\b(always|never|universally|global(?:ly)?|all\s+(?:tasks|prompts|suites|benchmarks)|"
    r"every\s+(?:task|prompt|suite|benchmark))\b",
    re.IGNORECASE,
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
    # RTG-55 MHS-4: prompt-side effect (``MutationEffect``) and its risk weight.
    effect: MutationEffect | None = None
    effect_risk: float | None = None


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
    # RTG-55 MHS-1: the typed return-effect the static screen assigned.
    effect: MutationEffect = MutationEffect.UNKNOWN
    effect_reason: str = ""
    # RTG-55 MHS-4: the ANTI-OVERRIDE prior weight of ``effect``.
    effect_risk: float = 1.0


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


def _trial_reference_count(text: str) -> int:
    return len({int(match) for match in _TRIAL_REF_RE.findall(text or "")})


def harness_content(prompts_dir: Path | None = None) -> _HarnessContent:
    """Shingles/tokens of the current prompts and the orchestrator code (cached by file identity).

    HIL-6 S2: the body moved to ``mutation_guards.harness_content``. This wrapper passes
    PromptForge's own roots, read at call time, so rebinding ``PROMPTS_DIR`` /
    ``NEW_FILE_MUTATION_ROOT`` / ``PROJECT_ROOT`` on this module still steers it.
    """
    return _guards_harness_content(
        prompts_dir if prompts_dir is not None else PROMPTS_DIR,
        code_root=NEW_FILE_MUTATION_ROOT,
        project_root=PROJECT_ROOT,
        code_files=CODE_MUTATION_ALLOWLIST,
    )


# --------------------------------------------------------------------------- TD-21.27: mutation extraction
#
# The mutation backend is the Claude CLI (`claude -p`, `_invoke_claude`
# below) -- an external process with no HTTP endpoint and no
# `response_format` support, so the shared `src.structured_output.repair`
# fish-then-repair idiom (TD-21.H) cannot reach it (per the TD-21 dispatch's
# HARD RULES: "apply deterministic fish improvements only and say so"). What
# follows is that deterministic-fish-only improvement, shared by both
# `_extract_mutation` (prompt bodies) and `_extract_code_mutation` (Python
# source) so a fenced-code mention inside the reply's own PROSE can never be
# mis-captured as the payload on either path -- the exact bug class
# `_extract_mutation` already guarded against alone before this change.

#: (site, status) -> count. "parsed" = a fenced/heuristic block was found;
#: "failed" = extraction found nothing and the mutation became a no-op.
#: TD-21.28's counter is `STRUCTURED_OUTPUT_REPAIR_COUNTS` (JSON-shaped);
#: this one is deliberately separate since extraction here is fenced TEXT,
#: not JSON, and there is no repair-turn dimension to key on.
MUTATION_EXTRACTION_COUNTS: dict[tuple[str, str], int] = {}


def _record_mutation_extraction(site: str, status: str) -> None:
    key = (site, status)
    MUTATION_EXTRACTION_COUNTS[key] = MUTATION_EXTRACTION_COUNTS.get(key, 0) + 1


_FENCE_BLOCK_RE = re.compile(
    r"^[ \t]*`{3,}[ \t]*([\w:.\-]*)[ \t]*\r?\n(.*?)(?:\r?\n[ \t]*`{3,}|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Line-anchored fenced blocks as ``(tag.lower(), body)``.

    Only fences whose opening backticks begin a line count as delimiters --
    an inline fenced-code mention inside prose (e.g. a sentence quoting
    ``result.index(...)``) can never be mis-captured as the payload. Shared
    by the prompt- and code-mutation extractors so both apply the identical
    rule instead of two near-copies with different edge-case coverage.
    """
    return [(tag.strip().lower(), body) for tag, body in _FENCE_BLOCK_RE.findall(text)]


class PromptForge:
    """Species 2: LLM-guided prompt mutation and optimization."""

    def __init__(
        self,
        prompts_dir: Path | None = None,
        timeout: int = 300,
        auto_commit: bool = True,
        eval_id_vocabulary: EvalIdVocabulary | None = None,
    ):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.timeout = timeout
        self.auto_commit = auto_commit
        self._session_id: str | None = None
        # RTG-55 MHS-3: injected vocabulary (tests) or the data-sourced default.
        self._injected_eval_id_vocabulary = eval_id_vocabulary

    def _eval_id_vocabulary(self) -> EvalIdVocabulary:
        if self._injected_eval_id_vocabulary is not None:
            return self._injected_eval_id_vocabulary
        return load_eval_id_vocabulary()

    def _leakage_reason(
        self,
        vocabulary: EvalIdVocabulary,
        original: str,
        mutated: str,
        failure_context: str = "",
    ) -> str | None:
        """RTG-55 MHS-3: the one leakage checker for prompt, GEPA and code mutations."""
        return eval_leakage_reason(
            _added_text(original, mutated),
            vocabulary,
            original=original,
            trace_context=failure_context,
            harness=harness_content(self.prompts_dir),
        )

    def list_prompts(self) -> list[str]:
        """List all hot-swappable prompt files (flat + roles/ subdirectory).

        HIL-6j: a prompt listed in the AutoPilot mutation-exclusions file is left out, so the
        controller is never offered it. With no exclusions the listing is unchanged.
        """
        if not self.prompts_dir.exists():
            return []
        files = self.prompts_dir.rglob("*.md")
        excluded = _mutation_exclusions.load()
        if excluded:
            files = [
                f
                for f in files
                if _mutation_exclusions.excluded_reason(self._exclusion_target(f), excluded) is None
            ]
        return sorted(f.name for f in files)

    def _exclusion_target(self, path: Path) -> str:
        """Repo-relative spelling (``orchestration/prompts/<rel>``) of a prompt file."""
        try:
            rel = path.resolve().relative_to(self.prompts_dir.resolve())
        except ValueError:
            rel = Path(path.name)
        return "orchestration/prompts/" + rel.as_posix()

    def _mutation_exclusion_reason(self, filename: str) -> str | None:
        """HIL-6j: ``mutation_exclusion:<target>`` when ``filename`` may not be mutated now."""
        excluded = _mutation_exclusions.load()
        if not excluded:
            return None
        try:
            path = self._resolve_prompt_path(filename)
        except FileNotFoundError:
            path = self.prompts_dir / filename
        return _mutation_exclusions.excluded_reason(self._exclusion_target(path), excluded)

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
        author_capture_context: dict[str, Any] | None = None,
    ) -> PromptMutation:
        """Propose a prompt mutation via Claude CLI or GEPA.

        When mutation_type="gepa", delegates to GEPA evolutionary optimization
        (AP-19). Requires eval_tower to be passed for orchestrator-based eval.

        Returns PromptMutation with the proposed changes.
        """
        if mutation_type not in MUTATION_TYPES:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        # HIL-6j: an excluded target gets no proposal (no CLI call, no GEPA run).
        exclusion = self._mutation_exclusion_reason(target_file)
        if exclusion is not None:
            original = self.read_prompt(target_file)
            log.warning("Prompt mutation refused (%s): %s", target_file, exclusion)
            return PromptMutation(
                file=target_file,
                mutation_type=mutation_type,
                description=f"{exclusion}: no mutation proposed",
                original_content=original,
                mutated_content=original,
                safety_valid=False,
                safety_reason=exclusion,
            )

        # AP-19: GEPA evolutionary optimization
        if mutation_type == "gepa":
            return self._propose_via_gepa(
                target_file=target_file,
                eval_tower=eval_tower,
                max_evals=gepa_max_evals,
                description=description,
            )

        original = self.read_prompt(target_file)

        try:  # CT-19b: flag-gated, warn-only scan of the failure traces that steer the rewrite
            from src.security.skill_scan import scan_fields

            scan_fields({"failure_context": failure_context}, "prompt_forge", subject=target_file)
        except Exception:
            pass

        prompt = self._build_mutation_prompt(
            target_file=target_file,
            mutation_type=mutation_type,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        # Best-effort metadata is captured at the complete-input boundary, before any author call.
        try:
            from mutation_author_capture import capture_author_context

            capture_author_context(
                prompt,
                operator=mutation_type,
                target=target_file,
                inputs={
                    "original_content": original,
                    "failure_context": failure_context,
                    "description": description,
                },
                context=author_capture_context,
                parameters={"timeout_s": self.timeout},
            )
        except Exception as exc:
            log.warning("Mutation author metadata unavailable (%s)", type(exc).__name__)
        result = self._invoke_claude(prompt)
        mutated_content, extracted_ok = self._extract_mutation(result, original)

        mutation = PromptMutation(
            file=target_file,
            mutation_type=mutation_type,
            description=description or f"{mutation_type} on {target_file}",
            original_content=original,
            mutated_content=mutated_content,
        )
        if not extracted_ok:
            # TD-21.27: extraction found nothing usable in the Claude CLI
            # reply -- an explicit, counted failure (`safety_valid=False`
            # reuses the EXISTING gate `_action_prompt_mutation` already
            # checks before `apply_mutation`/eval) instead of a mutation
            # that silently costs an eval round and a wasted commit
            # checkpoint for a no-op nobody can see happened.
            mutation.safety_valid = False
            mutation.safety_reason = "mutation_extraction_failed: no usable fenced block in reply"
            return mutation
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
        self._attach_prompt_effect(mutation)
        if not mutation.safety_valid:
            log.warning(
                "Prompt mutation rejected by transfer safety (%s): %s",
                target_file,
                mutation.safety_reason,
            )
            mutation.mutated_content = original
        return mutation

    def _attach_prompt_effect(self, mutation: PromptMutation) -> None:
        """RTG-55 MHS-4: classify a prompt mutation and apply the risk gate."""
        mutation.effect = classify_prompt_effect(
            mutation.original_content, mutation.mutated_content
        )
        mutation.effect_risk = mutation_effect_risk(mutation.effect)
        if not mutation.safety_valid:
            return
        gate_reason = mutation_risk_gate_reason(mutation.effect)
        if gate_reason is not None:
            mutation.safety_valid = False
            mutation.safety_reason = gate_reason

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
        # RTG-55 MHS-3: GEPA candidates can memorise the eval set too.
        vocabulary = self._eval_id_vocabulary()
        leakage = self._leakage_reason(
            vocabulary, mutation.original_content, mutation.mutated_content
        ) or suite_special_casing_reason(
            mutation.original_content,
            mutation.mutated_content,
            vocabulary,
            harness=harness_content(self.prompts_dir),
        )
        if leakage is not None:
            mutation.safety_valid = False
            mutation.safety_reason = leakage
            mutation.mutated_content = mutation.original_content
            log.warning(
                "GEPA prompt mutation rejected by leakage guard (%s): %s",
                mutation.file,
                leakage,
            )
            return mutation
        self._attach_prompt_effect(mutation)
        if not mutation.safety_valid:
            mutation.mutated_content = mutation.original_content
        else:
            # MHS-15: GEPA candidates get the same warn-only interference lint.
            mutation.safety_warnings = list(mutation.safety_warnings) + prompt_interference_lint(
                mutation.original_content, mutation.mutated_content
            )
        return mutation

    def apply_mutation(self, mutation: PromptMutation) -> dict[str, Any]:
        """Apply a mutation (write file + optional git commit)."""
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected mutation: {integrity_reason}")
        exclusion = self._mutation_exclusion_reason(mutation.file)
        if exclusion is not None:
            raise ValueError(f"prompt mutation refused: {exclusion}")

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
                f"autopilot: {mutation.mutation_type} on {mutation.file}\n\n{mutation.description}",
                paths=[self._resolve_prompt_path(mutation.file)],
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
                f"Reverted: {mutation.description}",
                paths=[self._resolve_prompt_path(mutation.file)],
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

    def _extract_mutation(self, result: str, original: str) -> tuple[str, bool]:
        """Extract mutated prompt from Claude's response.

        Only fences whose opening backticks begin a line are treated as block
        delimiters (`_fenced_blocks`). This prevents an inline fenced-code
        mention inside the reply's *prose* (e.g. a sentence quoting
        ``result.index(...)``) from being mis-captured as the payload — a bug
        that overwrote a prompt file with the model's prose and committed it.

        TD-21.27: returns ``(content, ok)``. ``ok=False`` means extraction
        found nothing usable — ``content`` is still ``original`` (the Claude
        CLI backend is unreachable for a repair turn, so a safe no-op is the
        only sound fallback), but the caller now gets an explicit signal
        instead of an indistinguishable "successful no-op mutation", so it
        can skip the mutation entirely (see ``propose_mutation``) rather than
        spend an eval cycle and a git commit around it.
        """
        blocks = _fenced_blocks(result)

        # Prefer an explicitly prose/markdown-tagged block.
        for tag, body in blocks:
            if tag in ("markdown", "md", "text"):
                _record_mutation_extraction("prompt_forge.mutation_extraction", "parsed")
                return body.strip(), True

        # Fallback: the largest block that is not a json/actions or object block.
        candidates = [
            body
            for tag, body in blocks
            if not tag.startswith("json")
            and not body.lstrip().startswith("{")
            and len(body.strip()) > 100
        ]
        if candidates:
            _record_mutation_extraction("prompt_forge.mutation_extraction", "parsed")
            return max(candidates, key=lambda b: len(b.strip())).strip(), True

        log.warning("Could not extract mutation from response, returning original")
        _record_mutation_extraction("prompt_forge.mutation_extraction", "failed")
        return original, False

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

    def _git_commit(self, message: str, *, paths: list[Path] | None = None) -> None:
        """Commit ONLY the prompt file(s) this mutation touched (never a directory sweep)."""
        if not paths:
            log.warning("Git commit skipped: no prompt path given (refusing a directory sweep)")
            return
        self._git_commit_paths(paths, message)

    @staticmethod
    def _git_commit_paths(paths: list[Path], message: str) -> bool:
        """Stage and commit exactly ``paths``; unrelated staged or dirty files stay out.

        ``git commit --only -- <paths>`` records just those paths and leaves every other
        index entry as it was, so an operator's staged work never rides along with an
        autopilot commit (gate-frontier review, 2026-09-16).
        """
        specs = [str(Path(p)) for p in paths]
        try:
            subprocess.run(
                ["git", "add", "-A", "--", *specs],
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
            )
            proc = subprocess.run(
                ["git", "commit", "--only", "-m", message, "--", *specs],
                timeout=10,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                log.warning(
                    "Git commit of %s failed: %s", specs, (proc.stderr or proc.stdout).strip()[:300]
                )
                return False
            log.info("Committed %s", ", ".join(Path(p).name for p in specs))
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("Git commit failed: %s", e)
            return False

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
        exclusion = self._mutation_exclusion_reason(mutation.file)
        if exclusion is not None:
            raise ValueError(f"isolated prompt mutation refused: {exclusion}")

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
        if MutationEffect.normalize(mutation.effect) is MutationEffect.UNSAFE:
            return {"status": "rejected", "reason": "effect_unsafe"}
        gate_reason = mutation_risk_gate_reason(mutation.effect)
        if gate_reason is not None:
            return {"status": "rejected", "reason": gate_reason}

        try:
            abs_path = _resolve_code_mutation_target(mutation.file)
        except FileNotFoundError:
            return {"status": "rejected", "reason": "target_invalid"}
        if mutation.mutation_type == "new_file":
            if mutation.original_content:
                return {"status": "rejected", "reason": "new_file_has_preimage"}
            if not _is_under_any(abs_path.parent, new_file_mutation_roots()):
                return {"status": "rejected", "reason": "new_file_target_invalid"}
            if abs_path.exists():
                return {"status": "rejected", "reason": "target_changed"}
        else:
            if mutation.file not in CODE_MUTATION_ALLOWLIST:
                return {"status": "rejected", "reason": "target_not_allowlisted"}
            try:
                current_content = abs_path.read_text()
            except FileNotFoundError:
                return {"status": "rejected", "reason": "target_missing"}
            if current_content != mutation.original_content:
                return {"status": "rejected", "reason": "target_changed"}

        rel_path = abs_path.relative_to(PROJECT_ROOT).as_posix()
        ctx.apply_file(rel_path, mutation.mutated_content)
        mutation.accepted = True
        return {
            "status": "applied_isolated",
            "file": rel_path,
            "mutation_type": mutation.mutation_type,
            "effect": MutationEffect.normalize(mutation.effect).value,
            "effect_risk": mutation_effect_risk(mutation.effect),
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
        author_capture_context: dict[str, Any] | None = None,
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

        # Best-effort metadata is captured at the complete-input boundary, before any author call.
        try:
            from mutation_author_capture import capture_author_context

            capture_author_context(
                prompt,
                operator=mutation_type,
                target=target_file,
                inputs={
                    "original_content": original,
                    "failure_context": failure_context,
                    "description": description,
                },
                context=author_capture_context,
                parameters={"timeout_s": self.timeout},
            )
        except Exception as exc:
            log.warning("Mutation author metadata unavailable (%s)", type(exc).__name__)
        result = self._invoke_claude(prompt)
        mutated_content, extracted_ok = self._extract_code_mutation(result, original)

        mutation = CodeMutation(
            file=target_file,
            mutation_type=mutation_type,
            description=description or f"{mutation_type} on {target_file}",
            original_content=original,
            mutated_content=mutated_content,
        )
        if not extracted_ok:
            # TD-21.27: same explicit, counted failure as the prompt-mutation
            # path. `_action_code_mutation` checks `syntax_valid` BEFORE
            # `safety_valid` -- syntax_valid=True here is not a claim that
            # anything was screened, it is what routes this rejection
            # through the transfer_safety gate (which DOES surface
            # `safety_reason` as `gate_detail`) instead of the generic,
            # reason-less syntax_validation gate, since mutated_content is
            # still the untouched, already-valid original.
            mutation.syntax_valid = True
            mutation.safety_valid = False
            mutation.safety_reason = "mutation_extraction_failed: no usable fenced block in reply"
            return mutation
        self._attach_transfer_safety(
            mutation,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        # Deep validation: syntax + shrinkage + public names + static safety screen.
        # STATIC ONLY — nothing is written to the repo and nothing is imported.
        report = self._screen_code_mutation(
            original,
            mutated_content,
            target_file,
            is_new_file=(mutation_type == "new_file"),
        )
        mutation.syntax_valid = report.safe
        mutation.effect = report.effect
        mutation.effect_reason = report.reason
        if not report.safe:
            log.warning("Code mutation rejected (%s): %s", target_file, report.reason)
            mutation.mutated_content = original
        if not mutation.safety_valid:
            log.warning(
                "Code mutation rejected by transfer safety (%s): %s",
                target_file,
                mutation.safety_reason,
            )
            mutation.mutated_content = original
            mutation.effect = MutationEffect.UNSAFE
            mutation.effect_reason = _truncate_reason(mutation.safety_reason)

        # RTG-55 MHS-4: ANTI-OVERRIDE risk prior + gate.
        mutation.effect_risk = mutation_effect_risk(mutation.effect)
        if report.safe and mutation.safety_valid:
            gate_reason = mutation_risk_gate_reason(mutation.effect)
            if gate_reason is not None:
                log.warning(
                    "Code mutation rejected by risk gate (%s): %s", target_file, gate_reason
                )
                mutation.safety_valid = False
                mutation.safety_reason = gate_reason
                mutation.mutated_content = original

        return mutation

    def apply_code_mutation(self, mutation: CodeMutation) -> dict[str, Any]:
        """Refuse legacy direct writes; code mutations require an experiment context."""
        del mutation
        return {"status": "rejected", "reason": "isolated_context_required"}

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
                self._git_commit_paths(
                    [abs_path],
                    f"autopilot: revert code mutation on {mutation.file}\n\n"
                    f"Reverted: {mutation.description}",
                )
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
        """Backwards-compatible ``(valid, reason)`` wrapper over the static screen."""
        report = self._screen_code_mutation(original, mutated, target_file, is_new_file=is_new_file)
        return report.safe, report.reason

    def _screen_code_mutation(
        self,
        original: str,
        mutated: str,
        target_file: str,
        *,
        is_new_file: bool = False,
    ) -> StaticSafetyReport:
        """Deep validation of a code mutation beyond syntax. STATIC ONLY.

        Returns (valid, reason). Checks:
        1. Syntax (ast.parse)
        2. No catastrophic size reduction (>60% shrinkage)
        3. Public names preserved (classes, functions defined at module level)
        4. RTG-55 static safety screen (MHS-12 UNSAFE takes precedence over 2/3)

        This function NEVER writes into the source tree and NEVER imports or
        execs the candidate in this process. The pre-RTG-55 step 4 wrote the
        model's code over the live repo file and then ``importlib`` -imported
        it, so the mutation's module top level ran unsandboxed and any
        concurrent reader saw the candidate on disk.
        """
        # 1. Syntax
        try:
            mutated_tree = ast.parse(mutated)
        except SyntaxError as e:
            return StaticSafetyReport(
                False, MutationEffect.UNSAFE, (_truncate_reason(f"syntax error: {e}"),)
            )

        # MHS-12 must report UNSAFE even when removing forced routing also
        # triggers a generic REPLACE rejection below. Cache the static report;
        # every other existing validation keeps its original precedence.
        report = screen_static_safety(mutated, original=original, strict=is_new_file)
        if any(
            violation.startswith(("eval_identity_added_read:", "force_role_handling_changed:"))
            for violation in report.violations
        ):
            return report

        # 2. Catastrophic shrinkage — reject if >60% of lines removed
        orig_lines = len(original.splitlines())
        new_lines = len(mutated.splitlines())
        if not is_new_file and orig_lines > 10 and new_lines < orig_lines * 0.4:
            return StaticSafetyReport(
                False,
                MutationEffect.REPLACE,
                (
                    f"catastrophic shrinkage: {orig_lines}→{new_lines} lines "
                    f"({100 * (1 - new_lines / orig_lines):.0f}% removed)",
                ),
            )

        # 3. Public names preserved — every class/function at module level
        #    in the original must still exist in the mutated version
        if not is_new_file:
            orig_names = _top_level_definition_names(ast.parse(original))
            missing = orig_names - _top_level_definition_names(mutated_tree)
            if missing:
                return StaticSafetyReport(
                    False,
                    MutationEffect.REPLACE,
                    (f"missing public names: {sorted(missing)}",),
                )

        # 4. Static safety screen (RTG-55 MHS-2). Replaces the pre-RTG-55
        #    "write the candidate over the live file and importlib-import it"
        #    step: no repo write, no in-process execution, no sys.path games.
        #    `new_file` proposals get the strict inertness denylist.
        if not report.safe:
            log.warning(
                "Static safety screen rejected mutation on %s: %s", target_file, report.reason
            )
        return report

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
                "- Create a default-inert schema/scaffold module describing a "
                "`MemoryAction` / `MemoryActionStore` schema proposal; importing "
                "it must not write files, start subprocesses, call inference, or "
                "touch the trace store. Importing it must DO NOTHING at all.\n"
                "- Express schema-evolution moves as prompt-free helpers, "
                "contracts, constants, or pure validators over "
                "APPEND/CREATE/UPSERT and the status/inventory/strategy/plan/log "
                "channels.\n"
                "- Do not change SafetyGate, Pareto admission, eval scoring, "
                "blacklists, thresholds, planner spend-breaker flags, or live "
                "runtime behavior.\n"
                "- Keep exports narrow and include explicit blockers when "
                "calibration, process, or validation evidence is missing.\n"
                "\n"
                "### Required SHAPE — a static validator rejects anything else\n"
                "Inertness is enforced by an AST denylist, not by trust. A "
                "proposal that breaks any rule below is DISCARDED WITHOUT "
                "REVIEW, so write to this shape exactly:\n"
                "- NO import statements of any kind — not even `dataclasses`, "
                "`enum`, `typing`, or `__future__`. Use only builtins.\n"
                "- NO `class` statements. Express the schema as plain module "
                "data: a `SCHEMA` dict, or a tuple of field-spec tuples/dicts, "
                "plus constants such as `SCHEMA_VERSION`, `ACTIONS`, "
                "`CHANNELS`.\n"
                "- NO underscore-prefixed names anywhere (no `_helper`, no "
                "`_CACHE`, no `obj._attr`). Every name must be public.\n"
                "- NO `raise`, `try`, `with`, `while`, `lambda`, `global`, "
                "`del`, `yield`, or `await`. A validator returns a "
                "`(ok, reason)` tuple instead of raising: "
                "`return (False, 'channel must be one of CHANNELS')`.\n"
                "- Module level may contain ONLY the docstring, constant "
                "assignments, and `def` statements. No calls, prints, or "
                "registration at module level.\n"
                "- Inside functions: `if`/`for`/`return`, comparisons, f-strings, "
                "and builtin calls (`len`, `isinstance`, `sorted`, `tuple`, "
                "`dict`, `str`) are all fine.\n"
                "\n"
                "Shape example (structure to copy, not content to reuse):\n"
                "```python\n"
                f"{MEMORY_SCHEMA_SHAPE_EXAMPLE}"
                "```"
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
            "best practices.\n"
            "- Never name a specific eval question, sample, item or task id (e.g. "
            "`<suite>_00042`, `question #12`); a change keyed to one instance memorises "
            "the eval set and is rejected automatically (RTG-55 MHS-3).\n"
            "- Prefer CONSTRAIN edits (add a check, block a bad path, re-prompt) over "
            "REPLACE edits (rewrite or force an action, hard-code an answer). Override "
            "edits carry the highest regression risk and are ranked last (RTG-55 MHS-4)."
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
        if isinstance(mutation, PromptMutation):
            # MHS-15: warn-only; appended after the verdict so it can never change validity.
            mutation.safety_warnings.extend(
                prompt_interference_lint(original_content, mutation.mutated_content)
            )
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

        # RTG-55 MHS-3: under-generalization (eval-instance leakage), fail-closed.
        # Scoped to the text the mutation ADDS; the description is controller metadata.
        vocabulary = self._eval_id_vocabulary()
        leakage = self._leakage_reason(
            vocabulary, original_content, mutated_content, failure_context
        )
        if leakage is not None:
            return TransferSafetyVerdict(
                valid=False,
                reason=leakage,
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

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

        # RTG-55 MHS-3 (2026-09-15): in-suite special-casing, which the check above allows.
        special = suite_special_casing_reason(
            original_content,
            mutated_content,
            vocabulary,
            harness=harness_content(self.prompts_dir),
        )
        if special is not None:
            return TransferSafetyVerdict(
                valid=False,
                reason=special,
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

    def _extract_code_mutation(self, result: str, original: str) -> tuple[str, bool]:
        """Extract mutated Python code from Claude's response.

        TD-21.27: rebuilt on the SAME line-anchored ``_fenced_blocks`` helper
        `_extract_mutation` already uses, closing the same bug class here —
        the old ``result.split("```")`` treated ANY triple-backtick anywhere
        in the reply's prose as a delimiter, with no line-anchoring at all.
        Returns ``(content, ok)``; see `_extract_mutation` for the ``ok``
        contract (extraction failure never repairs via the Claude CLI
        backend, only fishes better and reports the miss explicitly).
        """
        blocks = _fenced_blocks(result)

        for tag, body in blocks:
            if tag in ("python", "py"):
                _record_mutation_extraction("prompt_forge.code_mutation_extraction", "parsed")
                return body.strip(), True

        candidates = [
            body
            for tag, body in blocks
            if not tag.startswith("json")
            and not body.lstrip().startswith(("json", "{"))
            and len(body.strip()) > 100
        ]
        if candidates:
            chosen = max(candidates, key=lambda b: len(b.strip())).strip()
            # Byte-identical to the pre-TD-21.27 behaviour for a bare fence
            # whose language tag landed as the body's first LINE rather than
            # on the opening delimiter (some replies fence that way).
            lines = chosen.split("\n")
            if lines and lines[0].strip() in ("python", "py"):
                chosen = "\n".join(lines[1:]).strip()
            _record_mutation_extraction("prompt_forge.code_mutation_extraction", "parsed")
            return chosen, True

        log.warning("Could not extract code mutation from response, returning original")
        _record_mutation_extraction("prompt_forge.code_mutation_extraction", "failed")
        return original, False

    def _git_commit_file(self, path: Path, message: str) -> None:
        """Git add + commit exactly one file (no sweep of other staged changes)."""
        self._git_commit_paths([path], message)
