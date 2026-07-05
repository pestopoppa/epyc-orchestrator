"""Prompt hot-swap resolver: file → fallback constant, with A/B variant support.

Same pattern as _resolve_tools()/_resolve_rules() in builder.py: uncached file
read (~1ms) so edits take effect on the next request without API restart.

Usage:
    from src.prompt_builders.resolver import resolve_prompt

    _MY_FALLBACK = "You are a {role}. Answer: {question}"

    def build_my_prompt(role, question):
        return resolve_prompt(
            "my_prompt", _MY_FALLBACK,
            role=role, question=question,
        )

Variant selection (A/B testing):
    # Per-prompt: PROMPT_VARIANT__my_prompt=v2
    # Global:     PROMPT_VARIANT=v2
    # Creates lookup: orchestration/prompts/my_prompt.v2.md
"""

from __future__ import annotations

import logging
import os
import re
import contextlib
import contextvars
from collections.abc import Iterator
from pathlib import Path

from src.roles import Role

_log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "orchestration" / "prompts"
_PROMPT_DIR_OVERRIDE: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "orchestrator_prompt_dir_override",
    default=None,
)


@contextlib.contextmanager
def prompt_dir_override(prompt_dir: str | Path | None) -> Iterator[None]:
    """Temporarily resolve prompts from a request-local prompt tree."""
    if prompt_dir is None:
        yield
        return
    resolved = Path(prompt_dir).resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"prompt_dir override is not a directory: {resolved}")
    token = _PROMPT_DIR_OVERRIDE.set(resolved)
    try:
        yield
    finally:
        _PROMPT_DIR_OVERRIDE.reset(token)


def current_prompt_dir() -> Path:
    """Return the active prompt root for this request/context."""
    return _PROMPT_DIR_OVERRIDE.get() or PROMPT_DIR


class _SafeDict(dict):
    """Dict that returns '{key}' for missing keys instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _safe_format(template: str, variables: dict[str, str]) -> str:
    """Format a template string, leaving unmatched placeholders intact.

    Handles malformed templates gracefully (no crash on bad syntax).
    """
    try:
        return template.format_map(_SafeDict(variables))
    except (ValueError, IndexError):
        # Malformed format string (e.g. unmatched braces) — return as-is
        _log.debug("Malformed template, returning raw: %.80s", template)
        return template


def _get_variant(name: str) -> str | None:
    """Get variant suffix for a prompt name.

    Priority:
        1. PROMPT_VARIANT__{name} (per-prompt, e.g. PROMPT_VARIANT__architect_investigate=v2)
        2. PROMPT_VARIANT (global, e.g. PROMPT_VARIANT=v2)
        3. None (no variant)
    """
    # Per-prompt override (dots in name replaced with _ for env var compat)
    env_key = f"PROMPT_VARIANT__{name.replace('.', '_').replace('/', '_')}"
    variant = os.environ.get(env_key)
    if variant:
        return variant

    # Global variant
    variant = os.environ.get("PROMPT_VARIANT")
    if variant:
        return variant

    return None


def _path_within(base_dir: Path, filename: str) -> Path | None:
    """Return a prompt path only if it stays under base_dir."""
    base_resolved = base_dir.resolve(strict=False)
    candidate = (base_resolved / filename).resolve(strict=False)
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        _log.warning("Rejected prompt path escape: base=%s candidate=%s", base_resolved, candidate)
        return None
    return candidate


def _read_template(path: Path | None, *, label: str) -> str | None:
    if path is None:
        return None
    try:
        template = path.read_text()
    except OSError:
        _log.debug("Prompt file not found: %s", path)
        return None
    if not template.strip():
        _log.warning("Prompt provenance=%s path=%s empty; using next fallback", label, path)
        return None
    _log.debug("Prompt provenance=%s path=%s", label, path)
    return template


def resolve_prompt(
    name: str,
    fallback: str,
    *,
    variant: str | None = None,
    subdir: str = "",
    **template_vars: str,
) -> str:
    """Resolve a prompt: file (variant) -> file (default) -> fallback constant.

    File read is uncached (~1ms) to enable hot-swap: edit the .md file and the
    next request picks it up without restarting the API.

    Args:
        name: Prompt name (e.g. "architect_investigate", "frontdoor").
        fallback: Fallback prompt string if no file found.
        variant: Explicit variant override (takes priority over env vars).
        subdir: Subdirectory under PROMPT_DIR (e.g. "roles").
        **template_vars: Variables to interpolate into the template.

    Returns:
        Resolved and interpolated prompt string.
    """
    prompt_dir = current_prompt_dir()
    base_dir = prompt_dir / subdir if subdir else prompt_dir

    # Determine variant
    effective_variant = variant or _get_variant(name)

    # Try variant file first
    if effective_variant:
        variant_path = _path_within(base_dir, f"{name}.{effective_variant}.md")
        template = _read_template(variant_path, label="variant")
        if template is not None:
            return _safe_format(template, template_vars) if template_vars else template

    # Try default file
    default_path = _path_within(base_dir, f"{name}.md")
    template = _read_template(default_path, label="default")
    if template is not None:
        # Anti-self-correction for worker roles (primary path).
        # Without this, models generate 3x rewrites (389 tokens).
        if name.startswith("worker_"):
            template += '\n\nGive ONE answer. Do NOT self-correct, revise, or produce multiple versions. Write your final answer ONCE.'
        return _safe_format(template, template_vars) if template_vars else template

    # Try family fallback using canonical role truth first, then a structural
    # fallback for names that still only exist as legacy aliases.
    candidate_families: list[str] = []
    normalized = Role.from_string(name)
    if normalized is not None:
        family = normalized.value
        if family != name:
            candidate_families.append(family)
    if "_" in name:
        family = name.rsplit("_", 1)[0] + "_general"
        if family != name and family not in candidate_families:
            candidate_families.append(family)
    for family in candidate_families:
        family_path = _path_within(base_dir, f"{family}.md")
        template = _read_template(family_path, label="family")
        if template is not None:
            # Anti-self-correction: worker roles via family fallback
            # generate 3x rewrites ("Let me clarify", "I apologize").
            if name.startswith("worker_"):
                template += "\n\nGive ONE answer. Do NOT self-correct, revise, or produce multiple versions. Do NOT say \"Let me clarify\" or \"I apologize\"."
            return _safe_format(template, template_vars) if template_vars else template

    # Fallback to constant
    _log.debug("Prompt provenance=fallback name=%s subdir=%s", name, subdir)
    return _safe_format(fallback, template_vars) if template_vars else fallback


_DIRECT_ANSWER_ROLES = frozenset({Role.FRONTDOOR.value, Role.WORKER_GENERAL.value})

_TERSE_PREFIX = "Answer with ONLY the answer. No explanation.\n\n"
_LIST_PREFIX = "Respond with only the requested items, comma-separated.\n\n"

# Narrow: "What is <arithmetic>" only — excludes "Solve", "Calculate" etc.
# that match coding/physics/reasoning questions.
_WHAT_IS_ARITH = re.compile(
    r"^what is\s+\d[\d\s+\-*/^().]+\d", re.IGNORECASE
)
_LIST_EXACT = re.compile(r"^(list|name)\s+exactly\b", re.IGNORECASE)


def get_direct_answer_prefix(role: str, question: str = "") -> str:
    """Return a concise-answer directive for roles that need bare output.

    Used by _try_cheap_first in chat.py to prepend a formatting directive.
    Very selective: only fires for pure arithmetic ("What is 2+3") and
    explicit list requests ("List exactly ..."). Default is NO prefix.
    """
    normalized = Role.from_string(role)
    role_value = normalized.value if normalized is not None else role
    if role_value not in _DIRECT_ANSWER_ROLES or not question:
        return ""
    q = question.strip()
    if _WHAT_IS_ARITH.search(q):
        return _TERSE_PREFIX
    if _LIST_EXACT.search(q):
        return _LIST_PREFIX
    return ""
