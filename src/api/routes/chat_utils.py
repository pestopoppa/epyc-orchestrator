"""Utility functions and constants for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: token estimation, stub detection, answer resolution,
output quality heuristics, format enforcement, and shared constants.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.features import features
from src.prompt_builders import (
    build_formalizer_prompt,
    detect_format_constraints,
)
from src.roles import Role

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives
    from src.repl_environment.types import ExecutionResult
    from src.services.document_preprocessor import PreprocessingResult


# ── Role-specific timeouts (Phase 1b: KV cache bug mitigation) ──────────
# Sourced from centralized config (src.config.TimeoutsConfig).
from src.config import get_config as _get_config

_timeouts = _get_config().timeouts
ROLE_TIMEOUTS: dict[str, int] = _timeouts.role_timeouts_dict()
DEFAULT_TIMEOUT_S: int = _timeouts.default_request


@dataclass
class RoutingResult:
    """Encapsulates all routing decisions made before execution.

    Created by _route_request(), consumed by mode handlers and response builder.
    Frozen after creation — routing is a read-only decision.
    """

    task_id: str
    task_ir: dict
    use_mock: bool
    routing_decision: list = field(default_factory=list)
    routing_strategy: str = ""
    formalization_applied: bool = False
    timeout_s: int = DEFAULT_TIMEOUT_S
    document_result: PreprocessingResult | None = None
    tool_required: bool = False  # True when task needs tools (file search, computation)
    tool_hint: str | None = None  # Specific tool name if deterministic
    skill_context: str = ""  # Formatted skill text for prompt injection (SkillRL §3.2)
    skill_ids: list[str] = field(default_factory=list)  # IDs of retrieved skills
    # Factual-risk scoring (populated in shadow/enforce mode, empty when off)
    factual_risk_score: float = 0.0
    factual_risk_band: str = ""
    # Difficulty-adaptive signal (populated in shadow/enforce mode, empty when off)
    difficulty_score: float = 0.0
    difficulty_band: str = ""
    # Estimated pre-inference cost (relative units: tier_weight × estimated_tokens / 1M)
    estimated_cost: float = 0.0
    # Trinity tri-role axis (TR-2.1 of tri-role-coordinator-architecture.md).
    # Per-call assignment {"thinker", "worker", "verifier"}, orthogonal to
    # routing_decision (which selects the model). Default "worker" for backward
    # compat with pre-TR-2 callers. Logged in shadow mode regardless of feature
    # flag; only acted on when ORCHESTRATOR_ROLE_AWARE_ROUTING=1.
    assigned_role: str = "worker"

    @property
    def role(self) -> str:
        """Primary role for this request."""
        if self.routing_decision:
            return str(self.routing_decision[0])
        return str(Role.FRONTDOOR)

    def timeout_for_role(self, role: str) -> int:
        """Get timeout for a specific role (used during escalation)."""
        if role == "worker_explore":
            normalized = "worker_general"
        elif role == "worker_fast":
            normalized = "worker_fast"
        else:
            normalized = str(Role.from_string(role) or role)
        return ROLE_TIMEOUTS.get(normalized, DEFAULT_TIMEOUT_S)


# Three-stage summarization configuration — values sourced from centralized config
_chat_cfg = _get_config().chat
THREE_STAGE_CONFIG = {
    "enabled": True,
    "threshold_tokens": _chat_cfg.summarization_threshold_tokens,
    "multi_doc_discount": _chat_cfg.multi_doc_discount,
    "stage1_role": Role.FRONTDOOR,
    "stage2_role": Role.INGEST_LONG_CONTEXT,
    "compression": {
        "enabled": _chat_cfg.compression_enabled,
        "min_chars": _chat_cfg.compression_min_chars,
        "target_ratio": _chat_cfg.compression_target_ratio,
        "stage1_context_limit": _chat_cfg.stage1_context_limit,
    },
}

# Backwards compatibility alias
TWO_STAGE_CONFIG = THREE_STAGE_CONFIG

# Qwen chat-template stop token — prevents runaway generation past turn boundary
QWEN_STOP = _get_config().llm.qwen_stop_token

# Long context exploration configuration
LONG_CONTEXT_CONFIG = {
    "enabled": _chat_cfg.long_context_enabled,
    "threshold_chars": _chat_cfg.long_context_threshold_chars,
    "max_turns": _chat_cfg.long_context_max_turns,
}

def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough: 4 chars per token)."""
    return len(text) // 4


# ── Chat template application (per-role) ───────────────────────────────
#
# chat.py's direct-mode path used to hardcode the Qwen3 ChatML template
# unconditionally before sending to /completion. That broke silently after
# the 2026-05-08 worker_general swap to gemma-4-26B-A4B-it (gemma uses
# <start_of_turn>...<end_of_turn>, not <|im_start|>...<|im_end|>). Routes
# to worker_general/worker_summarize (and worker_explore aliases via
# worker_general) started producing
# 0 tokens — gemma4 saw the Qwen markers as random tokens and refused to
# generate, after which the orchestrator escalated to frontdoor (real cost:
# ~60s of dead-loop time per request).
#
# This helper inspects the chosen role's model name (from the live registry)
# and emits the correct turn-marker wrapper. Unknown model families return
# the prompt unchanged (defensive: never make things worse than the
# pass-through baseline).
#
# Family detection is case-insensitive substring match on `model.name`.
# Note: llama-server's /chat/completions endpoint applies the Jinja template
# server-side from the GGUF metadata — that's the more principled path. We
# keep this orchestrator-side wrapper because the direct path goes through
# /completion (which does not), and rewriting that backend is a larger
# refactor.

_TEMPLATE_QWEN_CHATML = (
    "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
)
# Gemma 2 / Gemma 3 use the classic `<start_of_turn>` / `<end_of_turn>`
# turn markers.
_TEMPLATE_GEMMA3 = (
    "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
)
# Gemma 4 (e.g. gemma-4-26B-A4B-it) actually ships a NEW multi-channel
# template with `<|turn>` / `<turn|>` markers + a `<|channel>thought`
# reasoning channel. Verified 2026-05-22 by reading the GGUF's embedded
# chat_template + calling the live server's /apply-template endpoint.
# HOWEVER: empirically (2026-05-22), feeding the proper gemma-4 template
# to ik_llama.cpp's /completion endpoint TIMES OUT with 0 tokens (tested
# with and without the channel prefix). The /v1/chat/completions
# endpoint with --jinja works fine (returns answers in 0.07s).
#
# Until worker_general's backend is switched to /chat/completions, fall
# back to the gemma-3 template — it produces output with some
# `<|channel>` artifacts in the response (visible bytes that need
# stripping downstream) but at least the request completes. The
# completion-quality issue is documented as a follow-up; see deferred
# section in the 2026-05-22 progress entry.
_TEMPLATE_GEMMA4 = _TEMPLATE_GEMMA3
_TEMPLATE_LLAMA3 = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def _detect_template_family(model_name: str) -> str:
    """Return a family identifier for `model_name`, or 'unknown'.

    Match order matters: more-specific variants first (e.g. gemma-4
    before gemma).
    """
    if not model_name:
        return "unknown"
    n = model_name.lower()
    # Gemma 4 — uses the new <|turn>/<channel|> template, different from
    # gemma 2/3. Match BEFORE generic "gemma" so we route to the right
    # family.
    if "gemma-4" in n or "gemma4" in n:
        return "gemma4"
    # Gemma 2/3 — classic start_of_turn/end_of_turn template.
    if "gemma" in n:
        return "gemma3"
    # MiniMax-M2: uses ChatML-ish but with different markers; pass-through for now
    if "minimax" in n:
        return "minimax"
    # Llama 3.x — needs the header-id template
    if "llama-3" in n or "llama3" in n or "meta-llama-3" in n:
        return "llama3"
    # Qwen family (Qwen2.5, Qwen3, Qwen3.5, Qwen3.6, Qwen3-Next, Qwen-VL, etc.)
    # This is the broadest match — keep it last so more specific families win.
    if "qwen" in n or "deepseek-r1-distill-qwen" in n:
        return "qwen"
    # Phi-4 uses Phi-4 chat template (close to ChatML); pass-through default
    if "phi-4" in n or "phi4" in n:
        return "phi"
    return "unknown"


def _wrap_for_family(user_prompt: str, family: str) -> str:
    """Apply turn-marker wrap for a known family; pass-through otherwise."""
    if family == "gemma4":
        return _TEMPLATE_GEMMA4.format(user=user_prompt)
    if family == "gemma3" or family == "gemma":  # alias kept for back-compat
        return _TEMPLATE_GEMMA3.format(user=user_prompt)
    if family == "llama3":
        return _TEMPLATE_LLAMA3.format(user=user_prompt)
    if family in ("qwen", "minimax", "phi"):
        # MiniMax + Phi-4 both accept Qwen-style ChatML markers per empirical
        # check. If a future model breaks, add a dedicated entry above.
        return _TEMPLATE_QWEN_CHATML.format(user=user_prompt)
    return user_prompt


def _is_already_templated(user_prompt: str) -> bool:
    """True if `user_prompt` already contains any known turn markers."""
    return (
        "<|im_start|>" in user_prompt
        or "<start_of_turn>" in user_prompt  # gemma 2/3
        or "<|turn>" in user_prompt  # gemma 4
        or "<|begin_of_text|>" in user_prompt
    )


def apply_chat_template_for_model(model_name: str, user_prompt: str) -> str:
    """Wrap `user_prompt` with markers appropriate for `model_name`.

    Use this when the caller already knows the model name (e.g. worker_pool
    has it via `WorkerConfig.model_path`'s basename). For callers that only
    know the role, use `apply_chat_template_for_role` instead.

    Returns the prompt unchanged if it's already templated or if the family
    is unknown. Never raises.
    """
    if not user_prompt or _is_already_templated(user_prompt):
        return user_prompt
    family = _detect_template_family(model_name)
    out = _wrap_for_family(user_prompt, family)
    if out is user_prompt and family == "unknown":
        log.debug(
            "apply_chat_template_for_model: no template for model=%s; passing through",
            model_name,
        )
    return out


def apply_chat_template_for_role(
    role_name: str,
    user_prompt: str,
    registry: "object | None" = None,
) -> str:
    """Wrap `user_prompt` with the role's chat-template turn markers.

    Looks up the role's model name from the registry and dispatches to
    `apply_chat_template_for_model`. Returns the original prompt unchanged
    if it's already templated, or wrapped with the Qwen ChatML fallback if
    registry/role lookup fails (preserves the legacy behavior used by the
    dominant Qwen-stack callers).

    Safe to call from chat.py and worker_pool; never raises.
    """
    if not user_prompt or _is_already_templated(user_prompt):
        return user_prompt
    canonical_role_name = str(Role.from_string(role_name) or role_name)
    if registry is None:
        # Legacy fallback: Qwen ChatML — matches pre-fix behavior for callers
        # that can't see the registry. The dominant stack is Qwen, so the
        # fallback is "right by default" for unknown-registry callers.
        return _TEMPLATE_QWEN_CHATML.format(user=user_prompt)
    try:
        role = registry.get_role(canonical_role_name)  # type: ignore[attr-defined]
        model_name = getattr(role.model, "name", "") or ""
    except Exception as exc:
        log.warning(
            "apply_chat_template_for_role: registry lookup failed for role=%s: %s — "
            "using Qwen ChatML fallback",
            role_name, exc,
        )
        return _TEMPLATE_QWEN_CHATML.format(user=user_prompt)
    return apply_chat_template_for_model(model_name, user_prompt)


def _is_stub_final(text: str) -> bool:
    """Detect when FINAL() arg is a stub pointing to printed output.

    Models often print their analysis via print(), then call
    FINAL("Analysis complete. See above.") — the real content
    is in result.output, not result.final_answer.
    """
    from src.classifiers import is_stub_final

    return is_stub_final(text)


def _strip_tool_outputs(text: str, tool_outputs: list[str]) -> str:
    """Strip known tool outputs from captured REPL output.

    Delegates to src.classifiers.output_parser.strip_tool_outputs.
    """
    from src.classifiers import strip_tool_outputs

    return strip_tool_outputs(text, tool_outputs)


# 2026-05-23: `strip_gemma4_channel_markers()` retired entirely.
# It was added 2026-05-22 (commit 3d69e97) as a stop-gap to clean
# `<|channel>thought\n<channel|>` artifacts that gemma-4-26B-A4B-it
# emitted into /completion responses. The /v1/chat/completions
# migration (commit 2c1711a, 2026-05-23) made the helper unreachable —
# all gemma-family worker roles now route via /v1/chat/completions
# which applies --jinja server-side and parses the multi-channel
# format cleanly. The wire-up in direct_stage.py was removed in
# ab889b1; this turn fully retires the helper to avoid carrying dead
# code. If a future gemma-family role ever lands on /completion, see
# wiki/chat-templates.md for the proper handling.


_FUNCTION_REPR_RE = re.compile(r"<(?:function|class|module) \w+ at 0x[0-9a-fA-F]+>")


def _resolve_answer(result: "ExecutionResult", tool_outputs: list[str] | None = None) -> str:
    """Extract the best answer from an ExecutionResult.

    Handles cases where the model prints content then uses a stub FINAL().
    Strips tool outputs (my_role, route_advice, list_dir) from captured output.
    Strips Python object repr strings that leak when model evaluates a bare name.
    """
    captured = result.output.strip() if result.output else ""
    final = result.final_answer or ""

    # Strip tool outputs from captured stdout
    if tool_outputs:
        captured = _strip_tool_outputs(captured, tool_outputs)

    # Guard: strip function/class repr strings from captured output.
    # These leak when the model's last REPL turn evaluates a bare function name
    # (e.g. `is_valid_parenthese`) without calling FINAL() — Python echoes
    # `<function is_valid_parenthese at 0x...>` which becomes the answer.
    if captured and _FUNCTION_REPR_RE.fullmatch(captured):
        captured = ""
    if final and _FUNCTION_REPR_RE.fullmatch(final.strip()):
        final = ""

    if captured and _is_stub_final(final):
        return captured
    elif captured and final and captured != final:
        # Prepend captured output if FINAL() doesn't already contain it
        if final not in captured:
            return f"{captured}\n\n{final}"
        return final
    else:
        return final


def _truncate_looped_answer(answer: str, prompt: str) -> str:
    """Truncate answer if prompt text reappears (loop detection).

    Delegates to src.classifiers.output_parser.truncate_looped_answer.
    """
    from src.classifiers import truncate_looped_answer

    return truncate_looped_answer(answer, prompt)


def _should_formalize(prompt: str) -> tuple[bool, str]:
    """Detect if the prompt has format constraints that need enforcement.

    Args:
        prompt: The user's prompt.

    Returns:
        Tuple of (should_formalize, format_spec_description).
    """
    if not features().output_formalizer:
        return False, ""

    constraints = detect_format_constraints(prompt)
    if constraints:
        return True, "; ".join(constraints)
    return False, ""


def _formalize_output(
    answer: str,
    prompt: str,
    format_spec: str,
    primitives: "LLMPrimitives",
) -> str:
    """Reformat an answer to satisfy detected format constraints.

    Uses the live general worker for fast reformatting.
    The answer content is correct — only format needs fixing.

    Args:
        answer: The correct-content answer to reformat.
        prompt: The original user prompt.
        format_spec: Description of format constraints to satisfy.
        primitives: LLM primitives for inference.

    Returns:
        Reformatted answer, or original if formalization fails.
    """
    formalizer_prompt = build_formalizer_prompt(answer, prompt, format_spec)
    try:
        result = primitives.llm_call(
            formalizer_prompt,
            role="worker_general",
            n_tokens=2000,
            skip_suffix=True,
        )
        reformatted = result.strip()
        if reformatted and len(reformatted) > 5:
            log.info(f"Formalized output for constraint: {format_spec}")
            return reformatted
        return answer
    except Exception as e:
        log.warning(f"Output formalization failed: {e}")
        return answer
