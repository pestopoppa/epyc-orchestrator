"""Delegation config + valid-role registry + recursion-depth helpers.

Extracted from src/api/routes/chat_delegation.py during the 2026-05-22 Task-C
Phase 1 refactor. chat_delegation.py re-exports every public name here so test
patches against src.api.routes.chat_delegation.* keep working.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from src.env_parsing import env_int as _env_int
from src.roles import Role
from src.registry.stack_priors import live_stack_role_records


_VALID_DELEGATE_ROLE_FALLBACKS = frozenset(
    {
        "coder_escalation",
        "worker_summarize",
        "worker_general",
        "worker_math",
        "worker_vision",
        "vision_escalation",
    }
)


def _normalize_delegate_role(role: object) -> str:
    """Normalize legacy/model-generated delegate labels to live stack roles."""
    if not isinstance(role, str):
        return "coder_escalation"
    normalized = Role.from_string(role)
    if normalized is not None:
        return normalized.value
    if role in {"worker_coder", "worker_code"}:
        return Role.CODER_ESCALATION.value
    return role


def _valid_delegate_roles() -> frozenset[str]:
    """Return the live delegate-role allowlist from stack priors."""
    live_roles = live_stack_role_records()
    allowed = {role for role in _VALID_DELEGATE_ROLE_FALLBACKS if role in live_roles}
    return frozenset(allowed or _VALID_DELEGATE_ROLE_FALLBACKS)


# Thread-local delegation depth counter to detect re-entrance
# (specialist escalating back to the architect starts a fresh loop counter)
_delegation_local = threading.local()


def _get_delegation_depth() -> int:
    return getattr(_delegation_local, "depth", 0)



@dataclass(frozen=True)
class DelegationConfig:
    specialist_turn_n_tokens: int
    specialist_turn_n_tokens_summary: int
    specialist_turn_n_tokens_code: int
    specialist_turn_n_tokens_default: int
    forced_synthesis_n_tokens: int
    specialist_max_turns_react: int
    specialist_max_turns_repl: int
    specialist_max_seconds: float
    total_max_seconds: float
    skip_synthesis_on_timeout: bool
    trace_enabled: bool
    summarize_long_reports: bool
    summarize_report_chars: int
    summarize_n_tokens: int
    specialist_question_chars: int
    specialist_brief_chars: int
    specialist_context_chars: int
    specialist_corpus_context: bool
    compact_specialist_prompt: bool
    report_handles: bool
    report_handle_chars: int
    architect_decision_n_tokens_override: int
    architect_compute_n_tokens_override: int

    @classmethod
    def from_env(cls) -> "DelegationConfig":
        return cls(
            specialist_turn_n_tokens=_env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_TURN_N_TOKENS", 256),
            specialist_turn_n_tokens_summary=_env_int(
                "ORCHESTRATOR_DELEGATION_SPECIALIST_TURN_N_TOKENS_SUMMARY", 192
            ),
            specialist_turn_n_tokens_code=_env_int(
                "ORCHESTRATOR_DELEGATION_SPECIALIST_TURN_N_TOKENS_CODE", 768
            ),
            specialist_turn_n_tokens_default=_env_int(
                "ORCHESTRATOR_DELEGATION_SPECIALIST_TURN_N_TOKENS_DEFAULT", 224
            ),
            forced_synthesis_n_tokens=max(
                64,
                _env_int("ORCHESTRATOR_DELEGATION_FORCED_SYNTHESIS_N_TOKENS", 128),
            ),
            specialist_max_turns_react=max(
                1,
                _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_MAX_TURNS_REACT", 3),
            ),
            specialist_max_turns_repl=max(
                1,
                _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_MAX_TURNS_REPL", 4),
            ),
            specialist_max_seconds=float(
                max(10, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_MAX_SECONDS", 45))
            ),
            total_max_seconds=float(
                max(20, _env_int("ORCHESTRATOR_DELEGATION_TOTAL_MAX_SECONDS", 110))
            ),
            skip_synthesis_on_timeout=os.environ.get(
                "ORCHESTRATOR_DELEGATION_SKIP_SYNTHESIS_ON_TIMEOUT", "1"
            ).strip().lower() in {"1", "true", "yes", "on"},
            trace_enabled=os.environ.get("ORCHESTRATOR_DELEGATION_TRACE", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            summarize_long_reports=os.environ.get(
                "ORCHESTRATOR_DELEGATION_SUMMARIZE_LONG_REPORTS", "1"
            ).strip().lower() in {"1", "true", "yes", "on"},
            summarize_report_chars=max(
                800,
                _env_int("ORCHESTRATOR_DELEGATION_SUMMARIZE_REPORT_CHARS", 2800),
            ),
            summarize_n_tokens=max(
                96,
                _env_int("ORCHESTRATOR_DELEGATION_SUMMARIZE_N_TOKENS", 220),
            ),
            specialist_question_chars=max(
                600,
                _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_QUESTION_CHARS", 2200),
            ),
            specialist_brief_chars=max(
                240,
                _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_BRIEF_CHARS", 700),
            ),
            specialist_context_chars=max(
                0,
                _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_CONTEXT_CHARS", 800),
            ),
            specialist_corpus_context=os.environ.get(
                "ORCHESTRATOR_DELEGATION_SPECIALIST_CORPUS_CONTEXT", "0"
            ).strip().lower() in {"1", "true", "yes", "on"},
            compact_specialist_prompt=os.environ.get(
                "ORCHESTRATOR_DELEGATION_COMPACT_SPECIALIST_PROMPT", "1"
            ).strip().lower() in {"1", "true", "yes", "on"},
            report_handles=os.environ.get(
                "ORCHESTRATOR_DELEGATION_REPORT_HANDLES", "1"
            ).strip().lower() in {"1", "true", "yes", "on"},
            report_handle_chars=max(
                1200,
                _env_int("ORCHESTRATOR_DELEGATION_REPORT_HANDLE_CHARS", 2600),
            ),
            architect_decision_n_tokens_override=_env_int(
                "ORCHESTRATOR_DELEGATION_ARCHITECT_DECISION_N_TOKENS", -1
            ),
            architect_compute_n_tokens_override=_env_int(
                "ORCHESTRATOR_DELEGATION_ARCHITECT_COMPUTE_N_TOKENS", -1
            ),
        )



def _delegation_config() -> DelegationConfig:
    return DelegationConfig.from_env()


def _delegation_specialist_turn_token_cap(
    delegate_mode: str,
    question: str,
    brief: str,
    delegate_to: str,
    cfg: DelegationConfig | None = None,
) -> int:
    """Task-aware specialist turn cap to reduce over-generation latency."""
    cfg = cfg or _delegation_config()
    base = cfg.specialist_turn_n_tokens
    q = f"{question}\n{brief}".lower()
    summary_signals = ("summarize", "summary", "extract key", "bullet")
    coding_signals = (
        "implement", "write code", "class ", "function", "refactor", "patch",
        "multi-file", "api", "middleware", "algorithm",
        "usaco", "codeforces", "leetcode", "sample input", "input format",
        "output format", "stdin", "write a python",
    )
    if delegate_to == "worker_summarize" or any(s in q for s in summary_signals):
        base = min(base, cfg.specialist_turn_n_tokens_summary)
    elif any(s in q for s in coding_signals):
        base = max(base, cfg.specialist_turn_n_tokens_code)
    else:
        base = min(base, cfg.specialist_turn_n_tokens_default)
    return max(96, base)
