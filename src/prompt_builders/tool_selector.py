"""Per-request tool selection for the REPL root prompt (codemode-ab probe, arm C). DEFAULT OFF.

Question (operator-approved 2026-11-17): does a per-request, decision-model-selected subset of
tool schemas in the INITIAL root prompt beat today's static tool list plus ``list_tools()``?

Two selector variants, both behind ONE flag, ``ORCHESTRATOR_TOOL_SELECTOR``:

* ``off`` (default, also unset/empty/anything else): :func:`maybe_selected_tools_block` returns
  ``None`` and ``PromptBuilder`` keeps its static tools text byte-for-byte. No live behaviour changes.
* ``embed`` (C1, embedding retrieval, the "REPL-retrieval alternative"): tool descriptions are
  embedded once with the document-side prefix, the request with the query-side prefix
  (``context_search.query_text`` / ``document_text``, i.e. the EG2 prefixes of ``embedding_dims``),
  and the top-k by cosine win. The embedder is any ``context_search.DenseEmbedder``.
* ``oneshot`` (C2, Jev-style one-shot classifier): ONE typed-decision call
  (``src.typed_decisions.run_typed_decisions``, value-only JSON, one NOUL question per tool GROUP)
  decides which groups the request needs; tools of flagged groups are ranked by lexical overlap and
  the top-k kept. The decision model is whatever ``primitives.llm_call`` is bound to (the worker).
  Abstain/failure falls back to lexical top-k and is recorded (never silent).

Wiring: a caller registers a provider with :func:`set_selector_provider`; the hook
``maybe_selected_tools_block(task)`` is called from ``PromptBuilder.build_root_lm_prompt`` and is a
no-op unless the flag is on AND a provider is set. Nothing here touches ``src/repl_environment``.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

ENV_FLAG = "ORCHESTRATOR_TOOL_SELECTOR"
ENV_K = "ORCHESTRATOR_TOOL_SELECTOR_K"
VARIANT_EMBED = "embed"
VARIANT_ONESHOT = "oneshot"
DEFAULT_K = 8

# REPL contract lines that stay in every selected prompt: they are the calling convention, not tools
# the selector chooses. Same wording as the CALL/TOOL/list_tools/FINAL lines of DEFAULT_ROOT_LM_TOOLS.
CORE_REPL_LINES = (
    "- `context`: str - full input. `artifacts`: dict - store intermediate results between turns.",
    "- `CALL(name, **kw)`: invoke a tool, returns JSON string. `TOOL(name, **kw)`: returns a Python object.",
    "- `list_tools()`: discover ALL available tools (use only if none of the tools below fits).",
    "- `FINAL(answer)`: signal completion with the final answer. REQUIRED for every task.",
)


def selector_variant(environ: Mapping[str, str] | None = None) -> str:
    """``"embed"`` | ``"oneshot"`` | ``"off"`` (default; any other value is off)."""
    env = os.environ if environ is None else environ
    v = str(env.get(ENV_FLAG, "")).strip().lower()
    return v if v in (VARIANT_EMBED, VARIANT_ONESHOT) else "off"


def selector_k(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        return max(1, int(str(env.get(ENV_K, DEFAULT_K))))
    except ValueError:
        return DEFAULT_K


@dataclass
class SelectionResult:
    variant: str
    k: int
    selected: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    selector_in_tokens: int = 0
    selector_out_tokens: int = 0
    fallback: str | None = None  # why the primary path was not used (never silent)
    detail: dict[str, Any] = field(default_factory=dict)


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("_", " "))


def tool_doc(tool: Mapping[str, Any]) -> str:
    """One embeddable/readable line per tool: name, description, parameter names."""
    params = ", ".join((tool.get("parameters") or {}).keys())
    return f"{tool['name']}({params}): {tool.get('description', '')}".strip()


def lexical_scores(query: str, tools: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Deterministic IDF-weighted token overlap; the C2 within-group ranker and the fallback."""
    docs = {t["name"]: set(_tokens(tool_doc(t))) for t in tools}
    n = max(1, len(docs))
    df: dict[str, int] = {}
    for toks in docs.values():
        for w in toks:
            df[w] = df.get(w, 0) + 1
    q = set(_tokens(query))
    return {
        name: sum(math.log(1 + n / df[w]) for w in (q & toks))
        for name, toks in docs.items()
    }


def _topk(scores: Mapping[str, float], order: Sequence[str], k: int) -> list[str]:
    # ties broken by registry order so the selection is deterministic
    pos = {n: i for i, n in enumerate(order)}
    return sorted(scores, key=lambda n: (-scores[n], pos.get(n, 1 << 30)))[:k]


class EmbeddingToolSelector:
    """C1. ``embedder``: a ``context_search.DenseEmbedder`` (``model_id`` + ``embed(texts, ...)``)."""

    def __init__(self, embedder: Any, tools: Sequence[Mapping[str, Any]], *, timeout_s: float = 120.0):
        from src.repl_environment.context_search import document_text

        self.embedder = embedder
        self.tools = list(tools)
        self.names = [t["name"] for t in self.tools]
        self._timeout = timeout_s
        res = embedder.embed(
            [document_text(embedder.model_id, tool_doc(t)) for t in self.tools],
            timeout_s=timeout_s, admission_wait_s=timeout_s,
        )
        self._vecs = _unit(res.vectors) if res.vectors is not None else None
        self._doc_reason = res.reason

    def select(self, query: str, k: int = DEFAULT_K) -> SelectionResult:
        from src.repl_environment.context_search import query_text

        t0 = time.perf_counter()
        if self._vecs is None:
            sc = lexical_scores(query, self.tools)
            return SelectionResult(VARIANT_EMBED, k, _topk(sc, self.names, k), sc,
                                   (time.perf_counter() - t0) * 1e3, fallback=f"lexical:{self._doc_reason}")
        res = self.embedder.embed([query_text(self.embedder.model_id, query)],
                                  timeout_s=self._timeout, admission_wait_s=self._timeout)
        if res.vectors is None:
            sc = lexical_scores(query, self.tools)
            return SelectionResult(VARIANT_EMBED, k, _topk(sc, self.names, k), sc,
                                   (time.perf_counter() - t0) * 1e3, fallback=f"lexical:{res.reason}")
        q = _unit(res.vectors)[0]
        sims = (self._vecs @ q).tolist()
        sc = dict(zip(self.names, (float(s) for s in sims)))
        return SelectionResult(VARIANT_EMBED, k, _topk(sc, self.names, k), sc,
                               (time.perf_counter() - t0) * 1e3,
                               detail={"embedder": self.embedder.model_id})


def _unit(vectors: Any) -> Any:
    import numpy as np

    a = np.asarray(vectors, dtype="float32")
    if a.ndim == 1:
        a = a[None, :]
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(n > 0, n, 1.0)


class OneShotToolSelector:
    """C2. ``groups``: group label -> tool names; ``group_desc``: group label -> one-line meaning.

    One ``run_typed_decisions`` call (JSON mode, value-only) asks, per group, whether the request
    needs a tool from it (NOUL). ``primitives.llm_call`` is the decision model seam; it should
    expose ``last_usage`` (``{"prompt_tokens", "completion_tokens"}``) for token accounting."""

    def __init__(self, primitives: Any, tools: Sequence[Mapping[str, Any]],
                 groups: Mapping[str, Sequence[str]], group_desc: Mapping[str, str], *, role: str = "worker"):
        self.primitives = primitives
        self.tools = list(tools)
        self.names = [t["name"] for t in self.tools]
        self.groups = {g: list(m) for g, m in groups.items()}
        self.group_desc = dict(group_desc)
        self.role = role

    def questions(self) -> list[Any]:
        from src.typed_decisions.types import Question, QuestionKind

        return [
            Question(id=f"needs_{g}", kind=QuestionKind.NOUL,
                     text=f"Does completing the request need a tool from group '{g}' ({self.group_desc.get(g, g)})?")
            for g in self.groups
        ]

    def select(self, query: str, k: int = DEFAULT_K) -> SelectionResult:
        from src.typed_decisions import run_typed_decisions

        t0 = time.perf_counter()
        before = dict(getattr(self.primitives, "usage_total", {}) or {})
        result = run_typed_decisions(self.primitives, state=f"REQUEST:\n{query}", questions=self.questions(),
                                     role=self.role, mode="json", max_retries=0, value_only=True)
        after = dict(getattr(self.primitives, "usage_total", {}) or {})
        tin = int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0))
        tout = int(after.get("completion_tokens", 0)) - int(before.get("completion_tokens", 0))
        flagged = [g for g in self.groups if any(d.question_id == f"needs_{g}" and d.value is True
                                                  for d in result.decisions)]
        sc = lexical_scores(query, self.tools)
        fallback = None
        if not flagged:
            fallback = "no_group_flagged:" + (result.failures[0].reason if result.failures else "empty")
            cand = sc
        else:
            allowed = {n for g in flagged for n in self.groups[g]}
            cand = {n: s for n, s in sc.items() if n in allowed}
        return SelectionResult(VARIANT_ONESHOT, k, _topk(cand, self.names, k), cand,
                               (time.perf_counter() - t0) * 1e3, tin, tout, fallback,
                               {"flagged_groups": flagged, "n_failures": len(result.failures)})


def render_tools_block(tools: Sequence[Mapping[str, Any]], selected: Sequence[str]) -> str:
    """The selected-tools replacement for the static tools text (same ``CALL`` convention)."""
    by = {t["name"]: t for t in tools}
    lines = ["## Tools (selected for this request)", *CORE_REPL_LINES, "", "**Selected tools** (call via CALL(name, **kw)):"]
    for name in selected:
        t = by.get(name)
        if t is None:
            continue
        ps = []
        for pn, spec in (t.get("parameters") or {}).items():
            ps.append(f"{pn}: {spec.get('type', 'string')}{'' if spec.get('required') else '?'}")
        lines.append(f"- `{name}({', '.join(ps)})`: {t.get('description', '')}")
    return "\n".join(lines)


# --- builder hook ----------------------------------------------------------------------------------
Provider = Callable[[str], "str | None"]
_PROVIDER: Provider | None = None


def set_selector_provider(provider: Provider | None) -> None:
    """Register ``provider(task_text) -> tools_block | None``. ``None`` unregisters."""
    global _PROVIDER
    _PROVIDER = provider


def maybe_selected_tools_block(task: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Hook for ``PromptBuilder``: None (static tools text kept) unless flag on AND provider set."""
    if selector_variant(environ) == "off" or _PROVIDER is None:
        return None
    try:
        return _PROVIDER(task)
    except Exception:  # noqa: BLE001 - a broken selector must never break prompt building
        return None
