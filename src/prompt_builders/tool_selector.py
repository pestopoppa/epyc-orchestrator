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
    "- `list_tools()`: discover ALL available tools (always available; the tools below are picks that ADD to it).",
    "- `search_tools(query)` / `describe_tool(name)`: find a tool by purpose / read one schema without listing everything.",
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
    from src.prompt_builders.codemode_stubs import render_stubs_block, stubs_enabled

    if stubs_enabled():  # codemode fix 4 (ORCHESTRATOR_CODEMODE_STUBS, default off): typed Python stubs
        return render_stubs_block(tools, selected)
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


# --- DIRECT native-tool-call path (CM-1 follow-through: the probe found direct calls beat REPL code mode) ----
# The client-supplied OpenAI ``tools`` list of a ``x_tool_mode="client"`` chat completion is narrowed to the
# per-request top-k BEFORE it reaches the backend. Same flag, same variants, DEFAULT OFF.
NativeProvider = Callable[[str, "list[dict[str, Any]]", int], "list[str] | None"]
_NATIVE_PROVIDER: NativeProvider | None = None
_NATIVE_CACHE: dict[tuple[str, str, tuple[str, ...], int], list[str]] = {}
_NATIVE_CACHE_MAX = 128
# Last decision, for observability/tests (never read by production logic).
LAST_NATIVE_SELECTION: dict[str, Any] = {}


def set_native_selector_provider(provider: NativeProvider | None) -> None:
    """Register ``provider(query, tool_dicts, k) -> selected names | None``. ``None`` unregisters and clears the cache."""
    global _NATIVE_PROVIDER
    from src.prompt_builders import tool_picker as tp

    _NATIVE_PROVIDER = provider
    _NATIVE_CACHE.clear()
    tp.clear_frozen()
    tp.clear_selectors()


def _native_name(tool: Mapping[str, Any]) -> str | None:
    fn = tool.get("function") if isinstance(tool, Mapping) else None
    name = fn.get("name") if isinstance(fn, Mapping) else None
    return name if isinstance(name, str) and name else None


def openai_tool_to_dict(tool: Mapping[str, Any]) -> dict[str, Any] | None:
    """OpenAI ``{"type":"function","function":{...}}`` -> the selector's ``{name, description, parameters}``."""
    name = _native_name(tool)
    if name is None:
        return None
    fn = tool["function"]
    props = (fn.get("parameters") or {}).get("properties") or {}
    req = set((fn.get("parameters") or {}).get("required") or [])
    params = {pn: {"type": (ps or {}).get("type", "string"), "required": pn in req} for pn, ps in props.items()}
    return {"name": name, "description": str(fn.get("description") or ""), "parameters": params}


def last_user_query(messages: Sequence[Mapping[str, Any]]) -> str:
    """Text of the latest user message ('' when none); content parts are joined."""
    for m in reversed(list(messages)):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(str(p.get("text", "")) for p in c if isinstance(p, Mapping))
    return ""


def _forced_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, Mapping):
        fn = tool_choice.get("function")
        if isinstance(fn, Mapping) and isinstance(fn.get("name"), str):
            return fn["name"]
    return None


def _history_tool_names(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    used: set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            n = (tc.get("function") or {}).get("name") if isinstance(tc, Mapping) else None
            if isinstance(n, str):
                used.add(n)
    return used


def first_user_query(messages: Sequence[Mapping[str, Any]]) -> str:
    """Text of the FIRST user message ('' when none): the conversation's stable pick query."""
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            return last_user_query(list(messages)[: i + 1])
    return ""


@dataclass
class PickContext:
    """What the route needs after ``select_native_tools_ex``: ids for the log, the frozen-pick key and a re-pick closure."""

    request_id: str
    conversation: str
    key: str
    variant: str
    sent: list[str]
    history: list[str]
    original: list[dict[str, Any]]  # the client's FULL tool list (OpenAI shape)
    query: str
    repick: Callable[[str], list[str]]
    environ: Mapping[str, str] | None = None
    scope: str = ""


def select_native_tools(
    tools: list[dict[str, Any]] | None,
    messages: Sequence[Mapping[str, Any]],
    tool_choice: Any = None,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]] | None:
    """Narrow a client ``tools`` list with the hardened picker (see ``tool_picker``). Returns ``tools`` itself (same object) whenever
    the flag is off, the list is already <= k, there is no user query, ``tool_choice`` is ``"none"``, the picker is low-confidence,
    slow, failing or its breaker is open (fail open to the unpicked list), or anything goes wrong."""
    return select_native_tools_ex(tools, messages, tool_choice, environ)[0]


def select_native_tools_ex(
    tools: list[dict[str, Any]] | None,
    messages: Sequence[Mapping[str, Any]],
    tool_choice: Any = None,
    environ: Mapping[str, str] | None = None,
    *,
    scope: str = "",
) -> tuple[list[dict[str, Any]] | None, PickContext | None]:
    """:func:`select_native_tools` plus the :class:`PickContext` (None whenever the list was returned unpicked and unlogged).

    The pick is made once per conversation (on the first user turn) and FROZEN: later turns reuse it and may only ADD tools
    (called earlier, forced, recovered), appended after the frozen names, so the tool block stays a stable cacheable prefix.
    The core set is pinned outside k and ``list_tools`` / ``search_tools`` / ``describe_tool`` (server-executed) are always sent
    with a narrowed list, so the pick ADDS to discoverability and never replaces it."""
    from src.prompt_builders import tool_picker as tp

    if not tools or selector_variant(environ) == "off" or tool_choice == "none":
        return tools, None
    k = selector_k(environ)
    if len(tools) <= k:
        return tools, None
    query = first_user_query(messages).strip()
    if not query:
        return tools, None
    try:
        variant = selector_variant(environ)
        cfg = tp.PickerConfig.from_env(environ)
        dicts = [d for d in (openai_tool_to_dict(t) for t in tools) if d is not None]
        names = [d["name"] for d in dicts]
        key = tp.conversation_key(messages, names, scope)
        req_id = tp.new_request_id()
        provider = _NATIVE_PROVIDER
        if provider is not None and getattr(provider, "shadow_only", False):
            provider = None  # C2 never picks on the critical path
        scorer = getattr(provider, "scorer", None) if provider is not None else None
        frozen = tp.frozen_get(key)
        hit = frozen is not None
        if frozen is None:
            if provider is not None and scorer is None:  # legacy provider: names only (no scores, no adaptive k)
                chosen = list(provider(query, dicts, k) or [])
                if not chosen:
                    return tools, None
                core = tp.core_names(names, environ)
                res = tp.PickResult("narrow", [n for n in chosen if n not in core], core, len(chosen), len(chosen), "provider")
            else:
                res = tp.pick(query, dicts, scorer, cfg, environ)
            keep = set(res.picks) | set(res.core)
            frozen = tp.Frozen(res.mode, [n for n in names if n in keep] if res.mode != "full" else [], {})
            tp.frozen_put(key, frozen)
            sent_now = names if res.mode == "full" else frozen.names
            tp.log_event(tp.selection_row(req_id, key, variant, scope, res, sent_now, len(tools), cfg, query, frozen_hit=False), environ)
            tp.maybe_shadow_rerank(req_id, query, dicts, res.picks, cfg.k_max, environ)
            LAST_NATIVE_SELECTION.clear()
            LAST_NATIVE_SELECTION.update(variant=variant, k=res.k_eff, n_in=len(tools), n_out=len(sent_now), selected=list(res.picks),
                                         fallback=res.rung if res.rung not in ("none", "widen") else None, mode=res.mode, core=res.core,
                                         rung=res.rung, elapsed_ms=res.elapsed_ms)
        if frozen.mode == "full":
            return tools, None
        history = sorted(_history_tool_names(messages))
        adds = [n for n in history if n in names]
        forced = _forced_name(tool_choice)
        if forced and forced in names:
            adds.append(forced)
        tp.frozen_add(key, adds)
        by = {_native_name(t): t for t in tools if _native_name(t)}
        ordered = [by[n] for n in frozen.names if n in by]
        extra = [t for t in tools if _native_name(t) is None]
        narrowed = tp.server_tool_defs() + ordered + extra
        sent = [n for n in frozen.names if n in by]

        def repick(extra_query: str) -> list[str]:
            r = tp.pick(f"{query}\n{extra_query}", dicts, scorer, cfg, environ)
            return [n for n in r.picks if n in by]

        if hit:
            tp.log_event({"type": "selection_reuse", "request_id": req_id, "conversation": key, "n_tools_out": len(sent), "added": adds}, environ)
        ctx = PickContext(req_id, key, key, variant, sent, history, list(tools), query, repick, environ, scope)
        return narrowed, ctx
    except Exception:  # noqa: BLE001 - a broken selector must never break the request
        return tools, None


class NativeEmbedProvider:
    """C1 native provider: callable (legacy names API) plus ``scorer`` (scores, used by the hardened picker)."""

    def __init__(self, embedder: Any):
        from src.prompt_builders.tool_picker import embed_scorer

        self.embedder = embedder
        self.scorer = embed_scorer(embedder)

    def __call__(self, query: str, tools: list[dict[str, Any]], k: int) -> list[str] | None:
        scores, _fb, _emb = self.scorer(query, tools)
        return _topk(scores, [t["name"] for t in tools], k)


def make_native_provider(variant: str, *, embedder: Any = None, primitives: Any = None, role: str = "worker") -> NativeProvider:
    """Provider for :func:`set_native_selector_provider`. ``embed``: C1 over the request's own tools (needs an embedder); the
    selector is MEMOISED per (embedder, tool-registry version), not rebuilt per call. ``oneshot``: C2 with one NOUL question per
    tool; it is SHADOW ONLY (``shadow_only``): the picker never calls it on the request's critical path. A missing dependency falls
    back to lexical top-k (never silent: it is logged)."""
    if variant == VARIANT_EMBED and embedder is not None:
        return NativeEmbedProvider(embedder)

    def provider(query: str, tools: list[dict[str, Any]], k: int) -> list[str] | None:
        if variant == VARIANT_ONESHOT and primitives is not None:
            groups = {t["name"]: [t["name"]] for t in tools}
            desc = {t["name"]: t["description"][:120] for t in tools}
            return OneShotToolSelector(primitives, tools, groups, desc, role=role).select(query, k).selected
        return _topk(lexical_scores(query, tools), [t["name"] for t in tools], k)

    provider.shadow_only = variant == VARIANT_ONESHOT  # type: ignore[attr-defined]
    return provider


# --- REPL (code mode) provider and startup wiring ----------------------------------------------------------------------
def make_repl_provider(embedder: Any, tools_source: Callable[[], Sequence[Mapping[str, Any]]]) -> Provider:
    """Provider for :func:`set_selector_provider`: ``provider(task) -> selected-tools block | None``. Same picker as the native path
    (memoised C1 selector, core pin, adaptive k, family cap, ladder, 500 ms budget, breaker). The pick is frozen per task text. ``None``
    (fail open) leaves the static tools text untouched."""
    from src.prompt_builders import tool_picker as tp

    scorer = tp.embed_scorer(embedder)

    def provider(task: str) -> str | None:
        dicts = [dict(t) for t in tools_source()]
        if not dicts or not task.strip():
            return None
        names = [d["name"] for d in dicts]
        cfg = tp.PickerConfig.from_env()
        if len(dicts) <= selector_k():
            return None
        key = tp.conversation_key([{"role": "user", "content": task}], names, "repl")
        frozen = tp.frozen_get(key)
        if frozen is None:
            res = tp.pick(task, dicts, scorer, cfg)
            if res.rung in ("timeout", "error", "breaker_open"):
                tp.log_event(tp.selection_row(tp.new_request_id(), key, "embed", "repl", res, [], len(dicts), cfg, task, frozen_hit=False))
                return None  # fail open: the static text stays
            keep = set(res.picks) | set(res.core)
            chosen = [n for n in names if n in keep] if res.mode == "narrow" else (names if res.mode == "full" else list(res.core))
            frozen = tp.Frozen(res.mode, chosen)
            tp.frozen_put(key, frozen)
            tp.log_event(tp.selection_row(tp.new_request_id(), key, "embed", "repl", res, chosen, len(dicts), cfg, task, frozen_hit=False))
        return render_tools_block(dicts, frozen.names)

    return provider


def wire_default_providers(environ: Mapping[str, str] | None = None, *, embedder: Any = None,
                           tools_source: Callable[[], Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Startup wiring (``src/api/__init__.py::lifespan``). A no-op unless ``ORCHESTRATOR_TOOL_SELECTOR`` is on. Registers the native
    and REPL providers over the pooled embedder. Never raises and never calls the network: document embedding stays lazy (first
    request, under the picker's latency budget). With no embedder nothing is registered and the picker fails open to the full list."""
    if selector_variant(environ) == "off":
        return {"wired": False, "reason": "flag_off"}
    try:
        emb = embedder
        if emb is None:
            from src.repl_environment.context_search import pool_dense_embedder

            emb = pool_dense_embedder()
        if emb is None:
            return {"wired": False, "reason": "no_embedder"}
        set_native_selector_provider(make_native_provider(VARIANT_EMBED, embedder=emb))
        if tools_source is not None:
            set_selector_provider(make_repl_provider(emb, tools_source))
        return {"wired": True, "embedder": str(getattr(emb, "model_id", "")), "repl": tools_source is not None}
    except Exception as exc:  # noqa: BLE001 - startup must survive a broken picker
        return {"wired": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}
