"""Hardened per-request tool picker (SELECTOR_HARDENING.md), all behind ``ORCHESTRATOR_TOOL_SELECTOR`` (default off).

The embedding picker (C1: BGE through the pooled embedder) is THE picker. This module adds what the CODEMODE-AB probe
showed it needs (every probe miss was a generic file-I/O tool; the task-specific tool was always picked):

* **core pin** outside k: ``read_file``/``peek``, ``write_file``/``file_write_safe``, ``list_dir`` (and a small generic-capability
  vocabulary for clients that name them ``read``/``write``/``ls``...), never counted against k, never dropped;
* **adaptive k** in ``[K_MIN, K_MAX]`` from the score margin to the top pick, with a **per-family cap** (at most 3 tools sharing a
  name prefix such as ``syn_sku_*``) so one domain cannot crowd out the list;
* **low-confidence ladder**: embedder fallback or a flat score profile -> widen to K_MAX; still flat -> the FULL list when it has
  at most ``FULL_LIST_MAX`` (40) tools, else core + the server tools (``list_tools``/``search_tools``);
* **latency cap + circuit breaker**: the picking step has a wall-clock budget (500 ms). A timeout or error fails OPEN to the
  unpicked list; 3 consecutive failures open the breaker for 60 s (requests get the unpicked list, logged ``breaker_open``);
* **memoised selector** built once per tool-registry version (the tool documents' fingerprint), not once per call;
* **frozen pick per conversation**: chosen on the first user turn; later turns reuse it and may only ADD tools (called earlier,
  forced, recovered), appended at the end, so the tool block stays a stable cacheable prefix;
* **per-request event log** (JSONL): selected set, scores, k, rung used, latency, and later ``called_not_selected``;
* **C2 shadow rerank**: optional (``ORCHESTRATOR_TOOL_SELECTOR_SHADOW=oneshot``), runs on a background thread, only logged,
  never on the request's critical path and never applied.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.flag_env import flag_env

ENV_K_MIN = "ORCHESTRATOR_TOOL_SELECTOR_K_MIN"
ENV_K_MAX = "ORCHESTRATOR_TOOL_SELECTOR_K_MAX"
ENV_DELTA = "ORCHESTRATOR_TOOL_SELECTOR_DELTA"
ENV_FAMILY_CAP = "ORCHESTRATOR_TOOL_SELECTOR_FAMILY_CAP"
ENV_FULL_LIST_MAX = "ORCHESTRATOR_TOOL_SELECTOR_FULL_LIST_MAX"
ENV_BUDGET_MS = "ORCHESTRATOR_TOOL_SELECTOR_BUDGET_MS"
ENV_CORE = "ORCHESTRATOR_TOOL_SELECTOR_CORE"
ENV_LOG = "ORCHESTRATOR_TOOL_SELECTOR_LOG"
ENV_SHADOW = "ORCHESTRATOR_TOOL_SELECTOR_SHADOW"

CORE_NAMES = ("read_file", "peek", "write_file", "file_write_safe", "list_dir")
# generic-capability names real clients use for the same three jobs (read / write / list)
CORE_VOCAB = frozenset({"read", "write", "edit", "ls", "list", "glob", "cat"})
CORE_MAX = 6
SERVER_TOOLS = ("list_tools", "search_tools", "describe_tool")  # executed by the server, always sent when the pick narrows

# Logged on every selection row so a log reader can tell which policy produced it; bump when the ladder/plateau/family logic changes.
PICKER_VERSION = "hardened-c1-v1"

DEFAULTS = {"k_min": 4, "k_max": 12, "delta": 0.05, "family_cap": 3, "full_list_max": 40, "budget_ms": 500.0}


def _num(env: Mapping[str, str], key: str, default: float, cast: Callable[[Any], Any] = float) -> Any:
    try:
        return cast(str(env.get(key, default)).strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class PickerConfig:
    k_min: int = 4
    k_max: int = 12
    delta: float = 0.05
    family_cap: int = 3
    full_list_max: int = 40
    budget_ms: float = 500.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PickerConfig:
        env = flag_env() if environ is None else environ
        k_min = max(1, _num(env, ENV_K_MIN, DEFAULTS["k_min"], int))
        return cls(k_min=k_min, k_max=max(k_min, _num(env, ENV_K_MAX, DEFAULTS["k_max"], int)),
                   delta=max(0.0, _num(env, ENV_DELTA, DEFAULTS["delta"])), family_cap=max(1, _num(env, ENV_FAMILY_CAP, DEFAULTS["family_cap"], int)),
                   full_list_max=max(0, _num(env, ENV_FULL_LIST_MAX, DEFAULTS["full_list_max"], int)),
                   budget_ms=max(1.0, _num(env, ENV_BUDGET_MS, DEFAULTS["budget_ms"])))


# --- core pin + family ------------------------------------------------------------------------------------
def core_names(names: Sequence[str], environ: Mapping[str, str] | None = None) -> list[str]:
    """The pinned names present in ``names`` (at most ``CORE_MAX``): the explicit core names first, then the vocabulary."""
    env = flag_env() if environ is None else environ
    override = [x.strip() for x in str(env.get(ENV_CORE, "")).split(",") if x.strip()]
    wanted = override or list(CORE_NAMES)
    have = set(names)
    out = [n for n in wanted if n in have]
    if not override:
        out += [n for n in names if n not in out and n.lower() in CORE_VOCAB]
    return out[:CORE_MAX]


def family(name: str) -> str:
    """Name family for the crowding cap: ``syn_sku_price`` -> ``syn_sku``; ``read_file`` -> ``read``."""
    parts = name.lower().split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else parts[0]


def adaptive_pick(scores: Mapping[str, float], order: Sequence[str], core: Sequence[str], cfg: PickerConfig) -> tuple[list[str], int, int]:
    """(picks, k_effective, plateau). Non-core tools ranked by score (ties by ``order``); ``plateau`` is the number within
    ``delta`` of the top score; ``k_effective = clamp(plateau, k_min, k_max)``; at most ``family_cap`` per family."""
    pos = {n: i for i, n in enumerate(order)}
    pinned = set(core)
    ranked = sorted((n for n in scores if n not in pinned), key=lambda n: (-scores[n], pos.get(n, 1 << 30)))
    if not ranked:
        return [], 0, 0
    top = scores[ranked[0]]
    plateau = sum(1 for n in ranked if scores[n] >= top - cfg.delta)
    k_eff = max(cfg.k_min, min(cfg.k_max, plateau))
    return _take(ranked, k_eff, cfg.family_cap), k_eff, plateau


def _take(ranked: Sequence[str], k: int, cap: int) -> list[str]:
    picks: list[str] = []
    per: dict[str, int] = {}
    for n in ranked:
        f = family(n)
        if per.get(f, 0) >= cap:
            continue
        per[f] = per.get(f, 0) + 1
        picks.append(n)
        if len(picks) >= k:
            break
    return picks


# --- circuit breaker ----------------------------------------------------------------------------------------
class Breaker:
    """Opens after ``threshold`` consecutive failures for ``cooldown_s``; ``clock`` is injectable for tests."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self._fails = 0
        self._open_until = 0.0
        self.opened = 0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            return self.clock() < self._open_until

    def success(self) -> None:
        with self._lock:
            self._fails = 0

    def failure(self) -> None:
        with self._lock:
            self._fails += 1
            if self._fails >= self.threshold:
                self._open_until = self.clock() + self.cooldown_s
                self._fails = 0
                self.opened += 1


BREAKER = Breaker()
_EXEC = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-picker")
_BG = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tool-picker-shadow")


# --- memoised selector --------------------------------------------------------------------------------------
_SELECTORS: OrderedDict[tuple[str, str], Any] = OrderedDict()
_SEL_LOCK = threading.Lock()
_SELECTOR_MAX = 8
SELECTOR_BUILDS = {"n": 0}  # observability / tests: how many selectors were actually constructed


def registry_fingerprint(tools: Sequence[Mapping[str, Any]]) -> str:
    """Stable id of a tool-registry version: names, descriptions and parameter names."""
    from src.prompt_builders.tool_selector import tool_doc

    h = hashlib.sha1()
    for t in tools:
        h.update(tool_doc(t).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def memoised_embedding_selector(embedder: Any, tools: Sequence[Mapping[str, Any]], *, timeout_s: float = 120.0) -> Any:
    """One ``EmbeddingToolSelector`` per (embedder model, registry version)."""
    from src.prompt_builders.tool_selector import EmbeddingToolSelector

    key = (str(getattr(embedder, "model_id", "")), registry_fingerprint(tools))
    with _SEL_LOCK:
        sel = _SELECTORS.get(key)
        if sel is not None:
            _SELECTORS.move_to_end(key)
            return sel
    sel = EmbeddingToolSelector(embedder, tools, timeout_s=timeout_s)  # outside the lock: may embed N documents
    with _SEL_LOCK:
        SELECTOR_BUILDS["n"] += 1
        _SELECTORS[key] = sel
        while len(_SELECTORS) > _SELECTOR_MAX:
            _SELECTORS.popitem(last=False)
    return sel


def clear_selectors() -> None:
    with _SEL_LOCK:
        _SELECTORS.clear()


# --- frozen pick per conversation ---------------------------------------------------------------------------------
@dataclass
class Frozen:
    mode: str  # "narrow" | "full" | "core_search"
    names: list[str] = field(default_factory=list)  # ordered; additions are APPENDED
    record: dict[str, Any] = field(default_factory=dict)


_FROZEN: OrderedDict[str, Frozen] = OrderedDict()
_FROZEN_MAX = 256
_FROZEN_LOCK = threading.Lock()


def conversation_key(messages: Sequence[Mapping[str, Any]], tool_names: Sequence[str], scope: str = "") -> str:
    """Stable across the turns of one conversation: the FIRST user message, the tool-list identity and ``scope`` (model)."""
    from src.prompt_builders.tool_selector import last_user_query

    first = last_user_query(list(messages)[:next((i + 1 for i, m in enumerate(messages) if m.get("role") == "user"), 0)])
    raw = "\x1f".join([scope, first.strip(), "\x1e".join(tool_names)])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def frozen_get(key: str) -> Frozen | None:
    with _FROZEN_LOCK:
        f = _FROZEN.get(key)
        if f is not None:
            _FROZEN.move_to_end(key)
        return f


def frozen_put(key: str, f: Frozen) -> None:
    with _FROZEN_LOCK:
        _FROZEN[key] = f
        _FROZEN.move_to_end(key)
        while len(_FROZEN) > _FROZEN_MAX:
            _FROZEN.popitem(last=False)


def frozen_add(key: str, names: Sequence[str]) -> list[str]:
    """Append ``names`` not yet in the frozen pick (never reorders). Returns the names actually added."""
    with _FROZEN_LOCK:
        f = _FROZEN.get(key)
        if f is None or f.mode == "full":
            return []
        added = [n for n in names if n not in f.names]
        f.names.extend(added)
        return added


def clear_frozen() -> None:
    with _FROZEN_LOCK:
        _FROZEN.clear()


# --- the pick ---------------------------------------------------------------------------------------------------------
@dataclass
class PickResult:
    mode: str  # narrow | full | core_search
    picks: list[str]
    core: list[str]
    k_eff: int = 0
    plateau: int = 0
    rung: str = "none"  # none | widen | full | core_search | lexical | timeout | breaker_open | error | provider
    scores: dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    embedder: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def decide(scores: Mapping[str, float], order: Sequence[str], core: Sequence[str], cfg: PickerConfig, *, lexical: bool = False,
           n_tools: int | None = None) -> PickResult:
    """The whole policy from scores. Rung ``none``: a sharp top (plateau <= k_min). Rung ``widen``: k widened up to k_max for a
    broader plateau. Low confidence (a lexical fallback, or a plateau beyond k_max that even k_max cannot cover) climbs the
    rest of the ladder: the FULL list when it has at most ``full_list_max`` tools, else core + the server tools."""
    n = len(order) if n_tools is None else n_tools
    picks, k_eff, plateau = adaptive_pick(scores, order, core, cfg)
    if not lexical and plateau <= cfg.k_max:
        return PickResult("narrow", picks, list(core), k_eff, plateau, "widen" if plateau > cfg.k_min else "none", dict(scores))
    why = "lexical" if lexical else "flat"
    if n <= cfg.full_list_max:
        return PickResult("full", [], list(core), cfg.k_max, plateau, "full", dict(scores), detail={"low_confidence": why})
    return PickResult("core_search", [], list(core), cfg.k_max, plateau, "core_search", dict(scores), detail={"low_confidence": why})


Scorer = Callable[[str, "list[dict[str, Any]]"], "tuple[dict[str, float], str | None, str]"]


def pick(query: str, dicts: list[dict[str, Any]], scorer: Scorer | None, cfg: PickerConfig, environ: Mapping[str, str] | None = None,
         *, breaker: Breaker | None = None) -> PickResult:
    """Run the scorer under the latency budget and the breaker. ANY failure returns ``mode='full'`` (fail open to the unpicked list)."""
    brk = BREAKER if breaker is None else breaker
    names = [d["name"] for d in dicts]
    core = core_names(names, environ)
    t0 = time.perf_counter()

    def fail_open(rung: str, **detail: Any) -> PickResult:
        return PickResult("full", [], core, rung=rung, elapsed_ms=(time.perf_counter() - t0) * 1e3, detail=detail)

    if brk.is_open():
        return fail_open("breaker_open")
    from src.prompt_builders.tool_selector import lexical_scores

    def work() -> tuple[dict[str, float], str | None, str]:
        if scorer is None:
            return lexical_scores(query, dicts), "lexical:no_provider", ""
        return scorer(query, dicts)

    fut = _EXEC.submit(work)
    try:
        scores, fallback, emb = fut.result(timeout=cfg.budget_ms / 1e3)
    except FutureTimeout:
        brk.failure()
        return fail_open("timeout", budget_ms=cfg.budget_ms)
    except Exception as exc:  # noqa: BLE001 - a broken picker never breaks the request
        brk.failure()
        return fail_open("error", error=f"{type(exc).__name__}: {exc}"[:160])
    brk.success()
    res = decide(scores, names, core, cfg, lexical=bool(fallback and str(fallback).startswith("lexical")), n_tools=len(names))
    res.elapsed_ms, res.embedder = (time.perf_counter() - t0) * 1e3, emb
    if fallback:
        res.detail["scorer_fallback"] = fallback
        if res.rung == "none":
            res.rung = "lexical"
    return res


def embed_scorer(embedder: Any) -> Scorer:
    """Scorer over the memoised C1 selector (``SelectionResult.scores`` = cosine per tool)."""

    def scorer(query: str, dicts: list[dict[str, Any]]) -> tuple[dict[str, float], str | None, str]:
        sel = memoised_embedding_selector(embedder, dicts)
        r = sel.select(query, k=len(dicts))
        return r.scores, r.fallback, str(getattr(embedder, "model_id", ""))

    return scorer


# --- event log + C2 shadow -----------------------------------------------------------------------------------------------
_LOG_LOCK = threading.Lock()


def log_path(environ: Mapping[str, str] | None = None) -> Path | None:
    env = flag_env() if environ is None else environ
    v = env.get(ENV_LOG)
    if v is not None:
        return Path(v) if v.strip() else None  # explicit empty string disables the log
    return Path(__file__).resolve().parents[2] / "logs" / "tool_selector_events.jsonl"


def log_event(row: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> None:
    """Append one JSON line; best effort (a log failure never reaches the request)."""
    p = log_path(environ)
    if p is None:
        return
    try:
        with _LOG_LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), **row}, default=str) + "\n")
    except OSError:
        pass


def selection_row(req_id: str, conv: str, variant: str, role: str, res: PickResult, sent: Sequence[str], n_in: int, cfg: PickerConfig,
                  query: str, *, frozen_hit: bool) -> dict[str, Any]:
    top = sorted(res.scores.items(), key=lambda kv: -kv[1])[:16]
    return {"type": "selection", "request_id": req_id, "conversation": conv, "variant": variant, "role": role, "mode": res.mode,
            "rung": res.rung, "n_tools_in": n_in, "n_tools_out": len(sent), "picker_version": PICKER_VERSION,
            "k_min": cfg.k_min, "k_max": cfg.k_max, "delta": cfg.delta, "family_cap": cfg.family_cap, "full_list_max": cfg.full_list_max,
            "k_effective": res.k_eff,
            "plateau": res.plateau, "core": res.core, "selected": list(res.picks), "sent": list(sent), "scores_top16": [[n, round(s, 4)] for n, s in top],
            "elapsed_ms": round(res.elapsed_ms, 2), "embedder": res.embedder, "frozen_hit": frozen_hit,
            "query_hash": hashlib.sha1(query.encode()).hexdigest()[:12], "query_head": query[:120], "detail": res.detail}


def outcome_row(req_id: str, conv: str, sent: Sequence[str], history: Sequence[str], called: Sequence[str]) -> dict[str, Any]:
    cns = sorted(set(called) - set(sent) - set(history))
    return {"type": "outcome", "request_id": req_id, "conversation": conv, "called": sorted(set(called)), "called_not_selected": cns,
            "proxy_recall": 1.0 - len(cns) / max(1, len(set(called)))}


_SHADOW_PRIMITIVES: Any = None


def set_shadow_primitives(primitives: Any) -> None:
    """Register the decision model for the optional C2 shadow rerank (``None`` unregisters)."""
    global _SHADOW_PRIMITIVES
    _SHADOW_PRIMITIVES = primitives


def maybe_shadow_rerank(req_id: str, query: str, dicts: list[dict[str, Any]], applied: Sequence[str], k: int,
                        environ: Mapping[str, str] | None = None) -> bool:
    """Off unless ``ORCHESTRATOR_TOOL_SELECTOR_SHADOW=oneshot`` AND primitives are registered. Submits to a background thread and
    returns at once; the C2 choice is only LOGGED next to the applied pick (never applied, never awaited)."""
    env = flag_env() if environ is None else environ
    prim = _SHADOW_PRIMITIVES
    if str(env.get(ENV_SHADOW, "")).strip().lower() != "oneshot" or prim is None:
        return False

    def run() -> None:
        try:
            from src.prompt_builders.tool_selector import OneShotToolSelector

            groups = {d["name"]: [d["name"]] for d in dicts}
            desc = {d["name"]: str(d.get("description", ""))[:120] for d in dicts}
            r = OneShotToolSelector(prim, dicts, groups, desc).select(query, k)
            log_event({"type": "shadow", "request_id": req_id, "variant": "oneshot", "applied": list(applied), "shadow_selected": r.selected,
                       "elapsed_ms": round(r.elapsed_ms, 1), "selector_in_tokens": r.selector_in_tokens,
                       "selector_out_tokens": r.selector_out_tokens, "fallback": r.fallback}, environ)
        except Exception as exc:  # noqa: BLE001
            log_event({"type": "shadow", "request_id": req_id, "error": f"{type(exc).__name__}: {exc}"[:160]}, environ)

    _BG.submit(run)
    return True


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


# --- server-executed discovery tools (direct-call path) ------------------------------------------------------------------------
def server_tool_defs() -> list[dict[str, Any]]:
    """OpenAI tool definitions the SERVER executes (the client never sees these calls). Sent with every narrowed list so the picked
    set ADDS to discoverability: the model can always list the tools, search them and pull one into its frozen set."""
    def fn(name: str, desc: str, props: dict[str, Any], req: list[str]) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": desc,
                                                  "parameters": {"type": "object", "properties": props, "required": req}}}

    return [
        fn("list_tools", "List ALL tools available to you (names and one-line descriptions), including ones missing from your function "
           "list. Use it when the tool you need is not listed.", {}, []),
        fn("search_tools", "Find tools by what they do. Matching tools become callable on your next step.",
           {"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
        fn("describe_tool", "Get the full schema of one tool by name; the tool becomes callable on your next step.",
           {"name": {"type": "string"}}, ["name"]),
    ]


def _short(d: Mapping[str, Any], n: int = 100) -> dict[str, str]:
    return {"name": d["name"], "description": " ".join(str(d.get("description", "")).split())[:n]}


def execute_server_tool(name: str, args: Mapping[str, Any], ctx: Any) -> str:
    """Run one server tool for the active pick context; tools it surfaces are ADDED to the frozen pick (appended)."""
    from src.prompt_builders.tool_selector import lexical_scores, openai_tool_to_dict

    dicts = [d for d in (openai_tool_to_dict(t) for t in ctx.original) if d is not None]
    by = {d["name"]: d for d in dicts}
    try:
        if name == "list_tools":
            return json.dumps([_short(d) for d in dicts])[:12000]
        if name == "search_tools":
            lim = max(1, min(int(args.get("limit") or 5), 8))
            sc = lexical_scores(str(args.get("query", "")), dicts)
            hits = [n for n in sorted(sc, key=lambda n: -sc[n]) if sc[n] > 0][:lim]
            added = frozen_add(ctx.key, hits)
            return json.dumps({"matches": [_short(by[n]) for n in hits], "now_callable": hits, "added": added})
        if name == "describe_tool":
            n = str(args.get("name", ""))
            if n not in by:
                close = sorted(by, key=lambda x: -len(set(x.split("_")) & set(n.split("_"))))[:3]
                return json.dumps({"error": f"unknown tool {n!r}", "did_you_mean": close})
            frozen_add(ctx.key, [n])
            return json.dumps({"name": n, "description": by[n].get("description", ""), "parameters": by[n].get("parameters", {}), "now_callable": True})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"[:200]})
    return json.dumps({"error": f"unknown server tool {name!r}"})


def rebuild_tools(ctx: Any) -> list[dict[str, Any]]:
    """The narrowed list again after the frozen pick grew (server tools first, frozen names in order)."""
    from src.prompt_builders.tool_selector import _native_name

    f = frozen_get(ctx.key)
    by = {_native_name(t): t for t in ctx.original if _native_name(t)}
    return server_tool_defs() + [by[n] for n in (f.names if f else []) if n in by] + [t for t in ctx.original if _native_name(t) is None]


def record_outcome(ctx: Any, called: Sequence[str]) -> None:
    """Log which tools the turn called vs the sent set (``called_not_selected``) and add real client tools called but not sent
    to the frozen pick so the next turn includes them."""
    from src.prompt_builders.tool_selector import _native_name

    names = {_native_name(t) for t in ctx.original}
    real = [c for c in called if c not in SERVER_TOOLS]
    log_event(outcome_row(ctx.request_id, ctx.conversation, ctx.sent, ctx.history, real), ctx.environ)
    frozen_add(ctx.key, [c for c in real if c in names])
