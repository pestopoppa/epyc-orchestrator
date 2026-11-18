"""REPL additions from the codemode intake (CM-2, CM-4, CM-5). Every function is DEFAULT OFF behind its own flag,
so with no flag set nothing here is installed and the REPL globals are byte-identical to before.

* CM-2 ``ORCHESTRATOR_REPL_TOOL_SEARCH=1``  -> ``search_tools(query, detail="names", k=8)`` and ``describe_tool(name)``.
  ``list_tools()`` is untouched (control arm). ``detail``: ``names`` (name only), ``brief`` (name + description),
  ``full`` (the list_tools() dict, i.e. with schema). A model pays for a schema only when it asks for it.
* CM-4 ``ORCHESTRATOR_REPL_CLASSIFY=1``      -> ``classify(items, labels, instruction="")`` labels N items with one
  typed-decision (CHOICE) call per item through the LLM primitives, so the items never enter the root prompt; only the
  returned labels do. Items that fail to parse are labelled ``None`` (never a silent default).
  Under the same flag the EXISTING deterministic classifiers are exposed read-only, in batch, with typed returns and
  a cost cap (``route_classify``, ``risk_bands``, and ``q_scores`` when a MemRL retriever is present); see CM-4b below.
* Fix 4 ``ORCHESTRATOR_CODEMODE_STUBS=1``    -> every registry tool the REPL does not already define is bound as a direct Python
  function that returns a Python object (``objectify``), matching the typed stubs of ``prompt_builders.codemode_stubs``.
* CM-5 ``ORCHESTRATOR_REPL_HANDLES=1``       -> ``store(name, value) -> handle`` / ``load(handle_or_name)`` /
  ``handles()``: a named JSON store capped at 256 KiB per value and 1 MiB total, written only by cells that
  SUCCEED (a failing cell's writes are rolled back) and snapshotted into REPL checkpoints so a restored/branched
  session replays it. ``peek`` is not needed to re-read a large value; the model passes the handle.
  With the handles flag on, replay recording (HIL-16) defaults ON unless ``REPLAY_RECORDING`` is set explicitly
  (``replay_default_on``); the seed of a recorded run carries the handle store so a replay starts from it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

ENV_SEARCH = "ORCHESTRATOR_REPL_TOOL_SEARCH"
ENV_CLASSIFY = "ORCHESTRATOR_REPL_CLASSIFY"
ENV_HANDLES = "ORCHESTRATOR_REPL_HANDLES"
ENV_STUBS = "ORCHESTRATOR_CODEMODE_STUBS"
ENV_CLASSIFY_BUDGET = "ORCHESTRATOR_REPL_CLASSIFY_BUDGET"
MAX_VALUE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024
DETAILS = ("names", "brief", "full")


def flag(name: str, environ: Mapping[str, str] | None = None) -> bool:
    return str((os.environ if environ is None else environ).get(name, "")).strip().lower() in ("1", "true", "on", "yes")


# --- CM-2 ---------------------------------------------------------------------------------------
def search_tools(tools: Sequence[Mapping[str, Any]], query: str, detail: str = "names", k: int = 8) -> list[Any]:
    """Ranked tool search over ``list_tools()``-shaped dicts (lexical IDF overlap; deterministic)."""
    from src.prompt_builders.tool_selector import _topk, lexical_scores

    if detail not in DETAILS:
        raise ValueError(f"detail must be one of {DETAILS}")
    tl = [dict(t) for t in tools]
    sc = lexical_scores(query, tl)
    order = [t["name"] for t in tl]
    hits = [n for n in _topk(sc, order, max(1, int(k))) if sc[n] > 0]
    by = {t["name"]: t for t in tl}
    if detail == "names":
        return hits
    if detail == "brief":
        return [{"name": n, "description": by[n].get("description", "")} for n in hits]
    return [by[n] for n in hits]


def describe_tool(tools: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    for t in tools:
        if t.get("name") == name:
            return dict(t)
    raise KeyError(f"unknown tool: {name!r} (use search_tools(query) to find one)")


# --- CM-4 ---------------------------------------------------------------------------------------
def classify_items(primitives: Any, items: Sequence[Any], labels: Sequence[str], instruction: str = "", *,
                   role: str = "worker", max_items: int = 500) -> list[str | None]:
    """One typed-decision CHOICE call per item (Jev technique: value-only JSON, constrained label set)."""
    from src.typed_decisions import run_typed_decisions
    from src.typed_decisions.types import Question, QuestionKind

    labels = [str(x) for x in labels]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("classify needs >= 2 unique labels")
    if len(items) > max_items:
        raise ValueError(f"classify is capped at {max_items} items per call")
    q = Question(id="label", kind=QuestionKind.CHOICE, options=tuple(labels),
                 text=(instruction.strip() or "Pick the single best label for the item."))
    out: list[str | None] = []
    for it in items:
        res = run_typed_decisions(primitives, state=f"ITEM:\n{it}", questions=[q], role=role, mode="json",
                                  max_retries=0, value_only=True)
        v = next((d.value for d in res.decisions if d.question_id == "label"), None)
        out.append(v if v in labels else None)
    return out


# --- CM-4b: read-only, capped, typed access to the existing classifiers / q-scorer reads ---------------
# Cost cap: at most MAX_CLASSIFIER_ITEMS_PER_CALL items per call and ``ORCHESTRATOR_REPL_CLASSIFY_BUDGET`` items per REPL
# session (default DEFAULT_CLASSIFIER_BUDGET) across route_classify / risk_bands / q_scores; each item is cut to
# MAX_ITEM_CHARS before scoring. Read-only: none of these write the episodic store, the q-values or any registry.
MAX_CLASSIFIER_ITEMS_PER_CALL = 200
DEFAULT_CLASSIFIER_BUDGET = 2000
MAX_ITEM_CHARS = 4000


class ClassifierBudget:
    """Per-session item budget shared by the read-only classifier functions."""

    def __init__(self, limit: int = DEFAULT_CLASSIFIER_BUDGET, per_call: int = MAX_CLASSIFIER_ITEMS_PER_CALL):
        self.limit, self.per_call, self.used = int(limit), int(per_call), 0

    def take(self, n: int, fn: str) -> None:
        if n > self.per_call:
            raise ValueError(f"{fn} is capped at {self.per_call} items per call (got {n})")
        if self.used + n > self.limit:
            raise ValueError(f"{fn}: classifier budget exhausted ({self.used}/{self.limit} items used this session)")
        self.used += n

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ClassifierBudget":
        raw = (os.environ if environ is None else environ).get(ENV_CLASSIFY_BUDGET, "")
        try:
            limit = int(raw) if str(raw).strip() else DEFAULT_CLASSIFIER_BUDGET
        except ValueError:
            limit = DEFAULT_CLASSIFIER_BUDGET
        return cls(limit=max(0, limit))


def _texts(items: Sequence[Any], budget: ClassifierBudget, fn: str) -> list[str]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError(f"{fn} takes a list of items, not {type(items).__name__}")
    budget.take(len(items), fn)
    return [str(i)[:MAX_ITEM_CHARS] for i in items]


def route_classify(items: Sequence[Any], budget: ClassifierBudget | None = None) -> list[dict[str, Any]]:
    """Rule-based routing label per item. Typed row: ``{role, strategy, confidence, matched_keywords, trinity_role,
    trinity_reason}`` (``classify_and_route`` + ``classify_role``; deterministic, no model call)."""
    from src.classifiers.keyword_matcher import classify_and_route
    from src.classifiers.role_classifier import classify_role

    out: list[dict[str, Any]] = []
    for t in _texts(items, budget or ClassifierBudget(), "route_classify"):
        d = classify_and_route(t)
        r = classify_role(t)
        out.append({"role": str(d.role), "strategy": str(d.strategy), "confidence": float(d.confidence),
                    "matched_keywords": [str(k) for k in d.matched_keywords],
                    "trinity_role": str(r.role), "trinity_reason": str(r.reason)})
    return out


def risk_bands(items: Sequence[Any], role: str = "", budget: ClassifierBudget | None = None) -> list[dict[str, Any]]:
    """Factual-risk and difficulty bands per item. Typed row: ``{risk_band, risk_score, difficulty_band,
    difficulty_score}`` (``assess_risk`` + ``assess_difficulty``; regex-only, deterministic)."""
    from src.classifiers.difficulty_signal import assess_difficulty
    from src.classifiers.factual_risk import assess_risk

    out: list[dict[str, Any]] = []
    for t in _texts(items, budget or ClassifierBudget(), "risk_bands"):
        rk, df = assess_risk(t, role), assess_difficulty(t, role)
        out.append({"risk_band": str(rk.risk_band), "risk_score": float(rk.adjusted_risk_score),
                    "difficulty_band": str(df.difficulty_band), "difficulty_score": float(df.difficulty_score)})
    return out


def q_scores(retriever: Any, items: Sequence[Any], k: int = 3, budget: ClassifierBudget | None = None) -> list[list[dict[str, Any]]]:
    """Read the MemRL routing q-values (the ones QScorer writes) for each item: the top ``k`` roles by mean q-value of the
    retrieved similar memories, as typed rows ``{role, q_value, count}``. Retrieval only; nothing is scored or written."""
    from src.constants import TASK_IR_OBJECTIVE_LEN
    from src.task_ir import canonicalize_task_ir

    if retriever is None:
        raise RuntimeError("q_scores is unavailable: this REPL has no MemRL retriever")
    k = max(1, min(int(k), 10))
    out: list[list[dict[str, Any]]] = []
    for t in _texts(items, budget or ClassifierBudget(), "q_scores"):
        ir = canonicalize_task_ir({"task_type": "chat", "objective": t[:TASK_IR_OBJECTIVE_LEN], "priority": "interactive"})
        stats: dict[str, list[float]] = {}
        for r in retriever.retrieve_for_routing(ir):
            role = str((r.memory.context or {}).get("role", r.memory.action))
            stats.setdefault(role, []).append(float(r.q_value))
        rows = [{"role": ro, "q_value": round(sum(v) / len(v), 3), "count": len(v)} for ro, v in stats.items()]
        rows.sort(key=lambda x: (-x["q_value"], x["role"]))
        out.append(rows[:k])
    return out


# --- CM-5 ---------------------------------------------------------------------------------------
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


class HandleStore:
    """Size-capped JSON store with per-cell staging. ``begin_cell`` .. ``commit`` | ``rollback``."""

    def __init__(self, max_value: int = MAX_VALUE_BYTES, max_total: int = MAX_TOTAL_BYTES):
        self.max_value, self.max_total = max_value, max_total
        self._data: dict[str, str] = {}  # name -> JSON text (committed)
        self._staged: dict[str, str] = {}

    def begin_cell(self) -> None:
        self._staged = {}

    def commit(self) -> None:
        self._data.update(self._staged)
        self._staged = {}

    def rollback(self) -> None:
        self._staged = {}

    def _view(self) -> dict[str, str]:
        return {**self._data, **self._staged}

    def store(self, name: str, value: Any) -> str:
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError("handle name must match [A-Za-z_][A-Za-z0-9_.-]{0,63}")
        try:
            text = json.dumps(value, sort_keys=True)
        except (TypeError, ValueError) as e:
            raise ValueError(f"value is not JSON-serializable: {e}") from e
        size = len(text.encode())
        if size > self.max_value:
            raise ValueError(f"value is {size} bytes; the per-value cap is {self.max_value}")
        total = sum(len(v.encode()) for k, v in self._view().items() if k != name) + size
        if total > self.max_total:
            raise ValueError(f"store would hold {total} bytes; the total cap is {self.max_total}")
        self._staged[name] = text
        return f"handle:{name}"

    def load(self, handle: str) -> Any:
        name = handle[7:] if isinstance(handle, str) and handle.startswith("handle:") else handle
        v = self._view().get(name)
        if v is None:
            raise KeyError(f"no such handle: {handle!r}")
        return json.loads(v)

    def handles(self) -> dict[str, int]:
        return {k: len(v.encode()) for k, v in sorted(self._view().items())}

    def snapshot(self) -> dict[str, str]:
        return dict(self._data)

    def restore(self, snap: Mapping[str, str] | None) -> None:
        self._data, self._staged = dict(snap or {}), {}


def replay_default_on(environ: Mapping[str, str] | None = None) -> bool:
    """CM-5: replay recording (HIL-16) is ON by default while handles are on, so a handle-using run can be replayed.
    Any explicit ``REPLAY_RECORDING`` setting (env in either spelling, runtime flags file, override) wins in BOTH
    directions; with the handles flag off this is always False and nothing changes."""
    if not flag(ENV_HANDLES, environ):
        return False
    env = os.environ if environ is None else environ
    if any(k in env for k in ("ORCHESTRATOR_REPLAY_RECORDING", "ORCHESTRATOR_FEATURE_REPLAY_RECORDING")):
        return False
    try:
        from src.features import feature_sources

        return str(feature_sources().get("replay_recording", "default")).startswith("default")
    except Exception:
        return True


# --- fix 4: stub-bound tools returning Python objects -----------------------------------------------
class ToolCallError(RuntimeError):
    """A bound tool failed; the message is the tool's own reason (a CM-3 envelope with ok=False, or the exception)."""


def objectify(result: Any, returns: str | None = None) -> Any:
    """Python object for a tool result. A CM-3 envelope (``ORCHESTRATOR_STRUCTURED_READ_ONLY``) is unwrapped, a failed one
    raises ``ToolCallError``. JSON *text* is parsed only when the stub PROMISES an object (``returns`` is not ``str``/``Any``),
    so a ``-> str`` tool such as ``read_file`` keeps returning the raw text."""
    if hasattr(result, "to_machine") and hasattr(result, "ok"):
        if not result.ok:
            raise ToolCallError(str(getattr(result, "output", "") or "tool failed"))
        result = result.output
    if isinstance(result, str) and returns and returns not in ("str", "Any"):
        t = result.lstrip()
        if t[:1] in ("[", "{"):
            try:
                return json.loads(t)
            except ValueError:
                return result
    return result


def _param_order(tool: Mapping[str, Any]) -> list[str]:
    params = tool.get("parameters") or {}
    return [k for k, v in params.items() if v.get("required")] + [k for k, v in params.items() if not v.get("required")]


def make_stub_callable(tool: Mapping[str, Any], invoke: Callable[..., Any]) -> Callable[..., Any]:
    """``f(*args, **kw)`` for one tool: positional args follow the stub's signature order (required first)."""
    name, order, returns = str(tool["name"]), _param_order(tool), tool.get("returns")

    def f(*args: Any, **kw: Any) -> Any:
        if len(args) > len(order):
            raise TypeError(f"{name}() takes at most {len(order)} arguments ({', '.join(order)}), got {len(args)}")
        merged = dict(zip(order, args))
        dup = set(merged) & set(kw)
        if dup:
            raise TypeError(f"{name}() got multiple values for {sorted(dup)}")
        merged.update(kw)
        missing = [k for k, v in (tool.get("parameters") or {}).items() if v.get("required") and k not in merged]
        if missing:
            raise TypeError(f"{name}() missing required argument(s): {', '.join(missing)}")
        return objectify(invoke(name, **merged), returns)

    f.__name__ = name
    return f


def bind_stub_tools(globals_dict: dict[str, Any], invoke: Callable[..., Any], tools: Sequence[Mapping[str, Any]],
                    names: Sequence[str] | None = None, *, override: bool = False) -> list[str]:
    """Bind ``names`` (default: all) from ``tools`` into ``globals_dict``. Existing globals are kept unless ``override``;
    returns the names bound."""
    bound: list[str] = []
    for t in tools:
        n = str(t.get("name", ""))
        if not n.isidentifier() or (names is not None and n not in names):
            continue
        if n in globals_dict and not override:
            continue
        globals_dict[n] = make_stub_callable(t, invoke)
        bound.append(n)
    return bound


def install(env: Any, globals_dict: dict[str, Any], environ: Mapping[str, str] | None = None) -> list[str]:
    """Add the flagged functions to a REPL globals dict. Returns the names added (empty when everything is off)."""
    added: list[str] = []
    # the picker is on -> search_tools/describe_tool are ALWAYS available (recovery when the needed tool was not picked)
    from src.prompt_builders.tool_selector import selector_variant

    search_on = flag(ENV_SEARCH, environ) or selector_variant(environ) != "off"
    if search_on and getattr(env, "tool_registry", None) is not None:
        globals_dict["search_tools"] = lambda query, detail="names", k=8: search_tools(env._list_tools(), query, detail, k)
        globals_dict["describe_tool"] = lambda name: describe_tool(env._list_tools(), name)
        added += ["search_tools", "describe_tool"]
    if flag(ENV_CLASSIFY, environ) and getattr(env, "llm_primitives", None) is not None:
        globals_dict["classify"] = lambda items, labels, instruction="": classify_items(
            env.llm_primitives, items, labels, instruction)
        added.append("classify")
    if flag(ENV_CLASSIFY, environ):
        budget = getattr(env, "_classifier_budget", None)
        if budget is None:
            budget = env._classifier_budget = ClassifierBudget.from_env(environ)
        globals_dict["route_classify"] = lambda items: route_classify(items, budget)
        globals_dict["risk_bands"] = lambda items, role="": risk_bands(items, role, budget)
        added += ["route_classify", "risk_bands"]
        retr = getattr(env, "_retriever", None) or getattr(getattr(env, "_hybrid_router", None), "retriever", None)
        if retr is not None:
            globals_dict["q_scores"] = lambda items, k=3: q_scores(retr, items, k, budget)
            added.append("q_scores")
    if flag(ENV_STUBS, environ) and getattr(env, "tool_registry", None) is not None:
        added += bind_stub_tools(globals_dict, env._invoke_tool, env._list_tools())
    if flag(ENV_HANDLES, environ):
        hs = getattr(env, "_handle_store", None)
        if hs is None:
            hs = env._handle_store = HandleStore()
        globals_dict.update(store=hs.store, load=hs.load, handles=hs.handles)
        added += ["store", "load", "handles"]
    return added


READ_ONLY_NAMES: frozenset[str] = frozenset({"search_tools", "describe_tool", "handles", "load",
                                             "route_classify", "risk_bands", "q_scores"})


# --- CM-3 consistency check ----------------------------------------------------------------------
def result_shape(x: Any) -> Any:
    """Structural fingerprint of a tool result: envelope fields for ToolOutput, key set for dicts, element types for lists."""
    if hasattr(x, "to_machine") and hasattr(x, "ok"):
        return ("envelope", x.ok, x.status, result_shape(x.output))
    if isinstance(x, dict):
        return ("dict", tuple(sorted(x)))
    if isinstance(x, (list, tuple)):
        return ("list", tuple(sorted({type(i).__name__ for i in x})))
    return (type(x).__name__,)


def check_shape_consistency(call: Callable[..., Any], make_kwargs: Callable[[int], dict[str, Any]],
                            small: int = 5, large: int = 1000) -> dict[str, Any]:
    """Probe-5-then-max-batch (intake-2675 failure mode): the tool must return the SAME shape at a small batch and at
    its maximum. ``make_kwargs(n)`` builds the call arguments for a result of about n items."""
    a, b = result_shape(call(**make_kwargs(small))), result_shape(call(**make_kwargs(large)))
    return {"small": a, "large": b, "consistent": a == b,
            "bare_string": a[0] == "str" or b[0] == "str"}
