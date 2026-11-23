"""Life-Harness distillation: four-layer harness taxonomy + zero-model trace-replay pre-check.

Operator decision 2026-11-23: the Life-Harness harness-optimization ideas are distilled INTO the autopilot
instead of running the full ALFWorld Life-Harness run. Two ideas are taken, both on EXISTING machinery:

1. **Four-layer taxonomy** (HIL-2 checklist item 24). Every harness proposal is annotated with ONE layer:

   * ``action_realization``   - how the model's intended action becomes an executable one (REPL syntax,
                                tool-call parse/markup, FINAL wrapping, write dedupe);
   * ``environment_contract`` - what the environment tells the model about its tools and results (tool
                                contract text, tool picker/search, output bounds);
   * ``trajectory_regulation`` - loop control across turns (nudges, failure-ratio guard, turn memory,
                                compaction, degeneration abort, escalation);
   * ``procedural_skill``     - reusable task procedure (worked examples, decomposition prompts, skillbank).

   plus the pre-existing proposal kinds ``prompt`` / ``routing`` / ``flag`` for surfaces that map onto no layer.
   The taxonomy is an ANNOTATION scheme: it ranks no layer (item 24: layer importance is environment-specific)
   and never gates a proposal. ``SURFACES`` maps each layer onto surfaces that ALREADY exist (feature flags,
   prompt files, allowlisted code files); ``tests/unit/test_harness_layers.py`` asserts every mapped name
   resolves, so this module can never invent a mutation surface.

2. **Trace-replay pre-check** (item 24, "before any paid evaluation ... trace replay on recorded traces"). A
   flag proposal is replayed, with NO model call, against the failing trace(s) that motivated it. Each
   replayable guard has a deterministic firing predicate built from the REAL guard's own patterns
   (imported, not mirrored) and written to OVER-approximate it: it may say "fires" when the real guard would
   not, never the reverse. A rejecting ``would_not_fire`` needs BOTH that and a whole-trajectory trace.

   The trace bank is currently fed from the last 50 lines of the inference tap: LLM prompt/response text only
   (REPL results live in ``repl_tap.log``), so a trace without a truncation marker is still NOT a complete
   trajectory. Bank items therefore carry ``complete`` / ``source`` provenance (default
   ``complete=False``, ``source="inference_tap_tail"``) and a rejection is returned ONLY when every
   motivating trace has ``complete=True`` and no truncation marker; otherwise the status is ``inconclusive``
   and the proposal proceeds to its eval. Today no capture sets ``complete=True``, so the pre-check is
   observe-only (it never rejects) until a whole-trajectory capture exists.

   Prompt and code edits, flags turned off, and guards that cannot be decided from text
   (``stream_degeneration_guard`` aborts on live draft acceptance; ``tool_call_parse_error_retry`` acts on
   server-side parse failures that never reach the transcript) are ``not_replayable``, never guessed.

Pure functions only: no I/O, no model call, no global state. Observe-only by construction except the one
documented effect of the pre-check (a ``would_not_fire`` flag proposal is skipped before its eval, which
needs whole-trajectory traces that no capture produces yet).
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Iterable, Mapping

LAYER_ACTION_REALIZATION = "action_realization"
LAYER_ENVIRONMENT_CONTRACT = "environment_contract"
LAYER_TRAJECTORY_REGULATION = "trajectory_regulation"
LAYER_PROCEDURAL_SKILL = "procedural_skill"

#: Priority order: also the tie-break when two layers implicate equally many failing traces.
LAYERS: tuple[str, ...] = (
    LAYER_ACTION_REALIZATION,
    LAYER_ENVIRONMENT_CONTRACT,
    LAYER_TRAJECTORY_REGULATION,
    LAYER_PROCEDURAL_SKILL,
)

#: Pre-existing proposal kinds; a proposal on a surface with no layer carries one of these.
LEGACY_KINDS: tuple[str, ...] = ("prompt", "routing", "flag")

#: Everything a proposal's ``layer`` may say.
LAYER_VOCABULARY: frozenset[str] = frozenset(LAYERS) | frozenset(LEGACY_KINDS)

LAYER_RECORD_SCHEMA = "harness_layer.v1"
REPLAY_RECORD_SCHEMA = "trace_replay_precheck.v1"

LAYER_TAXONOMY_ENV = "AUTOPILOT_LH_LAYER_TAXONOMY"
TRACE_REPLAY_PRECHECK_ENV = "AUTOPILOT_LH_TRACE_REPLAY_PRECHECK"

#: Failing traces consulted per proposal: the same k the critic trace IR shows the proposer.
MOTIVATING_K = 2

# ---------------------------------------------------------------------------------------------------------
# Layer -> EXISTING mutation surfaces. Every entry resolves to something that already ships:
#   flags        -> a FeatureSpec in src/features.py (and, where hot-swappable, HOT_SWAP_FEATURES)
#   prompt_files -> a path under orchestration/prompts/ (PromptForge targets)
#   code_files   -> a path in CODE_MUTATION_ALLOWLIST (src.autopilot_core.mutation_guards)
#   actions      -> an existing autopilot action type
# ---------------------------------------------------------------------------------------------------------
SURFACES: dict[str, dict[str, tuple[str, ...]]] = {
    LAYER_ACTION_REALIZATION: {
        "flags": (
            "repl_syntax_loop_breaker",
            "repl_explicit_final",
            "repl_explicit_final_strict",
            "repl_write_dedupe",
            "tool_call_parse_error_retry",
            "tool_call_arg_leak_guard",
            "structured_tool_output",
            "codemode_stubs",
        ),
        "prompt_files": ("root_lm_system.md",),
        "code_files": (),
        "actions": (),
    },
    LAYER_ENVIRONMENT_CONTRACT: {
        "flags": (
            "codemode_prompt_v2",
            "tool_selector",
            "tool_selector_shadow",
            "repl_tool_search",
            "repl_tool_arg_validation",
            "repl_handles",
            "repl_classify",
            "structured_read_only",
            "tool_output_bound",
            "tool_surface_cap",
        ),
        "prompt_files": ("tools.md",),
        "code_files": ("src/tool_policy.py",),
        "actions": (),
    },
    LAYER_TRAJECTORY_REGULATION: {
        "flags": (
            "repl_plan_narration_nudge",
            "harness_failure_ratio_guard",
            "repl_turn_memory",
            "history_append_only",
            "repl_answer_force",
            "stream_degeneration_guard",
            "session_compaction",
            "generation_monitor",
        ),
        "prompt_files": ("compaction_index.md",),
        "code_files": ("src/escalation.py", "src/graph/escalation_helpers.py"),
        "actions": (),
    },
    LAYER_PROCEDURAL_SKILL: {
        "flags": ("skillbank",),
        "prompt_files": (
            "rules.md",
            "task_decomposition.md",
            "planning_agent.md",
            "deep_search_agent.md",
        ),
        "code_files": (),
        "actions": ("distill_skillbank",),
    },
}

#: Surfaces that map onto no layer keep their pre-existing kind.
_UNLAYERED_CODE_FILES: dict[str, str] = {
    "src/prompt_builders/resolver.py": "prompt",
    "src/api/routes/chat.py": "routing",
}
_UNLAYERED_PROMPT_FILES: dict[str, str] = {"frontdoor.md": "routing"}


def _invert(field: str) -> dict[str, str]:
    return {name: layer for layer, surf in SURFACES.items() for name in surf[field]}


_FLAG_LAYER = _invert("flags")
_PROMPT_LAYER = _invert("prompt_files")
_CODE_LAYER = _invert("code_files")
_ACTION_LAYER = _invert("actions")


def flag_on(env_name: str) -> bool:
    """A default-off ``AUTOPILOT_LH_*`` flag, honouring a live ``POST /config`` override.

    Reads through ``src.flag_env`` (runtime-flag file > env > off) so the HOT_SWAP flags take effect in the
    autopilot process without a restart; falls back to the plain environment if that overlay is unavailable.
    """
    try:
        from src.flag_env import flag_env

        raw = flag_env().get(env_name)
    except Exception:  # noqa: BLE001 - an unreadable overlay reads as the plain env
        raw = os.environ.get(env_name)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _norm_prompt_path(path: str) -> str:
    p = str(path or "").strip().replace("\\", "/")
    marker = "orchestration/prompts/"
    if marker in p:
        p = p.split(marker, 1)[1]
    return p[2:] if p.startswith("./") else p


def surface_layer(action: Mapping[str, Any]) -> tuple[str, str, str]:
    """``(layer_or_kind, surface_kind, surface_name)`` for the surface an action edits.

    ``layer_or_kind`` is a layer when the surface maps onto one, else its pre-existing kind
    (``prompt`` / ``routing`` / ``flag``). Pure lookup over :data:`SURFACES`; never guesses.
    """
    atype = str(action.get("type") or "")
    if atype == "structural_experiment":
        flags = action.get("flags") if isinstance(action.get("flags"), Mapping) else {}
        name = next(iter(flags), "") if len(flags) == 1 else ""
        return _FLAG_LAYER.get(name, "flag"), "flag", str(name)
    if atype in ("prompt_mutation", "gepa_optimize", "structural_prune"):
        name = _norm_prompt_path(str(action.get("file") or ""))
        layer = _PROMPT_LAYER.get(name) or _UNLAYERED_PROMPT_FILES.get(name) or "prompt"
        return layer, "prompt_file", name
    if atype == "code_mutation":
        name = str(action.get("file") or "").strip()
        layer = _CODE_LAYER.get(name) or _UNLAYERED_CODE_FILES.get(name) or "prompt"
        return layer, "code_file", name
    if atype in _ACTION_LAYER:
        return _ACTION_LAYER[atype], "action", atype
    return "flag", "other", atype


# ---------------------------------------------------------------------------------------------------------
# Failure-trace diagnosis (deterministic signature scan)
# ---------------------------------------------------------------------------------------------------------
_I = re.IGNORECASE
_SIGNATURES: dict[str, tuple[tuple[str, "re.Pattern[str]"], ...]] = {
    LAYER_ACTION_REALIZATION: (
        ("syntax_error", re.compile(r"SyntaxError|IndentationError|unterminated|EOL while scanning")),
        ("tool_call_parse", re.compile(r"tool[_ ]call.{0,40}pars|failed to parse|parse[_ ]error|JSONDecodeError", _I)),
        ("tool_call_markup", re.compile(r"</?tool_call>|<function=|<parameter=|<arg_key>")),
    ),
    LAYER_ENVIRONMENT_CONTRACT: (
        ("unknown_tool", re.compile(r"unknown tool|no such tool|tool not found|unknown action|invalid action", _I)),
        ("bad_arguments", re.compile(r"unexpected keyword argument|missing \d* ?required (?:positional )?argument|invalid argument", _I)),
        ("undefined_name", re.compile(r"NameError|is not defined|has no attribute")),
        ("not_permitted", re.compile(r"not allowed|permission denied|not in allowlist", _I)),
    ),
    LAYER_TRAJECTORY_REGULATION: (
        ("turn_or_context_limit", re.compile(r"max(?:imum)? turns|turn limit|context (?:length|window) exceeded|exceeded the", _I)),
        ("no_progress", re.compile(r"no progress|nothing happens|same (?:action|command|call)|repeat(?:ed|ing)", _I)),
        ("plan_narration", re.compile(r"(?im)^\s*(?:I will|I'll|Let me|Next,? I|First,|Plan:)")),
        ("timeout", re.compile(r"timed?[ -]?out", _I)),
    ),
}


def _repeated_line(trace: str, *, min_len: int = 8, times: int = 3) -> bool:
    counts: dict[str, int] = {}
    for raw in trace.splitlines():
        line = raw.strip()
        if len(line) >= min_len:
            counts[line] = counts.get(line, 0) + 1
            if counts[line] >= times:
                return True
    return False


def classify_trace(trace: str) -> dict[str, Any]:
    """Which layer one failing trace implicates, with the signatures that say so.

    Primary layer = most signature hits (ties by :data:`LAYERS` order). A trace with no mechanical signature
    is the RESIDUAL case, labelled ``procedural_skill`` with ``basis="residual"``: the model ran without a
    harness defect we can name, so the failure is attributed to its procedure, and the record says so.
    """
    text = str(trace or "")
    hits: dict[str, list[str]] = {}
    for layer, sigs in _SIGNATURES.items():
        names = [name for name, rx in sigs if rx.search(text)]
        if layer == LAYER_TRAJECTORY_REGULATION and _repeated_line(text):
            names.append("repeated_line")
        if names:
            hits[layer] = names
    if not hits:
        return {"layer": LAYER_PROCEDURAL_SKILL, "basis": "residual", "signatures": []}
    layer = max(
        (l for l in LAYERS if l in hits), key=lambda l: (len(hits[l]), -LAYERS.index(l))
    )
    return {"layer": layer, "basis": "signature", "signatures": sorted(hits[layer])}


def motivating_failure_traces(state: Mapping[str, Any], k: int = MOTIVATING_K) -> list[dict[str, Any]]:
    """The failing trace(s) that motivate the next proposal: the newest ``k`` failures of the contrastive bank.

    These are exactly the failure examples ``build_critic_trace_ir`` shows the proposer. Items carry
    ``trace`` (possibly carrying a truncation marker), ``trial_id``, ``trace_hash`` and the provenance
    ``complete`` (True only when the bank item asserted a whole-trajectory capture; absent reads False) and
    ``source``.
    """
    bank = state.get("contrastive_trace_bank") if isinstance(state, Mapping) else None
    out: list[dict[str, Any]] = []
    for item in bank or []:
        if (
            isinstance(item, Mapping)
            and str(item.get("outcome") or "").lower() == "failure"
            and str(item.get("trace") or "").strip()
        ):
            text = str(item["trace"])
            out.append(
                {
                    "trial_id": item.get("trial_id"),
                    "trace_hash": str(item.get("trace_hash") or hashlib.sha256(text.encode()).hexdigest()[:12]),
                    "trace": text,
                    "complete": item.get("complete") is True,
                    "source": str(item.get("source") or "unknown"),
                }
            )
    return out[-k:] if k > 0 else []


def diagnose_failure_traces(traces: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate :func:`classify_trace` over the motivating failures into one implicated layer (or None)."""
    items = [t for t in traces if str(t.get("trace") or "").strip()]
    counts = {layer: 0 for layer in LAYERS}
    per_trace: list[dict[str, Any]] = []
    residual = 0
    for t in items:
        c = classify_trace(str(t["trace"]))
        counts[c["layer"]] += 1
        residual += 1 if c["basis"] == "residual" else 0
        per_trace.append(
            {"trial_id": t.get("trial_id"), "trace_hash": t.get("trace_hash"), **c}
        )
    implicated = None
    if items:
        implicated = max(LAYERS, key=lambda l: (counts[l], -LAYERS.index(l)))
    return {
        "schema": LAYER_RECORD_SCHEMA,
        "n_failures": len(items),
        "counts": counts,
        "residual": residual,
        "implicated": implicated,
        "per_trace": per_trace,
    }


def render_layer_block(diagnosis: Mapping[str, Any] | None) -> str:
    """Critic/proposer prompt block: the implicated layer and the existing surfaces each layer may edit."""
    if not isinstance(diagnosis, Mapping) or not diagnosis.get("implicated"):
        return ""
    counts = diagnosis.get("counts") or {}
    implicated = str(diagnosis["implicated"])
    lines = [
        "## Harness Layer Diagnosis (Life-Harness taxonomy, annotation only)",
        f"The {diagnosis.get('n_failures', 0)} failing trace(s) below implicate layer **{implicated}** "
        f"(per-layer trace counts: "
        + ", ".join(f"{layer}={counts.get(layer, 0)}" for layer in LAYERS)
        + f"; {diagnosis.get('residual', 0)} had no mechanical signature and default to procedural_skill).",
        "Target ONE layer per proposal, preferably the implicated one; a proposal on another layer states why. "
        "This is an annotation scheme, not a ranking, and no layer is presumed more valuable. "
        "Edit only these EXISTING surfaces:",
    ]
    for layer in LAYERS:
        surf = SURFACES[layer]
        parts = []
        if surf["flags"]:
            parts.append("flags " + ", ".join(surf["flags"]))
        if surf["prompt_files"]:
            parts.append("prompt files " + ", ".join(surf["prompt_files"]))
        if surf["code_files"]:
            parts.append("code files " + ", ".join(surf["code_files"]))
        if surf["actions"]:
            parts.append("actions " + ", ".join(surf["actions"]))
        marker = " (implicated)" if layer == implicated else ""
        lines.append(f"- {layer}{marker}: " + "; ".join(parts))
    lines.append(
        "Optional: put the layer in the action as \"layer\" "
        "(" + "|".join(sorted(LAYER_VOCABULARY)) + ")."
    )
    return "\n".join(lines)


def build_layer_record(
    action: Mapping[str, Any], diagnosis: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The journal annotation for one proposal: its layer, how it was determined, and alignment."""
    surface_value, surface_kind, surface_name = surface_layer(action)
    declared_raw = action.get("layer")
    declared = str(declared_raw).strip() if isinstance(declared_raw, str) else ""
    if declared and declared in LAYER_VOCABULARY:
        layer, source = declared, "declared"
    else:
        layer, source = surface_value, "surface"
    implicated = (diagnosis or {}).get("implicated") if isinstance(diagnosis, Mapping) else None
    aligned: bool | None = None
    if implicated and layer in LAYERS:
        aligned = layer == implicated
    return {
        "schema": LAYER_RECORD_SCHEMA,
        "layer": layer,
        "source": source,
        "declared": declared or None,
        "declared_invalid": bool(declared) and declared not in LAYER_VOCABULARY,
        "surface_layer": surface_value,
        "surface_kind": surface_kind,
        "surface": surface_name,
        "implicated": implicated,
        "aligned": aligned,
        "n_failures": int((diagnosis or {}).get("n_failures", 0)) if isinstance(diagnosis, Mapping) else 0,
    }


# ---------------------------------------------------------------------------------------------------------
# Trace replay (no model call)
# ---------------------------------------------------------------------------------------------------------
#: Prefix ``eval_tower_trace_feedback.trim_trace_text`` puts on a trace it cut (kept as a literal: pure module).
TRUNCATION_MARKER = "[trace truncated]"
#: The inference tap's own cut marker, e.g. ``... [1000 chars truncated]`` (src/runtime/inference_tap.py).
_TAP_TRUNCATION = re.compile(r"\[\d+ chars truncated\]")

#: Real-guard error shapes that make a turn a failing turn (src/graph/helpers.py ``_note_turn_health``: a REPL
#: error of any text, or tool output starting ``[ERROR`` / ``[TOOL ERROR``). Case-insensitive; over-approximate.
_FAIL_LINE = re.compile(
    r"traceback|error|exception|\bfailed\b|\bfailure\b|timed?[ -]?out|\[ERROR|\[TOOL ERROR", _I
)


def is_truncated(trace: str) -> bool:
    """True when the trace carries either truncation marker (the bank's or the inference tap's own)."""
    return TRUNCATION_MARKER in trace or bool(_TAP_TRUNCATION.search(trace))


def _real_leak_markers() -> tuple[str, ...] | None:
    """The real arg-leak guard's own markers (``src.backends.tool_arg_guard.LEAK_MARKERS``), or None."""
    try:
        from src.backends.tool_arg_guard import LEAK_MARKERS

        return tuple(LEAK_MARKERS)
    except Exception:  # noqa: BLE001 - unreadable real guard: cannot over-approximate it, so say "fires"
        return None


def _real_plan_line_re() -> "re.Pattern[str] | None":
    """The real plan-narration guard's own line pattern (``src.graph.answer_resolution._PLAN_LINE_RE``), or None."""
    try:
        from src.graph.answer_resolution import _PLAN_LINE_RE

        return _PLAN_LINE_RE
    except Exception:  # noqa: BLE001
        return None


def _fires_syntax_loop(trace: str) -> bool:
    # real guard: 2 CONSECUTIVE SyntaxError REPL turns. Two occurrences anywhere is a superset.
    return len(re.findall(r"SyntaxError", trace, _I)) >= 2


def _fires_plan_narration(trace: str) -> bool:
    # real guard (answer_resolution._looks_like_plan_narration): a line matching _PLAN_LINE_RE, UNLESS the text
    # also states an answer or contains a code fence. Ignoring those exclusions makes this a superset.
    pattern = _real_plan_line_re()
    if pattern is None:
        return True
    return any(pattern.match(line) for line in trace.splitlines() if line.strip())


def _fires_failure_ratio(trace: str) -> bool:
    # real guard: failures/turns >= 0.5 over >= 4 turns. Without turn structure, >= 2 failure lines is a superset.
    return sum(1 for ln in trace.splitlines() if _FAIL_LINE.search(ln)) >= 2


def _fires_arg_leak(trace: str) -> bool:
    # real guard (tool_arg_guard.find_leak): any LEAK_MARKERS substring in a tool-call argument value.
    markers = _real_leak_markers()
    if markers is None:
        return True
    return any(marker in trace for marker in markers)


#: Enabling flag -> deterministic over-approximating firing predicate. A flag absent here is NOT replayable.
FLAG_FIRING_PREDICATES = {
    "repl_syntax_loop_breaker": _fires_syntax_loop,
    "repl_plan_narration_nudge": _fires_plan_narration,
    "harness_failure_ratio_guard": _fires_failure_ratio,
    "tool_call_arg_leak_guard": _fires_arg_leak,
}

#: Guards that cannot be decided from transcript text, with the reason recorded in the journal.
NOT_REPLAYABLE_FLAGS: dict[str, str] = {
    "stream_degeneration_guard": "aborts on live rolling draft acceptance, which is not in the transcript text",
    "tool_call_parse_error_retry": "acts on server-side tool-call parse failures that never reach the transcript text",
}


def _replay(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"schema": REPLAY_RECORD_SCHEMA, "status": status, "reason": reason, **extra}


def replay_precheck(
    action: Mapping[str, Any], traces: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Replay a proposal against the failing trace(s) that motivated it. No model call, no I/O.

    ``status`` is one of ``would_fire`` (the proposal's guard fires on >= 1 motivating trace: proceed),
    ``would_not_fire`` (the predicate fires on none of them AND every one is a whole-trajectory capture with no
    truncation marker: reject before the eval), ``inconclusive`` (the predicate fires on none, but at least one
    trace is not provably whole: proceed to the eval) or ``not_replayable`` (no deterministic replay model, or
    no motivating trace: proceed). Only ``would_not_fire`` rejects; every other status leaves the proposal to
    its normal eval. Traces whose ``complete`` is not True (the default tap-tail capture) never reject.
    """
    atype = str(action.get("type") or "")
    if atype != "structural_experiment":
        return _replay("not_replayable", f"{atype or 'unknown'} edits have no deterministic replay model")
    flags = action.get("flags") if isinstance(action.get("flags"), Mapping) else {}
    if len(flags) != 1:
        return _replay("not_replayable", "needs exactly one flag (AP-9)")
    name, value = next(iter(flags.items()))
    if value is not True:
        return _replay("not_replayable", f"turning {name} off has no replay model", flag=name)
    if name in NOT_REPLAYABLE_FLAGS:
        return _replay("not_replayable", f"{name} is not replayable from text: {NOT_REPLAYABLE_FLAGS[name]}", flag=name)
    predicate = FLAG_FIRING_PREDICATES.get(name)
    if predicate is None:
        return _replay("not_replayable", f"{name} has no firing predicate", flag=name)
    items = [t for t in traces if str(t.get("trace") or "").strip()]
    if not items:
        return _replay("not_replayable", "no motivating failure trace", flag=name, motivating=0)
    fired = [t for t in items if predicate(str(t["trace"]))]
    truncated = sum(1 for t in items if is_truncated(str(t["trace"])))
    incomplete = sum(1 for t in items if t.get("complete") is not True)
    base = {
        "flag": name,
        "motivating": len(items),
        "fired": len(fired),
        "truncated": truncated,
        "incomplete": incomplete,
        "sources": sorted({str(t.get("source") or "unknown") for t in items}),
        "trace_hashes": [t.get("trace_hash") for t in items],
    }
    if fired:
        return _replay("would_fire", f"{name} fires on {len(fired)}/{len(items)} motivating trace(s)", **base)
    if truncated or incomplete:
        return _replay(
            "inconclusive",
            f"{name} does not fire on the visible text but {max(truncated, incomplete)}/{len(items)} trace(s) "
            "are not provably whole trajectories (truncated or not a whole-trajectory capture)",
            **base,
        )
    return _replay(
        "would_not_fire",
        f"{name} would not have fired on any of the {len(items)} whole-trajectory motivating failure trace(s)",
        **base,
    )


def replay_rejects(record: Mapping[str, Any] | None) -> bool:
    """True only for ``would_not_fire`` (the single status that skips the eval)."""
    return isinstance(record, Mapping) and record.get("status") == "would_not_fire"
