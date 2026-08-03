#!/usr/bin/env python3
"""Declared production posture for the runtime feature-flag overlay.

WHY THIS EXISTS
---------------
Three layers decide whether a feature is on in production:

1. ``src.features._FEATURE_REGISTRY`` — ``default_prod`` per flag. Tracked.
2. ``scripts.server.orchestrator_stack.PRODUCTION_FEATURE_WAVE_OVERRIDES`` —
   launch-time wave gating that overrides (1) in the ``ORCHESTRATOR_FEATURE_*``
   env block. Tracked.
3. ``orchestration/runtime_flags.json`` — written by the API at runtime
   (``set_by: api:127.0.0.1``) and read back by every worker. It wins over
   (1) and (2). **Gitignored, untracked, and correctly so** — it is live
   mutable state, rewritten constantly, and tracking it would churn every run
   and conflict across the sessions sharing this clone.

Layers 1+2 are reproducible from a fresh clone. Layer 3 is not, so a clone
cannot reproduce how the system actually behaves, and nothing records when a
flag changed or why.

WHAT THIS MODULE ADDS
---------------------
A tracked *declaration of intent* — ``orchestration/runtime_flags.spec.yaml`` —
that says, per flag, what production is expected to run and why it deviates.
It deliberately does NOT restate anything computable:

* flag NAMES, one-line meanings, dependencies, ``default_test``/``default_prod``
  come from ``_FEATURE_REGISTRY``;
* the wave gating comes from ``PRODUCTION_FEATURE_WAVE_OVERRIDES``;
* the spec file carries only ``expected`` + ``reason`` + ``since``, which are
  operator intent and exist nowhere in code.

The default spec value is the sentinel ``baseline``, meaning "whatever layers
1+2 compute" — so the spec cannot drift out of sync with a default that moves
in code. ``scripts/validate/runtime_flags_drift.py`` joins all four layers and
prints the difference; ``tests/unit/test_runtime_flag_spec.py`` fails if a flag
exists in code but is missing from the spec.

Read-only with respect to behaviour: nothing here writes ``runtime_flags.json``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "orchestration" / "runtime_flags.spec.yaml"

#: Sentinel meaning "expect whatever the code-derived production baseline says".
BASELINE = "baseline"

#: Source roots scanned by :func:`referenced_flag_names`. Tests are excluded on
#: purpose: a test may legitimately construct a Features-shaped stub.
CODE_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "orchestration",
)

#: Callables that return a :class:`~src.features.Features` instance.
_FEATURE_FACTORIES = frozenset({"features", "get_features"})

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSEY = frozenset({"0", "false", "no", "off", "disabled"})


class SpecError(ValueError):
    """Raised when the tracked spec file is malformed or names unknown flags."""


# ── Layers 1 + 2: the code-derived production baseline ─────────────────────


def _registry() -> tuple[Any, ...]:
    from src.features import _FEATURE_REGISTRY

    return _FEATURE_REGISTRY


def registry_flag_names() -> list[str]:
    """Every flag name declared in code, sorted. The spec must cover all of these."""
    return sorted(spec.name for spec in _registry())


def wave_overrides() -> dict[str, bool]:
    """Launch-time wave gating applied on top of ``default_prod``."""
    from scripts.server.orchestrator_stack import PRODUCTION_FEATURE_WAVE_OVERRIDES

    return dict(PRODUCTION_FEATURE_WAVE_OVERRIDES)


def baseline_posture() -> tuple[dict[str, bool], dict[str, str]]:
    """Production posture a fresh clone would launch with, plus provenance.

    Returns ``(values, sources)`` where every registry flag is present and each
    source is ``registry:default_prod`` or ``stack:wave_override``.
    """
    values = {spec.name: bool(spec.default_prod) for spec in _registry()}
    sources = {name: "registry:default_prod" for name in values}
    for name, enabled in wave_overrides().items():
        if name not in values:
            # A wave override naming an unknown flag is itself a defect; surface
            # it rather than silently dropping it.
            raise SpecError(
                f"PRODUCTION_FEATURE_WAVE_OVERRIDES names unknown flag {name!r}"
            )
        values[name] = bool(enabled)
        sources[name] = "stack:wave_override"
    return values, sources


def flag_metadata() -> dict[str, dict[str, Any]]:
    """Per-flag metadata derived from the registry (never hand-restated)."""
    return {
        spec.name: {
            "env_var": f"ORCHESTRATOR_FEATURE_{spec.env_var}",
            "default_test": bool(spec.default_test),
            "default_prod": bool(spec.default_prod),
            "description": spec.description,
            "dependencies": tuple(spec.dependencies),
        }
        for spec in _registry()
    }


# ── Layer 3: the live runtime file (read-only) ─────────────────────────────


def live_posture(path: Path | None = None) -> dict[str, bool]:
    """Effective overrides the running workers read, via the same loader they use."""
    from src.features import runtime_flag_overrides, runtime_flags_path

    return runtime_flag_overrides(path or runtime_flags_path())


def live_records(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Raw ``{name: {value, set_by, ts}}`` from the live file, unfiltered.

    ``src.features._runtime_records`` silently drops entries whose name is not
    in the registry. This keeps them so retired flags left behind in the live
    file can be reported instead of vanishing.
    """
    from src.features import runtime_flags_path

    path = path or runtime_flags_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("flags", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, record in raw.items():
        if isinstance(record, dict):
            out[str(name)] = {
                "value": _coerce_bool(record.get("value")),
                "set_by": str(record.get("set_by") or "unknown"),
                "ts": str(record.get("ts") or ""),
            }
        else:
            out[str(name)] = {"value": _coerce_bool(record), "set_by": "legacy", "ts": ""}
    return out


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUTHY:
            return True
        if text in _FALSEY:
            return False
    return None


# ── The tracked spec ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class SpecEntry:
    """Declared intent for one flag. ``expected is None`` means follow baseline."""

    name: str
    expected: bool | None = None
    reason: str = ""
    since: str = ""

    @property
    def follows_baseline(self) -> bool:
        return self.expected is None


@dataclass(frozen=True)
class StateGate:
    """A behaviour-gating key inside ``orchestration/autopilot_state.json``."""

    key: str
    expected: Any = None
    reason: str = ""
    since: str = ""


@dataclass
class Spec:
    version: int = 1
    flags: dict[str, SpecEntry] = field(default_factory=dict)
    autopilot_state: dict[str, StateGate] = field(default_factory=dict)
    path: Path | None = None

    def expected_posture(self) -> dict[str, bool]:
        """Declared production posture: baseline, overridden by explicit pins."""
        values, _sources = baseline_posture()
        for name, entry in self.flags.items():
            if entry.expected is not None and name in values:
                values[name] = entry.expected
        return values


def _parse_expected(raw: Any, *, where: str) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text == BASELINE:
        return None
    if text in _TRUTHY:
        return True
    if text in _FALSEY:
        return False
    raise SpecError(f"{where}: expected must be on/off/baseline, got {raw!r}")


def load_spec(path: Path | None = None, *, tolerant: bool = False) -> Spec:
    """Parse the tracked spec. Raises :class:`SpecError` on anything malformed.

    ``tolerant=True`` keeps entries naming flags that are no longer in the
    registry instead of rejecting them. Only :func:`sync_spec` should use it —
    dropping retired entries is precisely the repair sync performs, so refusing
    to parse them would make the guard forbid its own remedy.
    """
    path = path or SPEC_PATH
    if not path.exists():
        raise SpecError(f"spec file missing: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: top level must be a mapping")

    known = set(registry_flag_names())
    flags_raw = raw.get("flags") or {}
    if not isinstance(flags_raw, dict):
        raise SpecError(f"{path}: 'flags' must be a mapping")

    entries: dict[str, SpecEntry] = {}
    for name, value in flags_raw.items():
        where = f"{path}: flags.{name}"
        if name not in known and not tolerant:
            raise SpecError(
                f"{where}: not a known feature flag "
                f"(retired or misspelled — run runtime_flags_drift.py --sync-spec)"
            )
        if isinstance(value, dict):
            unknown_keys = set(value) - {"expected", "reason", "since"}
            if unknown_keys:
                raise SpecError(f"{where}: unknown keys {sorted(unknown_keys)}")
            entries[name] = SpecEntry(
                name=name,
                expected=_parse_expected(value.get("expected"), where=where),
                reason=str(value.get("reason") or ""),
                since=str(value.get("since") or ""),
            )
        else:
            entries[name] = SpecEntry(name=name, expected=_parse_expected(value, where=where))
        if entries[name].expected is not None and not entries[name].reason:
            raise SpecError(
                f"{where}: a pinned expectation must carry a 'reason' — an "
                f"undocumented pin is the drift it is meant to prevent"
            )

    gates_raw = raw.get("autopilot_state") or {}
    if not isinstance(gates_raw, dict):
        raise SpecError(f"{path}: 'autopilot_state' must be a mapping")
    gates: dict[str, StateGate] = {}
    for key, value in gates_raw.items():
        where = f"{path}: autopilot_state.{key}"
        if not isinstance(value, dict):
            raise SpecError(f"{where}: must be a mapping with 'expected' and 'reason'")
        unknown_keys = set(value) - {"expected", "reason", "since"}
        if unknown_keys:
            raise SpecError(f"{where}: unknown keys {sorted(unknown_keys)}")
        if not value.get("reason"):
            raise SpecError(f"{where}: requires a 'reason'")
        gates[key] = StateGate(
            key=key,
            expected=value.get("expected"),
            reason=str(value.get("reason")),
            since=str(value.get("since") or ""),
        )

    return Spec(
        version=int(raw.get("version") or 1),
        flags=entries,
        autopilot_state=gates,
        path=path,
    )


def spec_coverage(spec: Spec | None = None) -> tuple[list[str], list[str]]:
    """Return ``(missing_from_spec, unknown_in_spec)`` against the code registry."""
    spec = spec if spec is not None else load_spec()
    known = set(registry_flag_names())
    declared = set(spec.flags)
    return sorted(known - declared), sorted(declared - known)


# ── Spec file rendering (deterministic, so --sync-spec diffs stay small) ────

_SPEC_HEADER = """\
# Declared production posture for the runtime feature-flag overlay.
#
# GENERATED SHELL, HAND-MAINTAINED INTENT.
#   * The flag NAMES below are derived from src/features.py::_FEATURE_REGISTRY
#     by `scripts/validate/runtime_flags_drift.py --sync-spec`. Do not add or
#     remove names by hand — add a FeatureSpec in code and re-sync.
#   * Defaults, one-line meanings, env vars and dependencies are NOT restated
#     here; they live in the registry and are joined in at report time
#     (`runtime_flags_drift.py --show`).
#   * `baseline` means "expect whatever src/features.py::default_prod plus
#     scripts/server/orchestrator_stack.py::PRODUCTION_FEATURE_WAVE_OVERRIDES
#     compute". It stays correct when those move.
#
# WHY: orchestration/runtime_flags.json is live state the API rewrites
# (`set_by: api:127.0.0.1`); it is gitignored on purpose and must stay that
# way. This file is the tracked record of what production is SUPPOSED to run,
# so a fresh clone can reproduce behaviour and a changed flag has a reason.
#
# To pin a deviation, replace `baseline` with:
#     flag_name:
#       expected: on          # on | off | baseline
#       reason: "one line — why production deviates, and what unpins it"
#       since: YYYY-MM-DD
# A pin without a reason is rejected by the loader.
#
# Check drift:  python scripts/validate/runtime_flags_drift.py
#
# NOTE: every flag starts at `baseline` because nobody recorded why the live
# file deviates. `--strict` is therefore EXPECTED TO FAIL until each current
# `undeclared_override` is triaged — either pinned here with a reason, or the
# live value reverted through the API by the session that owns the stack. Do
# not silence it by pinning values you cannot justify; that reproduces the
# undocumented state in a tracked file instead of fixing it.
"""


def _yaml_scalar(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def render_spec(spec: Spec) -> str:
    """Render a spec back to YAML deterministically (sorted, stable formatting)."""
    lines = [_SPEC_HEADER, f"version: {spec.version}", "", "flags:"]
    for name in sorted(spec.flags):
        entry = spec.flags[name]
        if entry.follows_baseline and not entry.reason and not entry.since:
            lines.append(f"  {name}: {BASELINE}")
            continue
        lines.append(f"  {name}:")
        expected = BASELINE if entry.expected is None else ("on" if entry.expected else "off")
        lines.append(f"    expected: {expected}")
        if entry.reason:
            lines.append(f"    reason: {_yaml_scalar(entry.reason)}")
        if entry.since:
            lines.append(f"    since: {entry.since}")
    lines.append("")
    lines.append("# Behaviour-gating keys inside orchestration/autopilot_state.json.")
    lines.append("# That file has a code-derived bootstrap")
    lines.append("# (scripts/autopilot/autopilot.py::_default_state), so it is NOT restated")
    lines.append("# here — only the keys an operator flipped, which exist nowhere in code.")
    lines.append("autopilot_state:")
    if not spec.autopilot_state:
        lines.append("  {}")
    for key in sorted(spec.autopilot_state):
        gate = spec.autopilot_state[key]
        lines.append(f"  {key}:")
        lines.append(f"    expected: {json.dumps(gate.expected)}")
        lines.append(f"    reason: {_yaml_scalar(gate.reason)}")
        if gate.since:
            lines.append(f"    since: {gate.since}")
    return "\n".join(lines) + "\n"


def sync_spec(spec: Spec | None = None) -> tuple[Spec, list[str], list[str]]:
    """Add flags new in code as ``baseline``; drop entries for retired flags.

    Returns ``(synced_spec, added, removed)``. Hand-written pins survive.
    """
    if spec is None:
        try:
            spec = load_spec(tolerant=True)
        except SpecError:
            spec = Spec()
    known = set(registry_flag_names())
    added = sorted(known - set(spec.flags))
    removed = sorted(set(spec.flags) - known)
    flags = {name: entry for name, entry in spec.flags.items() if name in known}
    for name in added:
        flags[name] = SpecEntry(name=name)
    return (
        Spec(
            version=spec.version,
            flags=flags,
            autopilot_state=spec.autopilot_state,
            path=spec.path,
        ),
        added,
        removed,
    )


# ── Drift ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Drift:
    flag: str
    kind: str
    expected: bool | None
    effective: bool | None
    baseline: bool | None
    live_present: bool
    set_by: str = ""
    ts: str = ""
    reason: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


#: Drift kinds that mean "running config disagrees with tracked intent".
BLOCKING_KINDS = frozenset(
    {"undeclared_override", "contradicts_spec", "dependency_violation"}
)


def effective_posture(
    spec: Spec | None = None, live_path: Path | None = None
) -> dict[str, bool]:
    """What the workers actually resolve to: baseline, overridden by the live file."""
    values, _ = baseline_posture()
    values.update(live_posture(live_path))
    return values


def compute_drift(
    spec: Spec | None = None, live_path: Path | None = None
) -> list[Drift]:
    """Diff the live runtime flag file against the tracked spec.

    Kinds:
      ``undeclared_override``  live differs from the code baseline and the spec
                               says "baseline" — running config nobody declared.
      ``contradicts_spec``     the spec pins a value the running config does not
                               have (including "pinned but no live entry").
      ``redundant_override``   live entry equals the baseline — dead weight in a
                               file that otherwise records deliberate deviation.
      ``unknown_flag_in_live`` live entry for a flag not in the registry; the
                               loader silently ignores it, so it is dead config.
      ``dependency_violation`` the effective posture fails Features.validate().
    """
    spec = spec if spec is not None else load_spec()
    baseline, _sources = baseline_posture()
    expected = spec.expected_posture()
    records = live_records(live_path)
    known = set(baseline)

    drifts: list[Drift] = []
    for name in sorted(known):
        record = records.get(name)
        live_present = record is not None and record.get("value") is not None
        live_value = record["value"] if live_present else None
        effective = live_value if live_present else baseline[name]
        entry = spec.flags.get(name)
        pinned = entry is not None and entry.expected is not None
        base = baseline[name]

        if effective != expected[name]:
            if pinned:
                kind = "contradicts_spec"
                detail = (
                    "spec pins this value but the live file does not carry it"
                    if not live_present
                    else "live value contradicts the pinned expectation"
                )
            else:
                kind = "undeclared_override"
                detail = (
                    "live runtime_flags.json overrides the code baseline with no "
                    "tracked reason"
                )
            drifts.append(
                Drift(
                    flag=name,
                    kind=kind,
                    expected=expected[name],
                    effective=effective,
                    baseline=base,
                    live_present=live_present,
                    set_by=(record or {}).get("set_by", ""),
                    ts=(record or {}).get("ts", ""),
                    reason=(entry.reason if entry else ""),
                    detail=detail,
                )
            )
        elif live_present and live_value == base and not pinned:
            drifts.append(
                Drift(
                    flag=name,
                    kind="redundant_override",
                    expected=expected[name],
                    effective=effective,
                    baseline=base,
                    live_present=True,
                    set_by=record.get("set_by", ""),
                    ts=record.get("ts", ""),
                    detail="live entry matches the baseline; override adds nothing",
                )
            )

    for name in sorted(set(records) - known):
        record = records[name]
        drifts.append(
            Drift(
                flag=name,
                kind="unknown_flag_in_live",
                expected=None,
                effective=record.get("value"),
                baseline=None,
                live_present=True,
                set_by=record.get("set_by", ""),
                ts=record.get("ts", ""),
                detail="not in _FEATURE_REGISTRY; silently ignored at load time",
            )
        )

    for message in _dependency_errors(effective_posture(spec, live_path)):
        drifts.append(
            Drift(
                flag=message.split()[0],
                kind="dependency_violation",
                expected=None,
                effective=None,
                baseline=None,
                live_present=False,
                detail=message,
            )
        )
    return drifts


def _dependency_errors(posture: dict[str, bool]) -> list[str]:
    from src.features import Features

    try:
        candidate = Features(**posture)
    except TypeError as exc:  # pragma: no cover - registry/dataclass drift
        return [f"posture is not constructible: {exc}"]
    return [
        error
        for error in candidate.validate()
        # RestrictedPython availability is an environment fact, not flag drift.
        if "RestrictedPython library" not in error
    ]


def autopilot_state_drift(
    spec: Spec | None = None, state_path: Path | None = None
) -> list[dict[str, Any]]:
    """Compare declared autopilot_state gates against the live state file."""
    spec = spec if spec is not None else load_spec()
    if not spec.autopilot_state:
        return []
    state_path = state_path or (REPO_ROOT / "orchestration" / "autopilot_state.json")
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [
            {
                "key": "*",
                "status": "unreadable",
                "detail": f"{state_path}: {exc}",
            }
        ]
    rows: list[dict[str, Any]] = []
    for key in sorted(spec.autopilot_state):
        gate = spec.autopilot_state[key]
        present = key in state
        live = state.get(key)
        rows.append(
            {
                "key": key,
                "expected": gate.expected,
                "live": live,
                "present": present,
                "status": "ok" if (present and live == gate.expected) else "drift",
                "reason": gate.reason,
                "since": gate.since,
            }
        )
    return rows


# ── Rot guard: flag names actually read off a Features object ──────────────


class _FeatureAttrVisitor(ast.NodeVisitor):
    """Collect attribute reads on objects bound to ``features()``/``get_features()``.

    Scope-aware: a name bound in one function does not leak into another, which
    is what keeps short names like ``f`` from producing false positives.
    """

    def __init__(self) -> None:
        self.found: set[str] = set()
        self._scopes: list[set[str]] = [set()]

    # -- scope handling ----------------------------------------------------
    def _push(self) -> None:
        self._scopes.append(set())

    def _pop(self) -> None:
        self._scopes.pop()

    def _bound(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    def _bind(self, name: str) -> None:
        self._scopes[-1].add(name)

    def _visit_scope(self, node: ast.AST) -> None:
        self._push()
        self.generic_visit(node)
        self._pop()

    visit_FunctionDef = _visit_scope  # type: ignore[assignment]
    visit_AsyncFunctionDef = _visit_scope  # type: ignore[assignment]
    visit_Lambda = _visit_scope  # type: ignore[assignment]

    # -- binding detection -------------------------------------------------
    @staticmethod
    def _is_factory_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        return name in _FEATURE_FACTORIES

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_factory_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._bind(target.id)
                elif isinstance(target, ast.Attribute):
                    self._bind(target.attr)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and self._is_factory_call(node.value):
            if isinstance(node.target, ast.Name):
                self._bind(node.target.id)
            elif isinstance(node.target, ast.Attribute):
                self._bind(node.target.attr)
        self.generic_visit(node)

    # -- attribute reads ---------------------------------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        value = node.value
        if self._is_factory_call(value):
            self.found.add(node.attr)
        elif isinstance(value, ast.Name) and self._bound(value.id):
            self.found.add(node.attr)
        elif isinstance(value, ast.Attribute) and self._bound(value.attr):
            self.found.add(node.attr)
        self.generic_visit(node)


def _iter_python_files(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def referenced_flag_names(roots: Iterable[Path] | None = None) -> set[str]:
    """Flag-shaped attributes read off a ``Features`` object anywhere in the tree.

    This is a LOWER BOUND — flags reached via ``getattr``/``**kwargs`` are not
    visible to a static scan — so it must never be used as the authoritative
    flag list. Its job is the reverse direction: anything read off a Features
    object had better be in the registry (and therefore in the spec).
    """
    from src.features import Features

    method_names = {
        name
        for name in dir(Features)
        if not name.startswith("__")
    } - {f.name for f in dataclasses.fields(Features)}

    found: set[str] = set()
    for path in _iter_python_files(roots or CODE_ROOTS):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        visitor = _FeatureAttrVisitor()
        visitor.visit(tree)
        found |= visitor.found
    return found - method_names
