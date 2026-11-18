"""Runtime overlay for env-read feature knobs (flag surface, 2026-11-18).

Several codemode / tool-picker / REPL / document knobs were read straight from ``os.environ`` at call time
(``ORCHESTRATOR_TOOL_SELECTOR=embed``, ``ORCHESTRATOR_REPL_HANDLES=1`` ...). A running API cannot have its
environment changed, so ``POST /config`` (and with it the autopilot) could not toggle them. They are now
registered in ``src.features._FEATURE_REGISTRY`` (their env names and defaults unchanged) and the code paths read
them through :func:`flag_env` instead of ``os.environ``.

``flag_env()`` is a read-only ``Mapping`` over ``os.environ`` with ONE difference: for an env var bound to a
registered flag (a ``FeatureSpec`` with ``env_on``/``env_off``), a live runtime override in the shared runtime-flag
file wins. With no runtime override every lookup returns exactly what ``os.environ`` holds, so a flag nobody toggles
is byte-identical to the old behaviour (tests that ``monkeypatch.setenv`` keep working, no cache involved).

Runtime override semantics (same precedence as ``features()``: runtime file > env > default):

* override ON  -> the env value when it already says "on" for that flag (``=oneshot`` survives), else ``env_on``;
* override OFF -> ``env_off``.

The runtime file is shared by every API worker and re-read at most once per ``TTL_S`` (mtime-checked), so a
``POST /config`` reaches all workers within about a second, the same latency as ``features()``.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

TTL_S = 1.0
_FEATURE_PREFIX = "ORCHESTRATOR_FEATURE_"

_lock = threading.Lock()
_cache: dict[str, object] = {"path": None, "mtime": None, "at": 0.0, "overrides": {}}
_bindings_cache: dict[str, list] | None = None


def _bindings() -> dict[str, list]:
    """env var name -> the FeatureSpecs bound to it (built once from the registry)."""
    global _bindings_cache
    if _bindings_cache is None:
        from src.features import _FEATURE_REGISTRY, ENV_PREFIX

        out: dict[str, list] = {}
        for spec in _FEATURE_REGISTRY:
            if spec.env_on or spec.env_off or spec.env_name:
                out.setdefault(spec.env_name or f"{ENV_PREFIX}{spec.env_var}", []).append(spec)
        _bindings_cache = out
    return _bindings_cache


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def runtime_overrides() -> dict[str, bool]:
    """Valid, unexpired runtime overrides from the shared file (cached ``TTL_S``; reloaded early on an mtime change)."""
    from src.features import runtime_flag_overrides, runtime_flags_path

    path = runtime_flags_path()
    now = time.monotonic()
    mtime = _mtime(path)
    with _lock:
        fresh = (
            _cache["path"] == path
            and _cache["mtime"] == mtime
            and now - float(_cache["at"]) < TTL_S  # type: ignore[arg-type]
        )
        if not fresh:
            _cache.update(path=path, mtime=mtime, at=now, overrides=runtime_flag_overrides(path) if mtime is not None else {})
        return dict(_cache["overrides"])  # type: ignore[arg-type]


def reset_cache() -> None:
    """Drop the cached runtime-file read and the binding table (tests)."""
    global _bindings_cache
    with _lock:
        _cache.update(path=None, mtime=None, at=0.0, overrides={})
        _bindings_cache = None


def _says_on(spec, raw: str) -> bool:
    if spec.env_parse is not None:
        return bool(spec.env_parse(raw))
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(specs: list, raw: str | None, overrides: Mapping[str, bool]) -> str | None:
    """The env value a code path should see for ``specs`` (all bound to one env var)."""
    if len(specs) == 1:
        spec = specs[0]
        ov = overrides.get(spec.name)
        if ov is None:
            return raw
        if ov:
            return raw if (raw is not None and _says_on(spec, raw)) else spec.env_on
        return spec.env_off
    # REPL_EXPLICIT_FINAL: two flags share one env var (tools-mode on/off, strict on/off).
    by_name = {s.name: s for s in specs}
    tools, strict = by_name.get("repl_explicit_final"), by_name.get("repl_explicit_final_strict")
    if tools is None or strict is None:  # pragma: no cover - registry guard
        return raw
    o_tools, o_strict = overrides.get("repl_explicit_final"), overrides.get("repl_explicit_final_strict")
    if o_tools is None and o_strict is None:
        return raw
    if o_strict:
        return "strict"
    if o_strict is False and (raw or "").strip().lower() in ("1", "true", "on", "yes", "strict"):
        raw = None  # runtime says "not strict": drop the env's strict request, fall through to the tools flag
    if o_tools is False:
        return "0"
    if o_tools:
        return raw if (raw is not None and (raw or "").strip().lower() not in ("0", "false", "off", "no")) else "tools"
    return raw


class _FlagEnv(Mapping):
    """``os.environ`` with runtime overrides applied for flag-bound names (see module docstring)."""

    def get(self, key, default=None):  # type: ignore[override]
        raw = os.environ.get(key)
        specs = _bindings().get(key)
        if specs is None:
            return default if raw is None else raw
        if raw is None:  # the ORCHESTRATOR_FEATURE_<X> spelling is a plain bool for the same flag
            fkey = _FEATURE_PREFIX + key[len("ORCHESTRATOR_"):]
            fraw = os.environ.get(fkey)
            if fraw is not None and len(specs) == 1:
                on = fraw.strip().lower() in {"1", "true", "yes", "on"}
                off = fraw.strip().lower() in {"0", "false", "no", "off"}
                if on or off:
                    raw = specs[0].env_on if on else specs[0].env_off
        value = _resolve(specs, raw, runtime_overrides())
        return default if value is None else value

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key) -> bool:  # type: ignore[override]
        return self.get(key) is not None

    def __iter__(self) -> Iterator[str]:
        seen = set(os.environ)
        seen.update(k for k in _bindings() if self.get(k) is not None)
        return iter(seen)

    def __len__(self) -> int:
        return sum(1 for _ in self)


_FLAG_ENV = _FlagEnv()


def flag_env() -> Mapping[str, str]:
    """The environment code paths should read flag-bound knobs from (a runtime-overridable view of ``os.environ``)."""
    return _FLAG_ENV
