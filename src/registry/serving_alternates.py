"""``serving_shape.alternates`` -- named serving configs a champion ships beside its optimum.

PROMOTION-REGISTRY-SERVING-1 (operator 2026-11-24). A champion's serving-config artifact
carries the fastest screened config (``optimal``) plus 2-3 named alternates (``latency``,
``throughput``, ``low-temperature``...). The optimum is written into the registry's
``server_mode.<role>`` knobs and ``serving_shape``; each alternate is written as

    server_mode.<role>.serving_shape.alternates:
      <name>:
        argv_delta: {<knob or --flag>: <scalar | list of scalars>, ...}
        provenance: {row_id, result_json, result_sha256, [store]}

An alternate is a DELTA against the declared registry serving recipe (the optimum, once promoted),
never a second full recipe: it names only what differs. ``argv_delta`` keys are llama-server flags
(``-b``, ``--parallel``...) with their value (``true`` for a bare switch ONLY: every other flag requires a typed
value, and a string value is one argv token, so ``|`` and whitespace are rejected). A temperature knob is
0-1.0: no high-temperature testing (operator 2026-11-24), the temperature alternate is low-temperature,
T 0-0.6. It is declared data, not selected data -- nothing in the launcher reads it, and
the compiled argv is unchanged by its presence. It exists so that moving off the optimum (a
temperature move, a latency pin) is an informed, evidenced, one-key choice instead of a re-derivation.

The master registry is the source of truth; ``registry_compiler.compile_lean`` copies
``server_mode`` records whole, so alternates survive into the lean output, and it calls
:func:`validate_registry_alternates` so a malformed slot fails the compile instead of riding
into a signed promotion.
"""

from __future__ import annotations

import re
from typing import Any

ALTERNATES_KEY = "alternates"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: provenance keys every alternate must carry (kernel-manifests README: row id, RESULT.json
#: path and sha256; ``store`` -- the kernel store it ran on -- is accepted and recommended).
REQUIRED_PROVENANCE = ("row_id", "result_json", "result_sha256")


class ServingAlternatesError(ValueError):
    """A ``serving_shape.alternates`` declaration is malformed."""


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


# Knob classes. Mirrors scripts/champion/serving_config/validate_serving_config.py in epyc-root (BARE_FLAGS,
# INT_FLAGS, FLOAT_FLAGS); tests/unit/test_serving_alternates.py asserts the two stay in step. A flag is a
# bare switch only if listed in BARE_FLAGS; every OTHER ``-flag`` takes a value, so an unknown flag fails closed.
BARE_FLAGS = frozenset((
    "--kv-unified", "-kvu", "--no-kv-unified", "--no-warmup", "--no-mmap", "--mlock", "--cont-batching",
    "--no-cont-batching", "--jinja", "--no-context-shift", "--swa-full", "-nkvo", "--no-kv-offload"))
INT_FLAGS = frozenset((
    "-c", "--ctx-size", "-b", "--batch-size", "-ub", "--ubatch-size", "-np", "--parallel",
    "--ctx-checkpoints", "-ctxcp", "--cache-ram", "-cram", "-t", "--threads", "-ngl", "--n-gpu-layers",
    "--draft-max", "--draft-min", "--spec-draft-n-max", "-n", "--predict", "--top-k"))
FLOAT_FLAGS = frozenset(("--temp", "--temperature", "--top-p", "--min-p"))
TEMP_FLAGS = frozenset(("--temp", "--temperature"))
#: registry-knob spellings (an alternate may key by a registry knob name instead of a flag)
INT_KNOBS = frozenset(("cache_ram", "ctx_checkpoints", "slots", "np", "n_ctx", "batch", "ubatch", "threads",
                       "top_k"))
FLOAT_KNOBS = frozenset(("temp", "temperature", "top_p", "min_p"))
TEMP_KNOBS = frozenset(("temp", "temperature"))
#: operator 2026-11-24: no high-temperature testing (T > 1.0); the temperature alternate is T 0-0.6.
MAX_TEMPERATURE = 1.0


def _token_problem(value: str) -> str | None:
    if "|" in value:
        return "contains '|' (a sweep is not a value; name one arm)"
    if any(ch.isspace() for ch in value):
        return "contains whitespace (an argv value is a single token)"
    return None


def _check_knob_value(knob: str, value: Any, where: str) -> None:
    """Type-check one ``argv_delta`` entry. Raises :class:`ServingAlternatesError`."""
    is_flag = knob.startswith("-")
    if knob in BARE_FLAGS:
        if value is not True:
            raise ServingAlternatesError(f"{where} is a bare switch; its value must be true, got {value!r}")
        return
    if isinstance(value, list):
        if not all(_is_scalar(v) for v in value):
            raise ServingAlternatesError(f"{where} list must hold scalars only")
        if is_flag or knob in INT_KNOBS or knob in FLOAT_KNOBS:
            raise ServingAlternatesError(f"{where} takes one value, not a list")
        items = value
    elif not _is_scalar(value):
        raise ServingAlternatesError(
            f"{where} must be a scalar or a list of scalars, got {type(value).__name__}"
        )
    else:
        items = [value]
    numeric = knob in INT_FLAGS or knob in FLOAT_FLAGS or knob in INT_KNOBS or knob in FLOAT_KNOBS
    if value is None or isinstance(value, bool):
        if is_flag or numeric:
            raise ServingAlternatesError(
                f"{where} takes a value: a flag without one is not a delta (got {value!r}; only a known bare "
                "switch is true)"
            )
        if value is None:
            raise ServingAlternatesError(f"{where} must not be null")
        return
    for item in items:
        if isinstance(item, str):
            bad = _token_problem(item)
            if bad:
                raise ServingAlternatesError(f"{where} value {item!r} {bad}")
            if not item:
                raise ServingAlternatesError(f"{where} value must not be empty")
        if knob in INT_FLAGS or knob in INT_KNOBS:
            ok = (isinstance(item, int) and not isinstance(item, bool)) or (
                isinstance(item, str) and re.fullmatch(r"-?[0-9]+", item) is not None)
            if not ok or int(item) < (-1 if knob in ("-ngl", "--n-gpu-layers") else 0):
                raise ServingAlternatesError(f"{where} value {item!r} must be a non-negative integer")
        elif knob in FLOAT_FLAGS or knob in FLOAT_KNOBS:
            try:
                number = float(item)
            except (TypeError, ValueError):
                raise ServingAlternatesError(f"{where} value {item!r} must be a number") from None
            if number < 0:
                raise ServingAlternatesError(f"{where} value {item!r} must be non-negative")
            if (knob in TEMP_FLAGS or knob in TEMP_KNOBS) and number > MAX_TEMPERATURE:
                raise ServingAlternatesError(
                    f"{where} value {item!r} exceeds {MAX_TEMPERATURE}: no high-temperature testing "
                    "(operator 2026-11-24); the temperature alternate is low-temperature, T 0-0.6"
                )


def parse_alternates(raw: Any, *, source: str = "serving_shape.alternates") -> dict[str, dict[str, Any]]:
    """Validate and normalize an alternates mapping. ``None`` means undeclared (``{}``)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ServingAlternatesError(
            f"{source} must be a mapping of alternate name to {{argv_delta, provenance}}"
        )
    out: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        where = f"{source}.{name}"
        if not isinstance(name, str) or not name.strip():
            raise ServingAlternatesError(f"{source} has invalid alternate name {name!r}")
        if not isinstance(entry, dict):
            raise ServingAlternatesError(f"{where} must be a mapping with argv_delta and provenance")
        unknown = sorted(set(entry) - {"argv_delta", "provenance"})
        if unknown:
            raise ServingAlternatesError(
                f"{where} has unknown key(s) {unknown}; an alternate carries only argv_delta and provenance"
            )
        delta = entry.get("argv_delta")
        if not isinstance(delta, dict) or not delta:
            raise ServingAlternatesError(
                f"{where}.argv_delta must be a non-empty mapping (an alternate that changes nothing is not one)"
            )
        for knob, value in delta.items():
            if not isinstance(knob, str) or not knob.strip():
                raise ServingAlternatesError(f"{where}.argv_delta has invalid key {knob!r}")
            if _token_problem(knob):
                raise ServingAlternatesError(f"{where}.argv_delta key {knob!r} {_token_problem(knob)}")
            _check_knob_value(knob, value, f"{where}.argv_delta.{knob}")
        prov = entry.get("provenance")
        if not isinstance(prov, dict):
            raise ServingAlternatesError(f"{where}.provenance must be a mapping")
        missing = [k for k in REQUIRED_PROVENANCE if not prov.get(k)]
        if missing:
            raise ServingAlternatesError(
                f"{where}.provenance is missing {missing}; every alternate cites the row, RESULT.json and its sha256"
            )
        sha = prov["result_sha256"]
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            raise ServingAlternatesError(
                f"{where}.provenance.result_sha256 must be 64 lowercase hex characters"
            )
        out[name] = {"argv_delta": dict(delta), "provenance": dict(prov)}
    return out


def validate_registry_alternates(registry: dict[str, Any]) -> list[str]:
    """Validate every ``serving_shape.alternates`` in a registry document.

    Looks under ``server_mode.<name>.serving_shape`` and ``roles.<name>.serving_shape``.
    Returns the paths that declare alternates; raises :class:`ServingAlternatesError`
    on the first malformed one.
    """
    declared: list[str] = []
    for section in ("server_mode", "roles"):
        block = registry.get(section)
        if not isinstance(block, dict):
            continue
        for name, entry in block.items():
            shape = entry.get("serving_shape") if isinstance(entry, dict) else None
            if not isinstance(shape, dict) or ALTERNATES_KEY not in shape:
                continue
            path = f"{section}.{name}.serving_shape.{ALTERNATES_KEY}"
            parse_alternates(shape[ALTERNATES_KEY], source=path)
            declared.append(path)
    return declared
