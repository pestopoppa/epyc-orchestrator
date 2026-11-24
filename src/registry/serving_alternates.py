"""``serving_shape.alternates`` -- named serving configs a champion ships beside its optimum.

PROMOTION-REGISTRY-SERVING-1 (operator 2026-11-24). A champion's serving-config artifact
carries the fastest screened config (``optimal``) plus 2-3 named alternates (``latency``,
``throughput``, ``high-temperature``...). The optimum is written into the registry's
``server_mode.<role>`` knobs and ``serving_shape``; each alternate is written as

    server_mode.<role>.serving_shape.alternates:
      <name>:
        argv_delta: {<knob or --flag>: <scalar | list of scalars>, ...}
        provenance: {row_id, result_json, result_sha256, [store]}

An alternate is a DELTA against the declared registry serving recipe (the optimum, once promoted),
never a second full recipe: it names only what differs. ``argv_delta`` keys are llama-server flags
(``-b``, ``--parallel``...) with their value (``true`` for a bare flag). It is declared data, not selected data -- nothing in the launcher reads it, and
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
            if isinstance(value, list):
                if not all(_is_scalar(v) for v in value):
                    raise ServingAlternatesError(
                        f"{where}.argv_delta.{knob} list must hold scalars only"
                    )
            elif not _is_scalar(value):
                raise ServingAlternatesError(
                    f"{where}.argv_delta.{knob} must be a scalar or a list of scalars, got {type(value).__name__}"
                )
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
