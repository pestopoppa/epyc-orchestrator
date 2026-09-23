"""EV-6b: shared cross-family verification classification + role resolution.

``scripts/autopilot/eval_tower.py`` (the rubric-judge gate, ``check_cross_family``
/ ``check_cross_family_status``, also injected into
``scripts/autopilot/skill_efficacy.require_cross_family``) and
``scripts/benchmark/debug_scorer.py`` (the ``llm_judge`` scoring path,
``_score_llm_judge``) both need to answer the same question before letting one
model's verdict stand in judgment of another's answer: are the generator and
the verifier from different model FAMILIES? Confirmation bias from a same-family
verifier amplifies 52%->87% (see ``handoffs/.../eval-tower-verification.md``).

Until EV-6b this lived only in ``eval_tower.py``, as a plain substring match over
whatever two strings a caller happened to pass in. On its only production call
site (``eval_tower.py`` rubric-judge dispatch) the caller passes a ROLE NAME
(e.g. ``"architect_general"``) as the verifier — no family pattern matches a
role name, so the family resolved to ``"unknown"``, and a permissive default
(``gen_family == "unknown"``) returned True for every generator. The check was
vacuous on its only real caller. The ``llm_judge`` scoring path never ran any
check at all.

This module is the one place that:
  1. classifies a model NAME/PATH into a family (substring match, same
     patterns as before, extended with gpt-oss/glm/kimi/nemotron/minimax), and
  2. resolves a caller's input to a model name FIRST when it names a registry
     ROLE (``roles[role].model.name`` in the lean registry), and
  3. fails CLOSED: an unknown family on either side is NEVER independent, and
     is labelled ``"unverified"`` so it can never be read as a confirmed
     cross-family judgment (as opposed to ``"same_family"``, also not
     independent, but for a different, known reason).

One copy of an admissibility rule, mirroring the ``measurement_guards.py``
precedent in this same package: two copies drift silently, and the drift only
ever shows up as two paths disagreeing about the same measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "VERIFICATION_FAMILIES",
    "UNKNOWN_FAMILY",
    "classify_model_family",
    "resolve_family_input",
    "cross_family_status",
]

_DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"


def _load_lean_registry(path: Path) -> dict[str, Any]:
    """Read the lean registry; an unreadable file yields {} (roles then stay unverified)."""
    try:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception:  # noqa: BLE001 - fail closed via UNKNOWN_FAMILY, never crash a scorer
        return {}
    return data if isinstance(data, dict) else {}


def _served_model_for_role(registry: dict[str, Any], role: str) -> str | None:
    """``roles[role].model.name`` from a lean-registry-shaped dict, or None."""
    block = ((registry or {}).get("roles") or {}).get(role) or {}
    model = block.get("model") if isinstance(block, dict) else None
    name = model.get("name") if isinstance(model, dict) else None
    return str(name) if name is not None else None


# EV-6: Cross-family verification constraint. Verifier model must be from a
# different family than generator to avoid confirmation bias. See
# eval-tower-verification.md for research basis (confirmation bias amplifies
# 52%->87%).
VERIFICATION_FAMILIES: dict[str, set[str]] = {
    "qwen": {"Qwen", "qwen", "QwQ"},
    "llama": {"Llama", "llama", "Meta-Llama"},
    "deepseek": {"DeepSeek", "deepseek"},
    "ouro": {"Ouro", "ouro", "ByteDance"},
    "mistral": {"Mistral", "mistral"},
    "gemma": {"Gemma", "gemma", "Google"},
    # EV-6b (2026-09-23): the lineup now includes served models from these
    # families; a missing pattern is exactly what makes a pairing silently
    # "unknown" and, before EV-6b, silently permissive.
    "gpt-oss": {"gpt-oss", "GPT-OSS", "gptoss"},
    "glm": {"GLM", "glm"},
    "kimi": {"Kimi", "kimi"},
    "nemotron": {"Nemotron", "nemotron"},
    "minimax": {"MiniMax", "minimax", "Minimax"},
}

UNKNOWN_FAMILY = "unknown"


def classify_model_family(model_name: str) -> str:
    """Substring-match ``model_name`` against ``VERIFICATION_FAMILIES``.

    Case-insensitive, same matching rule for every family. Returns
    ``UNKNOWN_FAMILY`` when nothing matches.
    """
    name = str(model_name or "")
    for family, patterns in VERIFICATION_FAMILIES.items():
        if any(p.lower() in name.lower() for p in patterns):
            return family
    return UNKNOWN_FAMILY


_REGISTRY_CACHE: dict[str, Any] | None = None


def _registry() -> dict[str, Any]:
    """Lazily load + cache the lean model registry for role->model lookups.

    A module-level indirection (rather than a call-site default argument) so
    tests can monkeypatch this function directly to inject a fixture registry
    without touching disk or the live production registry.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _load_lean_registry(_DEFAULT_REGISTRY)
    return _REGISTRY_CACHE


def resolve_family_input(name: str, *, registry: dict[str, Any] | None = None) -> str:
    """Resolve ``name`` to its served model name if it names a registry ROLE.

    A caller may pass either a role (``"architect_general"``) or an
    already-resolved model name/path (``"Qwen3.8-27B-Q8_0"``,
    ``/mnt/.../model.gguf``). Only the former is a registry key, so this
    tries the role lookup first and falls back to the input unchanged when
    it isn't one (idempotent on an already-resolved model name/path, and
    idempotent when the registry can't be loaded at all — an unresolved role
    then simply stays unresolved and classifies as ``UNKNOWN_FAMILY``, which
    is the correct fail-closed outcome, not a crash).
    """
    cleaned = str(name or "").strip()
    if not cleaned:
        return cleaned
    reg = registry if registry is not None else _registry()
    return _served_model_for_role(reg, cleaned) or cleaned


def cross_family_status(
    generator: str, verifier: str, *, registry: dict[str, Any] | None = None
) -> tuple[bool, str]:
    """Return ``(independent, status)`` for a generator/verifier pairing.

    FAILS CLOSED: ``independent`` is True only when BOTH sides resolve to a
    KNOWN, DIFFERENT family. ``status`` is one of:

      * ``"cross_family"``  — both known, different families (independent)
      * ``"same_family"``   — both known, same family (not independent)
      * ``"unverified"``    — at least one side's family could not be
        classified (not independent — this is the fix for the vacuous
        permissive default: an unknown family used to count as safe)

    ``generator``/``verifier`` may each be a role name or an already-resolved
    model name/path; both are resolved via ``resolve_family_input`` first.
    """
    gen_resolved = resolve_family_input(generator, registry=registry)
    ver_resolved = resolve_family_input(verifier, registry=registry)
    gen_family = classify_model_family(gen_resolved)
    ver_family = classify_model_family(ver_resolved)
    if gen_family == UNKNOWN_FAMILY or ver_family == UNKNOWN_FAMILY:
        return False, "unverified"
    if gen_family == ver_family:
        return False, "same_family"
    return True, "cross_family"
