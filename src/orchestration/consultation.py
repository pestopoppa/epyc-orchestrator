"""One-shot internal consult helper for the Interaction lifecycle.

This module is intentionally not wired into any live route by itself. Callers
must opt in explicitly, which keeps the P2 consult path inert while the P1 bake
gate is still collecting evidence.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import jsonschema
import yaml

from src.orchestration.interaction import INTERACTION_POLICY_VERSION


ORCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS_PATH = ORCH_ROOT / "orchestration" / "interaction_skills.yaml"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class ConsultationDenied(RuntimeError):
    """Consultation could not produce an advisory, but caller may proceed."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{reason}{suffix}")


@dataclass(frozen=True)
class InteractionSkill:
    consultant_role: str
    name: str
    kind: str
    description: str
    output_schema: dict[str, Any]
    max_output_tokens: int
    scheduler_defaults: dict[str, Any]
    tools_budget: int
    cache_ttl_seconds: int

    @property
    def schema_hash(self) -> str:
        canonical = json.dumps(self.output_schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConsultationDenied("skill_spec_missing", str(path)) from exc
    if not isinstance(loaded, dict):
        raise ConsultationDenied("skill_spec_invalid", "top-level YAML must be a mapping")
    return loaded


def load_interaction_skill(
    consultant_role: str,
    skill: str,
    *,
    path: Path = DEFAULT_SKILLS_PATH,
) -> InteractionSkill:
    """Load and validate a single interaction skill spec."""
    data = _load_yaml(path)
    skills = data.get("interaction_skills")
    if not isinstance(skills, dict):
        raise ConsultationDenied("skill_spec_invalid", "missing interaction_skills mapping")
    role_skills = skills.get(consultant_role)
    if not isinstance(role_skills, dict):
        raise ConsultationDenied("skill_not_found", f"{consultant_role}.{skill}")
    raw = role_skills.get(skill)
    if not isinstance(raw, dict):
        raise ConsultationDenied("skill_not_found", f"{consultant_role}.{skill}")

    output_schema = raw.get("output_schema")
    scheduler_defaults = raw.get("scheduler_defaults", {})
    if not isinstance(output_schema, dict):
        raise ConsultationDenied("skill_spec_invalid", "output_schema must be a mapping")
    if not isinstance(scheduler_defaults, dict):
        raise ConsultationDenied("skill_spec_invalid", "scheduler_defaults must be a mapping")
    try:
        max_output_tokens = int(raw.get("max_output_tokens", 400))
        tools_budget = int(raw.get("tools_budget", 0))
        cache_ttl_seconds = int(raw.get("cache_ttl_seconds", 1800))
    except (TypeError, ValueError) as exc:
        raise ConsultationDenied("skill_spec_invalid", "numeric fields must be integers") from exc
    if str(raw.get("kind", "")).strip() != "consult":
        raise ConsultationDenied("skill_spec_invalid", "skill kind must be consult")
    if tools_budget != 0:
        raise ConsultationDenied("skill_spec_invalid", "P2 consult tools_budget must be 0")

    return InteractionSkill(
        consultant_role=consultant_role,
        name=skill,
        kind="consult",
        description=str(raw.get("description", "")).strip(),
        output_schema=output_schema,
        max_output_tokens=max_output_tokens,
        scheduler_defaults=dict(scheduler_defaults),
        tools_budget=tools_budget,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ConsultationDenied("schema_violation", "empty advisory")
    candidates = [text]
    match = _JSON_FENCE_RE.search(text)
    if match:
        candidates.insert(0, match.group(1).strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ConsultationDenied("schema_violation", "advisory is not a JSON object")


def _validate_advisory(advisory: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        jsonschema.validate(advisory, schema)
    except jsonschema.ValidationError as exc:
        raise ConsultationDenied("schema_violation", exc.message) from exc


def build_consult_prompt(
    *,
    requester_role: str,
    consultant_role: str,
    skill: InteractionSkill,
    context: str,
) -> str:
    """Build the one-shot advisory prompt sent to the consultant role."""
    schema_json = json.dumps(skill.output_schema, sort_keys=True)
    return (
        f"You are {consultant_role} providing a one-shot internal consult for "
        f"{requester_role}.\n"
        f"Skill: {skill.name}\n"
        f"Purpose: {skill.description}\n\n"
        "Review the draft/context below before commit. Return only a JSON object "
        "that satisfies the schema. Do not include prose outside JSON.\n\n"
        f"JSON schema:\n{schema_json}\n\n"
        f"Draft/context:\n{context}"
    )


def _request_context(primitives: Any, scheduler_defaults: dict[str, Any], override_priority: str | None):
    request_context = getattr(primitives, "request_context", None)
    if not callable(request_context):
        return nullcontext()
    priority = override_priority or scheduler_defaults.get("priority") or "background"
    workload_class = scheduler_defaults.get("workload_class") or "consult"
    max_queue_wait_ms = scheduler_defaults.get("max_queue_wait_ms")
    try:
        max_queue_wait_ms = None if max_queue_wait_ms is None else int(max_queue_wait_ms)
    except (TypeError, ValueError):
        max_queue_wait_ms = None
    return request_context(
        priority=str(priority),
        workload_class=str(workload_class),
        max_queue_wait_ms=max_queue_wait_ms,
    )


def consult(
    consultant_role: str,
    requester_role: str,
    skill: str,
    context: str,
    primitives: Any,
    *,
    override_max_tokens: int | None = None,
    override_priority: str | None = None,
    skills_path: Path = DEFAULT_SKILLS_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one bounded advisory consult and return ``(advisory, stats)``.

    ``ConsultationDenied`` is non-fatal by design; callers should log the reason
    and continue without advisory unless they explicitly choose otherwise.
    """
    spec = load_interaction_skill(consultant_role, skill, path=skills_path)
    prompt = build_consult_prompt(
        requester_role=requester_role,
        consultant_role=consultant_role,
        skill=spec,
        context=context,
    )
    n_tokens = int(override_max_tokens or spec.max_output_tokens)
    try:
        from src.scheduling.contention_gate import ContentionDenied
    except Exception:  # pragma: no cover - import exists in production/tests
        ContentionDenied = RuntimeError  # type: ignore[assignment]

    try:
        with _request_context(primitives, spec.scheduler_defaults, override_priority):
            raw = primitives.llm_call(
                prompt,
                role=consultant_role,
                n_tokens=n_tokens,
                json_schema=spec.output_schema,
            )
    except ContentionDenied as exc:  # type: ignore[misc]
        raise ConsultationDenied("contention_skip", str(exc)) from exc

    advisory = _parse_json_object(str(raw or ""))
    _validate_advisory(advisory, spec.output_schema)
    stats = {
        "interaction_type": "consult",
        "interaction_policy_version": INTERACTION_POLICY_VERSION,
        "consultant_role": consultant_role,
        "requester_role": requester_role,
        "skill": skill,
        "schema_hash": spec.schema_hash,
        "max_output_tokens": n_tokens,
        "cache_ttl_seconds": spec.cache_ttl_seconds,
    }
    return advisory, stats
