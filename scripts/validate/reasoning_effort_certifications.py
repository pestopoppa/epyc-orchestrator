#!/usr/bin/env python3
"""Fail closed when a role's prompt-effort certification no longer matches its model.

The prompt-effort ladder is independent of native ``reasoning_budget`` / ``<think>``
settings.  A role becomes subject to this validator only when it declares
``reasoning_effort.level`` in the model registry.  Its certificate must then name
the same model, quant, active kernel era, and chat-template sha256 as the
currently bound role (E-7 amendment 2026-08-21: the chat template is a
calibration-voiding axis the original ``(model, quant, kernel_era)`` stamp did
not cover).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
DEFAULT_CERTIFICATIONS = REPO_ROOT / "orchestration" / "reasoning_effort_certifications.yaml"
DEFAULT_ERAS = REPO_ROOT / "orchestration" / "instrument_eras.yaml"
REQUIRED_CERTIFICATION_FIELDS = ("model", "quant", "kernel_era", "template_sha")


@dataclass(frozen=True)
class EffortCertificationResult:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return value


def _normalize_template_sha(value: Any) -> str | None:
    """Normalize a template sha for comparison.

    Certificates may record the full 64-hex sha256 of the served template, while
    the registry's ``served_template_sha256_12`` stores the 12-char short form.
    Either side is truncated to its first 12 hex chars when longer than 12, so a
    full sha (``1443ea9ab4bb0a2663b1...``) compares equal to its short form
    (``1443ea9ab4bb``).  Non-string and blank values normalize to ``None``.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped[:12] if len(stripped) > 12 else stripped


def _bound_template_sha(
    role: dict[str, Any], role_name: str, server_mode: dict[str, Any]
) -> str | None:
    """Resolve the chat-template sha the role is bound to, if any.

    The role-level ``chat_template`` block (``served_template_sha256_12``) is
    preferred when it yields a usable value; otherwise the
    ``server_mode.<role_name>.chat_template`` block is consulted.  A role with
    no ``chat_template`` block binds no template axis, and the template check
    is vacuously satisfied (the certificate must still carry ``template_sha``).
    """
    for block in (role.get("chat_template"), server_mode.get(role_name, {}).get("chat_template")):
        if not isinstance(block, dict):
            continue
        bound = _normalize_template_sha(block.get("served_template_sha256_12"))
        if bound is not None:
            return bound
    return None


def _configured_effort_level(role: dict[str, Any]) -> str | None:
    effort = role.get("reasoning_effort")
    if not isinstance(effort, dict):
        return None
    level = effort.get("level")
    return level.strip() if isinstance(level, str) and level.strip() else None


def _bound_model(role: dict[str, Any]) -> tuple[str | None, str | None]:
    model = role.get("model")
    if not isinstance(model, dict):
        return None, None
    name = model.get("name")
    quant = model.get("quant")
    return (
        name.strip() if isinstance(name, str) and name.strip() else None,
        quant.strip() if isinstance(quant, str) and quant.strip() else None,
    )


def _parse_epoch(value: Any) -> float | None:
    """Parse an era-boundary timestamp (ISO string or epoch number) to epoch seconds."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _current_cpu_kernel_era(
    eras_path: Path, now: datetime | None = None
) -> tuple[str, str]:
    """Resolve the CURRENT production cpu-kernel era from the eras registry.

    ``orchestration/instrument_eras.yaml`` is the promotion record: every kernel
    cutover adds a new ``*-cpu-kernel`` row (scope ``cpu_bench``) with a machine-
    readable ``kernel_name`` (e.g. ``production-consolidated-v9``) beside the
    existing ``binary_version`` / ``kernel_commit``.  The current era is the
    newest such row whose window includes ``now``.

    This is the enforcement link that closes the 2026-08-23 gap (ledger
    ``active_kernel_era`` drifted stale because nothing compared it to the era
    authority): the prompt-effort ledger MUST agree with this value, so a kernel
    promotion that updates the eras file but forgets the ledger now FAILS the
    stack-change pipeline instead of silently diverging.

    Returns ``(era_id, kernel_name)``.  Raises ``ValueError`` when the eras file
    is unreadable, no cpu-kernel era is active, or the current era lacks a
    ``kernel_name`` — every one of those is a promotion-side defect that must
    fail closed, never pass silently.
    """
    try:
        doc = _load_mapping(eras_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"instrument eras registry invalid: {exc}") from exc
    rows = doc.get("eras")
    if not isinstance(rows, list):
        raise ValueError("instrument eras registry missing 'eras' list")

    now_epoch = _parse_epoch(now) if now is not None else None
    if now_epoch is None:
        now_epoch = datetime.now(timezone.utc).timestamp()

    current: dict[str, Any] | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        era_id = row.get("id")
        if not isinstance(era_id, str) or not era_id.endswith("-cpu-kernel"):
            continue
        if row.get("scope") != "cpu_bench":
            continue
        start = _parse_epoch(row.get("from"))
        end = _parse_epoch(row.get("until"))
        if start is not None and now_epoch < start:
            continue
        if end is not None and now_epoch >= end:
            continue
        if current is None or start > _parse_epoch(current.get("from")):
            current = row
    if current is None:
        raise ValueError("no active cpu-kernel era found in instrument_eras.yaml")

    era_id = str(current.get("id") or "")
    kernel_name = current.get("kernel_name")
    if not isinstance(kernel_name, str) or not kernel_name.strip():
        raise ValueError(
            f"current cpu-kernel era {era_id!r} lacks machine-readable kernel_name — "
            "add it at the same boundary as the kernel promotion"
        )
    return era_id, kernel_name.strip()


def validate_reasoning_effort_certifications(
    registry_path: Path = DEFAULT_REGISTRY,
    certifications_path: Path = DEFAULT_CERTIFICATIONS,
    eras_path: Path = DEFAULT_ERAS,
) -> EffortCertificationResult:
    """Validate declared prompt-effort defaults against their certified identity.

    The ledger's ``active_kernel_era`` is intentionally a distinct deployment
    stamp.  Updating it during a kernel promotion immediately invalidates every
    old certificate until its curve is remeasured and restamped.

    The ledger's era must ALSO agree with the current cpu-kernel era in
    ``orchestration/instrument_eras.yaml`` (``_current_cpu_kernel_era``) — the
    promotion record, updated at every kernel cutover.  This cross-check is what
    makes the era roll-forward enforceable rather than a comment: a promotion
    that forgets to roll the ledger now fails the stack-change pipeline with a
    named error.  The eras file being unreadable, having no active cpu-kernel
    era, or carrying no ``kernel_name`` on the current one are all fail-closed
    errors too.

    The chat template is a calibration-voiding axis too: a certificate must also
    name ``template_sha``, compared against the template the role is bound to.
    The bound sha is read from the role's own ``chat_template`` block when it
    yields a usable ``served_template_sha256_12``, falling back to
    ``server_mode.<role>.chat_template`` (the location the pilot roles carry
    today).  A role that binds no template has a vacuous template check.  Shas
    are compared in normalized 12-char form via ``_normalize_template_sha``.
    """
    try:
        registry = _load_mapping(registry_path)
        ledger = _load_mapping(certifications_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return EffortCertificationResult(errors=[f"effort certification input invalid: {exc}"])
    except Exception as exc:  # noqa: BLE001
        return EffortCertificationResult(errors=[f"effort certification input invalid: {exc}"])

    errors: list[str] = []
    active_kernel_era = ledger.get("active_kernel_era")
    if not isinstance(active_kernel_era, str) or not active_kernel_era.strip():
        errors.append("effort certification ledger missing non-empty active_kernel_era")
        active_kernel_era = None

    try:
        current_era_id, current_kernel_name = _current_cpu_kernel_era(eras_path)
    except ValueError as exc:
        errors.append(str(exc))
        current_kernel_name = None
    else:
        if active_kernel_era is not None and active_kernel_era != current_kernel_name:
            errors.append(
                f"effort certification ledger active_kernel_era {active_kernel_era!r} does not "
                f"match the current production cpu-kernel era {current_era_id!r} "
                f"(kernel_name {current_kernel_name!r}) — roll the ledger at the same boundary "
                "as the kernel promotion"
            )
        elif active_kernel_era is None:
            errors.append(
                f"effort certification ledger must name the current production cpu-kernel era "
                f"{current_kernel_name!r} ({current_era_id})"
            )

    certificates = ledger.get("role_certifications", {})
    if not isinstance(certificates, dict):
        errors.append("effort certification ledger role_certifications must be a mapping")
        certificates = {}

    roles = registry.get("roles", {})
    if not isinstance(roles, dict):
        return EffortCertificationResult(errors=[*errors, "registry roles must be a mapping"])
    server_mode = registry.get("server_mode", {})
    if not isinstance(server_mode, dict):
        server_mode = {}

    for role_name, role in sorted(roles.items()):
        if not isinstance(role_name, str) or not isinstance(role, dict):
            continue
        effort_level = _configured_effort_level(role)
        certificate = certificates.get(role_name)
        if effort_level is None:
            continue
        if not isinstance(certificate, dict):
            errors.append(
                f"role {role_name!r} declares reasoning_effort.level={effort_level!r} "
                "but has no role_certifications entry"
            )
            continue

        certificate_level = certificate.get("level")
        if certificate_level != effort_level:
            errors.append(
                f"role {role_name!r} effort level {effort_level!r} does not match "
                f"certified level {certificate_level!r}"
            )

        curve_artifact = certificate.get("curve_artifact")
        if not isinstance(curve_artifact, str) or not curve_artifact.strip():
            errors.append(f"role {role_name!r} certificate missing non-empty curve_artifact")

        certified_against = certificate.get("certified_against")
        if not isinstance(certified_against, dict):
            errors.append(f"role {role_name!r} certificate missing certified_against mapping")
            continue
        missing = [
            field for field in REQUIRED_CERTIFICATION_FIELDS if not certified_against.get(field)
        ]
        if missing:
            errors.append(
                f"role {role_name!r} certificate missing certified_against field(s): "
                + ", ".join(missing)
            )
            continue

        bound_name, bound_quant = _bound_model(role)
        if bound_name is None or bound_quant is None:
            errors.append(
                f"role {role_name!r} declares effort but has no bound model name and quant"
            )
            continue
        if certified_against["model"] != bound_name:
            errors.append(
                f"role {role_name!r} bound model {bound_name!r} differs from certified "
                f"model {certified_against['model']!r}"
            )
        if certified_against["quant"] != bound_quant:
            errors.append(
                f"role {role_name!r} bound quant {bound_quant!r} differs from certified "
                f"quant {certified_against['quant']!r}"
            )
        if active_kernel_era is not None and certified_against["kernel_era"] != active_kernel_era:
            errors.append(
                f"role {role_name!r} certified kernel era {certified_against['kernel_era']!r} "
                f"differs from active kernel era {active_kernel_era!r}"
            )
        bound_template_sha = _bound_template_sha(role, role_name, server_mode)
        if bound_template_sha is not None:
            if _normalize_template_sha(certified_against["template_sha"]) != bound_template_sha:
                errors.append(
                    f"role {role_name!r} certified template sha "
                    f"{certified_against['template_sha']!r} differs from bound "
                    f"template sha {bound_template_sha!r}"
                )

    return EffortCertificationResult(errors=errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--certifications", type=Path, default=DEFAULT_CERTIFICATIONS)
    parser.add_argument("--eras", type=Path, default=DEFAULT_ERAS)
    args = parser.parse_args(argv)
    result = validate_reasoning_effort_certifications(
        args.registry, args.certifications, args.eras
    )
    for error in result.errors:
        print(f"error: {error}")
    if result.ok:
        print("reasoning-effort certifications: ok")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
