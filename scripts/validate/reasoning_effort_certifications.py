#!/usr/bin/env python3
"""Fail closed when a role's prompt-effort certification no longer matches its model.

The prompt-effort ladder is independent of native ``reasoning_budget`` / ``<think>``
settings.  A role becomes subject to this validator only when it declares
``reasoning_effort.level`` in the model registry.  Its certificate must then name
the same model, quant, and active kernel era as the currently bound role.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
DEFAULT_CERTIFICATIONS = REPO_ROOT / "orchestration" / "reasoning_effort_certifications.yaml"
REQUIRED_CERTIFICATION_FIELDS = ("model", "quant", "kernel_era")


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


def validate_reasoning_effort_certifications(
    registry_path: Path = DEFAULT_REGISTRY,
    certifications_path: Path = DEFAULT_CERTIFICATIONS,
) -> EffortCertificationResult:
    """Validate declared prompt-effort defaults against their certified identity.

    The ledger's ``active_kernel_era`` is intentionally a distinct deployment
    stamp.  Updating it during a kernel promotion immediately invalidates every
    old certificate until its curve is remeasured and restamped.
    """
    try:
        registry = _load_mapping(registry_path)
        ledger = _load_mapping(certifications_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return EffortCertificationResult(errors=[f"effort certification input invalid: {exc}"])

    errors: list[str] = []
    active_kernel_era = ledger.get("active_kernel_era")
    if not isinstance(active_kernel_era, str) or not active_kernel_era.strip():
        errors.append("effort certification ledger missing non-empty active_kernel_era")
        active_kernel_era = None

    certificates = ledger.get("role_certifications", {})
    if not isinstance(certificates, dict):
        errors.append("effort certification ledger role_certifications must be a mapping")
        certificates = {}

    roles = registry.get("roles", {})
    if not isinstance(roles, dict):
        return EffortCertificationResult(errors=[*errors, "registry roles must be a mapping"])

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
            errors.append(
                f"role {role_name!r} certificate missing non-empty curve_artifact"
            )

        certified_against = certificate.get("certified_against")
        if not isinstance(certified_against, dict):
            errors.append(f"role {role_name!r} certificate missing certified_against mapping")
            continue
        missing = [field for field in REQUIRED_CERTIFICATION_FIELDS if not certified_against.get(field)]
        if missing:
            errors.append(
                f"role {role_name!r} certificate missing certified_against field(s): "
                + ", ".join(missing)
            )
            continue

        bound_name, bound_quant = _bound_model(role)
        if bound_name is None or bound_quant is None:
            errors.append(f"role {role_name!r} declares effort but has no bound model name and quant")
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

    return EffortCertificationResult(errors=errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--certifications", type=Path, default=DEFAULT_CERTIFICATIONS)
    args = parser.parse_args(argv)
    result = validate_reasoning_effort_certifications(args.registry, args.certifications)
    for error in result.errors:
        print(f"error: {error}")
    if result.ok:
        print("reasoning-effort certifications: ok")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
