from __future__ import annotations

from pathlib import Path

import yaml

from scripts.validate.reasoning_effort_certifications import (
    validate_reasoning_effort_certifications,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _registry(*, model: str = "Qwen-Test", quant: str = "Q8_0", level: str | None = "L2") -> dict:
    role: dict = {"model": {"name": model, "quant": quant}}
    if level is not None:
        role["reasoning_effort"] = {"level": level}
    return {"roles": {"frontdoor": role}}


def _ledger(*, model: str = "Qwen-Test", quant: str = "Q8_0", era: str = "v8", level: str = "L2") -> dict:
    return {
        "schema_version": 1,
        "active_kernel_era": era,
        "role_certifications": {
            "frontdoor": {
                "level": level,
                "curve_artifact": "artifacts/reasoning-effort/frontdoor.json",
                "certified_against": {"model": model, "quant": quant, "kernel_era": era},
            }
        },
    }


def test_accepts_matching_role_model_quant_and_kernel_era(tmp_path: Path) -> None:
    result = validate_reasoning_effort_certifications(
        _write(tmp_path / "registry.yaml", _registry()),
        _write(tmp_path / "certifications.yaml", _ledger()),
    )
    assert result.ok


def test_rejects_model_swap(tmp_path: Path) -> None:
    result = validate_reasoning_effort_certifications(
        _write(tmp_path / "registry.yaml", _registry(model="Replacement")),
        _write(tmp_path / "certifications.yaml", _ledger()),
    )
    assert any("bound model" in error for error in result.errors)


def test_rejects_quant_change(tmp_path: Path) -> None:
    result = validate_reasoning_effort_certifications(
        _write(tmp_path / "registry.yaml", _registry(quant="IQ2_XS")),
        _write(tmp_path / "certifications.yaml", _ledger()),
    )
    assert any("bound quant" in error for error in result.errors)


def test_rejects_kernel_promotion_until_curve_is_recertified(tmp_path: Path) -> None:
    ledger = _ledger()
    ledger["active_kernel_era"] = "v9"
    result = validate_reasoning_effort_certifications(
        _write(tmp_path / "registry.yaml", _registry()),
        _write(tmp_path / "certifications.yaml", ledger),
    )
    assert any("kernel era" in error for error in result.errors)


def test_rejects_declared_effort_without_certificate(tmp_path: Path) -> None:
    result = validate_reasoning_effort_certifications(
        _write(tmp_path / "registry.yaml", _registry()),
        _write(
            tmp_path / "certifications.yaml",
            {"schema_version": 1, "active_kernel_era": "v8", "role_certifications": {}},
        ),
    )
    assert any("has no role_certifications entry" in error for error in result.errors)


def test_rejects_certificate_without_a_curve_record(tmp_path: Path) -> None:
    ledger = _ledger()
    del ledger["role_certifications"]["frontdoor"]["curve_artifact"]
    result = validate_reasoning_effort_certifications(
        _write(tmp_path / "registry.yaml", _registry()),
        _write(tmp_path / "certifications.yaml", ledger),
    )
    assert any("curve_artifact" in error for error in result.errors)


def test_ignores_roles_without_a_prompt_effort_default(tmp_path: Path) -> None:
    result = validate_reasoning_effort_certifications(
        _write(tmp_path / "registry.yaml", _registry(level=None)),
        _write(
            tmp_path / "certifications.yaml",
            {"schema_version": 1, "active_kernel_era": "v8", "role_certifications": {}},
        ),
    )
    assert result.ok
