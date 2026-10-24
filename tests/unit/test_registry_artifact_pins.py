"""SEC-ART-1: hf_repo / hf_revision / sha256 pins on roles.<role>.model (strict by default since SEC-ART-1.1)."""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import pytest
import yaml

from src.registry import registry_validator as rv

LIVE = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
REV = "a" * 40
SHA = "b" * 64


@pytest.fixture()
def live() -> dict:
    reg = yaml.safe_load(LIVE.read_text())
    assert any(
        isinstance(c, dict) and isinstance(c.get("model"), dict)
        and str(c["model"].get("path", "")).endswith(".gguf")
        for c in reg["roles"].values()
    ), "no GGUF lineup role in live registry: pin checks would pass vacuously"
    return reg


@pytest.fixture()
def unpinned(live) -> dict:
    """The live lineup with every pin stripped: the missing-pin paths need a registry that HAS gaps."""
    reg = copy.deepcopy(live)
    for cfg in reg["roles"].values():
        if isinstance(cfg, dict) and isinstance(cfg.get("model"), dict):
            for field in (*rv.ARTIFACT_PIN_FIELDS, "shard_sha256"):
                cfg["model"].pop(field, None)
    return reg


def test_live_lineup_is_fully_pinned(live):
    """SEC-ART-1.1 acceptance: the compiled lineup carries every pin, so the strict default validates clean."""
    assert rv.check_artifact_pins(live) == ([], [])


def _write(tmp_path: Path, reg: dict) -> Path:
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(reg, sort_keys=False))
    return p


def _first_gguf_role(reg: dict) -> str:
    return next(r for r, c in reg["roles"].items()
                if isinstance(c, dict) and isinstance(c.get("model"), dict)
                and str(c["model"].get("path", "")).endswith(".gguf"))


def test_missing_pins_are_listed_not_errors(unpinned):
    malformed, missing = rv.check_artifact_pins(unpinned)
    assert malformed == []
    assert missing and all("lacks hf_repo, hf_revision, sha256" in m for m in missing)


def test_explicit_opt_out_is_warn_only(unpinned, tmp_path, caplog, monkeypatch):
    monkeypatch.setenv(rv.ARTIFACT_PINS_STRICT_ENV, "0")
    with caplog.at_level(logging.WARNING, logger=rv.__name__):
        errors = rv.validate_all(_write(tmp_path, unpinned))
    assert not [e for e in errors if "artifact pin" in e]
    assert any("SEC-ART-1" in r.message for r in caplog.records)


def test_strict_switch_via_argument_and_env(unpinned, tmp_path, monkeypatch):
    p = _write(tmp_path, unpinned)
    monkeypatch.delenv(rv.ARTIFACT_PINS_STRICT_ENV, raising=False)
    assert [e for e in rv.validate_all(p) if "artifact pin (strict)" in e]          # the default
    assert [e for e in rv.validate_all(p, strict_artifact_pins=True) if "artifact pin (strict)" in e]
    monkeypatch.setenv(rv.ARTIFACT_PINS_STRICT_ENV, "1")
    assert [e for e in rv.validate_all(p) if "artifact pin (strict)" in e]
    assert not [e for e in rv.validate_all(p, strict_artifact_pins=False) if "artifact pin" in e]


def test_fully_pinned_role_is_clean_and_other_roles_still_missing(unpinned):
    reg = copy.deepcopy(unpinned)
    role = _first_gguf_role(reg)
    reg["roles"][role]["model"].update(hf_repo="org/repo", hf_revision=REV, sha256=SHA)
    malformed, missing = rv.check_artifact_pins(reg)
    assert malformed == []
    assert missing and not any(f"roles.{role}.model" in m for m in missing)


@pytest.mark.parametrize("field,bad", [
    ("hf_repo", "no-slash"), ("hf_revision", "main"), ("hf_revision", "A" * 40),
    ("sha256", "b" * 12), ("sha256", "g" * 64), ("sha256", 123),
])
def test_malformed_pin_is_always_an_error_even_when_not_strict(live, tmp_path, monkeypatch, field, bad):
    monkeypatch.delenv(rv.ARTIFACT_PINS_STRICT_ENV, raising=False)
    reg = copy.deepcopy(live)
    role = _first_gguf_role(reg)
    reg["roles"][role]["model"].update(hf_repo="org/repo", hf_revision=REV, sha256=SHA)
    reg["roles"][role]["model"][field] = bad
    errors = rv.validate_all(_write(tmp_path, reg), strict_artifact_pins=False)
    assert any(f"roles.{role}.model.{field}" in e for e in errors), errors


def test_null_counts_as_absent_and_non_gguf_roles_are_skipped():
    reg = {"roles": {
        "a": {"model": {"path": "/m/a.gguf", "hf_repo": None}},
        "remote": {"model": {"path": "https://x/y"}},
        "nomodel": {},
    }}
    malformed, missing = rv.check_artifact_pins(reg)
    assert malformed == [] and len(missing) == 1 and "roles.a.model" in missing[0]


def test_validator_is_strict_by_default_and_ignores_the_launcher_ledger_flag(tmp_path, monkeypatch):
    """SEC-ART-1.1: with no env override a lineup role lacking pins is an ERROR, even when the registry's
    launcher switch runtime_defaults.artifact_integrity.pins_strict is false; =0 is the explicit opt-out."""
    reg = {
        "runtime_defaults": {"artifact_integrity": {"pins_strict": False}},
        "roles": {"drafter": {"model": {"path": "/m/d.gguf"}}},
    }
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(reg))
    monkeypatch.delenv("REGISTRY_ARTIFACT_PINS_STRICT", raising=False)
    assert rv.validator_artifact_pins_strict() is True
    assert [e for e in rv.validate_all(p) if "artifact pin (strict)" in e and "drafter" in e]
    for off in ("0", "false", "off"):
        monkeypatch.setenv("REGISTRY_ARTIFACT_PINS_STRICT", off)
        assert not [e for e in rv.validate_all(p) if "artifact pin" in e]
    monkeypatch.setenv("REGISTRY_ARTIFACT_PINS_STRICT", "1")
    assert [e for e in rv.validate_all(p) if "artifact pin (strict)" in e]
