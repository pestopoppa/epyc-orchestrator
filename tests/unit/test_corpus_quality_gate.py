"""Tests for corpus_quality_gate stack-derived model selection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
_SPEC = importlib.util.spec_from_file_location(
    "corpus_quality_gate_test",
    _BENCH / "corpus_quality_gate.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["corpus_quality_gate_test"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_load_live_models_reads_stack_prior_ports(tmp_path: Path) -> None:
    stack_priors = tmp_path / "stack_priors.yaml"
    stack_priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
    display_name: Frontdoor Display
    serving:
      endpoint: http://localhost:9100
      ports: [9100]
  worker_general:
    deployment_status: live_stack
    serving:
      ports: [9200]
  candidate:
    deployment_status: benchmark_or_candidate
    serving:
      endpoint: http://localhost:9900
""",
        encoding="utf-8",
    )

    models = _MOD._load_live_models(stack_priors)

    assert models["frontdoor"] == {
        "port": 9100,
        "name": "Frontdoor Display",
        "role": "frontdoor",
    }
    assert models["worker_general"]["port"] == 9200
    assert "candidate" not in models


def test_load_live_models_falls_back_to_ports_when_endpoint_port_is_invalid(
    tmp_path: Path,
) -> None:
    stack_priors = tmp_path / "stack_priors.yaml"
    stack_priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:notaport
      ports: [9100]
  worker_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:notaport
""",
        encoding="utf-8",
    )

    models = _MOD._load_live_models(stack_priors)

    assert models["frontdoor"]["port"] == 9100
    assert "worker_general" not in models


def test_fallback_models_derive_ports_from_manifest(monkeypatch) -> None:
    monkeypatch.setattr(
        _MOD,
        "PORT_MAP",
        {"frontdoor": 9001, "worker_general": 9002, "architect_general": 9003},
    )
    monkeypatch.setattr(
        _MOD,
        "HOT_ROLES",
        {"frontdoor", "worker_general", "architect_general"},
    )

    models = _MOD._fallback_models()

    assert models == {
        "frontdoor": {
            "port": 9001,
            "name": "frontdoor (manifest fallback)",
            "role": "frontdoor",
        },
        "worker_general": {
            "port": 9002,
            "name": "worker_general (manifest fallback)",
            "role": "worker_general",
        },
        "architect_general": {
            "port": 9003,
            "name": "architect_general (manifest fallback)",
            "role": "architect_general",
        },
    }


def test_preferred_fallback_model_roles_filters_missing_manifest_entries(monkeypatch) -> None:
    monkeypatch.setattr(
        _MOD,
        "PORT_MAP",
        {"frontdoor": 9001, "worker_general": 9002},
    )
    monkeypatch.setattr(
        _MOD,
        "HOT_ROLES",
        {"frontdoor", "worker_general"},
    )

    assert _MOD._preferred_fallback_model_roles() == ("frontdoor", "worker_general")


def test_default_model_keys_are_valid_loaded_roles() -> None:
    models = {
        "frontdoor": {"port": 8070},
        "worker_general": {"port": 8072},
        "custom_role": {"port": 9000},
    }

    defaults = _MOD._default_model_keys(models)

    assert defaults == ["frontdoor", "worker_general"]
    assert set(defaults) <= set(models)


def test_default_model_keys_falls_back_to_available_models() -> None:
    models = {"custom_role": {"port": 9000}}

    assert _MOD._default_model_keys(models) == ["custom_role"]
