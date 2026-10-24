"""Tests for additive diagnostics in src.config.validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import get_config, reset_config
from src.config.validation import (
    _load_registry_runtime_defaults,
    _load_registry_timeouts,
    _registry_timeout,
    get_validation_diagnostics,
    reset_validation_caches,
)


@pytest.fixture(autouse=True)
def _fresh_registry_caches():
    # Autouse, so it tears down after monkeypatch has restored the env: no test
    # leaves a tmp registry's values in the process-wide caches.
    yield
    reset_validation_caches()
    reset_config()


def test_runtime_defaults_records_missing_registry(monkeypatch, tmp_path: Path):
    reset_validation_caches()
    monkeypatch.setenv(
        "ORCHESTRATOR_PATHS_REGISTRY_PATH",
        str(tmp_path / "missing-model-registry.yaml"),
    )

    assert _load_registry_runtime_defaults() == {}

    diag = get_validation_diagnostics()["runtime_defaults"]
    assert diag["status"] == "missing"
    assert diag["failure_reason"] == "missing_registry"


def test_missing_registry_is_logged_as_a_warning(monkeypatch, tmp_path: Path, caplog):
    reset_validation_caches()
    missing = tmp_path / "missing-model-registry.yaml"
    monkeypatch.setenv("ORCHESTRATOR_PATHS_REGISTRY_PATH", str(missing))

    with caplog.at_level("WARNING", logger="src.config.validation"):
        _load_registry_runtime_defaults()

    assert str(missing) in caplog.text


def test_default_registry_path_does_not_depend_on_cwd(monkeypatch, tmp_path: Path):
    """A3-06: the default was CWD-relative, so any other cwd lost every registry timeout."""
    monkeypatch.delenv("ORCHESTRATOR_PATHS_REGISTRY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    reset_validation_caches()

    assert _registry_timeout("roles", "frontdoor", -1) != -1
    assert get_validation_diagnostics()["runtime_defaults"]["status"] == "loaded"


def test_timeouts_records_yaml_parse_fallback(monkeypatch, tmp_path: Path):
    reset_validation_caches()
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text("runtime_defaults:\n  timeouts: [broken\n", encoding="utf-8")
    monkeypatch.setenv("ORCHESTRATOR_PATHS_REGISTRY_PATH", str(registry_path))

    assert _load_registry_timeouts() == {"default": 600}

    diag = get_validation_diagnostics()["timeouts"]
    assert diag["status"] == "fallback"
    assert diag["failure_reason"]
    assert diag["entries"] == 1


def test_reset_config_rereads_registry_defaults(monkeypatch, tmp_path: Path):
    """A3-07: Settings class bodies froze registry values at import."""
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "runtime_defaults": {
                    "timeouts": {"roles": {"frontdoor": 4242, "architect_general": 4343}},
                    "llm": {"depth_override_max_depth": 9},
                }
            }
        ),
        encoding="utf-8",
    )
    get_config()
    monkeypatch.setenv("ORCHESTRATOR_PATHS_REGISTRY_PATH", str(registry_path))
    reset_config()

    cfg = get_config()
    assert cfg.timeouts.frontdoor == 4242
    assert cfg.timeouts.architect_general == 4343
    assert cfg.llm.depth_override_max_depth == 9


def test_concurrent_first_access_does_not_cache_a_fallback(monkeypatch, tmp_path: Path):
    """A3-05: a second thread saw the loading flag, got {} and cached {'default': 600}."""
    import threading
    import time

    from src.config import validation

    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"runtime_defaults": {"timeouts": {"roles": {"frontdoor": 4242}}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORCHESTRATOR_PATHS_REGISTRY_PATH", str(registry_path))
    monkeypatch.setattr(validation, "_PARSED_RUNTIME_DEFAULTS", None)
    reset_validation_caches()
    parsing = threading.Event()
    real_load = yaml.load

    def slow_load(*args, **kwargs):
        parsing.set()
        time.sleep(0.3)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(yaml, "load", slow_load)
    other = threading.Thread(target=validation._registry_runtime_value, args=(("timeouts",), None))
    other.start()
    assert parsing.wait(5)
    assert _registry_timeout("roles", "frontdoor", -1) == 4242
    other.join(5)
    assert _registry_timeout("roles", "frontdoor", -1) == 4242
