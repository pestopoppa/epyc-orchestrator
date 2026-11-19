"""M-27d.1: the episodic source label is a registered, runtime-toggleable, autopilot-surface flag."""
from __future__ import annotations

import asyncio
import pathlib
from types import SimpleNamespace

import pytest

from src import flag_env as flag_env_module
from src.api.routes.config import update_config
from src.features import _REGISTRY_BY_NAME, features, reset_features
from src.flag_env import episodic_source_label_on
from src.runtime import config_attestation

NAME = "episodic_source_label"
ENV = "ORCHESTRATOR_EPISODIC_SOURCE_LABEL"


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(tmp_path / "runtime_flags.json"))
    monkeypatch.setattr(config_attestation, "attestation_dir", lambda: tmp_path / "attestations")
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(f"ORCHESTRATOR_FEATURE_{_REGISTRY_BY_NAME[NAME].env_var}", raising=False)
    reset_features()
    flag_env_module.reset_cache()
    yield
    reset_features()
    flag_env_module.reset_cache()


def _post(**flags):
    class Request:
        client = SimpleNamespace(host="127.0.0.1")

        async def json(self):
            return dict(flags)

    flag_env_module.reset_cache()
    return asyncio.run(update_config(Request(), current=features()))


def test_registered_default_off_with_legacy_env_name():
    spec = _REGISTRY_BY_NAME[NAME]
    assert spec.default_prod is False and spec.default_test is False
    assert f"ORCHESTRATOR_{spec.env_var}" == ENV
    assert episodic_source_label_on() is False


def test_env_var_still_enables_it(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert episodic_source_label_on() is True


def test_post_config_toggles_the_real_reader():
    _post(**{NAME: True})
    assert episodic_source_label_on() is True
    _post(**{NAME: False})
    assert episodic_source_label_on() is False


def test_on_autopilot_surface():
    from scripts.autopilot.config_applicator import HOT_SWAP_FEATURES

    assert NAME in HOT_SWAP_FEATURES
    assert f"{NAME}: baseline" in pathlib.Path("orchestration/runtime_flags.spec.yaml").read_text()
    assert f"`{NAME}`" in pathlib.Path("scripts/autopilot/program.md").read_text()
