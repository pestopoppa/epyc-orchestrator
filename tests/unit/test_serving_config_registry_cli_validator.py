"""N6 (final-gate audit): the serving_config_registry CLI must not silently skip validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.registry import serving_config_registry as scr

# The shapes are reused from the module's own fixture file so the two cannot drift.
from tests.unit.test_serving_config_registry import _config, _master


def _files(tmp_path: Path) -> tuple[Path, Path]:
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(_config()))
    master = tmp_path / "master.yaml"
    master.write_text(yaml.safe_dump(_master()))
    return cfg, master


@pytest.mark.parametrize("cmd", ["diff", "dryrun", "section"])
def test_cli_without_validator_refuses(cmd, tmp_path, capsys):
    cfg, master = _files(tmp_path)
    out = tmp_path / "out.json"
    rc = scr.main([cmd, "--config", str(cfg), "--master", str(master), "--out", str(out)])
    err = capsys.readouterr().err
    assert rc == 2 and "--validator" in err and "--no-validate" in err
    assert not out.exists()          # nothing was produced from an unvalidated artifact


@pytest.mark.parametrize("cmd", ["diff", "section"])
def test_cli_no_validate_is_explicit_and_warns(cmd, tmp_path, capsys):
    cfg, master = _files(tmp_path)
    out = tmp_path / "out.json"
    extra = ["--no-dry-run"] if cmd == "section" else []
    rc = scr.main([cmd, "--config", str(cfg), "--master", str(master), "--out", str(out), "--no-validate", *extra])
    assert rc == 0 and out.is_file()
    assert "NOT validated" in capsys.readouterr().err


def test_cli_validator_and_no_validate_conflict(tmp_path, capsys):
    cfg, master = _files(tmp_path)
    rc = scr.main(["diff", "--config", str(cfg), "--master", str(master), "--validator", str(cfg), "--no-validate"])
    assert rc == 2 and "mutually exclusive" in capsys.readouterr().err


def test_cli_with_a_validator_still_validates(tmp_path, capsys):
    cfg, master = _files(tmp_path)
    bad = tmp_path / "validator.py"
    bad.write_text("class R:\n    errors = ['forced validator error']\n\ndef validate(cfg, strict, registry):\n    return R()\n")
    rc = scr.main(["diff", "--config", str(cfg), "--master", str(master), "--validator", str(bad)])
    assert rc == 2 and "forced validator error" in capsys.readouterr().err
