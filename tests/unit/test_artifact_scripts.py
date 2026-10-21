"""register_gguf_fingerprints / backfill_template_pins CLIs and the stack-change `artifact_integrity` step."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from src.registry import artifact_integrity as ai
from tests.unit.test_artifact_integrity import TEMPLATE, make_gguf, sha

_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reg_cli = _load("scripts/validate/register_gguf_fingerprints.py", "reg_cli")
tpl_cli = _load("scripts/validate/backfill_template_pins.py", "tpl_cli")


def _registry(tmp_path, gguf, pin):
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump({"roles": {"frontdoor": {"model": {"path": str(gguf), **({"sha256": pin} if pin else {})}}}}))
    return p


def test_register_cli_hashes_pinned_file_and_check_reports_ok(tmp_path, monkeypatch, capsys):
    g = make_gguf(tmp_path / "m.gguf")
    monkeypatch.setenv(ai.LEDGER_ENV, str(tmp_path / "ledger.json"))
    r = _registry(tmp_path, g, sha(g))
    assert reg_cli.main(["register", "--registry", str(r)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"][0]["status"] == "registered"
    assert reg_cli.main(["check", "--registry", str(r)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"][0]["status"] == "ok"


def test_register_cli_refuses_a_digest_that_differs_from_the_pin(tmp_path, monkeypatch, capsys):
    g = make_gguf(tmp_path / "m.gguf")
    monkeypatch.setenv(ai.LEDGER_ENV, str(tmp_path / "ledger.json"))
    r = _registry(tmp_path, g, "0" * 64)
    assert reg_cli.main(["register", "--registry", str(r)]) == 1
    assert "REFUSED" in json.loads(capsys.readouterr().out)["rows"][0]["status"]
    assert ai.load_ledger(tmp_path / "ledger.json") == {}


def test_register_cli_skips_unpinned_unless_adopting(tmp_path, monkeypatch, capsys):
    g = make_gguf(tmp_path / "m.gguf")
    monkeypatch.setenv(ai.LEDGER_ENV, str(tmp_path / "ledger.json"))
    r = _registry(tmp_path, g, None)
    assert reg_cli.main(["register", "--registry", str(r)]) == 0
    assert "skipped_unpinned" in json.loads(capsys.readouterr().out)["rows"][0]["status"]
    assert reg_cli.main(["register", "--registry", str(r), "--adopt-unpinned"]) == 0
    row = json.loads(capsys.readouterr().out)["rows"][0]
    assert row["status"] == "adopted_proposal" and row["sha256"] == sha(g)


def test_template_backfill_plan_file_vs_embedded_and_conflict(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    t = tmp_path / "t.jinja"
    t.write_bytes(TEMPLATE)
    digest = ai.template_sha256(TEMPLATE)
    reg = {"server_mode": {
        "with_file": {"model": str(g), "chat_template_file": str(t), "chat_template": {"served_template_sha256_12": digest[:12]}},
        "embedded": {"model": str(g)},
        "stale12": {"model": str(g), "chat_template_file": str(t), "chat_template": {"served_template_sha256_12": "deadbeefcafe"}},
        "alias": {"model": str(g), "alias_of": "embedded"},
        "aux": {"model": "whisper-large"},
    }}
    rows = {r["role"]: r for r in tpl_cli.plan(reg)}
    assert set(rows) == {"with_file", "embedded", "stale12"}
    assert rows["with_file"]["status"] == "pin" and rows["with_file"]["source"].startswith("file:")
    assert rows["embedded"]["status"] == "pin" and rows["embedded"]["source"].startswith("gguf:")
    assert rows["embedded"]["served_template_sha256"] == digest
    assert rows["stale12"]["status"] == "CONFLICT_existing_12hex_differs"


def test_pipeline_artifact_integrity_step_errors_and_warnings(tmp_path, monkeypatch):
    from scripts.registry import stack_change_pipeline as pipeline

    monkeypatch.delenv("REGISTRY_ARTIFACT_PINS_STRICT", raising=False)
    monkeypatch.delenv("REGISTRY_TEMPLATE_PINS_STRICT", raising=False)
    lean = tmp_path / "lean.yaml"
    reg = {
        "roles": {"frontdoor": {"model": {"path": "/m/x.gguf"}, "tool_permissions": {"allowed_categories": ["file"]}}},
        "server_mode": {"frontdoor": {"model": "/m/x.gguf"}},
    }
    lean.write_text(yaml.safe_dump(reg))
    cfg = pipeline.StackChangePipelineConfig(mode="check", lean_registry=lean, repo_root=tmp_path)
    step = pipeline._artifact_integrity_step(cfg)
    assert step.name == "artifact_integrity" and step.ok and step.warnings        # missing pins warn only
    monkeypatch.setenv("REGISTRY_ARTIFACT_PINS_STRICT", "1")
    step = pipeline._artifact_integrity_step(cfg)
    assert not step.ok and any("artifact pin missing" in e or "uploader_class" in e for e in step.errors)
    reg["roles"]["frontdoor"]["model"]["sha256"] = "zz"
    lean.write_text(yaml.safe_dump(reg))
    monkeypatch.delenv("REGISTRY_ARTIFACT_PINS_STRICT")
    step = pipeline._artifact_integrity_step(cfg)
    assert not step.ok and any("sha256" in e for e in step.errors)               # malformed is always an error
