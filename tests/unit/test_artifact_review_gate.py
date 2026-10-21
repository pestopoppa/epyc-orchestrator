"""SEC-ART-3 uploader_class + review-record gate, and SEC-ART-4 template-pin registry checks."""
from __future__ import annotations

import logging

import pytest
import yaml

from src.registry import artifact_review as ar
from src.registry import registry_validator as rv

SHA = "c" * 64


def _reg(klass=None, review=None, tools=True, generation=False):
    model = {"path": "/m/x.gguf"}
    if klass:
        model["uploader_class"] = klass
    if review:
        model["artifact_review"] = review
    role = {"model": model}
    if tools:
        role["tool_permissions"] = {"web_access": False, "allowed_categories": ["file"]}
    reg = {"roles": {"frontdoor": role}}
    if generation:
        reg["server_mode"] = {"frontdoor": {"chat_template_file": "/t.jinja", "chat_template_kwargs": {"enable_thinking": False}}}
    return reg


def _record(tmp_path, **over):
    rec = {
        "reviewed_by": "operator", "reviewed_on": "2026-10-21", "provenance": "HF tree compared, no sidecar files",
        "base_model_diff": "none_no_base_model",
        "flag_check": {"--chat-template-file": {"label": "terse template", "function": "renders terse", "verdict": "consistent"},
                       "chat_template_kwargs": {"label": "thinking off", "function": "disables thinking", "verdict": "consistent"}},
    }
    rec.update(over)
    p = tmp_path / "review.yaml"
    p.write_text(yaml.safe_dump(rec))
    return {"record": str(p)}


def test_community_tool_role_without_review_is_an_error(tmp_path):
    errs, _ = ar.check_uploader_review(_reg("community"), repo_root=tmp_path)
    assert errs and "artifact_review.record" in errs[0]


def test_community_tool_role_with_complete_review_passes(tmp_path):
    errs, warns = ar.check_uploader_review(_reg("community", _record(tmp_path), generation=True), repo_root=tmp_path)
    assert errs == [] and warns == []


@pytest.mark.parametrize("missing", ["reviewed_by", "reviewed_on", "provenance", "base_model_diff", "flag_check"])
def test_incomplete_record_names_the_missing_part(tmp_path, missing):
    ref = _record(tmp_path, **{missing: None})
    errs, _ = ar.check_uploader_review(_reg("community", ref), repo_root=tmp_path)
    assert any(missing in e for e in errs), errs


def test_review_must_label_check_every_declared_generation_flag(tmp_path):
    rec = _record(tmp_path, flag_check={"--chat-template-file": {"label": "a", "function": "b", "verdict": "consistent"}})
    errs, _ = ar.check_uploader_review(_reg("community", rec, generation=True), repo_root=tmp_path)
    assert any("chat_template_kwargs" in e for e in errs)
    rec = _record(tmp_path, flag_check={
        "--chat-template-file": {"label": "a", "function": "b", "verdict": "MISMATCH"},
        "chat_template_kwargs": {"label": "a", "function": "b", "verdict": "consistent"}})
    errs, _ = ar.check_uploader_review(_reg("community", rec, generation=True), repo_root=tmp_path)
    assert any("--chat-template-file" in e for e in errs)


def test_base_model_diff_path_must_exist(tmp_path):
    errs, _ = ar.check_uploader_review(_reg("community", _record(tmp_path, base_model_diff="nope.diff")), repo_root=tmp_path)
    assert any("does not exist" in e for e in errs)
    (tmp_path / "base.diff").write_text("x")
    errs, _ = ar.check_uploader_review(_reg("community", _record(tmp_path, base_model_diff="base.diff")), repo_root=tmp_path)
    assert errs == []


def test_first_party_and_allowlisted_need_no_review_and_non_tool_roles_are_exempt(tmp_path):
    for klass in ("first_party", "allowlisted_requantizer"):
        assert ar.check_uploader_review(_reg(klass), repo_root=tmp_path) == ([], [])
    assert ar.check_uploader_review(_reg("community", tools=False), repo_root=tmp_path) == ([], [])


def test_unset_class_on_tool_role_warns_then_errors_when_strict(tmp_path):
    errs, warns = ar.check_uploader_review(_reg(None), repo_root=tmp_path)
    assert errs == [] and warns and "uploader_class" in warns[0]
    errs, _ = ar.check_uploader_review(_reg(None), strict=True, repo_root=tmp_path)
    assert errs


def test_unknown_class_is_an_error(tmp_path):
    errs, _ = ar.check_uploader_review(_reg("sketchy"), repo_root=tmp_path)
    assert errs and "uploader_class" in errs[0]


def test_validate_all_runs_the_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("REGISTRY_ARTIFACT_PINS_STRICT", raising=False)
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(_reg("community")))
    errors = rv.validate_all(p, strict_artifact_pins=False)
    assert any(e.startswith("artifact review:") for e in errors)


# ---------------------------------------------------------------- SEC-ART-4 registry checks

def _server(pin=None, short=None, alias=False):
    block = {}
    if pin is not None:
        block["served_template_sha256"] = pin
    if short is not None:
        block["served_template_sha256_12"] = short
    cfg = {"model": "/m/x.gguf", "chat_template": block}
    if alias:
        cfg["alias_of"] = "frontdoor"
    return {"server_mode": {"frontdoor": cfg}}


def test_template_pin_shape_and_prefix_are_always_errors():
    bad, _ = rv.check_template_pins(_server(pin="xyz"))
    assert bad and "64 lowercase hex" in bad[0]
    bad, _ = rv.check_template_pins(_server(pin=SHA, short="aaaaaaaaaaaa"))
    assert bad and "first 12" in bad[0]
    assert rv.check_template_pins(_server(pin=SHA, short=SHA[:12])) == ([], [])


def test_missing_template_pin_listed_and_alias_skipped():
    _, missing = rv.check_template_pins(_server())
    assert missing == ["server_mode.frontdoor.chat_template.served_template_sha256"]
    assert rv.check_template_pins(_server(alias=True)) == ([], [])


def test_template_strict_switch(tmp_path, monkeypatch, caplog):
    reg = _server()
    reg["roles"] = {}
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(reg))
    monkeypatch.delenv("REGISTRY_TEMPLATE_PINS_STRICT", raising=False)
    with caplog.at_level(logging.WARNING, logger=rv.__name__):
        assert not [e for e in rv.validate_all(p) if "template pin" in e]
    assert any("SEC-ART-4" in r.message for r in caplog.records)
    monkeypatch.setenv("REGISTRY_TEMPLATE_PINS_STRICT", "1")
    assert [e for e in rv.validate_all(p) if "template pin (strict)" in e]
    reg["runtime_defaults"] = {"artifact_integrity": {"template_pins_strict": True}}
    p.write_text(yaml.safe_dump(reg))
    monkeypatch.delenv("REGISTRY_TEMPLATE_PINS_STRICT")
    assert [e for e in rv.validate_all(p) if "template pin (strict)" in e]


def test_shard_sha256_shape_is_checked():
    reg = {"roles": {"a": {"model": {"path": "/m/a-00001-of-00002.gguf", "shard_sha256": {"a-00002-of-00002.gguf": "nothex"}}}}}
    malformed, _ = rv.check_artifact_pins(reg)
    assert any("shard_sha256" in m for m in malformed)
