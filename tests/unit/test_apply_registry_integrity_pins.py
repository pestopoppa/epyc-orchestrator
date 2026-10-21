"""Comment-preserving registry pin applier + SEC-ART-5 template provenance row builder."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts/validate" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ap = _load("apply_registry_integrity_pins")
tp = _load("template_provenance")

REG = """\
# header comment kept
runtime_defaults:
  x: 1
server_mode:
  frontdoor:
    # keep me
    model: a.gguf
    chat_template_file: /t.jinja
    chat_template:
      provenance: epyc
      served_template_sha256_12: 1443ea9ab4bb
  architect_general:
    model: /m/g-00001-of-00003.gguf
    model_role: g
    port: 8074
roles:
  # role comment
  frontdoor:
    tier: A
    model:
      name: FD
      path: /m/a.gguf
      quant: Q8_0
  architect_general:
    model:
      name: G
      path: /m/g-00001-of-00003.gguf
  other:
    model:
      path: /m/other.gguf
"""
D64 = "d" * 64
PIN_ROWS = [
    {"role": "frontdoor", "status": "pin", "hf_repo": "o/r", "hf_revision": "a" * 40, "sha256": D64},
    {"role": "architect_general", "status": "pin", "hf_repo": "o/g", "hf_revision": "b" * 40, "sha256": "e" * 64,
     "shard_sha256": {"g-00002-of-00003.gguf": "1" * 64, "g-00003-of-00003.gguf": "2" * 64}},
    {"role": "other", "status": "needs_local_hash"},
]
TPL_ROWS = [
    {"role": "frontdoor", "status": "pin", "served_template_sha256": "f" * 64, "upstream_template_sha256": "9" * 64,
     "template_diff": "data/template_reviews/frontdoor.diff", "upstream_ref": "Q/x", "upstream_revision": "c" * 40},
    {"role": "architect_general", "status": "pin", "served_template_sha256": "8" * 64},
]


def test_applies_pins_preserving_comments_and_everything_else():
    new, notes = ap.apply(REG, PIN_ROWS, TPL_ROWS, {"frontdoor": "community"}, reviewed_note="generated; pending sign-off")
    assert notes == []
    for kept in ("# header comment kept", "    # keep me", "  # role comment"):
        assert kept in new
    doc = yaml.safe_load(new)
    m = doc["roles"]["frontdoor"]["model"]
    assert (m["hf_repo"], m["hf_revision"], m["sha256"], m["uploader_class"]) == ("o/r", "a" * 40, D64, "community")
    assert doc["roles"]["architect_general"]["model"]["shard_sha256"]["g-00003-of-00003.gguf"] == "2" * 64
    assert "sha256" not in doc["roles"]["other"]["model"]                      # non-pin rows untouched
    ct = doc["server_mode"]["frontdoor"]["chat_template"]
    assert ct["served_template_sha256"] == "f" * 64 and ct["served_template_sha256_12"] == "1443ea9ab4bb"
    assert ct["template_diff"].endswith("frontdoor.diff") and ct["template_diff_reviewed"] == "generated; pending sign-off"
    assert doc["server_mode"]["architect_general"]["chat_template"] == {"served_template_sha256": "8" * 64}


def test_apply_is_idempotent_and_never_overwrites():
    once, _ = ap.apply(REG, PIN_ROWS, TPL_ROWS, {})
    twice, notes = ap.apply(once, PIN_ROWS, TPL_ROWS, {})
    assert twice == once and notes == []
    changed = [dict(PIN_ROWS[0], sha256="0" * 64)]
    again, notes = ap.apply(once, changed, [], {})
    assert again == once and any("left alone" in n for n in notes)


def test_unknown_role_and_bad_uploader_class_refuse():
    with pytest.raises(ap.ApplyError):
        ap.apply(REG, [{"role": "ghost", "status": "pin", "hf_repo": "o/r", "hf_revision": "a" * 40, "sha256": D64}], [], {})
    with pytest.raises(ap.ApplyError):
        ap.apply(REG, [], [], {"frontdoor": "trusted-ish"})


def test_inline_model_block_is_refused_not_mangled():
    reg = REG.replace("    model:\n      name: FD\n      path: /m/a.gguf\n      quant: Q8_0\n", "    model: {path: /m/a.gguf}\n")
    with pytest.raises(ap.ApplyError):
        ap.apply(reg, PIN_ROWS[:1], [], {})


# ---------------------------------------------------------------- template_provenance

def test_build_row_writes_diff_and_reports_hashes(tmp_path):
    served = b"line1\nline2 served\n"
    upstream = b"line1\nline2 upstream\n"
    row = tp.build_row("frontdoor", served, "file:/t", upstream,
                       {"upstream_ref": "Q/x", "upstream_revision": "c" * 40}, tmp_path, "data/template_reviews")
    assert row["status"] == "pin" and row["identical"] is False
    assert row["template_diff"] == "data/template_reviews/frontdoor.diff"
    body = (tmp_path / "frontdoor.diff").read_text()
    assert "-line2 upstream" in body and "+line2 served" in body
    assert row["served_template_sha256"] != row["upstream_template_sha256"]


def test_identical_templates_still_record_a_nonempty_diff_file(tmp_path):
    row = tp.build_row("r", b"same", "gguf:x", b"same", {"upstream_ref": "Q/x", "upstream_revision": "c" * 40}, tmp_path, "d")
    assert row["identical"] is True and (tmp_path / "r.diff").read_text().startswith("# identical")


def test_unreachable_upstream_is_not_a_pin(tmp_path):
    row = tp.build_row("r", b"served", "gguf:x", None, {"upstream_error": "boom"}, tmp_path, "d")
    assert row["status"] == "upstream_unavailable" and "template_diff" not in row
    assert tp.build_row("r", None, "gguf:x", b"u", {}, tmp_path, "d")["status"] == "no_served_template"


def test_fetch_upstream_prefers_jinja_then_tokenizer_config():
    def get_factory(files):
        def get(url):
            if "/revision/" in url:
                return json.dumps({"sha": "c" * 40}).encode()
            name = url.rsplit("/", 1)[-1]
            if name in files:
                return files[name]
            raise OSError("404")
        return get

    data, sha, src = tp.fetch_upstream_template("o/r", "main", get_factory({"chat_template.jinja": b"J"}))
    assert (data, sha, src) == (b"J", "c" * 40, "chat_template.jinja")
    cfg = json.dumps({"chat_template": [{"name": "rag", "template": "R"}, {"name": "default", "template": "D"}]}).encode()
    data, _sha, src = tp.fetch_upstream_template("o/r", "main", get_factory({"tokenizer_config.json": cfg}))
    assert (data, src) == (b"D", "tokenizer_config.json")
    with pytest.raises(LookupError):
        tp.fetch_upstream_template("o/r", "main", get_factory({}))
