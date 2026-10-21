"""SEC-ART-6: content-addressed read-only chat-template store. Fixtures only; no server, no live store."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from src.registry import template_store as ts

TEMPLATE = b"{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"


@pytest.fixture()
def src(tmp_path: Path) -> Path:
    p = tmp_path / "tmpl.jinja"
    p.write_bytes(TEMPLATE)
    return p


@pytest.fixture()
def root(tmp_path: Path):
    r = tmp_path / "store" / "sha256"
    yield r
    if r.exists():                      # let pytest clean the read-only dir
        os.chmod(r, 0o755)


def test_publish_is_content_addressed_and_read_only(src, root):
    sha, target = ts.publish(src, root)
    assert sha == ts.sha256_bytes(TEMPLATE)
    assert target == root / f"{sha}.jinja" == ts.store_path(sha, root)
    assert target.read_bytes() == TEMPLATE
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    assert ts.verify_store(root) == []


def test_publish_is_idempotent_and_never_overwrites(src, root):
    sha, target = ts.publish(src, root)
    assert ts.publish(src, root) == (sha, target)
    os.chmod(root, 0o755)
    os.chmod(target, 0o644)
    target.write_bytes(TEMPLATE + b" ")                 # in-place tamper
    os.chmod(root, 0o555)
    with pytest.raises(ts.TemplateStoreError, match="tampered"):
        ts.publish(src, root)
    assert target.read_bytes() == TEMPLATE + b" "        # not healed silently


def test_empty_template_refused(tmp_path, root):
    empty = tmp_path / "e.jinja"
    empty.write_bytes(b"")
    with pytest.raises(ts.TemplateStoreError, match="empty"):
        ts.publish(empty, root)


def test_check_accepts_the_pinned_path_and_the_12_hex_prefix(src, root):
    sha, target = ts.publish(src, root)
    assert ts.check_template_path(target, root=root) == sha
    assert ts.check_template_path(target, expected_sha=sha, root=root) == sha
    assert ts.check_template_path(target, expected_sha=sha[:12], root=root) == sha


def test_one_byte_modified_template_is_detected(src, root):
    sha, target = ts.publish(src, root)
    os.chmod(root, 0o755)
    os.chmod(target, 0o644)
    data = bytearray(target.read_bytes())
    data[3] ^= 0x01
    target.write_bytes(bytes(data))
    os.chmod(target, 0o444)
    os.chmod(root, 0o555)
    with pytest.raises(ts.TemplateStoreError, match="modified in place"):
        ts.check_template_path(target, expected_sha=sha, root=root)
    assert any("modified in place" in p for p in ts.verify_store(root))


def test_pin_mismatch_and_bad_pin_shape(src, root):
    sha, target = ts.publish(src, root)
    with pytest.raises(ts.TemplateStoreError, match="does not match"):
        ts.check_template_path(target, expected_sha="0" * 64, root=root)
    with pytest.raises(ts.TemplateStoreError, match="does not match"):
        ts.check_template_path(target, expected_sha="0" * 12, root=root)
    with pytest.raises(ts.TemplateStoreError, match="neither 64-hex nor 12-hex"):
        ts.check_template_path(target, expected_sha="abc", root=root)


def test_path_outside_store_and_symlink_escape_refused(src, root, tmp_path):
    ts.publish(src, root)
    with pytest.raises(ts.TemplateStoreError, match="outside the template store"):
        ts.check_template_path(src, root=root)           # a plain override file
    outside = tmp_path / f"{ts.sha256_bytes(TEMPLATE)}.jinja"
    outside.write_bytes(TEMPLATE)                          # right name, right bytes, wrong place
    with pytest.raises(ts.TemplateStoreError, match="outside the template store"):
        ts.check_template_path(outside, root=root)
    os.chmod(root, 0o755)
    link = root / f"{'1' * 64}.jinja"
    link.symlink_to(outside)                               # symlink planted inside the store
    os.chmod(root, 0o555)
    with pytest.raises(ts.TemplateStoreError, match="outside the template store"):
        ts.check_template_path(link, root=root)
    assert any("not a regular file" in p for p in ts.verify_store(root))


def test_non_hash_name_refused(src, root):
    ts.publish(src, root)
    os.chmod(root, 0o755)
    odd = root / "terse.jinja"
    odd.write_bytes(TEMPLATE)
    os.chmod(root, 0o555)
    with pytest.raises(ts.TemplateStoreError, match="not <64hex>"):
        ts.check_template_path(odd, root=root)


def test_group_or_other_writable_refused_but_owner_bit_is_not_the_barrier(src, root):
    sha, target = ts.publish(src, root)
    os.chmod(target, 0o664)
    with pytest.raises(ts.TemplateStoreError, match="group/other-writable"):
        ts.check_template_path(target, root=root)
    assert ts.check_template_path(target, root=root, require_readonly=False) == sha
    assert any("owner-writable" in p or "group/other" in p for p in ts.verify_store(root))


def test_verify_missing_store_is_a_problem(tmp_path):
    assert ts.verify_store(tmp_path / "nope")[0].endswith("does not exist")


def test_registry_check_server_mode(src, root):
    sha, target = ts.publish(src, root)
    reg = {"server_mode": {
        "good": {"chat_template_file": str(target), "chat_template": {"served_template_sha256_12": sha[:12]}},
        "bad_pin": {"chat_template_file": str(target), "chat_template": {"served_template_sha256": "f" * 64}},
        "legacy": {"chat_template_file": "/mnt/raid0/llm/models/chat-templates/epyc-qwen3x-v1-terse.jinja"},
        "none": {"enable_thinking": True},
    }}
    errors, warnings = ts.check_registry_template_store(reg, strict=False, root=root)
    assert len(errors) == 1 and "server_mode.bad_pin" in errors[0]
    assert len(warnings) == 1 and "server_mode.legacy" in warnings[0]
    errors, warnings = ts.check_registry_template_store(reg, strict=True, root=root)
    assert len(errors) == 2 and warnings == []


def test_registry_strict_env(src, root, monkeypatch):
    ts.publish(src, root)
    reg = {"server_mode": {"legacy": {"chat_template_file": "/x/y.jinja"}}}
    monkeypatch.delenv(ts.STRICT_ENV, raising=False)
    assert ts.check_registry_template_store(reg, root=root)[0] == []
    monkeypatch.setenv(ts.STRICT_ENV, "1")
    assert len(ts.check_registry_template_store(reg, root=root)[0]) == 1


def test_cli_publish_verify_path(src, root, capsys):
    assert ts.main(["--root", str(root), "publish", str(src)]) == 0
    sha = ts.sha256_bytes(TEMPLATE)
    assert sha in capsys.readouterr().out
    assert ts.main(["--root", str(root), "verify"]) == 0
    assert ts.main(["--root", str(root), "path", sha]) == 0
    assert ts.main(["--root", str(root), "path", "short"]) == 2
