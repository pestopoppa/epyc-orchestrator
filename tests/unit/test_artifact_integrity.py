"""SEC-ART-2 (GGUF fingerprint pins) and SEC-ART-4 (served-template pins): fixtures only, no server, no weights."""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import pytest

from src.registry import artifact_integrity as ai

TEMPLATE = b"{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"


def _s(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def make_gguf(path: Path, *, template: bytes | None = TEMPLATE, version: int = 3, pad_to: int = 0) -> Path:
    """Minimal GGUF: header, KVs (string, uint32, string-array, optional chat_template), zero tensors."""
    kvs: list[bytes] = [
        _s(b"general.architecture") + struct.pack("<I", 8) + _s(b"qwen35"),
        _s(b"general.file_type") + struct.pack("<I", 4) + struct.pack("<I", 7),
        _s(b"tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 3) + _s(b"a") + _s(b"bb") + _s(b"ccc"),
    ]
    if template is not None:
        kvs.append(_s(b"tokenizer.chat_template") + struct.pack("<I", 8) + _s(template))
    kvs.append(_s(b"general.name") + struct.pack("<I", 8) + _s(b"fixture"))
    blob = b"GGUF" + struct.pack("<I", version) + struct.pack("<QQ", 0, len(kvs)) + b"".join(kvs)
    with open(path, "wb") as fh:
        fh.write(blob)
        if pad_to > len(blob):
            fh.truncate(pad_to)  # sparse "tensor data": must never be read
    return path


def corrupt(path: Path) -> None:
    """Flip one byte near the end (size unchanged) and move mtime forward so the stat fingerprint differs."""
    st = path.stat()
    data = bytearray(path.read_bytes())
    data[-3] ^= 0x01
    path.write_bytes(bytes(data))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CountingOpen:
    """Wraps open(): sums the bytes actually returned by read()."""

    def __init__(self) -> None:
        self.bytes_read = 0

    def __call__(self, path, mode="rb"):
        real = open(path, mode)
        outer = self

        class _F:
            def read(self, n=-1):
                data = real.read(n)
                outer.bytes_read += len(data)
                return data

            def seek(self, *a):
                return real.seek(*a)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                real.close()

        return _F()


# ------------------------------------------------------------------ GGUF header template reader

def test_reads_exact_template_bytes_skipping_array_and_scalar_kvs(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    assert ai.read_gguf_chat_template(g) == TEMPLATE


def test_absent_key_returns_none(tmp_path):
    assert ai.read_gguf_chat_template(make_gguf(tmp_path / "m.gguf", template=None)) is None


def test_bad_magic_truncation_and_old_version_raise(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(ai.ArtifactIntegrityError):
        ai.read_gguf_chat_template(bad)
    v1 = make_gguf(tmp_path / "v1.gguf", version=1)
    with pytest.raises(ai.ArtifactIntegrityError, match="version 1"):
        ai.read_gguf_chat_template(v1)
    full = make_gguf(tmp_path / "full.gguf").read_bytes()
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(full[:60])
    with pytest.raises(ai.ArtifactIntegrityError):
        ai.read_gguf_chat_template(cut)


def test_header_read_never_touches_tensor_data(tmp_path, monkeypatch):
    g = make_gguf(tmp_path / "big.gguf", pad_to=1 << 30)  # 1 GiB sparse
    counter = CountingOpen()
    monkeypatch.setattr(ai, "_open", counter)
    assert ai.read_gguf_chat_template(g) == TEMPLATE
    assert 0 < counter.bytes_read < (1 << 20)


# ------------------------------------------------------------------ effective template + pin check

def test_file_wins_over_embedded_and_labels_source(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    f = tmp_path / "t.jinja"
    f.write_bytes(b"override")
    data, src = ai.effective_template_bytes(["llama-server", "-m", str(g), "--chat-template-file", str(f)])
    assert data == b"override" and src == f"file:{f}"
    data, src = ai.effective_template_bytes(["llama-server", "-m", str(g)])
    assert data == TEMPLATE and src == f"gguf:{g}"
    data, src = ai.effective_template_bytes(["llama-server", f"--model={g}", f"--chat-template-file={f}"])
    assert data == b"override"


def test_inline_chat_template_is_refused(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    with pytest.raises(ai.ArtifactIntegrityError, match="inline --chat-template"):
        ai.effective_template_bytes(["llama-server", "-m", str(g), "--chat-template", "chatml"])


def test_one_byte_modified_template_file_is_refused_with_both_digests(tmp_path):
    f = tmp_path / "t.jinja"
    f.write_bytes(TEMPLATE)
    pin = ai.template_sha256(TEMPLATE)
    assert ai.check_template_pin("frontdoor", f.read_bytes(), f"file:{f}", pin, strict=False) == []
    evil = bytearray(TEMPLATE)
    evil[10] ^= 0x01
    f.write_bytes(bytes(evil))
    with pytest.raises(ai.TemplatePinError) as ei:
        ai.check_template_pin("frontdoor", f.read_bytes(), f"file:{f}", pin, strict=False)
    msg = str(ei.value)
    assert pin in msg and ai.template_sha256(bytes(evil)) in msg and "frontdoor" in msg and str(f) in msg


def test_one_byte_modified_embedded_template_is_refused(tmp_path):
    good = make_gguf(tmp_path / "good.gguf")
    pin = ai.template_sha256(ai.read_gguf_chat_template(good))
    bad = make_gguf(tmp_path / "bad.gguf", template=TEMPLATE.replace(b"im_end", b"im_enD"))
    data, src = ai.effective_template_bytes(["llama-server", "-m", str(bad)])
    with pytest.raises(ai.TemplatePinError):
        ai.check_template_pin("architect_general", data, src, pin, strict=False)


def test_missing_template_pin_warns_unless_strict():
    warns = ai.check_template_pin("r", TEMPLATE, "gguf:x", None, strict=False)
    assert warns and "no served_template_sha256" in warns[0]
    with pytest.raises(ai.TemplatePinError, match="strict"):
        ai.check_template_pin("r", TEMPLATE, "gguf:x", None, strict=True)


def test_gguf_without_template_cannot_satisfy_a_pin():
    with pytest.raises(ai.TemplatePinError):
        ai.check_template_pin("r", None, "gguf:x", "a" * 64, strict=False)


# ------------------------------------------------------------------ GGUF fingerprint pins (SEC-ART-2)

def _exploding_hasher(_p):
    raise AssertionError("fingerprint hit must not re-hash")


def test_register_then_launch_uses_stat_fingerprint_only(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    led = tmp_path / "ledger.json"
    pin = ai.GgufPin(role="r", path=str(g), sha256=sha(g))
    entry = ai.register_gguf(g, pin_sha256=pin.sha256, ledger=led)
    assert entry["sha256"] == pin.sha256 and led.exists()
    assert ai.verify_gguf_for_launch(g, pin, strict=True, ledger=led, hasher=_exploding_hasher) == []


def test_touched_copy_is_rehashed_and_accepted_when_bytes_equal(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    led = tmp_path / "ledger.json"
    pin = ai.GgufPin(role="r", path=str(g), sha256=sha(g))
    ai.register_gguf(g, pin_sha256=pin.sha256, ledger=led)
    st = g.stat()
    os.utime(g, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # touch: mtime changes, bytes do not
    calls = []

    def hasher(p):
        calls.append(p)
        return ai.sha256_file(p)

    assert ai.verify_gguf_for_launch(g, pin, strict=True, ledger=led, hasher=hasher) == []
    assert len(calls) == 1                                               # re-hashed because stat changed
    ai.verify_gguf_for_launch(g, pin, strict=True, ledger=led, hasher=_exploding_hasher)  # ledger refreshed


def test_touched_and_modified_copy_is_refused(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    led = tmp_path / "ledger.json"
    pin = ai.GgufPin(role="r", path=str(g), sha256=sha(g))
    ai.register_gguf(g, pin_sha256=pin.sha256, ledger=led)
    corrupt(g)                                                           # size same, mtime changed, content differs
    with pytest.raises(ai.GgufPinError) as ei:
        ai.verify_gguf_for_launch(g, pin, strict=False, ledger=led)
    assert pin.sha256 in str(ei.value) and str(g) in str(ei.value)


def test_replaced_file_with_same_bytes_new_inode_is_rehashed(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    led = tmp_path / "ledger.json"
    pin = ai.GgufPin(role="r", path=str(g), sha256=sha(g))
    ai.register_gguf(g, pin_sha256=pin.sha256, ledger=led)
    copy = tmp_path / "copy.gguf"
    copy.write_bytes(g.read_bytes())
    os.replace(copy, g)                                                  # new inode, same bytes
    calls = []
    ai.verify_gguf_for_launch(g, pin, strict=True, ledger=led, hasher=lambda p: (calls.append(p), ai.sha256_file(p))[1])
    assert len(calls) == 1


def test_register_refuses_mismatch_and_records_nothing(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    led = tmp_path / "ledger.json"
    with pytest.raises(ai.GgufPinError):
        ai.register_gguf(g, pin_sha256="0" * 64, ledger=led)
    assert not led.exists() or ai.load_ledger(led) == {}


def test_unregistered_file_warns_non_strict_and_refuses_strict(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    pin = ai.GgufPin(role="r", path=str(g), sha256=sha(g))
    led = tmp_path / "none.json"
    warns = ai.verify_gguf_for_launch(g, pin, strict=False, ledger=led)
    assert warns and "never registered" in warns[0]
    with pytest.raises(ai.ArtifactIntegrityError, match="strict"):
        ai.verify_gguf_for_launch(g, pin, strict=True, ledger=led)


def test_unpinned_file_warns_non_strict_and_refuses_strict(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    assert ai.verify_gguf_for_launch(g, None, strict=False, ledger=tmp_path / "l.json")
    with pytest.raises(ai.ArtifactIntegrityError):
        ai.verify_gguf_for_launch(g, None, strict=True, ledger=tmp_path / "l.json")


def test_split_gguf_checks_every_shard(tmp_path):
    shards = [make_gguf(tmp_path / f"m-0000{i}-of-00003.gguf", template=TEMPLATE + str(i).encode()) for i in (1, 2, 3)]
    pin = ai.GgufPin(role="r", path=str(shards[0]), sha256=sha(shards[0]),
                     shard_sha256={shards[1].name: sha(shards[1]), shards[2].name: sha(shards[2])})
    led = tmp_path / "ledger.json"
    for s in shards:
        ai.register_gguf(s, pin_sha256=pin.sha256 if s == shards[0] else pin.shard_sha256[s.name], ledger=led)
    assert ai.verify_gguf_for_launch(shards[0], pin, strict=True, ledger=led, hasher=_exploding_hasher) == []
    corrupt(shards[2])
    with pytest.raises(ai.GgufPinError, match="00003"):
        ai.verify_gguf_for_launch(shards[0], pin, strict=True, ledger=led)


# ------------------------------------------------------------------ the launch funnel

def _registry(gguf: Path, *, pin: str | None, strict: bool | None = None) -> dict:
    model = {"path": str(gguf)}
    if pin:
        model["sha256"] = pin
    reg: dict = {"roles": {"frontdoor": {"model": model}}}
    if strict is not None:
        reg["runtime_defaults"] = {"artifact_integrity": {"pins_strict": strict, "template_pins_strict": strict}}
    return reg


def test_verify_launch_argv_accepts_matching_artifacts(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    led = tmp_path / "l.json"
    ai.register_gguf(g, pin_sha256=sha(g), ledger=led)
    argv = ["llama-server", "-m", str(g), "--port", "1"]
    assert ai.verify_launch_argv(argv, role="frontdoor", template_pin=ai.template_sha256(TEMPLATE),
                                 check_template=True, registry=_registry(g, pin=sha(g)), ledger=led) == []


def test_verify_launch_argv_refuses_changed_gguf_and_changed_template(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    led = tmp_path / "l.json"
    ai.register_gguf(g, pin_sha256=sha(g), ledger=led)
    reg = _registry(g, pin=sha(g))
    argv = ["llama-server", "-m", str(g)]
    with pytest.raises(ai.TemplatePinError):
        ai.verify_launch_argv(argv, role="frontdoor", template_pin="f" * 64, check_template=True, registry=reg, ledger=led)
    corrupt(g)
    with pytest.raises(ai.GgufPinError):
        ai.verify_launch_argv(argv, role="frontdoor", template_pin=None, check_template=False, registry=reg, ledger=led)


def test_embedders_skip_template_check(tmp_path):
    g = make_gguf(tmp_path / "m.gguf", template=None)
    ai.verify_launch_argv(["llama-server", "-m", str(g)], role=None, template_pin=None, check_template=False,
                          registry=_registry(g, pin=None), ledger=tmp_path / "l.json")


def test_strict_switch_from_registry_and_env(tmp_path, monkeypatch):
    reg = {"runtime_defaults": {"artifact_integrity": {"pins_strict": True}}}
    monkeypatch.delenv("REGISTRY_ARTIFACT_PINS_STRICT", raising=False)
    assert ai.strict_enabled(ai.PINS_STRICT, reg) is True
    assert ai.strict_enabled(ai.PINS_STRICT, {}) is False
    monkeypatch.setenv("REGISTRY_ARTIFACT_PINS_STRICT", "0")
    assert ai.strict_enabled(ai.PINS_STRICT, reg) is False       # explicit env off wins
    monkeypatch.setenv("REGISTRY_ARTIFACT_PINS_STRICT", "1")
    assert ai.strict_enabled(ai.PINS_STRICT, {}) is True


def test_strict_launch_refuses_unpinned_model(tmp_path):
    g = make_gguf(tmp_path / "m.gguf")
    with pytest.raises(ai.ArtifactIntegrityError):
        ai.verify_launch_argv(["llama-server", "-m", str(g)], role="frontdoor", template_pin=None,
                              check_template=False, registry=_registry(g, pin=None, strict=True),
                              ledger=tmp_path / "l.json")
